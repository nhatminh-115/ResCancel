"""
patch_fungibility/v1_pipeline.py

Execution pipeline for Patch Fungibility V1:
- Extracts calibration activations and computes calibration statistics (mu_l, sigma_l, PCA).
- Evaluates clean baselines and validates forward equivalence.
- Runs Phase 1 (Depth Sweep: 5, 7, 8, 9, 10).
- Runs Phase 2 (Fraction & Geometry Controls at Depth 8: 25%, 50%, 75%).
- Runs Phase 3 (100% Spatial Replacement Diversity Sweep: Shared vs Independent vs Isotropic).
- Runs Phase 4 (1D Variation: Natural PC1 vs Natural PC2 vs Random 1D).
- Checks Phase 5 (Deterministic Secondary Adapted-Depth Rule).
- Produces image-level parquets and summary CSVs.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits, ParquetImageSubset
from patch_fungibility.v1_models import (
    get_vitb_model,
    get_dinov2_model,
    forward_vitb_manual,
    forward_dinov2_manual,
    validate_manual_forwards
)
from patch_fungibility.v1_interventions import (
    build_nested_masks,
    compute_calibration_statistics,
    get_coordinate_permutations,
    get_random_1d_directions,
    PatchInterventionHook
)
from patch_fungibility.v1_decision import (
    compute_paired_statistics,
    apply_bh_fdr,
    evaluate_model_signatures,
    evaluate_v1_decision
)
from patch_fungibility.v1_validation import verify_v1_protocol_assertions


def extract_cached_activations(
    model: nn.Module,
    loader: DataLoader,
    model_type: str,
    depths: Tuple[int, ...],
    device: torch.device
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Passes data through the model to extract:
      - hidden states at requested block outputs (stored on CPU RAM)
      - logits
      - labels
    """
    model.eval()
    cached_h = {d: [] for d in depths}
    all_logits = []
    all_labels = []

    forward_fn = forward_vitb_manual if model_type == "vitb" else forward_dinov2_manual

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits, collected = forward_fn(
                model, x=images, start_depth=0, collect_depths=depths
            )
            for d in depths:
                cached_h[d].append(collected[d].cpu())
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    concatenated_h = {d: torch.cat(cached_h[d], dim=0) for d in depths}
    concatenated_logits = torch.cat(all_logits, dim=0)
    concatenated_labels = torch.cat(all_labels, dim=0)

    return concatenated_h, concatenated_logits, concatenated_labels


def evaluate_condition_from_cache(
    model: nn.Module,
    model_type: str,
    start_depth: int,
    cached_h_start: torch.Tensor,
    labels: torch.Tensor,
    hook: Optional[PatchInterventionHook],
    batch_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Evaluates an intervention condition starting from cached hidden states at start_depth.
    Computes logits, margins, and correctness.
    """
    model.eval()
    forward_fn = forward_vitb_manual if model_type == "vitb" else forward_dinov2_manual

    all_logits = []
    n_samples = cached_h_start.size(0)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            h_batch = cached_h_start[i : i + batch_size].to(device)
            # If intervention is at start_depth, apply hook to h_batch directly before forwarding
            # or pass intervention_fn. Since hook checks depth == target_depth:
            if hook is not None and hook.target_depth == start_depth:
                h_batch = hook(start_depth, h_batch)
                logits, _ = forward_fn(model, start_depth=start_depth, h_start=h_batch)
            else:
                logits, _ = forward_fn(model, start_depth=start_depth, h_start=h_batch, intervention_fn=hook)
            all_logits.append(logits.cpu())

    logits = torch.cat(all_logits, dim=0)  # (N, C)

    # Compute margin and metrics
    N, C = logits.shape
    y = labels.numpy()
    logits_np = logits.numpy()

    true_class_logits = logits_np[np.arange(N), y]
    # Strongest incorrect logit
    mask_incorrect = np.ones((N, C), dtype=bool)
    mask_incorrect[np.arange(N), y] = False
    incorrect_logits = np.where(mask_incorrect, logits_np, -1e9)
    strongest_incorrect = np.max(incorrect_logits, axis=1)
    margins = true_class_logits - strongest_incorrect

    predictions = np.argmax(logits_np, axis=1)
    correctness = (predictions == y).astype(int)

    df_image = pd.DataFrame({
        "sample_id": np.arange(N),
        "label": y,
        "true_class_logit": true_class_logits,
        "strongest_incorrect_logit": strongest_incorrect,
        "true_class_margin": margins,
        "top1_prediction": predictions,
        "correctness": correctness
    })

    return logits, df_image


def run_full_v1_evaluation(
    output_dir: str = "outputs/fungibility_v1",
    smoke_test: bool = False,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the complete Patch Fungibility V1 evaluation across ViT-B and DINOv2.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if device_str is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[V1 Pipeline] Starting execution on device: {device}")

    # 1. Load models and metadata
    print("[V1 Pipeline] Loading models...")
    vitb_model, vitb_transform, vitb_meta = get_vitb_model(device)
    dinov2_model, dinov2_transform, dinov2_meta = get_dinov2_model(device)

    model_metadata = {
        "model_a": vitb_meta,
        "model_b": dinov2_meta,
        "smoke_test": smoke_test,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(model_metadata, f, indent=2)

    # 2. Load dataset splits
    print("[V1 Pipeline] Loading disjoint ImageNet splits...")
    n_per_split = 64 if smoke_test else 1000
    calib_ds_raw, eval_ds_raw, calib_manifest, eval_manifest = get_disjoint_imagenet_splits(
        calib_seed=9101, eval_seed=9201, n_per_split=n_per_split
    )

    calib_samples = calib_ds_raw._samples
    eval_samples = eval_ds_raw._samples

    vitb_calib_ds = ParquetImageSubset(calib_samples, transform=vitb_transform)
    vitb_eval_ds = ParquetImageSubset(eval_samples, transform=vitb_transform)

    dinov2_calib_ds = ParquetImageSubset(calib_samples, transform=dinov2_transform)
    dinov2_eval_ds = ParquetImageSubset(eval_samples, transform=dinov2_transform)

    batch_size_vitb = 16
    batch_size_dinov2 = 32

    loader_calib_vitb = DataLoader(vitb_calib_ds, batch_size=batch_size_vitb, shuffle=False)
    loader_eval_vitb = DataLoader(vitb_eval_ds, batch_size=batch_size_vitb, shuffle=False)

    loader_calib_dinov2 = DataLoader(dinov2_calib_ds, batch_size=batch_size_dinov2, shuffle=False)
    loader_eval_dinov2 = DataLoader(dinov2_eval_ds, batch_size=batch_size_dinov2, shuffle=False)

    # 3. Validate manual forward implementations
    print("[V1 Pipeline] Validating manual forward paths against official forwards...")
    manual_val_results = validate_manual_forwards(
        vitb_model, dinov2_model, loader_eval_vitb, loader_eval_dinov2, device, n_images=64
    )
    with open(os.path.join(output_dir, "manual_forward_validation.json"), "w") as f:
        json.dump(manual_val_results, f, indent=2)
    print(f"  ViT-B max abs diff: {manual_val_results['vit_base_patch16_224.augreg_in1k']['max_abs_logit_diff']:.2e}")
    print(f"  DINOv2 max abs diff: {manual_val_results['dinov2_vits14_lc']['max_abs_logit_diff']:.2e}")

    # 4. Build nested masks
    print("[V1 Pipeline] Building nested spatial patch masks (seed=21001)...")
    vitb_masks = build_nested_masks(n_patches=196, seed=21001)
    dinov2_masks = build_nested_masks(n_patches=256, seed=21001)

    # 5. Extract calibration activations & compute statistics
    print("[V1 Pipeline] Extracting calibration activations at depths {5, 7, 8, 9, 10}...")
    target_depths = (5, 7, 8, 9, 10)

    # ViT-B calibration
    t0 = time.time()
    calib_h_vitb, _, _ = extract_cached_activations(
        vitb_model, loader_calib_vitb, "vitb", target_depths, device
    )
    # Patch activations only: slice out index 0 (CLS)
    calib_patches_vitb = {d: calib_h_vitb[d][:, 1:, :] for d in target_depths}
    stats_vitb = compute_calibration_statistics(calib_patches_vitb)
    del calib_h_vitb  # Free memory

    # DINOv2 calibration
    calib_h_dinov2, _, _ = extract_cached_activations(
        dinov2_model, loader_calib_dinov2, "dinov2", target_depths, device
    )
    calib_patches_dinov2 = {d: calib_h_dinov2[d][:, 1:, :] for d in target_depths}
    stats_dinov2 = compute_calibration_statistics(calib_patches_dinov2)
    del calib_h_dinov2  # Free memory

    # Save calibration statistics
    np.savez_compressed(
        os.path.join(output_dir, "calibration_statistics.npz"),
        **{f"vitb_{k}": v for k, v in stats_vitb.items()},
        **{f"dinov2_{k}": v for k, v in stats_dinov2.items()}
    )
    print(f"[V1 Pipeline] Calibration completed in {time.time() - t0:.2f}s")

    # 6. Coordinate permutations & Random 1D directions
    coord_perms_vitb = get_coordinate_permutations(dim=768, seeds=[23001, 23002, 23003])
    coord_perms_dinov2 = get_coordinate_permutations(dim=384, seeds=[23001, 23002, 23003])

    rand_1d_vitb = get_random_1d_directions(dim=768, seeds=[25001, 25002, 25003])
    rand_1d_dinov2 = get_random_1d_directions(dim=384, seeds=[25001, 25002, 25003])

    # 7. Extract evaluation hidden states at depths {5, 7, 8, 9, 10}
    print("[V1 Pipeline] Extracting evaluation hidden states for ViT-B...")
    eval_h_vitb, clean_logits_vitb, eval_labels_vitb = extract_cached_activations(
        vitb_model, loader_eval_vitb, "vitb", target_depths, device
    )

    print("[V1 Pipeline] Extracting evaluation hidden states for DINOv2...")
    eval_h_dinov2, clean_logits_dinov2, eval_labels_dinov2 = extract_cached_activations(
        dinov2_model, loader_eval_dinov2, "dinov2", target_depths, device
    )

    # 8. Evaluate models across all conditions
    def run_model_experiments(
        model: nn.Module,
        model_type: str,
        eval_h: Dict[int, torch.Tensor],
        labels: torch.Tensor,
        stats: Dict[str, Any],
        masks: Dict[float, np.ndarray],
        coord_perms: Dict[int, np.ndarray],
        rand_1d: Dict[int, np.ndarray],
        batch_size: int
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        image_results = {}
        dim = 768 if model_type == "vitb" else 384

        # Clean baseline
        _, df_clean = evaluate_condition_from_cache(
            model, model_type, start_depth=8, cached_h_start=eval_h[8],
            labels=labels, hook=None, batch_size=batch_size, device=device
        )
        df_clean["margin_damage_vs_clean"] = 0.0
        df_clean["flip_corr_to_incorr"] = 0
        df_clean["flip_incorr_to_corr"] = 0
        image_results["CLEAN"] = df_clean

        clean_margins = df_clean["true_class_margin"].values
        clean_corrects = df_clean["correctness"].values

        def eval_and_record(cond_name: str, depth: int, hook: Optional[PatchInterventionHook]) -> pd.DataFrame:
            _, df = evaluate_condition_from_cache(
                model, model_type, start_depth=depth, cached_h_start=eval_h[depth],
                labels=labels, hook=hook, batch_size=batch_size, device=device
            )
            margins = df["true_class_margin"].values
            corrects = df["correctness"].values
            df["margin_damage_vs_clean"] = clean_margins - margins
            df["flip_corr_to_incorr"] = ((clean_corrects == 1) & (corrects == 0)).astype(int)
            df["flip_incorr_to_corr"] = ((clean_corrects == 0) & (corrects == 1)).astype(int)
            image_results[cond_name] = df
            return df

        # --- Phase 1: Depth Sweep (25% replacement) ---
        print(f"[{model_type.upper()}] Running Phase 1: Depth Sweep (25% replacement)...")
        depth_rows = []
        # Add clean row
        depth_rows.append({
            "depth": 0, "condition": "CLEAN", "seed": None,
            "top1_accuracy": float(np.mean(clean_corrects)),
            "mean_margin": float(np.mean(clean_margins)),
            "median_margin": float(np.median(clean_margins)),
            "margin_damage": 0.0,
            "corr_to_incorr": 0, "incorr_to_corr": 0
        })

        for d in [5, 7, 8, 9, 10]:
            mu_d = stats[f"mu_{d}"]
            sigma_d = stats[f"sigma_{d}"]
            m_25 = masks[0.25]

            # Zero
            hk_zero = PatchInterventionHook(d, m_25, "ZERO", device=device)
            df_z = eval_and_record(f"ZERO_depth_{d}", d, hk_zero)
            depth_rows.append({
                "depth": d, "condition": "ZERO", "seed": None,
                "top1_accuracy": float(np.mean(df_z["correctness"])),
                "mean_margin": float(np.mean(df_z["true_class_margin"])),
                "median_margin": float(np.median(df_z["true_class_margin"])),
                "margin_damage": float(np.mean(df_z["margin_damage_vs_clean"])),
                "corr_to_incorr": int(df_z["flip_corr_to_incorr"].sum()),
                "incorr_to_corr": int(df_z["flip_incorr_to_corr"].sum())
            })

            # Centroid
            hk_cent = PatchInterventionHook(d, m_25, "CENTROID", mu=mu_d, device=device)
            df_c = eval_and_record(f"CENTROID_depth_{d}", d, hk_cent)
            depth_rows.append({
                "depth": d, "condition": "CENTROID", "seed": None,
                "top1_accuracy": float(np.mean(df_c["correctness"])),
                "mean_margin": float(np.mean(df_c["true_class_margin"])),
                "median_margin": float(np.median(df_c["true_class_margin"])),
                "margin_damage": float(np.mean(df_c["margin_damage_vs_clean"])),
                "corr_to_incorr": int(df_c["flip_corr_to_incorr"].sum()),
                "incorr_to_corr": int(df_c["flip_incorr_to_corr"].sum())
            })

            # Gaussian (seeds 22001, 22002, 22003)
            for s in [22001, 22002, 22003]:
                hk_g = PatchInterventionHook(d, m_25, "DIAGONAL_GAUSSIAN", mu=mu_d, sigma=sigma_d, seed=s, device=device)
                df_g = eval_and_record(f"GAUSSIAN_depth_{d}_seed_{s}", d, hk_g)
                depth_rows.append({
                    "depth": d, "condition": "DIAGONAL_GAUSSIAN", "seed": s,
                    "top1_accuracy": float(np.mean(df_g["correctness"])),
                    "mean_margin": float(np.mean(df_g["true_class_margin"])),
                    "median_margin": float(np.median(df_g["true_class_margin"])),
                    "margin_damage": float(np.mean(df_g["margin_damage_vs_clean"])),
                    "corr_to_incorr": int(df_g["flip_corr_to_incorr"].sum()),
                    "incorr_to_corr": int(df_g["flip_incorr_to_corr"].sum())
                })

        df_depth = pd.DataFrame(depth_rows)

        # --- Phase 2: Fraction & Geometry Sweep at Depth 8 ---
        print(f"[{model_type.upper()}] Running Phase 2: Fraction & Geometry Controls (Depth 8)...")
        frac_geom_rows = []
        mu_8 = stats["mu_8"]
        sigma_8 = stats["sigma_8"]

        for frac in [0.25, 0.50, 0.75]:
            m_f = masks[frac]

            # Zero
            hk_z = PatchInterventionHook(8, m_f, "ZERO", device=device)
            df_z = eval_and_record(f"ZERO_d8_f{frac}", 8, hk_z)
            frac_geom_rows.append({
                "fraction": frac, "condition": "ZERO", "seed": None,
                "top1_accuracy": float(np.mean(df_z["correctness"])),
                "mean_margin": float(np.mean(df_z["true_class_margin"])),
                "median_margin": float(np.median(df_z["true_class_margin"])),
                "margin_damage": float(np.mean(df_z["margin_damage_vs_clean"])),
                "corr_to_incorr": int(df_z["flip_corr_to_incorr"].sum()),
                "incorr_to_corr": int(df_z["flip_incorr_to_corr"].sum())
            })

            # Centroid
            hk_c = PatchInterventionHook(8, m_f, "CENTROID", mu=mu_8, device=device)
            df_c = eval_and_record(f"CENTROID_d8_f{frac}", 8, hk_c)
            frac_geom_rows.append({
                "fraction": frac, "condition": "CENTROID", "seed": None,
                "top1_accuracy": float(np.mean(df_c["correctness"])),
                "mean_margin": float(np.mean(df_c["true_class_margin"])),
                "median_margin": float(np.median(df_c["true_class_margin"])),
                "margin_damage": float(np.mean(df_c["margin_damage_vs_clean"])),
                "corr_to_incorr": int(df_c["flip_corr_to_incorr"].sum()),
                "incorr_to_corr": int(df_c["flip_incorr_to_corr"].sum())
            })

            # Gaussian (seeds 22001, 22002, 22003)
            for s in [22001, 22002, 22003]:
                hk_g = PatchInterventionHook(8, m_f, "DIAGONAL_GAUSSIAN", mu=mu_8, sigma=sigma_8, seed=s, device=device)
                df_g = eval_and_record(f"GAUSSIAN_d8_f{frac}_s{s}", 8, hk_g)
                frac_geom_rows.append({
                    "fraction": frac, "condition": "DIAGONAL_GAUSSIAN", "seed": s,
                    "top1_accuracy": float(np.mean(df_g["correctness"])),
                    "mean_margin": float(np.mean(df_g["true_class_margin"])),
                    "median_margin": float(np.median(df_g["true_class_margin"])),
                    "margin_damage": float(np.mean(df_g["margin_damage_vs_clean"])),
                    "corr_to_incorr": int(df_g["flip_corr_to_incorr"].sum()),
                    "incorr_to_corr": int(df_g["flip_incorr_to_corr"].sum())
                })

            # Coordinate Permuted Centroid (seeds 23001, 23002, 23003)
            for s in [23001, 23002, 23003]:
                perm = coord_perms[s]
                hk_p = PatchInterventionHook(8, m_f, "COORDINATE_PERMUTED_CENTROID", mu=mu_8, extra_args={"perm": perm}, device=device)
                df_p = eval_and_record(f"COORD_PERM_d8_f{frac}_s{s}", 8, hk_p)
                frac_geom_rows.append({
                    "fraction": frac, "condition": "COORDINATE_PERMUTED_CENTROID", "seed": s,
                    "top1_accuracy": float(np.mean(df_p["correctness"])),
                    "mean_margin": float(np.mean(df_p["true_class_margin"])),
                    "median_margin": float(np.median(df_p["true_class_margin"])),
                    "margin_damage": float(np.mean(df_p["margin_damage_vs_clean"])),
                    "corr_to_incorr": int(df_p["flip_corr_to_incorr"].sum()),
                    "incorr_to_corr": int(df_p["flip_incorr_to_corr"].sum())
                })

            # Sign-flipped centroid
            hk_sf = PatchInterventionHook(8, m_f, "SIGN_FLIPPED_CENTROID", mu=mu_8, device=device)
            df_sf = eval_and_record(f"SIGN_FLIPPED_CENTROID_d8_f{frac}", 8, hk_sf)
            frac_geom_rows.append({
                "fraction": frac, "condition": "SIGN_FLIPPED_CENTROID", "seed": None,
                "top1_accuracy": float(np.mean(df_sf["correctness"])),
                "mean_margin": float(np.mean(df_sf["true_class_margin"])),
                "median_margin": float(np.median(df_sf["true_class_margin"])),
                "margin_damage": float(np.mean(df_sf["margin_damage_vs_clean"])),
                "corr_to_incorr": int(df_sf["flip_corr_to_incorr"].sum()),
                "incorr_to_corr": int(df_sf["flip_incorr_to_corr"].sum())
            })

        df_frac_geom = pd.DataFrame(frac_geom_rows)

        # --- Phase 3: 100% Spatial Replacement Diversity Sweep (Depth 8) ---
        print(f"[{model_type.upper()}] Running Phase 3: 100% Diversity Sweep (Depth 8)...")
        diversity_rows = []
        m_100 = masks[1.00]

        # Static Centroid
        hk_sc = PatchInterventionHook(8, m_100, "STATIC_CENTROID", mu=mu_8, device=device)
        df_sc = eval_and_record("STATIC_CENTROID_100", 8, hk_sc)
        diversity_rows.append({
            "condition": "STATIC_CENTROID", "seed": None,
            "top1_accuracy": float(np.mean(df_sc["correctness"])),
            "mean_margin": float(np.mean(df_sc["true_class_margin"])),
            "median_margin": float(np.median(df_sc["true_class_margin"])),
            "margin_damage": float(np.mean(df_sc["margin_damage_vs_clean"])),
            "corr_to_incorr": int(df_sc["flip_corr_to_incorr"].sum()),
            "incorr_to_corr": int(df_sc["flip_incorr_to_corr"].sum())
        })

        for s in [24001, 24002, 24003, 24004, 24005]:
            # Shared Gaussian
            hk_shared = PatchInterventionHook(8, m_100, "SHARED_GAUSSIAN_NOISE", mu=mu_8, sigma=sigma_8, seed=s, device=device)
            df_sh = eval_and_record(f"SHARED_GAUSSIAN_s{s}", 8, hk_shared)
            diversity_rows.append({
                "condition": "SHARED_GAUSSIAN", "seed": s,
                "top1_accuracy": float(np.mean(df_sh["correctness"])),
                "mean_margin": float(np.mean(df_sh["true_class_margin"])),
                "median_margin": float(np.median(df_sh["true_class_margin"])),
                "margin_damage": float(np.mean(df_sh["margin_damage_vs_clean"])),
                "corr_to_incorr": int(df_sh["flip_corr_to_incorr"].sum()),
                "incorr_to_corr": int(df_sh["flip_incorr_to_corr"].sum())
            })

            # Independent Gaussian
            hk_ind = PatchInterventionHook(8, m_100, "INDEPENDENT_GAUSSIAN_NOISE", mu=mu_8, sigma=sigma_8, seed=s, device=device)
            df_ind = eval_and_record(f"INDEPENDENT_GAUSSIAN_s{s}", 8, hk_ind)
            diversity_rows.append({
                "condition": "INDEPENDENT_GAUSSIAN", "seed": s,
                "top1_accuracy": float(np.mean(df_ind["correctness"])),
                "mean_margin": float(np.mean(df_ind["true_class_margin"])),
                "median_margin": float(np.median(df_ind["true_class_margin"])),
                "margin_damage": float(np.mean(df_ind["margin_damage_vs_clean"])),
                "corr_to_incorr": int(df_ind["flip_corr_to_incorr"].sum()),
                "incorr_to_corr": int(df_ind["flip_incorr_to_corr"].sum())
            })

            # Isotropic Independent
            hk_iso = PatchInterventionHook(8, m_100, "ISOTROPIC_INDEPENDENT_NOISE", mu=mu_8, sigma=sigma_8, seed=s, device=device)
            df_iso = eval_and_record(f"ISOTROPIC_INDEPENDENT_s{s}", 8, hk_iso)
            diversity_rows.append({
                "condition": "ISOTROPIC_INDEPENDENT", "seed": s,
                "top1_accuracy": float(np.mean(df_iso["correctness"])),
                "mean_margin": float(np.mean(df_iso["true_class_margin"])),
                "median_margin": float(np.median(df_iso["true_class_margin"])),
                "margin_damage": float(np.mean(df_iso["margin_damage_vs_clean"])),
                "corr_to_incorr": int(df_iso["flip_corr_to_incorr"].sum()),
                "incorr_to_corr": int(df_iso["flip_incorr_to_corr"].sum())
            })

        df_diversity = pd.DataFrame(diversity_rows)

        # --- Phase 4: 1D Variation (Depth 8, 100% replacement) ---
        print(f"[{model_type.upper()}] Running Phase 4: 1D Subspace Sweep (Depth 8)...")
        pca_lambdas = stats["pca_lambdas_8"]
        pca_vectors = stats["pca_vectors_8"]
        v_1 = pca_vectors[:, 0]
        lambda_1 = float(pca_lambdas[0])
        v_2 = pca_vectors[:, 1]
        lambda_2 = float(pca_lambdas[1])

        oned_rows = []
        # Include Static Centroid as baseline comparison
        oned_rows.append({
            "condition": "STATIC_CENTROID", "seed": None,
            "top1_accuracy": float(np.mean(df_sc["correctness"])),
            "mean_margin": float(np.mean(df_sc["true_class_margin"])),
            "median_margin": float(np.median(df_sc["true_class_margin"])),
            "margin_damage": float(np.mean(df_sc["margin_damage_vs_clean"])),
            "corr_to_incorr": int(df_sc["flip_corr_to_incorr"].sum()),
            "incorr_to_corr": int(df_sc["flip_incorr_to_corr"].sum())
        })

        for s in [25001, 25002, 25003]:
            # Natural PC1
            hk_pc1 = PatchInterventionHook(8, m_100, "NATURAL_PC1", mu=mu_8, seed=s, extra_args={"v_1": v_1, "lambda_1": lambda_1}, device=device)
            df_pc1 = eval_and_record(f"NATURAL_PC1_s{s}", 8, hk_pc1)
            oned_rows.append({
                "condition": "NATURAL_PC1", "seed": s,
                "top1_accuracy": float(np.mean(df_pc1["correctness"])),
                "mean_margin": float(np.mean(df_pc1["true_class_margin"])),
                "median_margin": float(np.median(df_pc1["true_class_margin"])),
                "margin_damage": float(np.mean(df_pc1["margin_damage_vs_clean"])),
                "corr_to_incorr": int(df_pc1["flip_corr_to_incorr"].sum()),
                "incorr_to_corr": int(df_pc1["flip_incorr_to_corr"].sum())
            })

            # Natural PC2
            hk_pc2 = PatchInterventionHook(8, m_100, "NATURAL_PC2", mu=mu_8, seed=s, extra_args={"v_2": v_2, "lambda_2": lambda_2}, device=device)
            df_pc2 = eval_and_record(f"NATURAL_PC2_s{s}", 8, hk_pc2)
            oned_rows.append({
                "condition": "NATURAL_PC2", "seed": s,
                "top1_accuracy": float(np.mean(df_pc2["correctness"])),
                "mean_margin": float(np.mean(df_pc2["true_class_margin"])),
                "median_margin": float(np.median(df_pc2["true_class_margin"])),
                "margin_damage": float(np.mean(df_pc2["margin_damage_vs_clean"])),
                "corr_to_incorr": int(df_pc2["flip_corr_to_incorr"].sum()),
                "incorr_to_corr": int(df_pc2["flip_incorr_to_corr"].sum())
            })

            # Random 1D
            u = rand_1d[s]
            hk_r1d = PatchInterventionHook(8, m_100, "RANDOM_1D", mu=mu_8, seed=s, extra_args={"u": u, "lambda_1": lambda_1}, device=device)
            df_r1d = eval_and_record(f"RANDOM_1D_s{s}", 8, hk_r1d)
            oned_rows.append({
                "condition": "RANDOM_1D", "seed": s,
                "top1_accuracy": float(np.mean(df_r1d["correctness"])),
                "mean_margin": float(np.mean(df_r1d["true_class_margin"])),
                "median_margin": float(np.median(df_r1d["true_class_margin"])),
                "margin_damage": float(np.mean(df_r1d["margin_damage_vs_clean"])),
                "corr_to_incorr": int(df_r1d["flip_corr_to_incorr"].sum()),
                "incorr_to_corr": int(df_r1d["flip_incorr_to_corr"].sum())
            })
        df_oned = pd.DataFrame(oned_rows)

        # --- Phase 5: Secondary Depth-Adapted Confirmation Check ---
        best_adv = -1e9
        best_adapted_d = None
        for d in [7, 8, 9, 10]:
            z_rows = df_depth[(df_depth["depth"] == d) & (df_depth["condition"] == "ZERO")]
            c_rows = df_depth[(df_depth["depth"] == d) & (df_depth["condition"] == "CENTROID")]
            if len(z_rows) > 0 and len(c_rows) > 0:
                z_dmg = z_rows.iloc[0]["margin_damage"]
                c_dmg = c_rows.iloc[0]["margin_damage"]
                if z_dmg >= 0.10:
                    adv = z_dmg - c_dmg
                    if adv > best_adv:
                        best_adv = adv
                        best_adapted_d = d

        if best_adapted_d is not None and best_adapted_d != 8:
            print(f"[{model_type.upper()}] Triggering SECONDARY DEPTH-ADAPTED CONFIRMATION at Depth {best_adapted_d} (Advantage: {best_adv:.4f})...")
            mu_ad = stats[f"mu_{best_adapted_d}"]
            sigma_ad = stats[f"sigma_{best_adapted_d}"]
            for frac in [0.25, 0.50, 0.75]:
                m_f = masks[frac]
                hk_z = PatchInterventionHook(best_adapted_d, m_f, "ZERO", device=device)
                eval_and_record(f"ZERO_d{best_adapted_d}_f{frac}_ADAPTED", best_adapted_d, hk_z)
                hk_c = PatchInterventionHook(best_adapted_d, m_f, "CENTROID", mu=mu_ad, device=device)
                eval_and_record(f"CENTROID_d{best_adapted_d}_f{frac}_ADAPTED", best_adapted_d, hk_c)
                for s in [22001, 22002, 22003]:
                    hk_g = PatchInterventionHook(best_adapted_d, m_f, "DIAGONAL_GAUSSIAN", mu=mu_ad, sigma=sigma_ad, seed=s, device=device)
                    eval_and_record(f"GAUSSIAN_d{best_adapted_d}_f{frac}_s{s}_ADAPTED", best_adapted_d, hk_g)
                for s in [23001, 23002, 23003]:
                    perm = coord_perms[s]
                    hk_p = PatchInterventionHook(best_adapted_d, m_f, "COORDINATE_PERMUTED_CENTROID", mu=mu_ad, extra_args={"perm": perm}, device=device)
                    eval_and_record(f"COORD_PERM_d{best_adapted_d}_f{frac}_s{s}_ADAPTED", best_adapted_d, hk_p)
                hk_sf = PatchInterventionHook(best_adapted_d, m_f, "SIGN_FLIPPED_CENTROID", mu=mu_ad, device=device)
                eval_and_record(f"SIGN_FLIPPED_CENTROID_d{best_adapted_d}_f{frac}_ADAPTED", best_adapted_d, hk_sf)

            hk_sc = PatchInterventionHook(best_adapted_d, m_100, "STATIC_CENTROID", mu=mu_ad, device=device)
            eval_and_record(f"STATIC_CENTROID_100_d{best_adapted_d}_ADAPTED", best_adapted_d, hk_sc)
            for s in [24001, 24002, 24003, 24004, 24005]:
                hk_shared = PatchInterventionHook(best_adapted_d, m_100, "SHARED_GAUSSIAN_NOISE", mu=mu_ad, sigma=sigma_ad, seed=s, device=device)
                eval_and_record(f"SHARED_GAUSSIAN_s{s}_d{best_adapted_d}_ADAPTED", best_adapted_d, hk_shared)
                hk_ind = PatchInterventionHook(best_adapted_d, m_100, "INDEPENDENT_GAUSSIAN_NOISE", mu=mu_ad, sigma=sigma_ad, seed=s, device=device)
                eval_and_record(f"INDEPENDENT_GAUSSIAN_s{s}_d{best_adapted_d}_ADAPTED", best_adapted_d, hk_ind)

        return image_results, df_depth, df_frac_geom, df_diversity, df_oned


    # Run ViT-B experiments
    print("\n================== EVALUATING MODEL A: ViT-B AugReg ==================")
    t0_vb = time.time()
    img_res_vitb, df_depth_vb, df_frac_vb, df_div_vb, df_oned_vb = run_model_experiments(
        vitb_model, "vitb", eval_h_vitb, eval_labels_vitb, stats_vitb, vitb_masks,
        coord_perms_vitb, rand_1d_vitb, batch_size_vitb
    )
    vb_time = time.time() - t0_vb
    print(f"ViT-B evaluation completed in {vb_time:.2f}s")

    # Run DINOv2 experiments
    print("\n================== EVALUATING MODEL B: DINOv2 ViT-S/14 ==================")
    t0_dino = time.time()
    img_res_dino, df_depth_dino, df_frac_dino, df_div_dino, df_oned_dino = run_model_experiments(
        dinov2_model, "dinov2", eval_h_dinov2, eval_labels_dinov2, stats_dinov2, dinov2_masks,
        coord_perms_dinov2, rand_1d_dinov2, batch_size_dinov2
    )
    dino_time = time.time() - t0_dino
    print(f"DINOv2 evaluation completed in {dino_time:.2f}s")

    # Save summary CSVs
    print("[V1 Pipeline] Saving CSV tables...")
    df_depth_vb.to_csv(os.path.join(output_dir, "vitb_depth_results.csv"), index=False)
    df_depth_dino.to_csv(os.path.join(output_dir, "dinov2_depth_results.csv"), index=False)

    df_frac_vb.to_csv(os.path.join(output_dir, "vitb_fraction_results.csv"), index=False)
    df_frac_dino.to_csv(os.path.join(output_dir, "dinov2_fraction_results.csv"), index=False)

    # Geometry results split
    geom_conds = ["CENTROID", "COORDINATE_PERMUTED_CENTROID", "SIGN_FLIPPED_CENTROID"]
    df_geom_vb = df_frac_vb[df_frac_vb["condition"].isin(geom_conds)].copy()
    df_geom_dino = df_frac_dino[df_frac_dino["condition"].isin(geom_conds)].copy()
    df_geom_vb.to_csv(os.path.join(output_dir, "vitb_geometry_results.csv"), index=False)
    df_geom_dino.to_csv(os.path.join(output_dir, "dinov2_geometry_results.csv"), index=False)

    df_div_vb.to_csv(os.path.join(output_dir, "vitb_diversity_results.csv"), index=False)
    df_div_dino.to_csv(os.path.join(output_dir, "dinov2_diversity_results.csv"), index=False)

    df_oned_vb.to_csv(os.path.join(output_dir, "vitb_1d_results.csv"), index=False)
    df_oned_dino.to_csv(os.path.join(output_dir, "dinov2_1d_results.csv"), index=False)

    # Consolidate and save image results parquets
    print("[V1 Pipeline] Saving image-level parquets...")
    def consolidate_parquets(img_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        dfs = []
        for cond, df in img_dict.items():
            df_c = df.copy()
            df_c["condition_key"] = cond
            dfs.append(df_c)
        return pd.concat(dfs, ignore_index=True)

    parquet_vb = consolidate_parquets(img_res_vitb)
    parquet_dino = consolidate_parquets(img_res_dino)
    parquet_vb.to_parquet(os.path.join(output_dir, "vitb_image_results.parquet"), index=False)
    parquet_dino.to_parquet(os.path.join(output_dir, "dinov2_image_results.parquet"), index=False)

    # 9. Evaluate Signatures & Outcomes
    print("[V1 Pipeline] Evaluating Signatures and Decision Engine...")
    vitb_eval = evaluate_model_signatures("ViT-B AugReg", df_depth_vb, df_frac_vb, df_div_vb, img_res_vitb)
    dinov2_eval = evaluate_model_signatures("DINOv2 ViT-S/14", df_depth_dino, df_frac_dino, df_div_dino, img_res_dino)

    # 10. Check protocol assertions
    calib_indices = [int(x) for x in calib_manifest["global_index"].values]
    eval_indices = [int(x) for x in eval_manifest["global_index"].values]
    val_status = verify_v1_protocol_assertions(
        vitb_model, dinov2_model, manual_val_results, calib_indices, eval_indices,
        vitb_masks, dinov2_masks, coord_perms_vitb, coord_perms_dinov2,
        {"vitb": stats_vitb, "dinov2": stats_dinov2}, list(target_depths),
        smoke_test=smoke_test
    )

    with open(os.path.join(output_dir, "validation_results.json"), "w") as f:
        json.dump(val_status, f, indent=2)

    decision = evaluate_v1_decision(vitb_eval, dinov2_eval, val_status)
    with open(os.path.join(output_dir, "decision_summary.json"), "w") as f:
        json.dump({
            "decision": decision,
            "vitb_evaluation": vitb_eval,
            "dinov2_evaluation": dinov2_eval
        }, f, indent=2)

    # Experiment manifest
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0
    manifest = {
        "experiment": "PATCH_FUNGIBILITY_V1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "smoke_test": smoke_test,
        "device": str(device),
        "peak_vram_gb": peak_vram_gb,
        "vitb_runtime_sec": vb_time,
        "dinov2_runtime_sec": dino_time,
        "decision": decision,
        "validation_passed": val_status["all_passed"]
    }
    with open(os.path.join(output_dir, "experiment_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n================== V1 EVALUATION COMPLETE ==================")
    print(f"OUTCOME: {decision['outcome']}")
    print(f"CLAIM LEVEL: {decision['supported_level']}")
    print(f"INTERPRETATION: {decision['interpretation']}")
    print(f"READY FOR PAPER: {decision['ready_for_paper']}")
    print(f"ViT-B Signatures: {decision['vitb_signatures']}")
    print(f"DINOv2 Signatures: {decision['dinov2_signatures']}")

    return {
        "decision": decision,
        "vitb_eval": vitb_eval,
        "dinov2_eval": dinov2_eval,
        "val_status": val_status
    }
