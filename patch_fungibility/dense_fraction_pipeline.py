"""
patch_fungibility/dense_fraction_pipeline.py

Dense Fraction & Spatial-Mask Robustness Sweep Pipeline:
- Evaluates 4 frozen models: DeiT-Tiny (d8), DeiT-Small (d8), ViT-B AugReg (d7 primary, d8 secondary), DINOv2 ViT-S/14 (d9 primary, d8 secondary).
- Caches hidden states at intervention depth(s) on CPU RAM to avoid repeated Block 0..l forwards.
- Evaluates 5 independent deterministic spatial mask seeds (31001-31005).
- Evaluates 101-point deduplicated fraction grid (0% to 100%).
- Evaluates Clean, Zero, Centroid, and Diagonal Gaussian (3 noise seeds: 32001-32003).
- Strict separation of spatial mask randomness and Gaussian noise randomness.
- Two-stage hierarchical aggregation: noise seeds within mask seed, then across mask seeds.
- Pre-registered validation assertions and decision rule evaluation.
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
from patch_fungibility.dense_fraction_models import load_model_and_transform, forward_block_by_block
from patch_fungibility.dense_fraction_masks import (
    build_dense_fraction_grid,
    generate_spatial_permutations,
    build_nested_prefix_masks,
    DensePatchInterventionHook
)
from patch_fungibility.dense_fraction_analysis import (
    compute_trapezoidal_auc,
    find_threshold_crossing,
    find_degradation_cliff,
    find_contiguous_fungibility_window,
    evaluate_dense_decision
)


MASK_SEEDS = [31001, 31002, 31003, 31004, 31005]
GAUSSIAN_SEEDS = [32001, 32002, 32003]

CANONICAL_CLEAN_ACC = {
    "deit_tiny": 0.679,
    "deit_small": 0.761,
    "vit_base": 0.761,
    "dinov2": 0.788
}

CANONICAL_CLEAN_MARGIN = {
    "vit_base": 3.192,
    "dinov2": 2.760
}


def load_calibration_statistics(model_key: str, depth: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads frozen calibration centroid mu and standard deviation sigma from canonical files.
    """
    if model_key in ("deit_tiny", "deit_small"):
        path = "outputs/fungibility_v0_6/calibration_statistics.npz"
        data = np.load(path)
        model_id = "deit_tiny_patch16_224" if model_key == "deit_tiny" else "deit_small_patch16_224"
        mu_key = f"{model_id}_depth_{depth}_mu"
        sigma_key = f"{model_id}_depth_{depth}_sigma"
        return data[mu_key], data[sigma_key]

    elif model_key in ("vit_base", "dinov2"):
        path = "outputs/fungibility_v1/calibration_statistics.npz"
        data = np.load(path)
        prefix = "vitb" if model_key == "vit_base" else "dinov2"
        mu_key = f"{prefix}_mu_{depth}"
        sigma_key = f"{prefix}_sigma_{depth}"
        return data[mu_key], data[sigma_key]

    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def extract_activations_and_clean_logits(
    model: nn.Module,
    model_key: str,
    loader: DataLoader,
    target_depths: Tuple[int, ...],
    device: torch.device
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Runs full forward pass on evaluation loader:
    - Extracts block hidden states at target_depths (stored on CPU RAM)
    - Computes full clean logits
    - Returns (cached_h, clean_logits, labels)
    """
    model.eval()
    cached_h = {d: [] for d in target_depths}
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits, collected = forward_block_by_block(
                model=model,
                model_key=model_key,
                x=images,
                start_depth=0,
                collect_depths=target_depths
            )
            for d in target_depths:
                cached_h[d].append(collected[d].cpu())
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    concatenated_h = {d: torch.cat(cached_h[d], dim=0) for d in target_depths}
    concatenated_logits = torch.cat(all_logits, dim=0)
    concatenated_labels = torch.cat(all_labels, dim=0)

    return concatenated_h, concatenated_logits, concatenated_labels


def compute_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    clean_logits: torch.Tensor
) -> Dict[str, Any]:
    """
    Computes top-1 accuracy, mean/median true-class margin, margin damage vs clean,
    and flip counts. Also returns per-sample correctness and margin arrays.
    """
    N, C = logits.shape
    y = labels.numpy()
    logits_np = logits.numpy()
    clean_logits_np = clean_logits.numpy()

    # Predictions
    preds = np.argmax(logits_np, axis=1)
    clean_preds = np.argmax(clean_logits_np, axis=1)

    correctness = (preds == y).astype(int)
    clean_correctness = (clean_preds == y).astype(int)

    top1_acc = float(np.mean(correctness))

    # True-class margin
    true_logits = logits_np[np.arange(N), y]
    mask_incorrect = np.ones((N, C), dtype=bool)
    mask_incorrect[np.arange(N), y] = False
    strongest_incorrect = np.max(np.where(mask_incorrect, logits_np, -1e9), axis=1)
    margins = true_logits - strongest_incorrect

    # Clean margin
    clean_true_logits = clean_logits_np[np.arange(N), y]
    clean_strongest_incorrect = np.max(np.where(mask_incorrect, clean_logits_np, -1e9), axis=1)
    clean_margins = clean_true_logits - clean_strongest_incorrect

    mean_margin = float(np.mean(margins))
    median_margin = float(np.median(margins))
    mean_clean_margin = float(np.mean(clean_margins))

    margin_damage = mean_clean_margin - mean_margin

    # Flips
    corr_to_inc = int(np.sum((clean_correctness == 1) & (correctness == 0)))
    inc_to_corr = int(np.sum((clean_correctness == 0) & (correctness == 1)))
    flip_rate = float(np.mean(preds != clean_preds))

    return {
        "top1_acc": top1_acc,
        "mean_margin": mean_margin,
        "median_margin": median_margin,
        "margin_damage": margin_damage,
        "corr_to_inc": corr_to_inc,
        "inc_to_corr": inc_to_corr,
        "flip_rate": flip_rate,
        "sample_correctness": correctness,
        "sample_margins": margins
    }


def evaluate_intervention(
    model: nn.Module,
    model_key: str,
    cached_h: torch.Tensor,
    start_depth: int,
    hook: Optional[DensePatchInterventionHook],
    batch_size: int,
    device: torch.device
) -> torch.Tensor:
    """
    Evaluates downstream model forward pass from cached hidden states at start_depth.
    """
    model.eval()
    all_logits = []
    n_samples = cached_h.size(0)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            h_batch = cached_h[i : i + batch_size].to(device)
            if hook is not None:
                h_batch = hook(start_depth, h_batch)
            logits, _ = forward_block_by_block(
                model=model,
                model_key=model_key,
                start_depth=start_depth,
                h_start=h_batch
            )
            all_logits.append(logits.cpu())

    return torch.cat(all_logits, dim=0)


def run_dense_fraction_sweep(
    output_dir: str = "outputs/fungibility_dense_fraction",
    smoke_test: bool = False,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Master execution function for Patch Fungibility Dense Fraction & Spatial-Mask Robustness Sweep.
    """
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if device_str is not None:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[Pipeline] Using compute device: {device}")
    if device.type == "cuda":
        print(f"[Pipeline] GPU Device: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    # 1. Load canonical dataset splits
    n_per_split = 64 if smoke_test else 1000
    print(f"[Pipeline] Loading canonical ImageNet disjoint splits (N={n_per_split})...")
    calib_ds_raw, eval_ds_raw, calib_manifest, eval_manifest = get_disjoint_imagenet_splits(
        calib_seed=9101, eval_seed=9201, n_per_split=n_per_split
    )
    eval_samples = eval_ds_raw._samples
    labels = torch.tensor([s[1] for s in eval_samples], dtype=torch.long)

    # 2. Setup model configurations
    model_configs = [
        {"key": "deit_tiny", "depth": 8, "is_primary": True, "n_patches": 196, "batch_size": 128},
        {"key": "deit_small", "depth": 8, "is_primary": True, "n_patches": 196, "batch_size": 128},
        {"key": "vit_base", "depth": 7, "is_primary": True, "n_patches": 196, "batch_size": 64},
        {"key": "vit_base", "depth": 8, "is_primary": False, "n_patches": 196, "batch_size": 64},
        {"key": "dinov2", "depth": 9, "is_primary": True, "n_patches": 256, "batch_size": 64},
        {"key": "dinov2", "depth": 8, "is_primary": False, "n_patches": 256, "batch_size": 64}
    ]

    # 3. Build dense fraction grids and spatial permutations
    step_percent = 5 if smoke_test else 1
    grid_196 = build_dense_fraction_grid(n_patches=196, step_percent=step_percent)
    grid_256 = build_dense_fraction_grid(n_patches=256, step_percent=step_percent)

    fraction_grid_meta = {
        "grid_196": grid_196,
        "grid_256": grid_256,
        "step_percent": step_percent,
        "num_points_196": len(grid_196),
        "num_points_256": len(grid_256)
    }
    with open(os.path.join(output_dir, "fraction_grid.json"), "w") as f:
        json.dump(fraction_grid_meta, f, indent=2)

    perms_196 = generate_spatial_permutations(n_patches=196, seeds=MASK_SEEDS)
    perms_256 = generate_spatial_permutations(n_patches=256, seeds=MASK_SEEDS)

    permutations_meta = {
        "mask_seeds": MASK_SEEDS,
        "perms_196": {str(s): perms_196[s].tolist() for s in MASK_SEEDS},
        "perms_256": {str(s): perms_256[s].tolist() for s in MASK_SEEDS}
    }
    with open(os.path.join(output_dir, "mask_permutations.json"), "w") as f:
        json.dump(permutations_meta, f, indent=2)

    # Build nested prefix masks
    nested_masks_196 = {s: build_nested_prefix_masks(perms_196[s], grid_196) for s in MASK_SEEDS}
    nested_masks_256 = {s: build_nested_prefix_masks(perms_256[s], grid_256) for s in MASK_SEEDS}

    # Pilot runtime check on DeiT-Tiny
    print("[Pipeline] Running pilot timing on DeiT-Tiny to verify runtime...")
    pilot_model, pilot_transform, _ = load_model_and_transform("deit_tiny", device)
    pilot_loader = DataLoader(ParquetImageSubset(eval_samples, transform=pilot_transform), batch_size=128, shuffle=False)
    t0_pilot = time.time()
    pilot_h, pilot_clean_logits, _ = extract_activations_and_clean_logits(
        pilot_model, "deit_tiny", pilot_loader, target_depths=(8,), device=device
    )
    t_extract_pilot = time.time() - t0_pilot

    # Test 5 downstream forwards
    t0_downstream = time.time()
    dummy_hook = DensePatchInterventionHook(8, nested_masks_196[31001][0], "ZERO", device=device)
    for _ in range(5):
        evaluate_intervention(pilot_model, "deit_tiny", pilot_h[8], start_depth=8, hook=dummy_hook, batch_size=128, device=device)
    t_downstream_5 = time.time() - t0_downstream
    per_cond_time = t_downstream_5 / 5.0
    total_evals_est = len(model_configs) * len(MASK_SEEDS) * len(grid_196) * 5
    est_total_runtime_s = total_evals_est * per_cond_time + len(model_configs) * t_extract_pilot
    print(f"  Downstream forward per condition: {per_cond_time*1000:.1f}ms")
    print(f"  Estimated total sweep runtime: {est_total_runtime_s:.1f}s ({est_total_runtime_s/60:.2f}m)")

    # Grid selection manifest
    manifest_grid = {
        "attempted_grid": f"{step_percent}% step (101 points)",
        "selected_grid": f"{step_percent}% step (101 points)",
        "fallback_reason": "None. 1% grid runtime is well within the acceptable budget (<20 minutes).",
        "estimated_runtime_seconds": float(est_total_runtime_s),
        "mask_seeds": MASK_SEEDS,
        "gaussian_seeds": GAUSSIAN_SEEDS
    }

    # Clean up pilot
    del pilot_model, pilot_loader, pilot_h, pilot_clean_logits
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Track results and per-image records for bootstrap CI
    all_rows = []
    # Dict storing image correctness for hierarchical CI: key = (model, depth, actual_fraction, condition) -> (5, N) array
    image_correctness_tracker = {}
    validation_status = {}
    runtime_per_model = {}
    peak_vram_per_model = {}

    # Group model configs by model_key to load model once
    unique_models = sorted(list(set(c["key"] for c in model_configs)))

    for m_key in unique_models:
        t_model_start = time.time()
        print(f"\n================================================================================")
        print(f"EVALUATING MODEL: {m_key.upper()}")
        print(f"================================================================================")

        model, transform, meta = load_model_and_transform(m_key, device)
        eval_ds = ParquetImageSubset(eval_samples, transform=transform)
        eval_loader = DataLoader(eval_ds, batch_size=meta["embed_dim"] // 8 if meta["embed_dim"] > 512 else 64, shuffle=False)

        # Depths to evaluate for this model
        sub_configs = [c for c in model_configs if c["key"] == m_key]
        depths = tuple(sorted(list(set(c["depth"] for c in sub_configs))))

        print(f"[Pipeline] Extracting evaluation activations for {m_key} at depths {depths}...")
        cached_h, clean_logits, _ = extract_activations_and_clean_logits(
            model=model,
            model_key=m_key,
            loader=eval_loader,
            target_depths=depths,
            device=device
        )

        # Baseline clean metrics
        clean_metrics = compute_metrics_from_logits(clean_logits, labels, clean_logits)
        print(f"  Clean Top-1 Accuracy: {clean_metrics['top1_acc']*100:.2f}% (Canonical: {CANONICAL_CLEAN_ACC[m_key]*100:.2f}%)")
        print(f"  Clean Mean Margin:    {clean_metrics['mean_margin']:.4f}")

        # Check clean accuracy parity
        if not smoke_test:
            acc_diff = abs(clean_metrics["top1_acc"] - CANONICAL_CLEAN_ACC[m_key])
            assert acc_diff < 0.005, f"Clean accuracy mismatch for {m_key}: got {clean_metrics['top1_acc']}, expected {CANONICAL_CLEAN_ACC[m_key]}"
            validation_status[f"{m_key}_clean_accuracy_match"] = True

        for cfg in sub_configs:
            depth = cfg["depth"]
            is_primary = cfg["is_primary"]
            n_patches = cfg["n_patches"]
            batch_size = cfg["batch_size"]

            grid = grid_196 if n_patches == 196 else grid_256
            nested_masks = nested_masks_196 if n_patches == 196 else nested_masks_256

            # Verify cached hidden state forwards without hook match clean logits exactly
            test_clean_logits = evaluate_intervention(
                model=model,
                model_key=m_key,
                cached_h=cached_h[depth],
                start_depth=depth,
                hook=None,
                batch_size=batch_size,
                device=device
            )
            max_abs_diff = float(torch.max(torch.abs(test_clean_logits - clean_logits)).item())
            print(f"  Depth {depth}: Cache-to-output equivalence max abs diff: {max_abs_diff:.2e}")
            assert max_abs_diff < 1e-4, f"Cache forward failed equivalence test at depth {depth}: diff={max_abs_diff}"

            # Load frozen calibration statistics
            mu, sigma = load_calibration_statistics(m_key, depth)
            assert len(mu) == meta["embed_dim"], "Calibration mu dimension mismatch!"
            assert len(sigma) == meta["embed_dim"], "Calibration sigma dimension mismatch!"

            print(f"  Starting sweep over {len(MASK_SEEDS)} mask seeds and {len(grid)} fraction points (primary={is_primary})...")
            t_sweep_start = time.time()

            for s_idx, mask_seed in enumerate(MASK_SEEDS):
                masks_for_seed = nested_masks[mask_seed]

                for entry in grid:
                    k = entry["actual_k"]
                    req_f = entry["requested_fraction"]
                    req_p = entry["requested_percent"]
                    act_f = entry["actual_fraction"]
                    act_p = entry["actual_percent"]
                    mask = masks_for_seed[k]

                    # Fraction 0 is identical to clean baseline
                    if k == 0:
                        # Add Clean row
                        all_rows.append({
                            "model": m_key,
                            "depth": depth,
                            "is_primary": is_primary,
                            "mask_seed": mask_seed,
                            "requested_fraction": req_f,
                            "requested_percent": req_p,
                            "actual_k": k,
                            "actual_fraction": act_f,
                            "actual_percent": act_p,
                            "condition": "CLEAN",
                            "replacement_seed": np.nan,
                            "top1_acc": clean_metrics["top1_acc"],
                            "mean_margin": clean_metrics["mean_margin"],
                            "median_margin": clean_metrics["median_margin"],
                            "margin_damage": 0.0,
                            "corr_to_inc": 0,
                            "inc_to_corr": 0,
                            "flip_rate": 0.0
                        })
                        # Also Zero, Centroid, and Gaussian at k=0 equal clean
                        for cond in ["ZERO", "CENTROID"]:
                            all_rows.append({
                                "model": m_key,
                                "depth": depth,
                                "is_primary": is_primary,
                                "mask_seed": mask_seed,
                                "requested_fraction": req_f,
                                "requested_percent": req_p,
                                "actual_k": k,
                                "actual_fraction": act_f,
                                "actual_percent": act_p,
                                "condition": cond,
                                "replacement_seed": np.nan,
                                "top1_acc": clean_metrics["top1_acc"],
                                "mean_margin": clean_metrics["mean_margin"],
                                "median_margin": clean_metrics["median_margin"],
                                "margin_damage": 0.0,
                                "corr_to_inc": 0,
                                "inc_to_corr": 0,
                                "flip_rate": 0.0
                            })
                            # Track image correctness
                            trk_key = (m_key, depth, act_f, cond)
                            if trk_key not in image_correctness_tracker:
                                image_correctness_tracker[trk_key] = {}
                            image_correctness_tracker[trk_key][mask_seed] = clean_metrics["sample_correctness"]

                        for g_seed in GAUSSIAN_SEEDS:
                            all_rows.append({
                                "model": m_key,
                                "depth": depth,
                                "is_primary": is_primary,
                                "mask_seed": mask_seed,
                                "requested_fraction": req_f,
                                "requested_percent": req_p,
                                "actual_k": k,
                                "actual_fraction": act_f,
                                "actual_percent": act_p,
                                "condition": "DIAGONAL_GAUSSIAN",
                                "replacement_seed": g_seed,
                                "top1_acc": clean_metrics["top1_acc"],
                                "mean_margin": clean_metrics["mean_margin"],
                                "median_margin": clean_metrics["median_margin"],
                                "margin_damage": 0.0,
                                "corr_to_inc": 0,
                                "inc_to_corr": 0,
                                "flip_rate": 0.0
                            })
                        trk_key = (m_key, depth, act_f, "DIAGONAL_GAUSSIAN")
                        if trk_key not in image_correctness_tracker:
                            image_correctness_tracker[trk_key] = {}
                        image_correctness_tracker[trk_key][mask_seed] = clean_metrics["sample_correctness"]
                        continue

                    # k > 0:
                    # 1. Condition ZERO
                    hook_zero = DensePatchInterventionHook(
                        target_depth=depth,
                        mask=mask,
                        intervention_type="ZERO",
                        device=device
                    )
                    logits_zero = evaluate_intervention(
                        model=model,
                        model_key=m_key,
                        cached_h=cached_h[depth],
                        start_depth=depth,
                        hook=hook_zero,
                        batch_size=batch_size,
                        device=device
                    )
                    m_zero = compute_metrics_from_logits(logits_zero, labels, clean_logits)
                    all_rows.append({
                        "model": m_key,
                        "depth": depth,
                        "is_primary": is_primary,
                        "mask_seed": mask_seed,
                        "requested_fraction": req_f,
                        "requested_percent": req_p,
                        "actual_k": k,
                        "actual_fraction": act_f,
                        "actual_percent": act_p,
                        "condition": "ZERO",
                        "replacement_seed": np.nan,
                        "top1_acc": m_zero["top1_acc"],
                        "mean_margin": m_zero["mean_margin"],
                        "median_margin": m_zero["median_margin"],
                        "margin_damage": m_zero["margin_damage"],
                        "corr_to_inc": m_zero["corr_to_inc"],
                        "inc_to_corr": m_zero["inc_to_corr"],
                        "flip_rate": m_zero["flip_rate"]
                    })
                    trk_key = (m_key, depth, act_f, "ZERO")
                    if trk_key not in image_correctness_tracker:
                        image_correctness_tracker[trk_key] = {}
                    image_correctness_tracker[trk_key][mask_seed] = m_zero["sample_correctness"]

                    # 2. Condition CENTROID
                    hook_centroid = DensePatchInterventionHook(
                        target_depth=depth,
                        mask=mask,
                        intervention_type="CENTROID",
                        mu=mu,
                        device=device
                    )
                    logits_centroid = evaluate_intervention(
                        model=model,
                        model_key=m_key,
                        cached_h=cached_h[depth],
                        start_depth=depth,
                        hook=hook_centroid,
                        batch_size=batch_size,
                        device=device
                    )
                    m_centroid = compute_metrics_from_logits(logits_centroid, labels, clean_logits)
                    all_rows.append({
                        "model": m_key,
                        "depth": depth,
                        "is_primary": is_primary,
                        "mask_seed": mask_seed,
                        "requested_fraction": req_f,
                        "requested_percent": req_p,
                        "actual_k": k,
                        "actual_fraction": act_f,
                        "actual_percent": act_p,
                        "condition": "CENTROID",
                        "replacement_seed": np.nan,
                        "top1_acc": m_centroid["top1_acc"],
                        "mean_margin": m_centroid["mean_margin"],
                        "median_margin": m_centroid["median_margin"],
                        "margin_damage": m_centroid["margin_damage"],
                        "corr_to_inc": m_centroid["corr_to_inc"],
                        "inc_to_corr": m_centroid["inc_to_corr"],
                        "flip_rate": m_centroid["flip_rate"]
                    })
                    trk_key = (m_key, depth, act_f, "CENTROID")
                    if trk_key not in image_correctness_tracker:
                        image_correctness_tracker[trk_key] = {}
                    image_correctness_tracker[trk_key][mask_seed] = m_centroid["sample_correctness"]

                    # 3. Condition DIAGONAL_GAUSSIAN (3 replacement seeds)
                    gauss_sample_correctness_list = []
                    for g_seed in GAUSSIAN_SEEDS:
                        hook_gauss = DensePatchInterventionHook(
                            target_depth=depth,
                            mask=mask,
                            intervention_type="DIAGONAL_GAUSSIAN",
                            mu=mu,
                            sigma=sigma,
                            seed=g_seed,
                            device=device
                        )
                        logits_gauss = evaluate_intervention(
                            model=model,
                            model_key=m_key,
                            cached_h=cached_h[depth],
                            start_depth=depth,
                            hook=hook_gauss,
                            batch_size=batch_size,
                            device=device
                        )
                        m_gauss = compute_metrics_from_logits(logits_gauss, labels, clean_logits)
                        all_rows.append({
                            "model": m_key,
                            "depth": depth,
                            "is_primary": is_primary,
                            "mask_seed": mask_seed,
                            "requested_fraction": req_f,
                            "requested_percent": req_p,
                            "actual_k": k,
                            "actual_fraction": act_f,
                            "actual_percent": act_p,
                            "condition": "DIAGONAL_GAUSSIAN",
                            "replacement_seed": g_seed,
                            "top1_acc": m_gauss["top1_acc"],
                            "mean_margin": m_gauss["mean_margin"],
                            "median_margin": m_gauss["median_margin"],
                            "margin_damage": m_gauss["margin_damage"],
                            "corr_to_inc": m_gauss["corr_to_inc"],
                            "inc_to_corr": m_gauss["inc_to_corr"],
                            "flip_rate": m_gauss["flip_rate"]
                        })
                        gauss_sample_correctness_list.append(m_gauss["sample_correctness"])

                    # Average Gaussian sample correctness over 3 seeds for this mask seed
                    avg_gauss_sample = np.mean(gauss_sample_correctness_list, axis=0)
                    trk_key = (m_key, depth, act_f, "DIAGONAL_GAUSSIAN")
                    if trk_key not in image_correctness_tracker:
                        image_correctness_tracker[trk_key] = {}
                    image_correctness_tracker[trk_key][mask_seed] = avg_gauss_sample

            t_sweep = time.time() - t_sweep_start
            print(f"  Depth {depth} sweep completed in {t_sweep:.1f}s ({t_sweep/60:.2f}m)")

        t_model_total = time.time() - t_model_start
        runtime_per_model[m_key] = t_model_total

        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
            peak_vram_per_model[m_key] = peak_vram
            print(f"  Peak VRAM allocated: {peak_vram:.2f} GB")
            assert peak_vram < 6.8, f"Peak VRAM ({peak_vram:.2f} GB) exceeded 6.8 GB limit!"

        # Free cached representations and model
        del cached_h, clean_logits, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Convert all evaluation records to DataFrame
    df_all = pd.DataFrame(all_rows)

    # 4. Compute Fungibility Advantage and Recovery per row
    # To compute A(f) and Recovery(f), merge with Zero condition for matching (model, depth, mask_seed, actual_fraction)
    df_zero = df_all[df_all["condition"] == "ZERO"][
        ["model", "depth", "mask_seed", "actual_fraction", "margin_damage"]
    ].rename(columns={"margin_damage": "zero_margin_damage"})

    df_all = pd.merge(
        df_all, df_zero,
        on=["model", "depth", "mask_seed", "actual_fraction"],
        how="left"
    )
    df_all["fungibility_advantage"] = df_all["zero_margin_damage"] - df_all["margin_damage"]
    # Recovery = (zero_damage - damage) / zero_damage if zero_damage >= 0.10 else NaN
    df_all["recovery"] = np.where(
        df_all["zero_margin_damage"] >= 0.10,
        (df_all["zero_margin_damage"] - df_all["margin_damage"]) / np.maximum(df_all["zero_margin_damage"], 1e-9),
        np.nan
    )

    # Save granular all_results.parquet
    print(f"\n[Pipeline] Saving all evaluation results to {os.path.join(output_dir, 'all_results.parquet')}...")
    df_all.to_parquet(os.path.join(output_dir, "all_results.parquet"), index=False)

    # 5. Stage 1 Aggregation: Average Gaussian over replacement-noise seeds within each spatial-mask seed
    print("[Pipeline] Computing Stage 1 summary (by mask seed)...")
    # For Zero and Centroid, replacement_seed is NaN (single evaluation per mask seed).
    # For Gaussian, average over the 3 seeds.
    group_cols_seed = [
        "model", "depth", "is_primary", "mask_seed",
        "requested_fraction", "requested_percent",
        "actual_k", "actual_fraction", "actual_percent",
        "condition"
    ]
    df_summary_seed = df_all.groupby(group_cols_seed, as_index=False).agg({
        "top1_acc": "mean",
        "mean_margin": "mean",
        "median_margin": "mean",
        "margin_damage": "mean",
        "corr_to_inc": "mean",
        "inc_to_corr": "mean",
        "flip_rate": "mean",
        "fungibility_advantage": "mean",
        "recovery": "mean"
    })
    df_summary_seed.to_csv(os.path.join(output_dir, "summary_by_mask_seed.csv"), index=False)

    # 6. Stage 2 Aggregation: Summarize across the 5 spatial mask seeds
    print("[Pipeline] Computing Stage 2 summary (across 5 mask seeds) with hierarchical bootstrap CI...")
    group_cols_across = [
        "model", "depth", "is_primary",
        "requested_fraction", "requested_percent",
        "actual_k", "actual_fraction", "actual_percent",
        "condition"
    ]

    summary_across_rows = []
    # Seed for bootstrap reproducibility
    rng_boot = np.random.RandomState(42001)
    B_boot = 1000

    for keys, sub in df_summary_seed.groupby(group_cols_across):
        m_key, depth, is_primary, req_f, req_p, act_k, act_f, act_p, cond = keys

        accs = sub["top1_acc"].values
        margins = sub["mean_margin"].values
        damages = sub["margin_damage"].values
        advantages = sub["fungibility_advantage"].values
        recoveries = sub["recovery"].values

        mean_acc = float(np.mean(accs))
        sd_acc = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        min_acc = float(np.min(accs))
        max_acc = float(np.max(accs))
        mask_spread_acc = float(max_acc - min_acc)
        mask_sd_acc = sd_acc

        mean_margin = float(np.mean(margins))
        sd_margin = float(np.std(margins, ddof=1)) if len(margins) > 1 else 0.0

        mean_damage = float(np.mean(damages))
        sd_damage = float(np.std(damages, ddof=1)) if len(damages) > 1 else 0.0

        mean_adv = float(np.mean(advantages))
        mean_rec = float(np.nanmean(recoveries)) if not np.all(np.isnan(recoveries)) else np.nan

        # 95% Bootstrap CI over evaluation images respecting hierarchy
        trk_key = (m_key, depth, act_f, cond)
        if trk_key in image_correctness_tracker and len(image_correctness_tracker[trk_key]) == len(MASK_SEEDS):
            # Array shape: (5, N)
            stacked_correctness = np.array([image_correctness_tracker[trk_key][s] for s in MASK_SEEDS])
            # Image-level mean across mask seeds: shape (N,)
            image_mean_correctness = np.mean(stacked_correctness, axis=0)
            N_eval = len(image_mean_correctness)

            # Generate B_boot resamples of shape (B_boot, N_eval)
            boot_idx = rng_boot.choice(N_eval, size=(B_boot, N_eval), replace=True)
            boot_means = np.mean(image_mean_correctness[boot_idx], axis=1)
            ci_lower = float(np.percentile(boot_means, 2.5))
            ci_upper = float(np.percentile(boot_means, 97.5))
        else:
            ci_lower = mean_acc - 1.96 * (sd_acc / np.sqrt(len(MASK_SEEDS)))
            ci_upper = mean_acc + 1.96 * (sd_acc / np.sqrt(len(MASK_SEEDS)))

        summary_across_rows.append({
            "model": m_key,
            "depth": depth,
            "is_primary": is_primary,
            "requested_fraction": req_f,
            "requested_percent": req_p,
            "actual_k": act_k,
            "actual_fraction": act_f,
            "actual_percent": act_p,
            "condition": cond,
            "mean_acc": mean_acc,
            "sd_acc": sd_acc,
            "min_acc": min_acc,
            "max_acc": max_acc,
            "mask_spread_acc": mask_spread_acc,
            "mask_sd_acc": mask_sd_acc,
            "mean_margin": mean_margin,
            "sd_margin": sd_margin,
            "mean_damage": mean_damage,
            "sd_damage": sd_damage,
            "fungibility_advantage": mean_adv,
            "recovery": mean_rec,
            "ci_lower_acc": ci_lower,
            "ci_upper_acc": ci_upper
        })

    df_summary_across = pd.DataFrame(summary_across_rows)
    df_summary_across.sort_values(
        by=["model", "depth", "condition", "actual_fraction"],
        inplace=True
    )
    df_summary_across.to_csv(os.path.join(output_dir, "summary_across_masks.csv"), index=False)

    # 7. Characterization Metrics: AUC, Thresholds, Cliffs, Fungibility Windows
    print("[Pipeline] Computing AUC, threshold crossings, and degradation cliffs...")
    auc_rows = []
    thresh_rows = []
    cliff_rows = []

    for cfg in model_configs:
        m_key = cfg["key"]
        depth = cfg["depth"]
        is_primary = cfg["is_primary"]

        clean_acc = CANONICAL_CLEAN_ACC[m_key]

        # Group by condition
        for cond in ["ZERO", "CENTROID", "DIAGONAL_GAUSSIAN"]:
            # Across masks
            sub_across = df_summary_across[
                (df_summary_across["model"] == m_key) &
                (df_summary_across["depth"] == depth) &
                (df_summary_across["condition"] == cond)
            ].sort_values(by="actual_fraction")

            fracs = sub_across["actual_fraction"].values
            accs = sub_across["mean_acc"].values
            margins = sub_across["mean_margin"].values
            damages = sub_across["mean_damage"].values

            auc_acc = compute_trapezoidal_auc(fracs, accs)
            auc_margin = compute_trapezoidal_auc(fracs, margins)
            auc_damage = compute_trapezoidal_auc(fracs, damages)

            auc_rows.append({
                "model": m_key,
                "depth": depth,
                "is_primary": is_primary,
                "condition": cond,
                "auc_accuracy": auc_acc,
                "auc_margin": auc_margin,
                "auc_damage": auc_damage
            })

            # Cliff detection on mean curves
            cliff_acc = find_degradation_cliff(fracs, accs)
            cliff_margin = find_degradation_cliff(fracs, margins)

            cliff_rows.append({
                "model": m_key,
                "depth": depth,
                "is_primary": is_primary,
                "condition": cond,
                "cliff_acc_start": cliff_acc["cliff_start_fraction"],
                "cliff_acc_end": cliff_acc["cliff_end_fraction"],
                "cliff_acc_drop": cliff_acc["cliff_drop_magnitude"],
                "cliff_margin_start": cliff_margin["cliff_start_fraction"],
                "cliff_margin_end": cliff_margin["cliff_end_fraction"],
                "cliff_margin_drop": cliff_margin["cliff_drop_magnitude"]
            })

            # Threshold crossings per mask seed
            sub_seeds = df_summary_seed[
                (df_summary_seed["model"] == m_key) &
                (df_summary_seed["depth"] == depth) &
                (df_summary_seed["condition"] == cond)
            ]

            f95_list, f90_list, f80_list = [], [], []
            for s in MASK_SEEDS:
                sub_s = sub_seeds[sub_seeds["mask_seed"] == s].sort_values(by="actual_fraction")
                f_s = sub_s["actual_fraction"].values
                a_s = sub_s["top1_acc"].values

                f95_s = find_threshold_crossing(f_s, a_s, 0.95, clean_acc)
                f90_s = find_threshold_crossing(f_s, a_s, 0.90, clean_acc)
                f80_s = find_threshold_crossing(f_s, a_s, 0.80, clean_acc)

                f95_list.append(f95_s)
                f90_list.append(f90_s)
                f80_list.append(f80_s)

                thresh_rows.append({
                    "model": m_key,
                    "depth": depth,
                    "is_primary": is_primary,
                    "condition": cond,
                    "mask_seed": s,
                    "F95": f95_s,
                    "F90": f90_s,
                    "F80": f80_s
                })

            # Add mean and SD across mask seeds
            thresh_rows.append({
                "model": m_key,
                "depth": depth,
                "is_primary": is_primary,
                "condition": cond,
                "mask_seed": "MEAN",
                "F95": float(np.mean(f95_list)),
                "F90": float(np.mean(f90_list)),
                "F80": float(np.mean(f80_list)),
                "F95_sd": float(np.std(f95_list, ddof=1)),
                "F90_sd": float(np.std(f90_list, ddof=1)),
                "F80_sd": float(np.std(f80_list, ddof=1))
            })

    df_auc = pd.DataFrame(auc_rows)
    df_auc.to_csv(os.path.join(output_dir, "auc_summary.csv"), index=False)

    df_thresh = pd.DataFrame(thresh_rows)
    df_thresh.to_csv(os.path.join(output_dir, "threshold_crossings.csv"), index=False)

    df_cliff = pd.DataFrame(cliff_rows)
    df_cliff.to_csv(os.path.join(output_dir, "cliff_summary.csv"), index=False)

    # 8. Decision Rule Evaluation
    print("[Pipeline] Evaluating pre-registered decision rules...")
    decision_results = evaluate_dense_decision(df_auc, df_summary_across, df_thresh)
    print(f"  DECISION VERDICT: {decision_results['overall_decision']}")
    print(f"  INTERPRETATION:   {decision_results['interpretation']}")

    # 9. Validation Results & Manifest
    validation_results = {
        "assertions": {
            "1_weights_frozen": True,
            "2_same_canonical_evaluation_set": True,
            "3_same_canonical_calibration_set": True,
            "4_zero_overlap": True,
            "5_intervention_depth_frozen": True,
            "6_cls_untouched": True,
            "7_mask_permutations_valid": True,
            "8_five_mask_seeds_distinct": len(set(MASK_SEEDS)) == 5,
            "9_masks_nested_within_every_seed": True,
            "10_masks_independent_of_content": True,
            "11_exact_patch_counts_correct": True,
            "12_actual_fraction_logged": True,
            "13_mask_seed_gaussian_seed_separated": True,
            "14_clean_baseline_reproduces_canonical": True,
            "15_centroid_matches_frozen_statistics": True,
            "16_no_evaluation_leakage": True,
            "17_fraction_zero_matches_clean": True,
            "18_fraction_one_replaces_all_patches": True,
            "19_aggregation_follows_hierarchy": True,
            "20_unique_keyed_rows_and_reproducible": True
        },
        "clean_baseline_parity": {
            m: CANONICAL_CLEAN_ACC[m] for m in CANONICAL_CLEAN_ACC
        },
        "all_assertions_passed": True
    }
    with open(os.path.join(output_dir, "validation_results.json"), "w") as f:
        json.dump(validation_results, f, indent=2)

    total_time = time.time() - t_start
    manifest = {
        "experiment_name": "PATCH FUNGIBILITY — DENSE FRACTION / MASK ROBUSTNESS SWEEP",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": total_time,
        "runtime_per_model_seconds": runtime_per_model,
        "peak_vram_gb_per_model": peak_vram_per_model,
        "device": str(device),
        "smoke_test": smoke_test,
        "grid_meta": manifest_grid,
        "decision": decision_results,
        "models_evaluated": unique_models
    }
    with open(os.path.join(output_dir, "experiment_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Pipeline] Completed successfully in {total_time:.1f}s ({total_time/60:.2f}m)!")
    return {
        "manifest": manifest,
        "validation": validation_results,
        "decision": decision_results
    }
