import os
import time
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from patch_fungibility.models import load_deit_model, get_model_parameter_hash
from patch_fungibility.pipeline import compute_logits_and_margins
from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits
from patch_fungibility.v0_7_masks import get_nested_patch_masks_v0_7
from patch_fungibility.v0_7_prototypes import generate_v0_7_prototypes, cosine_sim
from patch_fungibility.metrics import (
    evaluate_paired_margin_diff,
    evaluate_paired_accuracy,
    benjamini_hochberg_fdr
)


def apply_vector_to_mask(h: torch.Tensor, seq_mask: List[int], vec: torch.Tensor) -> torch.Tensor:
    """
    Replaces tokens at seq_mask in h (B, T, D) with vec (D,).
    Leaves all other tokens (including CLS at index 0) strictly untouched.
    """
    h_mod = h.clone()
    h_mod[:, seq_mask, :] = vec.to(h.device)
    return h_mod


def apply_zero_to_mask(h: torch.Tensor, seq_mask: List[int]) -> torch.Tensor:
    h_mod = h.clone()
    h_mod[:, seq_mask, :] = 0.0
    return h_mod


def apply_gaussian_to_mask(
    h: torch.Tensor,
    seq_mask: List[int],
    mu: torch.Tensor,
    sigma: torch.Tensor,
    seed: int
) -> torch.Tensor:
    h_mod = h.clone()
    B = h.shape[0]
    M = len(seq_mask)
    D = h.shape[2]
    rng = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.randn((B, M, D), generator=rng, dtype=h.dtype)
    mu_bc = mu.to(h.device).unsqueeze(0).unsqueeze(0)
    sigma_bc = sigma.to(h.device).unsqueeze(0).unsqueeze(0)
    samples = mu_bc + eps.to(h.device) * sigma_bc
    h_mod[:, seq_mask, :] = samples
    return h_mod


class PatchFungibilityV07Pipeline:
    def __init__(
        self,
        model_name: str = "deit_tiny_patch16_224",
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "outputs/fungibility_v0_7",
        fractions: List[str] = ["25%", "50%", "75%", "100%"],
        gaussian_seeds: List[int] = [9701, 9702, 9703],
        coord_perm_seeds: List[int] = [9801, 9802, 9803],
        sign_flip_seed: int = 9901,
        cosine_seeds: List[int] = [10001, 10002, 10003],
        target_cosines: List[float] = [0.0, 0.25, 0.50, 0.75],
        scale_factors: List[float] = [0.25, 0.50, 1.00, 2.00, 4.00],
        wrong_depths: List[int] = [5, 6, 7, 9, 10],
        mask_seed: int = 9601,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.output_dir = output_dir
        self.target_depth = 8
        self.fractions = fractions
        self.gaussian_seeds = gaussian_seeds
        self.coord_perm_seeds = coord_perm_seeds
        self.sign_flip_seed = sign_flip_seed
        self.cosine_seeds = cosine_seeds
        self.target_cosines = target_cosines
        self.scale_factors = scale_factors
        self.wrong_depths = wrong_depths
        self.mask_seed = mask_seed

        os.makedirs(self.output_dir, exist_ok=True)
        self.nested_masks = get_nested_patch_masks_v0_7(seed=self.mask_seed)

    def run(
        self,
        calib_loader: DataLoader,
        eval_loader: DataLoader,
        n_samples: int = 1000
    ) -> Dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.time()
        model, initial_hash = load_deit_model(self.model_name, self.device)
        total_blocks = len(model.blocks)
        embed_dim = model.embed_dim

        # -------------------------------------------------------------
        # STEP 1: Generate & Validate Prototype Vectors
        # -------------------------------------------------------------
        print(f"\n[{self.model_name}] Step 1: Generating V0.7 Prototype Vectors...")
        proto_data = generate_v0_7_prototypes(
            model_name=self.model_name,
            calibration_stats_npz_path="outputs/fungibility_v0_6/calibration_statistics.npz",
            gaussian_seeds=self.gaussian_seeds,
            perm_seeds=self.coord_perm_seeds,
            sign_seed=self.sign_flip_seed,
            cosine_seeds=self.cosine_seeds,
            target_cosines=self.target_cosines,
            scale_factors=self.scale_factors,
            wrong_depths=self.wrong_depths
        )
        vectors = proto_data["vectors"]
        mu_8_tensor = torch.from_numpy(vectors["mu_8"]).to(self.device)
        sigma_8_tensor = torch.from_numpy(proto_data["sigma_8"]).to(self.device)

        # -------------------------------------------------------------
        # STEP 2: Compute Calibration Mean CLS Vector (for Optional Asymmetry Control)
        # -------------------------------------------------------------
        print(f"[{self.model_name}] Step 2: Computing Calibration Mean CLS Vector for Asymmetry Test...")
        cls_calib_tokens = []
        with torch.no_grad():
            for c_images, _ in calib_loader:
                c_images = c_images.to(self.device)
                feat = model.patch_embed(c_images)
                feat = model._pos_embed(feat)
                feat = model.patch_drop(feat)
                feat = model.norm_pre(feat)
                for b in range(self.target_depth + 1):
                    feat = model.blocks[b](feat)
                cls_calib_tokens.append(feat[:, 0, :].detach().cpu())
        mu_8_cls = torch.cat(cls_calib_tokens, dim=0).mean(dim=0).to(self.device)

        # -------------------------------------------------------------
        # STEP 3: Forward Evaluation Images to Depth 8 & Record Baseline
        # -------------------------------------------------------------
        print(f"[{self.model_name}] Step 3: Forwarding Evaluation Images & Caching Depth-8 Representations...")
        cached_h8: List[torch.Tensor] = []
        all_targets: List[int] = []
        baseline_preds: List[int] = []
        baseline_corrects: List[int] = []
        baseline_margins: List[float] = []

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(eval_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                feat = model.patch_embed(images)
                feat = model._pos_embed(feat)
                feat = model.patch_drop(feat)
                feat = model.norm_pre(feat)

                for b in range(self.target_depth + 1):
                    feat = model.blocks[b](feat)

                cached_h8.append(feat.detach().cpu())

                # Continue forward through blocks 9..11
                for b in range(self.target_depth + 1, total_blocks):
                    feat = model.blocks[b](feat)

                logits = model.forward_head(model.norm(feat))
                preds, corrects, _, _, margins = compute_logits_and_margins(logits, targets)

                all_targets.extend(targets.cpu().numpy().tolist())
                baseline_preds.extend(preds.cpu().numpy().tolist())
                baseline_corrects.extend(corrects.cpu().numpy().tolist())
                baseline_margins.extend(margins.cpu().numpy().tolist())

        n_eval = len(all_targets)
        eval_h8 = torch.cat(cached_h8, dim=0)  # Shape (N, 197, D)
        targets_tensor = torch.tensor(all_targets, dtype=torch.long)

        base_acc = float(np.mean(baseline_corrects))
        base_m = np.array(baseline_margins, dtype=np.float32)
        base_c = np.array(baseline_corrects, dtype=np.int32)
        base_p = np.array(baseline_preds, dtype=np.int32)
        print(f"[{self.model_name}] Clean Baseline: Acc = {base_acc*100:.2f}%, Mean Margin = {np.mean(base_m):.4f}")

        # -------------------------------------------------------------
        # STEP 4: Define Conditions List
        # -------------------------------------------------------------
        # Static vector conditions:
        static_cond_keys = [
            "mu_8",
            "variance_normalized",
        ]
        # Wrong depth
        for wd in self.wrong_depths:
            static_cond_keys.append(f"wrong_depth_{wd}")
            static_cond_keys.append(f"wrong_depth_norm_{wd}")
        # Coordinate permutations
        for s in self.coord_perm_seeds:
            static_cond_keys.append(f"coord_perm_seed_{s}")
        # Sign flips
        static_cond_keys.extend(["sign_flip_25%", "sign_flip_50%", "sign_flip_75%", "sign_flip_100%"])
        # Scale sweep
        for sc in self.scale_factors:
            static_cond_keys.append(f"scale_{sc:.2f}x")
        # Cosine sweep
        for alpha in self.target_cosines:
            for s in self.cosine_seeds:
                static_cond_keys.append(f"cosine_{alpha:.2f}_seed_{s}")

        # All condition names evaluated per fraction
        all_eval_conds = ["zero"] + static_cond_keys + [f"gaussian_seed_{s}" for s in self.gaussian_seeds]

        # -------------------------------------------------------------
        # STEP 5: Evaluate Interventions Across Fractions
        # -------------------------------------------------------------
        print(f"[{self.model_name}] Step 5: Evaluating Interventions Across Fractions (25%, 50%, 75%, 100%)...")
        fraction_data: Dict[str, Dict[str, Dict[str, List]]] = {}
        cls_untouched_all = True
        num_batches = (n_eval + self.batch_size - 1) // self.batch_size

        for frac in self.fractions:
            mask_info = self.nested_masks[frac]
            seq_mask = mask_info["seq_mask"]
            count = mask_info["count"]
            fraction_data[frac] = {c: {"preds": [], "corrects": [], "margins": []} for c in all_eval_conds}

            with torch.no_grad():
                for b_idx in range(num_batches):
                    b_start = b_idx * self.batch_size
                    b_end = min(b_start + self.batch_size, n_eval)

                    h_batch = eval_h8[b_start:b_end].to(self.device)
                    b_targets = targets_tensor[b_start:b_end].to(self.device)

                    for cond in all_eval_conds:
                        if cond == "zero":
                            h_mod = apply_zero_to_mask(h_batch, seq_mask)
                        elif cond.startswith("gaussian_seed_"):
                            s = int(cond.split("_")[-1])
                            h_mod = apply_gaussian_to_mask(h_batch, seq_mask, mu_8_tensor, sigma_8_tensor, seed=s + b_idx * 1000)
                        else:
                            # Static prototype vector
                            v_tensor = torch.from_numpy(vectors[cond]).to(self.device)
                            h_mod = apply_vector_to_mask(h_batch, seq_mask, v_tensor)

                        # Programmatic invariant: verify CLS token untouched
                        if not torch.allclose(h_mod[:, 0, :], h_batch[:, 0, :], atol=1e-6):
                            cls_untouched_all = False

                        # Pass through downstream blocks 9..11
                        h_curr = h_mod
                        for blk in range(self.target_depth + 1, total_blocks):
                            h_curr = model.blocks[blk](h_curr)

                        logits = model.forward_head(model.norm(h_curr))
                        preds, corrects, _, _, margins = compute_logits_and_margins(logits, b_targets)

                        fraction_data[frac][cond]["preds"].extend(preds.cpu().numpy().tolist())
                        fraction_data[frac][cond]["corrects"].extend(corrects.cpu().numpy().tolist())
                        fraction_data[frac][cond]["margins"].extend(margins.cpu().numpy().tolist())

        # -------------------------------------------------------------
        # STEP 6: Optional CLS Replacement Asymmetry Control
        # -------------------------------------------------------------
        print(f"[{self.model_name}] Step 6: Evaluating Optional CLS Replacement Asymmetry Control...")
        cls_control_names = ["cls_calib_mean", "cls_cross_image", "cls_zero"]
        cls_control_data = {c: {"preds": [], "corrects": [], "margins": []} for c in cls_control_names}

        with torch.no_grad():
            for b_idx in range(num_batches):
                b_start = b_idx * self.batch_size
                b_end = min(b_start + self.batch_size, n_eval)

                h_batch = eval_h8[b_start:b_end].to(self.device)
                b_targets = targets_tensor[b_start:b_end].to(self.device)

                for c_name in cls_control_names:
                    h_mod = h_batch.clone()
                    if c_name == "cls_calib_mean":
                        h_mod[:, 0, :] = mu_8_cls
                    elif c_name == "cls_cross_image":
                        h_mod[:, 0, :] = torch.roll(h_batch[:, 0, :], shifts=1, dims=0)
                    elif c_name == "cls_zero":
                        h_mod[:, 0, :] = 0.0

                    h_curr = h_mod
                    for blk in range(self.target_depth + 1, total_blocks):
                        h_curr = model.blocks[blk](h_curr)

                    logits = model.forward_head(model.norm(h_curr))
                    preds, corrects, _, _, margins = compute_logits_and_margins(logits, b_targets)

                    cls_control_data[c_name]["preds"].extend(preds.cpu().numpy().tolist())
                    cls_control_data[c_name]["corrects"].extend(corrects.cpu().numpy().tolist())
                    cls_control_data[c_name]["margins"].extend(margins.cpu().numpy().tolist())

        # -------------------------------------------------------------
        # STEP 7: Statistical Analysis & Programmatic Aggregation
        # -------------------------------------------------------------
        print(f"[{self.model_name}] Step 7: Performing Statistical Analysis (Bootstraps, Paired Tests, BH-FDR)...")
        # Precompute aggregated conditions per fraction
        # Gaussian average across 3 seeds
        for frac in self.fractions:
            g_margins = [np.array(fraction_data[frac][f"gaussian_seed_{s}"]["margins"]) for s in self.gaussian_seeds]
            g_corrects = [np.array(fraction_data[frac][f"gaussian_seed_{s}"]["corrects"]) for s in self.gaussian_seeds]
            g_avg_m = np.mean(g_margins, axis=0)
            g_avg_c = np.mean(g_corrects, axis=0)
            fraction_data[frac]["gaussian_avg"] = {"margins": g_avg_m, "corrects": g_avg_c}

            # Coordinate permutation average across 3 seeds
            p_margins = [np.array(fraction_data[frac][f"coord_perm_seed_{s}"]["margins"]) for s in self.coord_perm_seeds]
            p_corrects = [np.array(fraction_data[frac][f"coord_perm_seed_{s}"]["corrects"]) for s in self.coord_perm_seeds]
            fraction_data[frac]["coord_perm_avg"] = {"margins": np.mean(p_margins, axis=0), "corrects": np.mean(p_corrects, axis=0)}

            # Cosine averages per alpha across 3 seeds
            for alpha in self.target_cosines:
                cos_m = [np.array(fraction_data[frac][f"cosine_{alpha:.2f}_seed_{s}"]["margins"]) for s in self.cosine_seeds]
                cos_c = [np.array(fraction_data[frac][f"cosine_{alpha:.2f}_seed_{s}"]["corrects"]) for s in self.cosine_seeds]
                fraction_data[frac][f"cosine_{alpha:.2f}_avg"] = {"margins": np.mean(cos_m, axis=0), "corrects": np.mean(cos_c, axis=0)}

        # Build Fraction Summary Rows
        fraction_summary_rows = []
        for frac in self.fractions:
            m_zero = np.array(fraction_data[frac]["zero"]["margins"])
            c_zero = np.array(fraction_data[frac]["zero"]["corrects"])
            m_mu8 = np.array(fraction_data[frac]["mu_8"]["margins"])
            c_mu8 = np.array(fraction_data[frac]["mu_8"]["corrects"])
            m_gauss = fraction_data[frac]["gaussian_avg"]["margins"]
            c_gauss = fraction_data[frac]["gaussian_avg"]["corrects"]

            zero_dmg = float(np.mean(base_m - m_zero))
            mu8_dmg = float(np.mean(base_m - m_mu8))
            gauss_dmg = float(np.mean(base_m - m_gauss))

            retention = float((zero_dmg - mu8_dmg) / zero_dmg) if zero_dmg >= 0.10 else np.nan
            retention_str = f"{retention*100:.1f}%" if not np.isnan(retention) else "N/A"

            # Fraction of originally correct evaluation samples remaining correct under mu_8
            originally_correct_idx = np.where(base_c == 1)[0]
            retention_of_correct = float(np.mean(c_mu8[originally_correct_idx])) if len(originally_correct_idx) > 0 else 0.0

            fraction_summary_rows.append({
                "model": self.model_name,
                "fraction": frac,
                "clean_acc": base_acc,
                "zero_acc": float(np.mean(c_zero)),
                "mu8_acc": float(np.mean(c_mu8)),
                "gaussian_acc": float(np.mean(c_gauss)),
                "zero_damage": zero_dmg,
                "mu8_damage": mu8_dmg,
                "gaussian_damage": gauss_dmg,
                "prototype_retention": retention,
                "retention_of_originally_correct": retention_of_correct,
            })

        # Build Pairwise Contrasts with BH-FDR correction within families
        comparison_rows = []
        # Define contrast families
        # Contrast: mu_8 vs Control (Diff = mu_8 - Control, Advantage = Margin(mu_8) - Margin(Control))
        all_family_contrasts: Dict[str, List[Dict[str, Any]]] = {
            "wrong_depth_raw": [],
            "wrong_depth_norm": [],
            "coord_perm": [],
            "sign_flip": [],
            "scale_sweep": [],
            "cosine_sweep": [],
            "reference": [],
            "exploratory": []
        }

        for frac in self.fractions:
            m_mu8 = np.array(fraction_data[frac]["mu_8"]["margins"])
            c_mu8 = np.array(fraction_data[frac]["mu_8"]["corrects"])

            # 1. Reference (Zero, Gaussian avg)
            for ref_name, ref_m, ref_c in [
                ("zero", np.array(fraction_data[frac]["zero"]["margins"]), np.array(fraction_data[frac]["zero"]["corrects"])),
                ("gaussian_avg", fraction_data[frac]["gaussian_avg"]["margins"], (fraction_data[frac]["gaussian_avg"]["corrects"] >= 0.5).astype(int))
            ]:
                diff_stat = evaluate_paired_margin_diff(m_mu8, ref_m)
                acc_stat = evaluate_paired_accuracy(c_mu8, ref_c)
                all_family_contrasts["reference"].append({
                    "fraction": frac, "control": ref_name, "diff_stat": diff_stat, "acc_stat": acc_stat
                })

            # 2. Wrong depth raw
            for wd in self.wrong_depths:
                m_c = np.array(fraction_data[frac][f"wrong_depth_{wd}"]["margins"])
                c_c = np.array(fraction_data[frac][f"wrong_depth_{wd}"]["corrects"])
                diff_stat = evaluate_paired_margin_diff(m_mu8, m_c)
                acc_stat = evaluate_paired_accuracy(c_mu8, c_c)
                all_family_contrasts["wrong_depth_raw"].append({
                    "fraction": frac, "control": f"wrong_depth_{wd}", "diff_stat": diff_stat, "acc_stat": acc_stat
                })

            # 3. Wrong depth norm-matched
            for wd in self.wrong_depths:
                m_c = np.array(fraction_data[frac][f"wrong_depth_norm_{wd}"]["margins"])
                c_c = np.array(fraction_data[frac][f"wrong_depth_norm_{wd}"]["corrects"])
                diff_stat = evaluate_paired_margin_diff(m_mu8, m_c)
                acc_stat = evaluate_paired_accuracy(c_mu8, c_c)
                all_family_contrasts["wrong_depth_norm"].append({
                    "fraction": frac, "control": f"wrong_depth_norm_{wd}", "diff_stat": diff_stat, "acc_stat": acc_stat
                })

            # 4. Coordinate permutations
            for s in self.coord_perm_seeds:
                m_c = np.array(fraction_data[frac][f"coord_perm_seed_{s}"]["margins"])
                c_c = np.array(fraction_data[frac][f"coord_perm_seed_{s}"]["corrects"])
                diff_stat = evaluate_paired_margin_diff(m_mu8, m_c)
                acc_stat = evaluate_paired_accuracy(c_mu8, c_c)
                all_family_contrasts["coord_perm"].append({
                    "fraction": frac, "control": f"coord_perm_seed_{s}", "diff_stat": diff_stat, "acc_stat": acc_stat
                })
            # Also coord perm avg
            m_cp_avg = fraction_data[frac]["coord_perm_avg"]["margins"]
            c_cp_avg = (fraction_data[frac]["coord_perm_avg"]["corrects"] >= 0.5).astype(int)
            diff_stat = evaluate_paired_margin_diff(m_mu8, m_cp_avg)
            acc_stat = evaluate_paired_accuracy(c_mu8, c_cp_avg)
            all_family_contrasts["coord_perm"].append({
                "fraction": frac, "control": "coord_perm_avg", "diff_stat": diff_stat, "acc_stat": acc_stat
            })

            # 5. Sign flips
            for s_flip in ["sign_flip_25%", "sign_flip_50%", "sign_flip_75%", "sign_flip_100%"]:
                m_c = np.array(fraction_data[frac][s_flip]["margins"])
                c_c = np.array(fraction_data[frac][s_flip]["corrects"])
                diff_stat = evaluate_paired_margin_diff(m_mu8, m_c)
                acc_stat = evaluate_paired_accuracy(c_mu8, c_c)
                all_family_contrasts["sign_flip"].append({
                    "fraction": frac, "control": s_flip, "diff_stat": diff_stat, "acc_stat": acc_stat
                })

            # 6. Scale sweep
            for sc in self.scale_factors:
                if sc == 1.0:
                    continue
                sc_name = f"scale_{sc:.2f}x"
                m_c = np.array(fraction_data[frac][sc_name]["margins"])
                c_c = np.array(fraction_data[frac][sc_name]["corrects"])
                diff_stat = evaluate_paired_margin_diff(m_mu8, m_c)
                acc_stat = evaluate_paired_accuracy(c_mu8, c_c)
                all_family_contrasts["scale_sweep"].append({
                    "fraction": frac, "control": sc_name, "diff_stat": diff_stat, "acc_stat": acc_stat
                })

            # 7. Cosine sweep
            for alpha in self.target_cosines:
                for s in self.cosine_seeds:
                    c_name = f"cosine_{alpha:.2f}_seed_{s}"
                    m_c = np.array(fraction_data[frac][c_name]["margins"])
                    c_c = np.array(fraction_data[frac][c_name]["corrects"])
                    diff_stat = evaluate_paired_margin_diff(m_mu8, m_c)
                    acc_stat = evaluate_paired_accuracy(c_mu8, c_c)
                    all_family_contrasts["cosine_sweep"].append({
                        "fraction": frac, "control": c_name, "diff_stat": diff_stat, "acc_stat": acc_stat
                    })
                # Avg
                c_avg_name = f"cosine_{alpha:.2f}_avg"
                m_c = fraction_data[frac][c_avg_name]["margins"]
                c_c = (fraction_data[frac][c_avg_name]["corrects"] >= 0.5).astype(int)
                diff_stat = evaluate_paired_margin_diff(m_mu8, m_c)
                acc_stat = evaluate_paired_accuracy(c_mu8, c_c)
                all_family_contrasts["cosine_sweep"].append({
                    "fraction": frac, "control": c_avg_name, "diff_stat": diff_stat, "acc_stat": acc_stat
                })

            # 8. Variance normalized
            m_c = np.array(fraction_data[frac]["variance_normalized"]["margins"])
            c_c = np.array(fraction_data[frac]["variance_normalized"]["corrects"])
            diff_stat = evaluate_paired_margin_diff(m_mu8, m_c)
            acc_stat = evaluate_paired_accuracy(c_mu8, c_c)
            all_family_contrasts["exploratory"].append({
                "fraction": frac, "control": "variance_normalized", "diff_stat": diff_stat, "acc_stat": acc_stat
            })

        # Apply BH-FDR within each family
        for fam_name, contrast_list in all_family_contrasts.items():
            p_vals = [c["diff_stat"]["p_value"] for c in contrast_list]
            q_vals = benjamini_hochberg_fdr(p_vals)
            for idx, c in enumerate(contrast_list):
                diff_s = c["diff_stat"]
                acc_s = c["acc_stat"]
                comparison_rows.append({
                    "model": self.model_name,
                    "family": fam_name,
                    "fraction": c["fraction"],
                    "control": c["control"],
                    "prototype_advantage": diff_s["mean"],
                    "advantage_median": diff_s["median"],
                    "bootstrap_ci_lower": diff_s["bootstrap_ci_95"][0],
                    "bootstrap_ci_upper": diff_s["bootstrap_ci_95"][1],
                    "cohens_dz": diff_s["cohens_dz"],
                    "t_pval": diff_s["p_value"],
                    "fdr_qval": q_vals[idx],
                    "wilcoxon_pval": diff_s["wilcoxon_p_value"],
                    "acc_mu8": acc_s["acc_a"],
                    "acc_control": acc_s["acc_b"],
                    "acc_diff": acc_s["acc_diff"],
                    "mcnemar_pval": acc_s["exact_mcnemar_p_value"]
                })

        # Specific sweep summary tables for reporting
        scale_sweep_rows = []
        cosine_sweep_rows = []
        sign_flip_rows = []

        for frac in self.fractions:
            # Scale
            for sc in self.scale_factors:
                sc_name = f"scale_{sc:.2f}x"
                m_sc = np.array(fraction_data[frac][sc_name]["margins"])
                c_sc = np.array(fraction_data[frac][sc_name]["corrects"])
                dmg = float(np.mean(base_m - m_sc))
                scale_sweep_rows.append({
                    "model": self.model_name,
                    "fraction": frac,
                    "scale": sc,
                    "accuracy": float(np.mean(c_sc)),
                    "mean_margin": float(np.mean(m_sc)),
                    "damage": dmg,
                })

            # Cosine
            for alpha in self.target_cosines:
                c_avg_name = f"cosine_{alpha:.2f}_avg"
                m_cos = fraction_data[frac][c_avg_name]["margins"]
                c_cos = fraction_data[frac][c_avg_name]["corrects"]
                dmg = float(np.mean(base_m - m_cos))
                cosine_sweep_rows.append({
                    "model": self.model_name,
                    "fraction": frac,
                    "target_cosine": alpha,
                    "accuracy": float(np.mean(c_cos)),
                    "mean_margin": float(np.mean(m_cos)),
                    "damage": dmg,
                })

            # Sign flip
            for s_flip in ["sign_flip_25%", "sign_flip_50%", "sign_flip_75%", "sign_flip_100%"]:
                m_sf = np.array(fraction_data[frac][s_flip]["margins"])
                c_sf = np.array(fraction_data[frac][s_flip]["corrects"])
                dmg = float(np.mean(base_m - m_sf))
                sign_flip_rows.append({
                    "model": self.model_name,
                    "fraction": frac,
                    "flip_condition": s_flip,
                    "accuracy": float(np.mean(c_sf)),
                    "mean_margin": float(np.mean(m_sf)),
                    "damage": dmg,
                })

        # CLS vs Patch Replacement comparison rows
        cls_comparison_rows = []
        for c_name in cls_control_names:
            m_c = np.array(cls_control_data[c_name]["margins"])
            c_c = np.array(cls_control_data[c_name]["corrects"])
            cls_comparison_rows.append({
                "model": self.model_name,
                "condition": c_name,
                "accuracy": float(np.mean(c_c)),
                "mean_margin": float(np.mean(m_c)),
                "damage": float(np.mean(base_m - m_c)),
            })

        # -------------------------------------------------------------
        # STEP 8: Export DataFrames and Parquet Files
        # -------------------------------------------------------------
        prefix = "tiny" if "tiny" in self.model_name else "small"

        # 1. Per-image results dataframe
        per_image_dict = {
            "image_idx": np.arange(n_eval),
            "target": all_targets,
            "baseline_pred": baseline_preds,
            "baseline_correct": baseline_corrects,
            "baseline_margin": baseline_margins,
        }
        # Add key conditions at 100% and 50%
        for frac in ["50%", "100%"]:
            for cond in ["zero", "mu_8", "gaussian_avg", "coord_perm_avg", "sign_flip_100%"]:
                m_cond = fraction_data[frac][cond]["margins"]
                c_cond = fraction_data[frac][cond]["corrects"]
                per_image_dict[f"margin_{frac}_{cond}"] = m_cond
                per_image_dict[f"correct_{frac}_{cond}"] = c_cond
                per_image_dict[f"damage_{frac}_{cond}"] = base_m - np.array(m_cond)

        df_per_image = pd.DataFrame(per_image_dict)
        parquet_path = os.path.join(self.output_dir, f"{prefix}_image_results.parquet")
        df_per_image.to_parquet(parquet_path, index=False)

        # 2. Prototype comparison CSV
        df_comp = pd.DataFrame(comparison_rows)
        comp_csv_path = os.path.join(self.output_dir, f"{prefix}_prototype_comparison.csv")
        df_comp.to_csv(comp_csv_path, index=False)

        # 3. Fraction summary CSV
        df_frac = pd.DataFrame(fraction_summary_rows)
        frac_csv_path = os.path.join(self.output_dir, f"{prefix}_fraction_summary.csv")
        df_frac.to_csv(frac_csv_path, index=False)

        # 4. Sweep CSVs
        pd.DataFrame(scale_sweep_rows).to_csv(os.path.join(self.output_dir, f"{prefix}_scale_sweep.csv"), index=False)
        pd.DataFrame(cosine_sweep_rows).to_csv(os.path.join(self.output_dir, f"{prefix}_cosine_sweep.csv"), index=False)
        pd.DataFrame(sign_flip_rows).to_csv(os.path.join(self.output_dir, f"{prefix}_sign_flip_sweep.csv"), index=False)
        pd.DataFrame(cls_comparison_rows).to_csv(os.path.join(self.output_dir, f"{prefix}_cls_comparison.csv"), index=False)

        runtime = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

        # Verify backbone hash unchanged
        final_hash = get_model_parameter_hash(model)
        assert initial_hash == final_hash, "Backbone weights were altered during evaluation!"

        return {
            "model_name": self.model_name,
            "prefix": prefix,
            "runtime": runtime,
            "peak_vram_mb": peak_vram,
            "baseline_accuracy": base_acc,
            "baseline_margin_mean": float(np.mean(base_m)),
            "cls_untouched_all": cls_untouched_all,
            "fraction_summary": fraction_summary_rows,
            "comparison_rows": comparison_rows,
            "scale_sweep_rows": scale_sweep_rows,
            "cosine_sweep_rows": cosine_sweep_rows,
            "sign_flip_rows": sign_flip_rows,
            "cls_comparison_rows": cls_comparison_rows,
            "proto_metadata": proto_data["metadata_rows"],
            "proto_vectors": proto_data["vectors"]
        }
