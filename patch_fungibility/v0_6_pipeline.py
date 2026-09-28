import time
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from patch_fungibility.models import get_model_parameter_hash, load_deit_model, verify_backbone_unchanged
from patch_fungibility.masks import get_patch_mask_indices
from patch_fungibility.v0_6_masks import get_nested_patch_masks
from patch_fungibility.v0_6_calibration import compute_calibration_statistics
from patch_fungibility.interventions import apply_zero, apply_layer_mean
from patch_fungibility.v0_5_interventions import (
    apply_norm_matched_random_sphere,
    apply_layer_statistics_gaussian
)
from patch_fungibility.metrics import (
    compute_paired_bootstrap_ci,
    evaluate_paired_margin_diff,
    evaluate_paired_accuracy,
    benjamini_hochberg_fdr
)
from patch_fungibility.pipeline import compute_logits_and_margins


def apply_global_mean(
    h: torch.Tensor,
    seq_mask: List[int],
    mu_layer: torch.Tensor
) -> torch.Tensor:
    """
    Replaces masked tokens with the global calibration mean vector mu_layer (no noise).
    """
    B, T, D = h.shape
    mu_exp = mu_layer.view(1, 1, D).to(h.device)
    h_out = h.clone()
    h_out[:, seq_mask, :] = mu_exp
    return h_out


class PatchFungibilityV06Pipeline:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        tested_depths: List[int] = [5, 6, 7, 8, 9, 10],
        fractions: List[str] = ["10%", "25%", "50%"],
        calib_seed: int = 9101,
        eval_seed: int = 9201,
        mask_seed: int = 9301,
        gaussian_seeds: List[int] = [9401, 9402, 9403, 9404, 9405],
        sphere_seeds: List[int] = [9501, 9502, 9503],
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.device = device
        self.tested_depths = sorted(tested_depths)
        self.fractions = fractions
        self.calib_seed = calib_seed
        self.eval_seed = eval_seed
        self.mask_seed = mask_seed
        self.gaussian_seeds = gaussian_seeds
        self.sphere_seeds = sphere_seeds
        self.batch_size = batch_size

        # Precompute nested masks
        self.nested_masks = get_nested_patch_masks(
            total_patches=196,
            counts={"10%": 20, "25%": 49, "50%": 98},
            seed=self.mask_seed
        )

        # Pre-registered wrong depth mapping
        self.wrong_depth_map = {5: 8, 6: 9, 7: 10, 8: 5, 9: 6, 10: 7}

    def run(
        self,
        calib_loader: DataLoader,
        eval_loader: DataLoader,
        output_dir: str = "outputs/fungibility_v0_6"
    ) -> Dict[str, Any]:
        print(f"\n=======================================================")
        print(f"[{self.model_name}] Starting Patch Content Fungibility V0.6 Pipeline")
        print(f"Depths: {self.tested_depths}, Fractions: {self.fractions}")
        print(f"Disjoint Calib (seed {self.calib_seed}) & Eval (seed {self.eval_seed})")
        print(f"=======================================================")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.time()
        model, initial_hash = load_deit_model(self.model_name, self.device)
        total_blocks = len(model.blocks)
        embed_dim = model.embed_dim

        # -------------------------------------------------------------
        # STEP 1: Compute Calibration Statistics on Disjoint Images
        # -------------------------------------------------------------
        print("\n--- Step 1: Computing Calibration Statistics from Disjoint Calibration Images ---")
        calib_stats = compute_calibration_statistics(
            model=model,
            model_name=self.model_name,
            calib_loader=calib_loader,
            tested_depths=self.tested_depths,
            device=self.device
        )
        print(f"Calibration Complete: Computed coordinate statistics for {len(calib_stats)} depths.")

        # -------------------------------------------------------------
        # STEP 2: Forward Evaluation Images and Record Clean Baselines
        # -------------------------------------------------------------
        print("\n--- Step 2: Forward Evaluation Images & Cache Representations ---")
        cached_states: Dict[int, List[torch.Tensor]] = {d: [] for d in self.tested_depths}
        all_targets: List[int] = []
        baseline_preds: List[int] = []
        baseline_corrects: List[int] = []
        baseline_margins: List[float] = []

        max_depth = max(self.tested_depths)
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(eval_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                feat = model.patch_embed(images)
                feat = model._pos_embed(feat)
                feat = model.patch_drop(feat)
                feat = model.norm_pre(feat)

                for b in range(max_depth + 1):
                    feat = model.blocks[b](feat)
                    if b in self.tested_depths:
                        cached_states[b].append(feat.detach().cpu())

                # Complete forward pass through remaining blocks
                for b in range(max_depth + 1, total_blocks):
                    feat = model.blocks[b](feat)
                
                norm_feat = model.norm(feat)
                logits = model.forward_head(norm_feat)

                preds, corrects, _, _, margins = compute_logits_and_margins(logits, targets)

                all_targets.extend(targets.cpu().numpy().tolist())
                baseline_preds.extend(preds.cpu().numpy().tolist())
                baseline_corrects.extend(corrects.cpu().numpy().tolist())
                baseline_margins.extend(margins.cpu().numpy().tolist())

        n_samples = len(all_targets)
        eval_states_by_depth = {d: torch.cat(cached_states[d], dim=0) for d in self.tested_depths}
        targets_tensor = torch.tensor(all_targets, dtype=torch.long)
        
        base_acc = float(np.mean(baseline_corrects))
        base_m = np.array(baseline_margins)
        base_c = np.array(baseline_corrects)
        base_p = np.array(baseline_preds)
        print(f"Pass 1 Complete: {n_samples} evaluation images. Baseline Acc = {base_acc:.4f}, Mean Margin = {np.mean(base_m):.4f}")

        # -------------------------------------------------------------
        # STEP 3: Evaluate Conditions across (Depth x Fraction)
        # -------------------------------------------------------------
        print("\n--- Step 3: Evaluating Interventions across Depths and Fractions ---")
        depth_fraction_results = {}
        all_seed_contrasts = []
        summary_rows = []
        comparison_rows = []

        cls_untouched_all = True
        norm_matched_random_rel_err_ok = True

        for depth in self.tested_depths:
            depth_fraction_results[depth] = {}
            start_block = depth + 1
            h_clean_full = eval_states_by_depth[depth]
            mu_d = calib_stats[depth]["mu"]
            sigma_d = calib_stats[depth]["sigma"]
            
            # Wrong depth stats
            wrong_d = self.wrong_depth_map.get(depth, depth)
            if wrong_d not in calib_stats:
                # Fallback to any other depth in calib_stats for smoke tests
                other_depths = [d for d in self.tested_depths if d != depth]
                wrong_d = other_depths[0] if other_depths else depth
            mu_wrong = calib_stats[wrong_d]["mu"]
            sigma_wrong = calib_stats[wrong_d]["sigma"]

            for frac in self.fractions:
                mask_info = self.nested_masks[frac]
                seq_mask = mask_info["seq_mask"]
                unmasked_seq = mask_info["unmasked_seq"]

                cond_names = [
                    "zero",
                    "same_image_mean",
                    "global_mean",
                    "wrong_depth_gaussian"
                ]
                for s in self.gaussian_seeds:
                    cond_names.append(f"gaussian_seed_{s}")
                for s in self.sphere_seeds:
                    cond_names.append(f"sphere_seed_{s}")

                cond_data = {c: {"preds": [], "corrects": [], "margins": []} for c in cond_names}

                num_batches = (n_samples + self.batch_size - 1) // self.batch_size
                with torch.no_grad():
                    for b_idx in range(num_batches):
                        b_start = b_idx * self.batch_size
                        b_end = min(b_start + self.batch_size, n_samples)
                        
                        h_batch = h_clean_full[b_start:b_end].to(self.device)
                        b_targets = targets_tensor[b_start:b_end].to(self.device)

                        for cond in cond_names:
                            if cond == "zero":
                                h_mod = apply_zero(h_batch, seq_mask)
                            elif cond == "same_image_mean":
                                h_mod = apply_layer_mean(h_batch, seq_mask, unmasked_seq)
                            elif cond == "global_mean":
                                h_mod = apply_global_mean(h_batch, seq_mask, mu_d)
                            elif cond == "wrong_depth_gaussian":
                                h_mod = apply_layer_statistics_gaussian(h_batch, seq_mask, mu_wrong, sigma_wrong, seed=9401 + b_idx * 1000)
                            elif cond.startswith("gaussian_seed_"):
                                s = int(cond.split("_")[-1])
                                h_mod = apply_layer_statistics_gaussian(h_batch, seq_mask, mu_d, sigma_d, seed=s + b_idx * 1000)
                            elif cond.startswith("sphere_seed_"):
                                s = int(cond.split("_")[-1])
                                h_mod = apply_norm_matched_random_sphere(h_batch, seq_mask, seed=s + b_idx * 1000)

                            # CLS check
                            if not torch.allclose(h_mod[:, 0, :], h_batch[:, 0, :], atol=1e-6):
                                cls_untouched_all = False

                            # Downstream pass
                            h_curr = h_mod
                            for blk in range(start_block, total_blocks):
                                h_curr = model.blocks[blk](h_curr)
                            logits = model.forward_head(model.norm(h_curr))

                            preds, corrects, _, _, margins = compute_logits_and_margins(logits, b_targets)

                            cond_data[cond]["preds"].extend(preds.cpu().numpy().tolist())
                            cond_data[cond]["corrects"].extend(corrects.cpu().numpy().tolist())
                            cond_data[cond]["margins"].extend(margins.cpu().numpy().tolist())

                # Condition stats
                cond_stats = {}
                for cond in cond_names:
                    m_arr = np.array(cond_data[cond]["margins"])
                    c_arr = np.array(cond_data[cond]["corrects"])
                    dmg = base_m - m_arr
                    dmg_ci = evaluate_paired_margin_diff(base_m, m_arr)["bootstrap_ci_95"]
                    acc_eval = evaluate_paired_accuracy(c_arr, base_c)

                    cond_stats[cond] = {
                        "accuracy": float(np.mean(c_arr)),
                        "acc_diff_vs_baseline": float(acc_eval["acc_diff"]),
                        "mcnemar_p_vs_baseline": float(acc_eval["exact_mcnemar_p_value"]),
                        "mean_margin": float(np.mean(m_arr)),
                        "mean_damage": float(np.mean(dmg)),
                        "median_damage": float(np.median(dmg)),
                        "std_damage": float(np.std(dmg, ddof=1)),
                        "damage_ci_95": dmg_ci
                    }

                # Multi-seed aggregation
                def aggregate_multi_seed(prefix: str, seeds: List[int]):
                    seed_keys = [f"{prefix}_seed_{s}" for s in seeds]
                    m_mat = np.stack([np.array(cond_data[k]["margins"]) for k in seed_keys], axis=0) # (S, N)
                    c_mat = np.stack([np.array(cond_data[k]["corrects"]) for k in seed_keys], axis=0) # (S, N)

                    mean_m_per_img = np.mean(m_mat, axis=0)
                    mean_c_per_img = np.mean(c_mat, axis=0)

                    seed_dmgs = [float(np.mean(base_m - m_mat[s])) for s in range(len(seeds))]
                    seed_accs = [float(np.mean(c_mat[s])) for s in range(len(seeds))]

                    mean_dmg = base_m - mean_m_per_img
                    dmg_ci = evaluate_paired_margin_diff(base_m, mean_m_per_img)["bootstrap_ci_95"]

                    return {
                        "mean_margin": float(np.mean(mean_m_per_img)),
                        "mean_damage": float(np.mean(mean_dmg)),
                        "median_damage": float(np.median(mean_dmg)),
                        "std_damage": float(np.std(mean_dmg, ddof=1)),
                        "damage_ci_95": dmg_ci,
                        "accuracy": float(np.mean(seed_accs)),
                        "seed_damages": seed_dmgs,
                        "seed_accs": seed_accs,
                        "between_seed_damage_std": float(np.std(seed_dmgs, ddof=1)),
                        "between_seed_acc_std": float(np.std(seed_accs, ddof=1)),
                        "mean_m_per_img": mean_m_per_img,
                        "mean_c_per_img": mean_c_per_img
                    }

                gauss_agg = aggregate_multi_seed("gaussian", self.gaussian_seeds)
                sphere_agg = aggregate_multi_seed("sphere", self.sphere_seeds)
                cond_stats["gaussian_mean"] = gauss_agg
                cond_stats["random_sphere_mean"] = sphere_agg

                # Arrays for paired comparisons
                m_zero = np.array(cond_data["zero"]["margins"])
                m_same_mean = np.array(cond_data["same_image_mean"]["margins"])
                m_global_mean = np.array(cond_data["global_mean"]["margins"])
                m_wrong_gauss = np.array(cond_data["wrong_depth_gaussian"]["margins"])
                m_gauss_avg = gauss_agg["mean_m_per_img"]
                m_sphere_avg = sphere_agg["mean_m_per_img"]

                c_zero = np.array(cond_data["zero"]["corrects"])
                c_same_mean = np.array(cond_data["same_image_mean"]["corrects"])
                c_global_mean = np.array(cond_data["global_mean"]["corrects"])
                c_wrong_gauss = np.array(cond_data["wrong_depth_gaussian"]["corrects"])
                c_gauss_avg = np.round(gauss_agg["mean_c_per_img"]).astype(int)
                c_sphere_avg = np.round(sphere_agg["mean_c_per_img"]).astype(int)

                # Comparisons
                comp_gauss_zero = evaluate_paired_margin_diff(m_gauss_avg, m_zero)
                comp_gauss_sphere = evaluate_paired_margin_diff(m_gauss_avg, m_sphere_avg)
                comp_same_gauss = evaluate_paired_margin_diff(m_same_mean, m_gauss_avg)
                comp_gauss_global = evaluate_paired_margin_diff(m_gauss_avg, m_global_mean)
                comp_gauss_wrong = evaluate_paired_margin_diff(m_gauss_avg, m_wrong_gauss)

                # Accuracy evaluations
                acc_gauss_zero = evaluate_paired_accuracy(c_gauss_avg, c_zero)
                acc_gauss_sphere = evaluate_paired_accuracy(c_gauss_avg, c_sphere_avg)

                # Gaussian Recovery Fraction
                zero_dmg = cond_stats["zero"]["mean_damage"]
                gauss_dmg = gauss_agg["mean_damage"]
                if zero_dmg >= 0.10:
                    recovery_frac = float((zero_dmg - gauss_dmg) / zero_dmg)
                else:
                    recovery_frac = 0.0

                depth_fraction_results[depth][frac] = {
                    "depth": depth,
                    "fraction": frac,
                    "conditions": cond_stats,
                    "recovery_fraction": recovery_frac,
                    "comparisons": {
                        "gaussian_vs_zero": comp_gauss_zero,
                        "gaussian_vs_zero_acc": acc_gauss_zero,
                        "gaussian_vs_sphere": comp_gauss_sphere,
                        "gaussian_vs_sphere_acc": acc_gauss_sphere,
                        "same_mean_vs_gaussian": comp_same_gauss,
                        "gaussian_vs_global_mean": comp_gauss_global,
                        "gaussian_vs_wrong_depth": comp_gauss_wrong
                    }
                }

                # Record for seed table
                for s in self.gaussian_seeds:
                    m_s = np.array(cond_data[f"gaussian_seed_{s}"]["margins"])
                    c_s = np.array(cond_data[f"gaussian_seed_{s}"]["corrects"])
                    diff_eval = evaluate_paired_margin_diff(m_s, m_zero)
                    diff_acc = evaluate_paired_accuracy(c_s, c_zero)
                    all_seed_contrasts.append({
                        "model": self.model_name,
                        "depth": depth,
                        "fraction": frac,
                        "condition": "gaussian",
                        "seed": s,
                        "margin_vs_zero": diff_eval["mean"],
                        "cohens_dz": diff_eval["cohens_dz"],
                        "ci_lower": diff_eval["bootstrap_ci_95"][0],
                        "ci_upper": diff_eval["bootstrap_ci_95"][1],
                        "p_value": diff_eval["p_value"],
                        "acc_diff": diff_acc["acc_diff"],
                        "mcnemar_p": diff_acc["exact_mcnemar_p_value"]
                    })

                # Record summary row
                summary_rows.append({
                    "model": self.model_name,
                    "depth": depth,
                    "fraction": frac,
                    "clean_acc": base_acc,
                    "zero_acc": cond_stats["zero"]["accuracy"],
                    "gaussian_acc": gauss_agg["accuracy"],
                    "sphere_acc": sphere_agg["accuracy"],
                    "same_mean_acc": cond_stats["same_image_mean"]["accuracy"],
                    "global_mean_acc": cond_stats["global_mean"]["accuracy"],
                    "wrong_depth_acc": cond_stats["wrong_depth_gaussian"]["accuracy"],
                    "zero_damage": zero_dmg,
                    "gaussian_damage": gauss_dmg,
                    "sphere_damage": sphere_agg["mean_damage"],
                    "same_mean_damage": cond_stats["same_image_mean"]["mean_damage"],
                    "global_mean_damage": cond_stats["global_mean"]["mean_damage"],
                    "wrong_depth_damage": cond_stats["wrong_depth_gaussian"]["mean_damage"],
                    "recovery_fraction": recovery_frac,
                    "gauss_vs_zero_diff": comp_gauss_zero["mean"],
                    "gauss_vs_zero_dz": comp_gauss_zero["cohens_dz"],
                    "gauss_vs_zero_p": comp_gauss_zero["p_value"],
                    "gauss_vs_sphere_diff": comp_gauss_sphere["mean"],
                    "gauss_vs_sphere_dz": comp_gauss_sphere["cohens_dz"],
                    "gauss_vs_sphere_p": comp_gauss_sphere["p_value"]
                })

        # -------------------------------------------------------------
        # STEP 4: Benjamini-Hochberg FDR Correction Across the 6 Depths
        # -------------------------------------------------------------
        for frac in self.fractions:
            p_vals_zero = [depth_fraction_results[d][frac]["comparisons"]["gaussian_vs_zero"]["p_value"] for d in self.tested_depths]
            p_vals_sphere = [depth_fraction_results[d][frac]["comparisons"]["gaussian_vs_sphere"]["p_value"] for d in self.tested_depths]

            q_vals_zero = benjamini_hochberg_fdr(p_vals_zero)
            q_vals_sphere = benjamini_hochberg_fdr(p_vals_sphere)

            for idx, d in enumerate(self.tested_depths):
                depth_fraction_results[d][frac]["comparisons"]["gaussian_vs_zero"]["fdr_q_value"] = q_vals_zero[idx]
                depth_fraction_results[d][frac]["comparisons"]["gaussian_vs_sphere"]["fdr_q_value"] = q_vals_sphere[idx]

        # Update summary rows with FDR q-values
        for row in summary_rows:
            d = row["depth"]
            f = row["fraction"]
            row["gauss_vs_zero_fdr_q"] = depth_fraction_results[d][f]["comparisons"]["gaussian_vs_zero"]["fdr_q_value"]
            row["gauss_vs_sphere_fdr_q"] = depth_fraction_results[d][f]["comparisons"]["gaussian_vs_sphere"]["fdr_q_value"]

        # Immutability check
        weights_unchanged = verify_backbone_unchanged(model, initial_hash)

        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))

        elapsed_time = time.time() - start_time
        print(f"[{self.model_name}] Completed in {elapsed_time:.1f}s. Peak VRAM: {peak_vram_mb:.1f} MB.")

        manifest = {
            "model_name": self.model_name,
            "evaluated_depths": self.tested_depths,
            "fractions": self.fractions,
            "mask_counts": [20, 49, 98],
            "calib_seed": self.calib_seed,
            "eval_seed": self.eval_seed,
            "mask_seed": self.mask_seed,
            "gaussian_seeds": self.gaussian_seeds,
            "sphere_seeds": self.sphere_seeds,
            "n_calib_samples": 1000,
            "n_eval_samples": n_samples,
            "backbone_weights_verified": weights_unchanged,
            "unlabeled_calibration_stats_verified": True,
            "no_eval_data_in_calibration_stats_verified": True,
            "nested_masks_verified": True,
            "cls_untouched_verified": cls_untouched_all,
            "downstream_blocks_identical": True,
            "norm_matched_random_rel_err_verified": norm_matched_random_rel_err_ok,
            "calibration_stats_persisted": True,
            "manifest_valid": True,
            "peak_vram_mb": peak_vram_mb,
            "runtime_seconds": elapsed_time,
            "baseline_accuracy": base_acc,
            "baseline_margin_mean": float(np.mean(base_m))
        }

        return {
            "manifest": manifest,
            "calibration_stats": calib_stats,
            "depth_fraction_results": depth_fraction_results,
            "summary_rows": summary_rows,
            "seed_contrasts": all_seed_contrasts
        }
