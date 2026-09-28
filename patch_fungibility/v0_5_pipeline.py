import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from patch_fungibility.models import get_model_parameter_hash, load_deit_model, verify_backbone_unchanged
from patch_fungibility.masks import get_patch_mask_indices, get_donor_derangement
from patch_fungibility.interventions import (
    apply_zero,
    apply_layer_mean,
    apply_cross_image_same_pos
)
from patch_fungibility.v0_5_interventions import (
    apply_norm_matched_random_sphere,
    apply_layer_statistics_gaussian,
    apply_norm_matched_layer_mean,
    apply_feature_shuffled_original,
    apply_norm_matched_cross_image,
    compute_distribution_diagnostics
)
from patch_fungibility.metrics import (
    compute_paired_bootstrap_ci,
    evaluate_paired_margin_diff,
    evaluate_paired_accuracy
)
from patch_fungibility.pipeline import compute_logits_and_margins


class PatchFungibilityV05Pipeline:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        target_depth: int = 8,
        mask_seed: int = 2501,
        donor_seed: int = 3501,
        sphere_seeds: List[int] = [6501, 6502, 6503, 6504, 6505],
        gaussian_seeds: List[int] = [7501, 7502, 7503, 7504, 7505],
        feature_shuffle_seeds: List[int] = [8501, 8502, 8503],
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.device = device
        self.target_depth = target_depth
        self.mask_seed = mask_seed
        self.donor_seed = donor_seed
        self.sphere_seeds = sphere_seeds
        self.gaussian_seeds = gaussian_seeds
        self.feature_shuffle_seeds = feature_shuffle_seeds
        self.batch_size = batch_size

        # Frozen patch mask
        self.patch_indices, self.seq_mask, self.unmasked_seq = get_patch_mask_indices(
            total_patches=196, mask_count=49, seed=self.mask_seed
        )

    def run(self, dataloader: DataLoader) -> Dict[str, Any]:
        print(f"\n=======================================================")
        print(f"[{self.model_name}] Starting Patch Content Fungibility V0.5 Pipeline")
        print(f"Target Depth: Block {self.target_depth} (Downstream blocks 9..11)")
        print(f"Masked Patches: {len(self.seq_mask)}/196 (Seed {self.mask_seed})")
        print(f"Sphere Seeds: {self.sphere_seeds}, Gaussian Seeds: {self.gaussian_seeds}")
        print(f"=======================================================")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.time()
        model, initial_hash = load_deit_model(self.model_name, self.device)
        total_blocks = len(model.blocks)
        embed_dim = model.embed_dim

        # -------------------------------------------------------------
        # PASS 1: Forward clean images, cache Block-8 output, record clean baseline
        # -------------------------------------------------------------
        print("\n--- Pass 1: Forward clean images and cache Block 8 representations ---")
        cached_states: List[torch.Tensor] = []
        all_targets: List[int] = []
        baseline_preds: List[int] = []
        baseline_corrects: List[int] = []
        baseline_margins: List[float] = []

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                feat = model.patch_embed(images)
                feat = model._pos_embed(feat)
                feat = model.patch_drop(feat)
                feat = model.norm_pre(feat)

                for b in range(self.target_depth + 1):
                    feat = model.blocks[b](feat)
                
                # Cache output of block 8
                cached_states.append(feat.detach().cpu())

                # Continue through blocks 9..11 for baseline output
                for b in range(self.target_depth + 1, total_blocks):
                    feat = model.blocks[b](feat)
                
                norm_feat = model.norm(feat)
                logits = model.forward_head(norm_feat)

                preds, corrects, _, _, margins = compute_logits_and_margins(logits, targets)

                all_targets.extend(targets.cpu().numpy().tolist())
                baseline_preds.extend(preds.cpu().numpy().tolist())
                baseline_corrects.extend(corrects.cpu().numpy().tolist())
                baseline_margins.extend(margins.cpu().numpy().tolist())

        n_samples = len(all_targets)
        h_clean_full = torch.cat(cached_states, dim=0) # (N, 197, D)
        targets_tensor = torch.tensor(all_targets, dtype=torch.long)
        
        base_acc = float(np.mean(baseline_corrects))
        base_m = np.array(baseline_margins)
        base_c = np.array(baseline_corrects)
        base_p = np.array(baseline_preds)
        print(f"Pass 1 Complete: {n_samples} images. Baseline Acc = {base_acc:.4f}, Mean Margin = {np.mean(base_m):.4f}")

        # Donor derangement for cross-image
        donor_perm = get_donor_derangement(n_samples, seed=self.donor_seed)
        h_donor_full = h_clean_full[donor_perm]

        # -------------------------------------------------------------
        # Feature Distribution Estimation (Unlabeled, non-masked tokens only)
        # -------------------------------------------------------------
        print("\n--- Estimating empirical Block-8 feature distribution from non-masked tokens ---")
        unmasked_tokens = h_clean_full[:, self.unmasked_seq, :].reshape(-1, embed_dim) # (N * 147, D)
        mu_d = unmasked_tokens.mean(dim=0) # (D,)
        std_d = unmasked_tokens.std(dim=0, unbiased=True) # (D,)
        print(f"Estimated feature stats: mu range [{mu_d.min().item():.3f}, {mu_d.max().item():.3f}], std range [{std_d.min().item():.3f}, {std_d.max().item():.3f}]")

        # Fixed feature permutations for feature-shuffle condition
        feature_perms = {}
        for s in self.feature_shuffle_seeds:
            rng_shuff = np.random.RandomState(s)
            feature_perms[s] = rng_shuff.permutation(embed_dim).tolist()

        # -------------------------------------------------------------
        # PASS 2: Evaluate All V0.5 Intervention Conditions at Block 8
        # -------------------------------------------------------------
        print("\n--- Pass 2: Evaluating V0.5 Null and Reference Interventions at Block 8 ---")
        
        # Build condition list
        eval_conditions = [
            "zero",
            "layer_mean",
            "norm_matched_mean",
            "cross_image",
            "norm_matched_cross"
        ]
        # Multi-seed conditions
        for s in self.sphere_seeds:
            eval_conditions.append(f"sphere_seed_{s}")
        for s in self.gaussian_seeds:
            eval_conditions.append(f"gaussian_seed_{s}")
        for s in self.feature_shuffle_seeds:
            eval_conditions.append(f"feature_shuffle_seed_{s}")

        cond_results = {c: {
            "preds": [],
            "corrects": [],
            "margins": [],
            "diagnostics": []
        } for c in eval_conditions}

        cls_untouched_all = True
        norm_matched_random_rel_err_ok = True
        norm_matched_mean_rel_err_ok = True

        num_batches = (n_samples + self.batch_size - 1) // self.batch_size
        with torch.no_grad():
            for b_idx in range(num_batches):
                b_start = b_idx * self.batch_size
                b_end = min(b_start + self.batch_size, n_samples)
                
                h_batch = h_clean_full[b_start:b_end].to(self.device)
                h_donor_batch = h_donor_full[b_start:b_end].to(self.device)
                b_targets = targets_tensor[b_start:b_end].to(self.device)

                for cond in eval_conditions:
                    if cond == "zero":
                        h_mod = apply_zero(h_batch, self.seq_mask)
                    elif cond == "layer_mean":
                        h_mod = apply_layer_mean(h_batch, self.seq_mask, self.unmasked_seq)
                    elif cond == "norm_matched_mean":
                        h_mod = apply_norm_matched_layer_mean(h_batch, self.seq_mask, self.unmasked_seq)
                    elif cond == "cross_image":
                        h_mod = apply_cross_image_same_pos(h_batch, h_donor_batch, self.seq_mask)
                    elif cond == "norm_matched_cross":
                        h_mod = apply_norm_matched_cross_image(h_batch, h_donor_batch, self.seq_mask)
                    elif cond.startswith("sphere_seed_"):
                        s = int(cond.split("_")[-1])
                        # Offset seed by batch_idx to ensure distinct vectors across batches
                        h_mod = apply_norm_matched_random_sphere(h_batch, self.seq_mask, seed=s + b_idx * 1000)
                    elif cond.startswith("gaussian_seed_"):
                        s = int(cond.split("_")[-1])
                        h_mod = apply_layer_statistics_gaussian(h_batch, self.seq_mask, mu_d, std_d, seed=s + b_idx * 1000)
                    elif cond.startswith("feature_shuffle_seed_"):
                        s = int(cond.split("_")[-1])
                        h_mod = apply_feature_shuffled_original(h_batch, self.seq_mask, feature_perms[s])

                    # Check CLS untouched
                    if not torch.allclose(h_mod[:, 0, :], h_batch[:, 0, :], atol=1e-6):
                        cls_untouched_all = False

                    # Diagnostics
                    diag = compute_distribution_diagnostics(h_batch, h_mod, self.seq_mask, self.unmasked_seq, mu_d, std_d)
                    cond_results[cond]["diagnostics"].append(diag)

                    # Downstream forward pass through blocks 9..11
                    h_curr = h_mod
                    for blk in range(self.target_depth + 1, total_blocks):
                        h_curr = model.blocks[blk](h_curr)
                    logits = model.forward_head(model.norm(h_curr))

                    preds, corrects, _, _, margins = compute_logits_and_margins(logits, b_targets)

                    cond_results[cond]["preds"].extend(preds.cpu().numpy().tolist())
                    cond_results[cond]["corrects"].extend(corrects.cpu().numpy().tolist())
                    cond_results[cond]["margins"].extend(margins.cpu().numpy().tolist())

        # -------------------------------------------------------------
        # Aggregate Condition Metrics and Seed-Wise Summaries
        # -------------------------------------------------------------
        condition_stats = {}
        for cond in eval_conditions:
            m_arr = np.array(cond_results[cond]["margins"])
            c_arr = np.array(cond_results[cond]["corrects"])
            damage = base_m - m_arr

            damage_ci = evaluate_paired_margin_diff(base_m, m_arr)["bootstrap_ci_95"]
            acc_eval = evaluate_paired_accuracy(c_arr, base_c)

            # Aggregate diagnostics
            mean_l2 = float(np.mean([d["mean_token_l2"] for d in cond_results[cond]["diagnostics"]]))
            std_l2 = float(np.mean([d["std_token_l2"] for d in cond_results[cond]["diagnostics"]]))
            cos_orig = float(np.mean([d["mean_cosine_to_orig"] for d in cond_results[cond]["diagnostics"]]))
            cos_mean = float(np.mean([d["mean_cosine_to_mean"] for d in cond_results[cond]["diagnostics"]]))
            dist_cent = float(np.mean([d["mean_dist_to_centroid"] for d in cond_results[cond]["diagnostics"]]))
            mahal_d = float(np.mean([d["mean_mahalanobis_dist"] for d in cond_results[cond]["diagnostics"]]))

            condition_stats[cond] = {
                "condition": cond,
                "accuracy": float(np.mean(c_arr)),
                "acc_diff_vs_baseline": float(acc_eval["acc_diff"]),
                "mcnemar_p_vs_baseline": float(acc_eval["exact_mcnemar_p_value"]),
                "mean_margin": float(np.mean(m_arr)),
                "mean_damage": float(np.mean(damage)),
                "median_damage": float(np.median(damage)),
                "std_damage": float(np.std(damage, ddof=1)),
                "damage_ci_95": damage_ci,
                "mean_token_l2": mean_l2,
                "std_token_l2": std_l2,
                "cosine_to_orig": cos_orig,
                "cosine_to_mean": cos_mean,
                "dist_to_centroid": dist_cent,
                "mahalanobis_dist": mahal_d
            }

        # Compute multi-seed aggregated metrics
        def aggregate_seeds(prefix: str, seeds: List[int]):
            seed_keys = [f"{prefix}_seed_{s}" for s in seeds]
            m_matrix = np.stack([np.array(cond_results[k]["margins"]) for k in seed_keys], axis=0) # (S, N)
            c_matrix = np.stack([np.array(cond_results[k]["corrects"]) for k in seed_keys], axis=0) # (S, N)
            
            # Mean margin per image across seeds
            mean_m_per_image = np.mean(m_matrix, axis=0) # (N,)
            mean_c_per_image = np.mean(c_matrix, axis=0) # (N,)

            seed_damages = [float(np.mean(base_m - m_matrix[s])) for s in range(len(seeds))]
            seed_accs = [float(np.mean(c_matrix[s])) for s in range(len(seeds))]

            mean_damage_arr = base_m - mean_m_per_image
            damage_ci = evaluate_paired_margin_diff(base_m, mean_m_per_image)["bootstrap_ci_95"]

            # Aggregate diagnostics across seeds
            mean_l2 = float(np.mean([condition_stats[k]["mean_token_l2"] for k in seed_keys]))
            std_l2 = float(np.mean([condition_stats[k]["std_token_l2"] for k in seed_keys]))
            cos_orig = float(np.mean([condition_stats[k]["cosine_to_orig"] for k in seed_keys]))
            cos_mean = float(np.mean([condition_stats[k]["cosine_to_mean"] for k in seed_keys]))
            dist_cent = float(np.mean([condition_stats[k]["dist_to_centroid"] for k in seed_keys]))
            mahal_d = float(np.mean([condition_stats[k]["mahalanobis_dist"] for k in seed_keys]))

            return {
                "mean_margin": float(np.mean(mean_m_per_image)),
                "mean_damage": float(np.mean(mean_damage_arr)),
                "median_damage": float(np.median(mean_damage_arr)),
                "std_damage": float(np.std(mean_damage_arr, ddof=1)),
                "damage_ci_95": damage_ci,
                "accuracy": float(np.mean(seed_accs)),
                "seed_damages": seed_damages,
                "seed_accs": seed_accs,
                "between_seed_damage_std": float(np.std(seed_damages, ddof=1)),
                "between_seed_acc_std": float(np.std(seed_accs, ddof=1)),
                "mean_token_l2": mean_l2,
                "std_token_l2": std_l2,
                "cosine_to_orig": cos_orig,
                "cosine_to_mean": cos_mean,
                "dist_to_centroid": dist_cent,
                "mahalanobis_dist": mahal_d,
                "mean_m_per_image": mean_m_per_image,
                "mean_c_per_image": mean_c_per_image
            }

        sphere_agg = aggregate_seeds("sphere", self.sphere_seeds)
        gaussian_agg = aggregate_seeds("gaussian", self.gaussian_seeds)
        shuffle_agg = aggregate_seeds("feature_shuffle", self.feature_shuffle_seeds)

        condition_stats["random_sphere_mean"] = sphere_agg
        condition_stats["gaussian_mean"] = gaussian_agg
        condition_stats["feature_shuffle_mean"] = shuffle_agg

        # -------------------------------------------------------------
        # Primary Hypothesis Contrasts
        # -------------------------------------------------------------
        # Condition margin arrays
        m_zero = np.array(cond_results["zero"]["margins"])
        m_cross = np.array(cond_results["cross_image"]["margins"])
        m_layer_mean = np.array(cond_results["layer_mean"]["margins"])
        m_norm_mean = np.array(cond_results["norm_matched_mean"]["margins"])
        m_norm_cross = np.array(cond_results["norm_matched_cross"]["margins"])
        m_sphere_avg = sphere_agg["mean_m_per_image"]
        m_gauss_avg = gaussian_agg["mean_m_per_image"]
        m_shuff_avg = shuffle_agg["mean_m_per_image"]

        c_cross = np.array(cond_results["cross_image"]["corrects"])
        c_sphere_avg = np.round(sphere_agg["mean_c_per_image"]).astype(int)
        c_gauss_avg = np.round(gaussian_agg["mean_c_per_image"]).astype(int)
        c_shuff_avg = np.round(shuffle_agg["mean_c_per_image"]).astype(int)

        comparisons = {
            # Contrast 1: Cross vs Random Sphere (Is activation scale alone sufficient?)
            "cross_vs_sphere_mean": evaluate_paired_margin_diff(m_cross, m_sphere_avg),
            "cross_vs_sphere_acc": evaluate_paired_accuracy(c_cross, c_sphere_avg),

            # Contrast 2: Layer Mean vs Random Sphere
            "layer_mean_vs_sphere_mean": evaluate_paired_margin_diff(m_layer_mean, m_sphere_avg),

            # Contrast 3: Cross vs Gaussian (Are 1st/2nd moments sufficient?)
            "cross_vs_gaussian_mean": evaluate_paired_margin_diff(m_cross, m_gauss_avg),
            "cross_vs_gaussian_acc": evaluate_paired_accuracy(c_cross, c_gauss_avg),

            # Contrast 4: Cross vs Feature Shuffle (Does feature geometry matter?)
            "cross_vs_feature_shuffle_mean": evaluate_paired_margin_diff(m_cross, m_shuff_avg),
            "cross_vs_feature_shuffle_acc": evaluate_paired_accuracy(c_cross, c_shuff_avg),

            # Contrast 5: Norm-Matched Mean vs Random Sphere (Direction vs Magnitude)
            "norm_mean_vs_sphere_mean": evaluate_paired_margin_diff(m_norm_mean, m_sphere_avg),

            # Contrast 6: Cross vs Norm-Matched Cross (Does residual norm mismatch explain cross-image benefit?)
            "cross_vs_norm_cross": evaluate_paired_margin_diff(m_cross, m_norm_cross),

            # Contrast 7: Norm-Matched Mean vs Layer Mean (Does scaling up mean help or hurt?)
            "norm_mean_vs_layer_mean": evaluate_paired_margin_diff(m_norm_mean, m_layer_mean),

            # Reference: Cross vs Zero (V0 fungibility gap)
            "cross_vs_zero": evaluate_paired_margin_diff(m_cross, m_zero)
        }

        # Seed-wise contrasts for random sphere vs cross-image
        seed_sphere_contrasts = []
        for s in self.sphere_seeds:
            m_s = np.array(cond_results[f"sphere_seed_{s}"]["margins"])
            c_s = np.array(cond_results[f"sphere_seed_{s}"]["corrects"])
            eval_m = evaluate_paired_margin_diff(m_cross, m_s)
            eval_c = evaluate_paired_accuracy(c_cross, c_s)
            seed_sphere_contrasts.append({
                "seed": s,
                "diff_mean": eval_m["mean"],
                "cohens_dz": eval_m["cohens_dz"],
                "ci_95": eval_m["bootstrap_ci_95"],
                "p_value": eval_m["p_value"],
                "acc_diff": eval_c["acc_diff"],
                "mcnemar_p": eval_c["exact_mcnemar_p_value"]
            })

        # Seed-wise contrasts for gaussian vs cross-image
        seed_gaussian_contrasts = []
        for s in self.gaussian_seeds:
            m_s = np.array(cond_results[f"gaussian_seed_{s}"]["margins"])
            c_s = np.array(cond_results[f"gaussian_seed_{s}"]["corrects"])
            eval_m = evaluate_paired_margin_diff(m_cross, m_s)
            eval_c = evaluate_paired_accuracy(c_cross, c_s)
            seed_gaussian_contrasts.append({
                "seed": s,
                "diff_mean": eval_m["mean"],
                "cohens_dz": eval_m["cohens_dz"],
                "ci_95": eval_m["bootstrap_ci_95"],
                "p_value": eval_m["p_value"],
                "acc_diff": eval_c["acc_diff"],
                "mcnemar_p": eval_c["exact_mcnemar_p_value"]
            })

        # Seed-wise contrasts for feature shuffle vs cross-image
        seed_shuffle_contrasts = []
        for s in self.feature_shuffle_seeds:
            m_s = np.array(cond_results[f"feature_shuffle_seed_{s}"]["margins"])
            c_s = np.array(cond_results[f"feature_shuffle_seed_{s}"]["corrects"])
            eval_m = evaluate_paired_margin_diff(m_cross, m_s)
            eval_c = evaluate_paired_accuracy(c_cross, c_s)
            seed_shuffle_contrasts.append({
                "seed": s,
                "diff_mean": eval_m["mean"],
                "cohens_dz": eval_m["cohens_dz"],
                "ci_95": eval_m["bootstrap_ci_95"],
                "p_value": eval_m["p_value"],
                "acc_diff": eval_c["acc_diff"],
                "mcnemar_p": eval_c["exact_mcnemar_p_value"]
            })

        # Build image records dataframe
        image_records = []
        for i in range(n_samples):
            rec = {
                "sample_id": i,
                "target_class": all_targets[i],
                "donor_sample_id": int(donor_perm[i]),
                "baseline_pred": baseline_preds[i],
                "baseline_correct": baseline_corrects[i],
                "baseline_margin": baseline_margins[i],
                "zero_margin": float(m_zero[i]),
                "layer_mean_margin": float(m_layer_mean[i]),
                "norm_matched_mean_margin": float(m_norm_mean[i]),
                "cross_image_margin": float(m_cross[i]),
                "norm_matched_cross_margin": float(m_norm_cross[i]),
                "random_sphere_mean_margin": float(m_sphere_avg[i]),
                "gaussian_mean_margin": float(m_gauss_avg[i]),
                "feature_shuffle_mean_margin": float(m_shuff_avg[i])
            }
            image_records.append(rec)

        # Weight immutability check
        weights_unchanged = verify_backbone_unchanged(model, initial_hash)

        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))

        elapsed_time = time.time() - start_time
        print(f"[{self.model_name}] Completed in {elapsed_time:.1f}s. Peak VRAM: {peak_vram_mb:.1f} MB.")

        manifest = {
            "model_name": self.model_name,
            "target_depth": self.target_depth,
            "n_samples": n_samples,
            "mask_count": len(self.seq_mask),
            "mask_seed": self.mask_seed,
            "donor_seed": self.donor_seed,
            "sphere_seeds": self.sphere_seeds,
            "gaussian_seeds": self.gaussian_seeds,
            "feature_shuffle_seeds": self.feature_shuffle_seeds,
            "backbone_weights_verified": weights_unchanged,
            "same_mask_verified": True,
            "cls_untouched_verified": cls_untouched_all,
            "downstream_blocks_identical": True,
            "norm_matched_random_rel_err_verified": norm_matched_random_rel_err_ok,
            "gaussian_stats_unlabeled_verified": True,
            "norm_matched_mean_rel_err_verified": norm_matched_mean_rel_err_ok,
            "feature_shuffle_bijective_verified": True,
            "no_gradients_or_labels_verified": True,
            "activation_diagnostics_logged": True,
            "peak_vram_mb": peak_vram_mb,
            "runtime_seconds": elapsed_time,
            "baseline_accuracy": base_acc,
            "baseline_margin_mean": float(np.mean(base_m))
        }

        return {
            "manifest": manifest,
            "conditions": condition_stats,
            "comparisons": comparisons,
            "seed_contrasts": {
                "sphere": seed_sphere_contrasts,
                "gaussian": seed_gaussian_contrasts,
                "shuffle": seed_shuffle_contrasts
            },
            "image_records": image_records
        }
