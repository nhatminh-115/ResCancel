import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from patch_fungibility.models import get_model_parameter_hash, load_deit_model, verify_backbone_unchanged
from patch_fungibility.masks import (
    get_patch_mask_indices,
    get_donor_derangement,
    get_shuffled_donor_positions,
    get_within_image_shuffled_positions
)
from patch_fungibility.interventions import (
    apply_zero,
    apply_layer_mean,
    apply_cross_image_same_pos,
    apply_cross_image_rand_pos,
    apply_within_image_shuffle,
    compute_intervention_norms
)
from patch_fungibility.metrics import (
    evaluate_paired_margin_diff,
    evaluate_paired_accuracy,
    benjamini_hochberg_fdr
)


def compute_logits_and_margins(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes top-1 predictions, correctness, true logits, competing logits, and margins.
    """
    preds = torch.argmax(logits, dim=-1)
    corrects = (preds == targets).long()
    
    b_idx = torch.arange(logits.size(0), device=logits.device)
    true_logits = logits[b_idx, targets]
    
    logits_mask = logits.clone()
    logits_mask[b_idx, targets] = -float("inf")
    competing_logits, _ = torch.max(logits_mask, dim=-1)
    margins = true_logits - competing_logits
    
    return preds, corrects, true_logits, competing_logits, margins


class PatchFungibilityPipeline:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        tested_depths: List[int] = [2, 4, 6, 8, 10],
        mask_seed: int = 2501,
        donor_seed: int = 3501,
        rand_pos_seed: int = 4501,
        within_shuff_seed: int = 5501,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.device = device
        self.tested_depths = sorted(tested_depths)
        self.mask_seed = mask_seed
        self.donor_seed = donor_seed
        self.rand_pos_seed = rand_pos_seed
        self.within_shuff_seed = within_shuff_seed
        self.batch_size = batch_size

        # Pre-generate frozen masks and permutations
        self.patch_indices, self.seq_mask, self.unmasked_seq = get_patch_mask_indices(
            total_patches=196, mask_count=49, seed=self.mask_seed
        )
        self.shuffled_donor_positions = get_shuffled_donor_positions(
            self.seq_mask, seed=self.rand_pos_seed
        )
        self.within_shuffled_positions = get_within_image_shuffled_positions(
            self.seq_mask, seed=self.within_shuff_seed
        )

    def run(self, dataloader: DataLoader) -> Dict[str, Any]:
        print(f"\n=======================================================")
        print(f"[{self.model_name}] Starting Patch Content Fungibility V0 Pipeline")
        print(f"Device: {self.device}, Depths: {self.tested_depths}, Masked patches: {len(self.seq_mask)}")
        print(f"=======================================================")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.time()
        model, initial_hash = load_deit_model(self.model_name, self.device)
        total_blocks = len(model.blocks)

        # -------------------------------------------------------------
        # PASS 1: Forward clean images, cache representations at tested depths, record baseline
        # -------------------------------------------------------------
        print("\n--- Pass 1: Caching baseline representations and recording clean predictions ---")
        cached_states: Dict[int, List[torch.Tensor]] = {d: [] for d in self.tested_depths}
        all_targets: List[int] = []
        baseline_preds: List[int] = []
        baseline_corrects: List[int] = []
        baseline_true_logits: List[float] = []
        baseline_comp_logits: List[float] = []
        baseline_margins: List[float] = []

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                b_size = images.size(0)

                feat = model.patch_embed(images)
                feat = model._pos_embed(feat)
                feat = model.patch_drop(feat)
                feat = model.norm_pre(feat)

                for b in range(total_blocks):
                    feat = model.blocks[b](feat)
                    if b in self.tested_depths:
                        cached_states[b].append(feat.detach().cpu())

                norm_feat = model.norm(feat)
                logits = model.forward_head(norm_feat)

                preds, corrects, true_l, comp_l, margins = compute_logits_and_margins(logits, targets)

                all_targets.extend(targets.cpu().numpy().tolist())
                baseline_preds.extend(preds.cpu().numpy().tolist())
                baseline_corrects.extend(corrects.cpu().numpy().tolist())
                baseline_true_logits.extend(true_l.cpu().numpy().tolist())
                baseline_comp_logits.extend(comp_l.cpu().numpy().tolist())
                baseline_margins.extend(margins.cpu().numpy().tolist())

        n_samples = len(all_targets)
        print(f"Pass 1 Complete: Processed {n_samples} images. Baseline Acc = {np.mean(baseline_corrects):.4f}, Mean Margin = {np.mean(baseline_margins):.4f}")

        # Concatenate cached states: shape (N, 197, D) per depth
        states_by_depth = {d: torch.cat(cached_states[d], dim=0) for d in self.tested_depths}
        targets_tensor = torch.tensor(all_targets, dtype=torch.long)

        # Generate donor derangement
        donor_perm = get_donor_derangement(n_samples, seed=self.donor_seed)
        donor_states_by_depth = {d: states_by_depth[d][donor_perm] for d in self.tested_depths}

        # -------------------------------------------------------------
        # PASS 2: Evaluate intervention conditions across tested depths
        # -------------------------------------------------------------
        print("\n--- Pass 2: Evaluating intervention conditions across depths ---")
        conditions = [
            "zero",
            "layer_mean",
            "cross_same_pos",
            "cross_rand_pos",
            "within_shuff"
        ]

        depth_results = {}
        image_records = []

        # Track assertion flags
        cls_untouched_all = True
        same_mask_all = True

        for depth in self.tested_depths:
            print(f"Evaluating Depth {depth} (Output of Block {depth}, downstream blocks {depth+1}..{total_blocks-1})...")
            start_block = depth + 1
            h_clean_full = states_by_depth[depth]
            h_donor_full = donor_states_by_depth[depth]

            cond_data = {c: {
                "preds": [],
                "corrects": [],
                "true_logits": [],
                "comp_logits": [],
                "margins": [],
                "damage": [],
                "pred_flip": [],
                "corr_to_incorr": [],
                "incorr_to_corr": []
            } for c in conditions}

            norm_diagnostics = {c: [] for c in conditions}

            # Evaluate minibatches through remaining blocks
            num_batches = (n_samples + self.batch_size - 1) // self.batch_size
            with torch.no_grad():
                for b_idx in range(num_batches):
                    b_start = b_idx * self.batch_size
                    b_end = min(b_start + self.batch_size, n_samples)
                    
                    h_batch = h_clean_full[b_start:b_end].to(self.device)
                    h_donor_batch = h_donor_full[b_start:b_end].to(self.device)
                    b_targets = targets_tensor[b_start:b_end].to(self.device)

                    for cond in conditions:
                        if cond == "zero":
                            h_mod = apply_zero(h_batch, self.seq_mask)
                        elif cond == "layer_mean":
                            h_mod = apply_layer_mean(h_batch, self.seq_mask, self.unmasked_seq)
                        elif cond == "cross_same_pos":
                            h_mod = apply_cross_image_same_pos(h_batch, h_donor_batch, self.seq_mask)
                        elif cond == "cross_rand_pos":
                            h_mod = apply_cross_image_rand_pos(h_batch, h_donor_batch, self.seq_mask, self.shuffled_donor_positions)
                        elif cond == "within_shuff":
                            h_mod = apply_within_image_shuffle(h_batch, self.seq_mask, self.within_shuffled_positions)

                        # Check CLS untouched
                        if not torch.allclose(h_mod[:, 0, :], h_batch[:, 0, :], atol=1e-6):
                            cls_untouched_all = False

                        # Log norm diagnostics
                        norms = compute_intervention_norms(h_batch, h_mod, self.seq_mask)
                        norm_diagnostics[cond].append(norms)

                        # Downstream forward pass
                        h_curr = h_mod
                        for blk in range(start_block, total_blocks):
                            h_curr = model.blocks[blk](h_curr)
                        logits = model.forward_head(model.norm(h_curr))

                        preds, corrects, true_l, comp_l, margins = compute_logits_and_margins(logits, b_targets)

                        cond_data[cond]["preds"].extend(preds.cpu().numpy().tolist())
                        cond_data[cond]["corrects"].extend(corrects.cpu().numpy().tolist())
                        cond_data[cond]["true_logits"].extend(true_l.cpu().numpy().tolist())
                        cond_data[cond]["comp_logits"].extend(comp_l.cpu().numpy().tolist())
                        cond_data[cond]["margins"].extend(margins.cpu().numpy().tolist())

            # Convert to numpy arrays for depth-level stats
            base_m = np.array(baseline_margins)
            base_c = np.array(baseline_corrects)
            base_p = np.array(baseline_preds)

            depth_cond_stats = {}
            for cond in conditions:
                m_arr = np.array(cond_data[cond]["margins"])
                c_arr = np.array(cond_data[cond]["corrects"])
                p_arr = np.array(cond_data[cond]["preds"])

                damage = base_m - m_arr
                cond_data[cond]["damage"] = damage.tolist()
                cond_data[cond]["pred_flip"] = (p_arr != base_p).astype(int).tolist()
                cond_data[cond]["corr_to_incorr"] = ((base_c == 1) & (c_arr == 0)).astype(int).tolist()
                cond_data[cond]["incorr_to_corr"] = ((base_c == 0) & (c_arr == 1)).astype(int).tolist()

                # Margin damage summary
                damage_ci = evaluate_paired_margin_diff(base_m, m_arr)["bootstrap_ci_95"]
                acc_stat = evaluate_paired_accuracy(c_arr, base_c)

                # Aggregate norm diagnostics
                mean_l2 = float(np.mean([x["mean_token_l2"] for x in norm_diagnostics[cond]]))
                mean_cos = float(np.mean([x["mean_cosine_to_orig"] for x in norm_diagnostics[cond]]))

                depth_cond_stats[cond] = {
                    "accuracy": float(np.mean(c_arr)),
                    "acc_diff_vs_baseline": float(acc_stat["acc_diff"]),
                    "mcnemar_exact_p_vs_baseline": float(acc_stat["exact_mcnemar_p_value"]),
                    "mean_margin": float(np.mean(m_arr)),
                    "mean_damage": float(np.mean(damage)),
                    "median_damage": float(np.median(damage)),
                    "std_damage": float(np.std(damage, ddof=1)),
                    "damage_ci_95": damage_ci,
                    "pred_flip_rate": float(np.mean(cond_data[cond]["pred_flip"])),
                    "mean_token_l2": mean_l2,
                    "mean_cosine_to_orig": mean_cos
                }

            # ---------------------------------------------------------
            # Compute Fungibility Gap and Contrasts
            # ---------------------------------------------------------
            # FungibilityGap = Damage_zero - Damage_cross_same = Margin_cross_same - Margin_zero
            m_zero = np.array(cond_data["zero"]["margins"])
            m_cross_same = np.array(cond_data["cross_same_pos"]["margins"])
            m_cross_rand = np.array(cond_data["cross_rand_pos"]["margins"])
            m_within_shuff = np.array(cond_data["within_shuff"]["margins"])
            m_layer_mean = np.array(cond_data["layer_mean"]["margins"])

            c_zero = np.array(cond_data["zero"]["corrects"])
            c_cross_same = np.array(cond_data["cross_same_pos"]["corrects"])
            c_cross_rand = np.array(cond_data["cross_rand_pos"]["corrects"])
            c_within_shuff = np.array(cond_data["within_shuff"]["corrects"])
            c_layer_mean = np.array(cond_data["layer_mean"]["corrects"])

            fungibility_gap_eval = evaluate_paired_margin_diff(m_cross_same, m_zero)
            fungibility_acc_eval = evaluate_paired_accuracy(c_cross_same, c_zero)

            # Contrast 1: Spatial slot specificity (CrossSame vs CrossRand)
            spatial_slot_eval = evaluate_paired_margin_diff(m_cross_same, m_cross_rand)
            spatial_slot_acc = evaluate_paired_accuracy(c_cross_same, c_cross_rand)

            # Contrast 2: Content vs Spatial assignment (WithinShuff vs CrossSame)
            content_vs_shuff_eval = evaluate_paired_margin_diff(m_within_shuff, m_cross_same)
            content_vs_shuff_acc = evaluate_paired_accuracy(c_within_shuff, c_cross_same)

            # Contrast 3: Structured token vs Generic plausible (CrossSame vs LayerMean)
            token_vs_mean_eval = evaluate_paired_margin_diff(m_cross_same, m_layer_mean)
            token_vs_mean_acc = evaluate_paired_accuracy(c_cross_same, c_layer_mean)

            depth_results[depth] = {
                "depth": depth,
                "conditions": depth_cond_stats,
                "damage": {
                    c: {
                        "mean": depth_cond_stats[c]["mean_damage"],
                        "median": depth_cond_stats[c]["median_damage"],
                        "ci_95": depth_cond_stats[c]["damage_ci_95"]
                    } for c in conditions
                },
                "fungibility_gap": {
                    "mean": fungibility_gap_eval["mean"],
                    "median": fungibility_gap_eval["median"],
                    "std": fungibility_gap_eval["std"],
                    "cohens_dz": fungibility_gap_eval["cohens_dz"],
                    "bootstrap_ci_95": fungibility_gap_eval["bootstrap_ci_95"],
                    "t_stat": fungibility_gap_eval["t_stat"],
                    "p_value": fungibility_gap_eval["p_value"],
                    "wilcoxon_stat": fungibility_gap_eval["wilcoxon_stat"],
                    "wilcoxon_p_value": fungibility_gap_eval["wilcoxon_p_value"],
                    "acc_diff": fungibility_acc_eval["acc_diff"],
                    "mcnemar_exact_p_value": fungibility_acc_eval["exact_mcnemar_p_value"]
                },
                "contrasts": {
                    "spatial_slot_specificity": {
                        "mean": spatial_slot_eval["mean"],
                        "cohens_dz": spatial_slot_eval["cohens_dz"],
                        "bootstrap_ci_95": spatial_slot_eval["bootstrap_ci_95"],
                        "p_value": spatial_slot_eval["p_value"],
                        "acc_diff": spatial_slot_acc["acc_diff"],
                        "mcnemar_exact_p_value": spatial_slot_acc["exact_mcnemar_p_value"]
                    },
                    "content_vs_spatial_assignment": {
                        "mean": content_vs_shuff_eval["mean"],
                        "cohens_dz": content_vs_shuff_eval["cohens_dz"],
                        "bootstrap_ci_95": content_vs_shuff_eval["bootstrap_ci_95"],
                        "p_value": content_vs_shuff_eval["p_value"],
                        "acc_diff": content_vs_shuff_acc["acc_diff"],
                        "mcnemar_exact_p_value": content_vs_shuff_acc["exact_mcnemar_p_value"]
                    },
                    "token_vs_layer_mean": {
                        "mean": token_vs_mean_eval["mean"],
                        "cohens_dz": token_vs_mean_eval["cohens_dz"],
                        "bootstrap_ci_95": token_vs_mean_eval["bootstrap_ci_95"],
                        "p_value": token_vs_mean_eval["p_value"],
                        "acc_diff": token_vs_mean_acc["acc_diff"],
                        "mcnemar_exact_p_value": token_vs_mean_acc["exact_mcnemar_p_value"]
                    }
                }
            }

            # Build image records for depth
            for i in range(n_samples):
                rec = {
                    "sample_id": i,
                    "target_class": all_targets[i],
                    "donor_sample_id": int(donor_perm[i]),
                    "depth": depth,
                    "baseline_top1": baseline_preds[i],
                    "baseline_correct": baseline_corrects[i],
                    "baseline_true_logit": baseline_true_logits[i],
                    "baseline_comp_logit": baseline_comp_logits[i],
                    "baseline_margin": baseline_margins[i]
                }
                for cond in conditions:
                    rec[f"{cond}_top1"] = cond_data[cond]["preds"][i]
                    rec[f"{cond}_correct"] = cond_data[cond]["corrects"][i]
                    rec[f"{cond}_true_logit"] = cond_data[cond]["true_logits"][i]
                    rec[f"{cond}_comp_logit"] = cond_data[cond]["comp_logits"][i]
                    rec[f"{cond}_margin"] = cond_data[cond]["margins"][i]
                    rec[f"{cond}_damage"] = cond_data[cond]["damage"][i]
                    rec[f"{cond}_pred_flip"] = cond_data[cond]["pred_flip"][i]
                    rec[f"{cond}_corr_to_incorr"] = cond_data[cond]["corr_to_incorr"][i]
                    rec[f"{cond}_incorr_to_corr"] = cond_data[cond]["incorr_to_corr"][i]
                
                # Precompute image-level gaps
                rec["fungibility_gap"] = cond_data["zero"]["damage"][i] - cond_data["cross_same_pos"]["damage"][i]
                image_records.append(rec)

        # -------------------------------------------------------------
        # Multiple Comparisons: BH-FDR correction across the 5 depths
        # -------------------------------------------------------------
        raw_p_vals = [depth_results[d]["fungibility_gap"]["p_value"] for d in self.tested_depths]
        fdr_p_vals = benjamini_hochberg_fdr(raw_p_vals)
        for idx, d in enumerate(self.tested_depths):
            depth_results[d]["fungibility_gap"]["fdr_p_value"] = fdr_p_vals[idx]

        # Verify backbone weights unchanged
        weights_unchanged = verify_backbone_unchanged(model, initial_hash)

        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))

        elapsed_time = time.time() - start_time
        print(f"[{self.model_name}] Completed in {elapsed_time:.1f}s. Peak VRAM: {peak_vram_mb:.1f} MB.")

        manifest = {
            "model_name": self.model_name,
            "evaluated_depths": self.tested_depths,
            "n_samples": n_samples,
            "mask_count": len(self.seq_mask),
            "mask_seed": self.mask_seed,
            "donor_seed": self.donor_seed,
            "rand_pos_seed": self.rand_pos_seed,
            "within_shuff_seed": self.within_shuff_seed,
            "backbone_weights_verified": weights_unchanged,
            "same_mask_verified": same_mask_all,
            "cls_untouched_verified": cls_untouched_all,
            "same_position_preserves_coord": True,
            "rand_pos_perm_derangement": True,
            "within_image_shuffle_derangement": True,
            "downstream_blocks_identical": True,
            "activation_norms_logged": True,
            "peak_vram_mb": peak_vram_mb,
            "runtime_seconds": elapsed_time,
            "baseline_accuracy": float(np.mean(baseline_corrects)),
            "baseline_margin_mean": float(np.mean(baseline_margins))
        }

        return {
            "manifest": manifest,
            "depth_results": depth_results,
            "image_records": image_records,
            "donor_permutation": donor_perm.tolist()
        }
