import time
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

from delta_transport.models import load_deit_model, verify_backbone_unchanged
from delta_transport.oracle import (
    forward_to_injection,
    forward_from_injection,
    compute_true_class_margins,
    compute_candidate_deltas,
    compute_margin_gradient
)
from delta_transport.metrics import evaluate_paired_comparison, compute_paired_bootstrap_ci
from delta_transport.v0_5_controls import (
    generate_cross_image_derangement,
    generate_random_oracle_candidates,
    generate_spatial_shuffle_candidates,
    generate_pooled_delta_candidates,
    evaluate_matched_oracle_condition
)


class DeltaV05Pipeline:
    """
    Executes the DELTA TRANSPORT V0.5 matched-oracle specificity falsification experiment.
    """
    def __init__(
        self,
        model_name: str = "deit_tiny_patch16_224",
        device: str = "cuda",
        gamma: float = 0.05,
        random_oracle_seeds: List[int] = [2501, 2502, 2503, 2504, 2505],
        cross_image_seed: int = 3501,
        spatial_shuffle_seeds: List[int] = [4501, 4502, 4503],
        pooled_delta_seed: int = 5501,
        start_block: int = 9,
        total_blocks: int = 12,
        candidate_layers: List[int] = list(range(8))
    ):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.gamma = gamma
        self.random_oracle_seeds = random_oracle_seeds
        self.cross_image_seed = cross_image_seed
        self.spatial_shuffle_seeds = spatial_shuffle_seeds
        self.pooled_delta_seed = pooled_delta_seed
        self.start_block = start_block
        self.total_blocks = total_blocks
        self.candidate_layers = candidate_layers

    def run(
        self,
        dataloader: DataLoader,
        run_sensitivities: bool = True
    ) -> Dict[str, Any]:
        print(f"\n=======================================================")
        print(f"[{self.model_name}] Starting Delta Transport V0.5 Pipeline")
        print(f"Device: {self.device}, Gamma: {self.gamma}")
        print(f"Random Seeds: {self.random_oracle_seeds}")
        print(f"Cross-Image Seed: {self.cross_image_seed}, Shuffle Seeds: {self.spatial_shuffle_seeds}")
        print(f"=======================================================")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.time()
        model, initial_hash = load_deit_model(self.model_name, self.device)
        total_images = len(dataloader.dataset)

        # -----------------------------------------------------------------
        # PASS 1: Collect Historical Delta Bank for Cross-Image Donor Pool
        # -----------------------------------------------------------------
        print(f"[{self.model_name}] Pass 1: Caching historical deltas for cross-image donor pool...")
        all_hist_u_list = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                states = forward_to_injection(model, images, injection_layer=self.start_block - 1)
                _, u = compute_candidate_deltas(states, self.candidate_layers)
                # Store on CPU in float16 to conserve host RAM
                all_hist_u_list.append(u.cpu().half())

        all_hist_u_cpu = torch.cat(all_hist_u_list, dim=0) # (1000, 8, 196, d)
        print(f"[{self.model_name}] Cached donor delta bank shape: {all_hist_u_cpu.shape}, memory: {all_hist_u_cpu.element_size() * all_hist_u_cpu.nelement() / (1024*1024):.1f} MB.")

        # Generate deterministic cross-image derangement
        donor_indices_all = generate_cross_image_derangement(total_images, seed=self.cross_image_seed)

        # -----------------------------------------------------------------
        # PASS 2: Execute Matched-Oracle Specificity Falsification
        # -----------------------------------------------------------------
        print(f"[{self.model_name}] Pass 2: Running matched-capacity oracle evaluations...")
        
        # Arrays for paired statistics
        base_margins_all = []
        base_correct_all = []

        hist_margins_all = []
        hist_correct_all = []
        hist_diversity_all = []

        random_margins_all = {s: [] for s in self.random_oracle_seeds}
        random_correct_all = {s: [] for s in self.random_oracle_seeds}
        random_diversity_all = {s: [] for s in self.random_oracle_seeds}

        cross_margins_all = []
        cross_correct_all = []
        cross_diversity_all = []

        shuff_margins_all = {s: [] for s in self.spatial_shuffle_seeds}
        shuff_correct_all = {s: [] for s in self.spatial_shuffle_seeds}
        shuff_diversity_all = {s: [] for s in self.spatial_shuffle_seeds}

        pooled_margins_all = []
        pooled_correct_all = []
        pooled_diversity_all = []

        global_margins_all = []
        global_correct_all = []

        image_records = []
        img_counter = 0

        # Sensitivity storage (Historical vs Random Seed 2501)
        sens_results = {0.02: {"hist_m": [], "hist_c": [], "rand_m": [], "rand_c": []},
                        0.10: {"hist_m": [], "hist_c": [], "rand_m": [], "rand_c": []}}

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            b_size = images.size(0)
            curr_indices = np.arange(img_counter, img_counter + b_size)

            # 1. Forward to injection point
            states = forward_to_injection(model, images, injection_layer=self.start_block - 1)
            h_inj = states[self.start_block]
            patch_h = h_inj[:, 1:, :].detach()
            patch_norm = patch_h.norm(dim=-1, keepdim=True)
            cls_h = h_inj[:, :1, :].detach()

            # 2. Baseline & Gradients
            base_logits, base_margins, patch_grads = compute_margin_gradient(
                model, h_inj, targets, self.start_block, self.total_blocks
            )
            base_preds = base_logits.argmax(dim=1)
            base_correct = (base_preds == targets)

            base_margins_all.extend(base_margins.cpu().numpy())
            base_correct_all.extend(base_correct.cpu().numpy())

            # 3. True Historical Candidate Deltas
            _, u_hist = compute_candidate_deltas(states, self.candidate_layers)

            # --- Condition 0: TRUE HISTORICAL TOKENWISE ORACLE ---
            res_hist = evaluate_matched_oracle_condition(
                model, h_inj, patch_grads, patch_norm, cls_h, u_hist,
                self.gamma, self.start_block, self.total_blocks, targets
            )
            hist_margins_all.extend(res_hist["margins"])
            hist_correct_all.extend(res_hist["correct"])
            hist_diversity_all.extend(res_hist["diversity_stats"])

            # --- Control 1: RANDOM-DIRECTION TOKENWISE ORACLE (5 Seeds) ---
            res_random_batch = {}
            for r_seed in self.random_oracle_seeds:
                u_rand = generate_random_oracle_candidates(
                    b_size, len(self.candidate_layers), 196, h_inj.size(-1), r_seed + img_counter, self.device
                )
                res_rand = evaluate_matched_oracle_condition(
                    model, h_inj, patch_grads, patch_norm, cls_h, u_rand,
                    self.gamma, self.start_block, self.total_blocks, targets
                )
                random_margins_all[r_seed].extend(res_rand["margins"])
                random_correct_all[r_seed].extend(res_rand["correct"])
                random_diversity_all[r_seed].extend(res_rand["diversity_stats"])
                res_random_batch[r_seed] = res_rand

            # --- Control 2: CROSS-IMAGE HISTORICAL TOKENWISE ORACLE (Seed 3501) ---
            batch_donor_indices = donor_indices_all[curr_indices]
            # Verify donor is never target
            assert (batch_donor_indices != curr_indices).all(), "Donor image equals target image in batch!"
            u_cross = all_hist_u_cpu[batch_donor_indices].to(self.device).float()
            res_cross = evaluate_matched_oracle_condition(
                model, h_inj, patch_grads, patch_norm, cls_h, u_cross,
                self.gamma, self.start_block, self.total_blocks, targets
            )
            cross_margins_all.extend(res_cross["margins"])
            cross_correct_all.extend(res_cross["correct"])
            cross_diversity_all.extend(res_cross["diversity_stats"])

            # --- Control 3: SPATIALLY SHUFFLED HISTORICAL TOKENWISE ORACLE (3 Seeds) ---
            res_shuff_batch = {}
            for s_seed in self.spatial_shuffle_seeds:
                u_shuff = generate_spatial_shuffle_candidates(u_hist, seed=s_seed + img_counter)
                res_shuff = evaluate_matched_oracle_condition(
                    model, h_inj, patch_grads, patch_norm, cls_h, u_shuff,
                    self.gamma, self.start_block, self.total_blocks, targets
                )
                shuff_margins_all[s_seed].extend(res_shuff["margins"])
                shuff_correct_all[s_seed].extend(res_shuff["correct"])
                shuff_diversity_all[s_seed].extend(res_shuff["diversity_stats"])
                res_shuff_batch[s_seed] = res_shuff

            # --- Control 4: UNCOUPLED POOLED DELTA BANK (Seed 5501) ---
            u_pooled = generate_pooled_delta_candidates(u_hist, seed=self.pooled_delta_seed + img_counter)
            res_pooled = evaluate_matched_oracle_condition(
                model, h_inj, patch_grads, patch_norm, cls_h, u_pooled,
                self.gamma, self.start_block, self.total_blocks, targets
            )
            pooled_margins_all.extend(res_pooled["margins"])
            pooled_correct_all.extend(res_pooled["correct"])
            pooled_diversity_all.extend(res_pooled["diversity_stats"])

            # Also evaluate Global Oracle for reference
            s_hist = (u_hist * patch_grads.unsqueeze(1)).sum(dim=-1) # (B, 8, 196)
            global_l = s_hist.sum(dim=2).argmax(dim=1) # (B,)
            u_global = torch.gather(
                u_hist, 1, global_l.view(b_size, 1, 1, 1).expand(b_size, 1, 196, h_inj.size(-1))
            ).squeeze(1)
            pert_gl = self.gamma * patch_norm * u_global
            h_gl = torch.cat([cls_h, patch_h + pert_gl], dim=1)
            with torch.no_grad():
                gl_logits = forward_from_injection(model, h_gl, self.start_block, self.total_blocks)
            gl_margins = compute_true_class_margins(gl_logits, targets)
            global_margins_all.extend(gl_margins.cpu().numpy())
            global_correct_all.extend((gl_logits.argmax(dim=1) == targets).cpu().numpy())

            # Sensitivity evaluations
            if run_sensitivities:
                for s_gamma in [0.02, 0.10]:
                    sens_hist = evaluate_matched_oracle_condition(
                        model, h_inj, patch_grads, patch_norm, cls_h, u_hist,
                        s_gamma, self.start_block, self.total_blocks, targets
                    )
                    u_rand_sens = generate_random_oracle_candidates(
                        b_size, len(self.candidate_layers), 196, h_inj.size(-1), 2501 + img_counter, self.device
                    )
                    sens_rand = evaluate_matched_oracle_condition(
                        model, h_inj, patch_grads, patch_norm, cls_h, u_rand_sens,
                        s_gamma, self.start_block, self.total_blocks, targets
                    )
                    sens_results[s_gamma]["hist_m"].extend(sens_hist["margins"])
                    sens_results[s_gamma]["hist_c"].extend(sens_hist["correct"])
                    sens_results[s_gamma]["rand_m"].extend(sens_rand["margins"])
                    sens_results[s_gamma]["rand_c"].extend(sens_rand["correct"])

            # Build per-image records
            for b in range(b_size):
                global_idx = img_counter + b
                target_cls = int(targets[b].cpu().item())
                donor_id = int(donor_indices_all[global_idx])

                b_m = float(base_margins.cpu().numpy()[b])
                b_c = bool(base_correct.cpu().numpy()[b])

                h_m = float(res_hist["margins"][b])
                h_c = bool(res_hist["correct"][b])

                # Mean random margin across seeds for this image
                r_ms = [float(res_random_batch[s]["margins"][b]) for s in self.random_oracle_seeds]
                r_mean_m = float(np.mean(r_ms))

                c_m = float(res_cross["margins"][b])
                c_c = bool(res_cross["correct"][b])

                # Mean shuffled margin across seeds for this image
                sh_ms = [float(res_shuff_batch[s]["margins"][b]) for s in self.spatial_shuffle_seeds]
                sh_mean_m = float(np.mean(sh_ms))

                p_m = float(res_pooled["margins"][b])
                p_c = bool(res_pooled["correct"][b])

                gl_m = float(gl_margins.cpu().numpy()[b])

                image_records.append({
                    "sample_id": global_idx,
                    "target_class": target_cls,
                    "cross_image_donor_id": donor_id,
                    "base_margin": b_m,
                    "base_correct": b_c,
                    "historical_margin": h_m,
                    "historical_correct": h_c,
                    "historical_delta_vs_base": h_m - b_m,
                    "random_oracle_mean_margin": r_mean_m,
                    "random_oracle_delta_vs_base": r_mean_m - b_m,
                    "historical_vs_random_diff": h_m - r_mean_m,
                    "cross_image_margin": c_m,
                    "cross_image_correct": c_c,
                    "cross_image_delta_vs_base": c_m - b_m,
                    "historical_vs_cross_diff": h_m - c_m,
                    "spatial_shuffled_mean_margin": sh_mean_m,
                    "spatial_shuffled_delta_vs_base": sh_mean_m - b_m,
                    "historical_vs_shuffled_diff": h_m - sh_mean_m,
                    "pooled_delta_margin": p_m,
                    "pooled_delta_correct": p_c,
                    "global_oracle_margin": gl_m
                })

            img_counter += b_size
            if (batch_idx + 1) % 10 == 0 or img_counter == total_images:
                print(f"[{self.model_name}] Processed {img_counter}/{total_images} samples...")

        # Verify backbone weights unchanged
        verify_backbone_unchanged(model, initial_hash)

        runtime_s = time.time() - start_time
        peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

        # Convert arrays to numpy
        base_margins = np.array(base_margins_all)
        base_correct = np.array(base_correct_all, dtype=int)

        hist_margins = np.array(hist_margins_all)
        hist_correct = np.array(hist_correct_all, dtype=int)

        cross_margins = np.array(cross_margins_all)
        cross_correct = np.array(cross_correct_all, dtype=int)

        pooled_margins = np.array(pooled_margins_all)
        pooled_correct = np.array(pooled_correct_all, dtype=int)

        global_margins = np.array(global_margins_all)
        global_correct = np.array(global_correct_all, dtype=int)

        # Multi-seed evaluations
        # 1. Random seeds
        random_seed_comparisons = []
        for s in self.random_oracle_seeds:
            r_m = np.array(random_margins_all[s])
            r_c = np.array(random_correct_all[s], dtype=int)
            cmp_res = evaluate_paired_comparison(
                hist_margins, r_m, hist_correct, r_c,
                label_a="Historical", label_b=f"Random_{s}"
            )
            cmp_res["seed"] = s
            cmp_res["control_type"] = "random_direction"
            random_seed_comparisons.append(cmp_res)

        # Mean random across seeds
        random_margins_matrix = np.array([random_margins_all[s] for s in self.random_oracle_seeds]) # (5, 1000)
        random_mean_margins = np.mean(random_margins_matrix, axis=0) # (1000,)
        random_mean_correct = (np.mean([random_correct_all[s] for s in self.random_oracle_seeds], axis=0) >= 0.5).astype(int)

        hist_vs_random_mean = evaluate_paired_comparison(
            hist_margins, random_mean_margins, hist_correct, random_mean_correct,
            label_a="Historical", label_b="RandomOracleMean"
        )

        # 2. Spatial shuffle seeds
        shuff_seed_comparisons = []
        for s in self.spatial_shuffle_seeds:
            sh_m = np.array(shuff_margins_all[s])
            sh_c = np.array(shuff_correct_all[s], dtype=int)
            cmp_res = evaluate_paired_comparison(
                hist_margins, sh_m, hist_correct, sh_c,
                label_a="Historical", label_b=f"SpatialShuffle_{s}"
            )
            cmp_res["seed"] = s
            cmp_res["control_type"] = "spatial_shuffle"
            shuff_seed_comparisons.append(cmp_res)

        shuff_margins_matrix = np.array([shuff_margins_all[s] for s in self.spatial_shuffle_seeds]) # (3, 1000)
        shuff_mean_margins = np.mean(shuff_margins_matrix, axis=0)
        shuff_mean_correct = (np.mean([shuff_correct_all[s] for s in self.spatial_shuffle_seeds], axis=0) >= 0.5).astype(int)

        hist_vs_shuff_mean = evaluate_paired_comparison(
            hist_margins, shuff_mean_margins, hist_correct, shuff_mean_correct,
            label_a="Historical", label_b="SpatialShuffleMean"
        )

        # 3. Cross-Image comparison
        hist_vs_cross = evaluate_paired_comparison(
            hist_margins, cross_margins, hist_correct, cross_correct,
            label_a="Historical", label_b="CrossImage"
        )

        # 4. Pooled Delta comparison
        hist_vs_pooled = evaluate_paired_comparison(
            hist_margins, pooled_margins, hist_correct, pooled_correct,
            label_a="Historical", label_b="PooledDelta"
        )

        # Historical vs Baseline (Reference)
        hist_vs_base = evaluate_paired_comparison(
            hist_margins, base_margins, hist_correct, base_correct,
            label_a="Historical", label_b="Baseline"
        )

        # Sensitivity comparisons
        sensitivity_summaries = {}
        if run_sensitivities:
            for s_gamma in [0.02, 0.10]:
                h_m_s = np.array(sens_results[s_gamma]["hist_m"])
                h_c_s = np.array(sens_results[s_gamma]["hist_c"], dtype=int)
                r_m_s = np.array(sens_results[s_gamma]["rand_m"])
                r_c_s = np.array(sens_results[s_gamma]["rand_c"], dtype=int)
                cmp_sens = evaluate_paired_comparison(
                    h_m_s, r_m_s, h_c_s, r_c_s,
                    label_a="Historical", label_b=f"Random_gamma_{s_gamma}"
                )
                sensitivity_summaries[f"gamma_{s_gamma}"] = cmp_sens

        # Diversity metrics summary across conditions
        diversity_summary = {
            "historical": {
                "entropy_mean": float(np.mean([d["entropy"] for d in hist_diversity_all])),
                "dominant_frac_mean": float(np.mean([d["dominant_fraction"] for d in hist_diversity_all])),
                "distinct_mean": float(np.mean([d["num_distinct_candidates"] for d in hist_diversity_all]))
            },
            "random_oracle": {
                "entropy_mean": float(np.mean([d["entropy"] for s in self.random_oracle_seeds for d in random_diversity_all[s]])),
                "dominant_frac_mean": float(np.mean([d["dominant_fraction"] for s in self.random_oracle_seeds for d in random_diversity_all[s]])),
                "distinct_mean": float(np.mean([d["num_distinct_candidates"] for s in self.random_oracle_seeds for d in random_diversity_all[s]]))
            },
            "cross_image": {
                "entropy_mean": float(np.mean([d["entropy"] for d in cross_diversity_all])),
                "dominant_frac_mean": float(np.mean([d["dominant_fraction"] for d in cross_diversity_all])),
                "distinct_mean": float(np.mean([d["num_distinct_candidates"] for d in cross_diversity_all]))
            },
            "spatial_shuffle": {
                "entropy_mean": float(np.mean([d["entropy"] for s in self.spatial_shuffle_seeds for d in shuff_diversity_all[s]])),
                "dominant_frac_mean": float(np.mean([d["dominant_fraction"] for s in self.spatial_shuffle_seeds for d in shuff_diversity_all[s]])),
                "distinct_mean": float(np.mean([d["num_distinct_candidates"] for s in self.spatial_shuffle_seeds for d in shuff_diversity_all[s]]))
            }
        }

        # Build DataFrames
        image_df = pd.DataFrame(image_records)

        # Seed results table
        all_seed_rows = []
        for r_res in random_seed_comparisons:
            all_seed_rows.append({
                "control_type": "random_direction",
                "seed": r_res["seed"],
                "historical_mean_margin": float(np.mean(hist_margins)),
                "control_mean_margin": float(np.mean(random_margins_all[r_res["seed"]])),
                "margin_diff": r_res["mean_diff"],
                "ci_lower": r_res["bootstrap_ci_95"][0],
                "ci_upper": r_res["bootstrap_ci_95"][1],
                "cohens_dz": r_res["cohens_dz"],
                "p_value": r_res["p_value"],
                "historical_acc": r_res["acc_a"],
                "control_acc": r_res["acc_b"],
                "acc_diff": r_res["acc_diff"],
                "mcnemar_p_value": r_res["exact_mcnemar_p_value"]
            })
        for sh_res in shuff_seed_comparisons:
            all_seed_rows.append({
                "control_type": "spatial_shuffle",
                "seed": sh_res["seed"],
                "historical_mean_margin": float(np.mean(hist_margins)),
                "control_mean_margin": float(np.mean(shuff_margins_all[sh_res["seed"]])),
                "margin_diff": sh_res["mean_diff"],
                "ci_lower": sh_res["bootstrap_ci_95"][0],
                "ci_upper": sh_res["bootstrap_ci_95"][1],
                "cohens_dz": sh_res["cohens_dz"],
                "p_value": sh_res["p_value"],
                "historical_acc": sh_res["acc_a"],
                "control_acc": sh_res["acc_b"],
                "acc_diff": sh_res["acc_diff"],
                "mcnemar_p_value": sh_res["exact_mcnemar_p_value"]
            })
        seed_df = pd.DataFrame(all_seed_rows)

        # Matched Oracle Comparison Table
        comparison_rows = [
            {
                "condition": "baseline",
                "mean_margin": float(np.mean(base_margins)),
                "top1_acc": float(np.mean(base_correct)),
                "delta_vs_baseline": 0.0,
                "historical_minus_condition_diff": hist_vs_base["mean_diff"],
                "ci_lower": hist_vs_base["bootstrap_ci_95"][0],
                "ci_upper": hist_vs_base["bootstrap_ci_95"][1],
                "cohens_dz": hist_vs_base["cohens_dz"],
                "p_value": hist_vs_base["p_value"],
                "acc_diff": hist_vs_base["acc_diff"],
                "mcnemar_p_value": hist_vs_base["exact_mcnemar_p_value"]
            },
            {
                "condition": "random_direction_oracle_mean",
                "mean_margin": float(np.mean(random_mean_margins)),
                "top1_acc": float(np.mean([np.mean(random_correct_all[s]) for s in self.random_oracle_seeds])),
                "delta_vs_baseline": float(np.mean(random_mean_margins - base_margins)),
                "historical_minus_condition_diff": hist_vs_random_mean["mean_diff"],
                "ci_lower": hist_vs_random_mean["bootstrap_ci_95"][0],
                "ci_upper": hist_vs_random_mean["bootstrap_ci_95"][1],
                "cohens_dz": hist_vs_random_mean["cohens_dz"],
                "p_value": hist_vs_random_mean["p_value"],
                "acc_diff": hist_vs_random_mean["acc_diff"],
                "mcnemar_p_value": hist_vs_random_mean["exact_mcnemar_p_value"]
            },
            {
                "condition": "cross_image_oracle",
                "mean_margin": float(np.mean(cross_margins)),
                "top1_acc": float(np.mean(cross_correct)),
                "delta_vs_baseline": float(np.mean(cross_margins - base_margins)),
                "historical_minus_condition_diff": hist_vs_cross["mean_diff"],
                "ci_lower": hist_vs_cross["bootstrap_ci_95"][0],
                "ci_upper": hist_vs_cross["bootstrap_ci_95"][1],
                "cohens_dz": hist_vs_cross["cohens_dz"],
                "p_value": hist_vs_cross["p_value"],
                "acc_diff": hist_vs_cross["acc_diff"],
                "mcnemar_p_value": hist_vs_cross["exact_mcnemar_p_value"]
            },
            {
                "condition": "spatial_shuffled_oracle_mean",
                "mean_margin": float(np.mean(shuff_mean_margins)),
                "top1_acc": float(np.mean([np.mean(shuff_correct_all[s]) for s in self.spatial_shuffle_seeds])),
                "delta_vs_baseline": float(np.mean(shuff_mean_margins - base_margins)),
                "historical_minus_condition_diff": hist_vs_shuff_mean["mean_diff"],
                "ci_lower": hist_vs_shuff_mean["bootstrap_ci_95"][0],
                "ci_upper": hist_vs_shuff_mean["bootstrap_ci_95"][1],
                "cohens_dz": hist_vs_shuff_mean["cohens_dz"],
                "p_value": hist_vs_shuff_mean["p_value"],
                "acc_diff": hist_vs_shuff_mean["acc_diff"],
                "mcnemar_p_value": hist_vs_shuff_mean["exact_mcnemar_p_value"]
            },
            {
                "condition": "pooled_delta_oracle",
                "mean_margin": float(np.mean(pooled_margins)),
                "top1_acc": float(np.mean(pooled_correct)),
                "delta_vs_baseline": float(np.mean(pooled_margins - base_margins)),
                "historical_minus_condition_diff": hist_vs_pooled["mean_diff"],
                "ci_lower": hist_vs_pooled["bootstrap_ci_95"][0],
                "ci_upper": hist_vs_pooled["bootstrap_ci_95"][1],
                "cohens_dz": hist_vs_pooled["cohens_dz"],
                "p_value": hist_vs_pooled["p_value"],
                "acc_diff": hist_vs_pooled["acc_diff"],
                "mcnemar_p_value": hist_vs_pooled["exact_mcnemar_p_value"]
            },
            {
                "condition": "true_historical_oracle",
                "mean_margin": float(np.mean(hist_margins)),
                "top1_acc": float(np.mean(hist_correct)),
                "delta_vs_baseline": hist_vs_base["mean_diff"],
                "historical_minus_condition_diff": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "cohens_dz": 0.0,
                "p_value": 1.0,
                "acc_diff": 0.0,
                "mcnemar_p_value": 1.0
            }
        ]
        matched_comparison_df = pd.DataFrame(comparison_rows)

        # Selection Diversity Table
        diversity_df = pd.DataFrame([
            {"condition": cond, **vals} for cond, vals in diversity_summary.items()
        ])

        print(f"[{self.model_name}] Completed in {runtime_s:.2f}s. Peak VRAM: {peak_vram_mb:.1f} MB.")
        print(f"[{self.model_name}] Historical vs Matched Random: mean Delta_m = {hist_vs_random_mean['mean_diff']:+.4f} "
              f"(95% CI [{hist_vs_random_mean['bootstrap_ci_95'][0]:.4f}, {hist_vs_random_mean['bootstrap_ci_95'][1]:.4f}], "
              f"dz = {hist_vs_random_mean['cohens_dz']:.3f}, p = {hist_vs_random_mean['p_value']:.4e})")

        return {
            "model_name": self.model_name,
            "runtime_seconds": runtime_s,
            "peak_vram_mb": peak_vram_mb,
            "image_df": image_df,
            "matched_comparison_df": matched_comparison_df,
            "seed_df": seed_df,
            "diversity_df": diversity_df,
            "hist_vs_random_mean": hist_vs_random_mean,
            "random_seed_comparisons": random_seed_comparisons,
            "hist_vs_cross": hist_vs_cross,
            "hist_vs_shuff_mean": hist_vs_shuff_mean,
            "shuff_seed_comparisons": shuff_seed_comparisons,
            "hist_vs_pooled": hist_vs_pooled,
            "hist_vs_base": hist_vs_base,
            "diversity_summary": diversity_summary,
            "sensitivity_summaries": sensitivity_summaries
        }
