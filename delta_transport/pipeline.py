import time
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

from delta_transport.models import load_deit_model, verify_backbone_unchanged
from delta_transport.oracle import forward_to_injection, evaluate_policies_batch
from delta_transport.metrics import (
    evaluate_paired_comparison,
    summarize_spatial_diversity
)


class DeltaTransportPipeline:
    """
    Executes the frozen DELTA TRANSPORT V0 falsification experiment for a given model.
    """
    def __init__(
        self,
        model_name: str = "deit_tiny_patch16_224",
        device: str = "cuda",
        gamma: float = 0.05,
        random_seeds: List[int] = [2501, 2502, 2503],
        start_block: int = 9,
        total_blocks: int = 12,
        candidate_layers: List[int] = list(range(8))
    ):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.gamma = gamma
        self.random_seeds = random_seeds
        self.start_block = start_block
        self.total_blocks = total_blocks
        self.candidate_layers = candidate_layers

    def run(
        self,
        dataloader: DataLoader,
        run_sensitivities: bool = True
    ) -> Dict[str, Any]:
        """
        Runs the full evaluation on the dataset.
        """
        print(f"\n=======================================================")
        print(f"[{self.model_name}] Starting Delta Transport V0 Pipeline")
        print(f"Device: {self.device}, Gamma: {self.gamma}, Candidate Layers: {self.candidate_layers}")
        print(f"=======================================================")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.time()
        model, initial_hash = load_deit_model(self.model_name, self.device)

        image_records = []
        source_records = []
        diversity_records = []

        total_samples = 0
        img_counter = 0

        # We also collect arrays for paired statistical testing
        base_margins_all = []
        base_correct_all = []

        most_recent_margins_all = []
        most_recent_correct_all = []

        global_margins_all = []
        global_correct_all = []

        tokenwise_margins_all = []
        tokenwise_correct_all = []

        random_margins_all = {s: [] for s in self.random_seeds}
        random_correct_all = {s: [] for s in self.random_seeds}

        # Sensitivity storage
        sensitivity_results = {}
        if run_sensitivities:
            for sens_gamma in [0.02, 0.10]:
                sensitivity_results[sens_gamma] = {
                    "global_margins": [],
                    "tokenwise_margins": [],
                    "global_correct": [],
                    "tokenwise_correct": []
                }

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            b_size = images.size(0)

            # 1. Forward to injection point (output of block 8)
            states = forward_to_injection(model, images, injection_layer=self.start_block - 1)

            # 2. Evaluate primary policies at gamma = 0.05
            batch_res = evaluate_policies_batch(
                model=model,
                states=states,
                targets=targets,
                gamma=self.gamma,
                random_seeds=self.random_seeds,
                candidate_layers=self.candidate_layers,
                start_block=self.start_block,
                total_blocks=self.total_blocks
            )

            # 3. Optional sensitivity sweeps
            if run_sensitivities:
                for sens_gamma in [0.02, 0.10]:
                    sens_res = evaluate_policies_batch(
                        model=model,
                        states=states,
                        targets=targets,
                        gamma=sens_gamma,
                        random_seeds=self.random_seeds,
                        candidate_layers=self.candidate_layers,
                        start_block=self.start_block,
                        total_blocks=self.total_blocks
                    )
                    sensitivity_results[sens_gamma]["global_margins"].extend(sens_res["global_oracle"]["margins"])
                    sensitivity_results[sens_gamma]["tokenwise_margins"].extend(sens_res["tokenwise_oracle"]["margins"])
                    sensitivity_results[sens_gamma]["global_correct"].extend(sens_res["global_oracle"]["correct"])
                    sensitivity_results[sens_gamma]["tokenwise_correct"].extend(sens_res["tokenwise_oracle"]["correct"])

            # Collect primary policy arrays
            base_margins_all.extend(batch_res["base"]["margins"])
            base_correct_all.extend(batch_res["base"]["correct"])

            most_recent_margins_all.extend(batch_res["most_recent"]["margins"])
            most_recent_correct_all.extend(batch_res["most_recent"]["correct"])

            global_margins_all.extend(batch_res["global_oracle"]["margins"])
            global_correct_all.extend(batch_res["global_oracle"]["correct"])

            tokenwise_margins_all.extend(batch_res["tokenwise_oracle"]["margins"])
            tokenwise_correct_all.extend(batch_res["tokenwise_oracle"]["correct"])

            for s in self.random_seeds:
                random_margins_all[s].extend(batch_res["random_tokenwise"][s]["margins"])
                random_correct_all[s].extend(batch_res["random_tokenwise"][s]["correct"])

            diversity_records.extend(batch_res["diversity_stats"])

            # Build per-image records
            for b in range(b_size):
                img_id = img_counter
                t_class = int(targets[b].cpu().item())
                b_margin = float(batch_res["base"]["margins"][b])
                b_pred = int(batch_res["base"]["preds"][b])
                b_corr = bool(batch_res["base"]["correct"][b])

                mr_margin = float(batch_res["most_recent"]["margins"][b])
                mr_pred = int(batch_res["most_recent"]["preds"][b])
                mr_corr = bool(batch_res["most_recent"]["correct"][b])

                gl_margin = float(batch_res["global_oracle"]["margins"][b])
                gl_pred = int(batch_res["global_oracle"]["preds"][b])
                gl_corr = bool(batch_res["global_oracle"]["correct"][b])
                gl_layer = int(batch_res["global_oracle"]["selected_layer"][b])

                tk_margin = float(batch_res["tokenwise_oracle"]["margins"][b])
                tk_pred = int(batch_res["tokenwise_oracle"]["preds"][b])
                tk_corr = bool(batch_res["tokenwise_oracle"]["correct"][b])

                r_margins = [float(batch_res["random_tokenwise"][s]["margins"][b]) for s in self.random_seeds]
                r_mean = float(np.mean(r_margins))

                # Flip classification between tokenwise and global
                if tk_corr and gl_corr:
                    flip_cat = "both_correct"
                elif tk_corr and not gl_corr:
                    flip_cat = "tokenwise_only"
                elif not tk_corr and gl_corr:
                    flip_cat = "global_only"
                else:
                    flip_cat = "both_incorrect"

                image_records.append({
                    "sample_id": img_id,
                    "target_class": t_class,
                    "base_margin": b_margin,
                    "base_pred": b_pred,
                    "base_correct": b_corr,
                    "most_recent_margin": mr_margin,
                    "most_recent_pred": mr_pred,
                    "most_recent_correct": mr_corr,
                    "most_recent_delta": mr_margin - b_margin,
                    "rand_2501_margin": r_margins[0],
                    "rand_2502_margin": r_margins[1],
                    "rand_2503_margin": r_margins[2],
                    "rand_mean_margin": r_mean,
                    "rand_mean_delta": r_mean - b_margin,
                    "global_margin": gl_margin,
                    "global_pred": gl_pred,
                    "global_correct": gl_corr,
                    "global_delta": gl_margin - b_margin,
                    "global_selected_layer": gl_layer,
                    "tokenwise_margin": tk_margin,
                    "tokenwise_pred": tk_pred,
                    "tokenwise_correct": tk_corr,
                    "tokenwise_delta": tk_margin - b_margin,
                    "tokenwise_vs_global_margin_diff": tk_margin - gl_margin,
                    "flip_category": flip_cat
                })

                div_stat = batch_res["diversity_stats"][b]
                src_dict = {
                    "sample_id": img_id,
                    "global_selected_layer": gl_layer,
                    "entropy": div_stat["entropy"],
                    "dominant_fraction": div_stat["dominant_fraction"],
                    "fraction_matching_global": div_stat["fraction_matching_global"],
                    "num_distinct_layers": div_stat["num_distinct_layers"]
                }
                for l_idx, count_val in enumerate(div_stat["layer_counts"]):
                    src_dict[f"count_layer_{l_idx}"] = count_val
                source_records.append(src_dict)

                img_counter += 1

            total_samples += b_size
            if (batch_idx + 1) % 10 == 0 or total_samples == len(dataloader.dataset):
                print(f"[{self.model_name}] Processed {total_samples}/{len(dataloader.dataset)} samples...")

        # 4. Verify backbone weight immutability
        verify_backbone_unchanged(model, initial_hash)

        runtime_s = time.time() - start_time
        peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

        # Convert to numpy arrays for paired analysis
        base_margins = np.array(base_margins_all)
        base_correct = np.array(base_correct_all, dtype=int)

        most_recent_margins = np.array(most_recent_margins_all)
        most_recent_correct = np.array(most_recent_correct_all, dtype=int)

        global_margins = np.array(global_margins_all)
        global_correct = np.array(global_correct_all, dtype=int)

        tokenwise_margins = np.array(tokenwise_margins_all)
        tokenwise_correct = np.array(tokenwise_correct_all, dtype=int)

        # 5. Paired Statistical Comparisons
        token_vs_global = evaluate_paired_comparison(
            tokenwise_margins, global_margins, tokenwise_correct, global_correct,
            label_a="Tokenwise", label_b="Global"
        )
        global_vs_base = evaluate_paired_comparison(
            global_margins, base_margins, global_correct, base_correct,
            label_a="Global", label_b="Baseline"
        )
        token_vs_base = evaluate_paired_comparison(
            tokenwise_margins, base_margins, tokenwise_correct, base_correct,
            label_a="Tokenwise", label_b="Baseline"
        )
        mr_vs_base = evaluate_paired_comparison(
            most_recent_margins, base_margins, most_recent_correct, base_correct,
            label_a="MostRecent", label_b="Baseline"
        )

        rand_vs_base_list = []
        for s in self.random_seeds:
            r_margins = np.array(random_margins_all[s])
            r_correct = np.array(random_correct_all[s], dtype=int)
            cmp_res = evaluate_paired_comparison(
                r_margins, base_margins, r_correct, base_correct,
                label_a=f"Random_{s}", label_b="Baseline"
            )
            rand_vs_base_list.append(cmp_res)

        # 6. Spatial Diversity Summary
        spatial_summary = summarize_spatial_diversity(diversity_records, num_candidate_layers=len(self.candidate_layers))

        # 7. Build DataFrames
        image_df = pd.DataFrame(image_records)
        source_df = pd.DataFrame(source_records)

        # Policy Comparison Summary Table
        comparison_rows = [
            {
                "policy": "baseline",
                "mean_margin": float(np.mean(base_margins)),
                "std_margin": float(np.std(base_margins, ddof=1)),
                "top1_acc": float(np.mean(base_correct)),
                "margin_delta_vs_base": 0.0,
                "ci95_vs_base_lower": 0.0,
                "ci95_vs_base_upper": 0.0,
                "p_val_vs_base": 1.0,
                "margin_delta_vs_global": -global_vs_base["mean_diff"],
                "p_val_vs_global": global_vs_base["p_value"],
                "cohens_dz_vs_global": -global_vs_base["cohens_dz"]
            },
            {
                "policy": "most_recent_l7",
                "mean_margin": float(np.mean(most_recent_margins)),
                "std_margin": float(np.std(most_recent_margins, ddof=1)),
                "top1_acc": float(np.mean(most_recent_correct)),
                "margin_delta_vs_base": mr_vs_base["mean_diff"],
                "ci95_vs_base_lower": mr_vs_base["bootstrap_ci_95"][0],
                "ci95_vs_base_upper": mr_vs_base["bootstrap_ci_95"][1],
                "p_val_vs_base": mr_vs_base["p_value"],
                "margin_delta_vs_global": float(np.mean(most_recent_margins - global_margins)),
                "p_val_vs_global": float(stats.ttest_rel(most_recent_margins, global_margins)[1]),
                "cohens_dz_vs_global": float(np.mean(most_recent_margins - global_margins) / (np.std(most_recent_margins - global_margins, ddof=1) + 1e-12))
            },
            {
                "policy": "random_tokenwise_mean",
                "mean_margin": float(np.mean([r["mean_diff"] + np.mean(base_margins) for r in rand_vs_base_list])),
                "std_margin": float(np.mean([np.std(random_margins_all[s], ddof=1) for s in self.random_seeds])),
                "top1_acc": float(np.mean([r["acc_a"] for r in rand_vs_base_list])),
                "margin_delta_vs_base": float(np.mean([r["mean_diff"] for r in rand_vs_base_list])),
                "ci95_vs_base_lower": float(np.mean([r["bootstrap_ci_95"][0] for r in rand_vs_base_list])),
                "ci95_vs_base_upper": float(np.mean([r["bootstrap_ci_95"][1] for r in rand_vs_base_list])),
                "p_val_vs_base": float(np.mean([r["p_value"] for r in rand_vs_base_list])),
                "margin_delta_vs_global": float(np.mean([np.mean(np.array(random_margins_all[s]) - global_margins) for s in self.random_seeds])),
                "p_val_vs_global": float(np.mean([stats.ttest_rel(random_margins_all[s], global_margins)[1] for s in self.random_seeds])),
                "cohens_dz_vs_global": float(np.mean([
                    np.mean(np.array(random_margins_all[s]) - global_margins) / (np.std(np.array(random_margins_all[s]) - global_margins, ddof=1) + 1e-12)
                    for s in self.random_seeds
                ]))
            },
            {
                "policy": "global_oracle",
                "mean_margin": float(np.mean(global_margins)),
                "std_margin": float(np.std(global_margins, ddof=1)),
                "top1_acc": float(np.mean(global_correct)),
                "margin_delta_vs_base": global_vs_base["mean_diff"],
                "ci95_vs_base_lower": global_vs_base["bootstrap_ci_95"][0],
                "ci95_vs_base_upper": global_vs_base["bootstrap_ci_95"][1],
                "p_val_vs_base": global_vs_base["p_value"],
                "margin_delta_vs_global": 0.0,
                "p_val_vs_global": 1.0,
                "cohens_dz_vs_global": 0.0
            },
            {
                "policy": "tokenwise_oracle",
                "mean_margin": float(np.mean(tokenwise_margins)),
                "std_margin": float(np.std(tokenwise_margins, ddof=1)),
                "top1_acc": float(np.mean(tokenwise_correct)),
                "margin_delta_vs_base": token_vs_base["mean_diff"],
                "ci95_vs_base_lower": token_vs_base["bootstrap_ci_95"][0],
                "ci95_vs_base_upper": token_vs_base["bootstrap_ci_95"][1],
                "p_val_vs_base": token_vs_base["p_value"],
                "margin_delta_vs_global": token_vs_global["mean_diff"],
                "ci95_vs_global_lower": token_vs_global["bootstrap_ci_95"][0],
                "ci95_vs_global_upper": token_vs_global["bootstrap_ci_95"][1],
                "p_val_vs_global": token_vs_global["p_value"],
                "cohens_dz_vs_global": token_vs_global["cohens_dz"]
            }
        ]
        comparison_df = pd.DataFrame(comparison_rows)

        # Sensitivity comparisons
        sensitivity_summaries = {}
        if run_sensitivities:
            for sens_gamma, s_data in sensitivity_results.items():
                s_g_margins = np.array(s_data["global_margins"])
                s_t_margins = np.array(s_data["tokenwise_margins"])
                s_g_corr = np.array(s_data["global_correct"], dtype=int)
                s_t_corr = np.array(s_data["tokenwise_correct"], dtype=int)
                s_cmp = evaluate_paired_comparison(
                    s_t_margins, s_g_margins, s_t_corr, s_g_corr,
                    label_a="Tokenwise", label_b="Global"
                )
                sensitivity_summaries[f"gamma_{sens_gamma}"] = s_cmp

        print(f"[{self.model_name}] Completed in {runtime_s:.2f}s. Peak VRAM: {peak_vram_mb:.1f} MB.")
        print(f"[{self.model_name}] Tokenwise vs Global: mean Delta_m = {token_vs_global['mean_diff']:+.4f} "
              f"(95% CI [{token_vs_global['bootstrap_ci_95'][0]:.4f}, {token_vs_global['bootstrap_ci_95'][1]:.4f}], "
              f"p = {token_vs_global['p_value']:.4e}, dz = {token_vs_global['cohens_dz']:.3f})")

        return {
            "model_name": self.model_name,
            "runtime_seconds": runtime_s,
            "peak_vram_mb": peak_vram_mb,
            "image_df": image_df,
            "source_df": source_df,
            "comparison_df": comparison_df,
            "token_vs_global": token_vs_global,
            "global_vs_base": global_vs_base,
            "token_vs_base": token_vs_base,
            "spatial_summary": spatial_summary,
            "diversity_records": diversity_records,
            "sensitivity_summaries": sensitivity_summaries
        }
