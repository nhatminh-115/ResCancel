import os
import time
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

import timm
from rescancel.instrumentation import InstrumentedViT, InterventionConfig
from rescancel.metrics import compute_prediction_metrics, is_extreme_cancellation
from rescancel.dataset import get_imagenet_val_loader
from rescancel.corruptions import apply_gaussian_noise, apply_gaussian_blur
from rescancel.confound_matching import match_controls, compute_matched_effect_sizes, run_controlled_logistic_regression
from rescancel.intervention import run_intervention_sweep


class ResCancelPipeline:
    """
    Executes the frozen ResCancel V0 mechanistic falsification protocol.
    """
    def __init__(
        self,
        model_name: str = "deit_tiny_patch16_224",
        num_samples: int = 1000,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        output_dir: str = "outputs",
        figures_dir: str = "figures"
    ):
        self.model_name = model_name
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.output_dir = output_dir
        self.figures_dir = figures_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        
        # Reproducibility seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_model(self) -> Tuple[InstrumentedViT, Any]:
        """Loads and instruments the pretrained ViT model."""
        raw_model = timm.create_model(self.model_name, pretrained=True)
        raw_model.to(self.device)
        raw_model.eval()
        inst_vit = InstrumentedViT(raw_model)
        return inst_vit, raw_model

    def run_clean_and_perturbed_analysis(
        self,
        inst_vit: InstrumentedViT,
        loader: torch.utils.data.DataLoader
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs clean and perturbed forward passes, extracting:
        1. Image-level prediction table (top-1 accuracy, margin, entropy, perturbation flips)
        2. Event-level residual cancellation table across all layers, sites, and tokens.
        """
        model = inst_vit.model
        inst_vit.set_logging(True)
        inst_vit.clear_records()
        inst_vit.disable_intervention()

        image_records = []
        sample_idx_offset = 0

        print(f"[{self.model_name}] Running clean and perturbed forward passes on {self.num_samples} samples...")
        t0 = time.time()

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_size = batch_x.shape[0]

            # 1. Clean forward (records residual additions)
            with torch.no_grad():
                out_clean = model(batch_x)
                clean_metrics = compute_prediction_metrics(out_clean, batch_y)

                # 2. Perturbed forward (Gaussian noise)
                inst_vit.set_logging(False) # Don't log perturbations to avoid double recording
                x_noise = apply_gaussian_noise(batch_x, sigma=0.08, seed=self.seed)
                out_noise = model(x_noise)
                noise_metrics = compute_prediction_metrics(out_noise, batch_y)

                # 3. Perturbed forward (Gaussian blur)
                x_blur = apply_gaussian_blur(batch_x, kernel_size=5, sigma=1.0)
                out_blur = model(x_blur)
                blur_metrics = compute_prediction_metrics(out_blur, batch_y)
                inst_vit.set_logging(True)

            # Store image-level results
            for b in range(batch_size):
                s_id = sample_idx_offset + b
                clean_pred = clean_metrics["top1_pred"][b].item()
                target = batch_y[b].item()
                noise_pred = noise_metrics["top1_pred"][b].item()
                blur_pred = blur_metrics["top1_pred"][b].item()

                image_records.append({
                    "sample_id": s_id,
                    "target": target,
                    "clean_top1_pred": clean_pred,
                    "clean_correct": int(clean_pred == target),
                    "clean_prob": clean_metrics["top1_prob"][b].item(),
                    "clean_margin": clean_metrics["margin"][b].item(),
                    "clean_entropy": clean_metrics["entropy"][b].item(),
                    "noise_flipped": int(clean_pred != noise_pred),
                    "blur_flipped": int(clean_pred != blur_pred),
                    "any_flipped": int((clean_pred != noise_pred) or (clean_pred != blur_pred))
                })
            sample_idx_offset += batch_size

        img_df = pd.DataFrame(image_records)
        print(f"Completed forward passes in {time.time() - t0:.2f}s. Clean Acc: {img_df['clean_correct'].mean():.4f}")

        # Assemble residual event records
        # Each instrumented block has a list of records:
        # Call 2k: batch k, site "attn"
        # Call 2k+1: batch k, site "mlp"
        batch_sizes = []
        for bx, _ in loader:
            batch_sizes.append(bx.shape[0])
        batch_start_indices = [0] + list(np.cumsum(batch_sizes)[:-1])

        event_rows = []
        for inst_b in inst_vit.instrumented_blocks:
            for c_idx, rec in enumerate(inst_b.records):
                b_idx = c_idx // 2
                start_s_id = batch_start_indices[b_idx]
                batch_len = rec.cls_cos.shape[0]

                for i in range(batch_len):
                    s_id = start_s_id + i
                    s_info = img_df.loc[s_id]

                    cls_cos = rec.cls_cos[i].item()
                    cls_r = rec.cls_r[i].item()
                    cls_q = rec.cls_q[i].item()
                    cls_c_l1 = rec.cls_c_l1[i].item()
                    cls_x = rec.cls_x_norm[i].item()
                    cls_d = rec.cls_delta_norm[i].item()

                    is_ext = is_extreme_cancellation(cls_cos, cls_r, cls_q, cls_c_l1)

                    event_rows.append({
                        "sample_id": s_id,
                        "layer": rec.layer,
                        "site": rec.site,
                        "token_type": "cls",
                        "cos_sim": cls_cos,
                        "r_mag": cls_r,
                        "q_contract": cls_q,
                        "c_l1": cls_c_l1,
                        "x_norm": cls_x,
                        "delta_norm": cls_d,
                        "is_extreme": is_ext,
                        "clean_correct": s_info["clean_correct"],
                        "margin": s_info["clean_margin"],
                        "flipped": s_info["any_flipped"]
                    })

                    # Patch summary event
                    p_cos = rec.patch_cos_mean[i].item()
                    p_r = rec.patch_r_mean[i].item()
                    p_q = rec.patch_q_mean[i].item()
                    p_c_l1 = rec.patch_c_l1_mean[i].item()
                    p_x = rec.patch_x_norm_mean[i].item()
                    p_d = rec.patch_delta_norm_mean[i].item()
                    p_ext_frac = rec.patch_extreme_frac[i].item()

                    event_rows.append({
                        "sample_id": s_id,
                        "layer": rec.layer,
                        "site": rec.site,
                        "token_type": "patch",
                        "cos_sim": p_cos,
                        "r_mag": p_r,
                        "q_contract": p_q,
                        "c_l1": p_c_l1,
                        "x_norm": p_x,
                        "delta_norm": p_d,
                        "is_extreme": p_ext_frac > 0.05,
                        "patch_extreme_frac": p_ext_frac,
                        "clean_correct": s_info["clean_correct"],
                        "margin": s_info["clean_margin"],
                        "flipped": s_info["any_flipped"]
                    })

        event_df = pd.DataFrame(event_rows)
        return img_df, event_df

    def analyze_matched_confounds(self, event_df: pd.DataFrame) -> Dict[str, Any]:
        """Runs matched pair analysis and controlled regression."""
        print(f"[{self.model_name}] Running confound matching and regression...")
        matched_df, stats_summary = match_controls(event_df, is_extreme_col="is_extreme")
        
        effect_sizes_margin = compute_matched_effect_sizes(matched_df, outcome_col="margin")
        effect_sizes_flip = compute_matched_effect_sizes(matched_df, outcome_col="flipped")
        
        reg_results = run_controlled_logistic_regression(event_df, target_col="flipped", is_extreme_col="is_extreme")
        
        return {
            "matching_stats": stats_summary,
            "matched_margin_effect": effect_sizes_margin,
            "matched_flip_effect": effect_sizes_flip,
            "logistic_regression": reg_results,
            "matched_df": matched_df
        }

    def run_causal_falsification(
        self,
        inst_vit: InstrumentedViT,
        loader: torch.utils.data.DataLoader,
        event_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Identifies top cancellation site and runs causal intervention across alpha grid
        and matched controls.
        """
        # Find site with highest extreme cancellation rate
        extreme_events = event_df[event_df["is_extreme"] == True]
        if len(extreme_events) > 0:
            top_site_series = extreme_events.groupby(["layer", "site"]).size()
            top_layer, top_site = top_site_series.idxmax()
            print(f"[{self.model_name}] Top cancellation site: Layer {top_layer}, Site {top_site} ({top_site_series.max()} events)")
        else:
            # Fallback to middle layer attention
            top_layer, top_site = 6, "attn"

        sweep_df = run_intervention_sweep(
            inst_vit=inst_vit,
            loader=loader,
            target_layer=top_layer,
            target_site=top_site,
            target_token="cls",
            alphas=[1.0, 0.75, 0.50, 0.25],
            device=self.device,
            seed=self.seed
        )
        return sweep_df

    def evaluate_decision_rule(
        self,
        confound_results: Dict[str, Any],
        intervention_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluates frozen decision criteria:
        - Outcome A (KILL): Confound-controlled effect |d| < 0.10, p > 0.05, or intervention no better than random.
        - Outcome B (KILL Training Hypothesis): Targeted intervention consistently harms accuracy/margin or worsens fragility.
        - Outcome C (INTERESTING): Targeted intervention improves margin or stability significantly vs matched controls.
        """
        margin_effect = confound_results["matched_margin_effect"]
        flip_effect = confound_results["matched_flip_effect"]
        reg_results = confound_results["logistic_regression"]

        # Check intervention performance
        baseline_row = intervention_df[intervention_df["mode"] == "baseline"].iloc[0]
        weakened_rows = intervention_df[intervention_df["mode"] == "weaken_opposing"]
        random_rows = intervention_df[intervention_df["mode"] == "random_direction"]

        base_acc = baseline_row["clean_acc"]
        base_flip = baseline_row["flip_rate"]
        
        best_weaken = weakened_rows.sort_values(by="clean_acc", ascending=False).iloc[0] if len(weakened_rows) > 0 else baseline_row
        
        # Test if weakening consistently harms clean performance
        all_weakened_harm = (weakened_rows["clean_acc"] < base_acc - 0.005).all() if len(weakened_rows) > 0 else False
        all_weakened_worse_flip = (weakened_rows["flip_rate"] >= base_flip).all() if len(weakened_rows) > 0 else False

        # Controlled effect significance
        cohens_d = margin_effect.get("cohens_d", 0.0)
        p_val = margin_effect.get("p_value", 1.0)
        odds_ratio = reg_results.get("odds_ratio", 1.0)

        decision = "Outcome A: KILL"
        rationale = []

        if all_weakened_harm and all_weakened_worse_flip:
            decision = "Outcome B: KILL training hypothesis"
            rationale.append("Suppressing cancellation consistently degrades clean accuracy and fails to improve perturbation stability.")
            rationale.append("Residual cancellation is productive, required computation rather than a pathology.")
        elif abs(cohens_d) < 0.10 and p_val > 0.05:
            decision = "Outcome A: KILL"
            rationale.append(f"No significant confound-controlled effect (Cohen's d={cohens_d:.3f}, p={p_val:.4f}).")
            rationale.append("Cancellation is an epiphenomenon that explains no variance once ||x||, ||delta||, and margin are controlled.")
        elif best_weaken["flip_rate"] < base_flip - 0.02 and best_weaken["clean_acc"] >= base_acc - 0.01:
            decision = "Outcome C: INTERESTING"
            rationale.append(f"Selective weakening of opposing component reduced flip rate ({base_flip:.4f} -> {best_weaken['flip_rate']:.4f}) without destroying clean accuracy.")
            rationale.append("Supports existence of a distinct harmful cancellation regime.")
        else:
            decision = "Outcome B: KILL training hypothesis"
            rationale.append(f"Weakening opposing residual component leads to degradation (clean acc {base_acc:.4f} -> {best_weaken['clean_acc']:.4f}, flip rate {base_flip:.4f} -> {best_weaken['flip_rate']:.4f}).")
            rationale.append("Targeted intervention offers no causal benefit over the untouched model.")

        return {
            "decision": decision,
            "rationale": rationale,
            "cohens_d": cohens_d,
            "margin_p_val": p_val,
            "odds_ratio": odds_ratio,
            "baseline_clean_acc": base_acc,
            "best_weaken_clean_acc": float(best_weaken["clean_acc"]),
            "baseline_flip_rate": base_flip,
            "best_weaken_flip_rate": float(best_weaken["flip_rate"])
        }
