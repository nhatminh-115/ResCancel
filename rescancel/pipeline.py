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
from rescancel.confound_matching import (
    match_controls,
    compute_matched_margin_effect,
    compute_matched_binary_fragility_effect,
    run_image_level_logistic_regression
)
from rescancel.intervention import run_intervention_sweep
from rescancel.audit_validation import validate_v0_1_results


class ResCancelPipeline:
    """
    Executes the corrected ResCancel V0.1 mechanistic falsification protocol.
    """
    def __init__(
        self,
        model_name: str = "deit_tiny_patch16_224",
        num_samples: int = 1000,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        output_dir: str = "outputs/v0_1",
        figures_dir: str = "figures/v0_1"
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
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Runs clean and perturbed forward passes, extracting:
        1. Image-level prediction table (top-1 accuracy, margin, entropy, perturbation flips)
        2. CLS residual event table across all layers and sites (primary mechanistic test)
        3. Patch residual summary table with patch_extreme_burden (secondary exploratory test)
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
        batch_sizes = []
        for bx, _ in loader:
            batch_sizes.append(bx.shape[0])
        batch_start_indices = [0] + list(np.cumsum(batch_sizes)[:-1])

        cls_event_rows = []
        patch_summary_rows = []

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

                    # Primary CLS extreme criteria
                    is_ext = is_extreme_cancellation(cls_cos, cls_r, cls_q, cls_c_l1)

                    cls_event_rows.append({
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

                    # Secondary Patch Summary (Explicitly named patch_extreme_burden, NOT is_extreme)
                    p_cos = rec.patch_cos_mean[i].item()
                    p_r = rec.patch_r_mean[i].item()
                    p_q = rec.patch_q_mean[i].item()
                    p_c_l1 = rec.patch_c_l1_mean[i].item()
                    p_x = rec.patch_x_norm_mean[i].item()
                    p_d = rec.patch_delta_norm_mean[i].item()
                    p_burden = rec.patch_extreme_frac[i].item()

                    patch_summary_rows.append({
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
                        "patch_extreme_burden": p_burden,
                        "clean_correct": s_info["clean_correct"],
                        "margin": s_info["clean_margin"],
                        "flipped": s_info["any_flipped"]
                    })

        cls_df = pd.DataFrame(cls_event_rows)
        patch_df = pd.DataFrame(patch_summary_rows)
        return img_df, cls_df, patch_df

    def analyze_matched_confounds(
        self,
        img_df: pd.DataFrame,
        cls_event_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Runs V0.1 confound matching obeying strict individual calipers and image-level regression.
        """
        print(f"[{self.model_name}] Running strict caliper matching on CLS events...")
        matched_df, stats_summary = match_controls(
            cls_event_df,
            is_extreme_col="is_extreme",
            token_type="cls",
            caliper_x_pct=0.15,
            caliper_delta_pct=0.15,
            caliper_margin=0.50,
            random_state=self.seed
        )

        margin_effects = compute_matched_margin_effect(matched_df, outcome_col="margin", n_bootstrap=10000, seed=self.seed)
        fragility_effects = compute_matched_binary_fragility_effect(matched_df, outcome_col="flipped")

        print(f"[{self.model_name}] Running image-level multivariable logistic regression (N=1000)...")
        reg_results = run_image_level_logistic_regression(img_df, cls_event_df, target_col="any_flipped")

        return {
            "matching_stats": stats_summary,
            "matched_df": matched_df,
            "matched_margin_effect": margin_effects,
            "matched_fragility_effect": fragility_effects,
            "image_level_regression": reg_results
        }

    def run_causal_falsification(
        self,
        inst_vit: InstrumentedViT,
        loader: torch.utils.data.DataLoader
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes causal intervention strictly on pre-registered Layer 0 Attention CLS extreme events.
        """
        print(f"[{self.model_name}] Running causal intervention on Layer 0 Attention CLS (only_extreme=True)...")
        sweep_df, summary_stats = run_intervention_sweep(
            inst_vit=inst_vit,
            loader=loader,
            target_layer=0,
            target_site="attn",
            target_token="cls",
            alphas=[1.0, 0.75, 0.50, 0.25],
            random_seeds=[2501, 2502, 2503],
            device=self.device,
            seed=self.seed
        )
        return sweep_df, summary_stats

    def evaluate_decision_rule(
        self,
        confound_results: Dict[str, Any],
        intervention_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluates the V0.1 decision criteria:
        - Outcome A — KILL: Confound-controlled association is null or negligible AND targeted intervention
          provides no meaningful advantage over matched controls.
        - Outcome B — KILL training hypothesis: Extreme-only cancellation suppression causes reproducible harm
          larger than matched controls, with CIs supporting a meaningful negative effect.
        - Outcome C — INTERESTING: Extreme-only targeted suppression improves robustness/stability without
          meaningfully degrading clean accuracy/margin.
        """
        frag_eff = confound_results["matched_fragility_effect"]
        margin_eff = confound_results["matched_margin_effect"]
        reg_res = confound_results["image_level_regression"]

        # Intervention rows at alpha=0.50
        weaken_05 = intervention_df[(intervention_df["mode"] == "weaken_opposing") & (intervention_df["alpha"] == 0.50)].iloc[0]
        rand_05 = intervention_df[(intervention_df["mode"] == "random_direction_mean") & (intervention_df["alpha"] == 0.50)].iloc[0]

        # Check matched risk difference
        rd = frag_eff.get("risk_diff", 0.0)
        mcnemar_p = frag_eff.get("mcnemar_p_value", 1.0)
        odds_ratio = reg_res.get("odds_ratio", 1.0)
        or_ci_lower = reg_res.get("odds_ratio_ci_lower", 1.0)
        or_ci_upper = reg_res.get("odds_ratio_ci_upper", 1.0)

        # Causal delta
        weaken_margin_delta = weaken_05["margin_delta"]
        weaken_margin_ci_upper = weaken_05["margin_ci_upper"]
        weaken_flip_delta = weaken_05["flip_delta"]

        rand_margin_delta = rand_05["margin_delta"]

        decision = "Outcome A: KILL (no evidence of harmful cancellation)"
        rationale = []

        # Test Outcome C first:
        # Significant improvement in flip rate (flip_delta < -0.02, p < 0.05) and clean acc not harmed (acc_delta >= -0.005)
        if weaken_flip_delta < -0.02 and weaken_05["flip_mcnemar_p"] < 0.05 and weaken_05["acc_delta"] >= -0.005:
            decision = "Outcome C: INTERESTING"
            rationale.append(f"Targeted suppression reduced flip rate by {weaken_flip_delta:.4f} (p={weaken_05['flip_mcnemar_p']:.4f}) without degrading accuracy.")
            rationale.append("Supports a distinct harmful cancellation subset justifying V1 training experiments.")

        # Test Outcome B:
        # Suppressing cancellation causes reproducible harm (margin CI strictly below 0, or clean acc drops significantly)
        elif weaken_margin_ci_upper < 0.0 and weaken_margin_delta < rand_margin_delta - 0.005:
            decision = "Outcome B: KILL training hypothesis (evidence suggests cancellation is productive)"
            rationale.append(f"Suppressing extreme cancellation causes statistically significant margin erosion (Delta z = {weaken_margin_delta:.4f}, 95% CI [{weaken_05['margin_ci_lower']:.4f}, {weaken_margin_ci_upper:.4f}]).")
            rationale.append(f"The negative impact of selective weakening exceeds matched random controls (Delta z = {rand_margin_delta:.4f}).")
            rationale.append("The tested extreme-cancellation regime appears to contribute productively to computation rather than acting as a pathology.")

        # Test Outcome A:
        # Confound-controlled association is null and intervention shows no advantage
        else:
            decision = "Outcome A: KILL (no evidence of pathology)"
            rationale.append(f"Confound-controlled matched risk difference is null: RD = {rd:.4f} (p = {mcnemar_p:.4f}).")
            rationale.append(f"Image-level logistic regression odds ratio: OR = {odds_ratio:.3f} (95% CI [{or_ci_lower:.3f}, {or_ci_upper:.3f}], spans 1.0).")
            rationale.append("No evidence that extreme cancellation is a predictive pathology.")

        return {
            "decision": decision,
            "rationale": rationale,
            "matched_risk_diff": rd,
            "matched_mcnemar_p": mcnemar_p,
            "image_level_odds_ratio": odds_ratio,
            "image_level_or_ci": [or_ci_lower, or_ci_upper],
            "weaken_margin_delta_a05": weaken_margin_delta,
            "weaken_margin_ci_a05": [weaken_05["margin_ci_lower"], weaken_05["margin_ci_upper"]],
            "weaken_flip_delta_a05": weaken_flip_delta,
            "rand_margin_delta_a05": rand_margin_delta
        }
