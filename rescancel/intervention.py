import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from rescancel.instrumentation import InstrumentedViT, InterventionConfig
from rescancel.metrics import compute_prediction_metrics
from rescancel.corruptions import apply_gaussian_noise, apply_gaussian_blur


def run_intervention_sweep(
    inst_vit: InstrumentedViT,
    loader: torch.utils.data.DataLoader,
    target_layer: int,
    target_site: str,
    target_token: str = "cls",
    alphas: List[float] = [1.0, 0.75, 0.50, 0.25],
    device: str = "cuda",
    seed: int = 42
) -> pd.DataFrame:
    """
    Executes a pre-registered causal intervention sweep across alpha values
    and matched controls (random direction, uniform scaling).
    
    Evaluates:
      - Clean top-1 accuracy
      - Clean logit margin (top-1 vs top-2)
      - Perturbed accuracy (under mild Gaussian noise)
      - Perturbation flip rate
      - Fragile stabilization rate & degradation rate
    """
    model = inst_vit.model
    model.eval()
    
    modes = ["weaken_opposing", "random_direction", "uniform_scaling"]
    results = []

    # First collect baseline predictions (alpha=1.0)
    inst_vit.disable_intervention()
    inst_vit.set_logging(False)

    baseline_clean_preds = []
    baseline_clean_margins = []
    baseline_pert_preds = []
    all_targets = []
    all_inputs = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            all_targets.append(batch_y)
            all_inputs.append(batch_x)

            # Clean forward
            out_clean = model(batch_x)
            m_clean = compute_prediction_metrics(out_clean, batch_y)
            baseline_clean_preds.append(m_clean["top1_pred"])
            baseline_clean_margins.append(m_clean["margin"])

            # Perturbed forward (mild Gaussian noise)
            x_pert = apply_gaussian_noise(batch_x, sigma=0.08, seed=seed)
            out_pert = model(x_pert)
            m_pert = compute_prediction_metrics(out_pert, batch_y)
            baseline_pert_preds.append(m_pert["top1_pred"])

    baseline_clean_preds = torch.cat(baseline_clean_preds)
    baseline_clean_margins = torch.cat(baseline_clean_margins)
    baseline_pert_preds = torch.cat(baseline_pert_preds)
    all_targets = torch.cat(all_targets)
    
    base_acc = (baseline_clean_preds == all_targets).float().mean().item()
    base_pert_acc = (baseline_pert_preds == all_targets).float().mean().item()
    base_flip_rate = (baseline_clean_preds != baseline_pert_preds).float().mean().item()
    base_margin_mean = baseline_clean_margins.mean().item()

    # Log baseline entry (alpha = 1.0)
    results.append({
        "target_layer": target_layer,
        "target_site": target_site,
        "mode": "baseline",
        "alpha": 1.0,
        "clean_acc": base_acc,
        "pert_acc": base_pert_acc,
        "flip_rate": base_flip_rate,
        "mean_margin": base_margin_mean,
        "margin_delta": 0.0,
        "stabilized_frac": 0.0,
        "degraded_frac": 0.0,
    })

    # Sweep across modes and alpha < 1.0
    for mode in modes:
        for alpha in [a for a in alphas if a < 1.0]:
            config = InterventionConfig(
                active=True,
                target_layer=target_layer,
                target_site=target_site,
                target_token=target_token,
                mode=mode,
                alpha=alpha,
                seed=seed,
                only_extreme=False
            )
            inst_vit.set_intervention(config)

            int_clean_preds = []
            int_clean_margins = []
            int_pert_preds = []

            with torch.no_grad():
                for batch_x, batch_y in zip(all_inputs, all_targets.split(loader.batch_size)):
                    out_clean = model(batch_x)
                    m_clean = compute_prediction_metrics(out_clean, batch_y)
                    int_clean_preds.append(m_clean["top1_pred"])
                    int_clean_margins.append(m_clean["margin"])

                    # Perturbed input with same intervention active
                    x_pert = apply_gaussian_noise(batch_x, sigma=0.08, seed=seed)
                    out_pert = model(x_pert)
                    m_pert = compute_prediction_metrics(out_pert, batch_y)
                    int_pert_preds.append(m_pert["top1_pred"])

            int_clean_preds = torch.cat(int_clean_preds)
            int_clean_margins = torch.cat(int_clean_margins)
            int_pert_preds = torch.cat(int_pert_preds)

            clean_acc = (int_clean_preds == all_targets).float().mean().item()
            pert_acc = (int_pert_preds == all_targets).float().mean().item()
            flip_rate = (int_clean_preds != int_pert_preds).float().mean().item()
            mean_margin = int_clean_margins.mean().item()
            margin_delta = mean_margin - base_margin_mean

            # Stabilization: originally flipped under perturbation, now stable
            orig_flipped = (baseline_clean_preds != baseline_pert_preds)
            now_stable = (int_clean_preds == int_pert_preds)
            stabilized_frac = (orig_flipped & now_stable).float().sum().item() / max(orig_flipped.float().sum().item(), 1)

            # Degradation: originally correct on clean, now incorrect
            orig_correct = (baseline_clean_preds == all_targets)
            now_incorrect = (int_clean_preds != all_targets)
            degraded_frac = (orig_correct & now_incorrect).float().sum().item() / max(orig_correct.float().sum().item(), 1)

            results.append({
                "target_layer": target_layer,
                "target_site": target_site,
                "mode": mode,
                "alpha": alpha,
                "clean_acc": clean_acc,
                "pert_acc": pert_acc,
                "flip_rate": flip_rate,
                "mean_margin": mean_margin,
                "margin_delta": margin_delta,
                "stabilized_frac": stabilized_frac,
                "degraded_frac": degraded_frac,
            })

    # Disable intervention before returning
    inst_vit.disable_intervention()
    return pd.DataFrame(results)
