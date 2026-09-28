import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar

from rescancel.instrumentation import InstrumentedViT, InterventionConfig
from rescancel.metrics import compute_prediction_metrics
from rescancel.corruptions import apply_gaussian_noise, apply_gaussian_blur


def run_intervention_sweep(
    inst_vit: InstrumentedViT,
    loader: torch.utils.data.DataLoader,
    target_layer: int = 0,
    target_site: str = "attn",
    target_token: str = "cls",
    alphas: List[float] = [1.0, 0.75, 0.50, 0.25],
    random_seeds: List[int] = [2501, 2502, 2503],
    device: str = "cuda",
    seed: int = 42,
    n_bootstrap: int = 10000
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the corrected V0.1 pre-registered causal intervention sweep:
      - Intervenes ONLY on pre-registered extreme cancellation events (only_extreme=True).
      - Primary site: Layer 0, Attention, CLS token.
      - Compares weaken_opposing vs untouched baseline, uniform scaling, and multi-seed random direction controls.
      - Performs image-level paired statistical tests (McNemar, bootstrap CIs).
    """
    model = inst_vit.model
    model.eval()

    inst_vit.disable_intervention()
    inst_vit.set_logging(False)

    all_inputs = []
    all_targets = []
    baseline_clean_preds = []
    baseline_clean_margins = []
    baseline_pert_preds = []

    # 1. Baseline Run (alpha = 1.0, untouched model)
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            all_inputs.append(batch_x)
            all_targets.append(batch_y)

            out_clean = model(batch_x)
            m_clean = compute_prediction_metrics(out_clean, batch_y)
            baseline_clean_preds.append(m_clean["top1_pred"].detach().cpu())
            baseline_clean_margins.append(m_clean["margin"].detach().cpu())

            # Perturbed input (Gaussian noise sigma=0.08)
            x_pert = apply_gaussian_noise(batch_x, sigma=0.08, seed=seed)
            out_pert = model(x_pert)
            m_pert = compute_prediction_metrics(out_pert, batch_y)
            baseline_pert_preds.append(m_pert["top1_pred"].detach().cpu())

    baseline_clean_preds = torch.cat(baseline_clean_preds).numpy()
    baseline_clean_margins = torch.cat(baseline_clean_margins).numpy()
    baseline_pert_preds = torch.cat(baseline_pert_preds).numpy()
    targets_np = torch.cat([t.cpu() for t in all_targets]).numpy()

    n_images = len(targets_np)
    base_correct = (baseline_clean_preds == targets_np).astype(int)
    base_flipped = (baseline_clean_preds != baseline_pert_preds).astype(int)

    base_clean_acc = float(np.mean(base_correct))
    base_flip_rate = float(np.mean(base_flipped))
    base_mean_margin = float(np.mean(baseline_clean_margins))

    results = []
    # Baseline row
    results.append({
        "target_layer": target_layer,
        "target_site": target_site,
        "mode": "baseline",
        "alpha": 1.0,
        "seed": seed,
        "clean_acc": base_clean_acc,
        "flip_rate": base_flip_rate,
        "mean_margin": base_mean_margin,
        "margin_delta": 0.0,
        "margin_ci_lower": 0.0,
        "margin_ci_upper": 0.0,
        "margin_p_val": 1.0,
        "acc_delta": 0.0,
        "acc_mcnemar_p": 1.0,
        "flip_delta": 0.0,
        "flip_mcnemar_p": 1.0,
        "stabilized_frac": 0.0,
        "degraded_frac": 0.0,
        "intervened_event_count": 0
    })

    # Helper function for forward evaluation under a specific intervention config
    def evaluate_config(config: InterventionConfig) -> Dict[str, Any]:
        inst_vit.set_intervention(config)
        # Reset block tracking
        for b in inst_vit.instrumented_blocks:
            b.total_intervention_events = 0

        clean_preds = []
        clean_margins = []
        pert_preds = []

        with torch.no_grad():
            for bx, by in zip(all_inputs, all_targets):
                out_clean = model(bx)
                mc = compute_prediction_metrics(out_clean, by)
                clean_preds.append(mc["top1_pred"].detach().cpu())
                clean_margins.append(mc["margin"].detach().cpu())

                xp = apply_gaussian_noise(bx, sigma=0.08, seed=seed)
                out_p = model(xp)
                mp = compute_prediction_metrics(out_p, by)
                pert_preds.append(mp["top1_pred"].detach().cpu())

        clean_preds = torch.cat(clean_preds).numpy()
        clean_margins = torch.cat(clean_margins).numpy()
        pert_preds = torch.cat(pert_preds).numpy()

        correct = (clean_preds == targets_np).astype(int)
        flipped = (clean_preds != pert_preds).astype(int)

        # Count total intervention events
        intervened_count = sum(b.total_intervention_events for b in inst_vit.instrumented_blocks)

        # Image-level paired margin difference
        margin_diffs = clean_margins - baseline_clean_margins
        mean_margin_delta = float(np.mean(margin_diffs))

        # Bootstrap 95% CI on paired margin delta
        rng = np.random.RandomState(seed)
        boot_means = rng.choice(margin_diffs, size=(n_bootstrap, n_images), replace=True).mean(axis=1)
        m_ci_lower = float(np.percentile(boot_means, 2.5))
        m_ci_upper = float(np.percentile(boot_means, 97.5))
        m_t_stat, m_p_val = stats.ttest_1samp(margin_diffs, 0.0)

        # Image-level paired accuracy change
        acc_delta = float(np.mean(correct) - base_clean_acc)
        # McNemar on clean correctness
        n11_acc = int(((correct == 1) & (base_correct == 1)).sum())
        n10_acc = int(((correct == 1) & (base_correct == 0)).sum()) # became correct
        n01_acc = int(((correct == 0) & (base_correct == 1)).sum()) # degraded
        n00_acc = int(((correct == 0) & (base_correct == 0)).sum())
        if n10_acc + n01_acc == 0:
            acc_mcnemar_p = 1.0
        elif n10_acc + n01_acc < 25:
            acc_mcnemar_p = float(stats.binomtest(min(n10_acc, n01_acc), n10_acc + n01_acc, 0.5).pvalue)
        else:
            acc_mcnemar_p = float(mcnemar([[n11_acc, n10_acc], [n01_acc, n00_acc]], exact=False, correction=True).pvalue)

        # Image-level paired flip rate change
        flip_delta = float(np.mean(flipped) - base_flip_rate)
        # McNemar on flip status
        n11_flip = int(((flipped == 1) & (base_flipped == 1)).sum())
        n10_flip = int(((flipped == 1) & (base_flipped == 0)).sum()) # newly flipped
        n01_flip = int(((flipped == 0) & (base_flipped == 1)).sum()) # stabilized
        n00_flip = int(((flipped == 0) & (base_flipped == 0)).sum())
        if n10_flip + n01_flip == 0:
            flip_mcnemar_p = 1.0
        elif n10_flip + n01_flip < 25:
            flip_mcnemar_p = float(stats.binomtest(min(n10_flip, n01_flip), n10_flip + n01_flip, 0.5).pvalue)
        else:
            flip_mcnemar_p = float(mcnemar([[n11_flip, n10_flip], [n01_flip, n00_flip]], exact=False, correction=True).pvalue)

        # Stabilization: baseline flipped, intervention not flipped
        orig_flipped = (base_flipped == 1)
        stabilized_frac = float(n01_flip / max(orig_flipped.sum(), 1))

        # Degradation: baseline correct, intervention incorrect
        degraded_frac = float(n01_acc / max(base_correct.sum(), 1))

        return {
            "clean_preds": clean_preds,
            "clean_margins": clean_margins,
            "pert_preds": pert_preds,
            "clean_acc": float(np.mean(correct)),
            "flip_rate": float(np.mean(flipped)),
            "mean_margin": float(np.mean(clean_margins)),
            "margin_delta": mean_margin_delta,
            "margin_ci_lower": m_ci_lower,
            "margin_ci_upper": m_ci_upper,
            "margin_p_val": float(m_p_val),
            "acc_delta": acc_delta,
            "acc_mcnemar_p": acc_mcnemar_p,
            "flip_delta": flip_delta,
            "flip_mcnemar_p": flip_mcnemar_p,
            "stabilized_frac": stabilized_frac,
            "degraded_frac": degraded_frac,
            "intervened_event_count": intervened_count
        }

    # 2. Sweep across alphas < 1.0
    active_alphas = [a for a in alphas if a < 1.0]

    for alpha in active_alphas:
        # A. Weaken Opposing (Primary Causal Intervention, only_extreme=True)
        config_weaken = InterventionConfig(
            active=True,
            target_layer=target_layer,
            target_site=target_site,
            target_token=target_token,
            mode="weaken_opposing",
            alpha=alpha,
            seed=seed,
            only_extreme=True
        )
        res_weaken = evaluate_config(config_weaken)
        row_weaken = {
            "target_layer": target_layer,
            "target_site": target_site,
            "mode": "weaken_opposing",
            "alpha": alpha,
            "seed": seed,
            **{k: v for k, v in res_weaken.items() if not k.endswith("preds") and not k.endswith("margins")}
        }
        results.append(row_weaken)

        # B. Random Direction Controls (Across multi-seeds: 2501, 2502, 2503)
        rand_runs = []
        for r_seed in random_seeds:
            config_rand = InterventionConfig(
                active=True,
                target_layer=target_layer,
                target_site=target_site,
                target_token=target_token,
                mode="random_direction",
                alpha=alpha,
                seed=r_seed,
                only_extreme=True
            )
            res_rand = evaluate_config(config_rand)
            rand_runs.append(res_rand)
            results.append({
                "target_layer": target_layer,
                "target_site": target_site,
                "mode": f"random_direction_seed_{r_seed}",
                "alpha": alpha,
                "seed": r_seed,
                **{k: v for k, v in res_rand.items() if not k.endswith("preds") and not k.endswith("margins")}
            })

        # Summary of random direction controls across seeds
        results.append({
            "target_layer": target_layer,
            "target_site": target_site,
            "mode": "random_direction_mean",
            "alpha": alpha,
            "seed": -1,
            "clean_acc": float(np.mean([r["clean_acc"] for r in rand_runs])),
            "flip_rate": float(np.mean([r["flip_rate"] for r in rand_runs])),
            "mean_margin": float(np.mean([r["mean_margin"] for r in rand_runs])),
            "margin_delta": float(np.mean([r["margin_delta"] for r in rand_runs])),
            "margin_ci_lower": float(np.mean([r["margin_ci_lower"] for r in rand_runs])),
            "margin_ci_upper": float(np.mean([r["margin_ci_upper"] for r in rand_runs])),
            "margin_p_val": float(np.mean([r["margin_p_val"] for r in rand_runs])),
            "acc_delta": float(np.mean([r["acc_delta"] for r in rand_runs])),
            "acc_mcnemar_p": float(np.mean([r["acc_mcnemar_p"] for r in rand_runs])),
            "flip_delta": float(np.mean([r["flip_delta"] for r in rand_runs])),
            "flip_mcnemar_p": float(np.mean([r["flip_mcnemar_p"] for r in rand_runs])),
            "stabilized_frac": float(np.mean([r["stabilized_frac"] for r in rand_runs])),
            "degraded_frac": float(np.mean([r["degraded_frac"] for r in rand_runs])),
            "intervened_event_count": rand_runs[0]["intervened_event_count"]
        })

        # C. Uniform Scaling Control (Matched L2 magnitude change on same events)
        config_uniform = InterventionConfig(
            active=True,
            target_layer=target_layer,
            target_site=target_site,
            target_token=target_token,
            mode="uniform_scaling",
            alpha=alpha,
            seed=seed,
            only_extreme=True
        )
        res_uniform = evaluate_config(config_uniform)
        results.append({
            "target_layer": target_layer,
            "target_site": target_site,
            "mode": "uniform_scaling",
            "alpha": alpha,
            "seed": seed,
            **{k: v for k, v in res_uniform.items() if not k.endswith("preds") and not k.endswith("margins")}
        })

    inst_vit.disable_intervention()
    df_results = pd.DataFrame(results)

    summary_stats = {
        "target_layer": target_layer,
        "target_site": target_site,
        "target_token": target_token,
        "baseline_clean_acc": base_clean_acc,
        "baseline_flip_rate": base_flip_rate,
        "baseline_mean_margin": base_mean_margin,
        "random_seeds": random_seeds,
        "n_images": n_images
    }
    return df_results, summary_stats
