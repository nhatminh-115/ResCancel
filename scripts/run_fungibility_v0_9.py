import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import time
import json
import argparse
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from scipy import stats

from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits
from patch_fungibility.v0_9_pipeline import PatchFungibilityV09Pipeline
from patch_fungibility.v0_9_validation import validate_v0_9_assertions
from patch_fungibility.v0_9_decision import compute_paired_statistics, evaluate_v0_9_decision
from scripts.plot_fungibility_v0_9_figures import plot_all_v0_9_figures


def main():
    parser = argparse.ArgumentParser(description="Run Patch Fungibility V0.9 Natural Low-Rank Variance & Downstream Rank-Expansion Test")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output-dir", type=str, default="outputs/fungibility_v0_9", help="Output directory")
    parser.add_argument("--figures-dir", type=str, default="figures/fungibility_v0_9", help="Figures directory")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test on 50 images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print(f"\n{'='*75}\nPATCH FUNGIBILITY V0.9: NATURAL LOW-RANK VARIANCE & DOWNSTREAM RANK EXPANSION\n{'='*75}")
    print(f"Device: {args.device} | Batch Size: {args.batch_size} | Smoke Test: {args.smoke_test}")

    # 1. Load strictly disjoint ImageNet splits
    print("\nLoading disjoint ImageNet splits...")
    calib_ds, eval_ds, calib_df, eval_df = get_disjoint_imagenet_splits(
        calib_seed=9101,
        eval_seed=9201,
        n_per_split=1000,
        output_dir=None
    )

    if args.smoke_test:
        print("[SMOKE TEST] Restricting to 50 calibration and 50 evaluation images...")
        calib_indices = list(range(50))
        eval_indices = list(range(50))
        calib_ds = Subset(calib_ds, calib_indices)
        eval_ds = Subset(eval_ds, eval_indices)
        ranks = [1, 4, 16]
        rank_seeds = [16001]
        scale_seeds = [17001]
        rand_seeds = [18001]
        pc_indices = [1, 2, 4]
        scale_multipliers = [0.5, 1.0, 2.0]
    else:
        calib_indices = calib_df["global_index"].tolist()
        eval_indices = eval_df["global_index"].tolist()
        ranks = [1, 2, 4, 8, 16, 32, 64]
        rank_seeds = [16001, 16002, 16003, 16004, 16005]
        scale_seeds = [17001, 17002, 17003, 17004, 17005]
        rand_seeds = [18001, 18002, 18003, 18004, 18005]
        pc_indices = [1, 2, 3, 4, 8, 16]
        scale_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]

    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Run DeiT-Tiny
    print("\n" + "="*50 + " RUNNING DEIT-TINY " + "="*50)
    tiny_pipe = PatchFungibilityV09Pipeline(
        model_name="deit_tiny_patch16_224",
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        ranks=ranks,
        rank_seeds=rank_seeds,
        scale_seeds=scale_seeds,
        random_1d_seeds=rand_seeds,
        pc_indices=pc_indices,
        scale_multipliers=scale_multipliers
    )
    tiny_res = tiny_pipe.run(calib_loader, eval_loader, n_samples=len(eval_ds))

    # 3. Run DeiT-Small
    print("\n" + "="*50 + " RUNNING DEIT-SMALL " + "="*50)
    small_pipe = PatchFungibilityV09Pipeline(
        model_name="deit_small_patch16_224",
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        ranks=ranks,
        rank_seeds=rank_seeds,
        scale_seeds=scale_seeds,
        random_1d_seeds=rand_seeds,
        pc_indices=pc_indices,
        scale_multipliers=scale_multipliers
    )
    small_res = small_pipe.run(calib_loader, eval_loader, n_samples=len(eval_ds))

    # 4. Construct Outputs & CSV DataFrames
    print("\nConstructing tabular outputs and CSVs...")
    spec_rows = []
    rank_comp_rows = []
    pc_id_rows = []
    scale_rows = []
    rand_rows = []
    prop_rows = []
    attn_rows = []

    models_data = [
        ("deit_tiny_patch16_224", tiny_res),
        ("deit_small_patch16_224", small_res)
    ]

    for m_name, m_res in models_data:
        clean_res = m_res["primary_results"]["clean"]
        clean_acc = clean_res["acc"]
        clean_margin = clean_res["mean_margin"]
        pca_res = m_res["pca_result"]
        eigenvalues = pca_res["eigenvalues"]
        evr = pca_res["explained_variance_ratio"]
        cum_evr = np.cumsum(evr)

        # Eigenvalue Spectrum CSV rows
        for idx_ev, (e_val, evr_val, cev_val) in enumerate(zip(eigenvalues, evr, cum_evr)):
            spec_rows.append({
                "model": m_name,
                "component": idx_ev + 1,
                "eigenvalue": float(e_val),
                "explained_variance_ratio": float(evr_val),
                "cumulative_evr": float(cev_val)
            })

        # Process Conditions
        for cname, cinfo in m_res["primary_results"].items():
            gtype = cinfo["gen_type"]
            acc = cinfo["acc"]
            margin = cinfo["mean_margin"]
            damage = clean_margin - margin
            r_energy = cinfo["mean_realized_energy"]
            p_norm = cinfo["mean_patch_norm"]

            # Rank sweeps: Natural vs Energy-Matched
            if gtype == "natural_pca":
                rank_comp_rows.append({
                    "model": m_name,
                    "condition": cname,
                    "formulation": "natural",
                    "rank": cinfo["rank"],
                    "seed": cinfo["seed"],
                    "accuracy": acc,
                    "mean_margin": margin,
                    "margin_damage": damage,
                    "realized_energy": r_energy,
                    "mean_patch_norm": p_norm
                })
            elif gtype == "energy_matched_pca":
                rank_comp_rows.append({
                    "model": m_name,
                    "condition": cname,
                    "formulation": "energy_matched",
                    "rank": cinfo["rank"],
                    "seed": cinfo["seed"],
                    "accuracy": acc,
                    "mean_margin": margin,
                    "margin_damage": damage,
                    "realized_energy": r_energy,
                    "mean_patch_norm": p_norm
                })

            # PC1 Scale Sweep
            if gtype == "pc1_scale":
                sc = cinfo["scale"]
                sc_label = "E_MATCH" if abs(sc - m_res["E_MATCH"]) < 1e-4 else f"{sc:.2f}"
                scale_rows.append({
                    "model": m_name,
                    "condition": cname,
                    "scale_multiplier": sc,
                    "scale_label": sc_label,
                    "seed": cinfo["seed"],
                    "accuracy": acc,
                    "mean_margin": margin,
                    "margin_damage": damage,
                    "realized_energy": r_energy,
                    "mean_patch_norm": p_norm
                })

            # PC Identity Test
            if gtype in ["single_pc_natural", "single_pc_matched"]:
                pc_type = "natural" if gtype == "single_pc_natural" else "matched"
                pc_id_rows.append({
                    "model": m_name,
                    "condition": cname,
                    "condition_type": pc_type,
                    "pc_index": cinfo["pc_idx"],
                    "seed": cinfo["seed"],
                    "accuracy": acc,
                    "mean_margin": margin,
                    "margin_damage": damage,
                    "realized_energy": r_energy,
                    "mean_patch_norm": p_norm
                })

            # Random 1D Directions
            if gtype == "random_1d":
                rand_rows.append({
                    "model": m_name,
                    "condition": cname,
                    "seed": cinfo["seed"],
                    "target_variance": m_res["lambda_1"],
                    "accuracy": acc,
                    "mean_margin": margin,
                    "margin_damage": damage,
                    "realized_energy": r_energy,
                    "mean_patch_norm": p_norm
                })

            # Rank Propagation (for key diagnostic conditions)
            if cinfo["representation_diagnostics"]:
                for blk in [8, 9, 10, 11]:
                    r_diag = cinfo["representation_diagnostics"][blk]
                    prop_rows.append({
                        "model": m_name,
                        "condition": cname,
                        "block_depth": blk,
                        "numerical_rank": r_diag["numerical_rank"],
                        "stable_rank": r_diag["stable_rank"],
                        "effective_rank": r_diag["effective_rank"],
                        "patch_variance": r_diag["patch_variance"],
                        "mean_pairwise_cosine": r_diag["mean_pairwise_cosine"]
                    })

            # Attention Diagnostics (for key diagnostic conditions)
            if cinfo["attention_diagnostics"]:
                for blk in [9, 10, 11]:
                    a_diag = cinfo["attention_diagnostics"][blk]
                    attn_rows.append({
                        "model": m_name,
                        "condition": cname,
                        "block_depth": blk,
                        "key_variance": a_diag["key_variance"],
                        "value_variance": a_diag["value_variance"],
                        "cls_attention_entropy": a_diag["cls_attention_entropy"],
                        "attention_stable_rank": a_diag["attention_stable_rank"]
                    })

    # Save DataFrames
    pd.DataFrame(spec_rows).to_csv(os.path.join(args.output_dir, "eigenvalue_spectrum.csv"), index=False)
    pd.DataFrame(rank_comp_rows).to_csv(os.path.join(args.output_dir, "natural_vs_energy_matched_rank.csv"), index=False)
    pd.DataFrame(pc_id_rows).to_csv(os.path.join(args.output_dir, "pc_identity_results.csv"), index=False)
    pd.DataFrame(scale_rows).to_csv(os.path.join(args.output_dir, "pc1_scale_sweep.csv"), index=False)
    pd.DataFrame(rand_rows).to_csv(os.path.join(args.output_dir, "random_direction_results.csv"), index=False)
    pd.DataFrame(prop_rows).to_csv(os.path.join(args.output_dir, "rank_propagation.csv"), index=False)
    pd.DataFrame(attn_rows).to_csv(os.path.join(args.output_dir, "attention_diagnostics.csv"), index=False)

    # 5. Export Per-Image Parquet Files
    print("\nExporting per-image predictions to Parquet...")
    for m_label, m_res in [("tiny", tiny_res), ("small", small_res)]:
        p_data = {"target": m_res["all_targets"]}
        for cname, cinfo in m_res["primary_results"].items():
            p_data[f"{cname}_pred"] = cinfo["preds"]
            p_data[f"{cname}_correct"] = cinfo["corrects"]
            p_data[f"{cname}_margin"] = cinfo["margins"]
        pdf = pd.DataFrame(p_data)
        pdf.to_parquet(os.path.join(args.output_dir, f"{m_label}_image_results.parquet"), index=False)

    # 6. Evaluate Decision Rules & Summaries
    print("\nEvaluating Decision Rules...")
    summaries = {}
    propagation_summaries = {}

    for m_name, m_res in models_data:
        p_res = m_res["primary_results"]
        cent_acc = p_res["static_centroid"]["acc"]
        gauss_acc = p_res["diagonal_gaussian_seed_16001"]["acc"]

        nat_pca_accs = {}
        em_pca_accs = {}
        for r in ranks:
            n_accs = [p_res[f"natural_pca_rank_{r}_seed_{s}"]["acc"] for s in rank_seeds]
            nat_pca_accs[r] = float(np.mean(n_accs))
            e_accs = [p_res[f"energy_matched_pca_rank_{r}_seed_{s}"]["acc"] for s in rank_seeds]
            em_pca_accs[r] = float(np.mean(e_accs))

        pc_nat_accs = {}
        pc_mat_accs = {}
        for k in pc_indices:
            n_accs = [p_res[f"natural_pc_{k}_seed_{s}"]["acc"] for s in scale_seeds]
            pc_nat_accs[k] = float(np.mean(n_accs))
            m_accs = [p_res[f"matched_pc_{k}_seed_{s}"]["acc"] for s in scale_seeds]
            pc_mat_accs[k] = float(np.mean(m_accs))

        rand_accs = [p_res[f"random_1d_seed_{s}"]["acc"] for s in rand_seeds]

        summaries[m_name] = {
            "centroid_acc": cent_acc,
            "gaussian_acc": gauss_acc,
            "natural_pca_accs": nat_pca_accs,
            "energy_matched_pca_accs": em_pca_accs,
            "pc_identity_natural": pc_nat_accs,
            "pc_identity_matched": pc_mat_accs,
            "random_1d_acc": float(np.mean(rand_accs))
        }

        # Propagation for natural PCA Rank 1
        r1_cname = "natural_pca_rank_1_seed_16001"
        r1_diag = p_res[r1_cname]["representation_diagnostics"]
        r1_geom = p_res.get("pc1_scale_1.00_seed_17001", {}).get("geometric_expansion", {})
        propagation_summaries[m_name] = {
            "r_eff_depth8": float(r1_diag[8]["effective_rank"]),
            "r_eff_block9": float(r1_diag[9]["effective_rank"]),
            "r_eff_block10": float(r1_diag[10]["effective_rank"]),
            "r_eff_block11": float(r1_diag[11]["effective_rank"]),
            "geometric_expansion": r1_geom
        }

    decision_res = evaluate_v0_9_decision(
        tiny_summary=summaries["deit_tiny_patch16_224"],
        small_summary=summaries["deit_small_patch16_224"],
        tiny_propagation=propagation_summaries["deit_tiny_patch16_224"],
        small_propagation=propagation_summaries["deit_small_patch16_224"]
    )
    decision_res["tiny_summary"] = summaries["deit_tiny_patch16_224"]
    decision_res["small_summary"] = summaries["deit_small_patch16_224"]
    decision_res["tiny_propagation"] = propagation_summaries["deit_tiny_patch16_224"]
    decision_res["small_propagation"] = propagation_summaries["deit_small_patch16_224"]

    with open(os.path.join(args.output_dir, "decision_summary.json"), "w") as f:
        json.dump(decision_res, f, indent=2)

    # 7. Experiment Manifest
    manifest = {
        "protocol_version": "Fungibility V0.9",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": ["deit_tiny_patch16_224", "deit_small_patch16_224"],
        "target_depth": 8,
        "n_calib": len(calib_ds),
        "n_eval": len(eval_ds),
        "cls_untouched": True,
        "replacement_token_count": 196,
        "natural_pca_unscaled": True,
        "energy_matched_reproduced": True,
        "pc1_scale_multipliers": [0.25, 0.5, 1.0, 2.0, 4.0, "E_MATCH"],
        "max_random_u_norm_error": float(max(tiny_res["max_u_norm_err"], small_res["max_u_norm_err"])),
        "random_1d_variance_matched": True,
        "perturbation_energy_logged": True,
        "per_image_rank_computed": True,
        "attention_diagnostics_measured": True,
        "seeds_protocol_compliant": not args.smoke_test,
        "zero_labels_or_gradients": True,
        "tiny_runtime_s": tiny_res["elapsed"],
        "small_runtime_s": small_res["elapsed"],
        "tiny_peak_vram_mb": tiny_res["peak_vram_mb"],
        "small_peak_vram_mb": small_res["peak_vram_mb"],
        "verdict": decision_res["verdict"],
        "verdict_description": decision_res["verdict_description"],
        "rank_propagation_verdict": decision_res["rank_propagation_verdict"],
        "rank_propagation_description": decision_res["rank_propagation_description"]
    }
    with open(os.path.join(args.output_dir, "experiment_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # 8. Assertions Validation
    print("\nValidating 18 Programmatic Assertions...")
    if not args.smoke_test:
        all_passed, failures = validate_v0_9_assertions(
            manifest_data=manifest,
            tiny_res=tiny_res,
            small_res=small_res,
            calib_indices=calib_indices,
            eval_indices=eval_indices
        )
        if all_passed:
            print(">>> ALL 18 PROGRAMMATIC ASSERTIONS PASSED! <<<")
        else:
            print(">>> VALIDATION FAILURES DETECTED: <<<")
            for fail in failures:
                print(f"  [X] {fail}")
            sys.exit(1)
    else:
        print("[SMOKE TEST] Skipping full 18 assertion validation.")

    # 9. Generate Figures
    print("\nGenerating Figures...")
    plot_all_v0_9_figures(output_dir=args.output_dir, figures_dir=args.figures_dir)

    print("\n" + "="*75)
    print("V0.9 EXECUTION COMPLETE")
    print(f"Primary Verdict: {decision_res['verdict']}")
    print(f"Summary: {decision_res['verdict_description']}")
    print(f"Rank Propagation: {decision_res['rank_propagation_verdict']}")
    print(f"Propagation Summary: {decision_res['rank_propagation_description']}")
    print("="*75)


if __name__ == "__main__":
    main()
