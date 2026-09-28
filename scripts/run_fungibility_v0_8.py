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
from patch_fungibility.v0_8_pipeline import PatchFungibilityV08Pipeline
from patch_fungibility.v0_8_validation import validate_v0_8_assertions
from patch_fungibility.v0_8_decision import compute_paired_statistics, evaluate_v0_8_decision, compute_r95_and_k95
from scripts.plot_fungibility_v0_8_figures import plot_all_v0_8_figures


def main():
    parser = argparse.ArgumentParser(description="Run Patch Fungibility V0.8 Token Diversity / Effective-Rank Sufficiency Experiment")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output-dir", type=str, default="outputs/fungibility_v0_8", help="Output directory")
    parser.add_argument("--figures-dir", type=str, default="figures/fungibility_v0_8", help="Figures directory")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test on 50 images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print(f"\n{'='*75}\nPATCH FUNGIBILITY V0.8: TOKEN-DIVERSITY / EFFECTIVE-RANK SUFFICIENCY TEST\n{'='*75}")
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
        pca_ranks = [1, 4, 16]
        grouped_k = [1, 8, 196]
        diag_seeds = [11001]
        pca_seeds = [11001]
        rand_seeds = [12001]
        iso_seeds = [13001]
        shared_indep_seeds = [14001, 14002]
        grp_seeds = [15001]
    else:
        calib_indices = calib_df["global_index"].tolist()
        eval_indices = eval_df["global_index"].tolist()
        pca_ranks = [1, 2, 4, 8, 16, 32, 64]
        grouped_k = [1, 2, 4, 8, 16, 32, 64, 196]
        diag_seeds = [11001, 11002, 11003, 11004, 11005]
        pca_seeds = [11001, 11002, 11003]
        rand_seeds = [12001, 12002, 12003]
        iso_seeds = [13001, 13002, 13003]
        shared_indep_seeds = [14001, 14002, 14003, 14004, 14005]
        grp_seeds = [15001, 15002, 15003]

    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Run DeiT-Tiny
    print("\n" + "="*50 + " RUNNING DEIT-TINY " + "="*50)
    tiny_pipe = PatchFungibilityV08Pipeline(
        model_name="deit_tiny_patch16_224",
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        pca_ranks=pca_ranks,
        grouped_k=grouped_k,
        diag_gaussian_seeds=diag_seeds,
        pca_rank_seeds=pca_seeds,
        random_rank_seeds=rand_seeds,
        isotropic_seeds=iso_seeds,
        shared_vs_indep_seeds=shared_indep_seeds,
        grouped_seeds=grp_seeds
    )
    tiny_res = tiny_pipe.run(calib_loader, eval_loader, n_samples=len(eval_ds))

    # 3. Run DeiT-Small
    print("\n" + "="*50 + " RUNNING DEIT-SMALL " + "="*50)
    small_pipe = PatchFungibilityV08Pipeline(
        model_name="deit_small_patch16_224",
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        pca_ranks=pca_ranks,
        grouped_k=grouped_k,
        diag_gaussian_seeds=diag_seeds,
        pca_rank_seeds=pca_seeds,
        random_rank_seeds=rand_seeds,
        isotropic_seeds=iso_seeds,
        shared_vs_indep_seeds=shared_indep_seeds,
        grouped_seeds=grp_seeds
    )
    small_res = small_pipe.run(calib_loader, eval_loader, n_samples=len(eval_ds))

    # 4. Save PCA Basis Metadata
    print("\nSaving PCA Basis Metadata...")
    pca_meta = {
        "deit_tiny": {
            "n_tokens": tiny_res["pca_result"]["n_tokens"],
            "embed_dim": tiny_res["pca_result"]["embed_dim"],
            "E_full": tiny_res["pca_result"]["E_full"],
            "ortho_error": tiny_res["pca_result"]["ortho_error"],
            "eigenvalues": tiny_res["pca_result"]["eigenvalues"].tolist(),
            "explained_variance_ratio": tiny_res["pca_result"]["explained_variance_ratio"]
        },
        "deit_small": {
            "n_tokens": small_res["pca_result"]["n_tokens"],
            "embed_dim": small_res["pca_result"]["embed_dim"],
            "E_full": small_res["pca_result"]["E_full"],
            "ortho_error": small_res["pca_result"]["ortho_error"],
            "eigenvalues": small_res["pca_result"]["eigenvalues"].tolist(),
            "explained_variance_ratio": small_res["pca_result"]["explained_variance_ratio"]
        }
    }
    with open(os.path.join(args.output_dir, "pca_basis_metadata.json"), "w") as f:
        json.dump(pca_meta, f, indent=2)

    # 5. Build Tabular Outputs
    print("\nConstructing tabular datasets and CSVs...")
    rank_rows = []
    grouped_rows = []
    shared_rows = []
    rep_rows = []
    attn_rows = []
    sec75_rows = []
    energy_rows = []
    stat_rows = []

    models_data = [
        ("deit_tiny_patch16_224", tiny_res),
        ("deit_small_patch16_224", small_res)
    ]

    for m_name, m_res in models_data:
        clean_res = m_res["primary_results"]["clean"]
        clean_acc = clean_res["acc"]
        clean_margin = clean_res["mean_margin"]
        clean_margins_arr = np.array(clean_res["margins"])
        clean_corrects_arr = np.array(clean_res["corrects"])
        E_full = m_res["pca_result"]["E_full"]

        # Primary conditions
        for cname, cinfo in m_res["primary_results"].items():
            gtype = cinfo["gen_type"]
            acc = cinfo["acc"]
            margin = cinfo["mean_margin"]
            damage = clean_margin - margin
            margins_arr = np.array(cinfo["margins"])
            corrects_arr = np.array(cinfo["corrects"])

            r_depth8 = cinfo["representation_diagnostics"][8]["effective_rank"]
            r_block11 = cinfo["representation_diagnostics"][11]["effective_rank"]

            # Energy verification
            if gtype not in ["clean", "zero", "centroid"]:
                realized_e = cinfo["mean_realized_energy"]
                rel_err = cinfo["energy_rel_error"]
                energy_rows.append({
                    "model": m_name,
                    "condition": cname,
                    "gen_type": gtype,
                    "target_energy": E_full,
                    "realized_energy": realized_e,
                    "energy_rel_error": rel_err,
                    "passed_2pct_tolerance": rel_err <= 0.02
                })

            # Representation diagnostics across blocks
            for blk in [8, 9, 10, 11]:
                r_diag = cinfo["representation_diagnostics"][blk]
                rep_rows.append({
                    "model": m_name,
                    "condition": cname,
                    "block_depth": blk,
                    "numerical_rank": r_diag["numerical_rank"],
                    "stable_rank": r_diag["stable_rank"],
                    "effective_rank": r_diag["effective_rank"],
                    "mean_pairwise_cosine": r_diag["mean_pairwise_cosine"]
                })

            # Attention diagnostics across blocks
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

            # Rank sweeps (PCA & Random)
            if gtype == "pca_rank":
                rank_rows.append({
                    "model": m_name,
                    "condition": cname,
                    "basis_type": "pca",
                    "rank": cinfo["rank"],
                    "seed": cinfo["seed"],
                    "accuracy": acc,
                    "mean_margin": margin,
                    "margin_damage": damage,
                    "effective_rank_depth8": r_depth8,
                    "effective_rank_block11": r_block11
                })
            elif gtype == "random_rank":
                rank_rows.append({
                    "model": m_name,
                    "condition": cname,
                    "basis_type": "random",
                    "rank": cinfo["rank"],
                    "seed": cinfo["seed"],
                    "accuracy": acc,
                    "mean_margin": margin,
                    "margin_damage": damage,
                    "effective_rank_depth8": r_depth8,
                    "effective_rank_block11": r_block11
                })

            # Grouped diversity
            if gtype == "grouped":
                grouped_rows.append({
                    "model": m_name,
                    "condition": cname,
                    "k": cinfo["k_val"],
                    "seed": cinfo["seed"],
                    "accuracy": acc,
                    "mean_margin": margin,
                    "margin_damage": damage,
                    "effective_rank_depth8": r_depth8
                })

            # Shared vs Independent
            if gtype in ["centroid", "shared_noise", "independent_noise"]:
                shared_rows.append({
                    "model": m_name,
                    "condition": cname,
                    "condition_type": gtype if gtype != "centroid" else "static_centroid",
                    "seed": cinfo["seed"] if cinfo["seed"] is not None else 0,
                    "accuracy": acc,
                    "mean_margin": margin,
                    "margin_damage": damage
                })

            # Compute paired statistics vs clean baseline
            stats_dict = compute_paired_statistics(clean_margins_arr, margins_arr, clean_corrects_arr, corrects_arr)
            stat_rows.append({
                "model": m_name,
                "condition": cname,
                "gen_type": gtype,
                "accuracy": acc,
                "mean_margin": margin,
                "margin_damage": damage,
                "ci_lower": stats_dict["ci_lower"],
                "ci_upper": stats_dict["ci_upper"],
                "t_stat": stats_dict["t_stat"],
                "t_pval": stats_dict["t_pval"],
                "wilcoxon_stat": stats_dict["wilcoxon_stat"],
                "wilcoxon_pval": stats_dict["wilcoxon_pval"],
                "cohen_dz": stats_dict["cohen_dz"],
                "mcnemar_b": stats_dict["mcnemar_b"],
                "mcnemar_c": stats_dict["mcnemar_c"],
                "mcnemar_pval": stats_dict["mcnemar_pval"]
            })

        # Secondary 75%
        for cname, cinfo in m_res["secondary_75_results"].items():
            acc = cinfo["acc"]
            margin = cinfo["mean_margin"]
            damage = clean_margin - margin
            r_depth8 = cinfo["representation_diagnostics"][8]["effective_rank"]
            sec75_rows.append({
                "model": m_name,
                "condition": cname,
                "gen_type": cinfo["gen_type"],
                "accuracy": acc,
                "mean_margin": margin,
                "margin_damage": damage,
                "effective_rank_depth8": r_depth8
            })

    # Save DataFrames
    rank_df = pd.DataFrame(rank_rows)
    grouped_df = pd.DataFrame(grouped_rows)
    shared_df = pd.DataFrame(shared_rows)
    rep_df = pd.DataFrame(rep_rows)
    attn_df = pd.DataFrame(attn_rows)
    sec75_df = pd.DataFrame(sec75_rows)
    energy_df = pd.DataFrame(energy_rows)
    stat_df = pd.DataFrame(stat_rows)

    rank_df.to_csv(os.path.join(args.output_dir, "rank_condition_results.csv"), index=False)
    grouped_df.to_csv(os.path.join(args.output_dir, "grouped_diversity_results.csv"), index=False)
    shared_df.to_csv(os.path.join(args.output_dir, "shared_vs_independent_results.csv"), index=False)
    rep_df.to_csv(os.path.join(args.output_dir, "representation_rank_diagnostics.csv"), index=False)
    attn_df.to_csv(os.path.join(args.output_dir, "attention_rank_diagnostics.csv"), index=False)
    sec75_df.to_csv(os.path.join(args.output_dir, "secondary_75_results.csv"), index=False)
    energy_df.to_csv(os.path.join(args.output_dir, "energy_verification.csv"), index=False)
    stat_df.to_csv(os.path.join(args.output_dir, "statistical_comparisons.csv"), index=False)

    # 6. Save Per-Image Parquet Files
    print("\nExporting per-image predictions to Parquet...")
    for m_label, m_res in [("tiny", tiny_res), ("small", small_res)]:
        p_data = {"target": m_res["all_targets"]}
        for cname, cinfo in m_res["primary_results"].items():
            p_data[f"{cname}_pred"] = cinfo["preds"]
            p_data[f"{cname}_correct"] = cinfo["corrects"]
            p_data[f"{cname}_margin"] = cinfo["margins"]
        pdf = pd.DataFrame(p_data)
        pdf.to_parquet(os.path.join(args.output_dir, f"{m_label}_image_results.parquet"), index=False)

    # 7. Compute Model Summaries and Decision
    print("\nEvaluating Decision Rules and Spearman Correlations...")
    summaries = {}
    for m_name, m_res in models_data:
        p_res = m_res["primary_results"]
        cent_acc = p_res["static_centroid"]["acc"]
        gauss_accs = [p_res[f"diagonal_gaussian_seed_{s}"]["acc"] for s in diag_seeds]
        shared_accs = [p_res[f"shared_noise_seed_{s}"]["acc"] for s in shared_indep_seeds]
        indep_accs = [p_res[f"independent_noise_seed_{s}"]["acc"] for s in shared_indep_seeds]
        iso_accs = [p_res[f"isotropic_full_seed_{s}"]["acc"] for s in iso_seeds]

        pca_acc_dict = {}
        for r in pca_ranks:
            r_accs = [p_res[f"pca_rank_{r}_seed_{s}"]["acc"] for s in pca_seeds]
            pca_acc_dict[r] = float(np.mean(r_accs))

        rand_acc_dict = {}
        for r in pca_ranks:
            r_accs = [p_res[f"random_rank_{r}_seed_{s}"]["acc"] for s in rand_seeds]
            rand_acc_dict[r] = float(np.mean(r_accs))

        grp_acc_dict = {}
        for k in grouped_k:
            k_accs = [p_res[f"grouped_k_{k}_seed_{s}"]["acc"] for s in grp_seeds]
            grp_acc_dict[k] = float(np.mean(k_accs))

        # Spearman correlation between effective rank at Depth 8 and accuracy/margin
        m_ranks = []
        m_accs = []
        m_margins = []
        for cname, cinfo in p_res.items():
            if cinfo["gen_type"] not in ["clean", "zero"]:
                m_ranks.append(cinfo["representation_diagnostics"][8]["effective_rank"])
                m_accs.append(cinfo["acc"])
                m_margins.append(cinfo["mean_margin"])

        spearman_acc, _ = stats.spearmanr(m_ranks, m_accs)
        spearman_margin, _ = stats.spearmanr(m_ranks, m_margins)

        summaries[m_name] = {
            "centroid_acc": cent_acc,
            "gaussian_acc": float(np.mean(gauss_accs)),
            "shared_acc": float(np.mean(shared_accs)),
            "indep_acc": float(np.mean(indep_accs)),
            "iso_acc": float(np.mean(iso_accs)),
            "pca_accs": pca_acc_dict,
            "rand_accs": rand_acc_dict,
            "grouped_k_accs": grp_acc_dict,
            "spearman_eff_rank_accuracy": float(spearman_acc),
            "spearman_eff_rank_margin": float(spearman_margin)
        }

    decision_res = evaluate_v0_8_decision(
        tiny_summary=summaries["deit_tiny_patch16_224"],
        small_summary=summaries["deit_small_patch16_224"]
    )
    decision_res["tiny_summary"] = summaries["deit_tiny_patch16_224"]
    decision_res["small_summary"] = summaries["deit_small_patch16_224"]

    with open(os.path.join(args.output_dir, "decision_summary.json"), "w") as f:
        json.dump(decision_res, f, indent=2)

    # 8. Experiment Manifest
    manifest = {
        "protocol_version": "Fungibility V0.8",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": ["deit_tiny_patch16_224", "deit_small_patch16_224"],
        "target_depth": 8,
        "n_calib": len(calib_ds),
        "n_eval": len(eval_ds),
        "cls_untouched": True,
        "primary_replacement_token_count": 196,
        "pca_ranks": pca_ranks,
        "grouped_k": grouped_k,
        "max_random_ortho_error": float(max(tiny_res["max_random_ortho_error"], small_res["max_random_ortho_error"])),
        "shared_and_indep_identical_dist": True,
        "grouped_k_exact": True,
        "seeds_match_protocol": not args.smoke_test,
        "zero_labels_or_gradients": True,
        "representation_ranks_computed": True,
        "attention_diagnostics_computed": True,
        "machine_readable_exported": True,
        "tiny_runtime_s": tiny_res["elapsed"],
        "small_runtime_s": small_res["elapsed"],
        "tiny_peak_vram_mb": tiny_res["peak_vram_mb"],
        "small_peak_vram_mb": small_res["peak_vram_mb"],
        "verdict": decision_res["verdict"],
        "verdict_description": decision_res["verdict_description"]
    }
    with open(os.path.join(args.output_dir, "experiment_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # 9. Assertions Validation
    print("\nValidating 18 Programmatic Assertions...")
    if not args.smoke_test:
        all_passed, failures = validate_v0_8_assertions(
            manifest_data=manifest,
            tiny_res=tiny_res,
            small_res=small_res,
            energy_df=energy_df,
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

    # 10. Generate Figures
    print("\nGenerating Figures...")
    plot_all_v0_8_figures(output_dir=args.output_dir, figures_dir=args.figures_dir)

    print("\n" + "="*75)
    print("V0.8 EXECUTION COMPLETE")
    print(f"Verdict: {decision_res['verdict']}")
    print(f"Summary: {decision_res['verdict_description']}")
    print(f"Tiny: R_95={decision_res['tiny']['r_95']}, K_95={decision_res['tiny']['k_95']}")
    print(f"Small: R_95={decision_res['small']['r_95']}, K_95={decision_res['small']['k_95']}")
    print("="*75)


if __name__ == "__main__":
    main()
