import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import time
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits
from patch_fungibility.v0_6_pipeline import PatchFungibilityV06Pipeline
from patch_fungibility.v0_6_decision import evaluate_fungibility_v0_6_decision
from patch_fungibility.v0_6_validation import run_fungibility_v0_6_validations
from scripts.plot_fungibility_v0_6_figures import plot_all_fungibility_v0_6_figures


def parse_args():
    parser = argparse.ArgumentParser(description="Run Patch Content Fungibility V0.6 Experiment")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples per split (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size (default: 32)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="outputs/fungibility_v0_6", help="Output directory")
    parser.add_argument("--figures_dir", type=str, default="figures/fungibility_v0_6", help="Figures directory")
    parser.add_argument("--depths", nargs="+", type=int, default=[5, 6, 7, 8, 9, 10], help="Tested block depths")
    parser.add_argument("--fractions", nargs="+", type=str, default=["10%", "25%", "50%"], help="Tested replacement fractions")
    return parser.parse_args()


def export_v0_6_tables(res: Dict[str, Any], prefix: str, output_dir: str):
    """Exports required tables per model."""
    # 1. Seed results CSV
    pd.DataFrame(res["seed_contrasts"]).to_csv(os.path.join(output_dir, f"{prefix}_seed_results.csv"), index=False)

    # 2. Depth fraction summary CSV
    pd.DataFrame(res["summary_rows"]).to_csv(os.path.join(output_dir, f"{prefix}_depth_fraction_summary.csv"), index=False)

    # 3. Condition comparisons CSV
    comp_rows = []
    for d, d_dict in res["depth_fraction_results"].items():
        for f, f_dict in d_dict.items():
            for c_name, c_eval in f_dict["comparisons"].items():
                if isinstance(c_eval, dict) and "mean" in c_eval:
                    comp_rows.append({
                        "model": res["manifest"]["model_name"],
                        "depth": d,
                        "fraction": f,
                        "comparison": c_name,
                        "mean_diff": c_eval.get("mean", 0.0),
                        "median_diff": c_eval.get("median", 0.0),
                        "cohens_dz": c_eval.get("cohens_dz", 0.0),
                        "ci_lower": c_eval.get("bootstrap_ci_95", [0.0, 0.0])[0],
                        "ci_upper": c_eval.get("bootstrap_ci_95", [0.0, 0.0])[1],
                        "p_value": c_eval.get("p_value", 1.0),
                        "fdr_q_value": c_eval.get("fdr_q_value", 1.0)
                    })
    pd.DataFrame(comp_rows).to_csv(os.path.join(output_dir, f"{prefix}_condition_comparisons.csv"), index=False)


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print("=================================================================")
    print("PATCH CONTENT FUNGIBILITY V0.6: Held-Out Depth Transition Pipeline")
    print(f"Depths: {args.depths}, Fractions: {args.fractions}")
    print(f"Calibration & Evaluation Samples: {args.n_samples} each (Disjoint)")
    print(f"Device: {device}")
    print(f"Output Directory: {args.output_dir}")
    print("=================================================================")

    # 1. Create Disjoint Datasets
    print("\n>>> Loading ImageNet validation set and constructing strictly disjoint splits...")
    calib_dataset, eval_dataset, calib_df, eval_df = get_disjoint_imagenet_splits(
        calib_seed=9101,
        eval_seed=9201,
        n_per_split=args.n_samples,
        output_dir=args.output_dir
    )
    calib_loader = DataLoader(calib_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Splits verified: Calibration N={len(calib_dataset)}, Evaluation N={len(eval_dataset)}, Overlap=0.")

    # 2. Run DeiT-Tiny
    print("\n>>> Running DeiT-Tiny V0.6 Pipeline...")
    tiny_pipe = PatchFungibilityV06Pipeline(
        model_name="deit_tiny_patch16_224",
        device=device,
        tested_depths=args.depths,
        fractions=args.fractions,
        batch_size=args.batch_size
    )
    tiny_res = tiny_pipe.run(calib_loader, eval_loader, output_dir=args.output_dir)

    # 3. Run DeiT-Small
    print("\n>>> Running DeiT-Small V0.6 Pipeline...")
    small_pipe = PatchFungibilityV06Pipeline(
        model_name="deit_small_patch16_224",
        device=device,
        tested_depths=args.depths,
        fractions=args.fractions,
        batch_size=args.batch_size
    )
    small_res = small_pipe.run(calib_loader, eval_loader, output_dir=args.output_dir)

    # 4. Save Calibration Statistics to .npz
    print("\n>>> Persisting calibration statistics to calibration_statistics.npz...")
    calib_npz_data = {}
    for m_res in [tiny_res, small_res]:
        m_name = m_res["manifest"]["model_name"]
        for d, s_dict in m_res["calibration_stats"].items():
            calib_npz_data[f"{m_name}_depth_{d}_mu"] = s_dict["mu"].cpu().numpy()
            calib_npz_data[f"{m_name}_depth_{d}_sigma"] = s_dict["sigma"].cpu().numpy()
            calib_npz_data[f"{m_name}_depth_{d}_var"] = s_dict["var"].cpu().numpy()
    np.savez_compressed(os.path.join(args.output_dir, "calibration_statistics.npz"), **calib_npz_data)

    # 5. Programmatic Validations
    print("\n>>> Executing Pre-Registered Programmatic Validations...")
    tiny_validations = run_fungibility_v0_6_validations(
        tiny_res["manifest"], calib_df, eval_df, expected_n=args.n_samples, expected_depths=args.depths
    )
    small_validations = run_fungibility_v0_6_validations(
        small_res["manifest"], calib_df, eval_df, expected_n=args.n_samples, expected_depths=args.depths
    )
    print(f"DeiT-Tiny Validations: {len(tiny_validations)}/14 Passed.")
    print(f"DeiT-Small Validations: {len(small_validations)}/14 Passed.")

    # 6. Evaluate Decision Rules
    print("\n>>> Evaluating Pre-Registered Decision Rules...")
    decision_res = evaluate_fungibility_v0_6_decision(tiny_res, small_res)
    print(f"\n=======================================================")
    print(f"FINAL SCIENTIFIC VERDICT: {decision_res['decision']}")
    print(f"=======================================================")
    for r in decision_res["rationale"]:
        print(f"  * {r}")

    # 7. Export Model Tables and Manifest
    export_v0_6_tables(tiny_res, "tiny", args.output_dir)
    export_v0_6_tables(small_res, "small", args.output_dir)

    combined_summary_df = pd.concat([
        pd.DataFrame(tiny_res["summary_rows"]),
        pd.DataFrame(small_res["summary_rows"])
    ], ignore_index=True)
    combined_summary_df.to_csv(os.path.join(args.output_dir, "summary_all_models.csv"), index=False)

    manifest_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "device": str(device),
            "tested_depths": args.depths,
            "fractions": args.fractions,
            "mask_counts": [20, 49, 98],
            "seeds": {
                "calib_seed": 9101,
                "eval_seed": 9201,
                "mask_seed": 9301,
                "gaussian_seeds": [9401, 9402, 9403, 9404, 9405],
                "sphere_seeds": [9501, 9502, 9503]
            }
        },
        "tiny_manifest": tiny_res["manifest"],
        "small_manifest": small_res["manifest"],
        "validations": {
            "tiny": tiny_validations,
            "small": small_validations
        },
        "decision": decision_res
    }

    manifest_path = os.path.join(args.output_dir, "experiment_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    # 8. Generate Publication Figures
    print("\n>>> Generating publication figures...")
    plot_all_fungibility_v0_6_figures(combined_summary_df, output_dir=args.figures_dir)

    # 9. Print Formatted Results Table
    print("\n====================================================================================================================")
    print(f"{'Model':<12} | {'Depth':<6} | {'Budget':<7} | {'Zero Dmg':<9} | {'Gauss Dmg':<9} | {'Sphere Dmg':<10} | {'Recov %':<8} | {'dz(G-Z)':<8} | {'FDR q':<10}")
    print("====================================================================================================================")
    for _, row in combined_summary_df.iterrows():
        m_name = "DeiT-Tiny" if "tiny" in row["model"] else "DeiT-Small"
        d = row["depth"]
        f = row["fraction"]
        z_dmg = row["zero_damage"]
        g_dmg = row["gaussian_damage"]
        s_dmg = row["sphere_damage"]
        rec = row["recovery_fraction"] * 100.0
        dz = row["gauss_vs_zero_dz"]
        q = row["gauss_vs_zero_fdr_q"]
        print(f"{m_name:<12} | {d:<6} | {f:<7} | {z_dmg:<9.3f} | {g_dmg:<9.3f} | {s_dmg:<10.3f} | {rec:<8.1f} | {dz:<8.3f} | {q:<10.4e}")
    print("====================================================================================================================")


if __name__ == "__main__":
    main()
