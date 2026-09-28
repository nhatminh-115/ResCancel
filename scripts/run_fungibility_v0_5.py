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

from rescancel.dataset import ImageNetValidationSubset
from patch_fungibility.v0_5_pipeline import PatchFungibilityV05Pipeline
from patch_fungibility.v0_5_decision import evaluate_fungibility_v0_5_decision
from patch_fungibility.v0_5_validation import run_fungibility_v0_5_validations
from scripts.plot_fungibility_v0_5_figures import plot_all_fungibility_v0_5_figures


def parse_args():
    parser = argparse.ArgumentParser(description="Run Patch Content Fungibility V0.5 Experiment")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of ImageNet validation samples (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size (default: 32)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="outputs/fungibility_v0_5", help="Output directory")
    parser.add_argument("--figures_dir", type=str, default="figures/fungibility_v0_5", help="Figures directory")
    return parser.parse_args()


def export_model_tables(res: Dict[str, Any], prefix: str, output_dir: str):
    """Exports machine-readable tables required by protocol."""
    # 1. Image results parquet
    image_df = pd.DataFrame(res["image_records"])
    image_df.to_parquet(os.path.join(output_dir, f"{prefix}_image_results.parquet"), index=False)

    # 2. Seed results CSV
    seed_records = []
    for s_info in res["seed_contrasts"]["sphere"]:
        seed_records.append({
            "condition": "random_sphere",
            "seed": s_info["seed"],
            "cross_vs_seed_margin_diff": s_info["diff_mean"],
            "cohens_dz": s_info["cohens_dz"],
            "ci_lower": s_info["ci_95"][0],
            "ci_upper": s_info["ci_95"][1],
            "p_value": s_info["p_value"],
            "acc_diff": s_info["acc_diff"],
            "mcnemar_p": s_info["mcnemar_p"]
        })
    for s_info in res["seed_contrasts"]["gaussian"]:
        seed_records.append({
            "condition": "gaussian",
            "seed": s_info["seed"],
            "cross_vs_seed_margin_diff": s_info["diff_mean"],
            "cohens_dz": s_info["cohens_dz"],
            "ci_lower": s_info["ci_95"][0],
            "ci_upper": s_info["ci_95"][1],
            "p_value": s_info["p_value"],
            "acc_diff": s_info["acc_diff"],
            "mcnemar_p": s_info["mcnemar_p"]
        })
    for s_info in res["seed_contrasts"]["shuffle"]:
        seed_records.append({
            "condition": "feature_shuffle",
            "seed": s_info["seed"],
            "cross_vs_seed_margin_diff": s_info["diff_mean"],
            "cohens_dz": s_info["cohens_dz"],
            "ci_lower": s_info["ci_95"][0],
            "ci_upper": s_info["ci_95"][1],
            "p_value": s_info["p_value"],
            "acc_diff": s_info["acc_diff"],
            "mcnemar_p": s_info["mcnemar_p"]
        })
    pd.DataFrame(seed_records).to_csv(os.path.join(output_dir, f"{prefix}_seed_results.csv"), index=False)

    # 3. Condition comparison CSV
    comp_records = []
    for c_name, c_eval in res["comparisons"].items():
        comp_records.append({
            "comparison": c_name,
            "mean_margin_diff": c_eval.get("mean", 0.0),
            "median_margin_diff": c_eval.get("median", 0.0),
            "cohens_dz": c_eval.get("cohens_dz", 0.0),
            "ci_lower": c_eval.get("bootstrap_ci_95", [0.0, 0.0])[0],
            "ci_upper": c_eval.get("bootstrap_ci_95", [0.0, 0.0])[1],
            "p_value": c_eval.get("p_value", 1.0),
            "acc_diff": c_eval.get("acc_diff", 0.0)
        })
    pd.DataFrame(comp_records).to_csv(os.path.join(output_dir, f"{prefix}_condition_comparison.csv"), index=False)

    # 4. Activation diagnostics CSV
    diag_records = []
    for c_name, c_info in res["conditions"].items():
        if isinstance(c_info, dict) and "mean_token_l2" in c_info:
            diag_records.append({
                "condition": c_name,
                "accuracy": c_info.get("accuracy", 0.0),
                "mean_damage": c_info.get("mean_damage", 0.0),
                "mean_token_l2": c_info.get("mean_token_l2", 0.0),
                "std_token_l2": c_info.get("std_token_l2", 0.0),
                "cosine_to_orig": c_info.get("cosine_to_orig", 0.0),
                "cosine_to_mean": c_info.get("cosine_to_mean", 0.0),
                "dist_to_centroid": c_info.get("dist_to_centroid", 0.0),
                "mahalanobis_dist": c_info.get("mahalanobis_dist", 0.0)
            })
    pd.DataFrame(diag_records).to_csv(os.path.join(output_dir, f"{prefix}_activation_diagnostics.csv"), index=False)


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print("=================================================================")
    print("PATCH CONTENT FUNGIBILITY V0.5: Norm-Matched Null Falsification")
    print(f"Target Depth: Block 8 Output / Block 9 Input ONLY")
    print(f"Samples: {args.num_samples} (Stratified ImageNet-1k, Seed 42)")
    print(f"Device: {device}")
    print(f"Output Directory: {args.output_dir}")
    print("=================================================================")

    # 1. Load Dataset
    print("\nLoading ImageNet-1k validation subset...")
    dataset = ImageNetValidationSubset(num_samples=args.num_samples, seed=42, stratified=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Loaded {len(dataset)} samples successfully.")

    # 2. Run DeiT-Tiny
    print("\n>>> Running DeiT-Tiny V0.5 Pipeline...")
    tiny_pipe = PatchFungibilityV05Pipeline(
        model_name="deit_tiny_patch16_224",
        device=device,
        batch_size=args.batch_size
    )
    tiny_res = tiny_pipe.run(dataloader)

    # 3. Run DeiT-Small
    print("\n>>> Running DeiT-Small V0.5 Pipeline...")
    small_pipe = PatchFungibilityV05Pipeline(
        model_name="deit_small_patch16_224",
        device=device,
        batch_size=args.batch_size
    )
    small_res = small_pipe.run(dataloader)

    # 4. Programmatic Validations
    print("\n>>> Executing Pre-Registered Programmatic Validations...")
    tiny_df = pd.DataFrame(tiny_res["image_records"])
    small_df = pd.DataFrame(small_res["image_records"])

    tiny_validations = run_fungibility_v0_5_validations(tiny_df, tiny_res["manifest"], expected_n=args.num_samples)
    small_validations = run_fungibility_v0_5_validations(small_df, small_res["manifest"], expected_n=args.num_samples)
    print(f"DeiT-Tiny Validations: {len(tiny_validations)}/14 Passed.")
    print(f"DeiT-Small Validations: {len(small_validations)}/14 Passed.")

    # 5. Evaluate Decision Rules
    print("\n>>> Evaluating Pre-Registered Decision Rules...")
    decision_res = evaluate_fungibility_v0_5_decision(tiny_res, small_res)
    print(f"\n=======================================================")
    print(f"FINAL SCIENTIFIC VERDICT: {decision_res['decision']}")
    print(f"=======================================================")
    for r in decision_res["rationale"]:
        print(f"  * {r}")

    # 6. Save Manifests and Datasets
    manifest_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "device": str(device),
            "target_depth": 8,
            "mask_count": 49,
            "mask_fraction": 0.25,
            "seeds": {
                "dataset_seed": 42,
                "mask_seed": 2501,
                "donor_seed": 3501,
                "sphere_seeds": [6501, 6502, 6503, 6504, 6505],
                "gaussian_seeds": [7501, 7502, 7503, 7504, 7505],
                "feature_shuffle_seeds": [8501, 8502, 8503]
            }
        },
        "tiny_results": tiny_res,
        "small_results": small_res,
        "validations": {
            "tiny": tiny_validations,
            "small": small_validations
        },
        "decision": decision_res
    }

    manifest_path = os.path.join(args.output_dir, "experiment_manifest.json")
    with open(manifest_path, "w") as f:
        # Avoid serializing tensors or complex types by standard json dumping
        def default_serializer(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            return str(obj)
        json.dump(manifest_data, f, indent=2, default=default_serializer)

    export_model_tables(tiny_res, "tiny", args.output_dir)
    export_model_tables(small_res, "small", args.output_dir)
    print(f"\nSaved all machine-readable tables and manifest to {args.output_dir}/")

    # 7. Generate Figures
    print("\n>>> Generating publication figures...")
    plot_all_fungibility_v0_5_figures(manifest_data, output_dir=args.figures_dir)

    # 8. Print Formatted Results Table
    print("\n==========================================================================================")
    print("SUMMARY RESULTS TABLE: V0.5 NULL CONTROLS AT BLOCK 8")
    print("==========================================================================================")
    header = f"{'Model':<12} | {'Condition':<22} | {'Acc (%)':<8} | {'Dmg Mean':<9} | {'Token L2':<9} | {'Mahal Dist':<10}"
    print(header)
    print("-" * len(header))
    for m_name, res in [("DeiT-Tiny", tiny_res), ("DeiT-Small", small_res)]:
        conds = res["conditions"]
        key_list = [
            ("clean_baseline", res["manifest"]["baseline_accuracy"], 0.0, conds["cross_image"]["mean_token_l2"], 0.0),
            ("zero", conds["zero"]["accuracy"], conds["zero"]["mean_damage"], conds["zero"]["mean_token_l2"], conds["zero"]["mahalanobis_dist"]),
            ("random_sphere (mean)", conds["random_sphere_mean"]["accuracy"], conds["random_sphere_mean"]["mean_damage"], conds["random_sphere_mean"]["mean_token_l2"], conds["random_sphere_mean"]["mahalanobis_dist"]),
            ("gaussian (mean)", conds["gaussian_mean"]["accuracy"], conds["gaussian_mean"]["mean_damage"], conds["gaussian_mean"]["mean_token_l2"], conds["gaussian_mean"]["mahalanobis_dist"]),
            ("feature_shuffle (mean)", conds["feature_shuffle_mean"]["accuracy"], conds["feature_shuffle_mean"]["mean_damage"], conds["feature_shuffle_mean"]["mean_token_l2"], conds["feature_shuffle_mean"]["mahalanobis_dist"]),
            ("cross_image", conds["cross_image"]["accuracy"], conds["cross_image"]["mean_damage"], conds["cross_image"]["mean_token_l2"], conds["cross_image"]["mahalanobis_dist"]),
            ("norm_matched_cross", conds["norm_matched_cross"]["accuracy"], conds["norm_matched_cross"]["mean_damage"], conds["norm_matched_cross"]["mean_token_l2"], conds["norm_matched_cross"]["mahalanobis_dist"]),
            ("same_image_mean", conds["layer_mean"]["accuracy"], conds["layer_mean"]["mean_damage"], conds["layer_mean"]["mean_token_l2"], conds["layer_mean"]["mahalanobis_dist"]),
            ("norm_matched_mean", conds["norm_matched_mean"]["accuracy"], conds["norm_matched_mean"]["mean_damage"], conds["norm_matched_mean"]["mean_token_l2"], conds["norm_matched_mean"]["mahalanobis_dist"])
        ]
        for c_lbl, acc, dmg, l2, mahal in key_list:
            print(f"{m_name:<12} | {c_lbl:<22} | {acc*100:<8.1f} | {dmg:<9.3f} | {l2:<9.2f} | {mahal:<10.1f}")
        print("-" * len(header))
    print("==========================================================================================")


if __name__ == "__main__":
    main()
