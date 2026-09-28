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
from patch_fungibility.pipeline import PatchFungibilityPipeline
from patch_fungibility.decision import evaluate_fungibility_decision
from patch_fungibility.validation import run_fungibility_v0_validations
from scripts.plot_fungibility_v0_figures import plot_all_fungibility_figures


def parse_args():
    parser = argparse.ArgumentParser(description="Run Patch Content Fungibility V0 Experiment")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of ImageNet validation samples (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size (default: 32)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="outputs/fungibility_v0", help="Output directory for manifests and data")
    parser.add_argument("--figures_dir", type=str, default="figures/fungibility_v0", help="Directory for saved figures")
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4, 6, 8, 10], help="Tested block depths (default: 2 4 6 8 10)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print("=================================================================")
    print("PATCH CONTENT FUNGIBILITY V0: Mechanistic Falsification Pipeline")
    print(f"Samples: {args.num_samples} (Stratified ImageNet-1k, Seed 42)")
    print(f"Tested Depths: {args.depths}")
    print(f"Device: {device}")
    print(f"Output Directory: {args.output_dir}")
    print("=================================================================")

    # 1. Load Dataset
    print("\nLoading ImageNet-1k validation subset...")
    dataset = ImageNetValidationSubset(num_samples=args.num_samples, seed=42, stratified=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Loaded {len(dataset)} samples successfully.")

    # 2. Run DeiT-Tiny
    print("\n>>> Running DeiT-Tiny Pipeline...")
    tiny_pipe = PatchFungibilityPipeline(
        model_name="deit_tiny_patch16_224",
        device=device,
        tested_depths=args.depths,
        batch_size=args.batch_size
    )
    tiny_res = tiny_pipe.run(dataloader)

    # 3. Run DeiT-Small
    print("\n>>> Running DeiT-Small Pipeline...")
    small_pipe = PatchFungibilityPipeline(
        model_name="deit_small_patch16_224",
        device=device,
        tested_depths=args.depths,
        batch_size=args.batch_size
    )
    small_res = small_pipe.run(dataloader)

    # 4. Programmatic Validations
    print("\n>>> Executing Pre-Registered Programmatic Validations...")
    tiny_df = pd.DataFrame(tiny_res["image_records"])
    small_df = pd.DataFrame(small_res["image_records"])

    # Slice depth 2 image records for sample-level assertions (1000 unique rows)
    tiny_sample_df = tiny_df[tiny_df["depth"] == args.depths[0]]
    small_sample_df = small_df[small_df["depth"] == args.depths[0]]

    tiny_validations = run_fungibility_v0_validations(
        tiny_sample_df, tiny_res["manifest"], expected_n=args.num_samples, expected_depths=args.depths
    )
    small_validations = run_fungibility_v0_validations(
        small_sample_df, small_res["manifest"], expected_n=args.num_samples, expected_depths=args.depths
    )
    print(f"DeiT-Tiny Validations: {len(tiny_validations)}/14 Passed.")
    print(f"DeiT-Small Validations: {len(small_validations)}/14 Passed.")

    # 5. Evaluate Decision Rules
    print("\n>>> Evaluating Pre-Registered Decision Rules...")
    decision_res = evaluate_fungibility_decision(
        tiny_res["depth_results"],
        small_res["depth_results"],
        tested_depths=args.depths
    )
    print(f"\n=======================================================")
    print(f"FINAL SCIENTIFIC VERDICT: {decision_res['decision']}")
    print(f"=======================================================")
    for r in decision_res["rationale"]:
        print(f"  * {r}")

    # 6. Save Manifests and Datasets
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "device": str(device),
            "tested_depths": args.depths,
            "mask_count": 49,
            "mask_fraction": 0.25,
            "seeds": {
                "dataset_seed": 42,
                "mask_seed": 2501,
                "donor_seed": 3501,
                "rand_pos_seed": 4501,
                "within_shuff_seed": 5501
            }
        },
        "tiny_results": {
            "manifest": tiny_res["manifest"],
            "depth_results": tiny_res["depth_results"],
            "validations": tiny_validations
        },
        "small_results": {
            "manifest": small_res["manifest"],
            "depth_results": small_res["depth_results"],
            "validations": small_validations
        },
        "decision": decision_res
    }

    manifest_path = os.path.join(args.output_dir, "results_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    decision_path = os.path.join(args.output_dir, "decision_summary.json")
    with open(decision_path, "w") as f:
        json.dump(decision_res, f, indent=2)

    # Save image-level data to parquet
    tiny_df.to_parquet(os.path.join(args.output_dir, "image_records_tiny.parquet"), index=False)
    small_df.to_parquet(os.path.join(args.output_dir, "image_records_small.parquet"), index=False)
    print(f"\nSaved machine-readable records to {args.output_dir}/")

    # 7. Generate Figures
    print("\n>>> Generating publication figures...")
    plot_all_fungibility_figures(manifest, output_dir=args.figures_dir)

    # 8. Print Formatted Results Table
    print("\n==========================================================================================")
    print("SUMMARY RESULTS TABLE: FUNGIBILITY GAP ACROSS DEPTH")
    print("==========================================================================================")
    header = f"{'Model':<12} | {'Depth':<6} | {'Zero Dmg':<9} | {'Cross Dmg':<9} | {'Gap Mean':<9} | {'95% CI':<18} | {'dz':<6} | {'BH-FDR q':<10}"
    print(header)
    print("-" * len(header))
    for m_name, res in [("DeiT-Tiny", tiny_res), ("DeiT-Small", small_res)]:
        for d in args.depths:
            d_res = res["depth_results"][d]
            z_dmg = d_res["damage"]["zero"]["mean"]
            c_dmg = d_res["damage"]["cross_same_pos"]["mean"]
            gap = d_res["fungibility_gap"]
            ci_str = f"[{gap['bootstrap_ci_95'][0]:.3f}, {gap['bootstrap_ci_95'][1]:.3f}]"
            print(f"{m_name:<12} | {d:<6} | {z_dmg:<9.3f} | {c_dmg:<9.3f} | {gap['mean']:<9.3f} | {ci_str:<18} | {gap['cohens_dz']:<6.3f} | {gap['fdr_p_value']:<10.4e}")
    print("==========================================================================================")


if __name__ == "__main__":
    main()
