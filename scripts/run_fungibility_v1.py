"""
scripts/run_fungibility_v1.py

Master execution script for PATCH FUNGIBILITY V1:
Cross-Model and Training-Regime Generalization (ViT-B AugReg & DINOv2 ViT-S/14).

Usage:
  python scripts/run_fungibility_v1.py --smoke-test
  python scripts/run_fungibility_v1.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import time
import json
import torch

from patch_fungibility.v1_pipeline import run_full_v1_evaluation
from scripts.plot_fungibility_v1_figures import generate_all_v1_figures


def main():
    parser = argparse.ArgumentParser(description="Run Patch Fungibility V1 Replication & Generalization")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test on 64 images")
    parser.add_argument("--output-dir", type=str, default="outputs/fungibility_v1", help="Output directory")
    parser.add_argument("--figures-dir", type=str, default="figures/fungibility_v1", help="Figures directory")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    args = parser.parse_args()

    print("================================================================================")
    print("PATCH FUNGIBILITY V1: CROSS-MODEL AND TRAINING-REGIME GENERALIZATION")
    print(f"Mode: {'SMOKE TEST (64 images)' if args.smoke_test else 'FULL PRODUCTION RUN (1,000 images)'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Figures directory: {args.figures_dir}")
    print("================================================================================")

    t_start = time.time()
    results = run_full_v1_evaluation(
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
        device_str=args.device
    )

    print("\n[Master Script] Generating publication figures...")
    generate_all_v1_figures(args.output_dir, args.figures_dir)

    total_time = time.time() - t_start
    print(f"\n[Master Script] Total elapsed time: {total_time:.1f}s ({total_time / 60:.2f}m)")

    # Print summary
    dec = results["decision"]
    print("\n========================= V1 FINAL SCIENTIFIC VERDICT =========================")
    print(f"DECISION: {dec['outcome']}")
    print(f"CLAIM:    {dec['supported_level']}")
    print(f"STATUS:   {'READY FOR PAPER CONSTRUCTION' if dec['ready_for_paper'] else 'NOT READY'}")
    print(f"SUMMARY:  {dec['interpretation']}")
    print("================================================================================")


if __name__ == "__main__":
    main()
