"""
scripts/run_fungibility_dense_fraction.py

Master execution script for:
PATCH FUNGIBILITY — DENSE FRACTION / MASK ROBUSTNESS SWEEP

Runs evaluation across 4 frozen models, 5 independent spatial mask seeds,
and 101-point deduplicated fraction grid, followed by figure generation.

Usage:
  python scripts/run_fungibility_dense_fraction.py --smoke-test
  python scripts/run_fungibility_dense_fraction.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import time
import json
import torch

from patch_fungibility.dense_fraction_pipeline import run_dense_fraction_sweep
from scripts.plot_fungibility_dense_fraction import generate_all_dense_fraction_figures


def main():
    parser = argparse.ArgumentParser(description="Run Patch Fungibility Dense Fraction / Mask Robustness Sweep")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test (64 images, 5% grid)")
    parser.add_argument("--output-dir", type=str, default="outputs/fungibility_dense_fraction", help="Output directory")
    parser.add_argument("--figures-dir", type=str, default="figures/fungibility_dense_fraction", help="Figures directory")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    args = parser.parse_args()

    print("================================================================================")
    print("PATCH FUNGIBILITY — DENSE FRACTION / MASK ROBUSTNESS SWEEP")
    print(f"Mode: {'SMOKE TEST (64 images, 5% grid)' if args.smoke_test else 'FULL PRODUCTION RUN (1,000 images, 1% grid)'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Figures directory: {args.figures_dir}")
    print("================================================================================")

    t_start = time.time()
    results = run_dense_fraction_sweep(
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
        device_str=args.device
    )

    print("\n[Master Script] Generating publication figures...")
    generate_all_dense_fraction_figures(args.output_dir, args.figures_dir)

    total_time = time.time() - t_start
    print(f"\n[Master Script] Full execution finished in {total_time:.1f}s ({total_time / 60:.2f}m)")

    dec = results["decision"]
    print("\n========================= SCIENTIFIC DECISION =========================")
    print(f"DECISION: {dec['overall_decision']}")
    print(f"INTERPRETATION: {dec['interpretation']}")
    print("========================================================================")


if __name__ == "__main__":
    main()
