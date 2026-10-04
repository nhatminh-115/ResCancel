"""
scripts/run_residual_carrier_experiment.py

Master execution script for:
IMAGE-CONDITIONED GEOMETRY-COMPATIBLE RESIDUAL CARRIER POC
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

from patch_fungibility.residual_carrier_pipeline import run_residual_carrier_experiment
from scripts.plot_residual_carrier_figures import generate_residual_carrier_plots


def main():
    parser = argparse.ArgumentParser(description="Run Image-Conditioned Residual Carrier POC")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick smoke test on 64 images")
    parser.add_argument("--output_dir", type=str, default="outputs/fungibility_residual_carrier")
    parser.add_argument("--fig_dir", type=str, default="figures/fungibility_residual_carrier")
    args = parser.parse_args()

    print("=================================================================")
    print("STARTING EXPERIMENT: IMAGE-CONDITIONED RESIDUAL CARRIER POC")
    print(f"Output Directory: {args.output_dir}")
    print(f"Figures Directory: {args.fig_dir}")
    print(f"Smoke Test: {args.smoke_test}")
    print("=================================================================")

    # 1. Run pipeline
    results = run_residual_carrier_experiment(
        output_dir=args.output_dir,
        smoke_test=args.smoke_test
    )

    # 2. Generate publication plots
    print("\nGenerating publication figures...")
    generate_residual_carrier_plots(
        data_dir=args.output_dir,
        fig_dir=args.fig_dir
    )

    print("\n=================================================================")
    print(f"EXPERIMENT COMPLETED! Verdict:\n{results['verdict']}")
    print("=================================================================")


if __name__ == "__main__":
    main()
