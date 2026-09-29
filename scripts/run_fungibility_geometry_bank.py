"""
scripts/run_fungibility_geometry_bank.py

Master execution script for:
FUNGIBILITY-TO-COMPRESSION POC: GEOMETRY-DIVERSITY BANK POC
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from patch_fungibility.geometry_bank_pipeline import run_geometry_bank_experiment
from scripts.plot_fungibility_geometry_bank import generate_geometry_bank_plots

def main():
    parser = argparse.ArgumentParser(description="Run Geometry-Diversity Bank POC")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick smoke test on 64 images")
    parser.add_argument("--output_dir", type=str, default="outputs/fungibility_geometry_bank")
    parser.add_argument("--fig_dir", type=str, default="figures/fungibility_geometry_bank")
    args = parser.parse_args()

    print("=================================================================")
    print("STARTING EXPERIMENT: GEOMETRY-DIVERSITY BANK POC")
    print(f"Output Directory: {args.output_dir}")
    print(f"Figures Directory: {args.fig_dir}")
    print(f"Smoke Test: {args.smoke_test}")
    print("=================================================================")

    # 1. Run pipeline
    results = run_geometry_bank_experiment(
        output_dir=args.output_dir,
        smoke_test=args.smoke_test
    )

    # 2. Generate plots
    print("\nGenerating publication figures...")
    generate_geometry_bank_plots(
        data_dir=args.output_dir,
        fig_dir=args.fig_dir
    )

    print("\n=================================================================")
    print(f"EXPERIMENT COMPLETED! Verdict: {results['verdict']}")
    print("=================================================================")

if __name__ == "__main__":
    main()
