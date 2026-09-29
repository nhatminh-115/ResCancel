"""
scripts/run_fungibility_compression.py

Master execution script for:
FUNGIBILITY-TO-COMPRESSION POC: WEIGHTED CENTROID CARRIER

Runs:
- Phase A: Exact Numerical Equivalence Validation (N=64 images)
- Phase B: Full 1,000-Image Evaluation across 5 Mask Seeds
- Phase C: Matched-Budget Baseline Comparisons
- Phase D: Theoretical FLOP Accounting & Physical CUDA Latency Benchmarking
- Publication figure generation

Usage:
  python scripts/run_fungibility_compression.py --smoke-test
  python scripts/run_fungibility_compression.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import time
import json
import torch

from patch_fungibility.compression_pipeline import run_compression_poc_pipeline
from scripts.plot_fungibility_compression import generate_all_compression_figures


def main():
    parser = argparse.ArgumentParser(description="Run Fungibility-to-Compression POC: Weighted Centroid Carrier")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test on 64 images")
    parser.add_argument("--output-dir", type=str, default="outputs/fungibility_compression_poc", help="Output directory")
    parser.add_argument("--figures-dir", type=str, default="figures/fungibility_compression_poc", help="Figures directory")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    args = parser.parse_args()

    print("================================================================================")
    print("FUNGIBILITY-TO-COMPRESSION POC: WEIGHTED CENTROID CARRIER")
    print(f"Mode: {'SMOKE TEST (64 images)' if args.smoke_test else 'FULL PRODUCTION RUN (1,000 images, 5 mask seeds)'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Figures directory: {args.figures_dir}")
    print("================================================================================")

    t_start = time.time()
    results = run_compression_poc_pipeline(
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
        device_str=args.device
    )

    print("\n[Master Script] Generating publication figures...")
    generate_all_compression_figures(args.output_dir, args.figures_dir)

    total_time = time.time() - t_start
    print(f"\n[Master Script] Full execution finished in {total_time:.1f}s ({total_time / 60:.2f}m)")

    val = results["validation"]
    print("\n========================= SCIENTIFIC DECISION =========================")
    print(f"OUTCOME:        {val['outcome']}")
    print(f"CLASSIFICATION: {val['classification']}")
    print(f"INTERPRETATION: {val['interpretation']}")
    print("========================================================================")


if __name__ == "__main__":
    main()
