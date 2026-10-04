#!/usr/bin/env python3
"""
scripts/run_predictive_principle.py

Execution script for the Patch Fungibility Predictive Principle Experiment:
Evaluates whether clean-state representation geometry and downstream sensitivity
predict patch-content fungibility prior to any replacement intervention.
"""

import sys
import os

# Ensure repo root is on python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from patch_fungibility.predictive_principle import run_predictive_principle_experiment


def main():
    print("=================================================================")
    print("FUNGIBILITY PREDICTIVE PRINCIPLE EXPERIMENT")
    print("Predicting ViT Layer Fungibility from Clean Geometry & Sensitivity")
    print("=================================================================")
    manifest = run_predictive_principle_experiment()
    print("\nExperiment execution successfully completed.")
    print(f"Final Decision: {manifest['verdict']}")


if __name__ == "__main__":
    main()
