#!/usr/bin/env python3
"""
scripts/run_joint_stream_geometry.py

Execution script for the Fungibility Joint Stream Geometry Experiment:
Feature Direction x Token-Space Pattern mapping under strict Frobenius norm matching.
"""

import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from patch_fungibility.joint_stream_geometry import run_joint_stream_geometry_experiment


def main():
    print("=================================================================")
    print("FUNGIBILITY JOINT STREAM GEOMETRY")
    print("Feature Direction x Token-Space Pattern (Strict Frobenius Matching)")
    print("=================================================================")
    manifest = run_joint_stream_geometry_experiment()
    print("\nJoint stream geometry mapping successfully finished!")


if __name__ == "__main__":
    main()
