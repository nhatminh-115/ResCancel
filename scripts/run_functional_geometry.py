#!/usr/bin/env python3
"""
scripts/run_functional_geometry.py

Execution script for the Functional Equivalence Geometry Experiment:
Maps the anisotropic geometry of patch-content fungibility in ViTs.
"""

import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from patch_fungibility.functional_geometry import run_functional_geometry_experiment


def main():
    print("=================================================================")
    print("FUNCTIONAL EQUIVALENCE GEOMETRY OF PATCH FUNGIBILITY")
    print("Mapping Anisotropy, Finite Tolerance Radii, and Metric Null-Spaces")
    print("=================================================================")
    manifest = run_functional_geometry_experiment()
    print("\nFunctional geometry mapping successfully finished!")


if __name__ == "__main__":
    main()
