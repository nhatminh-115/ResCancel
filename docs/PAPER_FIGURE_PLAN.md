# Publication Figure & Table Plan: Patch Content Fungibility

**Repository:** `https://github.com/nhatminh-115/ResCancel`  
**Date:** 2026-09-30  
**Status:** Final submission-facing figure map  
**Style guide:** `docs/PAPER_FIGURE_STYLE.md`

## 1. Main manuscript

The main paper uses **6 figures + 1 synthesis table**. All submission-facing figures are regenerated from frozen machine-readable outputs by `scripts/render_paper_figures_final.py` and stored under `figures/paper_final/`.

### Figure 1 — Conceptual overview
**Path:** `figures/paper_final/main/fig01_conceptual_overview.png`  
**Role:** Define the activation-substitution estimand and summarize the three boundaries of the result: late content tolerance, geometry/diversity constraints, and replaceability versus compression. The schematic explicitly keeps upstream image processing, [CLS], token slots, sequence length, model weights, and downstream blocks fixed.

### Figure 2 — Depth emergence
**Path:** `figures/paper_final/main/fig02_depth_emergence.png`  
**Role:** Show the 25% replacement intervention across DeiT-Tiny, DeiT-Small, ViT-B/16 AugReg, and DINOv2 ViT-S/14. The claim is a depth-dependent transition into a late replacement-tolerant regime; no universal block index is claimed.

### Figure 3 — Dense fraction dose-response
**Path:** `figures/paper_final/main/fig03_dense_fraction.png`  
**Role:** Show the continuous 0–100% replacement sweep at the primary depth of each model, with variability across five independently sampled image-independent spatial masks. This supports robustness across the **tested masks**, not spatial invariance.

### Figure 4 — Geometry constraint
**Path:** `figures/paper_final/main/fig04_geometry_constraint.png`  
**Role:** Compare the aligned calibration centroid against coordinate permutation, sign inversion, and zero replacement. The conclusion is feature-space compatibility with learned coordinates/orientation, not full-manifold recovery.

### Figure 5 — Diversity constraint
**Path:** `figures/paper_final/main/fig05_diversity_constraint.png`  
**Role:** Compare static centroid, shared Gaussian, and independent Gaussian under complete patch-stream replacement. Accuracy rescue is strong in CLS-readout models; DINOv2 is represented honestly by its positive true-class-margin gain at an accuracy floor.

### Figure 6 — Direction-specific low-dimensional variation
**Path:** `figures/paper_final/main/fig06_lowdim_direction.png`  
**Role:** Compare static centroid, natural PC1/PC2, and matched random one-dimensional variation. The claim is that useful low-dimensional variation is **direction-specific and amplitude-dependent**, not that the natural representation is intrinsically rank-one.

### Table 1 — Cross-architecture synthesis
**Role:** Summarize architecture, training regime, readout topology, clean performance, primary depth, centroid recovery, geometry effect, diversity effect, and learned-1D evidence for all four models.

## 2. Supplementary figures

1. **S1 — Mask-seed robustness:** `figures/paper_final/supp/figS01_mask_robustness.png`
2. **S2 — Accuracy-retention thresholds:** `figures/paper_final/supp/figS02_retention_thresholds.png`
3. **S3 — Natural vs energy-matched PCA rank:** `figures/paper_final/supp/figS03_pca_rank.png`
4. **S4 — PC1 amplitude sensitivity:** `figures/paper_final/supp/figS04_pc1_amplitude.png`
5. **S5 — Effective-rank propagation:** `figures/paper_final/supp/figS05_rank_propagation.png`
6. **S6 — Exact multiplicity-aware carrier equivalence:** `figures/paper_final/supp/figS06_carrier_equivalence.png`
7. **S7 — Accuracy vs downstream token count:** `figures/paper_final/supp/figS07_accuracy_vs_tokens.png`
8. **S8 — Accuracy vs measured GPU latency:** `figures/paper_final/supp/figS08_accuracy_vs_latency.png`
9. **S9 — Synthetic geometry-bank delta vs random pruning:** `figures/paper_final/supp/figS09_geometry_bank_vs_pruning.png`

The compression figures are deliberately supplementary. They establish the boundary that exact representational replaceability and duplicate-state collapse do not imply a competitive new acceleration method.

## 3. Rendering policy

Historical experiment figures remain in their original directories for auditability. The manuscript must reference only `figures/paper_final/main/` and `figures/paper_final/supp/`. Final plots use the same typography, semantic colors, line weights, panel labels, and spacing. Each plot is emitted as a 400-DPI PNG and vector PDF; the current arXiv source package uses the PNG set for portable compilation.
