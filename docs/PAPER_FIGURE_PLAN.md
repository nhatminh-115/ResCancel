# Publication Figure & Table Plan: Patch Content Fungibility

**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Date**: 2026-09-29  
**Goal**: Design a compact, visually compelling main manuscript (5 figures, 1 master table) and a comprehensive supplementary appendix.

---

## 1. Main Manuscript Figures

### Figure 1: Conceptual Paradigm of Patch Content Fungibility
- **Type**: Schematic / Conceptual Diagram.
- **Visual Structure**:
  - **Panel A (Original Forward Pass)**: Input image passing through early ViT layers; patch tokens carry rich spatial features $h_{l,t}$.
  - **Panel B (Destructive Null Control - Zero)**: In late layers (e.g., Depth 8), spatial patches are set to $0 \to$ severe disruption to downstream attention and classification failure.
  - **Panel C (Content Fungibility - Aligned Centroid / Gaussian)**: Spatial patches are replaced by a class-agnostic prototype $\mu_l$ or coarse Gaussian $\mathcal{N}(\mu_l, \sigma_l^2) \to$ classification is preserved ($>81\% - 97\%$ recovery), demonstrating dispensability of exact image content.
  - **Panel D (Dual Constraints)**:
    - *Geometry Constraint*: Permuted coordinates $\pi(\mu)$ or inverted signs $-\mu$ collapse computation.
    - *Diversity Constraint*: Broadcasting identical tokens (Shared Noise) collapses spatial Key/Value rank; Independent diversity restores dynamic attention.
- **Target Location**: Section 1 (Page 1 or 2).

---

### Figure 2: Emergence of Content Fungibility Across Network Depth
- **Source File**: [`figures/fungibility_v1/depth_generalization.png`](file:///d:/Study/ResCancel/figures/fungibility_v1/depth_generalization.png) (supplemented with DeiT curves from V0.6).
- **Structure**: 2x2 grid (or 4-column horizontal banner):
  - Subplots for DeiT-Tiny, DeiT-Small, ViT-B AugReg, and DINOv2 ViT-S/14.
  - X-axis: Transformer Block Depth ($l \in \{5, 6, 7, 8, 9, 10, 11\}$).
  - Y-axis: Top-1 Accuracy (%) and Logit Margin ($m$).
  - Curves:
    - Clean Reference (dashed horizontal line).
    - Zero Replacement (red line with circular markers).
    - Held-out Calibration Centroid $\mu_l$ (blue line with square markers).
    - Held-out Diagonal Gaussian $\mathcal{N}(\mu_l, \sigma_l^2)$ (purple band with error bars).
- **Key Takeaway**: Early blocks show strong exact-content dependence; a distinct transition occurs across intermediate layers into a late replacement-tolerant window (Blocks 7–10).
- **Target Location**: Section 4 (Page 4).

---

### Figure 3: Geometric Feature Constraints: Feature-Space Alignment vs. Negative Controls
- **Source File**: [`figures/fungibility_v1/geometry_controls_across_models.png`](file:///d:/Study/ResCancel/figures/fungibility_v1/geometry_controls_across_models.png).
- **Structure**: Multi-panel plot comparing replacement fractions ($f \in \{25\%, 50\%, 75\%\}$):
  - X-axis: Replaced Spatial Patch Fraction (%).
  - Y-axis: Top-1 Accuracy (%) / Margin Damage ($\Delta m$).
  - Conditions compared:
    - True Calibration Centroid $\mu_8$ (blue line/squares).
    - Coordinate-Permuted Centroid $\pi(\mu_8)$ (orange line/diamonds with seed spread).
    - Sign-Flipped Centroid $-\mu_8$ (pink dotted line/crosses).
    - Zero Control (gray dashed line).
- **Key Takeaway**: Preserving activation scale, mean, and variance is insufficient; destroying coordinate correspondence or reversing orientation causes catastrophic collapse, proving downstream attention is strictly geometry-constrained.
- **Target Location**: Section 5 (Page 5).

---

### Figure 4: The Causal Role of Spatial Diversity Under Complete (100%) Patch Replacement
- **Source File**: [`figures/fungibility_v1/shared_vs_independent_across_models.png`](file:///d:/Study/ResCancel/figures/fungibility_v1/shared_vs_independent_across_models.png).
- **Structure**: Comparative bar chart with error bars across all four models:
  - Conditions per model:
    - Static Centroid (all patches identical $\mu_8$).
    - Shared Gaussian Noise (sampled once per image, broadcast to all patches).
    - Independent Gaussian Noise (independently sampled per patch).
    - Isotropic Independent Noise (matched total variance energy $E_{\text{full}}$).
- **Key Takeaway**: Shared noise collapses classification; independent noise dramatically rescues performance (+11.7% to +37.1% accuracy gain in supervised models; +1.12 margin gain in DINOv2).
- **Target Location**: Section 6 (Page 6).

---

### Figure 5: Low-Dimensional Representation Dynamics: Learned PC1 vs. Arbitrary Directions
- **Source File**: [`figures/fungibility_v1/learned_vs_random_1d.png`](file:///d:/Study/ResCancel/figures/fungibility_v1/learned_vs_random_1d.png).
- **Structure**: Grouped bar chart comparing single-direction variation at 100% spatial replacement:
  - Conditions:
    - Static Centroid baseline.
    - Natural PC1 ($v_1$, variance $\lambda_1$).
    - Natural PC2 ($v_2$, variance $\lambda_2$).
    - Energy-Matched Random 1D Direction ($u \in \mathbb{S}^{D-1}$, variance $\lambda_1$).
- **Key Takeaway**: Natural PC1 variation alone drives accuracy to $40.1\%$ in ViT-B (vs. $6.7\%$ for Random 1D), demonstrating that downstream computation is specifically receptive to variation aligned with learned late-layer covariance axes.
- **Target Location**: Section 7 (Page 7).

---

## 2. Main Manuscript Table

### Table 1: Cross-Architecture and Training-Regime Synthesis Table
- **Content**: Master synthesis comparing DeiT-Tiny, DeiT-Small, Supervised ViT-B AugReg, and Self-Supervised DINOv2 ViT-S/14.
- **Columns**: Model, Architecture Capacity ($D$), Spatial Token Count ($N$), Training Paradigm, Readout Topology, Clean Baseline Acc/Margin, Fungibility Window, Zero Sensitivity, Centroid Recovery Rate, Geometry Constraint ($d_z$), Diversity Advantage ($\Delta \text{Acc} / \Delta m$), Learned 1D Advantage.
- **Target Location**: Section 8 (Page 8).

---

## 3. Supplementary Appendix Figures & Tables

1. **Appendix Figure S1: Replacement Fraction Sweep to 100%**
   - Source: [`figures/fungibility_v1/replacement_fraction_generalization.png`](file:///d:/Study/ResCancel/figures/fungibility_v1/replacement_fraction_generalization.png).
   - Continuous curves from 25% to 100% replacement showing the transition where static prototypes collapse and stochastic diversity takes over.
2. **Appendix Figure S2: Downstream Rank Propagation Across Blocks**
   - Plots of effective representation rank $r_{\text{eff}}$ across Blocks 8, 9, 10, 11 (from V0.9), confirming the rank-preserving property.
3. **Appendix Figure S3: PC1 Amplitude Sensitivity Sweeps**
   - Response curves for scale multipliers $s \in \{0.25, 0.5, 1.0, 2.0, 4.0\}$ along PC1, showing Tiny's inverted U-curve and Small's saturating plateau.
4. **Appendix Figure S4: Attention Map Entropy and Spatial Key/Value Variance**
   - Diagnostic heatmaps demonstrating spatial attention collapse under Shared noise and its restoration under Independent noise.
5. **Appendix Table S1: Complete Statistical Test Matrix**
   - Full reporting of paired $t$-tests, Wilcoxon signed-rank tests, Cohen's $d_z$, McNemar exact tests, and Benjamini-Hochberg FDR $q$-values for all tested conditions.
