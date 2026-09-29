# Patch Fungibility V1: Cross-Version Synthesis Audit & Discrepancy Log

**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Date**: 2026-09-29  
**Audit Purpose**: Reconcile cross-version DeiT values in the V1 synthesis table against canonical frozen machine-readable outputs and audited reports (V0.6–V0.9).

---

## 1. Summary of Identified Synthesis Errors

In the initial draft of [`docs/FUNGIBILITY_V1_REPORT.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_V1_REPORT.md), the cross-family comparison table contained several historical values that blended statistics from earlier exploratory iterations (V0/V0.1/V0.5) instead of strictly citing the canonical frozen outputs from V0.6, V0.7, V0.8, and V0.9.

Below is the exhaustive item-by-item audit of discrepancies, root causes, and corrected canonical values.

---

## 2. Discrepancy Matrix & Canonical Reconciliations

### Item 1: DeiT Clean Baseline Logit Margins
- **Discrepancy in V1 Draft**: Listed DeiT-Tiny clean margin as `+2.78` and DeiT-Small clean margin as `+3.32`.
- **Root Cause**: These numbers were taken from an earlier uncalibrated or exploratory split (V0/V0.1), rather than the canonical 1,000-image disjoint evaluation set established in V0.6 and preserved throughout V0.7–V0.9.
- **Canonical Source**:
  - [`docs/FUNGIBILITY_V0_7_REPORT.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_V0_7_REPORT.md) Section 1 (Lines 91–93).
  - [`docs/FUNGIBILITY_V0_8_REPORT.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_V0_8_REPORT.md) and [`docs/FUNGIBILITY_V0_9_REPORT.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_V0_9_REPORT.md).
  - Machine-readable files: [`outputs/fungibility_v0_8/shared_vs_independent_results.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v0_8/shared_vs_independent_results.csv).
- **Audit Verification**:
  - DeiT-Tiny Clean Margin: **`1.1681`** (Clean Accuracy: **`67.90%`**)
  - DeiT-Small Clean Margin: **`2.2423`** (Clean Accuracy: **`76.10%`**)
- **Correction Applied**: Updated synthesis table to reflect exact canonical values: Tiny `1.1681`, Small `2.2423`.

---

### Item 2: DeiT V0.8 Complete-Replacement Diversity (Shared vs. Independent)
- **Discrepancy in V1 Draft**: Listed Tiny as Shared `2.5%`, Independent `14.4%` (gain `+11.9%`); Small as Shared `3.8%`, Independent `38.2%` (gain `+34.4%`).
- **Root Cause**: These values were approximate figures recalled from an intermediate analysis script rather than the canonical multi-seed table.
- **Canonical Source**:
  - [`outputs/fungibility_v0_8/shared_vs_independent_results.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v0_8/shared_vs_independent_results.csv).
  - [`docs/FUNGIBILITY_V0_8_REPORT.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_V0_8_REPORT.md) Section 3.1.
- **Audit Verification (5 Seeds: 14001..14005)**:
  - **DeiT-Tiny**:
    - Static Centroid: **10.20%** (Margin: `-1.2640`)
    - Shared Noise Seeds: `[0.8%, 1.4%, 1.3%, 0.7%, 0.9%]` $\to$ Mean: **`1.02% ± 0.31%`** (Margin: `-2.9796`)
    - Independent Noise Seeds: `[26.0%, 26.7%, 26.3%, 26.1%, 26.6%]` $\to$ Mean: **`26.34% ± 0.30%`** (Margin: `-0.8261`)
    - Exact Diversity Advantage: **`+25.32 percentage points`** (Accuracy), **`+2.1535`** (Margin).
  - **DeiT-Small**:
    - Static Centroid: **17.00%** (Margin: `-1.4211`)
    - Shared Noise Seeds: `[9.6%, 9.3%, 9.0%, 8.9%, 9.6%]` $\to$ Mean: **`9.28% ± 0.32%`** (Margin: `-2.1793`)
    - Independent Noise Seeds: `[46.9%, 45.2%, 46.4%, 47.2%, 46.1%]` $\to$ Mean: **`46.36% ± 0.77%`** (Margin: `-0.0639`)
    - Exact Diversity Advantage: **`+37.08 percentage points`** (Accuracy), **`+2.1154`** (Margin).
- **Correction Applied**: Updated synthesis table to cite exact canonical multi-seed means: Tiny Shared `1.02%` $\to$ Indep `26.34%` (`+25.32 pp`); Small Shared `9.28%` $\to$ Indep `46.36%` (`+37.08 pp`).

---

### Item 3: DeiT Centroid and Gaussian Recovery Rates (Depth 8, 25% Mask)
- **Discrepancy in V1 Draft**: Listed Tiny Centroid Recovery as `96.6%` (Gauss `88.5%`), Small Centroid Recovery as `100.0%` (Gauss `93.0%`).
- **Root Cause**: These recovery rates were derived from V0.7 prototype comparisons or earlier V0.6 narrative summaries that used different baseline rounding.
- **Canonical Source**:
  - Directly computed from [`outputs/fungibility_v0_6/tiny_depth_fraction_summary.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v0_6/tiny_depth_fraction_summary.csv) (Row 12: Depth 8, 25%) and [`outputs/fungibility_v0_6/small_depth_fraction_summary.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v0_6/small_depth_fraction_summary.csv) (Row 12: Depth 8, 25%).
- **Audit Verification**:
  - **DeiT-Tiny (Depth 8, 25% Spatial Replacement)**:
    - $\text{Damage}_{\text{zero}} = 0.763182$
    - $\text{Damage}_{\text{centroid}} = 0.047629$ (Acc = 67.2%)
    - $\text{Damage}_{\text{gauss}} = 0.092684$ (Acc = 66.68%)
    - $\text{Recovery}_{\text{centroid}} = \frac{0.763182 - 0.047629}{0.763182} = \frac{0.715553}{0.763182} = \mathbf{93.76\%}$
    - $\text{Recovery}_{\text{gauss}} = \frac{0.763182 - 0.092684}{0.763182} = \frac{0.670498}{0.763182} = \mathbf{87.86\%}$
  - **DeiT-Small (Depth 8, 25% Spatial Replacement)**:
    - $\text{Damage}_{\text{zero}} = 1.431745$
    - $\text{Damage}_{\text{centroid}} = 0.035208$ (Acc = 76.1%)
    - $\text{Damage}_{\text{gauss}} = 0.075707$ (Acc = 75.28%)
    - $\text{Recovery}_{\text{centroid}} = \frac{1.431745 - 0.035208}{1.431745} = \frac{1.396537}{1.431745} = \mathbf{97.54\%}$
    - $\text{Recovery}_{\text{gauss}} = \frac{1.431745 - 0.075707}{1.431745} = \frac{1.356038}{1.431745} = \mathbf{94.71\%}$
- **Correction Applied**: Updated synthesis table to reflect exact formulaic calculations from V0.6 canonical data: Tiny Centroid `93.8%` (Gauss `87.9%`); Small Centroid `97.5%` (Gauss `94.7%`).

---

### Item 4: DeiT Low-Dimensional 1D Variation (PC1 vs. Random 1D)
- **Discrepancy in V1 Draft**: Synthesized numbers correctly from V0.9, but did not explicitly distinguish natural unscaled variance from the confounded V0.8 energy-matched numbers in the explanatory notes.
- **Canonical Source**:
  - [`outputs/fungibility_v0_9/pc_identity_results.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v0_9/pc_identity_results.csv).
  - [`outputs/fungibility_v0_9/random_direction_results.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v0_9/random_direction_results.csv).
  - [`docs/FUNGIBILITY_V0_9_REPORT.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_V0_9_REPORT.md) Section 6.
- **Audit Verification**:
  - **DeiT-Tiny**:
    - Natural PC1 ($\lambda_1 = 15.50$): **`18.64% ± 0.17%`** (Margin: `-1.5246`)
    - Random 1D (Energy matched to $\lambda_1$): **`10.80% ± 0.42%`** (Margin: `-1.3665`)
    - Advantage of Learned Natural PC1: **`+7.84 pp`** ($p = 1.65 \times 10^{-5}$).
  - **DeiT-Small**:
    - Natural PC1 ($\lambda_1 = 297.86$): **`28.08% ± 0.67%`** (Margin: `-0.3186`)
    - Random 1D (Energy matched to $\lambda_1$): **`20.78% ± 0.44%`** (Margin: `-1.1729`)
    - Advantage of Learned Natural PC1: **`+7.30 pp`** ($p = 1.1 \times 10^{-4}$), with a major margin advantage ($+0.8543$).
- **Correction Applied**: Confirmed V0.9 Natural unscaled PC1 is cited everywhere as the canonical paper number, rejecting the V0.8 energy-matched Rank-1 artifact ($46.18\%$ in Small) as an over-scaled confound.

---

### Item 5: DINOv2 Token Diversity Framing
- **Discrepancy in V1 Draft**: The narrative stated that "Independent Gaussian noise rescues classification in DINOv2", which was inaccurate because Top-1 accuracy in DINOv2 remains at floor ($0.12\%$ vs $0.24\%$) under 100% spatial patch replacement.
- **Root Cause**: Over-generalizing the diversity benefit from margin space to accuracy space. DINOv2's 1-layer classification head directly consumes $[\text{CLS}_{\text{norm}} \ ; \ \text{mean}(\text{Patch}_{\text{norm}})]$. When 100% of spatial patches are replaced, the second half of the head's input is perturbed, preventing Top-1 accuracy recovery even though block-level attention dynamics improve.
- **Canonical Source**:
  - [`outputs/fungibility_v1/dinov2_diversity_results.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v1/dinov2_diversity_results.csv).
  - Protocol pre-registration rule in [`docs/FUNGIBILITY_V1_PROTOCOL.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_V1_PROTOCOL.md) Section 4.3.
- **Audit Verification**:
  - Shared Gaussian (5 seeds): Accuracy = `0.24% ± 0.17%`, Margin = `-9.077 ± 0.28`.
  - Independent Gaussian (5 seeds): Accuracy = `0.12% ± 0.04%`, Margin = `-7.953 ± 0.07`.
  - Margin Advantage: **`+1.124`**, Cohen's $d_z = \mathbf{0.301} \ge 0.20$.
- **Correction Applied**: Downgraded DINOv2 diversity claim. Replaced "rescues classification" with:  
  *"Independent token variation improves downstream compatibility at the margin level under complete DINOv2 patch-stream replacement (+1.12 margin gain, $d_z = 0.301$), although Top-1 accuracy remains at floor because the official linear classifier directly consumes the final spatial patch mean."*  
  DINOv2 is now properly presented as margin-level supporting evidence, while DeiT-Tiny, DeiT-Small, and ViT-B provide the strong accuracy-level evidence.

---

## 3. Impact on Experimental Verdict

Does the correction of these historical cross-version synthesis values alter the V1 verdict?

**NO.**  
- **V1 Outcome A (Broad Generalization) remains 100% empirically valid**:
  - ViT-B AugReg satisfies Signatures A, B, and C with massive margins (81.8% centroid recovery, sign-flip $d_z=0.335$, diversity gain $+11.74$ pp, $d_z=1.005$).
  - DINOv2 ViT-S/14 satisfies Signature A (93.4% recovery at Depth 9, $d_z > 2.0$), Signature B (sign-flip drops acc by $-74.5\%$, $d_z=2.509$), and Signature C at the pre-registered margin level ($+1.124$ margin gain, $d_z=0.301 \ge 0.20$).
- All 18 programmatic protocol assertions continue to pass without exception.
- The synthesis audit strengthens the paper's scientific integrity by replacing approximate narrative values with exact, audited machine-readable metrics.
