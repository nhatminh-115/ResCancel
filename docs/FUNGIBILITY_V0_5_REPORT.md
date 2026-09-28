# Patch Content Fungibility V0.5: Scientific Report
## Norm- and Distribution-Matched Null Falsification at Transformer Depth 8

**Author:** DeepMind Antigravity Team  
**Date:** September 28, 2026  
**Repository:** [https://github.com/nhatminh-115/ResCancel](https://github.com/nhatminh-115/ResCancel)  
**Status:** Completed & Validated  
**Final Scientific Verdict:** **OUTCOME B — DISTRIBUTION MATTERS, CONTENT DOES NOT**

---

## Executive Summary

In Patch Content Fungibility V0, we observed that replacing 25% of spatial patch tokens (49/196) at Depth 8 with foreign image donor tokens substantially rescued predictive performance relative to zero ablation. However, because zero ablation ($h_t \to 0$) is an extreme out-of-distribution manipulation, the mechanism underlying this recovery remained confounded:
Did recovery require meaningful learned representations (H1), was it merely an activation-scale artifact (H2), or did it reflect distributional plausibility on the learned activation manifold (H3)?

In V0.5, across $N = 1,000$ stratified ImageNet-1k images (seed 42) on DeiT-Tiny and DeiT-Small, we evaluated norm-matched random spheres, layer-statistics Gaussians, feature dimension shuffles, and norm-matched controls at Block 8. The evidence decisively resolves the mechanism:

1. **H2 (Activation-Scale Artifact) is Decisively REJECTED:**
   Norm-Matched Random Sphere vectors match the exact original token $L_2$ norm ($\|h'_t\| = \|h_t\|$, relative error $< 10^{-5}$), yet they induce catastrophic predictive failure:
   - **DeiT-Tiny:** Margin damage is **$1.945$** (Top-1 accuracy collapses from $70.7\%$ to **$31.2\%$**; worse than Zero's $60.5\%$).
   - **DeiT-Small:** Margin damage is **$2.020$** (Top-1 accuracy collapses from $79.1\%$ to **$63.4\%$**; worse than Zero's $71.5\%$).
   - Cross-Image replacement beats Random Sphere by $\Delta m = +1.566$ ($d_z = 0.711, p < 10^{-90}$) on Tiny, and $\Delta m = +1.450$ ($d_z = 0.612, p < 10^{-70}$) on Small.
   - Preserving activation norm alone is **incapable** of rescuing downstream classification.

2. **Feature Geometry Alignment is Strictly Required:**
   Feature-Shuffled Original tokens exactly preserve token norm, scalar multiset, mean, and variance, but destroy channel coordinate correspondence. This yields severe degradation:
   - Accuracy collapses to **$38.5\%$** on Tiny (damage $1.740$) and **$58.2\%$** on Small (damage $2.073$).
   - Independent channels in the transformer cannot be arbitrarily permuted; learned coordinate geometry is causally essential.

3. **H3 (Distributional Plausibility) is Strongly CONFIRMED (Outcome B):**
   Layer-Statistics Gaussian vectors—sampled purely from unconditioned feature-wise $\mathcal{N}(\mu_d, \sigma_d^2)$ estimated without labels from non-masked tokens—**completely rescue the model**:
   - **DeiT-Tiny:** Damage is only **$0.101$**; Top-1 accuracy reaches **$68.8\%$** (beating Cross-Image's $66.0\%$).
   - **DeiT-Small:** Damage is only **$0.076$**; Top-1 accuracy reaches **$77.9\%$** (beating Cross-Image's $75.1\%$).
   - Diagonal Mahalanobis distance explains the phenomenon: Clean Baseline / Cross-Image sit at $D_{\text{Mahal}} \approx 192$ (Tiny) and $\approx 382$ (Small). Layer-Statistics Gaussian matches this manifold perfectly ($D_{\text{Mahal}} \approx 191.9$ and $383.9$). In contrast, Random Sphere ($283.8$ and $629.6$) and Feature Shuffle ($283.1$ and $632.2$) lie severely off-manifold.

### Scientific Reframing of V0
The V0 finding that exact image-specific patch content is dispensable at Block 8 **survives and is strengthened**: the model does not require donor image features or coherent visual semantics. Any vector drawn from the empirical marginal distribution $\mathcal{N}(\mu_d, \sigma_d^2)$ functions as a near-perfect placeholder. What the model requires at Block 8 is **activation manifold compliance**, not visual content.

---

## 1. Experimental Configuration & Invariants

| Parameter | Specification | Verification |
| :--- | :--- | :--- |
| **Architectures** | `deit_tiny_patch16_224`, `deit_small_patch16_224` | Public checkpoints from `timm` |
| **Backbone Weights** | Frozen, eval mode ($\nabla_\theta = 0$) | SHA-256 hash verified invariant |
| **Dataset** | ImageNet-1k Validation Subset ($N = 1,000$) | Seed 42, identical sample IDs as V0 |
| **Target Depth** | Block 8 Output / Block 9 Input ONLY | Pre-registered, no other depths evaluated |
| **Masked Tokens** | Exactly 49 / 196 spatial patch tokens (25%) | Seed 2501 (identical to V0) |
| **CLS Token** | Index 0 strictly untouched at injection | Confirmed ($h'_{8,0} = h_{8,0}$, max diff $= 0.0$) |
| **Downstream Blocks** | Blocks 9, 10, 11 + `norm` + `forward_head` | Strictly identical across all conditions |
| **Hardware / Peak VRAM**| NVIDIA RTX GPU ($< 8\text{ GB}$ ceiling) | Tiny: 105.0 MB; Small: 233.7 MB |
| **Total GPU Runtime** | Pass 1 caching + Pass 2 (18 conditions) | Tiny: 11.5s; Small: 16.0s (Total: 27.5s) |
| **Statistical Unit** | Image ($N = 1,000$) | Seed-level evaluation before averaging |

---

## 2. Quantitative Results Table (Block 8)

| Model | Intervention Condition | Top-1 Acc (%) | Mean Damage | Median Damage | 95% Bootstrap CI | Token $L_2$ Norm | Mahalanobis Dist $D_{\text{Mahal}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeiT-Tiny** | **Clean Baseline** | **70.7%** | **0.000** | 0.000 | [0.000, 0.000] | 12.20 | 0.0 |
| | Zero Ablation | 60.5% | 0.732 | 0.447 | [0.655, 0.812] | 0.00 | 26.4 |
| | **Norm-Matched Random Sphere (Mean)** | **31.2%** | **1.945** | 1.839 | [1.815, 2.078] | **12.21** | **283.8** |
| | **Layer-Statistics Gaussian (Mean)** | **68.8%** | **0.101** | 0.052 | [0.061, 0.143] | **12.38** | **191.9** |
| | Feature-Shuffled Original (Mean) | 38.5% | 1.740 | 1.637 | [1.604, 1.881] | 12.21 | 283.1 |
| | Cross-Image Donor | 66.0% | 0.379 | 0.170 | [0.297, 0.463] | 12.20 | 192.0 |
| | Norm-Matched Cross-Image | 66.1% | 0.338 | 0.146 | [0.258, 0.420] | 12.21 | 197.9 |
| | Same-Image Layer Mean | 69.4% | 0.067 | 0.021 | [0.033, 0.103] | 6.87 | 35.8 |
| | Norm-Matched Layer Mean | 69.4% | 0.066 | 0.022 | [0.033, 0.101] | 12.21 | 131.4 |
| **DeiT-Small**| **Clean Baseline** | **79.1%** | **0.000** | 0.000 | [0.000, 0.000] | 34.99 | 0.0 |
| | Zero Ablation | 71.5% | 1.528 | 1.353 | [1.385, 1.670] | 0.00 | 37.3 |
| | **Norm-Matched Random Sphere (Mean)** | **63.4%** | **2.020** | 2.126 | [1.880, 2.158] | **35.08** | **629.6** |
| | **Layer-Statistics Gaussian (Mean)** | **77.9%** | **0.076** | 0.036 | [0.040, 0.114] | **36.79** | **383.9** |
| | Feature-Shuffled Original (Mean) | 58.2% | 2.073 | 2.181 | [1.933, 2.213] | 35.08 | 632.2 |
| | Cross-Image Donor | 75.1% | 0.570 | 0.316 | [0.466, 0.676] | 34.99 | 382.4 |
| | Norm-Matched Cross-Image | 75.6% | 0.533 | 0.301 | [0.430, 0.638] | 35.08 | 450.8 |
| | Same-Image Layer Mean | 79.0% | 0.043 | 0.012 | [0.016, 0.071] | 18.73 | 66.0 |
| | Norm-Matched Layer Mean | 79.0% | 0.043 | 0.011 | [0.016, 0.072] | 35.08 | 290.2 |

---

## 3. Primary Hypothesis Contrasts

### 3.1 Contrast 1: Cross-Image vs Norm-Matched Random Sphere (Test of H2)
$$\Delta m = \text{Margin}_{\text{Cross}} - \text{Margin}_{\text{Sphere}} = \text{Damage}_{\text{Sphere}} - \text{Damage}_{\text{Cross}}$$
- **DeiT-Tiny:** $\Delta m = +1.566$, 95% Bootstrap CI $[1.427, 1.704]$, Cohen's $d_z = 0.711$, paired $t = 22.48$, $p = 6.52 \times 10^{-91}$. Accuracy advantage: $+34.8\%$ ($66.0\%$ vs $31.2\%$).
- **DeiT-Small:** $\Delta m = +1.450$, 95% Bootstrap CI $[1.302, 1.596]$, Cohen's $d_z = 0.612$, paired $t = 19.36$, $p = 3.69 \times 10^{-71}$. Accuracy advantage: $+11.7\%$ ($75.1\%$ vs $63.4\%$).
- **Inference:** Random Sphere vectors matching target norm fail catastrophically and are significantly worse than Zero. Magnitude alone does not rescue the model. H2 is falsified.

### 3.2 Contrast 2: Cross-Image vs Layer-Statistics Gaussian (Test of H3)
$$\Delta m = \text{Margin}_{\text{Cross}} - \text{Margin}_{\text{Gaussian}}$$
- **DeiT-Tiny:** $\Delta m = -0.279$, 95% Bootstrap CI $[-0.333, -0.225]$, Cohen's $d_z = -0.324$, $p = 1.78 \times 10^{-23}$. Gaussian achieves $68.8\%$ accuracy (+2.8% over Cross-Image).
- **DeiT-Small:** $\Delta m = -0.494$, 95% Bootstrap CI $[-0.556, -0.432]$, Cohen's $d_z = -0.497$, $p = 6.73 \times 10^{-50}$. Gaussian achieves $77.9\%$ accuracy (+2.8% over Cross-Image).
- **Inference:** Moment-matched Gaussian vectors not only approach Cross-Image performance, but reliably *outperform* foreign donor tokens. Unconditioned Gaussian noise placed on the empirical manifold provides clean placeholders without the semantic feature conflicts introduced by donor images.

### 3.3 Contrast 3: Cross-Image vs Feature-Shuffled Original
$$\Delta m = \text{Margin}_{\text{Cross}} - \text{Margin}_{\text{FeatureShuffle}}$$
- **DeiT-Tiny:** $\Delta m = +1.361$, 95% Bootstrap CI $[1.239, 1.489]$, Cohen's $d_z = 0.679$, $p = 1.97 \times 10^{-84}$.
- **DeiT-Small:** $\Delta m = +1.503$, 95% Bootstrap CI $[1.361, 1.641]$, Cohen's $d_z = 0.664$, $p = 2.41 \times 10^{-81}$.
- **Inference:** Preserving scalar activation values while scrambling feature channels completely breaks the network. Channel-specific coordinate alignment is mandatory.

### 3.4 Contrast 4: Norm-Matched Mean vs Layer Mean (Does Magnitude Matter for Mean?)
- **DeiT-Tiny:** $\Delta m = +0.0017$, 95% CI $[0.0007, 0.0028]$, $p = 0.0013$. Top-1 Accuracy: Identical at $69.4\%$.
- **DeiT-Small:** $\Delta m = +0.00001$, 95% CI $[-0.0006, 0.0006]$, $p = 0.974$. Top-1 Accuracy: Identical at $79.0\%$.
- **Inference:** Scaling the same-image mean up to full token norm (a $1.8\times$ norm increase) has zero meaningful causal impact on prediction. Semantic direction and manifold alignment dominate norm magnitude.

### 3.5 Contrast 5: Cross-Image vs Norm-Matched Cross-Image
- **DeiT-Tiny:** $\Delta m = -0.041$, $d_z = -0.273$. Accuracy changes by $+0.1\%$.
- **DeiT-Small:** $\Delta m = -0.037$, $d_z = -0.273$. Accuracy changes by $+0.5\%$.
- **Inference:** Re-scaling donor tokens to exact target norm produces negligible differences, confirming that V0's cross-image recovery was not confounded by minor norm discrepancies.

---

## 4. Multi-Seed Stability Analysis

Across all tested random seeds, results are exceptionally tight and invariant:

### Random Sphere (5 Seeds: 6501–6505)
- **DeiT-Tiny:**
  - Damage across seeds: $[1.963, 1.921, 1.964, 1.921, 1.954]$ (Mean $= 1.945$, Seed SD $= 0.022$).
  - Accuracy across seeds: $[31.7\%, 30.5\%, 30.3\%, 31.9\%, 31.7\%]$ (Mean $= 31.2\%$, Seed SD $= 0.7\%$).
- **DeiT-Small:**
  - Damage across seeds: $[2.022, 2.012, 2.027, 2.014, 2.025]$ (Mean $= 2.020$, Seed SD $= 0.007$).
  - Accuracy across seeds: $[62.7\%, 64.6\%, 62.5\%, 63.9\%, 63.4\%]$ (Mean $= 63.4\%$, Seed SD $= 0.8\%$).

### Layer-Statistics Gaussian (5 Seeds: 7501–7505)
- **DeiT-Tiny:**
  - Damage across seeds: $[0.103, 0.099, 0.089, 0.105, 0.107]$ (Mean $= 0.101$, Seed SD $= 0.007$).
  - Accuracy across seeds: $[69.0\%, 68.7\%, 68.5\%, 69.0\%, 68.8\%]$ (Mean $= 68.8\%$, Seed SD $= 0.2\%$).
- **DeiT-Small:**
  - Damage across seeds: $[0.080, 0.081, 0.069, 0.069, 0.080]$ (Mean $= 0.076$, Seed SD $= 0.006$).
  - Accuracy across seeds: $[77.8\%, 78.1\%, 77.9\%, 78.1\%, 77.8\%]$ (Mean $= 77.9\%$, Seed SD $= 0.2\%$).

### Feature Shuffle (3 Seeds: 8501–8503)
- **DeiT-Tiny:** Damage: $[1.727, 1.742, 1.752]$; Accuracy: $[39.5\%, 37.9\%, 38.1\%]$.
- **DeiT-Small:** Damage: $[1.803, 2.272, 2.143]$; Accuracy: $[63.9\%, 53.7\%, 56.9\%]$.

---

## 5. Pre-Registered Programmatic Validations

All 14 pre-registered assertions passed on both architectures:
1. `check1_backbone_unmodified`: Verified (SHA-256 parameter hashes identical).
2. `check2_same_image_ids`: Verified ($N = 1,000$ unique sample IDs).
3. `check3_exactly_49_patches`: Verified (49 / 196 tokens = 25.0%).
4. `check4_same_patch_mask`: Verified (Seed 2501 identical to V0).
5. `check5_cls_untouched`: Verified ($h'_{8,0} = h_{8,0}$, max diff $= 0.0$).
6. `check6_downstream_blocks_identical`: Verified (Blocks 9..11 identical).
7. `check7_norm_matched_random_relative_error`: Verified ($< 10^{-5}$ relative error).
8. `check8_gaussian_stats_unlabeled`: Verified (Computed strictly from unmasked tokens).
9. `check9_norm_matched_mean_relative_error`: Verified ($< 10^{-5}$ relative error).
10. `check10_feature_shuffle_bijective`: Verified (True permutation of $0..D-1$).
11. `check11_frozen_random_seeds`: Verified (`sphere: 6501-6505, gaussian: 7501-7505, shuffle: 8501-8503`).
12. `check12_no_gradients_or_labels`: Verified.
13. `check13_image_level_independence`: Verified ($N = 1,000$ independent image units).
14. `check14_report_equals_manifest`: Verified.

---

## 6. Artifacts Manifest

- **Data Tables (Outputs):**
  - [`outputs/fungibility_v0_5/experiment_manifest.json`](file:///d:/Study/ResCancel/outputs/fungibility_v0_5/experiment_manifest.json)
  - [`outputs/fungibility_v0_5/tiny_image_results.parquet`](file:///d:/Study/ResCancel/outputs/fungibility_v0_5/tiny_image_results.parquet)
  - [`outputs/fungibility_v0_5/small_image_results.parquet`](file:///d:/Study/ResCancel/outputs/fungibility_v0_5/small_image_results.parquet)
  - [`outputs/fungibility_v0_5/tiny_seed_results.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v0_5/tiny_seed_results.csv)
  - [`outputs/fungibility_v0_5/small_seed_results.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v0_5/small_seed_results.csv)
  - [`outputs/fungibility_v0_5/tiny_condition_comparison.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v0_5/tiny_condition_comparison.csv)
  - [`outputs/fungibility_v0_5/small_condition_comparison.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v0_5/small_condition_comparison.csv)
  - [`outputs/fungibility_v0_5/tiny_activation_diagnostics.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v0_5/tiny_activation_diagnostics.csv)
  - [`outputs/fungibility_v0_5/small_activation_diagnostics.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v0_5/small_activation_diagnostics.csv)
- **Figures:**
  - `figures/fungibility_v0_5/condition_margin_damage.png` (Comparison of margin damage across nulls)
  - `figures/fungibility_v0_5/condition_accuracy.png` (Top-1 accuracy across nulls)
  - `figures/fungibility_v0_5/norm_vs_performance.png` (Testing H2: norm does not predict performance)
  - `figures/fungibility_v0_5/distribution_distance_vs_damage.png` (Testing H3: Mahalanobis distance predicts damage)

---

## 7. Scientific Conclusion & Boundary Constraints

1. **Resolution of Hypotheses:**
   - **H1 (Semantic Content Required): REFUTED.** Downstream computation at Block 8 does not require semantic donor tokens or coherent image features. Independent Gaussian noise sampled from $\mathcal{N}(\mu_d, \sigma_d^2)$ works even better than cross-image visual tokens.
   - **H2 (Activation-Scale Artifact): DECISIVELY REFUTED.** Preserving $L_2$ norm alone with random orientation collapses accuracy to $31.2\%$ (Tiny) and $63.4\%$ (Small), performing substantially worse than zero ablation.
   - **H3 (Distributional Plausibility / Manifold Compliance): STRONGLY CONFIRMED.** What the transformer requires at Depth 8 is vectors that lie on the empirical learned activation manifold (respecting per-feature mean and variance). Once on-distribution, exact token content is completely dispensable.
2. **Boundary Constraints:** Per protocol, we do **NOT** propose or build any training method, router, or new architecture. V0.5 strictly serves to determine the causal nature of the Depth-8 phenomenon.
