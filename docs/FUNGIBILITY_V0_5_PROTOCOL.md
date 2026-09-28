# Patch Content Fungibility V0.5: Protocol Specification
## Norm- and Distribution-Matched Null Falsification at Transformer Depth 8

**Author:** DeepMind Antigravity Team  
**Date:** September 28, 2026  
**Repository:** [https://github.com/nhatminh-115/ResCancel](https://github.com/nhatminh-115/ResCancel)  
**Status:** FROZEN BEFORE EXECUTION  

---

## 1. Scientific Context & Motivation

In Patch Content Fungibility V0, we mapped patch-token causal sensitivity across depths $l \in \{2, 4, 6, 8, 10\}$ in pretrained Vision Transformers (DeiT-Tiny and DeiT-Small) under a 25% spatial patch mask (49 out of 196 tokens). We discovered a replicated **Content-Fungible Regime** at Depth 8 (the output of Block 8, preceding Blocks 9–11):
- **DeiT-Tiny ($l=8$):** Baseline Acc = 70.7%, Zero = 60.5% (damage 0.732), Cross-Image = 66.0% (damage 0.379), Same-Image Mean = 69.4%.
- **DeiT-Small ($l=8$):** Baseline Acc = 79.1%, Zero = 71.5% (damage 1.528), Cross-Image = 75.1% (damage 0.570), Same-Image Mean = 79.0%.

However, **ZERO ablation ($h_t \to 0$) is an extreme out-of-distribution intervention**: it zeroes activation magnitude, completely disrupting LayerNorm dynamics, whereas cross-image and layer-mean replacements supply vectors with in-distribution scale and statistics.

This introduces a foundational mechanistic question:
> **Does the Depth-8 recovery require meaningful, learned representation structure, or can essentially ANY well-scaled replacement vector rescue the model relative to zero?**

### The Three Hypotheses:
1. **H1 — CONTENT FUNGIBILITY:** Exact patch-specific content is no longer required, but downstream computation still requires representations with meaningful learned feature structure and geometry.
2. **H2 — ACTIVATION-SCALE ARTIFACT:** Zero ablation is harmful primarily because its activation magnitude is zero. Any vector with approximately correct norm restores LayerNorm operating point and rescues prediction.
3. **H3 — DISTRIBUTIONAL PLAUSIBILITY:** Exact content is unnecessary, but replacement vectors must lie approximately on the learned activation distribution/manifold (matching first and second moments).

V0.5 is designed strictly to falsify H2 and H3 against H1 without training or architectural modifications.

---

## 2. Experimental Setup & Controls

- **Models:**
  - `deit_tiny_patch16_224` ($D = 192$, 12 blocks)
  - `deit_small_patch16_224` ($D = 384$, 12 blocks)
  - Public checkpoints from `timm`, strictly frozen weights ($\nabla_\theta = 0$, eval mode).
- **Dataset:**
  - $N = 1,000$ validation images from ImageNet-1k (stratified, 1 per class, seed 42).
  - Exact same sample IDs and preprocessing (Resize 256 bicubic $\to$ CenterCrop 224 $\to$ Normalize) as V0.
- **Intervention Depth:**
  - **Exclusively Block 8 output / Block 9 input (depth index 8 in 0-indexed terms)**.
  - No other depths are tested or reported.
- **Spatial Mask:**
  - Exactly 49 spatial patch tokens (25% of 196).
  - Deterministic frozen seed: `2501` (identical to V0).
- **CLS Token Invariant:**
  - Sequence index 0 is strictly untouched at injection: $h'_{8,0} = h_{8,0}$.
- **Downstream Blocks:**
  - Blocks 9, 10, 11 + `norm` + `forward_head` are strictly identical across all conditions.

---

## 3. Intervention Conditions

For each masked token slot $t \in \mathcal{M}$ (where $|\mathcal{M}| = 49$):

### Reference Conditions (from V0)
1. **A. Clean Baseline:**
   Untouched original representation $h_t$.
2. **B. Zero Ablation:**
   $h'_t = 0$. Extreme out-of-distribution baseline.
3. **C. Same-Image Layer Mean (Unscaled):**
   $\mu_i = \frac{1}{147} \sum_{s \notin \mathcal{M}} h_s$ from the SAME image. $h'_t = \mu_i$.
4. **D. Cross-Image Token (Unscaled):**
   $h'_t = c_t$, where $c_t$ is the activation at slot $t$ from donor image $j \neq i$ (deterministic derangement seed `3501`).

### Primary Null Conditions (New in V0.5)
5. **E. Norm-Matched Random Sphere (Primary Null for H2):**
   Sample $r \sim \mathcal{N}(0, I_D)$. Normalize $u = r / \|r\|_2$.
   Construct:
   $$h'_t = \|h_t\|_2 \cdot u$$
   - Each token independently receives a random direction on the unit sphere, scaled to **exactly match the $L_2$ norm of the original target token $h_t$**.
   - No label or gradient information used.
   - 5 frozen seeds: `6501, 6502, 6503, 6504, 6505`.
   - Core test: *Is correct activation magnitude alone sufficient to rescue the model?*

6. **F. Layer-Statistics Gaussian (Primary Null for H3):**
   For each model separately, estimate feature-wise mean $\mu_d$ and variance $\sigma_d^2$ across all $N=1,000$ images using **ONLY non-masked patch tokens** at Block 8:
   $$\mu_d = \frac{1}{N \cdot 147} \sum_{i=1}^N \sum_{s \notin \mathcal{M}} h_{i,s}[d], \quad \sigma_d^2 = \frac{1}{N \cdot 147} \sum_{i=1}^N \sum_{s \notin \mathcal{M}} (h_{i,s}[d] - \mu_d)^2$$
   Sample independently:
   $$h'_t[d] \sim \mathcal{N}(\mu_d, \sigma_d^2)$$
   - Matches the marginal first- and second-order empirical feature distribution without label information.
   - Token norm is NOT artificially forced, allowing empirical variance structure to dictate magnitude.
   - 5 frozen seeds: `7501, 7502, 7503, 7504, 7505`.
   - Core test: *Are first- and second-order feature statistics sufficient?*

7. **G. Norm-Matched Same-Image Layer Mean:**
   $$\mu_i = \frac{1}{147} \sum_{s \notin \mathcal{M}} h_s, \quad h'_t = \|h_t\|_2 \cdot \frac{\mu_i}{\|\mu_i\|_2}$$
   - Preserves the semantic direction of the target image's low-frequency background while restoring full token $L_2$ norm.
   - Distinguishes the semantic orientation of $\mu_i$ from its smaller norm ($\approx 50\%$ of token norm in V0).

8. **H. Feature-Shuffled Original:**
   Let $\pi: \{1, \dots, D\} \to \{1, \dots, D\}$ be a random permutation of feature channels.
   $$h'_t[d] = h_t[\pi(d)]$$
   - Exactly preserves token $L_2$ norm, scalar multiset, mean, and variance.
   - Completely destroys the learned semantic coordinate alignment between embedding dimensions.
   - 3 frozen seeds: `8501, 8502, 8503` (one fixed permutation per seed/model across all images and tokens).
   - Core test: *Does learned feature geometry and inter-channel coordination matter?*

9. **I. Norm-Matched Cross-Image Token:**
   $$h'_t = \|h_t\|_2 \cdot \frac{c_t}{\|c_t\|_2}$$
   - Re-scales the donor token $c_t$ to exactly match target $\|h_t\|_2$, testing whether residual donor-target norm mismatch affects cross-image performance.

---

## 4. Activation Diagnostics & Distributional Metrics

For every condition, we measure:
1. **Mean & Std Token $L_2$ Norm:** $\|h'_t\|_2$.
2. **Mean Cosine Similarity to Original Token:** $\cos(h'_t, h_t)$.
3. **Mean Cosine Similarity to Same-Image Mean:** $\cos(h'_t, \mu_i)$.
4. **Euclidean Distance to Empirical Activation Centroid:** $\|h'_t - \mu_{\text{layer}}\|_2$.
5. **Diagonal Mahalanobis Distance:**
   $$D_{\text{Mahal}}(h'_t) = \sum_{d=1}^D \left(\frac{h'_t[d] - \mu_d}{\sigma_d + 10^{-6}}\right)^2$$
   Quantifies how far replacement vectors lie from the empirical Block-8 activation manifold.

---

## 5. Statistical Evaluation & Unit of Independence

- **Unit of Independence:** The image ($N = 1,000$).
- **Multi-Seed Aggregation Rule:**
  - For conditions with multiple seeds (Random Sphere: 5 seeds, Gaussian: 5 seeds, Feature Shuffle: 3 seeds), statistics are computed **per seed first**.
  - We report the seed-wise effects, the mean across seeds, and standard deviation across seeds.
  - **Never pool 5,000 seed-image pairs as 5,000 independent samples.**
- **Paired Hypothesis Tests:**
  - True-class margin $m_i = z_y - \max_{c \neq y} z_c$.
  - Margin damage: $\text{Damage}_C = \text{BaselineMargin} - \text{Margin}_C$.
  - Paired difference between Condition A and Condition B: $\Delta m = m_A - m_B$.
  - 10,000-resample paired percentile bootstrap 95% Confidence Interval.
  - Paired Student's $t$-test ($t$-statistic and $p$-value).
  - Wilcoxon signed-rank test ($W$-statistic and $p$-value).
  - Paired effect size Cohen's $d_z = \text{mean}(\Delta m) / \text{std}(\Delta m)$.
  - Top-1 Accuracy difference $\Delta \text{Acc}$ and exact two-sided McNemar test.
- **Key Contrasts:**
  1. `Norm-Matched Random Sphere vs Cross-Image` (Test of H2)
  2. `Norm-Matched Random Sphere vs Same-Image Mean` (Test of H2)
  3. `Layer-Statistics Gaussian vs Cross-Image` (Test of H3)
  4. `Feature-Shuffled Original vs Cross-Image` (Test of learned feature geometry)
  5. `Norm-Matched Mean vs Norm-Matched Random Sphere` (Direction vs Magnitude)
  6. `Cross-Image vs Norm-Matched Cross-Image` (Norm sensitivity of donor)

---

## 6. Pre-Registered Decision Rules

### OUTCOME A — ZERO-ABLATION / SCALE ARTIFACT
Declare **OUTCOME A** if:
- Norm-Matched Random Sphere recovers $\ge 80\%$ of the recovery seen with Cross-Image over Zero:
  $$\frac{\text{Margin}_{\text{RandomSphere}} - \text{Margin}_{\text{Zero}}}{\text{Margin}_{\text{Cross}} - \text{Margin}_{\text{Zero}}} \ge 0.80$$
- AND either $|\text{RandomSphere vs Cross } d_z| < 0.20$ OR accuracy differs by $< 1.0$ percentage point on **both** DeiT-Tiny and DeiT-Small.
- **Conclusion:** V0's apparent fungibility effect is largely explained by zero ablation producing an abnormal activation magnitude. Arbitrary vectors with correct norm rescue the model. The Content-Fungibility hypothesis is rejected.

### OUTCOME B — DISTRIBUTION MATTERS, CONTENT DOES NOT
Declare **OUTCOME B** if:
- Norm-Matched Random Sphere remains substantially worse than Cross-Image ($d_z \ge 0.20$, $p < 0.05$),
- BUT Layer-Statistics Gaussian approaches Cross-Image performance (recovering $\ge 80\%$ of Cross advantage, or differing from Cross by $|d_z| < 0.20$),
- **Conclusion:** Patch-specific content is dispensable at Block 8, but downstream computation requires activations drawn from an appropriate learned marginal distribution.

### OUTCOME C — LEARNED REPRESENTATION STRUCTURE MATTERS
Declare **OUTCOME C** only if **BOTH** DeiT-Tiny and DeiT-Small show:
1. Norm-Matched Random Sphere remains substantially worse than Cross-Image ($d_z \ge 0.20$, bootstrap 95% CI strictly $> 0$ for Cross - RandomSphere);
2. Layer-Statistics Gaussian also remains substantially worse than Cross-Image or LayerMean ($d_z \ge 0.20$);
3. Feature-Dimension Shuffle substantially degrades performance relative to Cross-Image and Baseline ($d_z \ge 0.20$);
4. Same-Image Mean and/or Cross-Image retain strong performance;
5. Effects are stable across random seeds.
- **Conclusion:** Block-8 patch-specific content is partially dispensable, but arbitrary well-scaled or moment-matched activations are insufficient. Downstream computation requires structured, learned representation manifolds. Content fungibility is confirmed as a genuine structural property.

---

## 7. Pre-Registered Programmatic Validation Assertions

Before issuing a scientific verdict, the execution harness must verify 14 programmatic assertions:
1. `check1_backbone_unmodified`: SHA-256 weight hash identical before/after execution.
2. `check2_same_image_ids`: Exact same $N = 1,000$ image IDs as V0.
3. `check3_exactly_49_patches`: Exactly 49 spatial patch tokens modified per image.
4. `check4_same_patch_mask`: Mask indices identical across all conditions (seed 2501).
5. `check5_cls_untouched`: CLS token strictly untouched at injection ($h'_{8,0} = h_{8,0}$).
6. `check6_downstream_blocks_identical`: Downstream Blocks 9..11 identical across conditions.
7. `check7_norm_matched_random_relative_error`: Random Sphere tokens match original token norm within $10^{-5}$ relative error.
8. `check8_gaussian_stats_unlabeled`: Gaussian parameters estimated without label conditioning.
9. `check9_norm_matched_mean_relative_error`: Norm-Matched Mean matches original token norm within $10^{-5}$ relative error.
10. `check10_feature_shuffle_bijective`: Feature shuffle is a true bijection of feature dimensions $\{1, \dots, D\}$.
11. `check11_frozen_random_seeds`: Frozen seeds verified (`sphere: 6501-6505, gaussian: 7501-7505, shuffle: 8501-8503`).
12. `check12_no_gradients_or_labels`: Interventions constructed strictly without gradients or labels.
13. `check13_image_level_independence`: Statistics computed with $N = 1,000$ independent image units.
14. `check14_report_equals_manifest`: All reported values match machine-readable manifest outputs.
