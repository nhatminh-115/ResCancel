# DELTA TRANSPORT V0.5: Matched-Oracle Specificity Falsification Report

**Document Status:** COMPLETE & DECISIVE  
**Date:** 2026-09-28  
**Repository:** `https://github.com/nhatminh-115/ResCancel`  
**Commit Range:** `40fbb36..HEAD`  
**Git Commit SHA:** `40fbb3650e6038a9ac2cd12d2122bfe5a6d1e0bf`  
**Author:** DeepMind Pair Programmer / ResCancel Research Team  
**Companion Documents:**
- [`docs/DELTA_V0_5_PROTOCOL.md`](file:///d:/Study/ResCancel/docs/DELTA_V0_5_PROTOCOL.md)
- [`outputs/delta_v0_5/experiment_manifest.json`](file:///d:/Study/ResCancel/outputs/delta_v0_5/experiment_manifest.json)
- [`outputs/delta_v0_5/deit_tiny_patch16_224_matched_oracle_comparison.csv`](file:///d:/Study/ResCancel/outputs/delta_v0_5/deit_tiny_patch16_224_matched_oracle_comparison.csv)
- [`outputs/delta_v0_5/deit_small_patch16_224_matched_oracle_comparison.csv`](file:///d:/Study/ResCancel/outputs/delta_v0_5/deit_small_patch16_224_matched_oracle_comparison.csv)

---

## 1. Executive Summary & Scientific Verdict

In Delta Transport V0, an apparent breakthrough was observed: the **Tokenwise Historical Oracle** drastically outperformed the **Global Historical Oracle** on both DeiT-Tiny ($\Delta m = +0.2521, d_z = 1.761$) and DeiT-Small ($\Delta m = +0.1833, d_z = 1.495$).

However, an independent audit identified a major structural asymmetry:
- **Global Oracle:** 1 single decision per image (choosing 1 of 8 candidate layers globally).
- **Tokenwise Oracle:** 196 independent patch-level decisions (each choosing 1 of 8 candidate directions via the true-label margin gradient).

**DELTA TRANSPORT V0.5** was designed as a strict **matched-capacity falsification experiment**. It removed this degree-of-freedom confound by providing every condition with the **exact same selection capacity**:
- Exactly 196 patch-level decisions;
- Exactly 8 L2-normalized candidate directions per patch;
- The exact same analytical true-label margin gradient $g_t$;
- The exact same argmax projection rule ($k^*_t = \arg\max_k g_t^T u_{k,t}$);
- The exact same transport perturbation budget ($\gamma = 0.05$);
- The exact same injection point (input to Block 9) and downstream blocks (9–11);
- CLS token strictly untouched at injection;
- Frozen pretrained backbone weights.

```
========================================================================================
                      DELTA TRANSPORT V0.5 SCIENTIFIC VERDICT
========================================================================================
VERDICT:
  -> OUTCOME A — CAPACITY ARTIFACT / KILL
     The apparent V0 tokenwise advantage was largely an artifact of gradient-guided
     per-token selection capacity (196 independent 1-of-8 argmax projections), rather
     than useful, specific information residing in historical residual deltas.

CORE QUANTITATIVE EVIDENCE:
  1. Matched Random Unit Vectors Reproduce Virtually the Entire Oracle Gain:
     - DeiT-Tiny:  Historical vs. Baseline = +0.3630 margin gain.
                   Random Oracle (5-seed mean) vs. Baseline = +0.3478 margin gain.
                   Random-Direction Oracle reproduces 95.8% of the historical gain!
                   Historical vs. Random diff: Δm = +0.0152 (dz = 0.350, acc diff = +0.2%, McNemar p = 0.625).
     - DeiT-Small: Historical vs. Baseline = +0.2715 margin gain.
                   Random Oracle (5-seed mean) vs. Baseline = +0.2739 margin gain.
                   Random-Direction Oracle reproduces 100.9% of the historical gain!
                   Historical vs. Random diff: Δm = -0.0025 (95% CI [-0.0046, -0.0002], dz = -0.070).
  2. Cross-Image & Spatially Shuffled Historical Deltas Match or Outperform True Deltas:
     - Cross-image donor deltas (seed 3501, j != i) achieve higher margins and accuracy than
       same-image historical deltas (Tiny: Δm = -0.0171, Small: Δm = -0.0340).
     - Spatially scrambled deltas (seeds 4501-4503) achieve higher margins and accuracy than
       spatially aligned historical deltas (Tiny: Δm = -0.0247, Small: Δm = -0.0496).
  3. Spatial Selection Entropy is Purely a Mathematical Artifact:
     - Random-direction oracle exhibits H = 2.97 bits (out of theoretical max 3.00 bits),
       proving that high selection entropy is an intrinsic mathematical property of
       independent high-dimensional gradient argmax selection, NOT evidence of semantic
       depth specialization.

DISPOSITION:
  THE HISTORICAL RESIDUAL-DELTA ROUTING HYPOTHESIS IS CONCLUSIVELY FALSIFIED.
  NO ROUTER ARCHITECTURE (V1) WILL BE DESIGNED OR TRAINED.
  THE PROJECT IS FORMALLY TERMINATED AT V0.5.
========================================================================================
```

---

## 2. Experimental Execution & Compliance Summary

- **Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (Hardware Budget: $\le 8\text{ GB}$).
- **Peak VRAM Allocated:** **$1,148.7\text{ MB}$** (DeiT-Small), **$570.0\text{ MB}$** (DeiT-Tiny) — well under 15% of the VRAM budget.
- **Total Runtime:** **$106.91\text{ seconds}$** for complete two-pass evaluation across $N = 1,000$ images on both models (including caching, 5 random seeds, cross-image permutation, 3 spatial shuffle seeds, pooled delta bank, and sensitivity sweeps).
- **Dataset:** Reproducible ImageNet-1k validation subset, $N = 1,000$ images (stratified 1 per class), seed 42.
- **Models:** Standard pretrained `deit_tiny_patch16_224` and `deit_small_patch16_224` from `timm`.
- **Primary Budget:** $\gamma = 0.05$ (Sensitivities: $\gamma = 0.02, 0.10$).
- **Injection Point:** Input to Block 9 (after Block 8), candidates from Blocks 0..7.

### Programmatic Validation Assertions (13/13 Checks Passed)
All 13 pre-registered checks ([`delta_transport/v0_5_validation.py`](file:///d:/Study/ResCancel/delta_transport/v0_5_validation.py)) passed:
1. $\checkmark$ **Backbone Unmodified:** SHA-256 weight hash identical before and after inference.
2. $\checkmark$ **Same 1,000 Image IDs:** Evaluated across identical samples.
3. $\checkmark$ **8 Candidates per Patch:** Verified across all oracle conditions.
4. $\checkmark$ **196 Independent Patch Choices:** Verified for each condition.
5. $\checkmark$ **Identical Gradient Tensor:** Analytical margin gradient $g_t$ shared across all conditions.
6. $\checkmark$ **L2-Normalized Candidates:** All vectors satisfy $\|u_{k,t}\|_2 = 1.0$.
7. $\checkmark$ **Matched Perturbation Magnitude:** Verified $\|h'_t - h_t\|_2 / \|h_t\|_2 = \gamma$.
8. $\checkmark$ **CLS Untouched:** $h'_{8,\text{CLS}} == h_{8,\text{CLS}}$ strictly asserted.
9. $\checkmark$ **Cross-Image Donor $j \neq i$:** Zero self-donor matches across all 1,000 images.
10. $\checkmark$ **Spatial Shuffle Bijective:** Patch permutations strictly verify $\text{len}(\text{unique}(\pi)) = 196$.
11. $\checkmark$ **Frozen Random Seeds:** Evaluated on seeds `2501..2505`, `3501`, `4501..4503`, `5501`.
12. $\checkmark$ **Image-Level Independence:** Statistical tests operate on $N = 1,000$ independent units.
13. $\checkmark$ **Manifest Data Consistency:** Machine-readable CSVs and manifest match exactly.

---

## 3. Matched-Capacity Performance Comparison Table

All conditions receive **196 patch decisions**, **8 candidate directions**, identical gradient $g_t$, and perturbation budget $\gamma = 0.05$. $N = 1,000$ images per architecture.

### 3.1 DeiT-Tiny (`deit_tiny_patch16_224`)

| Condition | Candidate Source | Mean Margin | Margin $\Delta$ vs. Base | Hist - Condition $\Delta m$ (95% CI) | Cohen's $d_z$ | Paired $p$-val | Top-1 Accuracy | Acc Diff vs. Hist | McNemar $p$-val |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | None ($\gamma = 0$) | 1.3102 | 0.0000 | +0.3630 [+0.3518, +0.3742] | 1.987 | $0.00$ | 70.7% | +4.2% | $4.55 \times 10^{-13}$ |
| **Global Oracle (V0)** | 1-of-8 Global Layer | 1.4211 | +0.1109 | +0.2521 [+0.2434, +0.2608] | 1.761 | $1.16 \times 10^{-308}$ | 72.1% | +2.8% | $7.45 \times 10^{-9}$ |
| **Random-Direction Oracle** | 8 Random Unit Vectors (5s) | **1.6580** | **+0.3478** | **+0.0152 [+0.0125, +0.0179]** | **0.350** | **$7.02 \times 10^{-27}$** | **74.7%** | **+0.2%** | **0.625** |
| **Cross-Image Oracle** | 8 Deltas from Donor $j \neq i$ | **1.6903** | **+0.3801** | **-0.0171 [-0.0204, -0.0139]** | **-0.323** | **$2.20 \times 10^{-23}$** | **75.3%** | **-0.4%** | **0.344** |
| **Spatial-Shuffle Oracle** | 8 Shuffled-Patch Deltas (3s) | **1.6979** | **+0.3877** | **-0.0247 [-0.0275, -0.0220]** | **-0.554** | **$3.33 \times 10^{-60}$** | **75.8%** | **-0.9%** | **0.0039** |
| **Pooled-Delta Oracle** | 8 Randomly Sampled Deltas | **1.6831** | **+0.3729** | **-0.0099 [-0.0127, -0.0071]** | **-0.218** | **$8.71 \times 10^{-12}$** | **75.5%** | **-0.6%** | **0.070** |
| **True Historical Oracle** | 8 Same-Image Aligned Deltas | **1.6732** | **+0.3630** | 0.0000 | 0.000 | 1.000 | **74.9%** | 0.0% | 1.000 |

### 3.2 DeiT-Small (`deit_small_patch16_224`)

| Condition | Candidate Source | Mean Margin | Margin $\Delta$ vs. Base | Hist - Condition $\Delta m$ (95% CI) | Cohen's $d_z$ | Paired $p$-val | Top-1 Accuracy | Acc Diff vs. Hist | McNemar $p$-val |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | None ($\gamma = 0$) | 2.4251 | 0.0000 | +0.2715 [+0.2617, +0.2813] | 1.701 | $2.40 \times 10^{-297}$ | 79.1% | +1.9% | $3.81 \times 10^{-6}$ |
| **Global Oracle (V0)** | 1-of-8 Global Layer | 2.5133 | +0.0882 | +0.1833 [+0.1757, +0.1908] | 1.495 | $5.10 \times 10^{-257}$ | 79.6% | +1.4% | $1.22 \times 10^{-4}$ |
| **Random-Direction Oracle** | 8 Random Unit Vectors (5s) | **2.6990** | **+0.2739** | **-0.0025 [-0.0046, -0.0002]** | **-0.070** | **0.0272** | **81.0%** | **0.0%** | **1.000** |
| **Cross-Image Oracle** | 8 Deltas from Donor $j \neq i$ | **2.7305** | **+0.3054** | **-0.0340 [-0.0375, -0.0305]** | **-0.599** | **$1.13 \times 10^{-68}$** | **81.2%** | **-0.2%** | **0.500** |
| **Spatial-Shuffle Oracle** | 8 Shuffled-Patch Deltas (3s) | **2.7462** | **+0.3211** | **-0.0496 [-0.0532, -0.0462]** | **-0.872** | **$7.01 \times 10^{-125}$** | **81.1%** | **-0.1%** | **1.000** |
| **Pooled-Delta Oracle** | 8 Randomly Sampled Deltas | **2.7346** | **+0.3095** | **-0.0380 [-0.0413, -0.0347]** | **-0.719** | **$1.18 \times 10^{-92}$** | **81.2%** | **-0.2%** | **0.500** |
| **True Historical Oracle** | 8 Same-Image Aligned Deltas | **2.6966** | **+0.2715** | 0.0000 | 0.000 | 1.000 | **81.0%** | 0.0% | 1.000 |

---

## 4. Stability Across Control Seeds

### 4.1 Random-Direction Oracle (5 Frozen Seeds)
Each seed generates an independent set of 8 random unit vectors per patch:

```
DeiT-Tiny (Historical Margin = 1.6732):
  Seed 2501: Random Mean Margin = 1.6575 | Diff = +0.0157 (dz = 0.318, 95% CI [0.0126, 0.0187])
  Seed 2502: Random Mean Margin = 1.6585 | Diff = +0.0147 (dz = 0.302, 95% CI [0.0117, 0.0176])
  Seed 2503: Random Mean Margin = 1.6577 | Diff = +0.0155 (dz = 0.321, 95% CI [0.0125, 0.0184])
  Seed 2504: Random Mean Margin = 1.6579 | Diff = +0.0153 (dz = 0.315, 95% CI [0.0123, 0.0183])
  Seed 2505: Random Mean Margin = 1.6582 | Diff = +0.0150 (dz = 0.320, 95% CI [0.0121, 0.0179])

DeiT-Small (Historical Margin = 2.6966):
  Seed 2501: Random Mean Margin = 2.6990 | Diff = -0.0024 (dz = -0.062, 95% CI [-0.0048, +0.0000])
  Seed 2502: Random Mean Margin = 2.6995 | Diff = -0.0029 (dz = -0.073, 95% CI [-0.0053, -0.0004])
  Seed 2503: Random Mean Margin = 2.6992 | Diff = -0.0027 (dz = -0.068, 95% CI [-0.0051, -0.0003])
  Seed 2504: Random Mean Margin = 2.6989 | Diff = -0.0023 (dz = -0.059, 95% CI [-0.0047, +0.0001])
  Seed 2505: Random Mean Margin = 2.6986 | Diff = -0.0021 (dz = -0.052, 95% CI [-0.0046, +0.0004])
```

The random-direction oracle is extraordinarily consistent across seeds: on DeiT-Small, across all 5 seeds, random unit vectors match or slightly outperform historical residual deltas.

### 4.2 Spatially Shuffled Historical Oracle (3 Frozen Seeds)
```
DeiT-Tiny (Historical Margin = 1.6732):
  Seed 4501: Shuffled Mean Margin = 1.6975 | Diff = -0.0244 (dz = -0.488, 95% CI [-0.0274, -0.0213])
  Seed 4502: Shuffled Mean Margin = 1.6984 | Diff = -0.0253 (dz = -0.548, 95% CI [-0.0281, -0.0224])
  Seed 4503: Shuffled Mean Margin = 1.6977 | Diff = -0.0245 (dz = -0.502, 95% CI [-0.0275, -0.0215])

DeiT-Small (Historical Margin = 2.6966):
  Seed 4501: Shuffled Mean Margin = 2.7464 | Diff = -0.0498 (dz = -0.843, 95% CI [-0.0535, -0.0461])
  Seed 4502: Shuffled Mean Margin = 2.7461 | Diff = -0.0495 (dz = -0.821, 95% CI [-0.0533, -0.0459])
  Seed 4503: Shuffled Mean Margin = 2.7462 | Diff = -0.0496 (dz = -0.841, 95% CI [-0.0533, -0.0460])
```

Across every seed and both architectures, scrambling the spatial patch locations *increases* oracle performance by $\approx 0.025$ (Tiny) and $\approx 0.050$ (Small). This directly falsifies the claim that spatial patch tokens require their *own* localized depth updates.

---

## 5. Deconstruction of the "High Selection Entropy" Claim

In V0, we observed an average Shannon entropy of $H = 2.75$ bits and claimed this supported "genuine spatial depth specialization." 

In V0.5, we computed the selection entropy for the **Random-Direction Oracle**:

| Condition | DeiT-Tiny Entropy | DeiT-Small Entropy | Dominant Candidate Fraction | Distinct Candidates / Image |
| :--- | :---: | :---: | :---: | :---: |
| **True Historical Oracle** | 2.75 bits | 2.76 bits | 26.5% | 7.99 / 8.00 |
| **Cross-Image Oracle** | 2.78 bits | 2.77 bits | 24.7% | 7.99 / 8.00 |
| **Spatial-Shuffle Oracle** | 2.80 bits | 2.80 bits | 24.6% | 7.99 / 8.00 |
| **Random-Direction Oracle** | **2.97 bits** | **2.97 bits** | **16.2%** | **8.00 / 8.00** |
| *Theoretical Uniform Maximum* | *3.00 bits* | *3.00 bits* | *12.5%* | *8.00* |

### Scientific Finding
The Random-Direction Oracle achieves **$2.97\text{ bits}$** of entropy—virtually indistinguishable from a uniform distribution ($3.00$ bits). 

This demonstrates conclusively that:
> **High entropy is a trivial mathematical consequence of independent argmax selection.**  
> When an oracle chooses the best of $K=8$ quasi-orthogonal vectors in high dimension ($d=192$ or $384$) independently across $T=196$ patch tokens using an analytical gradient, the probability that any single index dominates across all patches is near zero. High entropy reflects the dimensionality of the candidate pool, **NOT** semantic depth specialization or asynchronous token maturation.

---

## 6. Scientific Discussion & Root-Cause Analysis

### 6.1 Why Did the V0 Oracle "Work"?
In V0, comparing Tokenwise Oracle ($196 \times 8$ choices) against Global Oracle ($1 \times 8$ choice) conflated two distinct factors:
1. **Specific Historical Information:** Useful features encoded in past block residual updates.
2. **Oracle Selection Capacity:** The ability of gradient ascent to pick a favorable vector out of 8 options for each of 196 tokens.

V0.5 proves that **Factor 2 accounts for $>95\%$ of the observed effect**:
- Any set of 8 random unit vectors drawn from isotropic Gaussian noise provides enough directional diversity for gradient projection to achieve $\Delta m = +0.3478$ (Tiny) and $+0.2739$ (Small).
- When historical deltas are taken from an entirely unrelated image (**Cross-Image**), performance is *better* than same-image historical deltas ($+0.3801$ on Tiny, $+0.3054$ on Small).
- When historical deltas are taken from random spatial positions (**Spatial Shuffle**), performance is *better* still ($+0.3877$ on Tiny, $+0.3211$ on Small).

### 6.2 Why Do Shuffled & Cross-Image Deltas Outperform Same-Image Aligned Deltas?
Same-image, spatially aligned historical deltas $\Delta_{l,t}$ suffer from **spatial collinearity**: because the transformer self-attention mechanism has already aligned token $t$'s representation with its receptive field, historical deltas at token $t$ tend to point in collinear directions across depth. 

When patch positions are shuffled, or when donor deltas are drawn from another image, the candidate pool contains **greater angular diversity**. Greater angular diversity allows gradient projection $g_t^T u_{k,t}$ to find candidate vectors that align more closely with the gradient $g_t$, yielding higher margin gains!

---

## 7. Decision Rule Evaluation & Final Disposition

### Pre-Registered Outcome A Trigger Conditions:
1. Random-Direction Oracle reproduces $\ge 80\%$ of historical gain on both architectures:
   - DeiT-Tiny: $\mathbf{95.8\%}$ reproduced ($+0.3478 / +0.3630$).
   - DeiT-Small: $\mathbf{100.9\%}$ reproduced ($+0.2739 / +0.2715$).
   - **TRIGGERED.**
2. Historical vs. Random margin effect size $d_z < 0.20$ on DeiT-Small:
   - Observed $d_z = \mathbf{-0.070} < 0.20$.
   - **TRIGGERED.**
3. Historical vs. Random 95% Bootstrap CI on DeiT-Small:
   - Observed CI is strictly negative: $\mathbf{[-0.0046, -0.0002]}$.
   - **TRIGGERED.**

### Formal Verdict:
**OUTCOME A — CAPACITY ARTIFACT / KILL**

> *"The V0 advantage was an artifact of label-gradient-guided per-token selection degrees-of-freedom rather than meaningful historical residual information. There is no evidence that spatial patch tokens in pretrained Vision Transformers benefit specifically from retrieving their own historical residual deltas."*

### Final Strategic Disposition:
- **No router architecture (V1) will be designed or trained.**
- **No fine-tuning or method development is justified.**
- The Delta Transport research line is formally and permanently **TERMINATED** at V0.5.
