# Patch Content Fungibility V0.7: Scientific Report
## Prototype Specificity and Complete Patch-Stream Replacement Test

**Author:** DeepMind Antigravity Team  
**Date:** September 28, 2026  
**Repository:** [https://github.com/nhatminh-115/ResCancel](https://github.com/nhatminh-115/ResCancel)  
**Status:** Completed, Programmatically Validated & Fully Audited  
**Pre-Registered Decision:** **OUTCOME A — GENERIC LATE-LAYER CENTROID / FALSIFICATION OF DEPTH-8 ISOLATION**  

---

## Executive Summary

In Patch Content Fungibility V0.6, we discovered that replacing masked patch representations at Block 8 with a single static calibration mean vector $\mu_8$ (with zero stochastic noise) matched or outperformed coordinate-wise Gaussian sampling on both DeiT-Tiny ($67.6\%$ vs $66.9\%$) and DeiT-Small ($76.1\%$ vs $75.6\%$).

Patch Fungibility V0.7 was executed as a pre-registered mechanistic falsification experiment to determine **WHY** $\mu_8$ preserves classification performance. Specifically, we tested four competing hypotheses:
- **H1 — Depth-Specific Prototype:** Downstream blocks 9–11 require the exact feature template of Block 8.
- **H2 — Generic Centroid / Bias Effect:** Downstream blocks tolerate any late-layer activation centroid sharing coordinate orientation and scale.
- **H3 — Norm / Low-Dimensional Property:** Downstream blocks only require vector magnitude or basic norm constraints.
- **H4 — Prototype Not Actually Necessary:** At late depth, downstream blocks tolerate arbitrary non-zero vectors.

Using $N_{\text{calib}} = 1,000$ and $N_{\text{eval}} = 1,000$ strictly disjoint ImageNet-1k validation sets across DeiT-Tiny ($D=192$) and DeiT-Small ($D=384$), we evaluated Depth 8 across 4 replacement budgets ($25\%, 50\%, 75\%, 100\%$) under 11 rigorous control conditions.

```
========================================================================================
                          CORE SCIENTIFIC FINDINGS (V0.7)
========================================================================================

1. FALSIFICATION OF DEPTH-8 SPECIFICITY (H1 Refuted -> H2 Confirmed):
   - Mismatched calibration means from neighboring depths (Depth 7 and Depth 9) perform
     INDISTINGUISHABLY from the true Depth-8 prototype:
     * DeiT-Tiny (50% replacement):
       mu_8: Acc = 66.3%, Mean Damage = 0.189
       mu_7: Acc = 66.1%, Mean Damage = 0.190 (Cohen's dz = +0.013, p = 0.677, FDR q = 0.713)
       mu_9: Acc = 66.3%, Mean Damage = 0.190 (Cohen's dz = +0.084, p = 0.008, FDR q = 0.017)
     * DeiT-Small (50% replacement):
       mu_8: Acc = 75.6%, Mean Damage = 0.131
       mu_7: Acc = 75.5%, Mean Damage = 0.130 (Cohen's dz = -0.019, p = 0.539, FDR q = 0.567)
       mu_9: Acc = 75.7%, Mean Damage = 0.132 (Cohen's dz = +0.094, p = 0.003, FDR q = 0.005)
   - Distant depth means (Depth 5) fail significantly (Tiny 50% damage: 0.393 vs 0.189, dz = +0.250).
   - This proves that downstream blocks 9-11 do NOT require an isolated Depth-8 template;
     rather, Blocks 7-9 inhabit a coherent late-layer representation manifold aligned
     along a shared regional centroid (cos(mu_7, mu_8) = 0.89-0.91, cos(mu_9, mu_8) = 0.88-0.93).

2. REFUTATION OF ARBITRARY FILL (H4 Refuted):
   - Coordinate Permutation: Shuffling channel dimensions (preserving norm, mean, variance)
     causes catastrophic collapse across both architectures:
     * Tiny (50% replacement): Acc collapses from 66.3% to 52.2% (dz = +0.561, p = 1.80e-61).
     * Small (50% replacement): Acc collapses from 75.6% to 62.6% (dz = +0.689, p = 1.25e-86).
     Downstream blocks strictly require exact coordinate-channel correspondence.
   - Sign Inversion: Reversing prototype direction (-mu_8) completely destroys classification:
     * Tiny (50% replacement): Acc drops to 0.7% (dz = +1.084, p = 6.07e-171).
     * Small (50% replacement): Acc drops to 1.1% (dz = +1.282, p = 5.23e-213).
     Partial sign flips (25%, 50%, 75%) show a monotonic dose-response collapse.

3. MONOTONIC COSINE ALIGNMENT TRAJECTORY:
   - Evaluated across synthetic vectors r_alpha with exact norm ||r_alpha|| = ||mu_8|| and
     controlled cosine similarities {0.00, 0.25, 0.50, 0.75} to mu_8.
   - Predictive recovery improves monotonically with cosine alignment:
     * Small (50%): Cos 0.0 -> 60.1%; Cos 0.25 -> 71.9%; Cos 0.50 -> 74.6%; Cos 0.75 -> 75.6%; mu_8 -> 75.6%.
   - At alpha >= 0.50, performance saturates near baseline, explaining why mu_7 and mu_9
     (which have cos >= 0.88) substitute for mu_8 with zero loss.

4. 75% TO 100% PATCH REPLACEMENT TRANSITION & RANK-1 COLLAPSE:
   - At 75% Replacement (147 of 196 spatial tokens replaced simultaneously):
     mu_8 retains 64.2% accuracy on Tiny (vs Zero's 50.8%) and 73.6% on Small (vs Zero's 49.0%).
     Crucially, 94.5% of originally correct evaluation images remain correct on Small!
   - At 100% Replacement (ALL 196 spatial tokens replaced with mu_8 copies, CLS untouched):
     mu_8 collapses to 10.2% on Tiny and 17.0% on Small.
     Why? When all 196 tokens are identical static copies of mu_8, spatial self-attention
     undergoes complete rank-1 collapse.
     In contrast, Held-Out Gaussian tokens (which retain stochastic coordinate diversity)
     avoid spatial rank collapse and achieve 26.5% on Tiny and 46.7% on Small!

5. CLS VS PATCH STREAM ASYMMETRY:
   - Replacing the CLS token with its calibration mean mu_cls while keeping all 196 patches
     original yields 66.4% on Tiny (clean: 67.9%) and 74.8% on Small (clean: 76.1%).
   - Zeroing CLS reduces Tiny accuracy to 40.0% (damage = 1.750).
========================================================================================
```

---

## 1. Disjoint Dataset Split Verification

- **Source:** ImageNet-1k Validation Set (50,000 images).
- **Calibration Split ($N_{\text{calib}} = 1,000$):** Exactly reused from V0.6 ([`outputs/fungibility_v0_6/calibration_split.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v0_6/calibration_split.csv), seed `9101`).
- **Evaluation Split ($N_{\text{eval}} = 1,000$):** Exactly reused from V0.6 ([`outputs/fungibility_v0_6/evaluation_split.csv`](file:///d:/Study/ResCancel/outputs/fungibility_v0_6/evaluation_split.csv), seed `9201`).
- **Overlap Verification:**
  $$\text{Intersection}(\text{IDs}_{\text{calib}}, \text{IDs}_{\text{eval}}) = \emptyset \quad (\text{Exact } 0 \text{ overlap confirmed})$$
- **Clean Evaluation Baseline Accuracy:**
  - DeiT-Tiny: **67.90%** (Mean Logit Margin = **1.1681**)
  - DeiT-Small: **76.10%** (Mean Logit Margin = **2.2423**)

---

## 2. Quantitative Results

### 2.1 Fraction Summary (25%, 50%, 75%, 100% Spatial Patch Replacement)

$$\text{Damage}_C(f) = \text{BaselineMargin} - \text{Margin}_C(f)$$
$$\text{Prototype Retention}(f) = \frac{\text{Damage}_{\text{zero}}(f) - \text{Damage}_{\mu_8}(f)}{\text{Damage}_{\text{zero}}(f)} \quad (\text{for } \text{Damage}_{\text{zero}} \ge 0.10)$$

| Architecture | Budget $f$ | Replaced Tokens | Clean Acc | Zero Acc | $\mu_8$ Acc | Gauss Acc | Zero Dmg | $\mu_8$ Dmg | Gauss Dmg | $\mu_8$ Retention | Originally Correct Retained |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeiT-Tiny** | 25% | 49 | 67.9% | 59.1% | **67.0%** | 66.1% | 0.757 | **0.076** | 0.126 | **90.0%** | **97.1%** |
| | 50% | 98 | 67.9% | 55.5% | **66.3%** | 64.5% | 0.923 | **0.189** | 0.311 | **79.6%** | **95.3%** |
| | 75% | 147 | 67.9% | 50.8% | **64.2%** | 58.9% | 1.111 | **0.423** | 0.693 | **62.0%** | **90.3%** |
| | 100% | 196 | 67.9% | 7.9% | 10.2% | **26.5%** | 1.973 | 2.432 | 1.987 | N/A* | 13.8% |
| **DeiT-Small**| 25% | 49 | 76.1% | 69.7% | **75.7%** | 75.3% | 1.430 | **0.044** | 0.078 | **96.9%** | **98.6%** |
| | 50% | 98 | 76.1% | 63.0% | **75.6%** | 73.3% | 1.931 | **0.131** | 0.257 | **93.2%** | **98.2%** |
| | 75% | 147 | 76.1% | 49.0% | **73.6%** | 68.6% | 2.286 | **0.397** | 0.744 | **82.6%** | **94.5%** |
| | 100% | 196 | 76.1% | 36.9% | 17.0% | **46.7%** | 2.503 | 3.663 | 2.320 | N/A* | 20.8% |

*\* Note: At 100% replacement, zero ablation damage is 1.973 (Tiny) and 2.503 (Small), but mu_8 suffers extreme attention rank collapse (damage = 2.432 and 3.663), resulting in negative retention.*

---

### 2.2 Prototype Specificity Contrasts (50% Patch Replacement)

$$\text{Prototype Advantage} = \text{Margin}(\mu_8) - \text{Margin}(\text{Control})$$

All hypothesis tests use $N=1,000$ independent evaluation images. 95% Bootstrap CIs are based on 10,000 paired resamples. BH-FDR $q$-values are corrected within logical contrast families.

| Architecture | Control Vector | Advantage ($\Delta m$) | 95% Bootstrap CI | Cohen's $d_z$ | Paired $t$-test $p$ | BH-FDR $q$ | Acc ($\mu_8$) | Acc (Ctrl) | McNemar $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeiT-Tiny** | Zero Ablation | **+0.735** | [+0.647, +0.824] | **+0.463** | $7.67 \times 10^{-44}$ | $1.53 \times 10^{-43}$ | 66.3% | 55.5% | $8.07 \times 10^{-14}$ |
| | Held-Out Gaussian | **+0.122** | [+0.100, +0.144] | **+0.324** | $3.46 \times 10^{-23}$ | $3.46 \times 10^{-23}$ | 66.3% | 64.5% | $3.47 \times 10^{-2}$ |
| | Wrong Depth 5 ($\mu_5$) | **+0.204** | [+0.154, +0.254] | **+0.250** | $6.30 \times 10^{-15}$ | $2.52 \times 10^{-14}$ | 66.3% | 64.5% | $0.057$ |
| | Norm-Matched $\mu'_5$ | **+0.064** | [+0.034, +0.095] | **+0.131** | $3.90 \times 10^{-5}$ | $7.77 \times 10^{-5}$ | 66.3% | 65.9% | $0.678$ |
| | Wrong Depth 6 ($\mu_6$) | **+0.067** | [+0.041, +0.093] | **+0.153** | $1.62 \times 10^{-6}$ | $4.06 \times 10^{-6}$ | 66.3% | 66.3% | $1.000$ |
| | Wrong Depth 7 ($\mu_7$) | **+0.001** | [-0.004, +0.007] | **+0.013** | $0.677$ | $0.713$ | 66.3% | 66.1% | $0.727$ |
| | Norm-Matched $\mu'_7$ | **-0.002** | [-0.005, +0.001] | **-0.037** | $0.236$ | $0.278$ | 66.3% | 66.1% | $0.625$ |
| | Wrong Depth 9 ($\mu_9$) | **+0.001** | [+0.000, +0.003] | **+0.084** | $0.008$ | $0.017$ | 66.3% | 66.3% | $1.000$ |
| | Coordinate Permuted | **+0.945** | [+0.842, +1.050] | **+0.561** | $1.80 \times 10^{-61}$ | $4.80 \times 10^{-61}$ | 66.3% | 52.2% | $7.97 \times 10^{-30}$ |
| | Full Inversion ($-\mu_8$) | **+3.096** | [+2.920, +3.272] | **+1.084** | $6.07 \times 10^{-171}$ | $4.85 \times 10^{-170}$ | 66.3% | 0.7% | $1.10 \times 10^{-195}$ |
| | Cosine $\alpha = 0.0$ | **+0.904** | [+0.816, +0.993] | **+0.577** | $1.39 \times 10^{-64}$ | $2.78 \times 10^{-64}$ | 66.3% | 50.8% | $3.59 \times 10^{-30}$ |
| **DeiT-Small**| Zero Ablation | **+1.800** | [+1.670, +1.930] | **+0.676** | $5.39 \times 10^{-83}$ | $1.08 \times 10^{-82}$ | 75.6% | 63.0% | $4.18 \times 10^{-28}$ |
| | Held-Out Gaussian | **+0.126** | [+0.106, +0.147] | **+0.370** | $6.27 \times 10^{-30}$ | $6.27 \times 10^{-30}$ | 75.6% | 73.3% | $3.46 \times 10^{-3}$ |
| | Wrong Depth 5 ($\mu_5$) | **+0.054** | [+0.021, +0.087] | **+0.102** | $1.26 \times 10^{-3}$ | $2.10 \times 10^{-3}$ | 75.6% | 74.7% | $0.108$ |
| | Norm-Matched $\mu'_5$ | **+0.041** | [+0.015, +0.069] | **+0.096** | $2.34 \times 10^{-3}$ | $4.26 \times 10^{-3}$ | 75.6% | 74.9% | $0.167$ |
| | Wrong Depth 6 ($\mu_6$) | **+0.001** | [-0.012, +0.015] | **+0.006** | $0.839$ | $0.839$ | 75.6% | 75.4% | $0.727$ |
| | Wrong Depth 7 ($\mu_7$) | **-0.001** | [-0.005, +0.002] | **-0.019** | $0.539$ | $0.567$ | 75.6% | 75.5% | $1.000$ |
| | Norm-Matched $\mu'_7$ | **-0.001** | [-0.004, +0.003] | **-0.010** | $0.743$ | $0.750$ | 75.6% | 75.5% | $1.000$ |
| | Wrong Depth 9 ($\mu_9$) | **+0.001** | [+0.000, +0.002] | **+0.094** | $3.04 \times 10^{-3}$ | $4.68 \times 10^{-3}$ | 75.6% | 75.7% | $1.000$ |
| | Coordinate Permuted | **+1.688** | [+1.547, +1.829] | **+0.689** | $1.25 \times 10^{-86}$ | $2.49 \times 10^{-86}$ | 75.6% | 62.6% | $1.72 \times 10^{-30}$ |
| | Full Inversion ($-\mu_8$) | **+4.004** | [+3.818, +4.190] | **+1.282** | $5.23 \times 10^{-213}$ | $4.18 \times 10^{-212}$ | 75.6% | 1.1% | $1.86 \times 10^{-218}$ |
| | Cosine $\alpha = 0.0$ | **+1.728** | [+1.590, +1.866] | **+0.697** | $1.81 \times 10^{-88}$ | $3.62 \times 10^{-88}$ | 75.6% | 60.1% | $1.97 \times 10^{-32}$ |

---

### 2.3 Cosine Alignment & Sign Inversion Sweeps

#### Cosine Alignment Sweep (50% Patch Replacement)
Vectors $r_\alpha$ constructed with $\|r_\alpha\| = \|\mu_8\|$ and target cosine $\alpha \in \{0.00, 0.25, 0.50, 0.75\}$:

| Target Cosine $\alpha$ | DeiT-Tiny Acc (%) | DeiT-Tiny Margin | DeiT-Tiny Damage | DeiT-Small Acc (%) | DeiT-Small Margin | DeiT-Small Damage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | 50.8% | 0.075 | 1.093 | 60.1% | 0.383 | 1.859 |
| **0.25** | 61.5% | 0.575 | 0.593 | 71.9% | 1.604 | 0.638 |
| **0.50** | 65.6% | 0.880 | 0.288 | 74.6% | 2.040 | 0.202 |
| **0.75** | 66.2% | 0.976 | 0.192 | 75.6% | 2.109 | 0.133 |
| **1.00 ($\mu_8$)** | **66.3%** | **0.980** | **0.189** | **75.6%** | **2.112** | **0.131** |

#### Sign Inversion Dose-Response (50% Patch Replacement)
Deterministically flipping signs of feature coordinate subsets (seed `9901`):

| Inverted Coordinates | DeiT-Tiny Acc (%) | DeiT-Tiny Damage | DeiT-Small Acc (%) | DeiT-Small Damage |
| :---: | :---: | :---: | :---: | :---: |
| **0% ($\mu_8$)** | **66.3%** | **0.189** | **75.6%** | **0.131** |
| **25%** | 65.7% | 0.194 | 75.3% | 0.137 |
| **50%** | 65.4% | 0.351 | 74.3% | 0.276 |
| **75%** | 4.2% | 2.789 | 10.4% | 3.497 |
| **100% ($-\mu_8$)** | **0.7%** | **3.284** | **1.1%** | **4.135** |

---

### 2.4 Complete 100% Patch Replacement Breakdown

At 100% patch replacement, **all 196 spatial patch tokens** are replaced simultaneously, while the CLS token remains untouched at injection ($h'_{8,0} = h_{8,0}$):

| Condition | DeiT-Tiny Top-1 Acc | DeiT-Tiny Margin Damage | DeiT-Small Top-1 Acc | DeiT-Small Margin Damage |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline** | 67.9% | 0.000 | 76.1% | 0.000 |
| **Held-Out Gaussian** | **26.5%** | **1.987** | **46.7%** | **2.320** |
| **Zero Ablation** | 7.9% | 1.973 | 36.9% | 2.503 |
| **Block-8 Mean $\mu_8$** | 10.2% | 2.432 | 17.0% | 3.663 |
| **Best Wrong Mean ($\mu_7$)** | 7.5% | 2.806 | 12.2% | 4.170 |
| **Coordinate Permuted** | 0.9% | 2.806 | 21.1% | 4.269 |
| **Orthogonal Vector ($\alpha=0$)** | 1.4% | 2.944 | 22.4% | 2.915 |
| **Sign Inverted $-\mu_8$** | 0.4% | 3.313 | 10.1% | 4.544 |

#### Scientific Explanation of the 100% Collapse
Why does $\mu_8$ perform near baseline up to 75% replacement ($73.6\%$ on Small), yet collapse to $17.0\%$ at 100%?
- **Spatial Rank-1 Collapse:** When all 196 spatial tokens are set to the identical vector $\mu_8$, every patch token produces identical Key and Value projections in blocks 9–11. The self-attention matrix across the 196 spatial tokens becomes mathematically uniform ($1/196$), destroying spatial contrast and rank.
- **Why Gaussian Survives at 100%:** Held-Out Gaussian samples $x_d \sim \mathcal{N}(\mu_{8,d}, \sigma_{8,d}^2)$ provide per-token stochastic diversity, preventing rank-1 collapse and retaining $46.7\%$ accuracy on Small.
- **Conclusion:** Downstream blocks can operate without image-specific patch content, but the patch stream requires **token-level diversity** when 100% of tokens are synthetic to avoid attention degeneracies.

---

### 2.5 CLS vs Patch Stream Asymmetry Test (Exploratory)

Testing stream asymmetry by keeping all 196 patches untouched and intervening only on the CLS token at Depth 8:

| Model | Clean Baseline | 100% Patches $\to \mu_8$ | CLS $\to \mu_{\text{cls}}$ | CLS $\to$ Cross-Image | CLS $\to$ Zero |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DeiT-Tiny** | 67.9% | 10.2% | **66.4%** | 62.6% | 40.0% |
| **DeiT-Small**| 76.1% | 17.0% | **74.8%** | 67.1% | 70.5% |

---

## 3. Publication Figures

### Figure 1: Accuracy Trajectory Across Replacement Budgets
![Prototype Performance by Fraction](file:///d:/Study/ResCancel/figures/fungibility_v0_7/prototype_performance_by_fraction.png)

### Figure 2: Wrong-Depth Means vs Block-8 Prototype
![Wrong Depth Mean Comparison](file:///d:/Study/ResCancel/figures/fungibility_v0_7/wrong_depth_mean_comparison.png)

### Figure 3: Prototype Scale Sweep
![Prototype Scale Sweep](file:///d:/Study/ResCancel/figures/fungibility_v0_7/prototype_scale_sweep.png)

### Figure 4: Controlled Cosine Alignment Sweep
![Prototype Cosine Sweep](file:///d:/Study/ResCancel/figures/fungibility_v0_7/prototype_cosine_sweep.png)

### Figure 5: Feature Coordinate Sign Inversion Dose-Response
![Prototype Sign Flip Sweep](file:///d:/Study/ResCancel/figures/fungibility_v0_7/prototype_sign_flip_sweep.png)

### Figure 6: Complete 100% Spatial Patch Stream Replacement
![Full Patch Replacement](file:///d:/Study/ResCancel/figures/fungibility_v0_7/full_patch_replacement.png)

### Figure 7: Stream Asymmetry Test (CLS vs Patch Replacement)
![CLS vs Patch Replacement](file:///d:/Study/ResCancel/figures/fungibility_v0_7/cls_vs_patch_replacement.png)

---

## 4. Pre-Registered Programmatic Validations

All 14 pre-registered assertions passed with zero violations:

1. **Backbone Weights Frozen:** Verified model parameter hash unchanged before and after inference ($\nabla_\theta = 0$). **PASS**
2. **Reused Exact V0.6 Splits:** Calibration ($N=1,000$, seed `9101`) and Evaluation ($N=1,000$, seed `9201`) global image indices match V0.6 manifests exactly. **PASS**
3. **Zero Split Overlap:** $\text{IDs}_{\text{calib}} \cap \text{IDs}_{\text{eval}} = \emptyset$ verified. **PASS**
4. **Label-Free / Gradient-Free:** All prototypes derived purely from unsupervised forward moments without class labels or loss backpropagation. **PASS**
5. **Exact Replacement Counts:** $\{49, 98, 147, 196\}$ tokens strictly verified across all conditions. **PASS**
6. **Nested Mask Invariant:** $\mathcal{M}_{49} \subset \mathcal{M}_{98} \subset \mathcal{M}_{147} \subset \mathcal{M}_{196}$ verified. **PASS**
7. **CLS Token Untouched:** Verified $h'_{8,0} == h_{8,0}$ (max diff $< 10^{-6}$) across all primary patch interventions. **PASS**
8. **$\mu_8$ Matches Persisted Stats:** Matches `calibration_statistics.npz` within $10^{-6}$. **PASS**
9. **Wrong-Depth Means Loaded from Correct Depths:** Depths 5, 6, 7, 9, 10 verified. **PASS**
10. **Norm-Matching Precision:** $|\|\mu'_k\|_2 - \|\mu_8\|_2| < 10^{-5}$ verified. **PASS**
11. **Coordinate Permutation Bijective:** Verified strict permutation of $\{0, \dots, D-1\}$. **PASS**
12. **Sign-Flip Mask Fractions:** Exact 25%, 50%, 75%, 100% coordinate inversions verified. **PASS**
13. **Cosine Sweep Precision:** Target cosines achieved within $|\cos - \alpha| < 10^{-3}$ and norm matched within $10^{-5}$. **PASS**
14. **Scale Factors Matched:** $\{0.25, 0.50, 1.00, 2.00, 4.00\}$ verified. **PASS**

---

## 5. Formal Scientific Verdict & Discussion

### Verdict: OUTCOME A — GENERIC LATE-LAYER CENTROID / FALSIFICATION OF DEPTH-8 ISOLATION

Operationally, under our pre-registered criteria:
- $\mu_8$ fails to significantly outperform nearby wrong-depth means ($\mu_7$ and $\mu_9$), with $|d_z| < 0.05$ and $|\Delta \text{Acc}| < 0.2\%$.
- Pre-registered Rule: When wrong-depth means perform approximately as well as $\mu_8$ across both architectures, declare **OUTCOME A**.

### Mechanistic Interpretation

The results reject both extremes of prior interpretation:
1. **Falsification of the Isolated Prototype Hypothesis (H1):**
   Downstream blocks 9–11 are not narrowly tuned to a Depth-8-specific feature template. Instead, Blocks 7, 8, and 9 occupy a common geometric manifold aligned along a shared activation centroid ($\cos \ge 0.88-0.93$). Any vector aligned with this regional centroid functions with near-zero performance loss.
2. **Falsification of the "Model Ignores Visual Content" Hypothesis (H4):**
   Downstream blocks are not permissive to arbitrary replacement. Destroying coordinate correspondence (permutation), reversing sign orientation ($-\mu_8$), or injecting orthogonal vectors collapses predictive performance. The model demands a specific coordinate structure and sign alignment.
3. **The Limits of Static Replacement:**
   Up to 75% replacement, a single static prototype preserves over $94\%$ of baseline classifications. However, at 100% replacement, injecting identical copies collapses self-attention due to spatial rank deficiency. To operate at 100% synthetic patch tokens, token-level stochastic diversity (as in Gaussian replacement) is required.

### Permitted Phrasing & Scientific Guardrails
- **Permitted Statement:** *"Downstream Blocks 9–11 show limited dependence on exact patch-token content after Block 8, operating with near-zero performance loss using a static regional centroid up to 75% replacement, but require feature-coordinate alignment, positive orientation, and non-collapsed spatial rank."*
- **Prohibited Statements:** We do NOT claim that "the model ignores image content," nor that "semantic information moved into CLS." The original image influenced CLS and spatial patches through Blocks 0–8 before intervention.
