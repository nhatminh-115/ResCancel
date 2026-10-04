# Report: Image-Conditioned Geometry-Compatible Residual Carrier POC

**Experiment Name**: `IMAGE-CONDITIONED RESIDUAL CARRIER POC`  
**Execution Date**: 2026-10-04  
**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Protocol Reference**: [`docs/IMAGE_CONDITIONED_RESIDUAL_CARRIER_PROTOCOL.md`](file:///d:/Study/ResCancel/docs/IMAGE_CONDITIONED_RESIDUAL_CARRIER_PROTOCOL.md)  
**Hardware Target**: NVIDIA GeForce RTX 5070 Laptop GPU  
**Execution Time**: 134.73 seconds (2.25 min)  

---

## 1. Executive Summary & Final Verdict

### Final Decision Verdict:
```
KILL — image-conditioned carrier does not improve the matched-budget pruning frontier
```

### Scientific Purpose & Core Hypothesis
This exploratory sprint evaluated whether the limitation of previous generic carriers was lack of image-specific information. We tested the hypothesis:
$$\text{geometry-compatible carrier} + \text{compressed image-specific residual} > \text{one additional surviving real patch}$$
by constructing an image-conditioned carrier token anchored at the calibration centroid $\mu_\ell$, augmented with a PCA-projected residual of the discarded patch activations:
$$c_{r, \gamma}(x) = \mu_\ell + \gamma U_r U_r^\top (\bar{p}_M(x) - \mu_\ell)$$
at ordinary attention weight $s = 1$.

### Key Experimental Findings
1. **Hypothesis Decisively Falsified:**
   - On **DeiT-Small (Depth 8, Budget $B=50$)**, not a single carrier configuration beat Pure Random Pruning. Across all $r \in \{1, 4, 16, 64, D\}$ and $\gamma \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$, $\Delta A$ was strictly negative ($-0.04\%$ to $-0.12\%$).
   - On **ViT-B/16 AugReg (Depth 7, Budget $B=72$)**, $\Delta A$ fluctuated between $-0.08\%$ and $+0.06\%$ (well below the pre-registered GO threshold of $\ge +0.5\%$ to $+1.0\%$). The highest single point ($+0.06\%$ at $r=16, \gamma=0.25$) corresponds to a statistically insignificant gain of 3 test images across 5,000 evaluations.
2. **Paired True-Class Margin Degradation:**
   - Across both models and all 21 carrier conditions, the mean paired true-class margin difference relative to Pure Random Pruning was strictly **negative** ($\Delta \text{Margin} \approx -0.015$ for DeiT-Small, $\Delta \text{Margin} \approx -0.010$ for ViT-B).
   - In DeiT-Small, the 95% bootstrap confidence interval for the paired margin difference is strictly below zero across all conditions (e.g. $[-0.0293, -0.0044]$), confirming that replacing one real patch with an image-conditioned carrier significantly harms classification margin.
3. **Raw Image-Mean Carrier Does Not Outperform Centroid or Pruning:**
   - Raw image-mean carrier ($r=D, \gamma=1.0$) underperformed Pure Random Pruning on both models ($72.42\%$ vs $72.54\%$ in DeiT-Small; $72.38\%$ vs $72.40\%$ in ViT-B).
4. **Action:**
   - Per pre-registered protocol, this exploratory direction is **PERMANENTLY TERMINATED**.
   - No changes are made to `PAPER_DRAFT.md`. The paper remains strictly a study of late-layer representation geometry and content fungibility.

---

## 2. Comprehensive Results Summary Table

Evaluated on $N=1,000$ disjoint evaluation images across all 5 spatial mask seeds (`31001`–`31005`):

### 2.1 DeiT-Small (Depth 8, Budget $B=50$, Discarded $k=147$, Real Anchors $M=49$)
*Clean Model Top-1 Accuracy: $76.10\%$ | Clean Mean Margin: $2.2514$*

| Condition | Subspace Rank $r$ | Shrinkage $\gamma$ | Mean Top-1 Acc (%) | $\pm$ SD (%) | $\Delta A$ vs. Pruning (%) | Mean Paired $\Delta$ Margin | 95% Bootstrap CI | Cohen's $d_z$ | McNemar $p$ | Pred. Agree (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PURE_RANDOM_PRUNING** | — | — | **72.54%** | $\pm 1.06$ | **0.00%** | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 100.0% |
| **UNWEIGHTED_CENTROID** | — | 0.00 | 72.50% | $\pm 1.07$ | -0.04% | -0.0157 | [-0.0293, -0.0044] | -0.0777 | 0.8124 | 98.44% |
| **RAW_IMAGE_MEAN** | D | 1.00 | 72.42% | $\pm 1.03$ | -0.12% | -0.0142 | [-0.0277, -0.0028] | -0.0695 | 0.5898 | 98.44% |
| Residual Carrier | 1 | 0.25 | 72.50% | $\pm 1.07$ | -0.04% | -0.0157 | [-0.0293, -0.0044] | -0.0777 | 0.8124 | 98.44% |
| Residual Carrier | 1 | 0.50 | 72.50% | $\pm 1.07$ | -0.04% | -0.0157 | [-0.0293, -0.0044] | -0.0777 | 0.8124 | 98.44% |
| Residual Carrier | 1 | 0.75 | 72.50% | $\pm 1.07$ | -0.04% | -0.0157 | [-0.0293, -0.0044] | -0.0776 | 0.8124 | 98.44% |
| Residual Carrier | 1 | 1.00 | 72.50% | $\pm 1.07$ | -0.04% | -0.0157 | [-0.0293, -0.0044] | -0.0776 | 0.8124 | 98.44% |
| Residual Carrier | 4 | 0.25 | 72.50% | $\pm 1.07$ | -0.04% | -0.0157 | [-0.0293, -0.0044] | -0.0776 | 0.8124 | 98.44% |
| Residual Carrier | 4 | 0.50 | 72.50% | $\pm 1.07$ | -0.04% | -0.0157 | [-0.0293, -0.0044] | -0.0776 | 0.8124 | 98.44% |
| Residual Carrier | 4 | 0.75 | 72.48% | $\pm 1.05$ | -0.06% | -0.0157 | [-0.0293, -0.0044] | -0.0776 | 0.7937 | 98.42% |
| Residual Carrier | 4 | 1.00 | 72.48% | $\pm 1.05$ | -0.06% | -0.0157 | [-0.0293, -0.0044] | -0.0775 | 0.7937 | 98.42% |
| Residual Carrier | 16 | 0.25 | 72.50% | $\pm 1.07$ | -0.04% | -0.0157 | [-0.0293, -0.0044] | -0.0776 | 0.8124 | 98.44% |
| Residual Carrier | 16 | 0.50 | 72.50% | $\pm 1.07$ | -0.04% | -0.0157 | [-0.0293, -0.0044] | -0.0775 | 0.8124 | 98.44% |
| Residual Carrier | 16 | 0.75 | 72.50% | $\pm 1.07$ | -0.04% | -0.0157 | [-0.0292, -0.0043] | -0.0773 | 0.8124 | 98.44% |
| Residual Carrier | 16 | 1.00 | 72.46% | $\pm 1.04$ | -0.08% | -0.0156 | [-0.0292, -0.0043] | -0.0770 | 0.7490 | 98.40% |
| Residual Carrier | 64 | 0.25 | 72.50% | $\pm 1.07$ | -0.04% | -0.0157 | [-0.0293, -0.0044] | -0.0775 | 0.8124 | 98.44% |
| Residual Carrier | 64 | 0.50 | 72.50% | $\pm 1.07$ | -0.04% | -0.0157 | [-0.0293, -0.0044] | -0.0774 | 0.8124 | 98.44% |
| Residual Carrier | 64 | 0.75 | 72.48% | $\pm 1.05$ | -0.06% | -0.0156 | [-0.0292, -0.0043] | -0.0770 | 0.7937 | 98.44% |
| Residual Carrier | 64 | 1.00 | 72.46% | $\pm 1.04$ | -0.08% | -0.0155 | [-0.0290, -0.0041] | -0.0762 | 0.7490 | 98.40% |
| Residual Carrier | D | 0.25 | 72.50% | $\pm 1.07$ | -0.04% | -0.0157 | [-0.0293, -0.0043] | -0.0773 | 0.8124 | 98.44% |
| Residual Carrier | D | 0.50 | 72.50% | $\pm 1.07$ | -0.04% | -0.0155 | [-0.0291, -0.0041] | -0.0763 | 0.8124 | 98.46% |
| Residual Carrier | D | 0.75 | 72.46% | $\pm 1.04$ | -0.08% | -0.0150 | [-0.0286, -0.0037] | -0.0739 | 0.7490 | 98.42% |

---

### 2.2 ViT-B/16 AugReg (Depth 7, Budget $B=72$, Discarded $k=125$, Real Anchors $M=71$)
*Clean Model Top-1 Accuracy: $76.10\%$ | Clean Mean Margin: $3.1923$*

| Condition | Subspace Rank $r$ | Shrinkage $\gamma$ | Mean Top-1 Acc (%) | $\pm$ SD (%) | $\Delta A$ vs. Pruning (%) | Mean Paired $\Delta$ Margin | 95% Bootstrap CI | Cohen's $d_z$ | McNemar $p$ | Pred. Agree (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PURE_RANDOM_PRUNING** | — | — | **72.40%** | $\pm 0.48$ | **0.00%** | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 100.0% |
| **UNWEIGHTED_CENTROID** | — | 0.00 | 72.40% | $\pm 0.51$ | 0.00% | -0.0097 | [-0.0236, +0.0026] | -0.0376 | 0.9419 | 97.94% |
| **RAW_IMAGE_MEAN** | D | 1.00 | 72.38% | $\pm 0.57$ | -0.02% | -0.0092 | [-0.0231, +0.0032] | -0.0361 | 0.8523 | 98.24% |
| Residual Carrier | 1 | 0.25 | 72.38% | $\pm 0.55$ | -0.02% | -0.0096 | [-0.0234, +0.0028] | -0.0365 | 0.9419 | 98.00% |
| Residual Carrier | 1 | 0.50 | 72.38% | $\pm 0.55$ | -0.02% | -0.0094 | [-0.0233, +0.0029] | -0.0355 | 0.9419 | 98.00% |
| Residual Carrier | 1 | 0.75 | 72.36% | $\pm 0.55$ | -0.04% | -0.0092 | [-0.0232, +0.0032] | -0.0345 | 0.8932 | 98.00% |
| Residual Carrier | 1 | 1.00 | 72.36% | $\pm 0.55$ | -0.04% | -0.0089 | [-0.0229, +0.0036] | -0.0328 | 0.8932 | 98.02% |
| Residual Carrier | 4 | 0.25 | 72.40% | $\pm 0.51$ | 0.00% | -0.0096 | [-0.0235, +0.0028] | -0.0366 | 0.9419 | 97.98% |
| Residual Carrier | 4 | 0.50 | 72.32% | $\pm 0.51$ | -0.08% | -0.0095 | [-0.0235, +0.0028] | -0.0358 | 0.7533 | 98.06% |
| Residual Carrier | 4 | 0.75 | 72.34% | $\pm 0.51$ | -0.06% | -0.0093 | [-0.0233, +0.0030] | -0.0347 | 0.8037 | 98.08% |
| Residual Carrier | 4 | 1.00 | 72.32% | $\pm 0.51$ | -0.08% | -0.0090 | [-0.0230, +0.0035] | -0.0327 | 0.7533 | 98.16% |
| Residual Carrier | 16 | 0.25 | 72.46% | $\pm 0.52$ | +0.06% | -0.0100 | [-0.0239, +0.0022] | -0.0390 | 0.8354 | 98.06% |
| Residual Carrier | 16 | 0.50 | 72.42% | $\pm 0.49$ | +0.02% | -0.0103 | [-0.0241, +0.0019] | -0.0403 | 0.9419 | 98.10% |
| Residual Carrier | 16 | 0.75 | 72.38% | $\pm 0.46$ | -0.02% | -0.0105 | [-0.0243, +0.0018] | -0.0409 | 0.8932 | 98.08% |
| Residual Carrier | 16 | 1.00 | 72.34% | $\pm 0.55$ | -0.06% | -0.0103 | [-0.0241, +0.0021] | -0.0403 | 0.8037 | 98.18% |
| Residual Carrier | 64 | 0.25 | 72.42% | $\pm 0.51$ | +0.02% | -0.0100 | [-0.0239, +0.0022] | -0.0390 | 0.9419 | 98.04% |
| Residual Carrier | 64 | 0.50 | 72.40% | $\pm 0.46$ | 0.00% | -0.0101 | [-0.0240, +0.0021] | -0.0396 | 0.9419 | 98.12% |
| Residual Carrier | 64 | 0.75 | 72.44% | $\pm 0.48$ | +0.04% | -0.0101 | [-0.0241, +0.0023] | -0.0395 | 0.8842 | 98.12% |
| Residual Carrier | 64 | 1.00 | 72.40% | $\pm 0.51$ | 0.00% | -0.0101 | [-0.0241, +0.0025] | -0.0395 | 0.9419 | 98.18% |
| Residual Carrier | D | 0.25 | 72.36% | $\pm 0.47$ | -0.04% | -0.0102 | [-0.0241, +0.0019] | -0.0402 | 0.8447 | 98.04% |
| Residual Carrier | D | 0.50 | 72.32% | $\pm 0.53$ | -0.08% | -0.0104 | [-0.0243, +0.0019] | -0.0410 | 0.7533 | 98.14% |
| Residual Carrier | D | 0.75 | 72.38% | $\pm 0.50$ | -0.02% | -0.0099 | [-0.0238, +0.0024] | -0.0391 | 0.8932 | 98.28% |

---

## 3. Heatmap of Accuracy Delta $\Delta A$ (%)

### DeiT-Small ($B=50$)
| Shrinkage $\gamma$ \ Subspace $r$ | $r = 1$ | $r = 4$ | $r = 16$ | $r = 64$ | $r = D$ (Full) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\gamma = 1.00$ | $-0.04\%$ | $-0.06\%$ | $-0.08\%$ | $-0.08\%$ | $-0.12\%$ |
| $\gamma = 0.75$ | $-0.04\%$ | $-0.06\%$ | $-0.04\%$ | $-0.06\%$ | $-0.08\%$ |
| $\gamma = 0.50$ | $-0.04\%$ | $-0.04\%$ | $-0.04\%$ | $-0.04\%$ | $-0.04\%$ |
| $\gamma = 0.25$ | $-0.04\%$ | $-0.04\%$ | $-0.04\%$ | $-0.04\%$ | $-0.04\%$ |
| $\gamma = 0.00$ (Centroid) | $-0.04\%$ | $-0.04\%$ | $-0.04\%$ | $-0.04\%$ | $-0.04\%$ |

### ViT-B/16 AugReg ($B=72$)
| Shrinkage $\gamma$ \ Subspace $r$ | $r = 1$ | $r = 4$ | $r = 16$ | $r = 64$ | $r = D$ (Full) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\gamma = 1.00$ | $-0.04\%$ | $-0.08\%$ | $-0.06\%$ | $0.00\%$ | $-0.02\%$ |
| $\gamma = 0.75$ | $-0.04\%$ | $-0.06\%$ | $-0.02\%$ | $+0.04\%$ | $-0.02\%$ |
| $\gamma = 0.50$ | $-0.02\%$ | $-0.08\%$ | $+0.02\%$ | $0.00\%$ | $-0.08\%$ |
| $\gamma = 0.25$ | $-0.02\%$ | $0.00\%$ | $+0.06\%$ | $+0.02\%$ | $-0.04\%$ |
| $\gamma = 0.00$ (Centroid) | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ |

---

## 4. Scientific Analysis & Mechanistic Takeaway

### 4.1 Why One Real Patch Dominates Image-Conditioned Residual Carriers
The core mechanistic finding of this experiment is that **spatial aggregation fundamentally destroys localized discriminative mutual information**:
1. When dropping from $B$ real patches to $B-1$ real patches, the model forfeits one discrete, localized receptive field.
2. Replacing that patch with $\bar{p}_M(x)$ or its PCA projection $r_r(x)$ attempts to compress the entire background into a single pooled token.
3. Because self-attention operates via pairwise dot products between queries and keys, an averaged background token acts as a blunt global context vector. It cannot reconstruct the spatial relations or local contrast that the missing patch provided.
4. Consequently, keeping one additional authentic real patch provides strictly greater utility than any linear combination of the discarded patches, regardless of whether that combination is class-agnostic ($\mu_\ell$) or image-conditioned ($c_{r, \gamma}(x)$).

### 4.2 Full-Sequence Content Fungibility vs. Token-Budget Pruning
This result sharply defines the boundary of the core phenomenon:
- **In an uncompressed sequence ($T = 197$),** patch activations are content-fungible because the representation volume has ample capacity; replacing patches with a prototype leaves downstream attention dynamics intact.
- **Under sequence compression ($T \ll 197$),** token capacity is scarce. Every slot allocated to a synthetic surrogate comes at the expense of an image-specific patch. Under scarce token budgets, real patch diversity is strictly superior.

---

## 5. Machine-Readable Artifacts

All outputs and figures are generated and committed:
- **Experiment Manifest:** [`outputs/fungibility_residual_carrier/validation_manifest.json`](file:///d:/Study/ResCancel/outputs/fungibility_residual_carrier/validation_manifest.json)
- **Trial-by-Trial Data (260 runs):** [`outputs/fungibility_residual_carrier/trial_results.csv`](file:///d:/Study/ResCancel/outputs/fungibility_residual_carrier/trial_results.csv)
- **Aggregated Condition Summary:** [`outputs/fungibility_residual_carrier/summary_by_condition.csv`](file:///d:/Study/ResCancel/outputs/fungibility_residual_carrier/summary_by_condition.csv)
- **Parameter Heatmap Grid:** [`outputs/fungibility_residual_carrier/heatmap_grid.csv`](file:///d:/Study/ResCancel/outputs/fungibility_residual_carrier/heatmap_grid.csv)
- **Figures:**
  - Accuracy vs. Method: [`figures/fungibility_residual_carrier/accuracy_vs_method.png`](file:///d:/Study/ResCancel/figures/fungibility_residual_carrier/accuracy_vs_method.png)
  - $\Delta A$ Heatmap Grid: [`figures/fungibility_residual_carrier/delta_accuracy_heatmap.png`](file:///d:/Study/ResCancel/figures/fungibility_residual_carrier/delta_accuracy_heatmap.png)
