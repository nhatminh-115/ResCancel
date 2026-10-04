# Protocol: Image-Conditioned Geometry-Compatible Residual Carrier POC

**Experiment Name**: `IMAGE-CONDITIONED RESIDUAL CARRIER POC`  
**Execution Date**: 2026-10-04  
**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Status**: Pre-registered and Frozen  

---

## 1. Scientific Motivation & Hypothesis

Previous experiments established:
1. **Late Patch Content Fungibility:** Large fractions of late-layer spatial patch activations can be replaced by class-agnostic prototypes without catastrophic accuracy collapse.
2. **Representation Geometry Constraint:** Valid late replacements must respect learned late-layer feature geometry (PCA directions and natural variance scales).
3. **Compression Reality of Generic Carriers:** At a matched downstream token budget $B$, class-agnostic centroid carriers (whether single-token or multi-token PCA/K-means banks) do **NOT** outperform pure Random Pruning. Generic synthetic carriers lack image-specific mutual information, and allocating token slots to them sacrifices surviving real patches that contain vital discriminative evidence.

This raises the primary falsifiable hypothesis of this exploratory POC:
$$\text{geometry-compatible carrier} + \text{compressed image-specific residual} > \text{one additional surviving real patch}$$

### Core Research Question:
> *"Can an image-conditioned carrier token—constructed by anchoring at the calibration centroid $\mu_\ell$ and adding a PCA-projected residual of the discarded patch activations—outperform pure Random Pruning at the exact same downstream token budget?"*

---

## 2. Experimental Design & Architecture Targets

### 2.1 Primary Models and Operating Points
The initial kill sprint is restricted strictly to two primary architectures at their established intervention depths:
1. **DeiT-Small (Depth 8):**
   - Embedding Dimension: $D = 384$
   - Original Patches: $N = 196$
   - Operating Point: 75% replacement ($k = 147$ discarded patches, $M_{\text{real}} = 49$ surviving real patches)
   - Downstream Spatial Budget: $B = M_{\text{real}} + 1 = 50$ patch tokens
   - Total Downstream Sequence Length: $1 + B = 51$ tokens (including [CLS])
2. **ViT-B/16 AugReg (Depth 7):**
   - Embedding Dimension: $D = 768$
   - Original Patches: $N = 196$
   - Operating Point: 63.8% replacement ($k = 125$ discarded patches, $M_{\text{real}} = 71$ surviving real patches)
   - Downstream Spatial Budget: $B = M_{\text{real}} + 1 = 72$ patch tokens
   - Total Downstream Sequence Length: $1 + B = 73$ tokens (including [CLS])

### 2.2 Canonical Frozen Splits & Mask Permutations
- **Calibration Split:** ImageNet-1k subset ($N = 1,000$ images, seed 9101). Used strictly for estimating $\mu_\ell$ and PCA eigenvectors $U_r$.
- **Evaluation Split:** ImageNet-1k subset ($N = 1,000$ images, seed 9201). Strictly disjoint from calibration (zero overlap).
- **Spatial Mask Permutations:** 5 deterministic seeds from the dense sweep: `31001, 31002, 31003, 31004, 31005`.
- **Anchor Selection Convention:**
  - Let $p_s$ be the permutation of patch indices $0 \dots N-1$.
  - Discarded patch set $M = p_s[:k]$.
  - Surviving real patch set for carrier methods: $M_{\text{real}} = p_s[k:]$ (size $B - 1$).
  - Surviving real patch set for Pure Random Pruning: $p_s[k-1:]$ (size $B$).
  - Pure Random Pruning retains the exact same $B-1$ real patches as the carrier method, plus exactly one additional real patch from $M$. Both conditions have identical downstream sequence length $1 + B$.

---

## 3. Carrier Formulation & Tested Conditions

### 3.1 Mathematical Definition
For an image $x$, let $M$ denote the indices of discarded patches at intervention depth $\ell$.
1. Compute the image-specific mean of discarded patches:
   $$\bar{p}_M(x) = \frac{1}{|M|} \sum_{i \in M} p_i^{(\ell)}(x)$$
2. Compute the image-specific residual relative to the calibration centroid $\mu_\ell$:
   $$r(x) = \bar{p}_M(x) - \mu_\ell$$
3. Project $r(x)$ onto the top $r$ principal directions $U_r \in \mathbb{R}^{D \times r}$ estimated only from the calibration set:
   $$r_r(x) = U_r U_r^\top r(x)$$
   *(When $r = D$, no projection is applied: $r_D(x) = r(x)$).*
4. Construct the carrier token:
   $$c_{r, \gamma}(x) = \mu_\ell + \gamma r_r(x)$$
   with ordinary attention weight $s = 1$ (no $+\log m$ multiplicity bias).

### 3.2 Tested Parameter Grid
- Subspace rank: $r \in \{1, 4, 16, 64, D\}$
- Residual shrinkage factor: $\gamma \in \{0.25, 0.5, 0.75, 1.0\}$
- Centroid control: $\gamma = 0$ (yields $c = \mu_\ell$ for all $r$)
- Total carrier configurations: $5 \times 4 + 1 = 21$ conditions.

### 3.3 Baselines
1. **A. Pure Random Pruning:** Keep $B$ real patches (and [CLS]). Total sequence length $= 1 + B$. Primary baseline to beat.
2. **B. Existing Unweighted Centroid Carrier:** Keep $B - 1$ real patches + 1 static centroid $c = \mu_\ell$ ($\gamma = 0$).
3. **C. Diagnostic Raw Image-Mean Carrier:** Keep $B - 1$ real patches + 1 unprojected mean carrier $c_{\text{mean}}(x) = \bar{p}_M(x)$ ($r = D, \gamma = 1.0$) at $s = 1$.

---

## 4. Evaluation Metrics & Statistical Analysis

For each condition, evaluated on all 1,000 evaluation images across all 5 mask seeds:
- **Top-1 Accuracy:** Mean and standard deviation across seeds.
- **Accuracy Delta vs. Pure Pruning:**
  $$\Delta A = A_{\text{carrier}} - A_{\text{pruning}}$$
- **True-Class Margin:** $\text{Margin}(x) = \text{logit}(y_{\text{true}}) - \max_{j \ne y_{\text{true}}} \text{logit}(j)$
- **Paired Margin Delta:** $\Delta \text{Margin}_i = \text{Margin}_{\text{carrier}}(x_i) - \text{Margin}_{\text{pruning}}(x_i)$
- **Paired Bootstrap 95% Confidence Interval:** 1,000 resamples of paired margin differences.
- **Effect Size:** Cohen's $d_z = \frac{\text{mean}(\Delta \text{Margin})}{\text{std}(\Delta \text{Margin})}$.
- **McNemar Contingency Counts:**
  - $n_{00}$: both incorrect
  - $n_{01}$: pruning correct, carrier incorrect
  - $n_{10}$: carrier correct, pruning incorrect
  - $n_{11}$: both correct
  - McNemar two-sided p-value.

---

## 5. Pre-registered Decision Rules

1. **`GO`**:
   - At least one coherent parameter region (not an isolated point) achieves:
     - $\Delta A \ge +0.5\%$ to $+1.0\%$ over Pure Random Pruning;
     - Statistically significant positive paired margin delta (95% bootstrap CI strictly $> 0$);
     - Consistent positive direction on **BOTH** DeiT-Small and ViT-B/16 across mask seeds.
   - *Next Action:* Freeze the winning rule, expand to DeiT-Tiny and DINOv2, evaluate stronger token-merging baselines (e.g. ToMe).

2. **`KILL`**:
   - $\Delta A \le 0.0\%$ or gains are negligible ($< +0.5\%$);
   - Gains occur on only one architecture;
   - Only one isolated $(r, \gamma)$ setting wins;
   - Pure Random Pruning remains the dominant strategy.
   - *Next Action:* Permanently terminate the image-conditioned carrier direction. Conclude that retaining one additional real image patch is strictly superior to compressing discarded patches into a surrogate carrier. Do not modify `PAPER_DRAFT.md`.
