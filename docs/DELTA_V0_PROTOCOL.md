# DELTA TRANSPORT V0: Frozen Experimental Protocol

**Document Status:** FROZEN  
**Date:** 2026-09-28  
**Repository:** `https://github.com/nhatminh-115/ResCancel`  
**Research Line:** Delta Transport V0 — Oracle Falsification of Spatially Selective Residual-Delta Reuse in Vision Transformers  

---

## 1. Scientific Objective & Research Question

In standard Vision Transformers (ViT), block updates accumulate sequentially:
$$h_{l+1} = h_l + \Delta_l \quad \text{where} \quad \Delta_l = h_{l+1} - h_l$$

Recent work demonstrates that historical residual deltas provide un-diluted directional information compared to cumulative hidden states. The core question investigated in this research line is:

> **Do different spatial patch tokens benefit from reusing DIFFERENT historical residual deltas across depth?**  
> Specifically: At equal perturbation / transport budget, does label-aware **PER-TOKEN selection** over historical block deltas produce meaningfully better downstream predictions than the best **GLOBAL depth selection** shared by every patch token?

This experiment is a **mechanistic oracle falsification test**. It uses ground-truth labels and exact gradients to compute an upper bound on what *any* spatial routing mechanism could theoretically achieve. If even an all-knowing oracle cannot produce a meaningful, statistically significant advantage for token-wise routing over global depth selection, the per-token hypothesis is killed immediately before any router architecture is designed or trained.

---

## 2. Experimental Setup & Frozen Configurations

### 2.1 Models
- **DeiT-Tiny:** `deit_tiny_patch16_224` (timm pretrained, 12 blocks, embed dim $d=192$, heads=3)
- **DeiT-Small:** `deit_small_patch16_224` (timm pretrained, 12 blocks, embed dim $d=384$, heads=6)
- **Backbone Weights:** Strictly frozen throughout all experiments. Zero gradient updates or fine-tuning.

### 2.2 Dataset
- **ImageNet-1k Validation Subset:** $N = 1,000$ images, 1 per class (stratified), deterministic seed $42$.
- **Preprocessing:** Standard bicubic resize to 256, center crop to 224, ImageNet normalization.

### 2.3 Residual Delta Definitions
- We measure **block-level residual deltas** across the 12 transformer encoder blocks (indexed $l = 0, \dots, 11$):
  $$\Delta_{l,t} = h_{l+1,t} - h_{l,t}$$
  where $h_{l,t} \in \mathbb{R}^d$ is the representation of token $t$ entering block $l$, and $h_{l+1,t}$ is the representation emerging from block $l$ (after multi-head self-attention and MLP residual additions).
- For an image with 196 patch tokens and 1 CLS token, patch tokens are indexed $t \in \{1, \dots, 196\}$ and CLS is $t = 0$.

### 2.4 Injection Architecture & Candidate Source Bank
- **Primary Injection Point:** Input to **Block 9** (i.e., state $h_8$ emerging from Block 8).
- **Candidate Historical Source Bank:** Blocks $l \in \{0, 1, 2, 3, 4, 5, 6, 7\}$.
- **Downstream Propagation:** Blocks 9, 10, and 11 (the final three blocks) remain unmodified and propagate the injected representations into the final classification head.
- *Strict Rule:* No search over injection depths in the primary experiment. Secondary injection depths may be tested only as sensitivity tests.

---

## 3. Label-Aware Oracle Formulation

For an image with ground-truth class label $y \in \{0, \dots, 999\}$:

### 3.1 Margin & Gradient Definition
1. Pass input through Blocks $0 \dots 11$ to obtain baseline logits $z \in \mathbb{R}^{1000}$.
2. Compute the true-class logit margin:
   $$m = z_y - \max_{c \neq y} z_c$$
3. Compute the analytical gradient of margin $m$ with respect to the patch token representations at the injection point (input to Block 9):
   $$g_t = \frac{\partial m}{\partial h_{8,t}} \in \mathbb{R}^d \quad \text{for each patch } t \in \{1, \dots, 196\}$$

### 3.2 Normalized Direction & First-Order Usefulness
For every historical source layer $l \in \{0, \dots, 7\}$ and spatial patch $t$:
$$u_{l,t} = \frac{\Delta_{l,t}}{\|\Delta_{l,t}\|_2 + \epsilon} \quad (\epsilon = 10^{-7})$$
The first-order linear usefulness score is:
$$s_{l,t} = g_t^T u_{l,t}$$
This score estimates whether injecting the directional delta from layer $l$ into patch $t$ will positively align with increasing the correct-class logit margin.

### 3.3 Perturbation & Transport Normalization
All intervention policies inject a normalized directional update with strictly identical transport magnitude:
$$h'_{8,t} = h_{8,t} + \gamma \cdot \|h_{8,t}\|_2 \cdot u_{\text{selected},t}$$
- **Primary Transport Budget:** $\gamma = 0.05$.
- **Sensitivity Budgets:** $\gamma \in \{0.02, 0.10\}$ (labelled secondary).
- **CLS Token Invariant:** $h'_{8,\text{CLS}} = h_{8,\text{CLS}}$ (CLS is untouched at injection).
- All policies pass $h'_{8}$ through Blocks 9, 10, and 11 to obtain the true resulting logits $z'$.

---

## 4. Compared Policies

1. **Policy A — BASELINE:**
   No historical delta added ($\gamma = 0$). Unmodified forward pass.
2. **Policy B — MOST-RECENT:**
   Every patch token receives the normalized delta from Block 7:
   $$u_{\text{selected},t} = u_{7,t} \quad \forall t$$
3. **Policy C — RANDOM-TOKENWISE:**
   Each patch token independently selects a uniform random historical source layer from $\{0, \dots, 7\}$. Evaluated over three frozen seeds: `2501, 2502, 2503`.
4. **Policy D — GLOBAL ORACLE:**
   Using the ground-truth margin gradient, select ONE historical source layer $l^*$ for the entire image that maximizes the aggregate first-order score across all patch tokens:
   $$l^* = \arg\max_{l \in \{0, \dots, 7\}} \sum_{t=1}^{196} s_{l,t}$$
   Every patch token receives the same source: $u_{\text{selected},t} = u_{l^*,t}$.
5. **Policy E — TOKENWISE ORACLE (Key Condition):**
   For every patch token independently:
   $$l^*_t = \arg\max_{l \in \{0, \dots, 7\}} s_{l,t}$$
   Each patch token receives its independently optimal historical delta: $u_{\text{selected},t} = u_{l^*_t, t}$.

### Key Experimental Invariants
- Alter the exact same number of patch tokens (196).
- Exact same per-token perturbation L2 norm ($\gamma \|h_t\|_2$).
- CLS untouched at injection.
- Propagate through the exact same remaining pretrained blocks (9, 10, 11).

---

## 5. Statistical Framework & Primary Estimands

The primary comparison is **TOKENWISE ORACLE vs. GLOBAL ORACLE**.

- **Unit of Independence:** The image ($N = 1,000$ independent observations per architecture).
- **Primary Estimand:** Paired margin difference:
  $$\Delta m_i = m_{i,\text{tokenwise}} - m_{i,\text{global}}$$
- **Hypothesis Testing:**
  1. **Bootstrap 95% Confidence Interval:** 10,000 resamples of paired differences $\Delta m_i$.
  2. **Paired Difference Tests:** Two-sided paired t-test and Wilcoxon signed-rank test.
  3. **Effect Size:** Paired Cohen's $d_z = \frac{\bar{D}}{s_D}$.
  4. **Categorical Correctness:** Top-1 accuracy difference and McNemar's exact test for paired prediction flips.

### Spatial Diversity Metrics
To verify whether the Tokenwise Oracle exhibits genuine spatial heterogeneity or trivial collapse:
- **Layer Selection Histogram:** Aggregate distribution of chosen layers $l \in \{0, \dots, 7\}$.
- **Selection Entropy:** $H = -\sum_{l=0}^7 p(l) \log_2 p(l)$ (bits, theoretical max $3.0$).
- **Dominant Fraction:** Fraction of patch tokens selecting the globally preferred layer $l^*$.
- **Distinct Layers Used:** Number of unique historical layers chosen per image (range 1 to 8).
- **Spatial Maps:** 2D grid visualizations of chosen layers for representative images.

---

## 6. Pre-Registered Decision Rules

### OUTCOME A — KILL (No Evidence of Spatial Selectivity Advantage)
Kill the per-token delta transport hypothesis if **ANY** of the following hold:
1. Tokenwise oracle fails to beat global oracle on both architectures ($\Delta m \le 0$ on both);
2. Paired Tokenwise-vs-Global margin difference has a bootstrap 95% CI including zero on both models;
3. Paired effect size $d_z < 0.20$ (negligible effect);
4. Tokenwise oracle improves margin relative to global only by degrading clean top-1 accuracy;
5. Tokenwise source choices collapse mostly to the same layer ($H \approx 0$ or dominant fraction $\ge 90\%$).

*Scientific Conclusion:* "No evidence that spatially token-specific historical delta selection provides useful capacity beyond global depth selection."  
*Action:* Kill the idea. Do NOT design or train a router.

### OUTCOME B — GLOBAL DELTA REUSE ONLY
Triggered if historical delta reuse (Global Oracle) improves over baseline, but Tokenwise does not materially outperform Global Oracle.  
*Scientific Conclusion:* "Historical delta reuse across depth may be useful, but spatially selective per-token routing is not supported."  
*Action:* Terminate the per-token architecture.

### OUTCOME C — INTERESTING (Per-Token Spatial Heterogeneity Supported)
Triggered **ONLY** if:
1. Tokenwise beats Global Oracle on **BOTH** DeiT-Tiny and DeiT-Small;
2. Paired bootstrap 95% CI is strictly above zero on both architectures;
3. Paired effect size $d_z \ge 0.20$ on both;
4. Tokenwise does not reduce clean top-1 accuracy by $>0.5$ percentage points relative to baseline;
5. Selected layers show genuine spatial diversity ($H \ge 1.0$, dominant fraction $< 75\%$).

*Scientific Conclusion:* "Evidence supports that spatial patch tokens have distinct, non-trivial historical delta requirements."  
*Action:* Justifies a follow-up V1 study to investigate learned token-level routers.

---

## 7. Programmatic Validation Assertions

Before any scientific verdict is rendered, the pipeline must programmatically assert:
1. Backbone parameters $\theta$ have zero gradient / zero weight change.
2. $\Delta_l = h_{l+1} - h_l$ numerically holds within $10^{-5}$ tolerance.
3. CLS token is strictly untouched at injection ($h'_{8,\text{CLS}} = h_{8,\text{CLS}}$).
4. All policies modify the exact same 196 spatial patch tokens.
5. Perturbation norm is identical across all active policies ($\gamma \|h_t\|_2$).
6. Global Oracle selects exactly one historical source layer per image.
7. Random controls use frozen seeds: 2501, 2502, 2503.
8. Statistical testing is conducted strictly on $N=1,000$ independent image units.
9. Report numbers match saved machine-readable CSVs and JSON manifest exactly.
