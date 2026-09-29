# Protocol: Geometry-Diversity Bank Proof-of-Concept (POC)

**Experiment Name**: `GEOMETRY-DIVERSITY BANK POC`  
**Execution Date**: 2026-09-29  
**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Status**: Pre-registered and Frozen  

---

## 1. Scientific Purpose & Core Question

The dense-fraction and weighted-carrier experiments established two core facts:
1. **Fungibility and Exact Equivalence:** Late-layer spatial patch activations can be replaced extensively by generic calibration prototypes without destroying classification, and duplicate identical surrogate tokens admit an exact mathematical sequence collapse into a single multiplicity-weighted carrier token ($s=m$, logit bias $+\log(s_j)$) down to numerical tolerance ($4.49 \times 10^{-5}$ max logit error, $100\%$ agreement).
2. **Compression Reality:** However, at a matched downstream token budget $B$, simple **Random Pruning** (keeping $B$ real patches and dropping the rest with no carrier) matches or outperforms the Weighted Centroid Carrier. Preserving the full uncompressed attention mass of fungible surrogate tokens ($+\log(m)$) is unnecessary and actively detrimental once sequence length shrinks, because it starves attention to the surviving real patches that carry image-specific discriminative signal.

This raises the final mechanistic-application question:
> *"At the same downstream token budget $B$, can a small set of class-agnostic synthetic tokens that explicitly preserve learned late-layer geometry and token-to-token diversity outperform simple Random Pruning?"*

If the answer is **NO**, this application direction is permanently terminated.

---

## 2. Experimental Design & Controlled Comparisons

### 2.1 Frozen Models and Target Intervention Depths
Strictly the primary frozen checkpoints and intervention depths:
1. **DeiT-Tiny:** Depth 8 ($D=192$, $N_{\text{patch}}=196$)
2. **DeiT-Small:** Depth 8 ($D=384$, $N_{\text{patch}}=196$)
3. **ViT-B/16 AugReg:** Depth 7 ($D=768$, $N_{\text{patch}}=196$)
4. **DINOv2 ViT-S/14:** Depth 9 ($D=384$, $N_{\text{patch}}=256$)

No depth searches or hyperparameter tuning.

### 2.2 Frozen Data Splits & Deterministic Spatial Masks
- **Calibration Split:** Canonical ImageNet-1k subset ($N=1,000$ images, seed 9101).
- **Evaluation Split:** Canonical ImageNet-1k subset ($N=1,000$ images, seed 9201). Strictly disjoint, zero overlap.
- **Spatial Mask Permutations:** Reusing the exact 5 deterministic seeds from the dense sweep:
  `31001, 31002, 31003, 31004, 31005`.
- **Anchor Selection:** Strictly deterministic surviving prefix from the spatial mask permutation. No attention-based, norm-based, or post-hoc saliency anchor selection.

### 2.3 Downstream Token Budgets ($B$)
Target budgets chosen in the previously established useful operating regime:
- **DeiT-Tiny:** $B \in \{45, 73\}$
- **DeiT-Small:** $B \in \{28, 52\}$
- **ViT-B/16:** $B \in \{73, 112\}$
- **DINOv2 ViT-S/14:** $B \in \{184, 199\}$

*(Note: $B$ denotes the number of downstream patch tokens. Total downstream sequence length including [CLS] is strictly $1 + B$. All comparisons at budget $B$ must have identical sequence length $1 + B$.)*

### 2.4 Bank Sizes ($K$) and Real Anchors ($M$)
For each patch budget $B$:
- Bank size: $K_{\text{synth}} \in \{4, 8, 16\}$, subject to $K_{\text{synth}} < B$.
- Number of surviving real patches: $M_{\text{real}} = B - K_{\text{synth}}$.
- Surviving real patches are taken as the first $M_{\text{real}}$ tokens of the deterministic surviving $B$ tokens for that mask permutation.

---

## 3. Evaluated Methods at Matched Token Budget ($B$)

At each budget $B$, exactly five conditions are evaluated at identical downstream token count $1 + B$:

1. **A. RANDOM PRUNING (Primary Baseline to Beat):**
   - Retain $B$ real patches (and [CLS]).
   - $M_{\text{real}} = B, K = 0$.
   - No synthetic carriers.

2. **B. REAL + SINGLE CENTROID:**
   - Retain $B - 1$ real patches + 1 unweighted calibration centroid $\mu$.
   - $M_{\text{real}} = B - 1, K = 1$.
   - Ordinary standard transformer attention (no log-multiplicity bias).

3. **C. REAL + PCA BANK:**
   - Retain $M_{\text{real}} = B - K$ real patches + $K$ PCA synthetic tokens.
   - PCA computed on calibration patch activations at target depth:
     $\mu \in \mathbb{R}^D$, top eigenvectors $v_i$, eigenvalues $\lambda_i$.
   - Symmetric tokens along top $K/2$ principal directions:
     $t_{2i-1} = \mu + \sqrt{\lambda_i} v_i$  
     $t_{2i} = \mu - \sqrt{\lambda_i} v_i$  
     for $i = 1, \dots, K/2$ (with natural scale $a_i = 1$).

4. **D. REAL + K-MEANS BANK:**
   - Retain $M_{\text{real}} = B - K$ real patches + $K$ cluster centroids $c_1, \dots, c_K$.
   - $K$-means fitted on calibration patch activations at target depth with $k = K$.
   - Each cluster centroid is used as one ordinary unweighted token.

5. **E. REAL + RANDOM-DIRECTION BANK (Geometric Ablation Control):**
   - Retain $M_{\text{real}} = B - K$ real patches + $K$ synthetic tokens along random orthonormal directions $u_i \in \mathbb{R}^D$ scaled by matched PCA eigenvalues:
     $t_{2i-1} = \mu + \sqrt{\lambda_i} u_i$  
     $t_{2i} = \mu - \sqrt{\lambda_i} u_i$  
   - Directly tests whether the learned orientation of representation geometry provides advantage over isotropic/random diversity.

---

## 4. Evaluation Metrics & Comparison Protocol

For every combination of `(model, depth, budget B, bank size K, mask seed, method)`:
- **Top-1 Accuracy:** Correct classifications on the 1,000 evaluation images.
- **Mean True-Class Margin:** $\text{Logit}(y_{\text{true}}) - \max_{c \ne y} \text{Logit}(c)$.
- **Accuracy Delta vs. Random Pruning:**  
  $$\Delta \text{Acc} = \text{Top-1}(\text{method}) - \text{Top-1}(\text{RandomPruning}_{B})$$
- **Margin Delta vs. Random Pruning:**  
  $$\Delta \text{Margin} = \text{Margin}(\text{method}) - \text{Margin}(\text{RandomPruning}_{B})$$
- **Prediction Flip Rate vs. Clean:** Fraction of samples where prediction differs from full clean model.
- **Cross-Mask Statistics:** Mean and standard deviation across all 5 spatial mask seeds.

---

## 5. Pre-registered Decision Rules

1. **`PROMISING`**:
   - At least one geometry/diversity bank (PCA or K-means) outperforms Random Pruning at matched token budget by **$\ge +1.0$ percentage point Top-1 accuracy** on at least **TWO model families**;
   - Does not materially lose on the other models ($\Delta \text{Acc} \ge -0.5\%$);
   - Shows consistent positive $\Delta \text{Margin}$.
   - *Action:* Triggers end-to-end physical CUDA latency benchmarking on RTX 5070 GPU for the winning bank configuration.

2. **`MIXED`**:
   - Gains occur on one model architecture but fail to generalize to others.
   - *Action:* No latency benchmarking; documented as architecture-specific nuance without general utility.

3. **`KILL APPLICATION`**:
   - Simple Random Pruning matches or beats both PCA and K-means banks across the tested budgets ($\Delta \text{Acc} \le 0$ on average).
   - *Action:* Terminate the application direction immediately. Do NOT run physical latency benchmarks. Conclude that scarce token slots are better allocated to surviving real image patches than generic synthetic carriers.
