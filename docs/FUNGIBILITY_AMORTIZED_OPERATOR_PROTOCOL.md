# Research Protocol: Amortized Operator-Aware Token Compression

**Status:** ACTIVE PROTOCOL  
**Date:** October 2026  
**Repository:** [nhatminh-115/Patch-Content-Fungibility](https://github.com/nhatminh-115/Patch-Content-Fungibility)  
**Related Documents:**  
- Confirmatory Report: [FUNGIBILITY_OPERATOR_COMPRESSION_CONFIRMATORY_REPORT.md](file:///d:/Study/ResCancel/docs/FUNGIBILITY_OPERATOR_COMPRESSION_CONFIRMATORY_REPORT.md)  
- Confirmatory Protocol: [FUNGIBILITY_OPERATOR_COMPRESSION_CONFIRMATORY_PROTOCOL.md](file:///d:/Study/ResCancel/docs/FUNGIBILITY_OPERATOR_COMPRESSION_CONFIRMATORY_PROTOCOL.md)  
**Implementation Modules:**  
- [amortized_operator.py](file:///d:/Study/ResCancel/patch_fungibility/amortized_operator.py)  
- [generate_operator_targets.py](file:///d:/Study/ResCancel/scripts/generate_operator_targets.py)  
- [train_amortized_operator.py](file:///d:/Study/ResCancel/scripts/train_amortized_operator.py)  
- [eval_amortized_operator.py](file:///d:/Study/ResCancel/scripts/eval_amortized_operator.py)

---

## 1. Scientific Motivation & Central Goal

The strict confirmatory benchmark established that downstream-invisible error steering:
$$\min_C \|J_{l \to L} \text{vec}((P - S C)^\top)\|_2^2 + \lambda \|P - S C\|_F^2$$
consistently outperforms ordinary Euclidean merging and matched-budget pruning. Crucially, truncating the downstream Jacobian $J_{l \to L}$ to its top $r \in \{16, 32\}$ singular modes captures $>98\%$ of the full-rank operator's benefit.

However, computing $J_{l \to L}$ via backward Vector-Jacobian Products (VJP) requires $14\text{--}77\text{ ms}$ per image, categorizing the method as an **Oracle** technique.

The objective of this research phase is to **amortize** this computation:
> **Central Research Question:** Can a lightweight neural predictor $H_\theta(P_l)$, operating strictly in the forward pass at intervention block $l$, accurately predict the top-$r$ downstream-visible subspace ($r \in \{16, 32\}$) of $J_{l \to L}$, and how much of the oracle accuracy-token frontier can be recovered without test-time backpropagation?

---

## 2. Mathematical Formulation & Subspace-Aware Objectives

### 2.1 The Subspace Prediction Formulation
Naively regressing singular vectors via elementwise Mean Squared Error (MSE) is mathematically flawed due to:
1. **Sign Ambiguity:** For any singular vector $v_k$, $-v_k$ spans the exact same subspace.
2. **Rotational Invariance:** For clustered singular values, any orthonormal rotation within the eigenspace is equivalent.

Let $V_{\text{true}} \in \mathbb{R}^{ND \times r}$ be the orthonormal basis of the top-$r$ right singular vectors of $J_{l \to L}$ ($V_{\text{true}}^\top V_{\text{true}} = I_r$), and let $\hat{V}_{\text{pred}} \in \mathbb{R}^{ND \times r}$ be the predicted orthonormal basis.

### 2.2 Primary Loss: Subspace Overlap Loss
The canonical invariant Grassmannian metric measuring the overlap between two $r$-dimensional subspaces is:
$$\text{Overlap}(V_{\text{true}}, \hat{V}_{\text{pred}}) = \frac{1}{r} \|V_{\text{true}}^\top \hat{V}_{\text{pred}}\|_F^2 = \frac{1}{r} \text{Tr}\left( (V_{\text{true}}^\top \hat{V}_{\text{pred}}) (V_{\text{true}}^\top \hat{V}_{\text{pred}})^\top \right)$$
When the predicted subspace matches the true subspace exactly, $V_{\text{true}}^\top \hat{V}_{\text{pred}}$ is an $r \times r$ orthogonal matrix, so $\|V_{\text{true}}^\top \hat{V}_{\text{pred}}\|_F^2 = r$ and $\text{Overlap} = 1.0$.

The primary training loss is:
$$\mathcal{L}_{\text{subspace}} = 1.0 - \text{Overlap}(V_{\text{true}}, \hat{V}_{\text{pred}})$$

### 2.3 Secondary Loss: Singular-Value-Weighted Subspace Loss
To prioritize directions with larger downstream amplification:
$$W = \text{diag}(\sigma_1^2, \dots, \sigma_r^2) / \sum_{k=1}^r \sigma_k^2$$
$$\mathcal{L}_{\text{weighted}} = 1.0 - \text{Tr}\left( (V_{\text{true}}^\top \hat{V}_{\text{pred}}) (\hat{V}_{\text{pred}}^\top V_{\text{true}}) W \right)$$

### 2.4 Functional Distillation Loss
As an alternative to geometric subspace alignment, we evaluate functional transmission distillation:
Given random probe errors $e \in \mathbb{R}^{ND}$, student matches the oracle transmission norm:
$$\mathcal{L}_{\text{distill}} = \mathbb{E}_{e \sim \mathcal{N}(0, I)} \left[ \left| \|\hat{V}_{\text{pred}}^\top e\|_2 - \|V_{\text{true}}^\top e\|_2 \right| \right]$$

---

## 3. Dataset Splits & Oracle Target Generation

All evaluations use ImageNet-1k validation data ($50,000$ images total). Splits are strictly stratified and mutually disjoint:

1. **Canonical Evaluation Split ($N = 1,000$ images):**  
   Seed `9201` (1 per class). Completely held out. Never used for predictor training, validation, or target fitting.
2. **Calibration Split ($N = 1,000$ images):**  
   Seed `9101` (1 per class). Used for static baseline computation.
3. **Predictor Training Split ($N = 2,000$ images):**  
   Seed `7101` (2 per class). Disjoint from calibration and evaluation sets.
4. **Predictor Validation Split ($N = 500$ images):**  
   Seed `7201`. Disjoint from training, calibration, and evaluation sets.

### Fast Oracle Target Extraction
For each training and validation image:
1. Run backbone up to block $l$.
2. Compute $J_{l \to L} \in \mathbb{R}^{D_{\text{readout}} \times ND}$ via batched VJP.
3. Compute exact top-$r$ left and right singular vectors via Gram eigendecomposition:
   $$G = J J^\top \in \mathbb{R}^{D_{\text{readout}} \times D_{\text{readout}}}$$
   $$G U = U \Lambda, \quad \sigma_k = \sqrt{\lambda_k}$$
   $$V_r = J^\top U_r \Sigma_r^{-1} \in \mathbb{R}^{ND \times r}$$
   Runtime: $< 2\text{ ms}$ per image for SVD extraction.

---

## 4. Evaluated Predictor Architectures

To identify the optimal compute-accuracy frontier, we evaluate four predictor families:

1. **Family A: Static Global Subspace Baseline:**  
   Dataset-level SVD on calibration targets: $V_{\text{static}} = \text{top-}r(\frac{1}{M}\sum_{i=1}^M V_{r, i} V_{r, i}^\top)$. Requires zero test-time compute.
2. **Family B: Linear / MLP Activation Predictor:**  
   Extracts pooled summary statistics from $P_l$ (CLS token $h_{\text{cls}}$, patch mean $\bar{p}$, patch variance $\sigma_p$) and maps through a 2-layer MLP to predicted subspace parameters.
3. **Family C: Factorized Mode Predictor (Primary Innovation):**  
   Leveraging the empirical discovery that each mode $Q_k \in \mathbb{R}^{N \times D}$ is strongly low-rank ($>68\%$ energy in rank 4), Family C decomposes each mode into rank-$R$ separable factors:
   $$Q_k \approx \sum_{s=1}^R a_{k, s} b_{k, s}^\top, \quad a_{k, s} \in \mathbb{R}^N, \quad b_{k, s} \in \mathbb{R}^D$$
   A lightweight cross-attention / token-MLP network outputs token factors $A \in \mathbb{R}^{N \times (r \cdot R)}$ and feature factors $B \in \mathbb{R}^{D \times (r \cdot R)}$. This reduces predictor output dimensionality from $ND \times r = 2.41\text{M}$ down to $(N + D) \times r \times R \approx 37\text{K}$ parameters ($65\times$ reduction).
4. **Family D: Direct Carrier Predictor Control:**  
   A direct regression network that predicts carrier adjustments $\Delta C \in \mathbb{R}^{B \times D}$ from $P_l$ without explicitly predicting operator modes, testing whether explicit subspace modeling provides superior generalization.

---

## 5. Confirmatory Evaluation Protocol & Metrics

Every trained predictor is evaluated on the canonical unseen $N=1,000$ split across all 5 budgets:
$$B \in \{147, 98, 72, 49, 32\} \quad (N=196)$$
$$B \in \{192, 128, 94, 64, 42\} \quad (N=256)$$

### Metrics:
1. **Subspace Overlap:** $\frac{1}{r} \|V_{\text{true}}^\top \hat{V}_{\text{pred}}\|_F^2 \in [0, 1]$.
2. **Principal Angles:** $\theta_1, \dots, \theta_r$ between predicted and oracle subspaces.
3. **Top-1 Accuracy & Retention:** $A(B)$ and $\rho(B) = A(B)/A_{\text{clean}}$.
4. **Oracle Recovery Ratio:**
   $$\text{Recovery} = \frac{A_{\text{predicted}}(B) - A_{\text{baseline}}(B)}{A_{\text{oracle}}(B) - A_{\text{baseline}}(B)}$$
5. **End-to-End Latency & FLOPs:** Full pipeline wall-clock time including prefix blocks $0 \dots l-1$, predictor $H_\theta$, carrier solve, and compressed suffix blocks $l \dots L-1$.

---

## 6. Pre-Registered Decision Levels

- **OUTCOME A — Near-Oracle Amortization:**  
  Forward predictor recovers $\ge 90\%$ of the oracle compression improvement over group-mean merging with zero backward passes and $<15\%$ predictor compute overhead.
- **OUTCOME B — Partial Amortization:**  
  Predictor recovers $50\%\text{--}90\%$ of oracle gain and significantly outperforms static subspace and standard merging baselines.
- **OUTCOME C — Static Structure Dominates:**  
  Image-conditioned predictors provide no statistically significant gain over a static calibration subspace.
- **OUTCOME D — Mode Prediction Intractable / Direct Carrier Wins:**  
  Subspace prediction fails to improve compression, while direct carrier regression performs better.
- **OUTCOME G — Practical Acceleration Success:**  
  Total forward inference latency is strictly faster than the uncompressed ViT while matching or exceeding ToMe accuracy.
