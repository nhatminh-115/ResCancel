# Fungibility Predictive Principle Protocol
## Predicting ViT Patch-Content Fungibility from Clean-State Representation Geometry and Downstream Sensitivity

**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Protocol Version**: 1.0 (Pre-Registered)  
**Date**: October 4, 2026  
**Status**: Frozen Prior to Experiment Execution  

---

## 1. Scientific Motivation and Objective

### 1.1 The Scientific Gap
Previous iterations of this research (V0–V1) established the empirical phenomenon of **Late Patch-Content Fungibility**:
- In late transformer blocks (e.g., depths 7–9 in DeiT-Tiny, DeiT-Small, ViT-B AugReg, and DINOv2 ViT-S/14), spatial patch tokens can be substituted by static calibration centroids or distribution-matched Gaussian noise with minimal loss of accuracy or margin damage ($80\% - 98\%$ recovery).
- In contrast, early-to-mid transformer blocks (e.g., depths 5–6) are fragile to substitution.
- Terminal readout blocks (e.g., depth 10 in supervised DeiT) exhibit near-zero patch-stream causal sensitivity, as the CLS token has already decoupled from the spatial stream.

However, all existing fungibility maps were established **post-hoc** via expensive combinatorial replacement sweeps ($N$ intervention sweeps $\times$ condition ablations $\times$ seeds). 

### 1.2 The Goal
To upgrade from an empirical phenomenon to a **predictive mechanistic principle**:
$$\text{Phenomenon} \longrightarrow \text{Measurable Mechanism} \longrightarrow \text{Predictive Principle}$$

**The Central Question**:
> *Can we predict, BEFORE running any replacement intervention, whether a ViT layer will be FUNGIBLE vs. FRAGILE, using only clean calibration activations and downstream network sensitivity measurements?*

The ideal outcome is a clean-state geometric/sensitivity metric that predicts the emergence and location of patch-content fungibility across distinct architectures without running replacement sweeps.

---

## 2. Target Variable: Dependent Fungibility Score ($F_l$)

### 2.1 Primary Continuous Target
We strictly reuse existing validated replacement outcomes from canonical experiment manifests (`outputs/fungibility_v0_6/` and `outputs/fungibility_v1/`). The target variable is never recomputed or redefined using predictor values.

For each architecture and intervention depth $l$, the primary continuous fungibility target is:
$$F_l = 1 - \frac{D_{\text{centroid}}(l)}{D_{\text{zero}}(l)}$$
where:
- $D_R(l)$ is the mean true-class margin damage produced by replacement condition $R$ at depth $l$ on the canonical evaluation split ($N=1,000$ images, 25% replacement budget):
  $$D_R(l) = \text{Margin}_{\text{clean}} - \text{Margin}_R(l)$$
- $D_{\text{centroid}}(l)$ is the damage under static calibration centroid replacement (the primary surrogate).
- $D_{\text{zero}}(l)$ is the damage under zero replacement.

### 2.2 Boundary Handling and Secondary Targets
- Where $D_{\text{zero}}(l) \le 0$ or $D_{\text{centroid}}(l) > D_{\text{zero}}(l)$ (such as terminal layers where patch stream is decoupled or damaged minimally), the raw ratio can become negative or undefined. We report:
  1. **Raw $F_l$**: Exact mathematical definition above.
  2. **Bounded $F_l^{\text{clip}} \in [0, 1]$**: $\text{clip}(F_l, 0.0, 1.0)$.
  3. **Gaussian Recovery Fraction**: $F_l^{\text{gauss}} = 1 - D_{\text{gaussian}}(l) / D_{\text{zero}}(l)$.
  4. **Centroid Top-1 Retention**: $\text{Acc}_{\text{centroid}}(l) / \text{Acc}_{\text{clean}}$.
  5. **Gaussian Top-1 Retention**: $\text{Acc}_{\text{gaussian}}(l) / \text{Acc}_{\text{clean}}$.
  6. **Binary Fungibility Indicator**: $\mathbb{I}[F_l \ge 0.80]$ (secondary threshold analysis).

The primary analysis remains **continuous regression and correlation**.

---

## 3. Models and Intervention Depths

Four frozen architectures are evaluated across all intervention depths for which validated replacement data exist:

| Model Identifier | Key | Blocks ($L$) | Hidden Dim ($D$) | Spatial Patches ($N$) | Evaluated Depths ($l$) | Source Outputs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `deit_tiny_patch16_224` | `deit_tiny` | 12 | 192 | 196 | $\{5, 6, 7, 8, 9, 10\}$ | `outputs/fungibility_v0_6/` |
| `deit_small_patch16_224` | `deit_small` | 12 | 384 | 196 | $\{5, 6, 7, 8, 9, 10\}$ | `outputs/fungibility_v0_6/` |
| `vit_base_patch16_224.augreg_in1k` | `vit_base` | 12 | 768 | 196 | $\{5, 7, 8, 9, 10\}$ | `outputs/fungibility_v1/` |
| `dinov2_vits14_lc` (layers=1) | `dinov2` | 12 | 384 | 256 | $\{5, 7, 8, 9, 10\}$ | `outputs/fungibility_v1/` |

**Total Model-Depth Target Points**: $N_{\text{points}} = 6 + 6 + 5 + 5 = 22$ points.  
*(In addition, clean predictors are computed across all blocks $l \in \{1, \dots, 12\}$ to map full architectural depth profiles).*

---

## 4. Data Split and Strict Information Barrier

All clean-state predictors are computed **strictly and exclusively** from the canonical calibration split:
- $N_{\text{calib}} = 1,000$ ImageNet-1k validation images (stratified 1 per class, random seed `9101`).
- Evaluation images ($N_{\text{eval}} = 1,000$, seed `9201`) are **strictly excluded**.
- Zero evaluation-image labels, zero replacement activations, and zero replacement performance measurements may enter the predictors.

---

## 5. Working Mechanistic Hypothesis & Candidate Metric Families

### 5.1 Primary Mechanistic Hypothesis: Sensitivity-Weighted Representation Geometry
**Core Hypothesis**:
> *Patch fungibility emerges when natural patch variation increasingly aligns with directions to which the downstream network is functionally insensitive.*

Let $p_{l, k} \in \mathbb{R}^D$ denote the $k$-th spatial patch activation after block $l$, with calibration covariance matrix:
$$\Sigma_l = \frac{1}{M-1} \sum_{i=1}^{N_{\text{calib}}} \sum_{k=1}^N (p_{l, k}^{(i)} - \mu_l)(p_{l, k}^{(i)} - \mu_l)^\top \in \mathbb{R}^{D \times D}$$
where $M = N_{\text{calib}} \times N$, and $\mu_l$ is the calibration centroid.

Let $J_l$ denote the downstream Jacobian of the network output with respect to the patch representation stream, and let $g_{l, k} = \nabla_{p_{l, k}} m(x) \in \mathbb{R}^D$ denote the gradient of the true-class/top-class margin $m(x) = y_{\hat{y}} - \max_{c \ne \hat{y}} y_c$ with respect to patch token $k$.

The primary candidate metric is the **Sensitivity-Weighted Geometric Energy**:
$$S_l = \text{tr}(J_l \Sigma_l J_l^\top) \quad \text{or for margin gradient: } S_l = \frac{1}{N} \sum_{k=1}^N g_{l, k}^\top \Sigma_l g_{l, k}$$
**Physical Meaning**:
- $\Sigma_l$ measures the directions and energy of natural variation among patch tokens.
- $g_{l, k}$ measures which directions the remaining downstream network uses to compute the classification margin.
- $g_{l, k}^\top \Sigma_l g_{l, k}$ quantifies how much natural patch variation falls into functionally sensitive downstream directions.
- **Prediction**: Low $S_l \implies$ High Fungibility ($F_l \approx 1$); High $S_l \implies$ Fragility ($F_l \ll 1$).

### 5.2 Candidate Metric Families

#### Family A: Clean Representation Geometry (Covariance Spectrum)
Computed directly from token covariance $\Sigma_l$:
1. **Total Covariance Trace**: $\text{tr}(\Sigma_l) = \sum_{d=1}^D \lambda_d$
2. **Mean Token Variance**: $V_l = \frac{1}{D} \text{tr}(\Sigma_l)$
3. **Entropy Effective Rank**:
   $$r_{\text{eff}}(\Sigma_l) = \exp\left( - \sum_{d=1}^D q_d \log q_d \right), \quad q_d = \frac{\lambda_d}{\sum_j \lambda_j}$$
4. **Leading Eigenvalue Concentration**: $C_1 = \lambda_1 / \text{tr}(\Sigma_l)$
5. **Cumulative Spectral Concentration**: $C_4 = \sum_{k=1}^4 \lambda_k / \text{tr}(\Sigma_l)$, $C_{16} = \sum_{k=1}^{16} \lambda_k / \text{tr}(\Sigma_l)$

#### Family B: Token Homogeneity and Spatial Mixing
Computed across spatial tokens within clean images:
1. **Mean Pairwise Cosine Similarity**:
   $$\text{CosSim}_{\text{patch}}(l) = \mathbb{E}_i \left[ \frac{2}{N(N-1)} \sum_{j < k} \frac{\langle p_{l, j}^{(i)}, p_{l, k}^{(i)} \rangle}{\|p_{l, j}^{(i)}\|_2 \|p_{l, k}^{(i)}\|_2} \right]$$
2. **Mean Distance to Patch Mean**: $\text{Dist}_{\text{mean}}(l) = \mathbb{E}_i \left[ \frac{1}{N} \sum_k \| p_{l, k}^{(i)} - \bar{p}_l^{(i)} \|_2 \right]$
3. **Mean Distance to Centroid**: $\text{Dist}_{\text{centroid}}(l) = \mathbb{E}_i \left[ \frac{1}{N} \sum_k \| p_{l, k}^{(i)} - \mu_l \|_2 \right]$
4. **Patch-to-CLS Cosine Similarity**: $\text{CosSim}_{\text{cls}}(l) = \mathbb{E}_i \left[ \frac{1}{N} \sum_k \frac{\langle p_{l, k}^{(i)}, c_l^{(i)} \rangle}{\|p_{l, k}^{(i)}\|_2 \|c_l^{(i)}\|_2} \right]$

#### Family C: Downstream Output Sensitivity
Computed by propagating gradients from downstream output back to block $l$ patch activations:
1. **Margin Gradient Frobenius Norm**:
   $$\|J_l^{\text{margin}}\|_F = \mathbb{E}_i \left[ \sqrt{ \sum_{k=1}^N \| \nabla_{p_{l, k}} m(x^{(i)}) \|_2^2 } \right]$$
2. **Mean Token Gradient Norm**:
   $$\|\bar{g}_l\| = \mathbb{E}_i \left[ \frac{1}{N} \sum_{k=1}^N \| \nabla_{p_{l, k}} m(x^{(i)}) \|_2 \right]$$
3. **Logit Jacobian Frobenius Norm (Hutchinson Estimate)**:
   $$\|J_l^{\text{logit}}\|_F = \mathbb{E}_{i, v \sim \mathcal{N}(0, I_C)} \left[ \| \nabla_{P_l} (v^\top y^{(i)}) \|_F \right]$$
4. **Variance-Normalized Sensitivity**: $\|J_l^{\text{margin}}\|_F / \sqrt{\text{tr}(\Sigma_l)}$

#### Family D: Sensitivity-Weighted Geometry (Primary Candidate)
1. **Margin Quadratic Sensitivity ($S_l^{\text{margin}}$)**:
   $$S_l^{\text{margin}} = \mathbb{E}_i \left[ \frac{1}{N} \sum_{k=1}^N g_{l, k}^{(i)\top} \Sigma_l g_{l, k}^{(i)} \right] = \mathbb{E}_i \left[ \frac{1}{N} \text{tr}\left( G_l^{(i)\top} G_l^{(i)} \Sigma_l \right) \right]$$
2. **Logit Hutchinson Sensitivity ($S_l^{\text{logit}}$)**:
   $$S_l^{\text{logit}} = \mathbb{E}_{i, v} \left[ \frac{1}{N} \text{tr}\left( \nabla_{P_l}(v^\top y)^\top \nabla_{P_l}(v^\top y) \Sigma_l \right) \right]$$
3. **Trace-Normalized Sensitivity**: $S_l^{\text{margin}} / \text{tr}(\Sigma_l)$
4. **Gradient-Normalized Sensitivity**: $S_l^{\text{margin}} / \|J_l^{\text{margin}}\|_F^2$

---

## 6. Mechanistic Baseline Control: Normalized Depth ($\ell / L$)

To prove that a candidate metric captures a genuine physical mechanism rather than acting as a trivial surrogate for layer index, every predictor is tested against:
$$\text{Normalized Depth} = \frac{l}{L} \quad (l \in \{1, \dots, 12\}, L=12)$$

We estimate three nested linear regressions for each predictor $X$:
1. **Model 1 (Depth Only)**: $F_l = \alpha_0 + \beta_d \cdot (l/L) + \epsilon$
2. **Model 2 (Predictor Only)**: $F_l = \alpha_0 + \beta_x \cdot X_l + \epsilon$
3. **Model 3 (Joint Model)**: $F_l = \alpha_0 + \beta_d \cdot (l/L) + \beta_x \cdot X_l + \epsilon$

We report:
- Univariate $R^2(X)$ vs $R^2(\text{depth})$
- Incremental Explanatory Power: $\Delta R^2 = R^2(\text{depth} + X) - R^2(\text{depth})$
- Partial correlation $r(F_l, X \mid \text{depth})$

**Rule**: If $\Delta R^2 \le 0.05$ or partial correlation is not statistically significant, the metric is classified as a mere depth proxy.

---

## 7. Statistical Evaluation & Leave-One-Architecture-Out (LOAO) Cross-Validation

### 7.1 In-Sample & Pooled Metrics
For each predictor:
- Pearson correlation $r$ with $F_l$ (pooled and per-architecture)
- Spearman rank correlation $\rho$ with $F_l$ (pooled and per-architecture)
- 95% Bootstrap Confidence Interval on pooled $\rho$ ($B=2,000$ resamples)

### 7.2 Leave-One-Architecture-Out (LOAO) Protocol
This is the **primary critical test**. For each of the 4 architectures:
1. **Train Set**: The remaining 3 architectures (e.g., train on Tiny + Small + ViT-B; hold out DINOv2).
2. **Fit**: Fit univariate linear relation $\hat{F}_l = a + b \cdot X_l$ on training architectures only.
3. **Predict**: Apply the frozen linear fit to the held-out architecture's clean-state predictor values.
4. **Evaluate Held-Out Metrics**:
   - Out-of-Sample $R^2_{\text{LOAO}} = 1 - \frac{\sum (F_l - \hat{F}_l)^2}{\sum (F_l - \bar{F}_{\text{test}})^2}$
   - Mean Absolute Error (MAE): $\frac{1}{|L_{\text{test}}|} \sum |F_l - \hat{F}_l|$
   - Held-Out Spearman Rank Correlation $\rho_{\text{heldout}}$

### 7.3 Transition Layer Prediction Analysis
- **Definition of Transition Layer ($l^*$)**: The first layer index where $F_l \ge 0.80$.
- **Predictor Threshold**: Fit threshold $T_{\text{train}}$ on the 3 training architectures where the predictor crosses the $F=0.80$ boundary.
- **Held-Out Transition Prediction ($\hat{l}^*$)**: The first layer on the held-out architecture where $X_l$ crosses $T_{\text{train}}$.
- **Transition Layer Error**: $|l^* - \hat{l}^*|$.

---

## 8. Implementation Validation & Numerical Sanity Checks

Before executing the full evaluation:
1. **Weight Invariance**: Assert all backbone weights are frozen (`requires_grad=False`).
2. **Hook Parity**: Verify clean activations extracted block-by-block match untouched forward passes to $< 10^{-5}$ tolerance.
3. **Finite-Difference Sanity Check**: On a tiny sample ($N=4$), verify that the autograd gradient $g_{l, k} = \nabla_{p_{l, k}} m(x)$ matches central finite-difference approximation:
   $$\frac{m(P + \epsilon e) - m(P - \epsilon e)}{2\epsilon} \approx g^\top e \quad (\text{relative error } < 10^{-2})$$
4. **Quadratic Form Equivalence**: Verify numerically that $\frac{1}{N} \sum_k g_k^\top \Sigma g_k = \frac{1}{N} \text{tr}(G^\top G \Sigma)$.
5. **No NaNs/Infs**: Strict verification of finite floating-point values across all matrices and spectra.

---

## 9. Pre-Registered Decision Criteria and Outcome Framework

| Outcome | Decision | Empirical Criteria |
| :--- | :--- | :--- |
| **OUTCOME A — STRONG PREDICTIVE PRINCIPLE** | Elevate to primary paper claim | Candidate metric (especially $S_l$) achieves pooled $|\rho| \ge 0.70$, $\Delta R^2 \ge 0.15$ beyond depth, passes LOAO with positive $R^2_{\text{LOAO}}$, and predicts held-out transition layer with mean error $\le 1.0$ block. |
| **OUTCOME B — DESCRIPTIVE CORRELATE ONLY** | Document as exploratory finding, do not elevate | Metric correlates strongly within individual models ($|\rho| \ge 0.70$), but fails cross-architecture LOAO generalization ($R^2_{\text{LOAO}} \le 0$ or transition error $> 2$ blocks). |
| **OUTCOME C — DEPTH REMAINS BEST PREDICTOR** | Retain depth transition description | Candidate geometric/sensitivity metrics fail to explain meaningful variance beyond normalized depth ($\Delta R^2 < 0.05$). |
| **OUTCOME D — KILL PREDICTIVE PRINCIPLE** | Terminate direction | No candidate metric exhibits a robust relationship with fungibility across the four architectures. |

---

## 10. Execution Plan and Deliverables

1. Commit protocol: `docs(fungibility): freeze predictive principle protocol`
2. Create pipeline module: `patch_fungibility/predictive_principle.py`
3. Execute clean-state activation extraction, geometry computation, and sensitivity analysis across all 4 models.
4. Output directory: `outputs/fungibility_predictive_principle/`:
   - `layer_metrics.csv`
   - `predictor_vs_fungibility.csv`
   - `leave_one_architecture_out.csv`
   - `transition_predictions.csv`
   - `validation_manifest.json`
5. Generate figures in `figures/fungibility_predictive_principle/`:
   - `figure_a_depth_alignment.png`
   - `figure_b_predictor_scatter.png`
   - `figure_c_loao_predicted_vs_observed.png`
   - `figure_d_spectral_sensitivity_decomposition.png`
6. Comprehensive report: `docs/FUNGIBILITY_PREDICTIVE_PRINCIPLE_REPORT.md` answering all 9 required questions.
7. Preserve `docs/PAPER_DRAFT.md` untouched.
