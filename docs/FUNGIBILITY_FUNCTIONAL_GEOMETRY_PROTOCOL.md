# Functional Equivalence Geometry Protocol
## Mapping the Anisotropic Geometry of ViT Patch-Content Fungibility

**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Protocol Version**: 1.0 (Pre-Registered)  
**Date**: October 4, 2026  
**Status**: Frozen Prior to Experiment Execution  

---

## 1. Scientific Motivation and Paradigm Shift

### 1.1 Beyond Scalar Predictors
Prior research established that late ViT patch activations can be replaced by class-agnostic calibration surrogates with minimal functional damage, while coordinate permutations and sign flips catastrophically destroy function. However, the previous mechanistic sprint proved that **clean-state scalar metrics** (such as $\text{tr}(J \Sigma J^\top)$, effective rank, and Jacobian norm) do not predict fungibility across architectures.

These negative results provide crucial scientific information:
> *Patch-content fungibility cannot be described by a single scalar "fungibility score" or an isotropic property of a layer.*

### 1.2 The Central Research Question
Instead of asking:
> *"How fungible is this layer, represented by one scalar?"*

We ask:
> *"What changes to a patch representation are functionally tolerated by the downstream network, and what is the geometry of that tolerated set?"*

### 1.3 The Working Conceptual Hypothesis
> **Patch Content Fungibility is an Anisotropic Functional Equivalence Geometry.**  
> At a given layer $l$, activation space is partitioned into highly sensitive directions (where small perturbations destroy downstream classification) and highly fungible directions (which permit large finite displacements with negligible downstream effect).

The objective is to **empirically map this geometry** without assuming the hypothesis is true.

---

## 2. Functional Distance and Equivalence Set Definition

Let $G_{>l}$ denote the frozen downstream network mapping post-block-$l$ hidden states $H_l = [c_l, P_l]$ to logits $y \in \mathbb{R}^C$.

For a patch activation tensor $P_l$ and perturbed state $Q_l = P_l + \Delta$, downstream functional distance is defined using three complementary observables:
1. **Margin Damage (Primary Observable)**:
   $$d_{\text{margin}}(P, Q) = m(P) - m(Q)$$
   where $m = y_{y^*} - \max_{c \ne y^*} y_c$ is the clean true-class margin.
2. **Relative Logit Distance**:
   $$d_{\text{logit}}(P, Q) = \frac{\|y(Q) - y(P)\|_2}{\|y(P)\|_2}$$
3. **Top-1 Prediction Flip**:
   $$\text{Flip}(P, Q) = \mathbb{I}\left[\arg\max y(Q) \ne \arg\max y(P)\right]$$
4. **Output KL Divergence**:
   $$D_{\text{KL}}(\sigma(y(P)) \parallel \sigma(y(Q)))$$

The functional equivalence set of radius $\epsilon$ is:
$$\mathcal{E}_l(P, \epsilon) = \left\{ Q : \mathbb{E}\left[d_{\text{margin}}(P, Q)\right] \le \epsilon \right\}$$

---

## 3. Directional Fungibility Spectrum & Finite Tolerance Radius

For a clean patch activation state, we perturb along a normalized unit direction $v \in \mathbb{R}^D$ ($\|v\|_2 = 1$):
$$Q(s, v) = P + \alpha(s) v, \quad \alpha(s) = s \cdot \sqrt{\text{tr}(\Sigma_l) / D}$$
where $s$ is a dimensionless perturbation scale calibrated to the natural token standard deviation $\sigma_l = \sqrt{\text{tr}(\Sigma_l) / D}$.

We sweep $s$ across a calibrated grid:
$$s \in \{0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2\}$$

### 3.1 Tolerance Radius ($r_l(v; \epsilon)$)
$$r_l(v; \epsilon) = \max \left\{ s : d_{\text{margin}}(s, v) \le \epsilon \right\}$$
with pre-registered primary threshold $\epsilon = 0.20$ (preserving $\ge 80\%$ of clean margin) and secondary threshold $\epsilon = 0.50$.
- Small $r_l(v) \implies$ Sensitive direction.
- Large $r_l(v) \implies$ Fungible direction.

### 3.2 Directional Fungibility Spectrum & Summaries
- **Fungibility Spectrum**: Vector of tolerance radii across tested directions: $(r_1, r_2, \dots, r_K)$.
- **Anisotropy Ratio**:
  $$\text{Anisotropy}_l = \frac{\max_k r_{l, k}}{\min_k r_{l, k}}$$
- **Fungible Dimension**:
  $$\text{FDim}_l(\epsilon, \tau) = \sum_{k=1}^K \mathbb{I}[r_{l, k}(\epsilon) \ge \tau], \quad \text{fdim}_l = \frac{\text{FDim}_l}{K}$$
  with pre-registered scale threshold $\tau = 0.80 \times \sigma_l$.

---

## 4. Structured Direction Families to Probe ($K$ Directions)

For each model and layer, we probe four distinct structured direction families:

### Family A: Natural Representation Covariance (PCA of $\Sigma_l$)
- **Top PCs**: $v_1$ (PC1), $v_2$ (PC2), $v_4$ (PC4)
- **Middle PCs**: $v_{D/4}$, $v_{D/2}$ (median-variance directions)
- **Bottom PCs**: $v_{3D/4}$, $v_D$ (lowest-variance / tail directions)

### Family B: Isotropic Random Directions
- Four mutually orthogonalized random unit vectors drawn uniformly on $\mathbb{S}^{D-1}$.

### Family C: Downstream Functional Geometry (Jacobian Eigenspace)
- $v_{\text{grad}}$: Normalized margin gradient direction $\bar{g}_l / \|\bar{g}_l\|_2$
- $v_{\text{jac\_top}}$: Top right singular vector of downstream Jacobian (via power iteration on $J^\top J$)
- $v_{\text{jac\_null}}$: Near-null direction of downstream Jacobian (lowest singular vector of $J^\top J$)

### Family D: Canonical Intervention Displacements
- Centroid direction: $\mu_l / \|\mu_l\|_2$
- Clean-to-Centroid displacement: $\delta_{\text{centroid}} = (\mu_l - \bar{p}) / \|\mu_l - \bar{p}\|_2$
- Permutation displacement: $\delta_{\text{perm}} = (\pi(\mu_l) - \mu_l) / \|\pi(\mu_l) - \mu_l\|_2$
- Sign-flip displacement: $\delta_{\text{flip}} = -\mu_l / \|\mu_l\|_2$
- Cross-image displacement: $(p_{\text{other}} - p) / \|p_{\text{other}} - p\|_2$

---

## 5. Local Functional Metric Matrix ($M_l$) and Subspace Alignment

We construct the local quadratic functional metric:
$$M_l = \mathbb{E}_{i \in \text{images}} \left[ J_l^{(i)\top} J_l^{(i)} \right] \in \mathbb{R}^{D \times D}$$
and analyze its eigenspectrum $M_l = U_M \Omega_M U_M^\top$ ($\omega_1 \ge \dots \ge \omega_D \ge 0$):
1. **Near-Null Subspace Dimensionality**: Number of eigenvalues $\omega_k \le 10^{-3} \omega_1$.
2. **Effective Metric Rank**: $r_{\text{eff}}(M_l) = \exp(-\sum \tilde{\omega}_k \log \tilde{\omega}_k)$.
3. **Alignment with Natural Covariance $\Sigma_l$**:
   - Trace of product: $\text{tr}(M_l \Sigma_l)$
   - Eigenspectrum of generalized eigenvalue problem $M_l v = \lambda \Sigma_l^{-1} v$ or $\Sigma_l^{1/2} M_l \Sigma_l^{1/2}$.
   - Projection of natural PC directions onto high vs. low eigenspaces of $M_l$.
4. **Decomposition of Existing Interventions**:
   Project $\delta_{\text{centroid}}$, $\delta_{\text{perm}}$, $\delta_{\text{flip}}$ onto:
   - Sensitive subspace $U_{\text{sensitive}} = \text{span}(u_1, \dots, u_m)$
   - Fungible subspace $U_{\text{fungible}} = \text{span}(u_{D-m+1}, \dots, u_D)$
   Test whether damage correlates with the fraction of energy in $U_{\text{sensitive}}$.

---

## 6. Image-Wise and Token-Wise Consistency

1. **Image Consistency**:
   Evaluate tolerance curves per-image on $N_{\text{pilot}} = 100$ disjoint validation images.
   Compute Spearman rank correlation matrix of direction tolerances between images:
   $$\bar{\rho}_{\text{images}} = \frac{2}{N(N-1)} \sum_{i < j} \text{Corr}(r_i, r_j)$$
   High $\bar{\rho} \implies$ shared global functional geometry; Low $\bar{\rho} \implies$ image-conditioned geometry.
2. **Token Heterogeneity**:
   Compare tolerance radii across patch positions: central tokens vs. peripheral tokens vs. high-attention tokens.

---

## 7. Constructive Test: Functional-Geometry Surrogate

If a stable near-null subspace $U_{\text{fungible}}$ of dimension $k$ exists, test a constructive surrogate distribution:
$$R_{\text{fungible}} = \mu_l + U_{\text{fungible}} z, \quad z \sim \mathcal{N}(0, \text{diag}(\sigma_{\text{fungible}}^2))$$
Compare downstream margin damage and top-1 accuracy against:
- Static Centroid $\mu_l$
- Diagonal Gaussian $\mathcal{N}(\mu_l, \Sigma_{\text{diag}})$
- Sensitive-subspace Gaussian $R_{\text{sensitive}} = \mu_l + U_{\text{sensitive}} z$
- Isotropic Random Subspace Gaussian

---

## 8. Models, Depths, and Compute Scope

### 8.1 Pilot Scope (Section 15)
- **Primary Architectures**:
  1. `deit_small_patch16_224` ($D=384$)
  2. `vit_base_patch16_224.augreg_in1k` ($D=768$)
- **Validation Secondary Architectures**:
  3. `dinov2_vits14_lc` ($D=384$)
  4. `deit_tiny_patch16_224` ($D=192$)
- **Evaluated Depths**:
  - Depth 5: Fragile baseline layer
  - Depth 7: Transition / peak (ViT-B peak)
  - Depth 8: Known peak (DeiT-Small peak)
  - Depth 10: Terminal layer
- **Image Sample**: $N = 100$ disjoint calibration/evaluation images (rich, deterministic, rapid GPU execution).

---

## 9. Implementation Validation and Numerical Sanity Checks

1. **Weight Freeze**: Verify all model parameters have `requires_grad=False`.
2. **Orthonormality**: Verify PCA eigenvectors satisfy $V^\top V = I_D$ to $< 10^{-6}$.
3. **Random Direction Orthonormality**: Verify random unit vectors satisfy $U^\top U = I$.
4. **Perturbation Scale Invariance**: Verify $\|\alpha(s) v\|_2 = \alpha(s)$ exactly.
5. **Autograd / Finite-Difference Parity**: Confirm downstream gradient $g_l$ matches finite difference to $< 10^{-3}$ relative error.
6. **No Label Leakage**: Geometry and metric matrices constructed strictly without evaluation labels.

---

## 10. Deliverables

- `outputs/fungibility_functional_geometry/`
  - `directional_tolerance.csv`
  - `functional_spectrum.csv`
  - `fungible_dimension.csv`
  - `covariance_function_alignment.csv`
  - `imagewise_consistency.csv`
  - `intervention_projection.csv`
  - `validation_manifest.json`
- `figures/fungibility_functional_geometry/`
  - `figure_1_directional_tolerance_curves.png`
  - `figure_2_fungibility_spectrum_depth.png`
  - `figure_3_functional_metric_eigenspectrum.png`
  - `figure_4_covariance_functional_alignment.png`
  - `figure_5_intervention_projections.png`
  - `figure_6_constructive_surrogate_test.png`
- `docs/FUNGIBILITY_FUNCTIONAL_GEOMETRY_REPORT.md` answering all 10 mandatory questions and concluding with the Research Map.
