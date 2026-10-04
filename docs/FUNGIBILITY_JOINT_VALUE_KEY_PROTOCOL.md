# Joint Value-Key Downstream Low-Transmission Protocol

## 1. Scientific Objective

In previous stages of the patch-content fungibility program, we established:
1. **Single-Block Nullity ($A_l$):** Patch perturbations are governed by a massive exact linear null space ($\ge 99.49\%$). However, single-block null modes leak into subsequent blocks as downstream low-transmission subspaces rotate ($\sim 49^\circ$ between adjacent blocks).
2. **Multi-Block Downstream Jacobian ($J_{l \to L}$):** Diagonalizing the downstream Jacobian improved out-of-sample damage prediction from $r = 0.753$ to $r = 0.975$, and multi-block null modes achieved substantially lower damage than single-block null modes.
3. **The Remaining Boundary:** While multi-block null modes eliminate first-order downstream transmission along the clean value/residual graph ($\|J_{l \to L} Q_{\text{multi-null}}\| = 0$), residual full-model damage still exhibits a non-zero envelope that scales approximately linearly ($p \approx 1.0$) rather than quadratically ($p \approx 2.0$). Mechanistic evidence indicates this residual is driven by **key-induced attention rerouting**:
   $$\Delta K = \Delta P W_K \implies \Delta s = \frac{Q (K + \Delta K)^\top}{\sqrt{d}} - \frac{Q K^\top}{\sqrt{d}} \implies \Delta A = \text{softmax}(s + \Delta s) - \text{softmax}(s) \ne 0.$$

The objective of this protocol is to construct, decompose, and evaluate a **Joint Value-Key Downstream Low-Transmission Geometry**:
Can patch-stream perturbations be simultaneously hidden from:
1. Direct **VALUE** transmission ($V_l$), and
2. Attention-logit / softmax **KEY** rerouting ($R_l$)?

---

## 2. Mathematical Formulation & Channel Decomposition

### 2.1 Local Attention Differential
At block $l$, consider an input token sequence $x \in \mathbb{R}^{(N+1) \times D}$, where index 0 is the readout (CLS) token and indices $1, \dots, N$ are patch tokens.
Let $x_{\text{norm}} = \text{LayerNorm}_1(x)$.
The clean Q, K, V projections for head $h \in \{1, \dots, H\}$ (head dimension $d_h = D / H$, scale $\kappa = 1/\sqrt{d_h}$) are:
$$q_h = x_{\text{norm}} W_{Q, h} \in \mathbb{R}^{(N+1) \times d_h}, \quad k_h = x_{\text{norm}} W_{K, h} \in \mathbb{R}^{(N+1) \times d_h}, \quad v_h = x_{\text{norm}} W_{V, h} \in \mathbb{R}^{(N+1) \times d_h}.$$

For the readout token (index 0), the attention logits are:
$$s_{h, i} = \kappa \, q_{h, 0}^\top k_{h, i}, \quad i \in \{0, 1, \dots, N\}.$$
The softmax attention weights are:
$$a_h = \text{softmax}(s_h) \in \mathbb{R}^{N+1}, \quad a_{h, i} = \frac{\exp(s_{h, i})}{\sum_{j=0}^N \exp(s_{h, j})}.$$
The attention head output for readout token 0 is:
$$\text{head\_out}_{h, 0} = a_h^\top v_h = \sum_{i=0}^N a_{h, i} v_{h, i} \in \mathbb{R}^{d_h}.$$
The projected multi-head output is:
$$\text{attn\_out}_0 = \sum_{h=1}^H W_{O, h} \, \text{head\_out}_{h, 0} \in \mathbb{R}^D.$$

### 2.2 First-Order Taylor Decomposition of Attention Output
Now consider an additive patch perturbation $\Delta P \in \mathbb{R}^{N \times D}$ injected into $x[:, 1:, :]$.
Because token 0 (CLS) is unperturbed at block $l$, $q_{h, 0}$ is unperturbed at first order.
The perturbation induces changes in both keys and values of the patch tokens:
$$k_{h, i} \to k_{h, i} + \delta k_{h, i}, \quad v_{h, i} \to v_{h, i} + \delta v_{h, i}, \quad i \in \{1, \dots, N\}.$$

The total first-order variation of the attention output is:
$$\delta \text{attn\_out}_0 = \sum_{h=1}^H W_{O, h} \left( a_{h, \text{clean}}^\top \delta v_h + \delta a_h^\top v_{h, \text{clean}} \right).$$

This cleanly decomposes into two distinct physical channels:
1. **Value Channel ($V_l$):**
   $$\delta z_{\text{value}} \equiv \sum_{h=1}^H W_{O, h} \left( a_{h, \text{clean}}^\top \delta v_h \right) = V_l \, \text{vec}(\Delta P^\top) \in \mathbb{R}^D.$$
   Here attention weights $a_h$ are held fixed at their clean values, and $\Delta P$ transmits solely through the value projection $W_V$ and output projection $W_O$.
2. **Key Rerouting Channel ($R_l$):**
   $$\delta z_{\text{reroute}} \equiv \sum_{h=1}^H W_{O, h} \left( \delta a_h^\top v_{h, \text{clean}} \right) = R_l \, \text{vec}(\Delta P^\top) \in \mathbb{R}^D.$$
   Here values $v_h$ are held fixed at their clean activations, and $\Delta P$ acts solely by perturbing keys, shifting attention logits $\delta s_{h, i} = \kappa q_{h, 0}^\top \delta k_{h, i}$, which perturbs softmax attention probabilities:
   $$\delta a_h = J_{\text{softmax}}(s_h) \delta s_h = (\text{diag}(a_h) - a_h a_h^\top) \delta s_h,$$
   re-routing the clean values $v_{h, \text{clean}}$.

### 2.3 Exact Operator Dimensions & Joint Stacking
Both $V_l$ and $R_l$ map $\mathbb{R}^{N \times D} \to \mathbb{R}^{D_{\text{readout}}}$:
$$V_l \in \mathbb{R}^{D_{\text{readout}} \times (ND)}, \quad R_l \in \mathbb{R}^{D_{\text{readout}} \times (ND)}.$$
To construct a unified joint operator without ad-hoc parameter tuning, we normalize each operator by its matrix Frobenius norm:
$$\lambda_V = \frac{1}{\|V_l\|_F}, \quad \lambda_K = \frac{1}{\|R_l\|_F}.$$
We define the **Joint Value-Key Operator**:
$$C_l \equiv \begin{bmatrix} \lambda_V V_l \\ \lambda_K R_l \end{bmatrix} \in \mathbb{R}^{(2 D_{\text{readout}}) \times (ND)}.$$
The joint transmission magnitude is:
$$\tau_{\text{joint}}(\Delta P) \equiv \|C_l \, \text{vec}(\Delta P^\top)\|_2 = \sqrt{\lambda_V^2 \|V_l \, \text{vec}(\Delta P^\top)\|^2 + \lambda_K^2 \|R_l \, \text{vec}(\Delta P^\top)\|^2}.$$

### 2.4 Multi-Block Cumulative Extension
If local suppression at block $l$ is insufficient due to downstream regeneration, we construct the cumulative multi-block joint operator:
$$C_{l \to L} \equiv \begin{bmatrix} \lambda_J J_{l \to L} \\ \lambda_{R_{\text{cum}}} R_{l \to L} \end{bmatrix} \in \mathbb{R}^{(D_{\text{readout}} + (L - l)D) \times (ND)}$$
where $R_{l \to L} = [R_{l, l}^\top, R_{l+1, l}^\top, \dots, R_{L-1, l}^\top]^\top$ captures key rerouting sensitivities at all downstream blocks with respect to $\Delta P_l$.

---

## 3. Subspace Geometry: The Hard Intersection View

1. **Row Spaces:**
   Let $U_V \in \mathbb{R}^{(ND) \times K_V}$ and $U_R \in \mathbb{R}^{(ND) \times K_R}$ be orthonormal bases for the row spaces of $V_l$ and $R_l$.
   We compute the SVD of their mutual projection matrix:
   $$M = U_V^\top U_R \in \mathbb{R}^{K_V \times K_R} \implies M = P \Sigma Q^\top, \quad \Sigma = \text{diag}(s_1, s_2, \dots, s_K).$$
   The singular values $s_k \in [0, 1]$ are the **principal cosines**, and $\theta_k = \arccos(s_k)$ are the **principal angles**.
2. **Null Spaces:**
   The exact shared null space is:
   $$\mathcal{N}_{\text{shared}} = \mathcal{N}(V_l) \cap \mathcal{N}(R_l) = \mathcal{N}(C_l).$$
   By the rank-nullity theorem:
   $$\dim(\mathcal{N}_{\text{shared}}) = ND - \text{rank}(C_l) \ge ND - 2 D_{\text{readout}}.$$
   For DeiT-Small ($ND = 75,264, D_{\text{readout}} = 384$):
   $$\dim(\mathcal{N}_{\text{shared}}) \ge 75,264 - 768 = 74,496 \quad (\ge 98.98\%).$$
   The shared null space is mathematically guaranteed to be massive. The scientific question is not whether it exists, but whether modes inside it protect against full-model nonlinear damage and attention shifts.

---

## 4. Experimental Suite & Deliverables

### Phase 1: Numerical Operator Validation
- Verify $V_l$ and $R_l$ against finite-difference step-sizes $\epsilon \in [10^{-1}, 10^{-6}]$ in float64.
- Confirm exact first-order Taylor scaling: error $\mathcal{O}(\epsilon)$.

### Phase 2: Spectral & Subspace Overlap Analysis
- Measure singular spectra of $V_l$, $R_l$, and $C_l$.
- Compute principal cosines, mean angle, and minimum angle between RowSpace($V_l$) and RowSpace($R_l$).
- Save to `value_operator_spectrum.csv`, `key_operator_spectrum.csv`, `vk_subspace_overlap.csv`, `joint_operator_spectrum.csv`.

### Phase 3: Finite-Radius Full-Model Sweeps
- Compare 6 modes at matched Frobenius norm across radius $s \in [0.0, 4.0]$:
  1. `top_high_transmission`: Top singular mode of $C_l$.
  2. `value_only_null`: Exact null mode of $V_l$.
  3. `key_only_null`: Exact null mode of $R_l$.
  4. `multiblock_J_null`: Exact null mode of downstream Jacobian $J_{l \to L}$.
  5. `joint_vk_null`: Exact null mode of $C_l$ ($\mathcal{N}_V \cap \mathcal{N}_K$).
  6. `random_matched_norm`: Isotropic Gaussian control.
- Measure: Logit $L_2$, KL divergence, margin damage, Top-1 accuracy, Top-1 flip rate, readout disturbance.
- Save to `finite_radius_curves.csv`.

### Phase 4: Attention Shift & Power-Law Scaling
- Measure attention shifts across radii:
  - $\|A_{\text{pert}} - A_{\text{clean}}\|_F$
  - Attention KL divergence
  - Attention entropy change
  - Readout-to-patch attention weight change
- Fit power-law scaling $\ln(\text{damage}) = p \ln(\alpha) + c$ on small radii ($s \in [0.05, 0.4]$).
- Save to `attention_shift.csv` and `scaling_exponents.csv`.

### Phase 5: Blockwise Rerouting & Downstream Tracing
- Trace modes through each downstream block:
  - Patch perturbation norm $\|\Delta P_b\|_F$
  - Attention shift $\|A_{\text{pert}, b} - A_{\text{clean}, b}\|_F$
  - Readout disturbance $\|\Delta z_b\|$
  - Query disturbance $\|\delta Q_b\|_F / \|Q_b\|_F$
- Save to `blockwise_rerouting.csv`.

### Phase 6: Out-of-Sample Damage Prediction ($M = 100$)
- Generate $M = 100$ held-out random perturbations.
- Compare predictive correlations (Pearson $r$, Spearman $\rho$, $R^2$) of:
  - $\tau_{\text{value}} = \|V_l \text{vec}(\Delta P^\top)\|$
  - $\tau_{\text{key}} = \|R_l \text{vec}(\Delta P^\top)\|$
  - $\tau_{\text{joint}} = \|C_l \text{vec}(\Delta P^\top)\|$
  - $\tau_{\text{multi}} = \|J_{l \to L} \text{vec}(\Delta P^\top)\|$
- Save to `random_prediction.csv`.

### Phase 7: Depth Comparison & Cross-Architecture Replication
- Depths: 2 (early fragile), 8 (peak fungibility), 11 (terminal late).
- Architectures: DeiT-Small, ViT-Base, DeiT-Tiny, DINOv2 ViT-S/14.
- Save to `depth_comparison.csv`, `replication_summary.csv`, `validation_manifest.json`.

---

## 5. Execution Safeguards
- Strictly avoid modifying `PAPER_DRAFT.md`.
- Never terminate background tasks without user instruction.
- Respect native dual pooling for DINOv2 (CLS + Patch Mean).
