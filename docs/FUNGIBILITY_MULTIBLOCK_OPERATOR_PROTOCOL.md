# Fungibility Multi-Block Operator Protocol: Downstream-Persistent Low-Transmission Geometry

## 1. Scientific Context & Central Question

In the previous phase, we constructed the single-block token $\times$ feature transmission operator:
$$A_l : \mathbb{R}^{N \times D} \to \mathbb{R}^{D_{\text{out}}}, \quad \Delta z_{\text{readout}}^{(l)} = A_l \, \text{vec}(\Delta P^\top).$$
This established:
1. $A_l$ has an exact mathematical null space occupying $\ge 99.49\%$ of the perturbation space.
2. Low-transmission and exact null modes of $A_l$ remain significantly more tolerant in the full nonlinear Transformer than top singular modes.
3. Clean linear transmission $\tau_l(\Delta P) = \|A_l \text{vec}(\Delta P^\top)\|$ strongly predicts held-out damage ($r = 0.9485$).
4. Canonical interventions (centroid, independent Gaussian, checkerboard, random-sign) are largely explained by their projection onto the spectrum of $A_l$.

### The Remaining Fundamental Bottleneck: Inter-Block Leakage
Despite having zero linear transmission at block $l$ ($A_l \Delta P = 0$), exact single-block null modes still produce modest non-zero damage downstream. Crucially, this residual damage scales **linearly** ($p \approx 1.0$), not quadratically ($p \approx 2.0$).

The physical mechanism is **inter-block leakage**:
- The single-block operator $A_l$ cancels the perturbation's transmission into the readout token at block $l$.
- However, the unperturbed patch residual stream carries the perturbation forward into block $l+1$:
  $$h_{\text{patch}}^{(l+1)} = h_{\text{patch}}^{(l)} + \Delta P + \Delta h_{\text{attn}} + \Delta h_{\text{mlp}}.$$
- Because adjacent blocks have differently oriented attention weights and value projections ($\mathcal{N}(A_l) \ne \mathcal{N}(A_{l+1})$), the perturbation leaks into the high-transmission modes of subsequent blocks.

### The Next Scientific Goal
To advance from **single-block transmission nullity** to **multi-block / downstream-persistent low-transmission geometry**:
> *Can we construct a clean-state operator that predicts which patch-stream perturbations remain weakly transmitted through the entire remaining downstream network, and identify perturbations that are simultaneously low-transmission across all subsequent blocks?*

---

## 2. Mathematical Formulation of Multi-Block Operators

Let $l$ be the intervention depth, and let $L$ be the terminal depth before the classification head.
Let $h_l \in \mathbb{R}^{(N+1) \times D}$ be the clean activation preceding block $l$, with patch stream $P_l = h_l[:, 1:, :] \in \mathbb{R}^{N \times D}$.
Let $\Delta P \in \mathbb{R}^{N \times D}$ be an arbitrary patch perturbation, with $x = \text{vec}(\Delta P^\top) \in \mathbb{R}^{ND}$.

We construct and analyze two complementary operators:

### 2.1 Operator B: Direct End-to-End Downstream Jacobian $J_{l \to L}$
Let $\mathcal{G}_{l \to L}(h)$ denote the frozen downstream mapping from activation at layer $l$ to final readout embedding $z_{\text{final}} \in \mathbb{R}^{D_{\text{readout}}}$ (or logits $y \in \mathbb{R}^C$).
The first-order downstream Taylor linearization around the clean activation $h_l$ is:
$$\delta z_{\text{final}} = J_{l \to L} \, \text{vec}(\Delta P^\top) + \mathcal{O}(\|\Delta P\|^2),$$
where $J_{l \to L} \in \mathbb{R}^{D_{\text{readout}} \times (ND)}$ is the exact clean-state Jacobian:
$$J_{l \to L} \equiv \frac{\partial z_{\text{final}}}{\partial \text{vec}(P_l^\top)}.$$

#### Output Spaces:
1. **Primary Output Space:** Final readout token embedding $z_{\text{readout}} \in \mathbb{R}^D$ (or $\mathbb{R}^{2D}$ for DINOv2).  
   Because the classification head is linear ($\text{logits} = z_{\text{final}} W_{\text{head}}^\top + b$), $J_{\text{logit}} = W_{\text{head}} J_{l \to L}$, sharing the exact same row space.
2. **Scalar Margin Gradient:**
   $$g_{\text{margin}} = \nabla_{\text{vec}(P_l^\top)} \left( \text{logit}_{\text{clean\_top1}} - \text{logit}_{\text{runner\_up}} \right) \in \mathbb{R}^{ND}.$$

#### Fast Exact SVD of $J_{l \to L}$ via Gram Matrix:
Because $D_{\text{readout}} \in \{192, 384, 768\} \ll ND \in \{37632, 75264, 150528\}$:
1. Compute the exact Jacobian $J_{l \to L}$ in $\approx 1.5$ seconds using batched vector-Jacobian products (VJPs) with unit basis vectors $e_d \in \mathbb{R}^{D_{\text{readout}}}$.
2. Form the Gram matrix $G_J = J_{l \to L} J_{l \to L}^\top \in \mathbb{R}^{D_{\text{readout}} \times D_{\text{readout}}}$.
3. Diagonalize: $G_J U_J = U_J \Lambda_J, \quad \sigma_k = \sqrt{\lambda_k}$.
4. Right singular vectors in $\mathbb{R}^{ND}$:
   $$v_k = \frac{1}{\sigma_k} J_{l \to L}^\top U_{J, :, k}, \quad Q_k = \text{reshape}(v_k, (N, D)).$$
5. Exact multi-block null mode via pseudo-inverse projection of random Gaussian $r \in \mathbb{R}^{ND}$:
   $$r_{\text{row}} = J_{l \to L}^\top (J_{l \to L} J_{l \to L}^\top)^{-1} J_{l \to L} r, \quad r_{\text{null}} = r - r_{\text{row}}, \quad v_{\text{multi-null}} = \frac{r_{\text{null}}}{\|r_{\text{null}}\|_2}.$$
   By construction, $\|J_{l \to L} v_{\text{multi-null}}\|_2 = 0$ to numerical precision.

---

### 2.2 Operator A: Local Single-Block Composition Model $C_{l \to L}$
To track the physical mechanism of leakage, we construct the sequence of single-block operators:
$$A_b \in \mathbb{R}^{D_{\text{out}} \times (ND)} \quad \text{for } b \in \{l, l+1, \dots, L\}.$$
For any perturbation $q$, we propagate $\Delta P_b$ through each downstream block $b$:
1. Readout disturbance at block $b$: $\tau_b = \|A_b \text{vec}(\Delta P_b^\top)\|$.
2. Patch-stream propagation: $\Delta P_{b+1} = \text{Block}_b(P_b + \Delta P_b) - \text{Block}_b(P_b)$.
3. Cumulative visibility score:
   $$\mathcal{V}_l(q) = \sum_{b=l}^L \|A_b \, \text{vec}(\Delta P_b^\top)\|^2.$$

---

## 3. Key Experimental Hypotheses

### Hypothesis 1: Downstream-Persistent Nullity vs Single-Block Nullity
A perturbation in $\mathcal{N}(A_l)$ is null only to the immediate block $l$. Its transmission through $J_{l \to L}$ will be non-zero due to inter-block leakage. Conversely, a perturbation in $\mathcal{N}(J_{l \to L})$ cancels downstream transmission through the entire network, and will therefore cause significantly lower damage in the full nonlinear model.

### Hypothesis 2: Null-Space Rotation Mechanism
Adjacent blocks have huge null spaces of identical dimension ($\ge 99.49\%$), but their orientation rotates significantly across depth:
$$\angle(\text{RowSpace}(A_b), \text{RowSpace}(A_{b+1})) \gg 0^\circ.$$
This rotation is the primary algebraic driver of leakage.

### Hypothesis 3: Residual Power-Law Scaling Suppression
If inter-block leakage is the primary source of the $p \approx 1.0$ linear residual damage in single-block null modes, then multi-block null modes (which eliminate first-order downstream transmission) should substantially reduce the linear coefficient.

---

## 4. Test Suite Execution Plan

1. **Spectrum Comparison:** Single-block $A_l$ spectrum vs Multi-block $J_{l \to L}$ spectrum across depths 2, 8, 11.
2. **Survival & Overlap Analysis:**
   - Survival ratio: $\|J_{l \to L} q_{\text{single-null}}\| / \|J_{l \to L} q_{\text{multi-top}}\|$.
   - Principal angles and canonical correlations between $\text{RowSpace}(A_b)$ and $\text{RowSpace}(A_{b+1})$.
3. **Downstream Leakage Trace:** Block-by-block trace of $\|\Delta P_b\|_F$, readout disturbance $\|A_b \Delta P_b\|$, and projection onto next block's high-transmission subspace.
4. **Finite-Radius Unconstrained Model Sweep:**
   - Perturbations: Single-block top, Multi-block top, Single-block null, Multi-block null, Random Gaussian.
   - Sweep scales: $s \in [0.0, 4.0]$.
   - Metrics: Logit $L_2$, KL divergence, margin damage, Top-1 flip rate, readout disturbance.
5. **Held-Out Perturbation Prediction ($M=100$):**
   - Compare single-block score $\tau_{\text{single}}$ vs multi-block score $\tau_{\text{multi}}$ vs cumulative visibility $\mathcal{V}$ in predicting full-model damage.
6. **Power-Law Scaling:** Fit $\ln(\text{Damage}) = p \ln(\alpha) + c$ on fine small-$\alpha$ grid ($s \in [0.01, 0.25]$).
7. **Historical Intervention Alignment:** Compare single-block vs multi-block projection of canonical interventions.
8. **Cross-Architecture Replication:** DeiT-Small, ViT-B/16, DeiT-Tiny, DINOv2 ViT-S/14.
