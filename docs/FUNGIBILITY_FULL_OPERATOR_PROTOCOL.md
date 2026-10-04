# Fungibility Full Operator Protocol: Unified Token × Feature Transmission Spectrum

## 1. Background & Scientific Objective

Prior investigations established three foundational properties of late patch-content fungibility in Vision Transformers:
1. **Strong Anisotropy:** Function preservation depends critically on alignment with low-sensitivity feature subspaces.
2. **Joint Token-Feature Geometry:** Downstream damage depends jointly on feature direction and token-space coherence ($\Delta P = \alpha a v^\top$).
3. **Causal Attention Mechanism:** Coherence effects are transmitted predominantly through the frozen-attention value pathway, described by the clean token-transmission operator $T_v a = \sum_h W_O^h [(w_h^\top a)(v^\top W_{V,h})]$.

However, $T_v$ suffered from an intrinsic limitation: **it required choosing a specific feature direction $v \in \mathbb{R}^D$ in advance**. 

The goal of this protocol is to construct, diagonalize, and validate a **single, unified mathematical operator $A_l$** acting directly on the entire unconstrained patch-stream perturbation space:
$$\Delta P \in \mathbb{R}^{N \times D}, \quad \text{vec}(\Delta P^\top) \in \mathbb{R}^{ND}.$$

The central question is:
> *Can we construct a full clean-state operator whose spectrum directly identifies the most fungible and most fragile perturbations jointly across token space $\times$ feature space, explaining empirical damage on held-out perturbations and finite perturbation radii?*

---

## 2. Mathematical Formulation of the Full Operator $A_l$

### 2.1 Standard CLS-Readout Topology (DeiT, ViT)

Let $l$ denote the perturbation depth, and consider the downstream attention block $l$. For an input activation $h \in \mathbb{R}^{(N+1) \times D}$ (with token 0 as CLS and tokens $1, \dots, N$ as patch tokens), let $\Delta P \in \mathbb{R}^{N \times D}$ be an arbitrary patch-stream perturbation:
$$\Delta h = \begin{bmatrix} 0 \\ \Delta P \end{bmatrix} \in \mathbb{R}^{(N+1) \times D}.$$

Under frozen clean attention weights $w_{h, i} \in [0, 1]$ (where $w_{h, i}$ denotes the attention weight head $h$ places on patch token $i \in \{1, \dots, N\}$ from the readout query), the value-path readout perturbation $\Delta z \in \mathbb{R}^D$ is:
$$\Delta z = \sum_{h=1}^H W_O^h \left[ \sum_{i=1}^N w_{h, i} \Delta p_i W_{V,h} \right],$$
where:
- $H$ is the number of attention heads,
- $d = D / H$ is the head dimension,
- $W_{V,h} \in \mathbb{R}^{d \times D}$ is the head-specific value projection matrix,
- $W_O^h \in \mathbb{R}^{D \times d}$ is the head-specific output projection slice,
- $\Delta p_i \in \mathbb{R}^{1 \times D}$ is row $i$ of $\Delta P$.

Let $M_h \equiv W_O^h W_{V,h} \in \mathbb{R}^{D \times D}$. Since $\Delta p_i W_{V,h} \in \mathbb{R}^{1 \times d}$, taking transposes or writing column-wise:
$$W_O^h (\Delta p_i W_{V,h})^\top = W_O^h W_{V,h} \Delta p_i^\top = M_h \Delta p_i^\top.$$
Summing over heads $h$, the transmission kernel for patch token $i$ is:
$$K_i \equiv \sum_{h=1}^H w_{h, i} M_h \in \mathbb{R}^{D \times D}.$$

Then the net readout perturbation is:
$$\Delta z = \sum_{i=1}^N K_i \Delta p_i^\top = \begin{bmatrix} K_1 & K_2 & \dots & K_N \end{bmatrix} \begin{bmatrix} \Delta p_1^\top \\ \Delta p_2^\top \\ \vdots \\ \Delta p_N^\top \end{bmatrix}.$$

Letting $x = \text{vec}(\Delta P^\top) \in \mathbb{R}^{ND}$, we define the **Full Token $\times$ Feature Transmission Operator**:
$$A_l \equiv \begin{bmatrix} K_1 & K_2 & \dots & K_N \end{bmatrix} \in \mathbb{R}^{D \times (ND)}.$$
Thus:
$$\Delta z = A_l \, \text{vec}(\Delta P^\top).$$

### 2.2 Dual-Channel Readout Topology (DINOv2)

DINOv2 feeds the downstream classification head a concatenated representation:
$$z_{\text{readout}} = \begin{bmatrix} z_{\text{CLS}} \\ z_{\text{patch\_mean}} \end{bmatrix} \in \mathbb{R}^{2D}.$$
For this native topology:
1. $w_{\text{CLS}, h, i}$ is the clean attention weight from CLS to patch $i$ in head $h$.
2. $w_{\text{patch}, h, i} = \frac{1}{N} \sum_{j=1}^N A_{\text{clean}}[h, j, i]$ is the mean attention weight from all patch queries to patch $i$.

We define:
$$K_{i, \text{CLS}} = \sum_{h=1}^H w_{\text{CLS}, h, i} M_h \in \mathbb{R}^{D \times D}, \quad K_{i, \text{patch}} = \sum_{h=1}^H w_{\text{patch}, h, i} M_h \in \mathbb{R}^{D \times D},$$
and stack them into:
$$K_i = \begin{bmatrix} K_{i, \text{CLS}} \\ K_{i, \text{patch}} \end{bmatrix} \in \mathbb{R}^{2D \times D}, \quad A_l = \begin{bmatrix} K_1 & K_2 & \dots & K_N \end{bmatrix} \in \mathbb{R}^{2D \times (ND)}.$$

---

## 3. Dimensionality, Rank, and Exact Null-Space Geometry

For a Vision Transformer with $N$ patch tokens and embedding dimension $D$:
- **DeiT-Small / ViT-S:** $N = 196$, $D = 384$, $ND = 75,264$. Operator $A_l \in \mathbb{R}^{384 \times 75,264}$.
- **ViT-B/16:** $N = 196$, $D = 768$, $ND = 150,528$. Operator $A_l \in \mathbb{R}^{768 \times 150,528}$.
- **DINOv2 ViT-S/14:** $N = 256$, $D = 384$, $ND = 98,304$. Operator $A_l \in \mathbb{R}^{768 \times 98,304}$.

### Rank Upper Bound
Because $A_l$ has row dimension $D_{\text{out}} \in \{D, 2D\}$:
$$\text{rank}(A_l) \le D_{\text{out}} \ll ND.$$
Therefore, the **exact linear null space** of $A_l$:
$$\mathcal{N}(A_l) = \{ x \in \mathbb{R}^{ND} : A_l x = 0 \}$$
has dimension:
$$\dim \mathcal{N}(A_l) = ND - \text{rank}(A_l) \ge ND - D_{\text{out}}.$$
- For DeiT-Small: $\dim \mathcal{N}(A_l) \ge 75,264 - 384 = \mathbf{74,880}$ (**99.49% of the perturbation space**).
- For ViT-Base: $\dim \mathcal{N}(A_l) \ge 150,528 - 768 = \mathbf{149,760}$ (**99.49% of the perturbation space**).
- For DINOv2 ViT-S/14: $\dim \mathcal{N}(A_l) \ge 98,304 - 768 = \mathbf{97,536}$ (**99.22% of the perturbation space**).

---

## 4. Exact SVD Computation via Gram Eigendecomposition

Because $D_{\text{out}} \ll ND$, materializing $A_l \in \mathbb{R}^{D_{\text{out}} \times (ND)}$ requires only $110\text{--}460$ MB of memory. SVD is computed instantly by forming the Gram matrix:
$$G = A_l A_l^\top \in \mathbb{R}^{D_{\text{out}} \times D_{\text{out}}}.$$
1. Compute orthonormal eigenvectors $U \in \mathbb{R}^{D_{\text{out}} \times D_{\text{out}}}$ and eigenvalues $\lambda_k$:
   $$G U = U \Lambda, \quad \sigma_k = \sqrt{\max(\lambda_k, 0)}.$$
2. For each non-zero singular value $\sigma_k > 0$, the corresponding right singular vector in $\mathbb{R}^{ND}$ is:
   $$v_k = \frac{1}{\sigma_k} A_l^\top U_{:, k}.$$
3. Reshape $v_k$ to matrix form $Q_k \in \mathbb{R}^{N \times D}$ such that $\|Q_k\|_F = 1$.

### Fast Exact Null Mode Extraction
For any random matrix $R \in \mathbb{R}^{N \times D}$ with $r = \text{vec}(R^\top) \in \mathbb{R}^{ND}$:
$$r_{\text{row}} = A_l^\top (A_l A_l^\top)^{-1} A_l r = A_l^\top U \Lambda^{-1} U^\top A_l r,$$
$$r_{\text{null}} = r - r_{\text{row}}, \quad Q_{\text{null}} = \frac{r_{\text{null}}}{\|r_{\text{null}}\|_2} \in \mathbb{R}^{N \times D}.$$
By construction, $A_l \, \text{vec}(Q_{\text{null}}^\top) = 0$ to numerical machine precision.

---

## 5. Experimental Test Suite

The protocol executes 10 core evaluations:

1. **Synthetic & Consistency Validation:**
   - Parity against naive materialization.
   - For separable perturbations $\Delta P = a v^\top$, verify $A_l \text{vec}(\Delta P^\top) \equiv T_v a$ within relative tolerance $< 10^{-5}$.
2. **Spectrum Characterization & Mode Extraction:**
   - Singular spectrum decay, condition number $\sigma_1 / \sigma_r$, effective rank.
   - Extract top modes ($Q_1, Q_2$), median modes ($Q_{\text{mid}}$), lowest non-zero modes ($Q_{\text{bot}}$), and exact null modes ($Q_{\text{null}}$).
3. **Mode Factorization (Separable vs Entangled):**
   - For each singular mode $Q_k \in \mathbb{R}^{N \times D}$, compute SVD of $Q_k$ and measure rank-1 energy fraction:
     $$E_1(Q_k) = \frac{\sigma_1(Q_k)^2}{\sum_i \sigma_i(Q_k)^2}.$$
4. **Real Full-Model Finite-Radius Sweep:**
   - For each singular mode, evaluate downstream unconstrained model damage across scale sweep:
     $$s = \alpha / \sigma_P \in \{0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.0\}.$$
   - Metrics: Logit $L_2$, KL divergence, margin damage, Top-1 accuracy / flip rate, readout disturbance.
5. **Random Held-Out Perturbation Prediction:**
   - Sample $M = 100$ arbitrary random Gaussian perturbations $\Delta P_j \sim \mathcal{N}(0, I)$ normalized to $\|\Delta P_j\|_F = 1$.
   - Predict linear transmission: $\tau_j = \|A_l \text{vec}(\Delta P_j^\top)\|_2$.
   - Measure actual unconstrained full-model damage at fixed finite radius $s = 0.4$.
   - Compute Pearson $r$, Spearman $\rho$, and calibration curves.
6. **Depth Evolution:**
   - Compute $A_l$ across fragile early depth ($l=2$), peak fungible depth ($l=8$), and terminal depth ($l=11$).
7. **Projection of Historical Interventions:**
   - Project canonical interventions (centroid replacement, Gaussian replacements, PC1 vs random directions, checkerboard, random-sign) onto $A_l$ high-transmission, mid-transmission, and null subspaces.
8. **Residual Damage Breakdown:**
   - For exact null modes where $A_l \Delta P = 0$, decompose downstream residual damage: first downstream block output, LayerNorm scale, MLP branch, and later blocks.
9. **Nonlinear Scaling Analysis:**
   - Measure damage across fine small-$\alpha$ grid ($\alpha / \sigma_P \in [0.01, 0.2]$) and fit power law:
     $$\text{Damage} \propto \alpha^p.$$
   - Distinguish missing first-order pathway ($p \approx 1$) from quadratic residual ($p \approx 2$).
10. **Cross-Architecture Replication:**
    - Replicate core findings on DeiT-Small, ViT-B/16, DeiT-Tiny, and DINOv2 ViT-S/14.
