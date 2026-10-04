# Experimental Protocol: Token-Transmission Predictive Operator of Patch-Content Fungibility

**Repository**: [https://github.com/nhatminh-115/Patch-Content-Fungibility](https://github.com/nhatminh-115/Patch-Content-Fungibility)  
**Experiment Name**: TOKEN-TRANSMISSION PREDICTIVE OPERATOR ($T_v$)  
**Status**: ACTIVE PROTOCOL  
**Created**: 2026-10-04  
**Target Output Namespace**: `outputs/fungibility_predictive_operator/`  
**Target Report**: `docs/FUNGIBILITY_PREDICTIVE_OPERATOR_REPORT.md`  
**Figures**: `figures/fungibility_predictive_operator/`  

---

## 1. Executive Summary & Scientific Motivation

Previous experiments established three decisive empirical facts:
1. Patch-stream perturbation damage is jointly controlled by **feature direction ($v \in \mathbb{R}^D$) $\times$ token coherence ($a \in \mathbb{R}^N$)**.
2. The coherence gap is overwhelmingly governed by the **frozen-attention value pathway**:
   $$\Delta z_{\text{readout}} \approx \sum_{h=1}^H W_O^h \left[ (w_h^\top a) (v^\top W_{V, h}) \right]$$
3. Incoherent/high-frequency patterns undergo linear destructive cancellation during attention pooling ($\sum_i w_i a_i \approx 0$), while coherent patterns accumulate constructively.

**However, until now, finding which token modes are fungible required running brute-force finite intervention sweeps.**

The goal of this research step is to **transition from post-hoc empirical discovery to an a priori predictive operator**:
> *"Can we inspect clean attention and value projections, construct a linear transmission operator $T_v$, compute its singular spectrum, and directly predict which collective token modes are functionally tolerated and which are fragile—without running an intervention sweep first?"*

---

## 2. Mathematical Definition of the Token-Transmission Operator $T_v$

For a fixed transformer layer $l$ (with $H$ attention heads, head dimension $d$, embedding dimension $D = H d$) and a fixed unit feature direction $v \in \mathbb{R}^D$ ($\|v\|_2 = 1.0$), we define the **Token-Transmission Operator**:
$$T_v : \mathbb{R}^N \to \mathbb{R}^{D_{\text{out}}}$$
such that for any token-space perturbation pattern $a \in \mathbb{R}^N$:
$$T_v a = \sum_{h=1}^H W_O^h \left[ (w_h^\top a) (v^\top W_{V, h}) \right]$$
where:
- $w_h \in \mathbb{R}^N$: Clean attention weights from the readout query to spatial patches for head $h$.
- $W_{V, h} \in \mathbb{R}^{d \times D}$: Value projection weight matrix slice for head $h$.
- $W_O^h \in \mathbb{R}^{D_{\text{out}} \times d}$: Output projection weight matrix slice for head $h$.

### 2.1 Native Matrix Representation
Because $T_v a$ is linear in $a$, we represent $T_v$ as a matrix $M_v \in \mathbb{R}^{D_{\text{out}} \times N}$.
For each patch column $i \in \{1, \dots, N\}$:
$$(M_v)_{:, i} = \sum_{h=1}^H w_{h, i} \left( (v^\top W_{V, h}) W_O^h \right)^\top$$
so that $T_v a = M_v a$.

### 2.2 Readout Topology Adaptations
1. **CLS-Readout Models (DeiT-Small, DeiT-Tiny, ViT-B/16)**:
   - Query is the `[CLS]` token (index 0).
   - $w_h = A_{h, 0, 1:} \in \mathbb{R}^N$.
   - $D_{\text{out}} = D$.
2. **Dual-Pooled Readout (DINOv2 ViT-S/14)**:
   - Native readout is $\text{concat}(z_{\text{CLS}}, \bar{z}_{\text{patch}}) \in \mathbb{R}^{2D}$.
   - $w_{h, \text{CLS}} = A_{h, 0, 1:} \in \mathbb{R}^N$.
   - $w_{h, \text{patch}} = \frac{1}{N} \sum_{k=1}^N A_{h, k, 1:} \in \mathbb{R}^N$.
   - $T_v = \begin{bmatrix} T_{v, \text{CLS}} \\ T_{v, \text{patch}} \end{bmatrix} \in \mathbb{R}^{2D \times N}$.

---

## 3. Singular Value Decomposition & Mathematical Objects

We compute the Singular Value Decomposition:
$$T_v = U \Sigma V^\top = \sum_{k=1}^{\min(D_{\text{out}}, N)} \sigma_k u_k q_k^\top$$
where:
- Singular values: $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_N \ge 0$.
- Right singular vectors $q_k \in \mathbb{R}^N$ form an **orthonormal basis of token space** ($\|q_k\|_2 = 1$, $q_j^\top q_k = \delta_{jk}$).

### 3.1 Predicted Transmission Energy
For any token pattern $a \in \mathbb{R}^N$:
$$\tau_v(a) = \|T_v a\|_2$$
For the singular modes $a = q_k$:
$$\tau_v(q_k) = \|T_v q_k\|_2 = \sigma_k$$

### 3.2 Formal Definition of the Approximate Fungible Subspace
For a relative threshold $\epsilon > 0$:
$$\mathcal{F}_v(\epsilon) = \left\{ a \in \mathbb{R}^N : \|a\|_2 = 1, \; \frac{\|T_v a\|_2}{\sigma_1} \le \epsilon \right\}$$
$$\mathcal{N}_v(\epsilon) = \text{span} \left\{ q_k : \frac{\sigma_k}{\sigma_1} \le \epsilon \right\}$$
Because $T_v$ is formed by a sum of $H$ rank-1 head matrices, its mathematical rank is **at most $H$**:
$$\text{rank}(T_v) \le H \ll N$$
Thus, there exists an exact kernel of dimension $\ge N - H$ ($190$ out of $196$ dimensions in DeiT-Small) where $\sigma_k \approx 0$.

---

## 4. Primary Hypotheses to Test

| Hypothesis | Theoretical Statement | Empirical Prediction |
| :--- | :--- | :--- |
| **H1: Monotonic Transmission Law** | $\sigma_k$ directly predicts real finite model damage without fitting. | Spearman $\rho(\sigma_k, \text{Damage}) \ge 0.85$, Pearson $r \ge 0.85$. |
| **H2: Extreme Contrast** | Top singular mode $q_1$ is maximally fragile; null modes $q_k \in \mathcal{N}_v$ are highly fungible at matched norm. | Real logit $L_2$ and margin drop of $q_1$ are $\ge 5\times - 10\times$ higher than $q_N$. |
| **H3: Finite-Radius Stability** | The ordering predicted by $\sigma_k$ persists across finite perturbation scales $s \in [0.0, 4.0]$. | Damage curves for $q_1$ cross tolerance thresholds early ($s < 1.0$), while null modes remain tolerated up to $s = 4.0$. |
| **H4: Out-of-Basis Generalization** | $\tau_v(a) = \|T_v a\|$ explains previously tested patterns (`global_coherent`, `random_sign`, `checkerboard`). | $\tau_v(\text{coherent}) \gg \tau_v(\text{random\_sign})$; $\tau_v(a)$ correlates with real damage across arbitrary patterns. |
| **H5: Unseen Random Pattern Prediction** | For random Gaussian patterns $a_j \sim \mathcal{N}(0, I)$, $\tau_v(a_j)$ predicts variation in damage. | Significant correlation ($r \ge 0.70$) between predicted $\tau_v(a_j)$ and observed damage. |
| **H6: Cross-Architecture Replication** | The predictive operator principle holds across supervised and self-supervised architectures. | Validated on DeiT-Small, ViT-B/16, DeiT-Tiny, and DINOv2. |

---

## 5. Experimental Design & Protocol

### 5.1 Models and Canonical Depths
1. **Exploratory Models**:
   - **DeiT-Small**: Peak-fungibility Depth 8, Fragile Depth 5, Terminal Depth 10.
   - **ViT-B/16 AugReg**: Peak-fungibility Depth 7, Fragile Depth 5, Terminal Depth 10.
2. **Confirmatory Replication Models**:
   - **DeiT-Tiny**: Depth 8.
   - **DINOv2 ViT-S/14**: Depth 8 (dual pooled readout).
3. **Evaluation Set**: 100 ImageNet validation images (disjoint splits, clean calibration).

### 5.2 Feature Directions
1. `jac_top`: Principal eigenvector of functional Hessian proxy $M_l = \mathbb{E}[J^\top J]$.
2. `jac_null`: Smallest non-zero eigenvector of $M_l$.
3. `pc1`: Leading principal component of clean activation covariance $\Sigma_l$.
4. `rand_dir`: Isotropic random unit vector.
5. `centroid_dir`: Direction of calibration spatial centroid.

### 5.3 Evaluation Battery
1. **Singular Mode Validation**:
   - Modes evaluated: $q_1$ (top), $q_2$, $q_{\text{mid}}$, $q_{95\%}$, $q_N$ (bottom).
   - Perturbation: $\Delta P = \alpha q_k v^\top$ with $\|\Delta P\|_F = \alpha = 1.0 \sigma_P$.
   - Tested on the **real unconstrained model** without freezing attention.
2. **Finite-Radius Curves**:
   - Scale grid: $s \in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.0]$.
   - Curves traced for $q_1$, $q_{\text{mid}}$, and $q_N$.
3. **Out-of-Basis Patterns**:
   - `global_coherent`, `random_sign`, `checkerboard`, `spatial_cluster_25%`, `random_gaussian`, `smooth_spatial`.
   - Measure predicted transmission $\tau_v(a) = \|T_v a\|$ vs full-model margin drop.
4. **Random Held-Out Patterns**:
   - $M = 50$ random patterns $a_j \sim \mathcal{N}(0, I)$, normalized.
   - Measure Pearson $r$ and Spearman $\rho$ between $\tau_v(a_j)$ and observed damage.
5. **Depth Spectrum Evolution**:
   - Track singular value distribution and approximate null space dimension:
     $$\text{NullDim}_v(\epsilon) = \# \{k : \sigma_k / \sigma_1 \le \epsilon\}$$
     for $\epsilon \in \{10^{-2}, 10^{-3}, 10^{-4}\}$.
6. **Feature $\times$ Token Bilinear Matrix**:
   - Predict $\tau(v_j, a_k) = \|T_{v_j} a_k\|$ and compare against observed damage matrix.

---

## 6. Machine-Readable Outputs & Figures

All outputs saved under:
- `outputs/fungibility_predictive_operator/`
  1. `operator_spectrum.csv`
  2. `singular_mode_validation.csv`
  3. `random_pattern_prediction.csv`
  4. `finite_radius_curves.csv`
  5. `depth_spectrum.csv`
  6. `feature_token_matrix.csv`
  7. `spatial_mode_analysis.csv`
  8. `replication_summary.csv`
  9. `validation_manifest.json`

- `figures/fungibility_predictive_operator/`
  1. `figure_a_operator_spectrum.png`
  2. `figure_b_predicted_vs_observed_damage.png`
  3. `figure_c_top_vs_null_modes.png`
  4. `figure_d_finite_radius_validation.png`
  5. `figure_e_depth_evolution.png`
  6. `figure_f_cross_architecture_replication.png`
  7. `figure_g_token_mode_spatial_structure.png` (optional)
