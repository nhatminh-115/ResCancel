# Protocol: Fungibility Joint Stream Geometry (Feature Direction × Token-Space Pattern)

**Date:** 2026-10-04  
**Status:** Frozen Prior to Execution  
**Target Repository:** `nhatminh-115/ResCancel` (Patch Content Fungibility)  
**Authors:** Mechanistic Research Program

---

## 1. Scientific Motivation & Core Objective

The previous functional geometry sprint established that:
1. Patch-content fungibility is **violently anisotropic** ($>32\times$ spread in tolerance radii).
2. Late transformer layers develop an expansive **functional near-null subspace** ($\ker M_l$ reaching $>50\%$ of all activation coordinates by Depth 10).
3. **Single-token perturbations are vastly more tolerated than multi-token perturbations** ($r \ge 4.0$ for single tokens, but collective collapse occurs along sensitive directions when 25% or 100% of tokens are perturbed).

This directly raises the central mechanistic question:

$$\text{Does patch-content fungibility depend jointly on FEATURE DIRECTION and TOKEN COHERENCE?}$$

Rather than treating the patch stream as independent slots or collapsing the stream into a single token vector, we empirically map the **joint space**:

$$\text{FEATURE DIRECTION } v \in \mathbb{R}^D \quad \times \quad \text{TOKEN-SPACE PATTERN } a \in \mathbb{R}^N$$

Under strict **Frobenius norm matching**, we investigate whether downstream functional tolerance is governed by:
- Feature Sensitivity alone ($v$-dominated),
- Token Coherence alone ($a$-dominated), or
- A non-trivial **Interaction** between feature direction and spatial token pattern.

---

## 2. Mathematical Formulation & Strict Norm Matching

At layer $l$, the patch stream activation tensor is $P_l \in \mathbb{R}^{N \times D}$, where $N$ is the number of spatial patch tokens ($N = 196$ for $14 \times 14$ ViTs; $N = 256$ for $16 \times 16$ DINOv2) and $D$ is the embedding dimension.

We construct structured stream perturbations of rank-1 outer product form:

$$\Delta P = \alpha \, a \, v^\top \in \mathbb{R}^{N \times D}$$

where:
- $v \in \mathbb{R}^D$ is a unit-norm feature-space vector: $\|v\|_2 = 1$.
- $a \in \mathbb{R}^N$ is a token-space distribution vector across the $N$ spatial positions, normalized such that:
  $$\|a\|_2 = \sqrt{\sum_{i=1}^N a_i^2} = 1.0$$
- $\alpha = s \cdot \sigma_{\text{norm}}$ is the calibrated perturbation amplitude, where $\sigma_{\text{norm}} = \sqrt{\text{tr}(\Sigma_l)}$ is the total natural activation standard deviation at depth $l$, and $s \in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]$.

### Proof of Invariant Frobenius Norm:
$$\|\Delta P\|_F = \sqrt{\sum_{i=1}^N \sum_{d=1}^D (\alpha a_i v_d)^2} = \alpha \sqrt{\sum_{i=1}^N a_i^2} \sqrt{\sum_{d=1}^D v_d^2} = \alpha \cdot \|a\|_2 \cdot \|v\|_2 = \alpha \cdot 1.0 \cdot 1.0 = \alpha$$

**Crucial experimental guarantee:** At any given scale $s$, every perturbation pattern injects exactly the same total energy $\|\Delta P\|_F = \alpha$ into the patch stream. No global pattern has more energy simply because more tokens participate.

---

## 3. Definition of Token-Space Patterns ($a \in \mathbb{R}^N$)

All patterns satisfy $\|a\|_2 = 1.0$:

1. **`single_central` (Localized baseline):**
   $$a_i = \begin{cases} 1.0 & \text{if } i = i_{\text{center}} \\ 0 & \text{otherwise} \end{cases}$$
2. **`single_corner` (Boundary baseline):**
   $$a_i = \begin{cases} 1.0 & \text{if } i = 0 \\ 0 & \text{otherwise} \end{cases}$$
3. **`global_coherent` (Maximum collective coherence):**
   $$a_i = \frac{1}{\sqrt{N}} \quad \forall i \in \{1, \dots, N\}$$
   All patches move uniformly in the exact same direction with positive sign.
4. **`random_sign` (Incoherent matched-norm control):**
   $$a_i = \frac{s_i}{\sqrt{N}}, \quad s_i \in \{-1, +1\} \text{ i.i.d. with } P(s_i = 1) = 0.5$$
   Every patch moves, but signs alternate randomly with $\mathbb{E}[a_i] = 0$.
5. **`random_gaussian` (Continuous zero-mean incoherent):**
   $$z_i \sim \mathcal{N}(0, 1), \quad a = \frac{z}{\|z\|_2}$$
6. **`spatial_cluster_25%` (Contiguous spatial block):**
   For a contiguous 2D spatial block $\mathcal{B}$ containing $k = \lfloor 0.25 N \rfloor$ tokens:
   $$a_i = \begin{cases} \frac{1}{\sqrt{k}} & \text{if } i \in \mathcal{B} \\ 0 & \text{otherwise} \end{cases}$$
7. **`checkerboard` (High-spatial-frequency alternating mode):**
   On the 2D grid $(r, c) \in [0, H_p-1] \times [0, W_p-1]$:
   $$a_{r, c} = \frac{(-1)^{r+c}}{\sqrt{N}}$$
   Preserves strong local pairwise cancellation between adjacent patches.
8. **`smooth_spatial` (Low-spatial-frequency cosine mode):**
   $$m(r, c) = \cos\left(\frac{\pi r}{H_p}\right) \cos\left(\frac{\pi c}{W_p}\right), \quad a = \frac{m}{\|m\|_2}$$

---

## 4. Definition of Feature-Space Directions ($v \in \mathbb{R}^D$)

All directions satisfy $\|v\|_2 = 1.0$:

1. **`jac_null` (Functionally Near-Null):** Smallest-eigenvalue eigenvector of downstream functional metric $M_l = \mathbb{E}[J^\top J]$.
2. **`jac_top` (Functionally Sensitive):** Largest-eigenvalue eigenvector of $M_l$.
3. **`pc1` (Natural Covariance PC1):** Top principal component of clean calibration activations $\Sigma_l$.
4. **`rand_dir` (Isotropic Control):** Matched-norm random unit vector.
5. **`centroid_dir` (Canonical Surrogate Vector):** Normalized calibration centroid $\mu_l / \|\mu_l\|_2$.

---

## 5. Primary Hypotheses & Falsification Criteria

### Hypothesis 1: Coherence Bottleneck (Attention Cancellation)
- *Claim:* At matched Frobenius norm, `global_coherent` causes significantly greater functional damage than `random_sign` or `checkerboard`.
- *Mechanism:* Downstream attention heads aggregate patch streams into the classification token: $h_{\text{CLS}} \leftarrow \sum_i w_i P_i$. When $a$ has zero mean ($\sum_i a_i \approx 0$), linear aggregation cancels the perturbation: $\sum_i w_i (\alpha a_i v) \approx 0$. When $a$ is coherent, errors sum constructively.
- *Falsification:* If `damage(global_coherent) ≈ damage(random_sign)` across scales and directions, the coherence hypothesis is falsified.

### Hypothesis 2: Feature-Space Dominance (Non-Separability)
- *Claim:* For `jac_null`, even `global_coherent` perturbations remain harmless ($r \ge 4.0$). Conversely, for `jac_top`, even incoherent perturbations cause severe damage.
- *Falsification:* If `global_coherent` destroys function along `jac_null`, then null-space fungibility requires incoherent/dispersed token patterns.

### Hypothesis 3: Spatial Frequency Cutoff
- *Claim:* Functional tolerance increases monotonically with spatial frequency of the token pattern (`global_coherent` < `smooth_spatial` < `spatial_cluster` < `checkerboard` < `random_sign`).
- *Falsification:* If high-frequency checkerboard perturbations are as damaging as smooth modes, spatial frequency is irrelevant to downstream tolerance.

---

## 6. Execution Plan & Scope

### Phase 1: Exploratory Joint Grid & Interaction Analysis
- **Architectures:** `deit_small`, `vit_base`
- **Depths:**
  - `deit_small`: Depth 5 (fragile), Depth 8 (peak fungible), Depth 10 (terminal)
  - `vit_base`: Depth 5 (fragile), Depth 7 (peak fungible), Depth 10 (terminal)
- **Dataset:** $N=100$ disjoint calibration/evaluation images from ImageNet-1k.
- **Evaluations:**
  1. Full $8 \times 5$ grid of (Token Pattern $\times$ Feature Direction) across 7 perturbation scales $s \in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]$.
  2. Two-way ANOVA / Variance decomposition:
     $$\text{Damage}(a, v) = \beta_0 + \sum_i \beta_a^{(i)} \mathbb{I}[a = a_i] + \sum_j \beta_v^{(j)} \mathbb{I}[v = v_j] + \sum_{i, j} \gamma_{ij} \mathbb{I}[a = a_i, v = v_j] + \epsilon$$
     Compute proportion of variance explained ($R^2$, $\eta^2$) by Token Pattern, Feature Direction, and Interaction.

### Phase 2: Fraction Sweep
- Vary participation fraction: $f \in \{1/N, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00\}$.
- Compare at fixed scale $s = 1.0$:
  - `coherent_fraction`: $a_i = 1/\sqrt{k}$ on $k = \lfloor f N \rfloor$ tokens.
  - `random_sign_fraction`: $a_i = \pm 1/\sqrt{k}$ on $k$ tokens.
  - `cluster_fraction`: $a_i = 1/\sqrt{k}$ on a contiguous 2D block of $k$ tokens.

### Phase 3: Confirmatory Replication
- Replicate the decisive contrast (`global_coherent` vs `random_sign` vs `single_token` across `jac_null`, `pc1`, `jac_top`) on:
  - `deit_tiny` (Depth 8)
  - `dinov2` (ViT-S/14, Depth 8)

---

## 7. Deliverables & Figures

- **Code:**
  - `patch_fungibility/joint_stream_geometry.py`
  - `scripts/run_joint_stream_geometry.py`
- **Outputs (`outputs/fungibility_joint_stream_geometry/`):**
  - `condition_results.csv`
  - `directional_curves.csv`
  - `fraction_sweep.csv`
  - `token_feature_interaction.csv`
  - `replication_summary.csv`
  - `validation_manifest.json`
- **Publication Figures (`figures/fungibility_joint_stream_geometry/`):**
  - **Figure A:** Heatmap: Token Mode $\times$ Feature Mode (Tolerance Radius $r_l(a, v)$).
  - **Figure B:** Fraction-of-Tokens Sweep: Margin Damage vs Fraction perturbed (Coherent vs Random Sign vs Cluster).
  - **Figure C:** Finite-Radius Response Curves for key combinations (`null` vs `sensitive` $\times$ `coherent` vs `random_sign`).
  - **Figure D:** Spatial Pattern Comparison: Contiguous vs Random vs Checkerboard vs Smooth mode.
  - **Figure E:** Replication Summary across `deit_small`, `vit_base`, `deit_tiny`, and `dinov2`.
- **Report:** `docs/FUNGIBILITY_JOINT_STREAM_GEOMETRY_REPORT.md` answering all questions in Section 20.
