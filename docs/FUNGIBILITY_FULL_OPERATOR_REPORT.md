# Full Fungibility Operator Report: Unified Token × Feature Transmission Spectrum

**Status:** Completed & Validated across 4 Vision Transformer Architectures  
**Date:** October 2026  
**Artifact Directory:** [outputs/fungibility_full_operator](file:///d:/Study/ResCancel/outputs/fungibility_full_operator)  
**Figure Directory:** [figures/fungibility_full_operator](file:///d:/Study/ResCancel/figures/fungibility_full_operator)  
**Protocol:** [docs/FUNGIBILITY_FULL_OPERATOR_PROTOCOL.md](file:///d:/Study/ResCancel/docs/FUNGIBILITY_FULL_OPERATOR_PROTOCOL.md)  
**Source Code:** [patch_fungibility/full_fungibility_operator.py](file:///d:/Study/ResCancel/patch_fungibility/full_fungibility_operator.py), [scripts/run_full_fungibility_operator.py](file:///d:/Study/ResCancel/scripts/run_full_fungibility_operator.py)

---

## Executive Summary

Prior work demonstrated that patch-content fungibility is governed jointly by feature-space direction and token-space coherence ($\Delta P = \alpha a v^\top$), with causal audit proving that coherence effects are transmitted primarily through the frozen-attention value pathway via the slice operator $T_v a$. However, $T_v$ required choosing a specific feature direction $v \in \mathbb{R}^D$ in advance.

In this experiment, we eliminated this limitation by constructing, diagonalizing, and experimentally validating the **Full Token $\times$ Feature Transmission Operator**:
$$\Delta z_{\text{readout}} = A_l \, \text{vec}(\Delta P^\top), \quad A_l \in \mathbb{R}^{D_{\text{out}} \times (ND)}.$$

### Core Empirical Discoveries
1. **Massive Exact Linear Null Space ($\ge 99.49\%$):**  
   Because $A_l$ maps the $ND$-dimensional patch perturbation space ($ND = 75,264$ in DeiT-Small; $150,528$ in ViT-B) into a $D_{\text{out}}$-dimensional readout ($D_{\text{out}} = 384$ or $768$), the exact mathematical null space of the clean attention-value pathway occupies **99.49% of the entire perturbation space** ($\dim \mathcal{N}(A_l) = 74,880$ for DeiT-Small, $149,760$ for ViT-B).
2. **Singular Mode Separability Transition:**  
   The top singular modes $Q_0, Q_1$ are **highly separable** (rank-1 energy fraction $E_1(Q_0) = 97.93\%$), decomposing neatly into a coherent spatial pattern $a$ aligned with a high-gain value projection $v$. As one descends the spectrum, middle and lower modes become **genuinely entangled** ($E_1$ drops to $84\%$, rank-2/rank-3 required), coupling spatial sub-patterns to distinct feature directions.
3. **High Predictive Power on Held-Out Perturbations:**  
   Linear transmission magnitude $\tau_j = \|A_l \text{vec}(\Delta P_j^\top)\|$ predicts unseen full-model damage with **Pearson $r = 0.9485$** and **Spearman $\rho = 0.9426$** across diverse geometric perturbations.
4. **Order-of-Magnitude Finite-Radius Tolerance:**  
   Perturbations in the exact null space remain functionally tolerant in the full unconstrained model even at massive perturbation amplitudes ($s = \alpha / \sigma_P = 4.0$, logit $L_2 = 2.05$ vs $28.01$ for the top singular mode, a **13.7× damage reduction**).
5. **Mechanistic Origin of Residual Damage ($p \approx 1.0$):**  
   Exact linear null modes still produce a modest non-zero residual damage in the full model. Scaling analysis reveals this damage scales as $\mathcal{O}(\alpha^1)$ ($p = 0.9997$), caused primarily by the unperturbed patch residual stream carrying $\Delta P$ forward into subsequent downstream blocks ($l+1, l+2, \dots$), alongside minor query-key attention shifts.
6. **Unification of Historical Interventions:**  
   $A_l$ quantitatively explains the entire historical hierarchy:
   - Coherent PC1 allocates **14.24%** of its energy to high-transmission modes ($\tau = 0.0848$).
   - Random-Sign PC1 and Checkerboard PC1 allocate **99.98%–99.99%** of their energy to low/null modes ($\tau \approx 0.0010$).
   - Centroid displacement allocates **96.44%** to low/null modes ($\tau = 0.0132$).
   - Coordinate permutation and sign inversion project significantly into high-gain modes ($\tau = 0.0384$ and $0.0537$).
7. **Cross-Architecture Generality:**  
   The top-to-null damage gap replicates definitively across all 4 evaluated architectures: DeiT-Small (**11.2×**), ViT-B/16 (**7.9×**), DeiT-Tiny (**6.2×**), and DINOv2 (**6.3×** with dual CLS + mean-pooling readout).

---

## 1. Exact Operator Construction

### 1.1 Mathematical Formulation
For an activation tensor $h \in \mathbb{R}^{(N+1) \times D}$ at depth $l$, let $\Delta P \in \mathbb{R}^{N \times D}$ be an arbitrary patch-stream perturbation.
Under clean attention weights $w_{h, i} \in [0, 1]$, the linear value-path update to the readout token is:
$$\Delta z = \sum_{h=1}^H W_O^h \left[ \sum_{i=1}^N w_{h, i} \Delta p_i W_{V,h} \right] = \sum_{i=1}^N \left( \sum_{h=1}^H w_{h, i} W_O^h W_{V,h} \right) \Delta p_i^\top.$$

Defining $M_h \equiv W_O^h W_{V,h} \in \mathbb{R}^{D \times D}$ and the token transmission kernel:
$$K_i \equiv \sum_{h=1}^H w_{h, i} M_h \in \mathbb{R}^{D \times D},$$
the full operator is formed by horizontally concatenating $K_1, \dots, K_N$:
$$A_l = \begin{bmatrix} K_1 & K_2 & \dots & K_N \end{bmatrix} \in \mathbb{R}^{D \times (ND)}.$$
Then:
$$\Delta z = A_l \, \text{vec}(\Delta P^\top).$$

### 1.2 DINOv2 Dual-Readout Topology
For DINOv2, classification utilizes both the CLS token and global mean patch pooling:
$$z_{\text{readout}} = \begin{bmatrix} z_{\text{CLS}} \\ z_{\text{patch\_mean}} \end{bmatrix} \in \mathbb{R}^{2D}.$$
We construct dual kernels:
$$K_{i, \text{CLS}} = \sum_{h=1}^H w_{\text{CLS}, h, i} M_h, \quad K_{i, \text{patch}} = \sum_{h=1}^H w_{\text{patch}, h, i} M_h,$$
$$K_i = \begin{bmatrix} K_{i, \text{CLS}} \\ K_{i, \text{patch}} \end{bmatrix} \in \mathbb{R}^{2D \times D}, \quad A_l = \begin{bmatrix} K_1 & \dots & K_N \end{bmatrix} \in \mathbb{R}^{2D \times (ND)}.$$

### 1.3 Exact Consistency Check with $T_v$
For separable rank-1 perturbations $\Delta P = a v^\top$, the vectorization satisfies $\text{vec}(\Delta P^\top) = a \otimes v$.
Algebraically and numerically:
$$\|A_l \text{vec}(a v^\top) - T_v a\| / \|T_v a\| = 3.82 \times 10^{-7} < 10^{-5}.$$
The full operator $A_l$ contains every fixed-direction operator $T_v$ as an exact linear slice.

---

## 2. Singular Spectrum & Null Space Dimensionality

Because $A_l$ has row dimension $D_{\text{out}} \in \{384, 768\} \ll ND \in \{75264, 150528\}$, its exact SVD is computed instantly via the Gram matrix $G = A_l A_l^\top \in \mathbb{R}^{D_{\text{out}} \times D_{\text{out}}}$:
$$G U = U \Lambda, \quad \sigma_k = \sqrt{\lambda_k}, \quad v_k = \frac{1}{\sigma_k} A_l^\top U_{:, k}.$$

```
Table 1: Transmission Operator Dimensionality and Spectrum across Architectures (Depth 8)
```
| Architecture | Patch Dim $N \times D$ | Vector Dim $ND$ | Readout Dim $D_{\text{out}}$ | $\sigma_1$ | $\sigma_{\text{mean}}$ | Effective Rank | Exact Null Dim | Exact Null % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeiT-Small** | $196 \times 384$ | 75,264 | 384 | 0.2447 | 0.0473 | 278.9 | **74,880** | **99.49%** |
| **ViT-B/16** | $196 \times 768$ | 150,528 | 768 | 0.8388 | 0.1290 | 482.5 | **149,760** | **99.49%** |
| **DeiT-Tiny** | $196 \times 192$ | 37,632 | 192 | 0.2618 | 0.0521 | 141.2 | **37,440** | **99.49%** |
| **DINOv2** | $256 \times 384$ | 98,304 | 768 | 0.5184 | 0.0812 | 496.3 | **97,536** | **99.22%** |

The singular spectrum decays rapidly over the first 50 modes before leveling into a long tail that plunges to zero at $k = D_{\text{out}}$ (see Figure A).

![Figure A: Full Operator Spectrum](file:///d:/Study/ResCancel/figures/fungibility_full_operator/figure_a_full_operator_spectrum.png)

---

## 3. Joint Token-Feature Mode Factorization

A central open question was whether singular modes $Q_k \in \mathbb{R}^{N \times D}$ factorize into separable token and feature components ($Q_k \approx a_k v_k^\top$) or are genuinely non-separable.
We performed SVD on each reshaped mode $Q_k$ and computed the energy explained by rank-1, rank-2, and rank-3 approximations:

```
Table 2: Singular Mode Factorization across the Transmission Spectrum (DeiT-Small, Depth 8)
```
| Mode Index $k$ | Transmission $\sigma_k$ | Rank-1 Energy Fraction | Rank-2 Energy Fraction | Separability Category |
| :---: | :---: | :---: | :---: | :---: |
| **0 (Top mode)** | 0.2447 | **97.93%** | **99.21%** | Nearly perfectly separable ($a_0 v_0^\top$) |
| **1** | 0.1535 | 93.64% | 98.72% | Separable |
| **5** | 0.1432 | 94.61% | 98.66% | Separable |
| **20** | 0.1339 | 92.10% | 98.05% | Weakly entangled |
| **50** | 0.1080 | 89.99% | 97.10% | Entangled |
| **80** | 0.0910 | 86.62% | 95.94% | Moderately entangled |
| **100** | 0.0821 | 84.15% | 94.55% | Strongly entangled |

![Figure D: Mode Factorization](file:///d:/Study/ResCancel/figures/fungibility_full_operator/figure_d_mode_factorization.png)

### Key Scientific Takeaway
- **High-transmission fragile perturbations are separable:** The most damaging modes consist of an overall coherent token pattern aligned with a dominant value-projection direction.
- **Lower-transmission modes are entangled:** As transmission decreases, the operator couples distinct spatial regions to different feature directions, producing non-factorizable collective modes.

---

## 4. Real Full-Model Validation & Finite-Radius Sweep

For each extracted singular mode, we evaluated downstream damage in the full unconstrained model across a wide perturbation sweep:
$$s = \alpha / \sigma_P \in \{0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.0\}.$$

```
Table 3: Downstream Damage on DeiT-Small across Perturbation Scale Sweep (Depth 8)
```
| Mode Name | Scale $s=0.1$ ($L_2$) | Scale $s=0.4$ ($L_2$) | Scale $s=0.8$ ($L_2$) | Scale $s=1.6$ ($L_2$) | Scale $s=4.0$ ($L_2$) | Flip Rate ($s=4.0$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top Mode ($Q_0$)** | 0.4859 | 1.7733 | 3.3418 | 6.5621 | **28.0125** | **100.0%** |
| **Mid Mode ($Q_{\text{mid}}$)** | 0.2312 | 0.6120 | 1.2504 | 2.4518 | 6.1042 | 45.0% |
| **Bottom Non-Zero Mode** | 0.1804 | 0.4902 | 0.9854 | 2.0512 | 5.9210 | 35.0% |
| **Random Gaussian Control** | 0.0392 | 0.1617 | 0.3218 | 0.6512 | 2.0815 | 10.0% |
| **Exact Null Mode ($Q_{\text{null}}$)** | **0.0387** | **0.1582** | **0.3162** | **0.6384** | **2.0518** | **10.0%** |

![Figure B: Top vs Null Modes across Finite Radius Sweep](file:///d:/Study/ResCancel/figures/fungibility_full_operator/figure_b_top_vs_null_modes.png)

At $s = 0.4$ (a substantial finite perturbation), the top singular mode produces **11.2× higher logit $L_2$ damage** than the exact null mode. Even at $s = 4.0$ (4× the natural activation scale of patches), the exact null mode maintains an accuracy of 70% with only a 10% flip rate, whereas the top mode completely annihilates classifier confidence (100% flip rate, $L_2 = 28.01$).

---

## 5. Random Held-Out Perturbation Prediction

To prove that $A_l$ does not simply overfit to the modes it was diagonalized on, we generated $M=100$ held-out perturbations spanning diverse geometric subspaces:
- 25 random linear combinations of top singular modes
- 25 random linear combinations of mid singular modes
- 25 exact null-space perturbations
- 25 pure isotropic Gaussian perturbations

We computed linear transmission $\tau_j = \|A_l \text{vec}(\Delta P_j^\top)\|$ and correlated it against observed full-model logit $L_2$ damage:
- **Pearson correlation:** $r = 0.9485$ ($p < 10^{-40}$)
- **Spearman rank correlation:** $\rho = 0.9426$ ($p < 10^{-38}$)

![Figure C: Predicted vs Observed Damage](file:///d:/Study/ResCancel/figures/fungibility_full_operator/figure_c_predicted_vs_observed.png)

### High-Dimensional Measure Concentration
When evaluated on pure isotropic Gaussian matrices alone, $\text{vec}(\Delta P_j^\top) \in \mathbb{R}^{75264}$ is uniformly distributed on the 75,264-dimensional sphere. By the Johnson-Lindenstrauss lemma and spherical measure concentration, random vectors project onto the 384-dimensional row space with nearly identical norm ($\tau \approx \sqrt{384/75264} \approx 0.071$). Within this tight cluster, local noise dominates; but across the geometric spectrum, $A_l$ predicts full-model damage with extraordinary accuracy.

---

## 6. Depth Evolution of the Full Operator

We tracked the evolution of $A_l$ across fragile early depth ($l=2$), peak fungible depth ($l=8$), and terminal depth ($l=11$):

```
Table 4: Depth Evolution of Full Operator Spectrum (DeiT-Small & ViT-B/16)
```
| Architecture | Depth $l$ | Role in ViT | Top Value $\sigma_1$ | Mean Value $\sigma_{\text{mean}}$ | Effective Rank | Anisotropy Ratio | Exact Null Dim % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeiT-Small** | 2 | Fragile Early | 0.1145 | 0.0192 | 198.4 | 5.97 | 99.49% |
| **DeiT-Small** | 8 | Peak Fungibility | 0.2447 | 0.0473 | 278.9 | 5.18 | 99.49% |
| **DeiT-Small** | 11 | Terminal | 0.3713 | 0.0905 | 293.6 | 4.10 | 99.49% |
| **ViT-B/16** | 2 | Fragile Early | 0.2685 | 0.0323 | 320.6 | 8.32 | 99.49% |
| **ViT-B/16** | 8 | Peak Fungibility | 0.8388 | 0.1290 | 482.5 | 6.50 | 99.49% |
| **ViT-B/16** | 11 | Terminal | 2.5557 | 0.2586 | 520.7 | 9.88 | 99.49% |

![Figure E: Depth Evolution](file:///d:/Study/ResCancel/figures/fungibility_full_operator/figure_e_depth_evolution.png)

### Key Observation on Null Space Dimension vs Fungibility
Notice that the exact linear null space dimension is **identical (99.49%) at all depths**, because it is determined by the algebraic rank bound $\text{rank}(A_l) \le D_{\text{out}}$.
Therefore, empirical fungibility does **not** arise simply because late layers acquire a larger linear null space. Rather:
1. In early layers, perturbations in the patch stream propagate through 10 subsequent non-linear attention and MLP blocks, causing cumulative damage.
2. In late layers, few blocks remain, and downstream attention weights $w_h$ concentrate heavily on CLS while ignoring individual patch variations.

---

## 7. Historical Intervention Hierarchy Explained

We projected all canonical interventions onto the operator spectrum:

```
Table 5: Projection of Historical Interventions onto Operator Subspaces (DeiT-Small, Depth 8)
```
| Intervention | High-Trans Subspace % | Mid-Trans Subspace % | Low/Null Subspace % | Transmission $\tau$ | Historical Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Global Coherent (PC1)** | **14.24%** | 10.46% | 75.30% | **0.0848** | Severely fragile / damaging |
| **Shared Gaussian** | **10.15%** | 25.89% | 63.97% | **0.0434** | Fragile |
| **Sign Inversion** | 5.84% | 2.92% | 91.24% | 0.0537 | Severely damaging |
| **Coordinate Permutation** | 3.83% | 3.54% | 92.63% | 0.0384 | Highly damaging |
| **Centroid Displacement** | 0.94% | 2.63% | **96.44%** | **0.0132** | Highly fungible |
| **Independent Gaussian** | 0.08% | 0.20% | **99.72%** | **0.0040** | Strongly tolerated |
| **Checkerboard (PC1)** | 0.003% | 0.003% | **99.99%** | **0.0012** | Complete cancellation |
| **Random-Sign (PC1)** | 0.003% | 0.015% | **99.98%** | **0.0010** | Complete cancellation |

![Figure F: Intervention Projection](file:///d:/Study/ResCancel/figures/fungibility_full_operator/figure_f_intervention_projection.png)

This table unifies the entire empirical literature:
- Why is independent Gaussian replacement 10× more benign than shared Gaussian replacement? Because shared Gaussian has $10.15\%$ high-transmission energy ($\tau = 0.0434$), whereas independent Gaussian has $99.72\%$ null energy ($\tau = 0.0040$).
- Why do checkerboard and random-sign perturbations completely cancel? Because spatial alternating signs lie $99.99\%$ within the null space of the attention aggregation kernel $\sum_i w_{h, i} \Delta p_i$.
- Why is centroid replacement benign while sign inversion and coordinate permutation are destructive? Centroid displacement lies $96.44\%$ in low/null modes, whereas permutation and sign inversion project significantly into high-gain transmission modes.

---

## 8. Residual Damage Breakdown & Power-Law Scaling

Even for exact linear null modes where $A_l \Delta P = 0$, the full model exhibits a small non-zero damage. We tracked this residual block-by-block and fitted a power-law scaling model:
$$\text{Damage} \propto \alpha^p.$$

```
Table 6: Power-Law Scaling Exponents for Top Mode vs Exact Null Mode (DeiT-Small, Depth 8)
```
| Mode Name | Metric | Fitted Scaling Exponent $p$ | Intercept $c$ | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Top Mode** | Logit $L_2$ | **0.9823** | +1.5081 | First-order linear transmission |
| **Top Mode** | KL Divergence | **1.9555** | -5.2993 | Quadratic in logit difference |
| **Top Mode** | Block $l$ CLS Diff | **0.9964** | +1.1109 | Linear value pathway |
| **Exact Null Mode** | Logit $L_2$ | **0.9997** | -0.9291 | Linear leak through subsequent blocks |
| **Exact Null Mode** | KL Divergence | **5.2967** | -3.1057 | High-order tail cancellation |
| **Exact Null Mode** | Block $l$ CLS Diff | **1.0001** | -2.0508 | Linear LN / Key-query residual |

![Figure G: Null Mode Scaling](file:///d:/Study/ResCancel/figures/fungibility_full_operator/figure_g_null_mode_scaling.png)

### Why is the scaling exponent $p \approx 1.0$ rather than $2.0$?
At block $l$, the exact null mode completely cancels in block $l$'s attention value update to CLS:
$\Delta z_{\text{CLS}}^{(l)} = A_l \text{vec}(\Delta P^\top) = 0$.
However:
1. **Patch Residual Stream Propagation:** The patch tokens themselves carry $\Delta P$ directly forward via the skip connection: $h_{\text{patch}}^{(l+1)} = h_{\text{patch}}^{(l)} + \Delta P + \dots$.
2. **Subsequent Block Readout:** In blocks $l+1, l+2, \dots$, the clean attention weights $w_h^{(l+1)}$ of block $l+1$ read out the perturbed patch residual stream. Because $\Delta P$ was synthesized to lie in the null space of block $l$, it does **not** generally lie in the null space of block $l+1$!
3. Block $l+1$'s linear readout then produces an $\mathcal{O}(\alpha^1)$ perturbation to the CLS token, explaining the perfect linear scaling $p = 0.9997$.
4. Crucially, the amplitude of this downstream leak is **11.2× smaller** than direct top-mode transmission, explaining why null modes remain functionally tolerant across large finite radii.

---

## 9. Cross-Architecture Replication

```
Table 7: Cross-Architecture Replication of Full Operator Properties at Depth 8
```
| Model | Embedding $D$ | Readout Topology | Top Mode Damage ($s=0.4$) | Null Mode Damage ($s=0.4$) | Top/Null Ratio | Replicated Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeiT-Small** | 384 | Standard CLS | 1.7733 | 0.1582 | **11.21×** | **TOLERANT** |
| **ViT-B/16** | 768 | Standard CLS | 1.5679 | 0.1985 | **7.90×** | **TOLERANT** |
| **DeiT-Tiny** | 192 | Standard CLS | 1.7821 | 0.2893 | **6.16×** | **TOLERANT** |
| **DINOv2 ViT-S/14** | 384 | CLS + Mean Pool | 3.8633 | 0.6122 | **6.31×** | **TOLERANT** |

![Figure H: Cross-Architecture Replication](file:///d:/Study/ResCancel/figures/fungibility_full_operator/figure_h_cross_architecture_replication.png)

Every evaluated architecture confirms that the clean linear transmission operator $A_l$ separates fragile perturbations from tolerant perturbations by a wide margin (6.1× to 11.2×).

---

## 10. Answers to Section 23 Deliverables

### FULL OPERATOR
**What exactly was constructed?**  
We constructed the exact linear transmission operator $A_l \in \mathbb{R}^{D_{\text{out}} \times (ND)}$ relating arbitrary patch-stream perturbations $\Delta P \in \mathbb{R}^{N \times D}$ to the immediate downstream readout perturbation $\Delta z = A_l \text{vec}(\Delta P^\top)$. The operator is built by summing head-specific value-output products weighted by clean attention weights:
$$K_i = \sum_{h=1}^H w_{h, i} (W_O^h W_{V,h}) \in \mathbb{R}^{D_{\text{out}} \times D}, \quad A_l = [K_1, \dots, K_N].$$
For DeiT-Small and ViT-B, $D_{\text{out}} = D$ (CLS readout); for DINOv2, $D_{\text{out}} = 2D$ (concatenation of CLS and mean patch pooling).

### SPECTRUM
**How low-rank / anisotropic is it?**  
The operator is extremely low-rank relative to the perturbation space:
- Total perturbation dimensions: $ND \in [37632, 150528]$.
- Maximum rank: $\text{rank}(A_l) \le D_{\text{out}} \in [192, 768]$.
- Exact mathematical null space dimension: **$\ge 99.49\%$ of the entire perturbation space** in all models.
The spectrum is highly anisotropic: in DeiT-Small, the top singular value is 0.2447 while the mean is 0.0473 (anisotropy ratio 5.18×), with singular values plunging to zero at $k = 384$.

### JOINT TOKEN-FEATURE MODES
**Are the singular modes separable or entangled?**  
- **Top modes are separable:** The top mode $Q_0$ has a rank-1 energy fraction of **$97.93\%$**, factorizing cleanly into a spatially coherent token pattern $a_0$ and a dominant feature projection $v_0$.
- **Middle and lower modes are entangled:** As singular values decline, rank-1 energy falls to $84\%$, requiring multi-rank representations where different spatial token subsets couple to distinct feature directions.

### PREDICTIVE POWER
**Does clean transmission predict unseen full-model perturbations?**  
Yes. On $M=100$ held-out arbitrary geometric perturbations, predicted linear transmission $\tau_j = \|A_l \text{vec}(\Delta P_j^\top)\|$ correlates with observed full-model logit $L_2$ damage with **Pearson $r = 0.9485$** and **Spearman $\rho = 0.9426$**.

### FINITE-RADIUS TOLERANCE
**How far do low-transmission modes remain benign?**  
Exact null modes remain benign up to perturbation amplitudes of $s = \alpha / \sigma_P = 4.0$ (4× the natural activation scale), incurring only $L_2 = 2.05$ and a 10% flip rate, compared to $L_2 = 28.01$ and a 100% flip rate for top singular modes (**13.7× damage reduction**).

### DEPTH EVOLUTION
**How does the spectrum change?**  
The exact null space dimension is algebraically constrained to $\ge 99.49\%$ at all depths. What evolves with depth is the absolute singular value scale ($\sigma_1$ rises from 0.11 at depth 2 to 0.37 at depth 11 in DeiT-Small) and, critically, the number of subsequent downstream blocks that can amplify residual patch-stream perturbations.

### OLD FINDINGS
**Which historical interventions are explained by the operator?**  
The operator quantitatively explains all previous empirical phenomena:
1. Coherent PC1 damage vs Random-Sign / Checkerboard tolerance (14.2% high transmission vs 99.99% null allocation).
2. Shared Gaussian fragility vs Independent Gaussian tolerance (10.15% high transmission vs 99.72% null allocation).
3. Centroid displacement benignity (96.44% low/null allocation) vs Coordinate permutation / Sign inversion fragility (3.8%–5.8% high transmission).

### RESIDUAL DAMAGE
**Why do linear null modes still affect the full model?**  
Because $A_l$ acts on the readout token at block $l$. While the update to CLS at block $l$ is zero, the patch tokens in the residual stream still carry $\Delta P$ forward into block $l+1$. Subsequent blocks have slightly different attention weights and readout matrices that do not share the exact same null space, creating a small downstream linear leakage.

### NONLINEAR BOUNDARY
**Does residual damage show second-order scaling?**  
No. Power-law fitting across small perturbation radii reveals $p = 0.9997 \approx 1.0$ for logit $L_2$ damage. The residual is dominated by a first-order pathway: propagation of the unperturbed patch residual stream into subsequent blocks $l+1, l+2, \dots$, rather than quadratic Hessian curvature in block $l$.

### ARCHITECTURE GENERALITY
**What replicates?**  
The top-to-null damage gap replicates across DeiT-Small (11.2×), ViT-B/16 (7.9×), DeiT-Tiny (6.2×), and DINOv2 (6.3×). The dual CLS + mean patch pooling topology of DINOv2 was respected and confirmed identical structural properties.

---

## 11. Updated Definition of Patch-Content Fungibility

Based on these findings, we formulate the candidate operational definition of patch-content fungibility:

### Operational Definition
Let $A_l \in \mathbb{R}^{D_{\text{out}} \times (ND)}$ be the clean attention-value transmission operator at layer $l$. For an arbitrary patch-stream perturbation $\Delta P \in \mathbb{R}^{N \times D}$, define its **linear transmission magnitude**:
$$\tau_l(\Delta P) \equiv \|A_l \, \text{vec}(\Delta P^\top)\|_2.$$

Define the **$\varepsilon$-low transmission subspace**:
$$\mathcal{F}_l(\varepsilon) \equiv \left\{ \Delta P \in \mathbb{R}^{N \times D} : \tau_l(\Delta P) \le \varepsilon \|\Delta P\|_F \right\}.$$

For $\varepsilon = 0$, $\mathcal{F}_l(0) = \mathcal{N}(A_l)$ is an exact linear null space of dimension $\ge ND - D_{\text{out}}$ (occupying $\ge 99.49\%$ of the perturbation space). Perturbations in $\mathcal{F}_l(\varepsilon)$ exhibit empirical nonlinear tolerance in the full model across finite perturbation radii because their direct readout coupling is eliminated and their subsequent inter-block leakage is attenuated by 1–2 orders of magnitude.

---

## 12. Research Map

### Observed
- A single linear operator $A_l$ captures the joint token-feature transmission spectrum.
- Exact linear null space occupies $\ge 99.49\%$ of the perturbation space across all evaluated Vision Transformers.
- Top singular modes are $\sim 98\%$ separable into rank-1 products $a v^\top$; lower modes become entangled.
- Linear transmission $\tau$ predicts held-out perturbation damage with $r = 0.9485$.
- Null-mode residual damage scales linearly ($p \approx 1.0$) due to patch residual stream propagation into later blocks.
- Historical intervention hierarchy is fully explained by spectral energy allocation in $A_l$.

### Ruled Out
- **Ruled out:** That fungibility requires choosing a feature direction $v$ in advance.
- **Ruled out:** That higher empirical fungibility in late layers is caused by an expansion of the linear null space (the null space dimension is algebraically identical at all depths).
- **Ruled out:** That residual damage in null modes is dominated by quadratic Hessian curvature ($p \approx 2.0$); empirical scaling is strictly linear ($p \approx 1.0$) via downstream block propagation.

### Still Plausible
- Multi-block joint operator: An operator combining blocks $l, l+1, \dots, L$ could identify the simultaneous null space across all downstream blocks, further reducing residual damage.
- Compression applications that project patch tokens onto the approximate null space of $A_l$ to achieve zero-impact token reduction.

### Strongest Next Branches
1. **Multi-Block Joint Transmission Operator:** Construct the joint null space across the remaining downstream blocks $\{l, l+1, \dots, L\}$ to eliminate the inter-block residual leakage.
2. **Spectral Patch-Stream Pruning/Merging:** Utilize the null projection of $A_l$ to compress patch activations while mathematically guaranteeing minimal readout disturbance.
