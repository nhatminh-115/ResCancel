# Experimental Protocol: Causal Audit of the Attention Aggregation Cancellation Mechanism

**Repository**: [https://github.com/nhatminh-115/Patch-Content-Fungibility](https://github.com/nhatminh-115/Patch-Content-Fungibility)  
**Experiment Name**: CAUSAL DECOMPOSITION OF THE COHERENCE EFFECT  
**Status**: ACTIVE PROTOCOL  
**Created**: 2026-10-04  
**Target Output Namespace**: `outputs/fungibility_attention_causal_audit/`  
**Target Report**: `docs/FUNGIBILITY_ATTENTION_CAUSAL_AUDIT_REPORT.md`  
**Figures**: `figures/fungibility_attention_causal_audit/`  

---

## 1. Executive Summary & Scientific Motivation

The preceding joint-stream geometry experiment established a decisive empirical finding:
$$\Delta P = \alpha \, a v^\top \in \mathbb{R}^{N \times D}$$
At identical total Frobenius perturbation energy $\|\Delta P\|_F = \alpha$, **spatially coherent perturbations along sensitive feature directions inflict up to $223\times$ higher downstream damage than incoherent (random-sign or checkerboard) perturbations**.

A candidate mechanism was proposed:
$$\Delta z_{\text{readout}} = \alpha \left( \sum_{i=1}^N w_{\text{readout}, i} a_i \right) (v^\top W_V)$$
This formula suggests that when $a_i$ is random-sign or checkerboard, broad attention distributions linearly cancel the sum $\sum_i w_i a_i \approx 0$, while coherent shifts accumulate constructively.

**However, this mechanism has NOT yet been causally established.**  
The simple linear derivation implicitly assumes that attention weights $w_i$ remain perfectly fixed. In an actual Vision Transformer, perturbing the patch stream also alters:
1. Keys ($K$) and Queries ($Q$),
2. Scaled dot-product attention logits ($Q K^\top / \sqrt{d}$),
3. Softmax routing weights ($A$),
4. Residual stream additions,
5. Multi-Layer Perceptron (MLP) non-linearities,
6. Multi-block compounding dynamics across downstream layers.

Therefore, this experiment executes a **rigorous causal path decomposition** to determine what actually produces the coherent-vs-incoherent damage gap.

---

## 2. Competing Hypotheses

| Hypothesis | Mechanism | Empirical Prediction |
| :--- | :--- | :--- |
| **A. Value-Path Cancellation** | Attention weights are approximately fixed. Perturbed values aggregate as $\sum_i w_i a_i V(v)$, canceling under alternating signs. | Frozen clean attention weights ($A_{\text{clean}} V_{\text{pert}}$) reproduce $\ge 80\%$ of the coherence damage gap. |
| **B. Attention-Weight Reconfiguration** | Perturbations alter $Q$ and $K$, warping softmax attention distributions. The coherence gap is caused by routing breakdown. | Attention rerouting alone ($A_{\text{pert}} V_{\text{clean}}$) accounts for the majority of the coherence gap; $A$ moves drastically under coherent perturbations. |
| **C. Multi-Block Compounding** | The first downstream block provides minimal cancellation; the coherence gap emerges gradually across subsequent blocks. | The immediate disturbance after block $l+1$ is similar for coherent and incoherent modes, diverging only in later blocks. |
| **D. Residual / MLP Pathway Effect** | Attention alone does not cancel perturbations; the residual stream preserves them, but subsequent LayerNorm/MLP blocks suppress them. | Perturbation norm remains intact through the attention branch and is suppressed specifically inside the MLP branch. |
| **E. Architecture-Specific Topology** | Supervised CLS-readout ViTs operate via value cancellation; self-supervised models with dense pooling behave differently. | DeiT and ViT-B exhibit pure value cancellation, while DINOv2 exhibits attention reconfiguration or residual suppression. |

---

## 3. Models, Canonical Depths, and Readout Topologies

To avoid confounds and maintain continuity with prior experiments:
1. **Exploratory Models**:
   - **DeiT-Small** (`deit_small_patch16_224`, $N=196, D=384, H=6$ heads):
     Intervention at peak-fungibility depth $l=8$ (first downstream block is block index 8, depth 9).
     Readout: `[CLS]` token (index 0).
   - **ViT-B/16 AugReg** (`vit_base_patch16_224.augreg_in1k`, $N=196, D=768, H=12$ heads):
     Intervention at peak-fungibility depth $l=7$ (first downstream block is block index 7, depth 8).
     Readout: `[CLS]` token (index 0).
2. **Confirmatory Replication Models**:
   - **DeiT-Tiny** (`deit_tiny_patch16_224`, $N=196, D=192, H=3$ heads): Depth 8.
   - **DINOv2 ViT-S/14** (`dinov2_vits14_lc`, $N=256, D=384, H=6$ heads): Depth 8.
     Readout: Dual topology respecting model architecture: $\text{concat}(z_{\text{CLS}}, \frac{1}{N} \sum_i z_i)$.

---

## 4. Controlled Causal Forward Conditions

For the first downstream block $B_{l}$ taking hidden state $h_l \to h_{l+1}$:
Let $h_{l, \text{clean}}$ be the clean activation and $h_{l, \text{pert}} = h_{l, \text{clean}} + \Delta P$ (patch tokens only).
From $h_{l, \text{clean}}$, we obtain clean projections: $Q_c, K_c, V_c$ and clean attention matrix $A_c = \text{softmax}(Q_c K_c^\top / \sqrt{d})$.  
From $h_{l, \text{pert}}$, we obtain perturbed projections: $Q_p, K_p, V_p$ and perturbed attention matrix $A_p = \text{softmax}(Q_p K_p^\top / \sqrt{d})$.

### Condition 0: Clean Baseline
Standard unperturbed forward pass.

### Condition 1: Full Perturbation
Standard perturbation. $Q_p, K_p, V_p$ are all active; attention matrix $A_p$ is used; residual is $h_{l, \text{pert}}$; subsequent blocks proceed naturally.

### Condition 2: Frozen Attention Weights (Value-Only Pathway)
- Attention context is forced to: $\text{ctx} = A_c V_p$.
- Evaluated with both:
  - `v_only_clean_res`: residual stream is $h_{l, \text{clean}}$ (isolating attention output).
  - `v_only_pert_res`: residual stream is $h_{l, \text{pert}}$ (standard residual).
- Subsequent blocks $l+1 \dots L$ run naturally to measure downstream logit damage.

### Condition 3: Attention-Rerouting Only
- Attention context is forced to: $\text{ctx} = A_p V_c$.
- Evaluated with both `reroute_clean_res` and `reroute_pert_res`.
- Directly isolates damage caused solely by shifts in routing weights $A$.

### Condition 4: Immediate First-Block Output
Evaluate the immediate state change at $h_{l+1}$:
- Readout token disturbance: $\|\Delta z_{l+1, \text{readout}}\|_2$.
- Patch stream Frobenius disturbance: $\|\Delta P_{l+1}\|_F$.
- Cosine similarity: $\cos(h_{l+1, \text{clean}}, h_{l+1, \text{pert}})$.

### Condition 5: Multi-Block Downstream Trace
Record disturbance norm, readout shift, and attention variation at every subsequent block $b \in \{l+1, l+2, \dots, L\}$.

---

## 5. Sub-Projection Decompositions ($Q$ vs $K$ vs $V$)

In block $l$, isolate the 6 projection combinations:
1. $V$ only: $(Q_c, K_c, V_p)$
2. $K$ only: $(Q_c, K_p, V_c)$
3. $Q$ only: $(Q_p, K_c, V_c)$
4. $K+V$: $(Q_c, K_p, V_p)$
5. $Q+K$: $(Q_p, K_p, V_c)$
6. $Q+K+V$: $(Q_p, K_p, V_p)$

---

## 6. Head-Wise Cancellation Scalar & Linear Prediction

For each attention head $h \in \{1, \dots, H\}$:
$$\Gamma_h(a) = \sum_{i=1}^N w_{h, \text{readout}, i} a_i$$
We evaluate:
1. **Head-Level Linear Prediction**:
   $$\Delta z_h^{\text{pred}} = \alpha \Gamma_h(a) (v^\top W_{V, h})$$
   Compare against measured head output change $\Delta z_h^{\text{obs}} = (A_c V_p - A_c V_c)_{h, \text{readout}, :}$.
   Report cosine similarity, norm ratio, Pearson $r$, and $R^2$.
2. **Aggregate Scalar Correlation**:
   Test whether $\bar{\Gamma}(a) = \frac{1}{H} \sum_h |\Gamma_h(a)|$ correlates with downstream model damage.

---

## 7. Spatial Smoothness Analysis

To empirically test whether attention distributions act as spatial low-pass filters:
1. **2D Fourier Transform of Attention Maps**:
   For clean readout attention weights $w \in \mathbb{R}^{H_p \times W_p}$, compute 2D FFT:
   $$\hat{w}(k_x, k_y) = \sum_{u, v} w(u, v) e^{-2\pi i (u k_x / H_p + v k_y / W_p)}$$
2. **Low-Frequency Power Ratio**:
   $$\text{LFPR} = \frac{\sum_{k_x \le 2, k_y \le 2} |\hat{w}(k_x, k_y)|^2}{\sum_{k_x, k_y} |\hat{w}(k_x, k_y)|^2}$$
3. **Total Variation (TV) & Neighbor Correlation**:
   Compute adjacent pixel correlation across the $14 \times 14$ grid.
4. **2D Fourier Spatial Response Map $H_l(k_x, k_y, v)$**:
   Evaluate downstream damage across 2D cosine spatial frequency modes $k_x, k_y \in [0, 4]$.

---

## 8. Quantitative Causal Attribution & Decision Criteria

To quantify causal contribution:
$$\text{Gap}_{\text{full}} = \text{Damage}(\text{coherent}, \text{full}) - \text{Damage}(\text{random\_sign}, \text{full})$$
$$\text{Gap}_{V} = \text{Damage}(\text{coherent}, V\text{-only}) - \text{Damage}(\text{random\_sign}, V\text{-only})$$
$$\text{Gap}_{\text{route}} = \text{Damage}(\text{coherent}, \text{reroute}) - \text{Damage}(\text{random\_sign}, \text{reroute})$$

- **Value-Aggregation Decisive Criterion**:
  If $\frac{\text{Gap}_V}{\text{Gap}_{\text{full}}} \ge 0.70$ and $\frac{\text{Gap}_{\text{route}}}{\text{Gap}_{\text{full}}} \le 0.30$, Value-Path Cancellation is causally established as the primary mechanism.
- **Attention-Rerouting Criterion**:
  If $\frac{\text{Gap}_{\text{route}}}{\text{Gap}_{\text{full}}} \ge 0.50$, Attention Rerouting is a necessary co-factor.
- **Compounding Criterion**:
  If $\text{ImmediateGap}_{l+1} \ll \text{FinalGap}_L$, multi-block amplification is essential.

---

## 9. Machine-Readable Outputs & Deliverables

All outputs will be saved in `outputs/fungibility_attention_causal_audit/`:
1. `causal_conditions.csv`
2. `qkv_decomposition.csv`
3. `headwise_gamma.csv`
4. `headwise_prediction.csv`
5. `attention_shift.csv`
6. `blockwise_propagation.csv`
7. `residual_path.csv`
8. `replication_summary.csv`
9. `validation_manifest.json`

Figures in `figures/fungibility_attention_causal_audit/`:
1. `figure_a_causal_path_decomposition.png`
2. `figure_b_headwise_gamma_prediction.png`
3. `figure_c_blockwise_propagation.png`
4. `figure_d_attention_shift.png`
5. `figure_e_replication.png`
6. `figure_f_spatial_frequency_response.png`

Comprehensive Report:
`docs/FUNGIBILITY_ATTENTION_CAUSAL_AUDIT_REPORT.md`
