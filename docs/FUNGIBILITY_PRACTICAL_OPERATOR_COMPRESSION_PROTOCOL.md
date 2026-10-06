# Protocol: Practical Operator Compression
# Early Intervention, Fast Solvers, and the Accuracy-Latency Frontier

**Status:** Pre-registered, Frozen Protocol  
**Date:** 2026-10-06  
**Repository:** `https://github.com/nhatminh-115/Patch-Content-Fungibility`  
**Working Directory:** `d:\Study\ResCancel`  
**Authors:** Patch Content Fungibility Research Program  

---

## 1. Executive Summary & Problem Formulation

The strict validation audit of Amortized Operator-Aware Token Compression established five core facts:
1. The prior "Oracle-above-clean" observation was an artifact of evaluated sample subsets (a high-accuracy 200-image subset vs. the 1,000-image canonical evaluation set); on identical inputs, Oracle Rank-32 is strictly $\le$ Clean.
2. The predicted Rank-32 subspace genuinely learns image-conditioned downstream functional geometry from intermediate activations ($R^2$ and overlap correlation with error reduction).
3. Predicted subspace overlap correlates directly with reduced downstream distortion ($\|J E\|_2$ and logit $L_2$).
4. Amortized inference eliminates the need for test-time VJP passes, running $100\times\text{--}140\times$ faster than Oracle VJP.
5. **Critically, current amortized inference ($24\text{--}32\text{ ms}$) is NOT faster than ordinary clean uncompressed inference ($3.5\text{--}6.7\text{ ms}$).**

Two distinct bottlenecks cause this deficit:
- **Bottleneck A (Implementation Overhead):** Carrier optimization ($\sim 11\text{ ms}$) and feature-similarity grouping ($\sim 6\text{--}8\text{ ms}$) alone consume $\sim 17\text{--}19\text{ ms}$.
- **Bottleneck B (Structural Depth Bottleneck):** Compression was applied at late layers ($l=8$ of 12), where the prefix forward pass already consumed $\ge 67\%$ of clean latency, leaving insufficient suffix computation to yield an end-to-end win.

### Core Scientific & Practical Question
Can downstream functional geometry enable **earlier token compression**, where sequence shortening saves enough remaining transformer computation to outweigh the compression overhead?
Specifically:
$$\text{End-to-End Amortized Latency } T_{\text{amortized}}(l) < T_{\text{clean}}$$
while maintaining a superior accuracy-latency Pareto frontier over matched-budget baselines (ToMe, Attention Pruning, Group Mean).

---

## 2. Predefined Architectural Scope & Intervention Depths

Four vision transformer backbones spanning diverse capacities and pretraining paradigms are evaluated:
1. **DeiT-Tiny** (`deit_tiny_patch16_224`): $N=196, D=192$, 12 blocks, supervised.
2. **DeiT-Small** (`deit_small_patch16_224`): $N=196, D=384$, 12 blocks, supervised.
3. **ViT-Base** (`vit_base_patch16_224.augreg_in1k`): $N=196, D=768$, 12 blocks, AugReg supervised.
4. **DINOv2-S/14** (`dinov2_vits14_lc`): $N=256, D=384$, 12 blocks, self-supervised with linear head.

To avoid cherry-picking, we predefine 3 representative intervention depths per architecture:
- **Early-Mid Depth:** $l = 3$ (DeiT-Tiny, DeiT-Small, ViT-Base, DINOv2)
- **Mid Depth:** $l = 6$ (DeiT-Tiny, DeiT-Small, DINOv2) / $l = 5$ (ViT-Base)
- **Audited Late Depth:** $l = 8$ (DeiT-Tiny, DeiT-Small, DINOv2) / $l = 7$ (ViT-Base)

---

## 3. Fundamental Latency Lower Bound Formulation

For an intervention at depth $l$, let:
- $T_{\text{prefix}}(l)$: Exact latency of blocks $1 \dots l$ starting from input image $x$.
- $T_{\text{suffix, clean}}(l)$: Exact latency of blocks $l+1 \dots 12$ plus readout head on full sequence $N+1$.
- $T_{\text{suffix, comp}}(l, B)$: Latency of blocks $l+1 \dots 12$ on compressed sequence $B+1$.

The **idealized theoretical latency lower bound** assuming zero overhead for grouping, subspace prediction, and carrier solving is:
$$T_{\text{ideal}}(l, B) = T_{\text{prefix}}(l) + T_{\text{suffix, comp}}(l, B)$$

**Feasibility Criterion:** Intervention at depth $l$ can only achieve practical speedup if:
$$T_{\text{ideal}}(l, B) < T_{\text{clean}}$$
$$\Delta_{\text{budget}}(l, B) = T_{\text{clean}} - T_{\text{ideal}}(l, B) > 0$$
where $\Delta_{\text{budget}}(l, B)$ represents the maximal allowable latency for the entire compression mechanism (grouping + prediction + carrier solve + token collapse).

---

## 4. Decomposing & Optimizing the Carrier Solve

The original `solve_amortized_carrier` required $11\text{ ms}$. We decompose the hot path into 10 granular stages:
1. Feature gathering: $P[S_j > 0]$
2. Group kernel construction: $C_{\text{mean}}$
3. $K_j$ accumulation: $\sum_{i \in G_j} V_i^\top$
4. Residual construction: $r_{\text{mean}} = J_{\text{proj}} E_{\text{mean}}$
5. Low-rank projection: $\tilde{K}_j = \frac{1}{\sqrt{m_j}} K_j$
6. Matrix formation: $\Sigma = \sum_j \tilde{K}_j \tilde{K}_j^\top$
7. Regularization: $\Sigma + \lambda I_r$
8. Cholesky factor / solve: $M \alpha = r_{\text{mean}}$
9. Carrier update: $\Delta C_j = \frac{1}{m_j} K_j^\top \alpha$
10. Python loops & CPU-GPU synchronizations (`.item()` calls)

### Vectorized Exact Carrier Solver
We vectorize all steps into batched GEMM / tensor operations:
- $C_{\text{mean}} = (S^\top P) \oslash m$ where $m = S^\top \mathbf{1}_N$
- $K = \text{einsum}('rnd,nj \to rjd', V_{\text{blocks}}, S)$
- $\Sigma = \tilde{K} \tilde{K}^\top$ where $\tilde{K} \in \mathbb{R}^{r \times (B \cdot D)}$
- $M = \Sigma + \lambda I_r$, solved via Cholesky decomposition in $\mathbb{R}^{r \times r}$ ($r \le 32$)
- $\Delta C = \text{einsum}('r, rbd \to bd', \alpha, K) \oslash m$
- Strict requirement: 0 Python loops over tokens/groups, 0 host `.item()` syncs, 100% GPU resident.

### Fast Approximate Solvers
To test whether sub-millisecond solving is achievable with minimal functional loss, we evaluate:
1. **Vectorized Exact:** Exact closed-form solution to the low-rank Tikhonov problem.
2. **Diagonal Approximation:** Approximates $\Sigma$ with its diagonal: $\alpha_{\text{diag}} = r_{\text{mean}} \oslash (\text{diag}(\Sigma) + \lambda \mathbf{1})$. Cost: $\mathcal{O}(r)$.
3. **One-Step Preconditioned Step:** Uses empirical calibration average $\bar{\Sigma}^{-1}$ as a constant preconditioner: $\alpha_{\text{pre}} = \bar{M}^{-1} r_{\text{mean}}$.
4. **Group Mean:** Pure unadjusted centroid $\Delta C = 0$.

---

## 5. Cheap & Vectorized Grouping Strategies

Grouping previously took $6\text{--}8\text{ ms}$ due to iterative sequential medoid selection on CPU/GPU. We compare:
- **Strategy A: Sequential Medoid Feature Similarity** (Original baseline)
- **Strategy B: Vectorized GPU Similarity** (Fast batched cosine distance + argmin)
- **Strategy C: Fixed Spatial Grid Grouping** (Deterministic 2D spatial patches: e.g., $2 \times 2$ or $4 \times 4$ pooling. Precomputed constant tensor $S$, **0.00 ms** inference cost).
- **Strategy D: 1D Regular Strided Grouping** (Uniform sequence subsampling).

---

## 6. Compression Budgets & Token Retention

We evaluate three canonical retention budgets:
- **Budget 50%:** $B = 98$ tokens (DeiT/ViT-B), $B = 128$ tokens (DINOv2)
- **Budget 25%:** $B = 49$ tokens (DeiT/ViT-B), $B = 64$ tokens (DINOv2)
- **Budget ~16%:** $B = 32$ tokens (DeiT/ViT-B), $B = 42$ tokens (DINOv2)

---

## 7. Comparative Baselines & Causal Controls

All methods are compared under identical image subsets, token budgets, and evaluation seeds:
1. **Clean Baseline:** Full $N+1$ token uncompressed model.
2. **Attention Pruning (EViT style):** Prunes tokens with smallest [CLS]-attention weights at depth $l$.
3. **Token Merging (ToMe):** Bipartite soft matching at depth $l$.
4. **Group Mean (Spatial & Similarity):** Mean centroid $C_{\text{mean}}$ without carrier correction.
5. **Oracle Rank-32 Operator:** Exact top-32 VJP SVD modes applied via exact solver.
6. **Predicted Rank-32 Operator:** Subspace predicted by `FactorizedModePredictor` trained at depth $l$.

---

## 8. Success Levels (Pre-Registered)

- **Level P1 — Engineering Recovery:** Vectorized grouping and solver reduce compression overhead from $18\text{ ms}$ to $<2\text{ ms}$, but late-depth compression ($l=8$) still cannot beat clean latency.
- **Level P2 — Earlier Functional Compression:** Operator-aware carrier optimization preserves accuracy at earlier depths ($l=3$ or $6$) where ordinary merging/pruning exhibit severe degradation.
- **Level P3 — End-to-End Acceleration:** At least one architecture and budget achieves measured whole-model latency $T_{\text{amortized}} < T_{\text{clean}}$ with competitive accuracy retention.
- **Level P4 — Accuracy-Latency Frontier Shift:** Operator-aware amortized compression defines a strictly superior Pareto frontier over ToMe, Pruning, and Clean across multiple architectures.
- **Level P5 — Mechanistic but Not Practical:** Operator geometry provides clear mathematical error reduction, but practical inference constraints prevent achieving hardware-level speedup.

---

## 9. Deliverables & Artifact Paths

1. **Protocol:** [`docs/FUNGIBILITY_PRACTICAL_OPERATOR_COMPRESSION_PROTOCOL.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_PRACTICAL_OPERATOR_COMPRESSION_PROTOCOL.md)
2. **Report:** [`docs/FUNGIBILITY_PRACTICAL_OPERATOR_COMPRESSION_REPORT.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_PRACTICAL_OPERATOR_COMPRESSION_REPORT.md)
3. **Core Engine:** [`patch_fungibility/practical_operator_compression.py`](file:///d:/Study/ResCancel/patch_fungibility/practical_operator_compression.py)
4. **Profiling Script:** [`scripts/profile_operator_compression.py`](file:///d:/Study/ResCancel/scripts/profile_operator_compression.py)
5. **Execution Script:** [`scripts/run_practical_operator_compression.py`](file:///d:/Study/ResCancel/scripts/run_practical_operator_compression.py)
6. **Outputs Directory:** `outputs/fungibility_practical_operator_compression/`
   - `detailed_solver_profile.csv`
   - `grouping_profile.csv`
   - `depth_latency_lower_bound.csv`
   - `depth_sweep.csv`
   - `fast_solver_ablation.csv`
   - `predictor_depth_ablation.csv`
   - `matched_budget_results.csv`
   - `latency_results.csv`
   - `accuracy_latency_frontier.csv`
   - `pareto_frontier.csv`
   - `validation_manifest.json`
7. **Figures Directory:** `figures/fungibility_practical_operator_compression/`
   - `figure_a_runtime_breakdown.png`
   - `figure_b_depth_latency_lower_bound.png`
   - `figure_c_accuracy_vs_depth.png`
   - `figure_d_operator_advantage_vs_depth.png`
   - `figure_e_fast_solver_tradeoff.png`
   - `figure_f_accuracy_token_frontier.png`
   - `figure_g_accuracy_latency_frontier.png`
   - `figure_h_pareto_summary.png`
