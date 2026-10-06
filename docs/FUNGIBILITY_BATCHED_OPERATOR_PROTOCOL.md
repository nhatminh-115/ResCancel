# Protocol: Batched Operator Compression
# Throughput Acceleration, FlashAttention Multiplicity, and Hybrid Dynamic Grouping

**Status:** Pre-registered, Frozen Protocol  
**Date:** 2026-10-06  
**Repository:** `https://github.com/nhatminh-115/ResCancel`  
**Working Directory:** `d:\Study\ResCancel`  
**Target Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8.15 GB VRAM, sm_120)  
**Authors:** Patch Content Fungibility Research Program  

---

## 1. Context, Motivation & Central Question

The preceding Practical Operator Compression benchmark established:
1. **Algorithmic Overhead Elimination:** The reported $20\text{--}50\text{ ms}$ carrier solve was largely host-side Python and synchronization overhead. Pure GPU vectorized Cholesky solving requires only **$1.41\text{--}1.51\text{ ms}$** with exact float32 fidelity ($\|\Delta C_{\text{fast}} - \Delta C_{\text{orig}}\|_F \le 2.2 \times 10^{-6}$).
2. **Zero-Cost Grouping:** Fixed 2D spatial grid grouping runs in **$0.007\text{--}0.018\text{ ms}$**, eliminating clustering overhead entirely.
3. **Hardware Regime at Batch Size 1:** Single-stream inference ($BS=1$) on modern GPUs is memory-bandwidth and kernel-launch bound. Shortening the sequence from 197 to 50 tokens yields only a $1.08\times$ block speedup, meaning clean ViT remains faster than any token reduction pipeline at $BS=1$.
4. **Compute Scaling at Batched Inference:** At $BS=8$, sequence shortening yields a **$3.37\times$** block speedup; at $BS=16$, a **$4.02\times$** block speedup.
5. **FlashAttention Multiplicity Penalty:** Passing log-multiplicity as an explicit attention bias currently disables FlashAttention / fused SDP kernels, inducing a $45\%$ slowdown in scaled dot-product attention.
6. **Semantic vs. Spatial Tradeoff:** Fixed spatial grouping is essentially free, but ignores semantic object boundaries at earlier transformer depths.

### Primary Research Question
> **At what batch size does Fast Amortized Operator Compression cross over from slower than clean inference to higher image throughput (images/sec) than clean inference, while preserving a Pareto accuracy advantage over matched token-compression baselines?**

### Secondary Research Questions
- **Question A (FlashAttention Compatibility):** Can token multiplicity semantics be preserved (exactly or with verified high fidelity) WITHOUT disabling the fused FlashAttention / memory-efficient SDP kernel?
- **Question B (Hybrid Dynamic Grouping):** Can a lightweight, non-iterative semantic grouping mechanism ($<0.5\text{ ms}$) outperform fixed spatial grouping at early/mid layers without sacrificing batched throughput?

---

## 2. Experimental Scope & Architectural Grid

We evaluate four diverse vision transformer architectures:
1. **DeiT-Tiny** (`deit_tiny_patch16_224`): $N=196, D=192$, 12 blocks.
2. **DeiT-Small** (`deit_small_patch16_224`): $N=196, D=384$, 12 blocks.
3. **ViT-Base** (`vit_base_patch16_224.augreg_in1k`): $N=196, D=768$, 12 blocks.
4. **DINOv2-S/14** (`dinov2_vits14_lc`): $N=256, D=384$, 12 blocks.

### Evaluated Batch Sizes
$$BS \in \{1, 2, 4, 8, 16, 32, 64\}$$
For smaller backbones (DeiT-Tiny, DeiT-Small), test $BS \in \{96, 128\}$ if GPU VRAM permits without OOM.

### Evaluated Intervention Depths
To maximize suffix acceleration while maintaining non-trivial subspace predictability:
- **DeiT-Tiny / DeiT-Small / DINOv2-S:** Mid-depth $l=6$, Late-depth $l=8$
- **ViT-Base:** Mid-depth $l=5$, Late-depth $l=7$

### Evaluated Budgets
- **50% Retention:** $B = 98$ tokens (DeiT / ViT-B), $B = 128$ tokens (DINOv2)
- **25% Retention:** $B = 49$ tokens (DeiT / ViT-B), $B = 64$ tokens (DINOv2)
- **~16% Retention:** $B = 32$ tokens (DeiT / ViT-B), $B = 42$ tokens (DINOv2)

---

## 3. Phase 1: Batched Throughput & Crossover Formulation

For each architecture, batch size $BS$, depth $l$, and budget $B$:
- Measure wall-clock batch latency $T_{\text{batch}}$ (ms) via synchronized CUDA events (`torch.cuda.Event`).
- Compute per-image latency: $T_{\text{img}} = T_{\text{batch}} / BS$ (ms).
- Compute image throughput: $\text{Throughput} = 1000 \times BS / T_{\text{batch}}$ (images/sec).
- Measure peak allocated and reserved VRAM (MB).

### Crossover Metric
$$\text{Throughput Gain}(BS) = \frac{\text{Throughput}_{\text{operator}}(BS)}{\text{Throughput}_{\text{clean}}(BS)}$$
$$\text{Crossover Batch Size } BS^* = \min \{ BS \mid \text{Throughput Gain}(BS) > 1.0 \}$$
A method qualifies as a **practical throughput accelerator** if $BS^*$ exists within standard serving batch limits ($BS^* \le 64$) while maintaining competitive Top-1 retention.

---

## 4. Phase 2: FlashAttention-Compatible Multiplicity Formulation

### Mathematical Derivation of Exact Multiplicity
When $m_j$ identical or merged tokens are collapsed into token $j$, the total attention mass allocated to group $j$ by query $q_i$ should equal the sum of uncompressed attentions:
$$A_{i, j} = \sum_{k \in G_j} \frac{\exp(q_i^\top k_k / \sqrt{d})}{\sum_l \exp(q_i^\top k_l / \sqrt{d})} \propto m_j \exp\left(\frac{q_i^\top k_j}{\sqrt{d}}\right) = \exp\left(\frac{q_i^\top k_j}{\sqrt{d}} + \log m_j\right)$$

### Evaluated Reformulations:
1. **Route A (Explicit Additive Bias - Reference):** Adds $\log m_j$ inside softmax. Semantically exact, but disables FlashAttention on standard PyTorch.
2. **Route B (Augmented Coordinate Formulation):** Augments query and key by 1 dimension:
   $$\tilde{q}_i = \left[ q_i \sqrt{\frac{d+1}{d}}, \, 1 \right], \quad \tilde{k}_j = \left[ k_j, \, \sqrt{d+1} \log m_j \right]$$
   Then:
   $$\frac{\tilde{q}_i^\top \tilde{k}_j}{\sqrt{d+1}} = \frac{q_i^\top k_j \sqrt{(d+1)/d} + \sqrt{d+1} \log m_j}{\sqrt{d+1}} = \frac{q_i^\top k_j}{\sqrt{d}} + \log m_j$$
   Test whether $d+1$ preserves tensor core FlashAttention support.
3. **Route C (Folded Key Scaling / Unmasked Approximation):** Scales key or value embeddings directly without additive bias.
4. **Route D (Uniform Multiplicity Assumption):** When using regular spatial grid partitions ($m_j \approx N/B$ is uniform across all tokens), $\log m_j$ is a row-constant shift that cancels out exactly in softmax!
   $$\text{Softmax}\left( \frac{q_i^\top k_j}{\sqrt{d}} + C \right) = \text{Softmax}\left( \frac{q_i^\top k_j}{\sqrt{d}} \right)$$
   For uniform spatial grouping, **multiplicity bias is mathematically redundant**!

All routes will be audited for numerical parity in `multiplicity_kernel_audit.csv`.

---

## 5. Phase 3: Hybrid Dynamic Grouping (IP-SM)

To address the weakness of static spatial grouping on fine-grained semantic boundaries:
- **Importance-Preserve + Spatial Merge (IP-SM):**
  1. Compute 1-pass patch saliency $w \in \mathbb{R}^N$ from intermediate activation norms or CLS attention ($<0.05\text{ ms}$).
  2. Select top-$K_{\text{keep}}$ foreground patches ($K_{\text{keep}} = B / 4$ or 16) and preserve them uncompressed.
  3. Spatially pool the remaining $N - K_{\text{keep}}$ background patches into $B - K_{\text{keep}}$ carrier tokens.
  4. Apply operator carrier optimization across the resulting partition.

### Controlled Comparison
On identical images, depths, and budgets, evaluate:
1. Spatial Group Mean (Centroid)
2. Spatial + Operator Carrier
3. Hybrid Group Mean (Centroid)
4. Hybrid + Operator Carrier
5. Attention Pruning (EViT)
6. ToMe

---

## 6. Pre-Registered Success Levels

- **Level B1 — Batched Crossover:** At least one architecture/budget achieves $\text{Throughput}_{\text{operator}} > \text{Throughput}_{\text{clean}}$ at $BS \ge 8$.
- **Level B2 — Multi-Architecture Throughput Gain:** Operator compression beats clean throughput on at least 3 out of 4 architectures at a realistic batch size.
- **Level B3 — Strong Baseline Frontier:** Operator method improves the accuracy-throughput Pareto frontier relative to Attention Pruning and ToMe on at least 2 architectures.
- **Level B4 — FlashAttention Recovery:** Exact or near-exact multiplicity handling preserves fast fused SDP kernels and materially accelerates batched suffix execution.
- **Level B5 — Hybrid Grouping Gain:** Cheap semantic-aware grouping improves early/mid-depth accuracy over fixed spatial grouping while adding $<0.5\text{ ms}$ overhead.
- **Level B6 — Throughput Practical Success:** Operator-aware compression delivers $>1.2\times$ clean throughput with $\le 2$ percentage points Top-1 accuracy loss on at least one major backbone.

---

## 7. Deliverables & Artifact Paths

1. **Protocol:** [`docs/FUNGIBILITY_BATCHED_OPERATOR_PROTOCOL.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_BATCHED_OPERATOR_PROTOCOL.md)
2. **Report:** [`docs/FUNGIBILITY_BATCHED_OPERATOR_REPORT.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_BATCHED_OPERATOR_REPORT.md)
3. **Core Engine:** [`patch_fungibility/batched_operator_compression.py`](file:///d:/Study/ResCancel/patch_fungibility/batched_operator_compression.py)
4. **Scripts:**
   - [`scripts/profile_batched_operator.py`](file:///d:/Study/ResCancel/scripts/profile_batched_operator.py)
   - [`scripts/eval_batched_operator.py`](file:///d:/Study/ResCancel/scripts/eval_batched_operator.py)
   - [`scripts/audit_multiplicity_flashattention.py`](file:///d:/Study/ResCancel/scripts/audit_multiplicity_flashattention.py)
   - [`scripts/eval_hybrid_grouping.py`](file:///d:/Study/ResCancel/scripts/eval_hybrid_grouping.py)
5. **Outputs Directory:** `outputs/fungibility_batched_operator/`
   - `batch_throughput.csv`
   - `batch_crossover.csv`
   - `batch_runtime_breakdown.csv`
   - `multiplicity_kernel_audit.csv`
   - `hybrid_grouping_results.csv`
   - `predictor_batch_ablation.csv`
   - `solver_batch_ablation.csv`
   - `depth_batch_ablation.csv`
   - `matched_budget_results.csv`
   - `throughput_frontier.csv`
   - `pareto_frontier.csv`
   - `validation_manifest.json`
6. **Figures Directory:** `figures/fungibility_batched_operator/`
   - `figure_a_batch_crossover.png`
   - `figure_b_throughput_vs_batch.png`
   - `figure_c_accuracy_throughput_frontier.png`
   - `figure_d_flashattention_multiplicity.png`
   - `figure_e_hybrid_grouping.png`
   - `figure_f_depth_batch_interaction.png`
   - `figure_g_predictor_batch_scaling.png`
   - `figure_h_cross_architecture_pareto.png`
