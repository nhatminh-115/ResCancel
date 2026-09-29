# Protocol: Fungibility-to-Compression Proof-of-Concept (Weighted Centroid Carrier)

**Experiment Name**: `FUNGIBILITY-TO-COMPRESSION POC: WEIGHTED CENTROID CARRIER`  
**Date**: 2026-09-29  
**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Execution Branch**: `main`  
**Scope**: Tightly scoped proof-of-concept application directly derived from the established Patch Content Fungibility mechanism.

---

## 1. Scientific & Engineering Context

### 1.1 From Mechanism to Compression
The dense fraction experiment established that Vision Transformers exhibit strong late-layer patch content fungibility:
- In DeiT-Small at Depth 8, up to **86.5%** of spatial patch tokens can be replaced by a single static calibration centroid $\mu_8$ while preserving $\ge 90\%$ of clean Top-1 accuracy.
- At this operating point, only $M \approx 26$ image-specific real anchor patches remain, while $m \approx 170$ patch slots contain identical copies of $\mu_8$.
- In standard forward passes, computing self-attention, LayerNorm, and MLP activations over $m = 170$ identical token copies wastes substantial downstream compute.

### 1.2 Central Engineering Question
> *"Can the many identical centroid-replaced late patch tokens be collapsed into ONE multiplicity-aware carrier token, preserving the predictive behavior of the uncompressed centroid intervention while reducing the downstream sequence length?"*

### 1.3 Prior Art Disclaimer (Non-Novelty of Multiplicity-Aware Attention)
Multiplicity-aware / proportional attention (adding $\log(s_j)$ to attention logits) is conceptually related to existing token-merging frameworks such as Token Merging (ToMe). **We do NOT claim this mathematical attention formulation as novel.**  
The novelty of this experiment lies in the **domain of application**: demonstrating that Vision Transformer representations naturally admit a late-depth regime where exact spatial patch content is fungible, allowing ordinary spatial tokens to be replaced by a class-agnostic generic prototype $\mu_l$ and compressed into a single weighted carrier without bespoke per-image clustering or training.

---

## 2. Mathematical Formulation & Equivalence Principle

### 2.1 Multiplicity-Aware Self-Attention
Suppose an uncompressed sequence at block $l$ contains $M$ real image-specific tokens and $m$ identical copies of centroid prototype $\mu_l$:
$$T_{\text{uncompressed}} = 1 \text{ (CLS)} + M \text{ (real anchors)} + m \text{ (duplicate centroids)} = 1 + N_{\text{patch}}$$

For any query vector $q_i$, the total attention contribution of the $m$ duplicate tokens is:
$$\sum_{c=1}^m \frac{\exp(q_i k_\mu / \sqrt{d})}{Z} v_\mu = \frac{m \cdot \exp(q_i k_\mu / \sqrt{d})}{Z} v_\mu$$
where $Z = \sum_{j \in \text{tokens}} \exp(q_i k_j / \sqrt{d})$.

By collapsing the $m$ duplicate tokens into **one carrier token** with multiplicity $s_{\text{carrier}} = m$ (and $s_j = 1$ for all other tokens), the attention logit is augmented:
$$A_{ij} = \frac{q_i k_j}{\sqrt{d}} + \log(s_j)$$
Thus:
$$\exp(A_{i,\text{carrier}}) = \exp\left(\frac{q_i k_\mu}{\sqrt{d}} + \log(m)\right) = m \cdot \exp\left(\frac{q_i k_\mu}{\sqrt{d}}\right)$$
The attention denominator and value contributions are mathematically identical:
$$Z_{\text{compressed}} = \sum_{r \in \text{real}} \exp(q_i k_r / \sqrt{d}) + m \cdot \exp(q_i k_\mu / \sqrt{d}) = Z_{\text{uncompressed}}$$
The compressed sequence length is:
$$T_{\text{compressed}} = 1 \text{ (CLS)} + M \text{ (real anchors)} + 1 \text{ (centroid carrier)} = M + 2$$
(If $m = 0$, no carrier is added, and $T = 1 + N_{\text{patch}}$).

### 2.2 Invariance of Downstream Block Operations
- **Query dimension:** Multiplicity is applied strictly to keys (keys and values). The single centroid carrier query represents all identical centroid queries, computing identical output $o_\mu$.
- **LayerNorm & MLP:** Point-wise per-token operations operate identically on the carrier state.
- **DINOv2 Readout Adjustment:** DINOv2 computes classification logits from $[CLS_{\text{norm}}; \text{mean}(Patch_{\text{norm}})]$. For compressed sequences, the patch mean must not be taken unweighted; instead, it is reconstructed using token multiplicities:
  $$\text{patch\_mean} = \frac{\sum_{i=1}^M \text{real}_{i,\text{norm}} + m \cdot \text{carrier}_{\text{norm}}}{N_{\text{original\_patches}}}$$

---

## 3. Experimental Parameters & Frozen Settings

### 3.1 Model Architectures & Depths
- **DeiT-Tiny (`deit_tiny_patch16_224`):** Depth 8 (Primary), $D=192, N_{\text{patch}}=196$.
- **DeiT-Small (`deit_small_patch16_224`):** Depth 8 (Primary), $D=384, N_{\text{patch}}=196$.
- **ViT-B/16 AugReg (`vit_base_patch16_224.augreg_in1k`):** Depth 7 (Primary late fungibility window), $D=768, N_{\text{patch}}=196$.
- **DINOv2 ViT-S/14 (`dinov2_vits14_lc`, layers=1):** Depth 9 (Primary late fungibility window), $D=384, N_{\text{patch}}=256$.
- *(Optional Secondary Depths: ViT-B Depth 8 and DINOv2 Depth 8 evaluated only if nearly free).*

### 3.2 Data Splits & Frozen Statistics
- Canonical ImageNet-1k validation subset: $N=1,000$ calibration (seed 9101), $N=1,000$ evaluation (seed 9201). Zero overlap.
- Calibration centroid vectors $\mu_l$ reused from frozen outputs (`outputs/fungibility_v0_6/` and `outputs/fungibility_v1/`).

### 3.3 Spatial Permutations & Anchor Selection
- Five frozen deterministic seeds: `31001, 31002, 31003, 31004, 31005`.
- Real anchors are the exact complement of the centroid replacement mask: $M = N_{\text{patch}} - k$. No saliency or attention-based selection in primary experiment.

### 3.4 Evaluated Fraction Grid (Compact Set)
Derived from frozen empirical thresholds:
- Standard fractions: $0\%, 25\%, 50\%, 75\%, 100\%$ (boundary diversity check).
- Model-specific retention points: Nearest actual patch counts corresponding to mean $F_{95}, F_{90}, F_{80}$ from dense sweep:
  - **DeiT-Tiny:** $F_{95} = 63.8\%$ ($k=125, M=71$), $F_{90} = 78.1\%$ ($k=153, M=43$), $F_{80} = 88.3\%$ ($k=173, M=23$).
  - **DeiT-Small:** $F_{95} = 74.5\%$ ($k=146, M=50$), $F_{90} = 86.7\%$ ($k=170, M=26$), $F_{80} = 90.8\%$ ($k=178, M=18$).
  - **ViT-B AugReg (D7):** $F_{95} = 43.9\%$ ($k=86, M=110$), $F_{90} = 63.8\%$ ($k=125, M=71$), $F_{80} = 79.6\%$ ($k=156, M=40$).
  - **DINOv2 (D9):** $F_{95} = 23.0\%$ ($k=59, M=197$), $F_{90} = 28.9\%$ ($k=74, M=182$), $F_{80} = 39.8\%$ ($k=102, M=154$).
- Deduplicate repeated fractions while preserving monotonic ordering.

---

## 4. Multi-Phase Execution Protocol

### Phase A: Exact Numerical Equivalence Validation (Mandatory Audit)
- Evaluate on at least $N = 64$ evaluation images.
- Compare:
  - **Condition A (Reference):** Full uncompressed sequence with $m$ duplicate $\mu_l$ tokens.
  - **Condition B (Compressed):** $M$ real anchors + 1 weighted carrier ($s_{\text{carrier}} = m$).
- Metrics tracked:
  - Prediction agreement (%)
  - Maximum absolute logit difference ($\max |l_{\text{ref}} - l_{\text{comp}}|$)
  - Mean absolute logit difference
  - Maximum hidden-state difference for CLS and real anchors across blocks.
- **Pass Target:** Prediction agreement = 100%; max absolute logit difference $\le 10^{-4}$ (ideally $\le 10^{-5}$).

### Phase B: Full 1,000-Image Evaluation
- Evaluate Weighted Centroid Carrier on all 1,000 evaluation images across all 5 spatial mask seeds.
- Confirm that accuracy, margin, and flip rates reproduce the uncompressed centroid baseline across all evaluated fraction points.

### Phase C: Matched-Budget Baseline Comparisons (Downstream Budget $B = M + 1$)
At the exact same downstream token budget $B$:
1. **Random Pruning:** Retain $B$ random real spatial patches, drop all others. (No carrier).
2. **Real Anchors + Unweighted Centroid ($s=1$):** Retain $M$ real anchors + 1 centroid carrier, but with unit weight $s=1$. Isolates the specific contribution of multiplicity-aware attention.
3. **Image-Mean Carrier:** Retain $M$ real anchors + 1 carrier initialized with the mean of discarded current-image patches ($s=m$). Tests whether exact image-specific discarded statistics provide value over the generic static calibration centroid $\mu_l$.
4. **ToMe Baseline:** Standard training-free token merging baseline at matched token budget (or document reason if incompatible).

### Phase D: FLOP Analysis & Physical Latency Benchmarking
1. **Theoretical FLOPs Accounting:**
   - Multi-head Attention FLOPs: $4 N D^2 + 2 N^2 D$
   - MLP FLOPs: $8 N D^2$ (or $6 N D^2$ for SwiGLU in DINOv2)
   - LayerNorm FLOPs: $2 N D$
   - Total downstream block FLOPs vs. full model FLOPs.
2. **Physical Latency Benchmarking:**
   - Target GPU: NVIDIA GeForce RTX 5070 Laptop GPU.
   - Batch sizes: $1$ and $16$.
   - Protocol: End-to-end (full image $\to$ final logits; NOT from cached activations).
   - Warmups: $\ge 5$, Timed iterations: $\ge 30$, timing via `torch.cuda.Event`.
   - Report: Mean latency, median latency, SD, tail speedup %, end-to-end speedup %.

---

## 5. Pre-Registered Decision Framework

### Equivalence Verdict:
- **Outcome A — EXACT CARRIER WORKS:** Multiplicity-aware compression reproduces the uncompressed centroid intervention to numerical tolerance ($\max \text{error} \le 10^{-4}$, prediction agreement $100\%$).
- **Outcome B — APPROXIMATE CARRIER WORKS:** Predictions remain effectively equivalent, but exact logits differ slightly for an understood numerical/architectural reason.
- **Outcome C — CARRIER EQUIVALENCE FAILS:** Compression materially changes predictions despite correct multiplicity handling. (Stop and diagnose).

### Practical Utility Verdict:
- **USEFUL APPLICATION:** Weighted carrier preserves centroid-reference accuracy, provides meaningful downstream token/FLOP reduction, produces measurable end-to-end latency speedup, and is competitive with matched-budget baselines.
- **MECHANISTIC DEMONSTRATION ONLY:** Carrier equivalence holds, but total end-to-end speed gain is negligible because intervention occurs too late in the network (e.g. only 3–4 blocks remaining).
- **NOT COMPETITIVE:** Matched-budget baselines (Random Pruning or Image-Mean Carrier) clearly dominate the accuracy-efficiency Pareto frontier.
