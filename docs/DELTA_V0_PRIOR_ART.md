# DELTA TRANSPORT V0: Prior-Art Audit and Novelty Boundary

**Document Status:** FROZEN  
**Date:** 2026-09-28  
**Repository:** `https://github.com/nhatminh-115/ResCancel`  
**Research Line:** Delta Transport V0 — Oracle Falsification of Spatially Selective Residual-Delta Reuse in Vision Transformers  
**Primary Novelty Boundary:**  
> *"Per-spatial-token selection over historical BLOCK RESIDUAL DELTAS in a standard pretrained Vision Transformer."*

---

## 1. Executive Summary of Prior-Art Audit

An exhaustive review of cross-layer routing, historical feature aggregation, and depth-wise attention was conducted across both language and vision transformer literature. The audit examined whether any existing work performs per-spatial-token selection over historical block residual deltas in pretrained Vision Transformers.

**Findings:**
- **No paper performs per-spatial-token selection over historical block residual deltas in standard Vision Transformers.**
- Recent advances such as **Delta Attention Residuals (Luo et al., May 2026)** investigate attending over sublayer/block deltas instead of cumulative hidden states, but this is restricted to autoregressive language models, alters the baseline residual connectivity, and requires extensive pretraining or fine-tuning.
- Other architectures like **DenseFormer (NeurIPS 2024)** use depth-weighted averaging over cumulative states, but apply token-invariant scalar weights.
- Vision Transformer architectures like **HAViT (IEEE CAI 2026)** and **ReViT** fuse attention maps across depth, but do not route or reuse residual deltas.
- Conditional computation approaches like **Mixture-of-Depths (Raposo et al., 2024)** and **DynamicViT (Rao et al., 2021)** route tokens to execute or skip entire transformer blocks to conserve FLOPs, rather than selecting historical deltas to inject into downstream representations.
- **Audit Decision:** **NO PRIOR-ART KILL**. The research line proceeds to the pre-registered oracle falsification experiment in **DELTA TRANSPORT V0**.

---

## 2. Structured Prior-Art Taxonomy & Collision Analysis

Below is the structured breakdown of audited literature against the pre-registered criteria.

### 2.1 Delta Attention Residuals (arXiv:2605.18855, May 2026)
- **Authors:** Cheng Luo, Zefan Cai, Junjie Hu.
- **What is routed:** Layer/block deltas $v_i = h_{i+1} - h_i$.
- **Across what axis:** Depth (preceding layers $0, \dots, l-1$).
- **Whether routing is token-specific:** Yes (via softmax attention over historical layer deltas).
- **Source objects:** Block/sublayer residual deltas ($\Delta_l$).
- **Modality:** Language Modeling (evaluated on 220M to 7.6B LLMs).
- **Pretrained vs. Training Required:** Requires fine-tuning or training from scratch; replaces the additive residual stream with learned softmax query-key attention over past deltas.
- **Exact Overlap with Hypothesis:** Shares the fundamental motivation that historical residual deltas ($\Delta$) provide sharper, non-redundant directional signals compared to cumulative hidden states ($h$).
- **Distinction from DELTA V0:** 
  1. *Scope & Modality:* Applied to 1D token sequences in LLMs, whereas DELTA V0 asks whether 2D spatial patch tokens in Vision Transformers require heterogeneous historical deltas.
  2. *Scientific Goal:* Delta Attention Residuals designs a trained architecture to improve validation perplexity. DELTA V0 is an *un-trained oracle falsification study* asking whether an omniscient label-aware per-token delta selector can beat a global depth selector on frozen, pretrained ViTs.
- **Collision Risk:** **MEDIUM**. Strong conceptual alignment regarding delta representations over cumulative states, but completely distinct in application, modality, and scientific formulation.

---

### 2.2 Attention Residuals (AttnRes) & Multi-Head Attention Residuals (MHAR) (arXiv:2603.15031, arXiv:2607.27230, 2026)
- **Authors:** Kimi Team / Moonshot AI.
- **What is routed:** Cumulative hidden states $h_0, \dots, h_{l-1}$.
- **Across what axis:** Depth across preceding layers.
- **Whether routing is token-specific:** Yes (per-token learned pseudo-queries attend across depth).
- **Source objects:** Cumulative hidden states ($h$).
- **Modality:** Language Modeling (Kimi Linear and Transformer models up to 48B).
- **Pretrained vs. Training Required:** Requires training from scratch or extensive adaptation.
- **Exact Overlap with Hypothesis:** Uses token-specific attention across depth.
- **Distinction from DELTA V0:** Routes cumulative hidden states, which suffer from depth-dilution and low-contrast routing collapse (as shown by Luo et al., 2026). Does not operate on residual deltas, does not study Vision Transformers, and requires training.
- **Collision Risk:** **MEDIUM**.

---

### 2.3 Depth-Attention: Cross-Layer Value Mixing (arXiv, mid-2026)
- **What is routed:** Key-Value (KV) cache activations across depth.
- **Across what axis:** Depth within self-attention modules.
- **Whether routing is token-specific:** Yes, token query attends over prior layers' keys at the same token position.
- **Source objects:** Attention Key-Value projections, not residual deltas.
- **Modality:** Transformer Language Models.
- **Pretrained vs. Training Required:** Requires trained architectural integration.
- **Exact Overlap with Hypothesis:** Reuses historical representations across depth.
- **Distinction from DELTA V0:** Operates inside the attention mechanism over KV states; does not interact with block residual updates or Vision Transformers.
- **Collision Risk:** **LOW**.

---

### 2.4 DenseFormer: Enhancing Information Flow via Depth Weighted Averaging (NeurIPS 2024)
- **Authors:** Matteo Pagliardini, Amirkeivan Mohtashami, François Fleuret, Martin Jaggi.
- **What is routed:** Cumulative hidden states from all past layers.
- **Across what axis:** Depth.
- **Whether routing is token-specific:** **NO**. DenseFormer uses Depth-Weighted-Average (DWA), where a set of learned scalar coefficients per layer weights previous representations. The weights are uniform across all tokens in the sequence.
- **Source objects:** Cumulative hidden states ($h_l$), not residual deltas.
- **Modality:** Language Modeling.
- **Pretrained vs. Training Required:** Requires training from scratch.
- **Exact Overlap with Hypothesis:** Acknowledges that intermediate representations retain valuable information lost in deep layers.
- **Distinction from DELTA V0:** Completely token-invariant (global per layer), operates on cumulative states rather than deltas, and requires training.
- **Collision Risk:** **LOW**.

---

### 2.5 HAViT: Historical Attention Vision Transformer (IEEE CAI 2026)
- **Authors:** Swarnendu Banik, Manish Das, Shiv Ram Dubey, Satish Kumar Singh.
- **What is routed:** Historical attention matrices ($A_l \in \mathbb{R}^{T \times T}$).
- **Across what axis:** Depth across self-attention blocks.
- **Whether routing is token-specific:** Operates at the token-to-token attention relation level.
- **Source objects:** Attention probability maps / scores.
- **Modality:** Computer Vision (ViTs).
- **Pretrained vs. Training Required:** New ViT backbone architecture trained from scratch.
- **Exact Overlap with Hypothesis:** Operates in Vision Transformers and links depth history.
- **Distinction from DELTA V0:** Fuses historical attention weights to enforce attention continuity; does not route, select, or modify residual delta updates ($\Delta_l$).
- **Collision Risk:** **LOW**.

---

### 2.6 ReViT: Residual Attention Vision Transformer (Diko et al., 2023/2024)
- **What is routed:** Attention maps and residual feature flows.
- **Across what axis:** Depth between consecutive self-attention modules.
- **Whether routing is token-specific:** Attention map level.
- **Source objects:** Self-attention maps to counteract feature collapse.
- **Modality:** Computer Vision.
- **Pretrained vs. Training Required:** Architecture trained from scratch.
- **Exact Overlap with Hypothesis:** Focuses on preserving local and low-level feature diversity across depth in ViTs.
- **Distinction from DELTA V0:** Modifies internal attention computation; does not perform selective historical block delta reuse.
- **Collision Risk:** **LOW**.

---

### 2.7 Mixture-of-Depths (MoD, Raposo et al., 2024) & DynamicViT (Rao et al., NeurIPS 2021)
- **What is routed:** Computational allocation (binary routing decision to compute or skip a block).
- **Across what axis:** Depth (per-layer token selection).
- **Whether routing is token-specific:** Yes, per-token routing based on top-$k$ router scores.
- **Source objects:** Router score derived from current token representation $h_l$. Non-selected tokens take the residual skip connection $h_{l+1} = h_l$.
- **Modality:** LLMs (MoD) and ViTs (DynamicViT, A-MoD).
- **Pretrained vs. Training Required:** Requires trained routers or fine-tuning.
- **Exact Overlap with Hypothesis:** Demonstrates that different spatial tokens have different depth requirements.
- **Distinction from DELTA V0:** MoD and DynamicViT dynamically *prune or skip* computation at layer $l$. They do not store a historical bank of past block residual updates $\Delta_{0..l}$ and selectively transport an earlier delta into a later layer.
- **Collision Risk:** **LOW**.

---

## 3. Summary Table: Prior Art vs. DELTA TRANSPORT V0

| Method | Modality | Source Object | Routing Axis | Token-Specific? | Requires Training? | Collision Risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Delta Attention Residuals (2026)** | NLP | Block/Sublayer $\Delta$ | Depth | Yes (Softmax Q-K) | Yes (Fine-tuning) | **MEDIUM** |
| **Attention Residuals (2026)** | NLP | Cumulative $h$ | Depth | Yes (Pseudo-query) | Yes (Pretraining) | **MEDIUM** |
| **Depth-Attention (2026)** | NLP | Attention KV pairs | Depth | Yes (Inside Attn) | Yes (Pretraining) | **LOW** |
| **DenseFormer (NeurIPS 2024)** | NLP | Cumulative $h$ | Depth | **No** (Token-invariant) | Yes (Pretraining) | **LOW** |
| **HAViT (IEEE CAI 2026)** | Vision | Attention maps $A$ | Depth | Yes (Token pairs) | Yes (From scratch) | **LOW** |
| **ReViT (2024)** | Vision | Attention maps $A$ | Depth | Yes (Token pairs) | Yes (From scratch) | **LOW** |
| **Mixture-of-Depths (2024)** | NLP/Vision | Execution flag (Compute vs Skip) | Depth | Yes (Top-$k$ router) | Yes (Router tuning) | **LOW** |
| **DELTA TRANSPORT V0 (Ours)** | **Vision (ViT)** | **Historical Block Deltas ($\Delta_{l,t}$)** | **Depth** | **Yes vs. Global (Oracle Test)** | **NO (Pretrained Oracle)** | **N/A (Target)** |

---

## 4. Prior-Art Conclusion & Novelty Boundary Statement

No existing work performs:
> *"Per-spatial-token selection over historical block residual deltas in standard pretrained Vision Transformers."*

Specifically:
1. Prior work using residual deltas as routing sources (**Delta Attention Residuals**) is confined to language modeling, alters the training architecture, and evaluates continuous perplexity improvements with trained query-key matrices.
2. Prior vision transformer work addressing cross-layer representation decay either fuses attention maps (**HAViT**, **ReViT**) or skips computation to save FLOPs (**DynamicViT**).
3. No study has conducted a **falsification test** to determine whether spatial patch tokens in standard pretrained ViTs (DeiT) actually benefit from *heterogeneous historical deltas* compared to a *globally optimal depth selection*.

Therefore, the **DELTA TRANSPORT V0** hypothesis possesses clear novelty and clear scientific boundary. We proceed directly to the frozen protocol and oracle falsification experiment.
