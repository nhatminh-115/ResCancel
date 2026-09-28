# PATCH CONTENT FUNGIBILITY V0: Prior-Art Audit and Novelty Boundary

**Document Status:** FROZEN  
**Date:** 2026-09-28  
**Repository:** `https://github.com/nhatminh-115/ResCancel`  
**Research Line:** Patch Content Fungibility V0 — Mechanistic Falsification of Depth-Wise Token Content Dependence  
**Primary Novelty Question:**  
> *"Has prior work directly mapped across depth whether standard ViT spatial patch-token slots remain causally necessary while their exact image-specific activation content becomes replaceable by in-distribution content?"*

---

## 1. Executive Summary & Audit Verdict

A focused review of literature at the intersection of Vision Transformer interpretability, causal ablation, activation patching, and token dynamics was conducted. The audit covered:
1. Zero-ablation vs. resampling/mean ablation in Vision Transformers;
2. DINO register tokens and high-norm feature map artifacts (Darcet et al., 2023/2024);
3. Activation patching and causal tracing in Vision Transformers (ViT-Prisma, CAAP);
4. Patch replacement and token mixing in training/augmentation (MAE, Panoptic Patch Learning, CutMix);
5. Dynamic token reduction and token merging (ToMe, DynamicViT).

**Findings:**
- **No prior work has systematically mapped the depth-wise emergence of patch-token content fungibility in standard pretrained Vision Transformers.**
- Prior interpretability research establishes that zero-ablation can induce out-of-distribution artifacts and that resampling ablation (patching from another image) maintains on-distribution activations. However, these techniques are predominantly deployed for *local feature attribution* (asking which specific spatial patch conveys the object's identity to the classification head), not for characterizing the layer-wise causal requirement of patch slots.
- Darcet et al. (ICLR 2024) demonstrated that certain patch tokens are repurposed by the network as global computational workspaces ("registers"), but did not investigate whether normal spatial patches exhibit depth-dependent content fungibility.
- **Audit Decision:** **NO PRIOR-ART KILL**. The research line proceeds to the pre-registered falsification protocol.

---

## 2. Structured Prior-Art Taxonomy & Collision Analysis

### 2.1 Causal Attribution via Activation Patching (CAAP) & ViT-Prisma
- **Scope:** Mechanistic interpretability of Vision Transformers (ViT, CLIP).
- **Core Mechanism:** Caches activations from a "clean" image and swaps them into a "corrupted" image (or vice versa) to locate causal sub-circuits or measure localized spatial importance.
- **Distinction from Fungibility V0:**
  - Activation patching asks: *"Does injecting the patch from image A into image B cause the model to output class A?"* (information presence and steering).
  - Fungibility V0 asks: *"Does injecting wrong-image content into image A prevent the model from outputting class A, or does the slot merely require a plausible token?"*
  - Fungibility V0 explicitly formalizes and measures the **FungibilityGap**:
    $$\text{FungibilityGap}(l) = \text{Damage}_{\text{zero}}(l) - \text{Damage}_{\text{cross}}(l)$$
    across multiple pre-registered transformer depths ($l \in \{2, 4, 6, 8, 10\}$) under strictly controlled 25% spatial ablations.
- **Collision Risk:** **LOW–MEDIUM**. High methodological similarity in using activation replacement, but completely different research question and estimand.

---

### 2.2 Vision Transformers Need Registers (Darcet et al., Meta FAIR / Inria, ICLR 2024)
- **Scope:** Investigates high-norm feature artifacts in self-supervised (DINOv2) and supervised ViTs.
- **Core Mechanism:** Shows that ViTs repurpose low-information background patches as "scratchpads" to hold global context, stripping them of local spatial details. Proposes adding explicit register tokens.
- **Distinction from Fungibility V0:**
  - Darcet et al. focused on identifying and mitigating high-norm outlier tokens that degrade dense downstream tasks (segmentation/object discovery).
  - Fungibility V0 investigates standard supervised models (DeiT) on general classification, testing arbitrary spatial patch positions (random 25% mask) rather than background outlier tokens, and systematically evaluates whether intermediate patch slots are *content-fungible* across depth.
- **Collision Risk:** **LOW**.

---

### 2.3 Zero-Ablation vs. Resampling / Mean Ablation Literature
- **Scope:** Methodological papers in transformer interpretability (e.g., Wang et al., 2022; Nanda et al.).
- **Core Mechanism:** Demonstrates that zeroing activations often produces massive performance drops due to artificial out-of-distribution (OOD) representation disruption, whereas mean or resampling ablation isolates true functional dependence.
- **Distinction from Fungibility V0:**
  - These papers discuss ablation methodology generally or apply it to language models.
  - Fungibility V0 explicitly leverages this methodological distinction as a scientific probe into the ViT computational graph: if zero-ablation causes severe damage while plausible cross-image replacement causes negligible damage, the slot's computational presence is required but its exact content is fungible.
- **Collision Risk:** **LOW**.

---

### 2.4 Token Merging (ToMe, Bolya et al., ICLR 2023) & Token Pruning (DynamicViT)
- **Scope:** Compute efficiency in Vision Transformers.
- **Core Mechanism:** Progressively merges or drops similar/redundant patch tokens based on key-key cosine similarity to accelerate inference.
- **Distinction from Fungibility V0:**
  - Token merging reduces sequence length by combining mutually redundant tokens from the *same image*.
  - Fungibility V0 maintains constant sequence length (197 tokens) and tests whether tokens can be replaced with *unrelated donor tokens from different images* or *spatially permuted tokens* without performance loss.
- **Collision Risk:** **LOW**.

---

## 3. Summary Table: Prior Art vs. PATCH CONTENT FUNGIBILITY V0

| Research Area | Key Papers | Primary Focus | Evaluates Cross-Image Resampling Across Depth? | Distinguishes Slot Requirement from Content Requirement? | Collision Risk |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Activation Patching in ViT** | CAAP, ViT-Prisma | Spatial circuit localization | No (focuses on target transfer) | No | **LOW–MEDIUM** |
| **ViT Registers** | Darcet et al. (ICLR 2024) | High-norm background artifacts | No | No (proposes registers) | **LOW** |
| **Ablation Methodology** | Nanda, Wang et al. | OOD effects of zero-ablation | No (general NLP focus) | Methodological only | **LOW** |
| **Token Merging** | ToMe, DynamicViT | Speedup via token reduction | No (merges same-image tokens) | No | **LOW** |
| **FUNGIBILITY V0 (Ours)** | **This Work** | **Depth-wise content fungibility** | **YES (Depths 2, 4, 6, 8, 10)** | **YES (Explicit FungibilityGap)** | **Target** |

---

## 4. Prior-Art Audit Conclusion

No prior work directly answers:
> *"Whether patch-token slots inside pretrained Vision Transformers continue to require the correct image-specific content across depth, or whether there exists a regime where the model still requires an active slot but no longer requires the exact original content."*

The novelty boundary is clear and uncontested. We proceed immediately to the pre-registered experimental protocol.
