# Patch Fungibility: Dense Fraction & Spatial-Mask Robustness Sweep Protocol

**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Protocol Version**: 1.0 (Frozen Pre-Registration)  
**Date**: 2026-09-29  
**Experiment Name**: `PATCH FUNGIBILITY — DENSE FRACTION / MASK ROBUSTNESS SWEEP`  

---

## 1. Scientific Purpose & Research Question

Previous iterations (V0.6–V1) established late-layer patch content fungibility across four model families using coarse spatial replacement budgets (25%, 50%, 75%, 100%) and a single deterministic spatial permutation per model. While these experiments established the phenomenon, they do not resolve key reviewer-facing questions:
1. **Mask Robustness**: Does content fungibility depend on a "lucky" spatial ordering of patch positions, or does it hold consistently across independently drawn spatial permutations?
2. **Dose-Response Shape**: What is the continuous functional relationship between the fraction of replaced patches and downstream classification degradation? Is degradation gradual or characterized by a sharp cliff?
3. **Continuous Advantage**: Does class-agnostic centroid and Gaussian replacement consistently outperform destructive Zero replacement across the entire range of fractions, or only at isolated benchmark points?
4. **Degradation Thresholds**: Exactly what percentage of the spatial patch stream can be substituted before accuracy drops below 95% ($F_{95}$), 90% ($F_{90}$), and 80% ($F_{80}$) of the clean baseline?

**Primary Research Question**:  
> *"Across independently sampled spatial mask orderings, how does classification accuracy and true-class logit margin change continuously as an increasing fraction of the patch stream is replaced?"*

This experiment is strictly a **robustness, characterization, and visualization study**. It does not invent new mechanisms, propose routers, train weights, or search for alternative depths.

---

## 2. Frozen Models & Architectures

Four frozen pretrained model families spanning two scales ($D=192, 384, 768$), two patch token counts ($N=196, 256$), three training paradigms (DeiT distillation, Supervised AugReg, Self-Supervised DINOv2), and two classifier readout topologies (CLS vs. CLS $\oplus$ Patch Mean):

1. **Model A: DeiT-Tiny** (`deit_tiny_patch16_224` via `timm`)
   - 12 blocks, $D=192$, 196 spatial patch tokens, ImageNet-1k CLS head.
2. **Model B: DeiT-Small** (`deit_small_patch16_224` via `timm`)
   - 12 blocks, $D=384$, 196 spatial patch tokens, ImageNet-1k CLS head.
3. **Model C: ViT-B/16 AugReg** (`vit_base_patch16_224.augreg_in1k` via `timm`)
   - 12 blocks, $D=768$, 196 spatial patch tokens, ImageNet-1k CLS head.
4. **Model D: DINOv2 ViT-S/14** (`dinov2_vits14_lc`, layers=1, pretrained=True via PyTorch Hub)
   - 12 blocks, $D=384$, **256 spatial patch tokens**, NO-REGISTER.
   - Readout consumes $[\text{CLS}_{\text{norm}} \ ; \ \text{mean}(\text{Patch}_{\text{norm}})] \to \text{linear\_head}$.

All pretrained weights are strictly frozen (`requires_grad=False`). Official model-specific preprocessing is applied independently.

---

## 3. Dataset Splits & Calibration Integrity

- **Evaluation Set**: Exactly canonical $N_{\text{eval}} = 1,000$ disjoint validation images (stratified 1 per class).
- **Calibration Set**: Exactly canonical $N_{\text{calib}} = 1,000$ disjoint validation images.
- **Strict Disjointness**: $\text{Intersection}(\text{IDs}_{\text{calib}}, \text{IDs}_{\text{eval}}) = \emptyset$. Zero image overlap.
- **No Evaluation Leakage**: Calibration vectors ($\mu_l, \sigma_l$) are drawn from frozen calibration statistics (`outputs/fungibility_v0_6/calibration_statistics.npz` and `outputs/fungibility_v1/calibration_statistics.npz`). No evaluation activations or labels are used.

---

## 4. Frozen Intervention Depths

Intervention depths are frozen strictly from existing empirical evidence:
- **DeiT-Tiny**: **Depth 8** (output Block 8 / input Block 9).
- **DeiT-Small**: **Depth 8** (output Block 8 / input Block 9).
- **ViT-B/16 AugReg**: **Depth 7** (Primary result; identified as peak fungibility window in V1 pre-registered sweep with advantage $+0.895$). Standardized Depth 8 is evaluated as a secondary reference.
- **DINOv2 ViT-S/14**: **Depth 9** (Primary result; identified as peak fungibility window in V1 pre-registered sweep with advantage $+8.005$). Standardized Depth 8 is evaluated as a secondary reference.

No post-hoc depth optimization is permitted.

---

## 5. Spatial Mask Selection & 5 Independent Seeds

To evaluate robustness against which spatial positions are replaced:
- **Five Deterministic Mask Seeds**: `31001`, `31002`, `31003`, `31004`, `31005`.
- **Image-Independent Permutation**: For each model and each seed, sample ONE permutation $\pi$ of spatial patch indices $\{0, \dots, N_{\text{patch}}-1\}$. The exact same ordered list is applied to every evaluation image for that seed.
- **Strictly Nested Prefix Masks**: For token count $k$, the mask $M(k)$ comprises the first $k$ indices in $\pi$. This guarantees:
  $$M(k) \subset M(k+1) \quad \forall k \in \{0, \dots, N_{\text{patch}}-1\}$$
  Ensuring that increasing fraction strictly represents progressively accumulating replaced positions rather than switching spatial coordinates.
- **CLS Isolation**: Token index 0 (CLS) is strictly untouched across all conditions and depths.
- **Token Counts**: $N_{\text{patch}} = 196$ for DeiT/ViT-B; $N_{\text{patch}} = 256$ for DINOv2.

---

## 6. Dense Fraction Grid & Fallback Hierarchy

- **Target Grid**: 101 requested fraction points: $f \in \{0.00, 0.01, 0.02, \dots, 1.00\}$.
- **Integer Token Mapping**: For each model with spatial count $N_{\text{patch}}$, map $k = \text{round}(f \cdot N_{\text{patch}})$.
- **Deduplication**: Deduplicate repeated integer $k$ values while preserving monotonic ordering.
- **Exact Landmarks**: Grid must contain exact fractions $0.00$, $0.25$, $0.50$, $0.75$, $1.00$ and exact $k = N_{\text{patch}}$.
- **Logging**: Both `requested_fraction`, `actual_k`, and `actual_fraction = actual_k / N_patch` are logged.
- **Fallback Hierarchy**:
  1. Default: Full 1% grid (~101 points).
  2. Fallback 1: 2% grid ($0, 0.02, \dots, 1.00$, ~51 points) if runtime is prohibitive.
  3. Fallback 2: 5% grid ($0, 0.05, \dots, 1.00$, 21 points) if 2% grid is impractical.
  4. Never coarser than 5%.

---

## 7. Intervention Conditions & Randomness Separation

For every fraction $k$ and spatial mask seed, evaluate:
1. **CLEAN**: Baseline untouched model pass (evaluated once per model).
2. **ZERO**: Masked patch activations $\to 0$.
3. **CENTROID**: Masked patch activations $\to \mu_l$.
4. **DIAGONAL GAUSSIAN**: Masked patches receive independent samples $h_t \sim \mathcal{N}(\mu_l, \text{diag}(\sigma_l^2))$. Evaluated across 3 deterministic replacement seeds: `32001`, `32002`, `32003`.

### Strict Randomness Separation:
- Mask seed controls ONLY the spatial positions selected.
- Gaussian replacement seed controls ONLY the stochastic noise vectors drawn.
- Aggregation is strictly hierarchical:
  1. Average over replacement seeds within each spatial-mask seed.
  2. Summarize across the 5 independent spatial-mask seeds.

---

## 8. Quantitative Metrics & Dose-Response Analysis

### Per-Condition Metrics:
- Top-1 Accuracy ($\text{Acc}$).
- Mean and Median true-class logit margin ($m = z_y - \max_{j \ne y} z_j$).
- Margin Damage $\Delta m(f) = m_{\text{clean}} - m(f)$.
- Fungibility Advantage $A(f) = \Delta m_{\text{zero}}(f) - \Delta m_{\text{replacement}}(f)$.
- Recovery Fraction $R(f) = \frac{\Delta m_{\text{zero}}(f) - \Delta m_{\text{replacement}}(f)}{\Delta m_{\text{zero}}(f)}$ for $\Delta m_{\text{zero}}(f) \ge 0.10$ (reported as N/A when $\Delta m_{\text{zero}} < 0.10$).

### Mask Robustness Summary:
- Mean $\pm$ SD across the 5 spatial mask seeds.
- $\text{MaskSpread}_{\text{acc}}(f) = \max_{\text{seed}} \text{Acc} - \min_{\text{seed}} \text{Acc}$.
- $\text{MaskSD}_{\text{acc}}(f) = \text{SD}_{\text{seeds}}(\text{Acc})$.

### Dose-Response Functional Characterization:
1. **Trapezoidal AUC**: Normalized over $x \in [0, 1]$ for Accuracy, Margin, and Damage.
2. **Threshold Crossings**: Directly identified by grid crossing:
   - $F_{95}$: Largest fraction retaining $\ge 95\%$ of clean Top-1 accuracy.
   - $F_{90}$: Largest fraction retaining $\ge 90\%$ of clean Top-1 accuracy.
   - $F_{80}$: Largest fraction retaining $\ge 80\%$ of clean Top-1 accuracy.
3. **Degradation Cliff**: Adjacent fraction interval with maximum local drop in accuracy/margin.
4. **Contiguous Fungibility Window**: Contiguous fraction interval where $\Delta m_{\text{zero}} \ge 0.10$ AND Recovery $\ge 50\%$ across the majority of mask seeds.

---

## 9. Computational Guardrails & Optimization

- **Hardware**: NVIDIA RTX GPU, 8 GB VRAM. Hard safety ceiling: peak allocated VRAM $< 6.8\text{ GB}$.
- **Activation Caching**: To make the dense fraction sweep computationally tractable, evaluation activations at the intervention depth are cached once per model. Downstream blocks $l+1 \dots L$ and classification heads are executed on cached hidden states.
- **Cache Validation**: Verified that forwarding cached activations matches full end-to-end forward to machine precision ($0.00\times 10^0$ logit difference).

---

## 10. Programmatic Protocol Assertions (20 Checks)

1. Pretrained weights strictly frozen (`requires_grad=False`).
2. Canonical evaluation split reused ($N=1,000$).
3. Canonical calibration split reused ($N=1,000$).
4. Zero image overlap between calibration and evaluation.
5. Intervention depths match pre-registered values (Tiny: 8, Small: 8, ViT-B: 7 [8], DINOv2: 9 [8]).
6. CLS token untouched in all interventions.
7. Mask permutations are true bijections of $\{0, \dots, N_{\text{patch}}-1\}$.
8. Five spatial mask seeds (`31001..31005`) are distinct and produce distinct permutations.
9. Prefix masks are strictly nested ($M(k) \subset M(k+1)$).
10. Masks are independent of image pixels, labels, attention, and activation norms.
11. Spatial patch counts exact: DeiT/ViT-B = 196, DINOv2 = 256.
12. Actual fractions and integer $k$ values logged.
13. Mask seed and Gaussian seed strictly separated.
14. Clean baseline reproduces canonical accuracy within numerical tolerance.
15. Calibration statistics ($\mu_l, \sigma_l$) match canonical frozen values.
16. Zero evaluation data leaks into calibration vectors.
17. 0% intervention ($k=0$) matches clean baseline identically.
18. 100% intervention ($k=N_{\text{patch}}$) replaces every spatial patch.
19. Two-stage hierarchical aggregation enforced.
20. All output tables uniquely keyed and reproducible.

---

## 11. Pre-Registered Decision Rules

- **`ROBUST`**: The Centroid/Gaussian > Zero advantage persists across the majority of the fraction range where Zero causes meaningful damage ($\Delta m \ge 0.10$), and the condition ordering is strictly consistent across all 5 spatial mask seeds.
- **`ROBUST WITH SPATIAL HETEROGENEITY`**: The advantage persists overall, but mask-to-mask variance is substantial ($\text{MaskSpread} > 10\%$) or specific fraction regions depend strongly on spatial position.
- **`MASK-SENSITIVE`**: The phenomenon depends heavily on which specific patch positions are replaced, failing to replicate across independent spatial permutations.
