# Manuscript Structure & Section Outline: Patch Content Fungibility in Vision Transformers

**Working Title**: *Patch Content Fungibility in Vision Transformers*  
**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Date**: 2026-09-29  

---

## Section-by-Section Manuscript Plan

### Section 1: Introduction
- **Core Narrative**: Vision Transformers partition images into spatial grids of patch tokens that are processed through stacked self-attention and MLP blocks. A standard assumption is that each spatial token continues to represent localized, image-specific visual features throughout the network depth. We investigate a fundamental mechanistic question: *What information must actually remain inside late spatial patch activations for downstream ViT classification to function?*
- **Central Finding**: Late Vision Transformer patch representations become *content-fungible*: their exact image-specific features can be replaced extensively by static class-agnostic prototypes or coarse Gaussians with minimal predictive loss. However, this fungibility is strictly *geometry-constrained* (orientation and feature coordinates matter) and *diversity-constrained* (token-to-token independence is required to prevent spatial attention collapse).
- **Primary Contributions**:
  1. Formalization of the Patch Content Fungibility interventionist paradigm.
  2. Identification of a coherent late-depth fungibility window (Blocks 7–10) across diverse models.
  3. Falsification of activation-norm and random-fill explanations via geometric controls.
  4. Causal demonstration of the necessity of spatial token diversity under complete stream replacement.
  5. Cross-family generalization across Supervised (DeiT, AugReg) and Self-Supervised (DINOv2) regimes.
- **Figure/Table Reference**: Figure 1 (Conceptual Diagram: Baseline vs. Zero vs. Aligned Replacement vs. Collapsed Shared Diversity).

---

### Section 2: Related Work & Conceptual Positioning
- **Core Positioning**: Delineating causal patch substitution from existing paradigms.
  - *Token Pruning & Merging (DynamicViT, ToMe)*: Pruning discards tokens to reduce FLOPs; we keep the sequence length $T$ constant and investigate representation necessity.
  - *Activation Attribution & Causal Tracing*: Attribution measures sensitivity of an existing forward pass; we perform synthetic drop-in substitutions to establish computational equivalence classes.
  - *Register Tokens*: Background artifacts identified in unregularized ViTs; our findings replicate in models without register tokens (ViT-B AugReg, DINOv2 no-register).
  - *Information Routing & Residual Propagation*: Contextualizing how residual streams aggregate global context into the CLS token while maintaining patch stream compatibility.
- **Caveat**: Explicitly note that our study is mechanistic; we do not propose a latency-reduction method.

---

### Section 3: The Patch Content Fungibility Framework
- **Methodology & Mathematical Formulation**:
  - Formulation of block forward: $h_{l+1} = \text{Block}_{l+1}(h_l)$, where $h_l = [h_{l,\text{cls}}, h_{l,1}, \dots, h_{l,N}]$.
  - Masking operator: $M \in \{0, 1\}^N$, defining replacement fraction $f \in \{0.25, 0.50, 0.75, 1.00\}$.
  - Surrogate replacements:
    - Destructive null: Zero ($h_t \to 0$).
    - Static Prototype: Calibration centroid $\mu_l = \mathbb{E}_{\text{calib}}[h_{l,t}]$.
    - Distributional Surrogate: Diagonal Gaussian $\mathcal{N}(\mu_l, \text{diag}(\sigma_l^2))$.
  - Metrics: True-class logit margin $m = z_y - \max_{j \ne y} z_j$, Margin damage $\Delta m = m_{\text{clean}} - m_{\text{intervention}}$, Damage Recovery Fraction $R = \frac{\Delta m_{\text{zero}} - \Delta m_{\text{replacement}}}{\Delta m_{\text{zero}}}$.
- **Experimental Integrity**:
  - Strict sample disjointness: $N_{\text{calib}}=1,000$, $N_{\text{eval}}=1,000$, zero image overlap.
  - Zero label or gradient leakage. Pretrained models strictly frozen.

---

### Section 4: Emergence of Late-Layer Content Fungibility
- **Exact Claim**: Content fungibility is not present in early layers, but emerges systematically across intermediate blocks into a robust late-layer window (Blocks 7–10).
- **Supporting Evidence**:
  - Depth sweeps across Depths $\{5, 7, 8, 9, 10\}$ at 25% replacement.
  - Early blocks (Depth 5) exhibit high sensitivity to replacement.
  - Late blocks exhibit high Zero sensitivity combined with $>81\% - 97\%$ Centroid/Gaussian recovery.
  - DeiT-Tiny: 93.8% recovery at Depth 8 ($d_z = 0.466$).
  - DeiT-Small: 97.5% recovery at Depth 8 ($d_z = 0.652$).
  - ViT-B: 81.8% recovery at Depth 7 ($d_z = 0.391$).
  - DINOv2: 93.4% recovery at Depth 9 ($d_z = 1.516$).
- **Figure/Table Reference**: Figure 2 (Depth Emergence curves: Accuracy and Logit Margin vs. Layer Depth).
- **Caveat**: The exact peak advantage depth varies between Depth 7 and Depth 9 across models, reflecting differences in depthwise information accumulation.

---

### Section 5: Feature-Space Geometry Constrains Valid Replacements
- **Exact Claim**: Content fungibility is not an artifact of arbitrary non-zero fill or coarse norm matching; valid surrogate activations must align with the coordinate structure and orientation of late-layer feature space.
- **Supporting Evidence**:
  - Coordinate Permutation $\pi(\mu_8)$: Preserves vector $L_2$ norm, scalar mean, and scalar variance, but destroys feature coordinate correspondence $\to$ causes catastrophic collapse across models.
  - Sign Flip $-\mu_8$: Preserves magnitude but inverts direction $\to$ collapses accuracy to near zero ($0.1\%$ in DINOv2, $27.9\%$ in ViT-B, $0.1\%$ in DeiT-Small).
  - Continuous Cosine Alignment Sweep (V0.7): Recovery scales monotonically with cosine similarity to the true centroid.
- **Figure/Table Reference**: Figure 3 (Geometry Controls: Centroid vs. Coordinate Permutation vs. Sign Flip across replacement fractions 25%, 50%, 75%).
- **Caveat**: Geometry constraints become progressively more severe as the replacement fraction increases from 25% to 75%.

---

### Section 6: Token-to-Token Diversity under Complete Stream Replacement
- **Exact Claim**: When 100% of spatial patch tokens are replaced simultaneously, token-to-token diversity is causally required; broadcasting identical tokens collapses downstream attention, while independent diversity rescues computation.
- **Supporting Evidence**:
  - Shared Gaussian (broadcast single sample per image) vs. Independent Gaussian (independent sample per patch) with matched marginal distribution and matched perturbation energy $E_{\text{full}}$.
  - DeiT-Tiny: Shared $1.02\% \to$ Independent **$26.34\%$** (+25.32 pp gain, $d_z = 1.142$).
  - DeiT-Small: Shared $9.28\% \to$ Independent **$46.36\%$** (+37.08 pp gain, $d_z = 1.341$).
  - ViT-B AugReg: Shared $2.94\% \to$ Independent **$14.68\%$** (+11.74 pp gain, $d_z = 1.005$).
  - Attention Key/Value diagnostics: Identical tokens eliminate spatial Key/Value variance, destroying dynamic attention routing in subsequent blocks.
- **Figure/Table Reference**: Figure 4 (Shared vs. Independent Gaussian Noise at 100% replacement across architectures).
- **Caveat**: In DINOv2, where the classifier directly consumes the spatial patch mean, 100% replacement floors Top-1 accuracy; the diversity benefit is manifested at the logit margin level (+1.124 margin gain, $d_z = 0.301$).

---

### Section 7: Low-Dimensional Representation Dynamics
- **Exact Claim**: Downstream computation can be partially restored under 100% replacement by variation along a single 1D axis, provided it aligns with learned principal components (Natural PC1) rather than random isotropic directions. Downstream blocks operate in a rank-preserving regime.
- **Supporting Evidence**:
  - ViT-B: Natural PC1 variation achieves **$40.10\%$ accuracy**, outperforming energy-matched Random 1D directions ($6.70\%$) by $+33.4$ pp.
  - DeiT-Tiny: Natural PC1 ($18.64\%$) outperforms Random 1D ($10.80\%$).
  - DeiT-Small: Natural PC1 ($28.08\%$) outperforms Random 1D ($20.78\%$).
  - Downstream Singular Value Decomposition (V0.9): Effective rank remains low ($r_{\text{eff}} < 2.0$) across all subsequent layers, demonstrating that computation is rank-preserving.
- **Figure/Table Reference**: Figure 5 (Natural PC1 vs. Natural PC2 vs. Energy-Matched Random 1D Direction).
- **Caveat**: Amplitude matters causally; arbitrary over-scaled rank-1 variations fail in capacity-constrained models like DeiT-Tiny.

---

### Section 8: Cross-Architecture & Training-Regime Generalization
- **Exact Claim**: The phenomenon of patch content fungibility, geometric constraint, and diversity dependence is not an idiosyncrasy of DeiT distillation, but generalizes broadly to large supervised models and self-supervised foundation backbones.
- **Supporting Evidence**:
  - Full comparative analysis across DeiT-Tiny, DeiT-Small, ViT-B AugReg, and DINOv2 ViT-S/14.
  - Evaluation of the dual readout structure in DINOv2 ($[\text{CLS} \ ; \ \text{mean}(\text{Patches})]$).
- **Figure/Table Reference**: Master Table 1 (Canonical Cross-Family Synthesis Table).
- **Caveat**: Differences in classifier head topologies modulate the observable manifestation of token interventions on top-1 accuracy vs. logit margins.

---

### Section 9: Discussion, Limitations & Future Directions
- **Mechanistic Implications**:
  - What does the patch stream do in late layers? It provides a geometric and diverse scaffold for attention-mediated computation, rather than transmitting localized visual features.
  - Explains why late-layer token pruning works empirically without severe degradation.
- **Limitations**:
  - Sequence length $T$ was held constant; findings motivate, but do not directly implement, inference speedups.
  - Studied standard classification heads; dense prediction tasks (segmentation, object detection) may retain greater spatial content dependence.
- **Future Work**: Designing geometry-guided token merging and low-rank token compression routers based on our causal constraints.

---

### Section 10: Conclusion
- Summary of Level 3 supported claim.
- Final synthesis of the paper thesis: *"Late Vision Transformer patch streams become content-fungible while remaining geometry- and diversity-constrained."*
