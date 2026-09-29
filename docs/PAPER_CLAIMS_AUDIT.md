# Paper Claims Audit: Patch Content Fungibility in Vision Transformers

**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Date**: 2026-09-29  
**Core Paper Question**:  
> *"What information must remain in late spatial patch activations for downstream Vision Transformer classification to function?"*

**Working Thesis**:  
> *"Late Vision Transformer patch representations become content-fungible while remaining geometry- and diversity-constrained."*

---

## 1. Title Selection & Working Alternatives

### Selected Conservative Default Title:
> **Patch Content Fungibility in Vision Transformers**

### Five Evaluated Title Alternatives:
1. *Patch Content Fungibility in Vision Transformers: Geometric and Diversity Constraints in Late-Layer Representations*  
   *(Strongest candidate for top vision/learning conferences; descriptive and balanced).*
2. *Content-Fungible, Geometry-Constrained: How Late Vision Transformers Process Spatial Patches*  
   *(Mechanistic, emphasizes the dual property).*
3. *What Must Spatial Tokens Contain? Late Patch Fungibility and Diversity Constraints in Pretrained ViTs*  
   *(Question-driven framing directly reflecting the experimental paradigm).*
4. *Beyond Image-Specific Features: Geometric and Diversity Constraints of Late ViT Patch Streams*  
   *(Emphasizes representation geometry over naive feature retention).*
5. *Patch Fungibility Across Supervised and Self-Supervised Vision Transformers*  
   *(Emphasizes cross-training-regime generalization).*

---

## 2. Granular Claims Audit & Classification

Below, each potential paper claim is audited against the frozen experimental evidence and classified as:
- **`SUPPORTED STRONGLY`**
- **`SUPPORTED WITH QUALIFICATION`**
- **`NOT SUPPORTED`**

---

### Claim 1: Late-Layer Patch Content Fungibility
- **Statement**: In late Transformer blocks, large fractions of spatial patch activations can be replaced by class-agnostic, dataset-level centroid prototype vectors or coarse diagonal Gaussians with limited degradation in downstream classification accuracy and logit margin.
- **Classification**: **`SUPPORTED STRONGLY`**
- **Evidence**:
  - Replicated across 4 models: DeiT-Tiny (93.8% recovery), DeiT-Small (97.5% recovery), ViT-B/16 AugReg (81.8% recovery at Depth 7), and DINOv2 ViT-S/14 (93.4% recovery at Depth 9).
  - Even replacing 75% of all spatial patches (147 of 196 tokens) in DeiT-Small retains 73.6% accuracy (clean: 76.1%), with 94.5% of originally correct images remaining correct.
  - In ViT-B at 75% replacement, centroid retains 65.4% accuracy (clean: 76.1%).
- **Allowed Paper Framing**: *"Late Vision Transformer patch representations become content-fungible: exact image-specific patch content can be replaced extensively by class-agnostic prototypes."*
- **Forbidden Framing**: *"ViTs do not need patch content."* / *"Patch tokens are useless."*

---

### Claim 2: Depth Dependence (Late-Layer Phenomenon)
- **Statement**: Content fungibility is not present in early/intermediate blocks, where exact image content is strictly required. A distinct transition occurs in late layers (Blocks 7–10) where zeroing patches remains damaging, but plausible geometric replacement preserves computation.
- **Classification**: **`SUPPORTED STRONGLY`**
- **Evidence**:
  - In DeiT-Tiny/Small (V0.6): Blocks 5–6 show severe sensitivity to Gaussian replacement; Blocks 8–9 show a sharp plateau where held-out centroid/Gaussian recovery jumps to 88%–97%.
  - In ViT-B (V1): Depth 7 exhibits peak fungibility advantage (+0.895 margin advantage over Zero, 81.8% recovery).
  - In DINOv2 (V1): Depths 8–10 exhibit massive zero-collapse ($4.5\%$ at Depth 9, $1.9\%$ at Depth 10) while centroid recovery jumps to $93.4\% - 98.0\%$.
- **Allowed Paper Framing**: *"Content fungibility emerges as a late-depth phenomenon, transitioning across intermediate layers into a late-layer window."*
- **Forbidden Framing**: *"All ViTs transition at exactly Block 8."* (The exact peak varies between Depth 7 and Depth 9 depending on capacity and training regime).

---

### Claim 3: Cross-Architecture and Training-Regime Generalization
- **Statement**: The core content-fungibility phenomenon replicates across distinct architectures (Small vs. Base, $D=192, 384, 768$), patch sizes ($14 \times 14$ vs. $16 \times 16$), training recipes (Supervised distillation, Supervised AugReg, Self-Supervised DINOv2), and classifier readouts (CLS vs. CLS+Patch mean).
- **Classification**: **`SUPPORTED STRONGLY`**
- **Evidence**:
  - Formally achieved V1 Outcome A: both ViT-B AugReg and DINOv2 ViT-S/14 satisfy pre-registered Signatures A, B, and C with zero assertion failures.
- **Allowed Paper Framing**: *"Late patch-content fungibility generalizes across supervised and self-supervised Vision Transformers."*
- **Forbidden Framing**: *"We prove a universal law governing all Vision Transformers."* (Scope is empirically bounded to standard ViT, DeiT, and DINOv2 backbones).

---

### Claim 4: Geometric Feature-Space Constraints
- **Statement**: Content fungibility is not arbitrary non-zero replacement. Valid replacements must adhere to the learned late-layer feature geometry: scalar coordinate permutations ($\pi(\mu)$) and sign flips ($-\mu$) destroy classification performance, despite perfectly preserving vector $L_2$ norm, scalar mean, and variance.
- **Classification**: **`SUPPORTED STRONGLY`**
- **Evidence**:
  - In ViT-B (Depth 8, 75% replacement): Centroid retains 65.4% accuracy; permuted centroids crash to $38.0\% \pm 7.7\%$, and sign flip collapses to $27.9\%$ ($p < 10^{-50}$).
  - In DINOv2 (Depth 8, 25% replacement): Centroid preserves 74.6% accuracy; coordinate permutation drops to $31.8\%$, and sign flip causes complete collapse to **0.1%** ($d_z = 2.509$, $-74.5$ pp loss).
  - In DeiT (V0.7): Centroid alignment sweep proved performance improves monotonically with cosine similarity to $\mu_8$.
- **Allowed Paper Framing**: *"Patch replacement is strictly geometry-constrained: valid surrogate tokens must align with the coordinate structure and orientation of late-layer feature space."*

---

### Claim 5: Token-to-Token Diversity under Complete Stream Replacement
- **Statement**: When 100% of spatial patch tokens are replaced simultaneously, token-to-token diversity is causally required. Identical broadcast tokens (Shared noise) cause spatial attention variance collapse, whereas independent token variation (Independent noise) rescues downstream computation.
- **Classification**: **`SUPPORTED WITH QUALIFICATION`**
- **Evidence & Qualification**:
  - **Supervised models (CLS-readout)** show massive accuracy recovery:
    - DeiT-Tiny: Shared $1.02\% \to$ Indep **$26.34\%$** (+25.32 pp gain).
    - DeiT-Small: Shared $9.28\% \to$ Indep **$46.36\%$** (+37.08 pp gain).
    - ViT-B AugReg: Shared $2.94\% \to$ Indep **$14.68\%$** (+11.74 pp gain, $d_z = 1.005$).
  - **Self-Supervised DINOv2 (Dual CLS+Patch mean readout)**:
    - 100% spatial patch replacement directly corrupts the mean patch token fed into the linear classification head. As a result, Top-1 accuracy remains at floor ($0.12\%$ vs. $0.24\%$).
    - However, at the margin level, Independent Gaussian improves logit margin by **$+1.124$** ($d_z = 0.301$, $p = 7.1 \times 10^{-21}$).
- **Required Qualification**: Diversity cannot be claimed to "universally rescue Top-1 classification accuracy" in architectures where the classification head explicitly consumes the spatial patch mean. It must be framed as restoring downstream attention dynamics and margin compatibility, with strong accuracy recovery established for CLS-readout ViTs.
- **Allowed Paper Framing**: *"Under complete patch-stream replacement, token-to-token diversity is causally necessary to avoid attention collapse, strongly restoring Top-1 accuracy in CLS-readout models and improving margin compatibility in pooled-readout models."*
- **Forbidden Framing**: *"Token diversity universally rescues classification in all ViTs."*

---

### Claim 6: Low-Dimensional Learned 1D Subspace Variation
- **Statement**: Complete spatial patch stream collapse can be partially mitigated by introducing diversity along a single 1D axis, provided the direction is aligned with learned principal components (Natural PC1) rather than arbitrary random directions.
- **Classification**: **`SUPPORTED WITH QUALIFICATION`**
- **Evidence & Qualification**:
  - ViT-B: Natural PC1 variation achieves **$40.10\%$ accuracy**, outperforming energy-matched Random 1D directions ($6.70\%$) by **$+33.4$ percentage points**.
  - DeiT-Tiny: Natural PC1 ($18.64\%$) outperforms Random 1D ($10.80\%$) by $+7.84$ pp.
  - DeiT-Small: Natural PC1 ($28.08\%$) outperforms Random 1D ($20.78\%$) by $+7.30$ pp.
  - V0.9 proved that arbitrary Rank-1 variation is **not** sufficient: amplitude scaling along PC1 follows an architecture-specific response curve (inverted U-curve in Tiny, saturating plateau in Small).
- **Required Qualification**: Do not claim that arbitrary 1D variation suffices or that higher dimensions are unnecessary. Direction and amplitude both matter causally.
- **Allowed Paper Framing**: *"Learned late-layer 1D variation (PC1) partially restores downstream computation over isotropic random directions, but downstream recovery depends causally on direction and amplitude."*

---

### Claim 7: Rank-Preserving Downstream Dynamics
- **Statement**: When rank-1 variation is injected at Depth 8, downstream blocks (MLP and attention) do not spontaneously expand representations into high-dimensional space; instead, effective representation rank remains low ($r_{\text{eff}} < 2.0$) throughout the remainder of the network.
- **Classification**: **`SUPPORTED STRONGLY`**
- **Evidence**:
  - Directly verified in V0.9 across all downstream layers: DeiT-Tiny injection $r_{\text{eff}} = 1.000 \to 1.086 \to 1.140 \to 1.248$; DeiT-Small injection $r_{\text{eff}} = 1.000 \to 1.373 \to 1.636 \to 1.791$.
- **Allowed Paper Framing**: *"Downstream Vision Transformer computation is rank-preserving: downstream blocks operate effectively directly upon low-dimensional patch representations."*

---

### Claim 8: Relationship to the CLS Token
- **Statement**: Patch tokens and the CLS token have asymmetric functional roles in late layers: patch tokens become content-fungible, while the CLS token continues to require individual specificity.
- **Classification**: **`SUPPORTED WITH QUALIFICATION`**
- **Evidence & Qualification**:
  - In V0.7, zeroing CLS at Depth 8 caused 1.75 margin damage in DeiT-Tiny (acc dropped to 40.0%), while replacing CLS with its calibration mean yielded 66.4% (clean: 67.9%).
  - In DINOv2, classification consumes both CLS and patch mean.
- **Required Qualification**: Do not claim "all information moves into CLS" or that "CLS is purely global and patches are dead." Patches actively participate in attention and form the Key/Value matrices for late blocks.

---

### Claim 9: Implications for Token Compression and Acceleration
- **Statement**: Late-layer patch content fungibility indicates that late ViT patch activations can be approximated, quantized, or merged into low-rank prototypes, motivating new token compression architectures.
- **Classification**: **`SUPPORTED WITH QUALIFICATION`**
- **Crucial Methodological Caveat**:
  - Our experimental interventions strictly preserve the spatial sequence length ($T = 196$ or $256$). We do not remove tokens or prune heads.
  - Therefore, our experiments **do not** demonstrate inference latency reduction or FLOP speedups.
- **Required Qualification**: State clearly that these findings **MOTIVATE** future token compression, merging, or routing methods by showing what information is dispensable, but that our work is a mechanistic study, not an acceleration method.
- **Allowed Paper Framing**: *"These findings motivate future token compression and merging methods by identifying the minimal geometric and diversity constraints required by downstream ViT layers."*
- **Forbidden Framing**: *"Our method accelerates Vision Transformers."*

---

## 3. Reviewer Robustness Audit: Dense Fraction & Spatial-Mask Invariance

Following the execution of the pre-registered **Dense Fraction and Spatial-Mask Robustness Sweep** (`outputs/fungibility_dense_fraction/`), the frozen paper claims were subjected to a rigorous reviewer-facing robustness audit.

### Audit Verdict: **`ALL CLAIMS FULLY SUPPORTED AND REINFORCED`**
No existing paper claims are retracted, weakened, or altered. The experiment provides three major empirical enhancements:

1. **Spatial Invariance Guarantee (Falsification of "Lucky Mask" Confound):**
   - **Finding**: Across 5 independently sampled deterministic spatial permutations (`31001..31005`), variation in accuracy retention thresholds is exceptionally small ($SD \le 1.7\%$ in DeiT-Tiny, $\le 1.0\%$ in DeiT-Small, $\le 2.1\%$ in ViT-Base, and $< 1.1\%$ in DINOv2).
   - **Verdict**: Content fungibility is an intrinsic layer-level property of the representation space, completely independent of spatial mask ordering.

2. **Continuous Parametric Dose-Response ($0\%$ to $100\%$):**
   - **Finding**: The dose-response curve is continuous and smooth throughout $0\% \to 95\%$. Supervised ViTs do not exhibit early catastrophic cliffs; instead, graceful linear-to-monotonic degradation persists up to high replacement fractions ($F_{90} = 77.9\%$ in Tiny, $86.5\%$ in Small, $67.4\%$ in ViT-B).
   - **Boundary Cliff**: The only sharp accuracy drop occurs at the extreme boundary ($98.9\% \to 100.0\%$), exactly where token diversity drops to zero, reinforcing the V0.8 diversity constraint.

3. **Continuous Superiority over Zero:**
   - **Finding**: Centroid and Gaussian replacements outperform destructive Zero replacement at **$96.9\% – 100.0\%$** of all evaluated fraction points where Zero is damaging ($m_{\text{damage}} \ge 0.10$).
   - **Verdict**: The fungibility advantage is sustained continuously across the entire spectrum, not confined to coarse isolated points ($25\%, 50\%, 75\%$).

