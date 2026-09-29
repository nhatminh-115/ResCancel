# Related Work & Conceptual Positioning: Patch Content Fungibility

**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Date**: 2026-09-29  

---

## 1. Delineation from Related Literature Paradigms

Understanding what representation properties must be preserved inside deep neural networks is an active area of deep learning research. To establish the precise scientific contribution of this work, we systematically distinguish **Patch Content Fungibility** from adjacent literature paradigms.

```
+------------------------------------+-------------------------------------------+-----------------------------------------------+
| Research Paradigm                  | Central Question Asked                    | Primary Methodological Manipulation           |
+------------------------------------+-------------------------------------------+-----------------------------------------------+
| Token Pruning                      | Which tokens can be safely dropped to     | Sequence length reduction:                    |
| (DynamicViT, EViT, SPViT)          | minimize FLOPs and latency?               | T_in -> T_out (T_out < T_in)                  |
+------------------------------------+-------------------------------------------+-----------------------------------------------+
| Token Merging                      | How can similar tokens be aggregated to   | Bipartite matching & weighted averaging:      |
| (ToMe, Token Pooling)              | preserve visual information with fewer Ts?| T_in -> T_out (T_out < T_in)                  |
+------------------------------------+-------------------------------------------+-----------------------------------------------+
| Early Exiting                      | Can easy samples skip late computation?   | Sub-network truncation;                       |
| (MSDNet, Dynamic ViTs)             |                                           | intermediate classifier readout               |
+------------------------------------+-------------------------------------------+-----------------------------------------------+
| Activation Attribution             | Which input pixels or hidden activations  | Gradient backprop, integrated gradients,      |
| (Attention Rollout, Causal Tracing)| are most causally important for y_hat?    | or causal mediation on existing activations   |
+------------------------------------+-------------------------------------------+-----------------------------------------------+
| Register Token Analysis            | Are extreme outlier background tokens     | Architectural inspection and intervention     |
| (Darcet et al., 2023)              | performing global scratchpad computation? | on specialized register tokens                |
+------------------------------------+-------------------------------------------+-----------------------------------------------+
| Patch Content Fungibility          | What information must remain in ordinary  | Causal synthetic substitution holding         |
| (This Work)                        | spatial patch streams for classification? | sequence length constant: T_in == T_out       |
+------------------------------------+-------------------------------------------+-----------------------------------------------+
```

---

### 1.1 Contrast with Token Pruning (DynamicViT, EViT, SPViT)
- **Token Pruning Literature**: Methods such as DynamicViT (Rao et al., 2021), EViT (Liang et al., 2022), and SPViT (Kong et al., 2022) train routing modules or compute saliency scores to permanently discard uninformative spatial tokens, shortening the sequence length $T$ across depth to achieve computational speedup.
- **Fundamental Distinction**:
  1. *Sequence Length*: Token pruning modifies sequence length ($T < 196$). Patch fungibility strictly preserves the sequence length ($T = 196$ or $256$).
  2. *Objective*: Pruning is an engineering optimization for inference efficiency. Fungibility is a scientific inquiry into representation necessity.
  3. *Mechanism Revealed*: Pruning works by removing tokens whose absence does not distort self-attention. Our work reveals a complementary mechanism: downstream attention blocks do not need image-specific features from spatial patches, but they *do* require a non-zero, geometry-aligned, and spatially diverse token stream to maintain dynamic Key/Value rank.

---

### 1.2 Contrast with Token Merging (ToMe)
- **Token Merging Literature**: Token Merging (ToMe; Bolya et al., 2023) uses bipartite soft matching to merge redundant tokens via weighted averaging, progressively compressing visual representations without retraining.
- **Fundamental Distinction**:
  - ToMe relies on representation *similarity* to cluster and average tokens that describe similar visual content.
  - In contrast, Patch Fungibility demonstrates that *synthetic, class-agnostic prototypes* ($\mu_l$) derived from an external calibration dataset can replace up to 75% of spatial tokens without clustering or image-specific aggregation. The downstream model functions effectively even when spatial tokens contain *no image-specific information at all*.

---

### 1.3 Contrast with Early Exiting & Dynamic Depths
- **Early Exiting Literature**: Works on early exit architectures attach auxiliary classification heads to intermediate layers, allowing high-confidence predictions to exit early without executing late blocks.
- **Fundamental Distinction**:
  - Early exiting stops execution at layer $l$, skipping layers $l+1 \dots L$.
  - Patch Fungibility intervenes *at layer $l$* and explicitly executes all subsequent Transformer blocks $l+1 \dots L$ and the final classifier head. It tests how subsequent blocks process corrupted vs. surrogate spatial tokens.

---

### 1.4 Contrast with Activation Attribution and Causal Tracing
- **Attribution Literature**: Attention rollout (Abnar & Zuidema, 2020), Integrated Gradients (Sundararajan et al., 2017), and causal tracing (Meng et al., 2022) analyze the sensitivity of an existing forward pass by inspecting attention weights or computing gradients.
- **Fundamental Distinction**:
  - Attribution methods quantify which existing features correlate with or mediate a prediction.
  - Patch Fungibility is an *interventionist substitution study*: by substituting synthetic activations from controlled geometric distributions (centroids, Gaussians, coordinate bijections, isotropic vectors, PC axes), it maps out the boundaries of the model's *computational equivalence class*.

---

### 1.5 Contrast with Vision Transformer Register Tokens
- **Register Token Literature**: Darcet et al. (2023) demonstrated that unregularized Vision Transformers (especially DINO and DeiT) tend to recruit tokens in low-information background patches as "registers"—high-norm artifact vectors that store global computation.
- **Fundamental Distinction**:
  - Our findings are not an artifact of register tokens.
  - In our protocol:
    1. ViT-B AugReg has **no register tokens** ($196$ patches + $1$ CLS token).
    2. The DINOv2 model evaluated is explicitly the official **NO-REGISTER** checkpoint (`dinov2_vits14_lc`, not `reg`).
    3. Content fungibility is demonstrated across random spatial subsets (25%, 50%, 75%), not isolated background artifact tokens. Ordinary spatial patch representations themselves become content-fungible.

---

### 1.6 Contextualizing with Representation Redundancy & CLS Aggregation
- **Information Flow in ViTs**: Several empirical studies have noted that representations across late Transformer layers become increasingly similar (cosine similarity across consecutive layers increases, sometimes termed representation collapse or uniformity; e.g., Raghu et al., 2021; Zhou et al., 2021).
- **Our Advance**:
  - Prior work observed that representations stabilize in late layers, often conjecturing that the CLS token has already extracted all necessary global visual semantics.
  - However, passive observation could not distinguish between two causal hypotheses:
    - *Hypothesis 1 (Passive Transmittance)*: Spatial patch activations continue to be causally necessary because their individual visual content is repeatedly consulted by remaining attention blocks.
    - *Hypothesis 2 (Content Fungibility with Geometric Constraints)*: Exact image-specific visual content in spatial patches is dispensable, but the spatial patch stream continues to provide a necessary geometric and diverse coordinate scaffold for multi-head attention.
  - Our experiments decisively falsify Hypothesis 1 and confirm Hypothesis 2.

---

## 2. Framing of Paper Contributions

To ensure academic rigor and avoid over-claiming:
1. **Mechanistic Study, Not an Acceleration Algorithm**:
   - The paper must explicitly clarify that our experiments do not achieve inference acceleration because the spatial sequence length is kept constant ($T=196$ or $256$).
   - The appropriate claim is that our findings **motivate and provide principled design rules** for future token compression, quantization, and merging algorithms by establishing the exact geometric and diversity constraints that downstream attention layers enforce.
2. **Causal Substitution Study of Ordinary Spatial Patches**:
   - The contribution is framed as a rigorous causal intervention establishing that late spatial patch streams become content-fungible while remaining geometry- and diversity-constrained.
3. **Avoidance of Absolute Novelty Claims**:
   - We do not claim to be the first to note that late layers aggregate global information.
   - We claim the specific mechanistic discovery that *ordinary spatial patch activations can be replaced by class-agnostic prototypes without significant predictive damage, provided feature-space geometry and token diversity are preserved*.
