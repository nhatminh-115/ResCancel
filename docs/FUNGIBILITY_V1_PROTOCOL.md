# Patch Fungibility V1 Protocol: Cross-Model & Training-Regime Generalization

## 1. Scientific Objective

Experiments V0 through V0.9 in `ResCancel` established that late Vision Transformer patch representations in DeiT-Tiny and DeiT-Small become strongly content-fungible:
- Exact image-specific patch content can be extensively substituted at late layers (Blocks 7–9).
- Arbitrary non-zero fill (coordinate permutation, sign inversion, low-cosine vectors) fails severely; replacement must align with late-layer feature geometry.
- At 100% spatial patch replacement, token-to-token diversity is strictly causal: identical patch streams collapse downstream computation to near-zero accuracy due to attention Key/Value variance collapse, whereas independent stochastic replacement restores substantial predictive accuracy.
- These phenomena are not explained by activation-norm artifacts, same-set calibration leakage, or variance-scaling confounds.

### The Remaining Threat: Model Specificity
The foundational findings were discovered on the **DeiT** family (DeiT-Tiny, DeiT-Small), trained with the DeiT-1 distillation/supervision recipe.
The central scientific question of **Patch Fungibility V1** is:
> **Does late-layer patch content fungibility with geometric and diversity constraints generalize across architecture scales, training regimes (supervised vs. self-supervised), and classifier readout architectures?**

V1 is strictly a **replication and generalization experiment**.
- No architecture modification.
- No model training or fine-tuning.
- No method, router, or gradient optimization.

---

## 2. Models & Architectures

### 2.1 Model A — Supervised Vanilla ViT
- **Checkpoint**: `vit_base_patch16_224.augreg_in1k` via `timm`.
- **Architecture**: ViT-Base/16 (12 Transformer blocks, embedding dimension $D=768$, 12 attention heads, 196 spatial patches at $224 \times 224$).
- **Training Recipe**: Supervised AugReg training (ImageNet-1k).
- **Readout**: Pre-LN + Head directly on final CLS token.
- **Preprocessing**: Official AugReg transform: Bicubic interpolation, crop_pct 0.9, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5).

### 2.2 Model B — Self-Supervised DINOv2
- **Checkpoint**: `dinov2_vits14_lc` with `layers=1, pretrained=True` via official Meta PyTorch Hub.
- **Architecture**: ViT-Small/14 (12 Transformer blocks, embedding dimension $D=384$, 6 attention heads, patch size 14, $16 \times 16 = 256$ spatial patches at $224 \times 224$).
- **Variant**: Strict **NO-REGISTER** model (`num_register_tokens = 0`).
- **Training Recipe**: Self-supervised DINOv2 pretraining + official frozen ImageNet-1k linear head (`layers=1`).
- **Readout Mechanism**:
  The official classifier head directly consumes:
  $$\text{Input}_{\text{head}} = [\text{CLS}_{\text{norm}} \,\|\, \text{mean}(\text{Patch}_{\text{norm}})] \in \mathbb{R}^{2D}$$
  followed by the pretrained linear layer $\mathbb{R}^{2D} \to \mathbb{R}^{1000}$.
  *Scientific Significance*: DINOv2's classifier head directly pools spatial patch representations. Patch interventions therefore affect both downstream multi-head self-attention and the final linear readout.
- **Preprocessing**: Official DINOv2 transform: Resize to 256 (bicubic), CenterCrop 224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225).

---

## 3. Implementation Verification & Numerical Tolerance

Before executing any scientific interventions:
- For both models, an intervention-compatible manual block-by-block forward execution path is implemented.
- Each manual forward path is verified against the untouched official model forward on at least 64 evaluation images across multiple batches.
- **Strict Verification Gate**:
  $$\max |\text{Logits}_{\text{manual}} - \text{Logits}_{\text{official}}| < 10^{-5}$$
  Prediction agreement must be **exact 100%**.
  If manual forward fails verification, execution halts immediately.

---

## 4. Datasets & Nested Mask Specifications

- **Datasets**: Strictly disjoint subsets of ImageNet-1k validation set from V0.6–V0.9:
  - Calibration: $N=1,000$ images (stratified 1 per class).
  - Evaluation: $N=1,000$ images.
  - Zero image overlap. Calibration statistics never observe evaluation images.
- **Nested Spatial Replacement Masks**:
  Deterministic permutation seed: `21001`.
  - **ViT-B/16** ($N_{\text{patch}} = 196$):
    - 25% = 49 tokens
    - 50% = 98 tokens
    - 75% = 147 tokens
    - 100% = 196 tokens
  - **DINOv2-S/14** ($N_{\text{patch}} = 256$):
    - 25% = 64 tokens
    - 50% = 128 tokens
    - 75% = 192 tokens
    - 100% = 256 tokens
- **CLS Token**: Index 0 remains strictly original and untouched in all primary interventions.

---

## 5. Experimental Phases

### Phase 1: Late-Depth Fungibility Sweep (Depths 5, 7, 8, 9, 10)
- **Intervention**: 25% patch replacement at block outputs $l \in \{5, 7, 8, 9, 10\}$.
- **Conditions**:
  1. Clean baseline.
  2. Zero replacement: $h'_t = 0$.
  3. Held-out calibration centroid: $h'_t = \mu_l$.
  4. Held-out diagonal Gaussian: $h'_t \sim \mathcal{N}(\mu_l, \text{diag}(\sigma_l^2))$ (Seeds: `22001, 22002, 22003`).
- **Question**: Does a coherent late-layer window exist where Zero remains damaging while Centroid/Gaussian preserves accuracy?

### Phase 2: Fraction & Geometry Controls at Standardized Depth 8
- **Intervention Depth**: Depth 8 (output Block 8 / input Block 9).
- **Fractions**: 25%, 50%, 75%.
- **Conditions**:
  1. Clean.
  2. Zero.
  3. Calibration Centroid ($\mu_8$).
  4. Diagonal Gaussian (Seeds `22001..22003`).
  5. Coordinate-Permuted Centroid: Permutation $\pi \in S_D$ applied to $\mu_8$ coordinates (Seeds `23001, 23002, 23003`). Preserves exact values, scalar mean, variance, and L2 norm, but destroys coordinate correspondence.
  6. Sign-Flipped Centroid: $-\mu_8$.

### Phase 3: Token Diversity Causal Test at 100% Replacement
- **Intervention Depth**: Depth 8, 100% spatial patch replacement.
- **Conditions**:
  1. Static Centroid: $h'_t = \mu_8$ for all $t$.
  2. Shared Gaussian Noise: Sample one $\epsilon \sim \mathcal{N}(0, \text{diag}(\sigma_8^2))$ per image, broadcast to all spatial patches ($K=1$).
  3. Independent Gaussian Noise: Independent $\epsilon_t \sim \mathcal{N}(0, \text{diag}(\sigma_8^2))$ per patch ($K=N_{\text{patch}}$).
  4. Isotropic Independent Noise: $\epsilon_t \sim \mathcal{N}(0, \frac{E_{\text{full}}}{D} I_D)$.
  - Seeds: `24001, 24002, 24003, 24004, 24005`.
- **Causal Question**: Does Independent diversity strongly outperform Shared noise under identical marginal distribution and total variance?

### Phase 4: 1D Subspace Direction Control at 100% Replacement
- **Intervention Depth**: Depth 8, 100% spatial patch replacement.
- **Conditions**:
  1. Static Centroid ($\mu_8$).
  2. Natural PC1: $h'_t = \mu_8 + z_t \sqrt{\lambda_1} v_1$.
  3. Natural PC2: $h'_t = \mu_8 + z_t \sqrt{\lambda_2} v_2$.
  4. Matched Random 1D Direction: $h'_t = \mu_8 + z_t \sqrt{\lambda_1} u$, where $\|u\|_2 = 1.0$.
  - Seeds: `25001, 25002, 25003`.

### Phase 5: Secondary Adapted-Depth Confirmation (If Triggered)
If Depth 8 is not a model's optimal fungibility window, deterministic selection rule evaluates depths $\{7, 8, 9, 10\}$:
$$\text{Fungibility Advantage} = \text{Damage}_{\text{zero}} - \text{Damage}_{\text{centroid}}$$
subject to $\text{Damage}_{\text{zero}} \ge 0.10$.
If optimal depth $d^* \neq 8$, evaluate Phase 2 and 3 at $d^*$ as Secondary Depth-Adapted Confirmation.

---

## 6. Pre-Registered Signatures & Decision Rules

### Core Signatures (Evaluated Per Model)
- **Signature A — Content Fungibility**:
  At at least one late depth $l \in \{7, 8, 9\}$:
  $\text{Damage}_{\text{zero}} \ge 0.10$ AND Centroid or Gaussian recovers $\ge 50\%$ of Zero margin damage:
  $$\text{Recovery} = \frac{\text{Damage}_{\text{zero}} - \text{Damage}_{\text{replacement}}}{\text{Damage}_{\text{zero}}} \ge 0.50$$
  with paired bootstrap 95% CI for $(\text{Margin}_{\text{replacement}} - \text{Margin}_{\text{zero}})$ strictly $> 0$.
- **Signature B — Geometric Constraint**:
  At Depth 8 (or secondary adapted depth), correct Centroid outperforms Coordinate-Permuted OR Sign-Flipped Centroid with Cohen's $d_z \ge 0.20$ and statistically significant margin difference.
- **Signature C — Token Diversity Requirement**:
  At 100% replacement, Independent Gaussian outperforms Shared Gaussian with Cohen's $d_z \ge 0.20$ and accuracy gain $\ge 5.0$ percentage points (or equivalently strong margin difference).

### Primary V1 Generalization Verdict
- **OUTCOME A — BROAD GENERALIZATION**:
  BOTH ViT-B AugReg AND DINOv2 ViT-S/14 satisfy Signature A, Signature B, and Signature C.
  *Scientific Conclusion: "Late-layer patch content fungibility with geometric and diversity constraints generalizes across supervised and self-supervised Vision Transformers."*
  $\implies$ **STOP experimental escalation. Move directly to paper writing.**
- **OUTCOME B — PARTIAL GENERALIZATION**:
  One model satisfies all core signatures while the other shows a partial subset, or content fungibility generalizes but geometry/diversity constraints differ.
- **OUTCOME C — DEIT-FAMILY PHENOMENON**:
  Neither model shows convincing late content fungibility despite verified implementation and non-zero damage.
- **OUTCOME D — UNINTERPRETABLE**:
  Manual forward verification failure or implementation defect.

### Supported Paper Claim Levels
- **Level 1**: *"DeiT patch content is fungible."*
- **Level 2**: *"Supervised ViTs show late patch-content fungibility."*
- **Level 3**: *"Late patch-content fungibility generalizes across supervised and self-supervised ViTs."*

---

## 7. Programmatic Assertions

1. All pretrained weights strictly frozen.
2. Manual forward path reproduces official forward ($< 10^{-5}$ logit diff, 100% prediction match).
3. Correct model-specific preprocessing used (AugReg vs. DINOv2).
4. Exact existing calibration ($N=1,000$) and evaluation ($N=1,000$) image identities reused.
5. Zero overlap between calibration and evaluation sets.
6. Calibration statistics never observe evaluation images.
7. CLS token strictly untouched at injection.
8. Spatial patch counts exact: ViT-B = 196, DINOv2 = 256.
9. Nested fraction masks exact for all fractions.
10. All requested depths evaluated.
11. Coordinate permutations are true bijections.
12. Shared-noise vectors are identical across all patches within each image.
13. Independent-noise vectors are sampled independently per patch.
14. Shared and independent distributions and expected norms match.
15. DINOv2 official linear head remains frozen and untouched.
16. DINOv2 final readout reproduces official CLS + patch-mean concatenation.
17. Seeds match protocol specifications.
18. Exactly $N=1,000$ independent evaluation images per model.

---

## 8. Artifacts & Deliverables

- `docs/FUNGIBILITY_V1_PROTOCOL.md`
- `docs/FUNGIBILITY_V1_REPORT.md`
- `outputs/fungibility_v1/`:
  - `model_metadata.json`
  - `manual_forward_validation.json`
  - `calibration_statistics.npz`
  - `vitb_depth_results.csv`
  - `dinov2_depth_results.csv`
  - `vitb_fraction_results.csv`
  - `dinov2_fraction_results.csv`
  - `vitb_geometry_results.csv`
  - `dinov2_geometry_results.csv`
  - `vitb_diversity_results.csv`
  - `dinov2_diversity_results.csv`
  - `vitb_1d_results.csv`
  - `dinov2_1d_results.csv`
  - `vitb_image_results.parquet`
  - `dinov2_image_results.parquet`
  - `experiment_manifest.json`
- `figures/fungibility_v1/`:
  - `depth_generalization.png`
  - `content_fungibility_across_models.png`
  - `geometry_controls_across_models.png`
  - `shared_vs_independent_across_models.png`
  - `replacement_fraction_generalization.png`
  - `learned_vs_random_1d.png`
