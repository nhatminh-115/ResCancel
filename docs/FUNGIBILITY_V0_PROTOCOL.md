# PATCH CONTENT FUNGIBILITY V0: Frozen Experimental Protocol

**Document Status:** FROZEN  
**Date:** 2026-09-28  
**Repository:** `https://github.com/nhatminh-115/ResCancel`  
**Research Line:** Patch Content Fungibility V0 — Mechanistic Falsification of Depth-Wise Token Content Dependence  

---

## 1. Scientific Objective & Research Question

In standard Vision Transformers (ViT), each image is partitioned into non-overlapping spatial patches mapped to sequence tokens $h_{l,t} \in \mathbb{R}^d$ across depth $l \in \{0, \dots, 11\}$. 

This research line investigates:
> **Does downstream prediction depend on:**
> **A. The existence of an active, well-scaled activation at token slot $t$, or**  
> **B. The exact, image-specific semantic content contained in that activation?**

We formally distinguish three functional regimes:
1. **CONTENT-SPECIFIC REGIME:**  
   Replacing the original token with plausible but incorrect content (e.g., from another image) causes severe prediction degradation, comparable to complete ablation.
2. **CONTENT-FUNGIBLE REGIME (Primary Phenomenon of Interest):**  
   Zero ablation (an out-of-distribution intervention) causes substantial damage, but replacing the token with plausible wrong-image content causes minimal damage.  
   *Interpretation:* The computational slot remains causally necessary to the network, but its exact image-specific content has become partly fungible.
3. **TOKEN-IRRELEVANT REGIME:**  
   Both zero ablation and wrong-content replacement have negligible effect on prediction.  
   *Interpretation:* The patch token has become causally inert.

---

## 2. Experimental Setup & Frozen Configurations

### 2.1 Models
- **DeiT-Tiny:** `deit_tiny_patch16_224` (timm pretrained, 12 blocks, $d=192$, heads=3)
- **DeiT-Small:** `deit_small_patch16_224` (timm pretrained, 12 blocks, $d=384$, heads=6)
- **Execution Mode:** Strictly `.eval()`, weights frozen ($\theta$ immutable). Zero training or fine-tuning.

### 2.2 Dataset
- **Reproducible ImageNet-1k Validation Subset:** $N = 1,000$ images (stratified 1 per class), seed $42$.
- **Preprocessing:** Standard bicubic resize to 256, center crop to 224, standard ImageNet normalization.

### 2.3 Pre-Registered Tested Depths
Interventions are executed at the **output of Block $l$** (which serves as the input to Block $l+1$):
- **Depths:** $l \in \{2, 4, 6, 8, 10\}$ (0-indexed out of 12 blocks).
  - Depth 2: Output of Block 2, propagates through Blocks 3–11 (9 downstream blocks).
  - Depth 4: Output of Block 4, propagates through Blocks 5–11 (7 downstream blocks).
  - Depth 6: Output of Block 6, propagates through Blocks 7–11 (5 downstream blocks).
  - Depth 8: Output of Block 8, propagates through Blocks 9–11 (3 downstream blocks).
  - Depth 10: Output of Block 10, propagates through Block 11 (1 downstream block).
- *Strict Rule:* All five depths are pre-registered. No cherry-picking or searching for optimal layers.

### 2.4 Deterministic Patch Mask
- **Ablation Budget:** Exactly $25\%$ of spatial patch tokens = **49 patch tokens per image** (out of 196 total patches).
- **Mask Generation:** Fixed random subset of 49 patch indices generated with deterministic seed `2501`.
- **Mask Invariant:** The exact same 49 spatial patch positions are modified across all intervention conditions for each image.
- *Strict Rule:* No saliency, gradient, or attention-guided patch selection.

---

## 3. Evaluated Intervention Conditions

For each selected patch token position $t \in \mathcal{M}$ (where $|\mathcal{M}| = 49$):

1. **Condition A — BASELINE:**  
   Original, untouched representation $h_{l,t}$.
2. **Condition B — ZERO ABLATION:**  
   Replace $h_{l,t} \leftarrow 0$.  
   *Purpose:* Intentionally out-of-distribution removal of the token's activation energy.
3. **Condition C — LAYER MEAN:**  
   Replace $h_{l,t} \leftarrow \mu_l(i)$, where:
   $$\mu_l(i) = \frac{1}{196 - 49} \sum_{t' \notin \mathcal{M}} h_{l,t'}(i)$$
   is the spatial mean of the *non-masked* patch tokens from the *same image* at layer $l$.  
   *Strict Rule:* Original masked values are strictly excluded from the mean computation.
4. **Condition D — CROSS-IMAGE SAME-POSITION:**  
   Replace $h_{l,t}(i) \leftarrow h_{l,t}(j)$, where donor image $j \neq i$ is determined by a fixed derangement of $\{0, \dots, 999\}$ generated with seed `3501`.  
   *Properties:* Preserves layer $l$, spatial position $t$, and empirical activation statistics, but substitutes foreign image content.
5. **Condition E — CROSS-IMAGE RANDOM-POSITION:**  
   Replace $h_{l,t}(i) \leftarrow h_{l,\pi(t)}(j)$ using donor image $j \neq i$, where patch position $\pi(t)$ is permuted using seed `4501`.  
   *Properties:* Destroys both image identity and spatial correspondence while preserving empirical activation scale.
6. **Condition F — WITHIN-IMAGE SPATIAL SHUFFLE:**  
   Permute the 49 masked tokens within the *same image*: $h_{l,t}(i) \leftarrow h_{l,\sigma(t)}(i)$ where $\sigma$ is a permutation of $\mathcal{M}$ using seed `5501`.  
   *Properties:* Preserves exact image content, but scrambles spatial slot assignment.

### Invariants Across Conditions
- Exactly 49 patch positions modified.
- CLS token strictly untouched at intervention point ($h'_{l,\text{CLS}} = h_{l,\text{CLS}}$).
- Identical downstream blocks propagate modified representations to the classification head.
- Exact same 1,000 image samples in identical batch sequence.

---

## 4. Mathematical Estimands & Metrics

For an image with label $y$ and logits $z$, the true-class logit margin is:
$$m = z_y - \max_{c \neq y} z_c$$

### 4.1 Damage Measures
For any intervention condition $C \in \{\text{zero}, \text{cross}, \text{mean}, \text{rand\_pos}, \text{shuffle}\}$:
$$\text{Damage}_C(l) = m_{\text{baseline}} - m_C(l)$$
Positive damage indicates prediction degradation relative to clean baseline.

### 4.2 Primary Estimand: The Fungibility Gap
$$\text{FungibilityGap}(l) = \text{Damage}_{\text{zero}}(l) - \text{Damage}_{\text{cross}}(l) = m_{\text{cross\_same}}(l) - m_{\text{zero}}(l)$$
- **Interpretation:**
  - $\text{FungibilityGap} \approx 0$: Plausible wrong-image content is just as damaging as zero ablation $\to$ **Content-Specific**.
  - $\text{FungibilityGap} \gg 0$: Zero ablation severely damages the prediction, but plausible wrong-image content preserves it $\to$ **Content-Fungible**.
  - $\text{Damage}_{\text{zero}} \approx 0$ and $\text{Damage}_{\text{cross}} \approx 0$: Patch tokens do not influence prediction $\to$ **Token-Irrelevant**.

### 4.3 Secondary Structural Dissociations
1. **Spatial Slot Specificity:**  
   $$\Delta_{\text{spatial}}(l) = m_{\text{cross\_same}}(l) - m_{\text{cross\_rand}}(l)$$
   Tests whether cross-image substitution requires spatial alignment.
2. **Content vs. Position Specificity:**  
   $$\Delta_{\text{content\_pos}}(l) = m_{\text{cross\_same}}(l) - m_{\text{shuffle}}(l)$$
   Compares foreign content in the correct slot vs. correct content in the wrong slot.
3. **Structured Token vs. Layer Mean:**  
   $$\Delta_{\text{mean}}(l) = m_{\text{cross\_same}}(l) - m_{\text{mean}}(l)$$
   Tests whether structured patch representations are required or if an unstructured mean vector suffices.

---

## 5. Statistical Framework & Inference

- **Unit of Independence:** The image ($N = 1,000$ independent observations per model). Tokens are never pooled as independent samples.
- **Hypothesis Testing for FungibilityGap:**
  - Mean paired difference $\bar{D}$ and median paired difference;
  - 10,000-resample paired percentile bootstrap 95% Confidence Interval;
  - Paired Student's $t$-test and Wilcoxon signed-rank test;
  - Cohen's $d_z = \frac{\bar{D}}{s_D}$.
- **Accuracy Metrics:**
  - Top-1 accuracy for each condition at each depth;
  - Paired risk difference and McNemar's exact test.
- **Multiple Testing Correction:**
  - Benjamini-Hochberg False Discovery Rate (BH-FDR, $q = 0.05$) across the 5 tested depths per architecture.
  - Both raw and FDR-adjusted $p$-values reported.

---

## 6. Pre-Registered Decision Rules

### OUTCOME A — KILL (No Evidence of Content Fungibility)
Triggered if:
- Cross-image same-position replacement damages prediction approximately as much as zero ablation across tested depths; OR
- $\text{FungibilityGap}$ is negligible: Cohen's $d_z < 0.20$ and/or bootstrap 95% CI includes zero across both architectures; OR
- Any apparent effect fails to replicate across both DeiT-Tiny and DeiT-Small.

*Scientific Conclusion:* "No evidence that patch-token slots become content-fungible across depth. Accurate predictions strictly require correct image-specific content."  
*Action:* Terminate the research line.

---

### OUTCOME B — TOKEN IRRELEVANCE ONLY
Triggered if:
- At late layers, both $\text{Damage}_{\text{zero}} \approx 0$ and $\text{Damage}_{\text{cross}} \approx 0$ (margin damage $< 0.10$ and accuracy drop $< 1\%$).

*Scientific Conclusion:* "Patch tokens simply become causally inert in late layers. This reflects token irrelevance rather than active slot fungibility."  
*Action:* Reject the content fungibility hypothesis.

---

### OUTCOME C — CONTENT FUNGIBILITY
Triggered **ONLY** if there exists at least one pre-registered depth $l \in \{2, 4, 6, 8, 10\}$ where **BOTH** DeiT-Tiny and DeiT-Small satisfy:
1. Zero ablation causes meaningful degradation ($\text{Damage}_{\text{zero}} > 0.50$ logits and accuracy drop $\ge 3\%$);
2. Cross-image same-position replacement causes substantially less degradation;
3. $\text{FungibilityGap}$ bootstrap 95% CI is strictly above zero on both architectures;
4. Effect size $d_z \ge 0.20$ on both;
5. Effect survives Benjamini-Hochberg FDR correction ($p_{\text{FDR}} < 0.05$);
6. Result is not explained by simple token irrelevance.

*Scientific Conclusion:* "At this depth, the model still requires an active, structured patch-token slot, but exact image-specific content is partly fungible."  
*Action:* Justifies deeper mechanistic investigation into slot mechanics. (No router or training in V0).

---

## 7. Programmatic Validation Assertions

Before issuing any scientific verdict, the pipeline must verify:
1. Backbone weights $\theta$ verified unchanged via SHA-256 hash.
2. Identical $N = 1,000$ image IDs evaluated in all conditions.
3. Exactly 49 spatial patches modified per image ($25\%$).
4. Patch mask is strictly identical across all intervention conditions.
5. CLS token is untouched at intervention point ($h'_{l,\text{CLS}} = h_{l,\text{CLS}}$).
6. Cross-image donor $j \neq i$ strictly holds for all 1,000 images.
7. Same-position donor preserves spatial patch coordinates.
8. Random-position donor modifies spatial patch coordinates.
9. Within-image shuffle is a valid permutation of the 49 masked slots.
10. Layer mean strictly excludes original masked token representations.
11. Downstream transformer blocks remain identical across conditions.
12. All statistical tests operate on $N = 1,000$ independent image units.
13. Activation replacement L2 norms are logged and verified against original norms.
14. Report values match saved machine-readable CSVs and JSON manifest exactly.
