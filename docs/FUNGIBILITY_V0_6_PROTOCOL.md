# Patch Content Fungibility V0.6: Protocol Specification
## Held-Out Generalization, Depth-Transition Validation, and Budget Scaling

**Author:** DeepMind Antigravity Team  
**Date:** September 28, 2026  
**Repository:** [https://github.com/nhatminh-115/ResCancel](https://github.com/nhatminh-115/ResCancel)  
**Status:** FROZEN BEFORE EXECUTION  

---

## 1. Scientific Context & Motivation

In Patch Content Fungibility V0.5, we falsified the hypothesis that the Block-8 recovery is merely an activation-scale artifact (H2):
- **Norm-Matched Random Sphere** vectors matching original token $L_2$ norm fail catastrophically (DeiT-Tiny Acc collapses to $31.2\%$, DeiT-Small to $63.4\%$).
- **Feature-Dimension Shuffling** collapses performance to $38.5\%$ (Tiny) and $58.2\%$ (Small), proving that channel coordinate alignment is strictly necessary.
- **Layer-Statistics Gaussian** samples drawn from per-feature $\mathcal{N}(\mu_d, \sigma_d^2)$ completely rescue performance ($68.8\%$ on Tiny, $77.9\%$ on Small), matching or outperforming foreign donor tokens.

However, three critical limitations prevent drawing broad scientific conclusions:
1. **In-Sample Statistics:** Gaussian moments ($\mu_d, \sigma_d$) were estimated from the non-masked tokens of the *same* 1,000 images on which they were evaluated.
2. **Depth Isolation:** Only Block 8 was tested under the new Gaussian null. Does this tolerance reflect a continuous depth transition or an isolated layer anomaly?
3. **Single Fraction:** The phenomenon was demonstrated solely at a 25% replacement fraction (49/196 tokens). Does it hold at 10% (20 tokens) and 50% (98 tokens)?

### Primary Scientific Questions (V0.6):
- **Q1. Held-Out Generalization:** Do distribution-matched Gaussian replacements still rescue the model when their activation statistics are estimated on images completely disjoint from the evaluation images?
- **Q2. Depth Specificity:** Is tolerance to distribution-matched synthetic tokens a genuine depth transition across intermediate/late layers ($l \in \{5, 6, 7, 8, 9, 10\}$), and does it depend on depth-specific distributions?
- **Q3. Budget Scaling:** Does the phenomenon persist when replacing 10%, 25%, and 50% of spatial patch tokens?
- **Q4. Architecture Replication:** Does the depth-transition trajectory replicate across DeiT-Tiny and DeiT-Small?

---

## 2. Experimental Setup & Controls

- **Models:**
  - `deit_tiny_patch16_224` ($D = 192$, 12 blocks)
  - `deit_small_patch16_224` ($D = 384$, 12 blocks)
  - Public checkpoints from `timm`, strictly frozen weights ($\nabla_\theta = 0$, eval mode).
- **Disjoint Dataset Splits (ImageNet-1k Validation Set, 50,000 images):**
  - **Calibration Set ($N_{\text{calib}} = 1,000$):** Deterministic seed `9101`, stratified 1 image per class. Used *exclusively* to estimate layer activation statistics.
  - **Evaluation Set ($N_{\text{eval}} = 1,000$):** Deterministic seed `9201`, stratified 1 image per class, sampled from the remaining 49 images per class.
  - **Strict Disjointness:** $\text{IDs}_{\text{calib}} \cap \text{IDs}_{\text{eval}} = \emptyset$. Zero file overlap, zero activation leakage.
  - Both split manifests saved to `outputs/fungibility_v0_6/calibration_split.csv` and `outputs/fungibility_v0_6/evaluation_split.csv`.
- **Primary Tested Depths:**
  - Blocks $l \in \{5, 6, 7, 8, 9, 10\}$ (0-indexed; output of Block $l$ / input to Block $l+1$).
  - All six depths pre-registered and reported; no selective depth pruning.
- **Intervention Fractions & Nested Masks:**
  - $10\% \to 20$ patch tokens
  - $25\% \to 49$ patch tokens
  - $50\% \to 98$ patch tokens
  - Deterministic permutation of all 196 spatial patch positions generated with seed `9301`.
  - Nested subset invariant: $\mathcal{M}_{20} \subset \mathcal{M}_{49} \subset \mathcal{M}_{98}$.
  - Token sequence indices: `idx + 1`. CLS token (sequence index 0) is strictly untouched at injection ($h'_{l,0} = h_{l,0}$).
- **Downstream Blocks:**
  - Downstream transformer blocks $l+1 \dots 11$ + `norm` + `forward_head` are strictly identical across conditions.

---

## 3. Evaluated Conditions

At every (Model, Depth $l$, Fraction $f$):

1. **A. Clean Baseline:**
   Original untouched representation $h_{l,t}$.
2. **B. Zero Ablation:**
   $h'_{l,t} = 0$ for $t \in \mathcal{M}_f$. Reference out-of-distribution ablation.
3. **C. Held-Out Gaussian (Primary Null):**
   Using *only* the $N=1,000$ Calibration images, forward to depth $l$. Using all 196 spatial patch tokens across calibration images ($N \times 196$ tokens), compute per-feature mean $\mu_{l,d}$ and standard deviation $\sigma_{l,d}$ without labels.
   For evaluation images:
   $$h'_{l,t}[d] \sim \mathcal{N}(\mu_{l,d}, \sigma_{l,d}^2)$$
   Evaluated with 5 frozen seeds: `9401, 9402, 9403, 9404, 9405`.
   Calibration moments persisted in `outputs/fungibility_v0_6/calibration_statistics.npz`.
4. **D. Norm-Matched Random Sphere:**
   Sample $r \sim \mathcal{N}(0, I_D)$, $u = r / \|r\|_2$, $h'_{l,t} = \|h_{l,t}\|_2 \cdot u$.
   Matches exact original token norm with random orientation.
   Evaluated with 3 frozen seeds: `9501, 9502, 9503`.
5. **E. Same-Image Layer Mean:**
   $\mu_{i,l} = \frac{1}{196 - |\mathcal{M}_f|} \sum_{s \notin \mathcal{M}_f} h_{i,l,s}$.
   $h'_{l,t} = \mu_{i,l}$ for all $t \in \mathcal{M}_f$. Image-conditioned reference.
6. **F. Global Mean (Fixed Prototype):**
   $h'_{l,t} = \mu_{l}$ (the global calibration mean vector with zero noise).
   Tests whether token-level stochastic diversity is necessary or if a static coordinate-wise prototype vector suffices.
7. **G. Wrong-Depth Gaussian:**
   Sample $r_d \sim \mathcal{N}(\mu_{l',d}, \sigma_{l',d}^2)$ using statistics from a mismatched depth $l'$ according to pre-registered permutation:
   - Target depth 5 $\leftarrow$ stats from depth 8
   - Target depth 6 $\leftarrow$ stats from depth 9
   - Target depth 7 $\leftarrow$ stats from depth 10
   - Target depth 8 $\leftarrow$ stats from depth 5
   - Target depth 9 $\leftarrow$ stats from depth 6
   - Target depth 10 $\leftarrow$ stats from depth 7
   Evaluated with seed `9401`. Tests whether downstream transformer blocks expect a depth-specific coordinate distribution.

---

## 4. Quantitative Metrics & Statistical Analysis

- **Unit of Independence:** The evaluation image ($N_{\text{eval}} = 1,000$).
- **Per-Image Metrics:**
  - True-class margin: $m_i = z_y - \max_{c \neq y} z_c$
  - Margin damage: $\text{Damage}_C(l,f) = \text{BaselineMargin} - \text{Margin}_C(l,f)$
  - Top-1 correctness ($0/1$), prediction flip indicator.
- **Gaussian Recovery Fraction:**
  $$\text{Recovery}(l,f) = \frac{\text{Damage}_{\text{zero}}(l,f) - \text{Damage}_{\text{gaussian}}(l,f)}{\text{Damage}_{\text{zero}}(l,f)}$$
  Computed whenever $\text{Damage}_{\text{zero}}(l,f) \ge 0.10$.
- **Hypothesis Testing:**
  - 10,000-resample paired percentile bootstrap 95% Confidence Interval for $\Delta m$.
  - Paired Student's $t$-test and Wilcoxon signed-rank test.
  - Paired effect size Cohen's $d_z = \text{mean}(\Delta m) / \text{std}(\Delta m)$.
  - Top-1 accuracy difference $\Delta \text{Acc}$ and exact two-sided McNemar test.
- **Multiple Comparisons:**
  - Benjamini-Hochberg FDR correction applied across the 6 tested depths separately within each model and replacement fraction.
- **Multi-Seed Aggregation Rule:**
  - Evaluate per seed first; report seed-wise effects, mean across seeds, and standard deviation across seeds.
  - Never pool seed-image observations as independent samples.

---

## 5. Pre-Registered Decision Rules

### OUTCOME A — DOES NOT GENERALIZE / KILL
Declare **OUTCOME A** if:
- At Depth 8, 25% fraction, Held-Out Gaussian fails to substantially outperform Zero on either architecture ($d_z < 0.20$, bootstrap 95% CI includes zero, or accuracy differs by $< 1\%$).
- **Conclusion:** V0.5's apparent Gaussian recovery was an in-sample statistical artifact that does not generalize to disjoint data. Stop the research line.

### OUTCOME B — LOCAL ROBUST PHENOMENON
Declare **OUTCOME B** if:
- Depth-8 held-out Gaussian recovery replicates on both models ($d_z \ge 0.20$, CI $> 0$),
- BUT the effect is narrow: fails at neighboring depths (e.g. only Block 8 works), or fails at higher intervention budgets (e.g. works at 10% but collapses at 25% and 50%), or fails cross-architecture consistency.
- **Conclusion:** Tolerance to distribution-matched tokens is real but localized to an isolated depth or low fraction.

### OUTCOME C — DEPTH-TRANSITION PHENOMENON
Declare **OUTCOME C** only if **BOTH** DeiT-Tiny and DeiT-Small show:
1. **Depth-8 Held-Out Replication:** Held-Out Gaussian reliably outperforms Zero ($d_z \ge 0.20$, bootstrap CI $> 0$, FDR $q < 0.05$) at depth 8, 25% fraction;
2. **Clear Superiority over Random Sphere:** Held-Out Gaussian significantly outperforms Norm-Matched Random Sphere ($d_z \ge 0.20$);
3. **Coherent Depth Region:** The Gaussian recovery extends across a coherent multi-layer transition window (e.g. intermediate blocks show severe sensitivity, mid-to-late blocks show high Gaussian recovery, late blocks approach token irrelevance);
4. **Interpretable Budget Scaling:** Effect is strongest at 10%, remains clear at 25%, and degrades smoothly at 50% while remaining superior to Zero;
5. **Stability Across Seeds:** Consistent results across all 5 Gaussian seeds.
- **Conclusion:** Pretrained Vision Transformers exhibit a genuine depth-dependent transition in which exact patch representations become replaceable by coordinate-wise distribution-matched activations before the patch stream becomes causally dispensable.

---

## 6. Pre-Registered Programmatic Validations

Before issuing a scientific verdict, the execution harness must verify 14 assertions:
1. `check1_backbone_unmodified`: Model weights verified invariant via SHA-256 hash.
2. `check2_disjoint_splits`: Calibration and Evaluation image sets are strictly disjoint ($0$ overlap).
3. `check3_unlabeled_calibration_stats`: Calibration statistics computed without label conditioning.
4. `check4_no_eval_data_in_stats`: Evaluation activations never contribute to calibration moments.
5. `check5_all_depths_evaluated`: All six depths $\{5, 6, 7, 8, 9, 10\}$ evaluated.
6. `check6_nested_masks`: Mask indices satisfy $\mathcal{M}_{20} \subset \mathcal{M}_{49} \subset \mathcal{M}_{98}$ from seed 9301.
7. `check7_mask_counts`: Exactly 20, 49, and 98 tokens modified per image at corresponding fractions.
8. `check8_cls_untouched`: CLS token strictly untouched at injection ($h'_{l,0} = h_{l,0}$, max diff $= 0.0$).
9. `check9_downstream_blocks_identical`: Downstream blocks $l+1 \dots 11$ identical across conditions.
10. `check10_norm_matched_random_rel_err`: Random Sphere tokens match target norm within $10^{-5}$ relative error.
11. `check11_frozen_seeds`: Frozen seeds verified (`calib: 9101, eval: 9201, mask: 9301, gauss: 9401-9405, sphere: 9501-9503`).
12. `check12_image_level_independence`: Statistics computed with $N_{\text{eval}} = 1,000$ independent image units.
13. `check13_calibration_stats_persisted`: `calibration_statistics.npz` saved and reproducible.
14. `check14_report_equals_manifest`: All reported values match saved machine-readable outputs.
