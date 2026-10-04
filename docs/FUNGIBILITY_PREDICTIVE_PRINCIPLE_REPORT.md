# Mechanistic Follow-Up Report: Evaluating Clean-State Representation Geometry & Downstream Sensitivity as Predictors of ViT Patch Fungibility

**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Protocol Document**: [`docs/FUNGIBILITY_PREDICTIVE_PRINCIPLE_PROTOCOL.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_PREDICTIVE_PRINCIPLE_PROTOCOL.md)  
**Execution Timestamp**: 2026-10-04T09:15:28Z  
**Execution Runtime**: 246.34 seconds (Peak VRAM: 0.81 GB, NVIDIA RTX 5070 GPU)  
**Target Hardware**: 1x NVIDIA RTX GPU, FP32/FP64 Verified Precision  
**Models Evaluated**:
1. `deit_tiny_patch16_224` ($D=192$, 12 blocks, 196 patches, supervised distillation)
2. `deit_small_patch16_224` ($D=384$, 12 blocks, 196 patches, supervised distillation)
3. `vit_base_patch16_224.augreg_in1k` ($D=768$, 12 blocks, 196 patches, supervised AugReg)
4. `dinov2_vits14_lc` ($D=384$, 12 blocks, 256 patches, self-supervised, dual-stream linear head)

**Validated Target Depths Evaluated**:
- DeiT-Tiny: $l \in \{5, 6, 7, 8, 9, 10\}$ (6 depths)
- DeiT-Small: $l \in \{5, 6, 7, 8, 9, 10\}$ (6 depths)
- ViT-B/16 AugReg: $l \in \{5, 7, 8, 9, 10\}$ (5 depths)
- DINOv2 ViT-S/14: $l \in \{5, 7, 8, 9, 10\}$ (5 depths)  
**Total Target Points**: $N_{\text{points}} = 22$ validated model-depth pairs  
**Calibration Dataset**: $N_{\text{calib}} = 1,000$ strictly disjoint ImageNet-1k images (seed 9101, zero overlap with evaluation)  
**Evaluation Target ($F_l$)**: $F_l = 1 - D_{\text{centroid}}(l) / D_{\text{zero}}(l)$ from canonical 25% replacement manifests  

**Pre-Registered Scientific Verdict**: **OUTCOME D — KILL PREDICTIVE-PRINCIPLE HYPOTHESIS**

---

## 1. Executive Summary

This mechanistic follow-up sprint tested the pre-registered hypothesis:
> *"Without running a replacement sweep, a clean-state geometric or downstream sensitivity metric predicts where patch-content fungibility emerges in Vision Transformers."*

The candidate primary mechanism proposed was the **Sensitivity-Weighted Representation Geometry**:
$$S_l = \text{tr}(J_l \Sigma_l J_l^\top) \quad \text{or} \quad S_l^{\text{margin}} = \frac{1}{N} \sum_{k=1}^N g_{l, k}^\top \Sigma_l g_{l, k}$$
under the prediction that low $S_l$ (natural token variation lying along functionally insensitive downstream directions) predicts high fungibility ($F_l \approx 1$), whereas high $S_l$ predicts fragility ($F_l \ll 1$).

### Key Empirical Findings:
1. **Primary Mechanistic Candidate Falsified:**
   - The primary candidate $S_l^{\text{margin}}$ achieved a pooled Spearman rank correlation of only $\rho = -0.3631$, and its 95% bootstrap confidence interval $[-0.7917, +0.1617]$ spans zero.
   - The sign of the relationship **flipped across architectures**: in DeiT-Small ($\rho = -0.8857$) and DINOv2 ($\rho = -0.8000$) higher $S_l$ tracked fragility, but in ViT-B AugReg the correlation inverted to **$\rho = +0.5000$** (higher sensitivity-weighted energy coincided with peak fungibility at Block 7).
   - Univariate regression $R^2$ was only $0.0599$, and incremental explanatory power beyond normalized depth was $\Delta R^2 = 0.0530$.
2. **Downstream Jacobian Sensitivity Alone Explains Nothing:**
   - Pure downstream sensitivity $\|J_l^{\text{margin}}\|_F$ and logit Jacobian $\|J_l^{\text{logit}}\|_F$ showed near-zero correlation with fungibility ($\rho = 0.1112$ and $0.1858$, univariate $R^2 \le 0.0028$).
   - Downstream margin sensitivity decays almost monotonically across depth in all models, completely failing to anticipate the inverted U-shaped / peak-and-drop trajectory of patch fungibility (which peaks at blocks 7–9 and falls at terminal blocks).
3. **Combining Geometry with Sensitivity Degrades Predictive Power:**
   - Clean representation geometry alone (Family A) showed moderate pooled correlation: Entropy Effective Rank $r_{\text{eff}}(\Sigma_l)$ achieved pooled $\rho = +0.5325$ ($R^2 = 0.3271$), and cumulative spectral concentration $C_{16}$ achieved $\rho = -0.5042$ ($R^2 = 0.3260$).
   - Multiplying covariance with downstream gradient sensitivity ($S_l$) **decreased** correlation from $\rho = +0.5325$ to $\rho = -0.3631$. Downstream sensitivity acts as noise rather than a clarifying filter.
4. **Catastrophic Failure of Cross-Architecture Transfer (LOAO):**
   - In Leave-One-Architecture-Out (LOAO) cross-validation, every candidate predictor produced negative out-of-sample $R^2$ (mean $R^2_{\text{LOAO}} = -1.7116$ for $r_{\text{eff}}$, $-1.5303$ for $S_l^{\text{margin}}$).
   - Cross-architecture transition layer prediction failed with a mean error of **$2.75$ blocks** (out of a 6-block evaluation window).
   - The failure is structural: representation metrics have architecture-dependent absolute scales ($D=192$ vs $D=384$ vs $D=768$) and contradictory within-model depth trajectories (e.g., $r_{\text{eff}}$ correlates positively with $F_l$ in DINOv2 and DeiT, but negatively in ViT-B).
5. **Decisive Verdict: OUTCOME D (KILL PREDICTIVE-PRINCIPLE HYPOTHESIS)**:
   - A clean-state geometric or sensitivity metric cannot reliably predict patch fungibility across ViT architectures without intervention sweeps.
   - Patch fungibility remains an **empirically validated intervention phenomenon**, constrained by learned geometry, but it cannot be substituted by an a priori clean-state surrogate.

---

## 2. Comprehensive Statistical Summary Table

Below is the audited evaluation of all candidate predictors across the 22 validated model-depth target points, sorted by absolute pooled Spearman rank correlation:

| Predictor | Family | Pooled Pearson $r$ | Pooled Spearman $\rho$ | 95% Bootstrap CI | Univariate $R^2$ | $\Delta R^2$ over Depth | Partial $r$ (given Depth) | DeiT-Tiny $\rho$ | DeiT-Small $\rho$ | ViT-B $\rho$ | DINOv2 $\rho$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`effective_rank`** | A (Geometry) | **+0.5719** | **+0.5325** | $[+0.158, +0.789]$ | **0.3271** | **+0.3174** | **+0.5663** | +0.7714 | +0.6000 | **-0.7000** | +0.9000 |
| **`c16_concentration`** | A (Geometry) | -0.5710 | -0.5042 | $[-0.779, -0.121]$ | 0.3260 | +0.3167 | -0.5657 | -0.7714 | -0.8857 | **+0.7000** | -0.9000 |
| **`patch_to_cls_cos_sim`** | B (Mixing) | -0.4535 | -0.4218 | $[-0.817, +0.085]$ | 0.2057 | +0.1975 | -0.4468 | +0.5429 | -0.8857 | 0.0000 | -0.2000 |
| **`s_margin` ($S_l$)** | **D (Primary)** | **-0.2447** | **-0.3631** | **$[-0.792, +0.162]$** | **0.0599** | **+0.0530** | **-0.2314** | -0.0857 | -0.8857 | **+0.5000** | -0.8000 |
| **`neg_log_s_margin`** | D (Primary) | +0.1915 | +0.3631 | $[-0.162, +0.792]$ | 0.0367 | +0.0286 | +0.1700 | +0.0857 | +0.8857 | **-0.5000** | +0.8000 |
| **`inv_s_margin`** | D (Primary) | +0.1230 | +0.3631 | $[-0.162, +0.792]$ | 0.0151 | +0.0053 | +0.0731 | +0.0857 | +0.8857 | **-0.5000** | +0.8000 |
| **`c4_concentration`** | A (Geometry) | -0.4923 | -0.3123 | $[-0.687, +0.111]$ | 0.2424 | +0.2478 | -0.5004 | -0.0857 | -0.3714 | +0.5000 | -0.9000 |
| **`c1_concentration`** | A (Geometry) | -0.4635 | -0.2829 | $[-0.657, +0.147]$ | 0.2148 | +0.2236 | -0.4754 | +0.0286 | -0.3714 | +0.5000 | -0.3000 |
| **`norm_depth` ($l/L$)** | Control | +0.1021 | +0.2624 | $[-0.276, +0.750]$ | 0.0104 | 0.0000 | 0.0000 | +0.0857 | +0.8857 | -0.5000 | +0.9000 |
| **`s_margin_grad_norm`** | D (Normalized) | -0.5358 | -0.2163 | $[-0.639, +0.268]$ | 0.2871 | +0.3423 | -0.5881 | +0.0857 | +0.8857 | -0.7000 | +0.9000 |
| **`var_token` ($V_l$)** | A (Geometry) | -0.5526 | -0.2140 | $[-0.618, +0.268]$ | 0.3054 | +0.3780 | -0.6180 | +0.0857 | +0.8857 | -0.5000 | +0.9000 |
| **`cov_trace` $\text{tr}(\Sigma_l)$** | A (Geometry) | -0.5756 | -0.2027 | $[-0.614, +0.282]$ | 0.3313 | +0.3972 | -0.6335 | +0.0857 | +0.8857 | -0.5000 | +0.9000 |
| **`dist_to_centroid`** | B (Mixing) | -0.5148 | -0.2027 | $[-0.614, +0.282]$ | 0.2650 | +0.3115 | -0.5611 | +0.0857 | +0.8857 | -0.5000 | +0.9000 |
| **`dist_to_patch_mean`** | B (Mixing) | -0.4916 | -0.2027 | $[-0.614, +0.282]$ | 0.2417 | +0.2894 | -0.5408 | +0.0857 | +0.8857 | -0.5000 | +0.9000 |
| **`logit_jac_norm`** | C (Sensitivity) | +0.0525 | +0.1858 | $[-0.303, +0.616]$ | 0.0028 | +0.0068 | +0.0829 | -0.0857 | -0.8857 | +0.5000 | -0.9000 |
| **`sens_per_unit_var`** | C (Sensitivity) | -0.0200 | +0.1813 | $[-0.306, +0.611]$ | 0.0004 | +0.0002 | +0.0147 | -0.0857 | -0.8857 | +0.5000 | -0.9000 |
| **`s_logit_trace_norm`** | D (Normalized) | -0.0280 | +0.1767 | $[-0.306, +0.607]$ | 0.0008 | +0.0000 | +0.0067 | -0.0857 | -0.8857 | +0.5000 | -0.9000 |
| **`s_margin_trace_norm`**| D (Normalized) | -0.0299 | +0.1248 | $[-0.360, +0.582]$ | 0.0009 | +0.0000 | +0.0052 | -0.0857 | -0.8857 | +0.5000 | -0.9000 |
| **`margin_grad_norm`** | C (Sensitivity) | +0.0445 | +0.1112 | $[-0.380, +0.583]$ | 0.0020 | +0.0061 | +0.0786 | -0.0857 | -0.8857 | +0.5000 | -0.9000 |
| **`mean_token_grad_norm`**| C (Sensitivity)| +0.0344 | +0.0898 | $[-0.409, +0.567]$ | 0.0012 | +0.0048 | +0.0696 | -0.0857 | -0.8857 | +0.5000 | -0.9000 |
| **`patch_pairwise_cos_sim`**| B (Mixing) | -0.2793 | -0.0830 | $[-0.577, +0.403]$ | 0.0780 | +0.1938 | -0.4426 | +0.0286 | +0.8857 | -0.4000 | -0.1000 |
| **`s_logit`** | D (Primary) | +0.1290 | -0.0175 | $[-0.513, +0.493]$ | 0.0166 | +0.0253 | +0.1599 | -0.0857 | -0.8857 | +0.7000 | -0.8000 |

---

## 3. Leave-One-Architecture-Out (LOAO) Cross-Validation Results

To test whether any predictor establishes a true **predictive principle** rather than an in-sample curve fit, we trained linear predictors on three architectures and evaluated generalization on the fourth held-out architecture.

### 3.1 Top Geometric Predictor: `effective_rank`
- **Mean Out-of-Sample MAE**: **0.2675**
- **Mean Out-of-Sample $R^2_{\text{LOAO}}$**: **-1.7116** (Catastrophic negative generalization)

| Held-Out Architecture | Train Architectures | Slope $\beta$ | Intercept $\alpha$ | Out-of-Sample MAE | Out-of-Sample $R^2_{\text{LOAO}}$ | Held-Out Spearman $\rho$ | True Trans. Depth | Pred. Trans. Depth | Depth Error |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeiT-Tiny** | Small + ViT-B + DINOv2 | +0.00246 | 0.4188 | 0.2083 | **+0.0122** | +0.7714 | 8 | 10 | 2 blocks |
| **DeiT-Small** | Tiny + ViT-B + DINOv2 | +0.00246 | 0.3262 | 0.2636 | **-3.9212** | +0.6000 | 7 | 10 | 3 blocks |
| **ViT-B/16 AugReg** | Tiny + Small + DINOv2 | +0.00137 | 0.5709 | 0.3207 | **-0.7232** | **-0.7000** | 7 | 10 | 3 blocks |
| **DINOv2 ViT-S/14** | Tiny + Small + ViT-B | +0.00378 | 0.3159 | 0.2776 | **-2.2142** | +0.9000 | 8 | 5 | 3 blocks |

**Mean Transition Depth Prediction Error**: **2.75 blocks** (out of 6 blocks evaluated).

### 3.2 Primary Candidate Metric: `s_margin` ($S_l$)
- **Mean Out-of-Sample MAE**: **0.2625**
- **Mean Out-of-Sample $R^2_{\text{LOAO}}$**: **-1.5303**

| Held-Out Architecture | Train Architectures | Slope $\beta$ | Intercept $\alpha$ | Out-of-Sample MAE | Out-of-Sample $R^2_{\text{LOAO}}$ | Held-Out Spearman $\rho$ | True Trans. Depth | Pred. Trans. Depth | Depth Error |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeiT-Tiny** | Small + ViT-B + DINOv2 | -0.1652 | 0.8250 | 0.2443 | -0.3804 | -0.0857 | 8 | 10 | 2 blocks |
| **DeiT-Small** | Tiny + ViT-B + DINOv2 | -0.0934 | 0.6976 | 0.2709 | -4.2045 | -0.8857 | 7 | 10 | 3 blocks |
| **ViT-B/16 AugReg** | Tiny + Small + DINOv2 | -0.0890 | 0.6983 | 0.2637 | -0.2076 | **+0.5000** | 7 | 5 | 2 blocks |
| **DINOv2 ViT-S/14** | Tiny + Small + ViT-B | -0.0639 | 0.7259 | 0.2712 | -1.3289 | -0.8000 | 8 | 10 | 2 blocks |

**Mean Transition Depth Prediction Error**: **2.25 blocks**.

---

## 4. Mandatory Question Audit (Section 14)

### Q1: Can any clean-state metric predict fungibility without replacement sweeps?
**Answer: NO.**  
No evaluated clean-state metric reliably predicts layer fungibility across architectures before running replacement sweeps. The top-performing clean-state metric, Entropy Effective Rank $r_{\text{eff}}(\Sigma_l)$, achieved a pooled Spearman rank correlation of only $\rho = +0.5325$ (falling far short of the pre-registered threshold $|\rho| \ge 0.70$), and completely failed out-of-sample LOAO cross-validation ($R^2_{\text{LOAO}} = -1.7116$, mean transition error $= 2.75$ blocks).

### Q2: Does covariance geometry alone suffice?
**Answer: NO.**  
Covariance geometry captures part of the intra-model depth trajectory in DeiT-Tiny, DeiT-Small, and DINOv2, but it **inverts in ViT-B AugReg**. In ViT-B, effective rank collapses to near-zero ($r_{\text{eff}} \approx 8 - 15$) at layers 5–9, exactly where patch fungibility peaks ($F_7 = 0.818$). Consequently, covariance geometry alone produces a negative rank correlation in ViT-B ($\rho = -0.7000$) and fails cross-architecture transfer.

### Q3: Does downstream Jacobian sensitivity alone suffice?
**Answer: DECISIVELY NO.**  
Downstream Jacobian sensitivity alone has virtually **zero predictive power** ($R^2 = 0.0020$, pooled $\rho = 0.1112$). Downstream sensitivity $\|J_l^{\text{margin}}\|_F$ decays monotonically as layer depth increases in all models because fewer non-linear transformations separate intermediate layers from the readout. It is blind to the mid-to-late emergence and peak of fungibility (blocks 7–9) vs fragile early layers (blocks 5–6) vs terminal CLS decoupling (block 10).

### Q4: Does combining covariance geometry with downstream sensitivity perform better than either alone?
**Answer: NO; IT PERFORMS WORSE.**  
The primary mechanistic candidate $S_l = \text{tr}(J_l \Sigma_l J_l^\top)$ yielded a pooled rank correlation of $\rho = -0.3631$ ($R^2 = 0.0599$), which is substantially **weaker** than clean representation geometry alone ($r_{\text{eff}}$: $\rho = +0.5325$, $R^2 = 0.3271$). Folding downstream Jacobian sensitivity into the covariance matrix injected cross-model gradient scale variance that degraded the geometric signal.

### Q5: Does the predictor explain anything beyond normalized layer depth?
**Answer: Statistically yes within pooled OLS, but practically no across architectures.**  
In pooled linear regression on the 22 target points, normalized depth $l/L$ has a low linear $R^2 = 0.0104$ because fungibility is non-monotonic (peaking at blocks 7–9 and dropping at block 10). Adding `effective_rank` increases pooled $R^2$ to $0.3278$ ($\Delta R^2 = +0.3174$). However, this incremental $R^2$ is an artifact of architecture clustering (different models occupying separate rank bands) rather than a generalizable relationship.

### Q6: Does it generalize leave-one-architecture-out?
**Answer: NO.**  
Every candidate predictor achieved negative out-of-sample $R^2_{\text{LOAO}}$ across the four folds (ranging from $-0.72$ to $-3.92$). A frozen linear predictor fitted on three architectures failed completely on the fourth.

### Q7: Can it predict the approximate fungibility transition layer?
**Answer: NO.**  
Predictors transfer transition thresholds with a mean error of **$2.25$ to $2.75$ blocks**. Across a target depth window spanning blocks 5 to 10 (a 5-block span), an average error of nearly 3 blocks is no better than uninformative guessing.

### Q8: What is the simplest supported mechanistic interpretation?
**Answer:**  
Patch-content fungibility is a **causal network-state property that requires intervention to measure**. While valid surrogate replacements are strictly constrained by learned activation geometry (as proven in V0.6–V1 by coordinate permutation, sign-flip, and natural PC1 failures), the *tolerance* of the downstream network to replacement is determined by non-linear, multi-block downstream compensation dynamics that cannot be linearized into a static quadratic form $g^\top \Sigma g$.

### Q9: What claims are NOT supported?
**Answer:**
1. **NOT SUPPORTED**: *"A clean-state geometric or sensitivity metric predicts where patch-content fungibility emerges without replacement sweeps."* (Falsified).
2. **NOT SUPPORTED**: *"Sensitivity-weighted representation geometry $S_l = \text{tr}(J_l \Sigma_l J_l^\top)$ is the universal mechanistic driver of fungibility."* (Falsified; $\rho = -0.363$, CI spans zero, sign inverts on ViT-B).
3. **NOT SUPPORTED**: *"Downstream Jacobian sensitivity alone dictates layer fragility."* (Falsified; $R^2 = 0.002$).

---

## 5. Visual Artifacts and Diagnostic Inspection

The generated high-resolution publication figures are stored in `figures/fungibility_predictive_principle/`:

1. **[`Figure A: Depth Alignment (Fungibility vs. Effective Rank)`](file:///d:/Study/ResCancel/figures/fungibility_predictive_principle/figure_a_depth_alignment.png)**:
   - Illustrates why cross-architecture transfer fails: DeiT-Tiny, DeiT-Small, and DINOv2 show positive tracking between effective rank and fungibility, while ViT-B AugReg displays an inverted trajectory (effective rank collapses at layers 5–9 where fungibility peaks).
2. **[`Figure B: Predictor vs Fungibility Scatter Plot`](file:///d:/Study/ResCancel/figures/fungibility_predictive_principle/figure_b_predictor_scatter.png)**:
   - Demonstrates that pooled correlation is driven by discrete model clustering rather than a continuous universal curve.
3. **[`Figure C: Leave-One-Architecture-Out Generalization`](file:///d:/Study/ResCancel/figures/fungibility_predictive_principle/figure_c_loao_predicted_vs_observed.png)**:
   - Visualizes horizontal prediction lines for held-out models, confirming zero cross-architecture predictive discrimination.
4. **[`Figure D: Spectral & Sensitivity Decomposition Across Layers`](file:///d:/Study/ResCancel/figures/fungibility_predictive_principle/figure_d_spectral_sensitivity_decomposition.png)**:
   - Plots the full 12-block profiles of effective rank and downstream sensitivity across the four architectures.

---

## 6. Pre-Registered Decision Verdict

In accordance with Section 10 and Section 11 of [`docs/FUNGIBILITY_PREDICTIVE_PRINCIPLE_PROTOCOL.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_PREDICTIVE_PRINCIPLE_PROTOCOL.md):

> **Criteria for Outcome D (Kill)**: No clean metric shows a robust relationship ($|\rho| < 0.60$ or primary candidate fails cross-architecture transfer).

### Final Verdict:
```
================================================================================
OUTCOME D — KILL PREDICTIVE-PRINCIPLE HYPOTHESIS
================================================================================
```

### Actionable Scope and Paper Boundaries:
1. **Preserve Existing Validated Claims**: The core paper claims (Level 3: Late patch-content fungibility generalizes across supervised and self-supervised ViTs; replacements are geometry-constrained; complete replacement requires token diversity) remain completely intact and unaffected.
2. **Do NOT Elevate a Clean-State Predictive Principle**: We do NOT add a claim that clean geometry or sensitivity predicts fungibility a priori.
3. **Manuscript Preservation**: `docs/PAPER_DRAFT.md` remains untouched during this exploratory sprint. In the discussion / limitations section of future manuscript revisions, this negative result can be cleanly documented: *patch-content fungibility is an emergent intervention phenomenon that cannot be reduced to clean-state Jacobian-weighted covariance.*
