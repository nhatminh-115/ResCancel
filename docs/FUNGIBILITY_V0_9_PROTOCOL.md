# Patch Fungibility V0.9 Protocol: Natural Low-Rank Variance & Downstream Rank-Expansion Test

## 1. Scientific Context & Purpose

In Patch Fungibility V0.8, we established:
1. Token-to-token diversity is strictly causal: identical patch streams collapse downstream computation to $\sim 1\% - 10\%$ accuracy, while independent stochastic replacement restores substantial predictive capability ($26.3\%$ in DeiT-Tiny, $46.4\%$ in DeiT-Small).
2. The failure of identical token streams is mechanistically driven by attention rank collapse: spatial patch Key and Value variance collapse to exact $0.000000$, and the attention matrix degenerates to rank $1.0$.
3. Under controlled subspace sweeps, DeiT-Small exhibited an unexpected result: a single principal component direction (PCA Rank 1) was sufficient to reach $46.10\%$ accuracy, matching the full diagonal Gaussian ceiling ($46.06\%$).
4. In contrast, DeiT-Tiny failed to saturate at low ranks ($12.2\% - 17.3\%$) and appeared to require high-dimensional variation ($R \ge 64, K = 196$).

### The Critical Confound
In V0.8, all rank conditions were forced to have the exact same total perturbation energy $E_{\text{full}} = \sum_{d=0}^{D-1} \sigma_8[d]^2$. For rank $R$, this concentrated the entire full-dimensional variance into only $R$ directions:
$$\epsilon_t = \sum_{r=1}^R a_{t, r} v_r, \quad a_{t, r} \sim \mathcal{N}\left(0, \frac{E_{\text{full}}}{\sum_{j=1}^R \lambda_j} \lambda_r\right)$$
For $R=1$, the PC1 perturbation was scaled by $\sqrt{E_{\text{full}} / \lambda_1}$ relative to its natural variance:
- In DeiT-Tiny: $\lambda_1 / E_{\text{full}} \approx 14.1\% \implies$ variance was inflated by **$7.08\times$** (amplitude scaled by $\mathbf{2.66\times}$).
- In DeiT-Small: $\lambda_1 / E_{\text{full}} \approx 25.2\% \implies$ variance was inflated by **$3.97\times$** (amplitude scaled by $\mathbf{1.99\times}$).

This artificial variance concentration may have created out-of-distribution coordinate amplitudes, particularly in DeiT-Tiny where the eigenvalue spectrum is diffuse. Consequently, we cannot yet conclude that DeiT-Tiny intrinsically requires high-dimensional variation.

### Objectives of V0.9
This is a narrow mechanism-closing experiment designed to answer two precise questions:
- **Objective A**: Does the apparent low-rank failure of DeiT-Tiny disappear when low-rank diversity uses natural unscaled empirical variance, or does a genuine architectural dichotomy exist?
- **Objective B**: When rank-1 or low-rank variation is injected after Block 8, does downstream computation remain low-rank, or does low-rank input diversity seed rapid representational rank expansion through Blocks 9–11?

---

## 2. Experimental Design & Interventions

### 2.1 Models, Data Splits, and Calibration Base
- **Models**: Pretrained `deit_tiny_patch16_224` ($D=192$) and `deit_small_patch16_224` ($D=384$) from `timm`. Strictly frozen.
- **Data Splits**:
  - Calibration split: $N=1,000$ images (stratified 1 per class), $196,000$ patch tokens. Zero labels or gradients.
  - Evaluation split: $N=1,000$ images. Zero overlap with calibration split.
- **Calibration PCA Base**:
  Reuse Block-8 calibration mean $\mu_8$, standard deviation $\sigma_8$, total energy $E_{\text{full}} = \sum_d \sigma_8[d]^2$, and eigendecomposition of centered patch covariance $\Sigma_{\text{calib}} = V \Lambda V^\top$ where $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D$.
- **Intervention Site**:
  Depth 8 (output of Block 8 / input to Block 9). Primary experiment replaces 100% of spatial patch tokens (indices 1..196). CLS token (index 0) remains original and untouched.

---

### 2.2 Conditions & Interventions

#### Reference Conditions
1. **Clean**: Original frozen model on evaluation split.
2. **Static Centroid**: $h'_t = \mu_8$ for all $t \in \{1 \dots 196\}$.
3. **Full Diagonal Gaussian**: $h'_t = \mu_8 + \epsilon_t, \epsilon_{t, d} \sim \mathcal{N}(0, \sigma_8[d]^2)$. (Seed 16001).
4. **Full Isotropic Gaussian**: $h'_t = \mu_8 + \epsilon_t, \epsilon_t \sim \mathcal{N}(0, \frac{E_{\text{full}}}{D} I_D)$. (Seed 16001).
5. **V0.8 Energy-Matched PCA Ranks**:
   $R \in \{1, 2, 4, 8, 16, 32, 64\}$. Reproduced using V0.8 formulation where $a_{t, r} \sim \mathcal{N}(0, \frac{E_{\text{full}}}{\sum_{j=1}^R \lambda_j} \lambda_r)$. Seeds: `16001, 16002, 16003, 16004, 16005`.

#### Key Experiment 1: Natural-Energy PCA Ranks (PCA-NATURAL-RANK-R)
For $R \in \{1, 2, 4, 8, 16, 32, 64\}$:
$$h'_t = \mu_8 + \sum_{r=1}^R a_{t, r} v_r, \quad a_{t, r} \sim \mathcal{N}(0, \lambda_r)$$
- Sampled independently per token $t$ and per component $r$ **WITHOUT any $E_{\text{full}}$ rescaling**.
- Expected perturbation energy is natural: $E_R = \sum_{r=1}^R \lambda_r$.
- Seeds: `16001, 16002, 16003, 16004, 16005`.

#### Key Experiment 2: PC1 Amplitude Sweep
Natural PC1 perturbation: $\epsilon_t = z_t \sqrt{\lambda_1} v_1$, where $z_t \sim \mathcal{N}(0, 1)$.
Test scale multiplier $s \in \{0.25, 0.5, 1.0, 2.0, 4.0, E_{\text{MATCH}}\}$, where $E_{\text{MATCH}} = \sqrt{E_{\text{full}} / \lambda_1}$:
$$h'_t = \mu_8 + s \epsilon_t$$
- Seeds: `17001, 17002, 17003, 17004, 17005`.
- Tests whether rank-1 performance has a broad useful amplitude regime or if V0.8 landed at an unusual scaling point.

#### Key Experiment 3: Principal Component Identity Test
Test single-component variation along specific eigenvectors:
$$k \in \{1, 2, 3, 4, 8, 16\}$$
Two conditions per $k$:
1. `NATURAL-PC-k`: $\epsilon_t = z_t \sqrt{\lambda_k} v_k$ (unscaled natural eigenvalue).
2. `PC1-ENERGY-MATCHED-PC-k`: $\epsilon_t = z_t \sqrt{\lambda_1} v_k$ (norm-matched to natural PC1 energy).
- Seeds: `17001, 17002, 17003, 17004, 17005`.
- Distinguishes whether "any 1D diversity works" or "PC1 carries a privileged variation direction."

#### Key Experiment 4: Matched Random 1D Direction Control
Generate random unit vectors $u \in \mathbb{R}^D$ ($\|u\|_2 = 1$).
$$\epsilon_t = z_t \sqrt{\lambda_1} u$$
- Exact variance matching to natural PC1: $\mathbb{E}[\|\epsilon_t\|_2^2] = \lambda_1$.
- Seeds: `18001, 18002, 18003, 18004, 18005`.
- Clean test of PCA PC1 vs. arbitrary 1D subspace at natural variance.

---

### 2.3 Downstream Representation Rank Expansion & Attention Diagnostics

#### Representation Rank Tracking Across Blocks
For designated Key Conditions:
- Clean
- Static Centroid
- Natural PCA Rank 1
- Natural PCA Rank 2
- Natural PCA Rank 4
- Energy-Matched PCA Rank 1
- Full Gaussian

Compute representation rank metrics on centered spatial patch representations $H_c = H_{\text{patch}} - \text{mean}_{\text{token}}(H_{\text{patch}}) \in \mathbb{R}^{196 \times D}$ at:
1. Depth 8 injection
2. Block 9 output
3. Block 10 output
4. Block 11 output

Metrics computed **per image first**, then averaged across $N=1,000$ evaluation images:
- Numerical rank: count of singular values with $s_i / s_1 > 10^{-3}$
- Stable rank: $r_{\text{stable}} = \frac{\|H_c\|_F^2}{\|H_c\|_2^2} = \frac{\sum s_i^2}{s_1^2}$
- Participation-ratio effective rank: $r_{\text{eff}} = \frac{(\sum s_i^2)^2}{\sum s_i^4}$
- Mean pairwise cosine similarity across all $\frac{196 \times 195}{2}$ patch pairs
- Spatial patch-feature variance: $\frac{1}{196} \|H_c\|_F^2$

#### Attention Diagnostics Across Blocks 9–11
During actual forward execution through Blocks 9, 10, 11:
- Spatial patch Key variance across 196 tokens
- Spatial patch Value variance across 196 tokens
- CLS-to-patch attention distribution entropy (nats)
- Softmax attention matrix stable rank

#### Geometric Expansion Diagnostic
For Natural PCA Rank 1:
Given token coefficient $z_t \sim \mathcal{N}(0, 1)$ at injection, measure:
1. Correlation between $z_t$ and downstream projection $\langle h_t^{(l)}, v_1 \rangle$ on original PC1.
2. Fraction of downstream patch variance contained in original PC1 direction vs. orthogonal subspace.

---

## 3. Pre-Registered Hypotheses & Decision Rules

### 3.1 Competing Hypotheses
- **H1 — V0.8 Tiny Failure Was Amplitude Artifact**:
  Natural-energy low-rank PCA substantially improves DeiT-Tiny relative to energy-matched low-rank PCA.
- **H2 — True Architectural Dichotomy**:
  Even after removing the amplitude confound, DeiT-Small retains low-rank sufficiency while DeiT-Tiny requires much higher $R$.
- **H3 — PC1-Specific Sufficiency**:
  PC1 strongly outperforms other single PCs ($k \in \{2, 3, 4, 8, 16\}$) and matched-energy random 1D directions ($d_z \ge 0.20$).
- **H4 — Generic Low-Rank Seeding**:
  Multiple single-PC or random 1D directions produce similar recovery once natural variance is used; the primary requirement is non-collapsed token variation rather than PC1 identity.

---

### 3.2 Decision Rules

#### Primary Outcome Classification
- **OUTCOME A — V0.8 ARCHITECTURAL DICHOTOMY WAS AN ENERGY ARTIFACT**:
  DeiT-Tiny natural-energy low-rank PCA performs substantially better than energy-matched low-rank PCA, and reaches near-full Gaussian performance at low ranks ($R \le 8$ reaches $\ge 90\%$ of Full-Gaussian recovery).
- **OUTCOME B — TRUE ARCHITECTURAL DICHOTOMY**:
  After natural-energy correction:
  - DeiT-Small: $R \le 4$ reaches $\ge 90\%$ of Full-Gaussian recovery.
  - DeiT-Tiny: requires $R \ge 32$ or fails to reach $90\%$ at low ranks.
  - Replicated across seeds.
- **OUTCOME C — SPECIFIC DOMINANT DIRECTION**:
  In BOTH models, PC1 substantially outperforms PC2, PC3, PC4, etc., and matched random 1D directions ($d_z \ge 0.20$, McNemar $p < 0.01$).
- **OUTCOME D — GENERIC LOW-DIMENSIONAL SEED**:
  Multiple single directions or random 1D directions produce similar recovery, and PC1 does not have a meaningful advantage over other directions.

#### Rank Propagation Classification (Evaluated Independently)
- **RANK-EXPANSION**:
  Natural PCA Rank 1 injection has $r_{\text{eff}} \le 1.5$ at Depth 8, but expands substantially ($r_{\text{eff}} \ge 3.0$) after Blocks 9–11.
  *Interpretation: "Rank-1 variation at the intervention point is sufficient to seed downstream representational diversification."*
- **RANK-PRESERVING**:
  Effective rank remains close to 1 throughout Blocks 9–11 ($r_{\text{eff}} < 2.0$), yet downstream classification functions.
  *Interpretation: "Downstream computation operates directly on rank-1 patch variation."*

---

## 4. Programmatic Assertions

Before issuing any scientific verdict, the pipeline must verify:
1. **Backbone Unchanged**: Parameter hashes identical before and after run.
2. **Split Isolation**: Exactly 1,000 calibration and 1,000 evaluation images with zero index overlap.
3. **No Evaluation Leakage**: Evaluation images never used for calibration PCA or Gaussian statistics.
4. **Calibration PCA Integrity**: PCA correctly loaded or recomputed identically.
5. **CLS Untouched**: CLS token at index 0 unmodified during patch replacement.
6. **Token Count Exact**: Exactly 196 spatial patch tokens replaced in primary experiment.
7. **Natural PCA Unscaled**: Natural PCA coefficients use unscaled empirical eigenvalues $\lambda_r$.
8. **Energy-Matched Formula Exact**: Energy-matched reference reproduces V0.8 formulation.
9. **Amplitude Factors Match**: PC1 scale multipliers are $\{0.25, 0.5, 1.0, 2.0, 4.0, E_{\text{MATCH}}\}$.
10. **Random Unit Vectors**: Random 1D directions have $\|u\|_2 = 1.0 \pm 10^{-5}$.
11. **Random Variance Matched**: Random 1D variance equals $\lambda_1$.
12. **PCA Orthonormality**: Max Gram matrix error $|V^\top V - I| < 10^{-4}$.
13. **Energy Logged**: Realized perturbation energy logged for every condition.
14. **Per-Image Rank Computation**: Effective and stable rank computed per image prior to averaging.
15. **Attention Diagnostics Measured**: Diagnostics captured directly from Blocks 9–11 forward passes.
16. **Seeds Protocol Compliant**: Seeds `16001..16005`, `17001..17005`, `18001..18005` used exactly.
17. **Zero Labels/Gradients**: No labels or gradients used to construct replacements.
18. **Evaluation Unit Exact**: Exactly $N=1,000$ independent evaluation images.

If any assertion fails: **DO NOT ISSUE SCIENTIFIC VERDICT.**

---

## 5. Artifacts and Outputs

- **Protocol**: `docs/FUNGIBILITY_V0_9_PROTOCOL.md`
- **Scientific Report**: `docs/FUNGIBILITY_V0_9_REPORT.md`
- **Output Directory**: `outputs/fungibility_v0_9/`
  - `eigenvalue_spectrum.csv`
  - `natural_vs_energy_matched_rank.csv`
  - `pc_identity_results.csv`
  - `pc1_scale_sweep.csv`
  - `random_direction_results.csv`
  - `rank_propagation.csv`
  - `attention_diagnostics.csv`
  - `tiny_image_results.parquet`
  - `small_image_results.parquet`
  - `experiment_manifest.json`
- **Figures Directory**: `figures/fungibility_v0_9/`
  - `eigenvalue_spectrum.png`
  - `natural_vs_energy_matched_rank.png`
  - `accuracy_vs_natural_rank.png`
  - `pc_identity_comparison.png`
  - `pc1_amplitude_sweep.png`
  - `rank_expansion_through_blocks.png`
  - `attention_recovery_through_blocks.png`
