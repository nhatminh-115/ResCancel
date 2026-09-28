# Patch Content Fungibility V0.8: Protocol Specification
## Token-Diversity / Effective-Rank Sufficiency Test

**Author:** DeepMind Antigravity Team  
**Date:** September 28, 2026  
**Repository:** [https://github.com/nhatminh-115/ResCancel](https://github.com/nhatminh-115/ResCancel)  
**Status:** FROZEN BEFORE EXECUTION  

---

## 1. Scientific Context & Motivation

In Patch Content Fungibility V0.7, we established two foundational facts:
1. **The Regional Late-Layer Centroid:** The static prototype effect is not an isolated Block-8 anomaly (H1 refuted), but reflects a late-layer representation manifold shared across Blocks 7–9. Up to 75% spatial patch replacement, a static centroid preserves over $94\%$ of baseline classifications.
2. **The 100% Replacement Divergence:** When ALL 196 spatial patch tokens are replaced simultaneously at Depth 8:
   - **Static Centroid ($\mu_8$):** Collapses severely to **10.2%** on DeiT-Tiny and **17.0%** on DeiT-Small.
   - **Held-Out Gaussian:** Preserves substantially higher performance: **26.5%** on DeiT-Tiny and **46.7%** on DeiT-Small.

### The Central Question of V0.8
Why does stochastic Gaussian replacement substantially outperform the static prototype at 100% patch replacement?
We hypothesize that injecting 196 identical copies of $\mu_8$ causes spatial self-attention degeneracies (e.g. rank-1 collapse where all Key and Value projections are identical), whereas stochastic sampling provides token-to-token representational diversity.

However, this mechanism has not yet been causally isolated. We must test three competing explanations:
- **H1 — Token-Diversity / Effective-Rank Requirement:** Downstream self-attention requires some degree of representational diversity across patch positions. The exact image content is dispensable, but a completely rank-collapsed patch stream causes attention failure.
- **H2 — Gaussian-Specific Distributional Effect:** Gaussian sampling succeeds because of its specific coordinate-wise variance structure or probability density, not because of rank or diversity per se.
- **H3 — Simple Perturbation Energy:** Any non-zero perturbation energy around $\mu_8$ rescues performance, independent of effective dimensionality or rank.

V0.8 executes a **mechanistic falsification experiment** to distinguish these hypotheses. No training, fine-tuning, routers, or architectures will be introduced.

---

## 2. Experimental Setup & Controls

- **Models:**
  - `deit_tiny_patch16_224` ($D = 192$, 12 blocks)
  - `deit_small_patch16_224` ($D = 384$, 12 blocks)
  - Public `timm` checkpoints, strictly frozen weights ($\nabla_\theta = 0$, eval mode).
- **Disjoint Dataset Splits (ImageNet-1k Validation Set):**
  - **Calibration Set ($N_{\text{calib}} = 1,000$):** Exactly reused from `outputs/fungibility_v0_6/calibration_split.csv` (seed `9101`).
  - **Evaluation Set ($N_{\text{eval}} = 1,000$):** Exactly reused from `outputs/fungibility_v0_6/evaluation_split.csv` (seed `9201`).
  - Strict zero overlap: $\text{IDs}_{\text{calib}} \cap \text{IDs}_{\text{eval}} = \emptyset$.
  - Evaluation image activations are strictly never used to estimate moments, PCA bases, or prototypes.
- **Intervention Depth:**
  - Primary experiment: **Depth 8 only** (output of Block 8 / input to Block 9).
  - Downstream blocks 9–11 + `norm` + `forward_head` are strictly identical across conditions.
- **Intervention Scope:**
  - Primary experiment: **100% replacement** of all 196 spatial patch tokens ($t \in \{1, \dots, 196\}$).
  - CLS token (sequence index 0) remains strictly **untouched** at injection ($h'_{8,0} = h_{8,0}$).
  - Secondary confirmation: **75% replacement** (147/196 tokens) on a reduced set.

---

## 3. Tested Conditions & Interventions (100% Patch Stream)

At Depth 8, replacing all 196 spatial patch tokens:

### 3.1 Reference Conditions
1. **Clean Baseline:** Untouched evaluation pass.
2. **Zero Ablation:** $h'_{8,t} = 0$ for all $t \ge 1$.
3. **Static Centroid ($\mu_8$):** $h'_{8,t} = \mu_8$ for all $t \ge 1$. Reference rank-1 collapsed condition.
4. **Held-Out Diagonal Gaussian:** $h'_{8,t}[d] \sim \mathcal{N}(\mu_{8,d}, \sigma_{8,d}^2)$ for $t \ge 1$, evaluated across 5 frozen seeds: `11001, 11002, 11003, 11004, 11005`. Full-diversity reference.

### 3.2 Total Variance Energy Matching Principle
Let the empirical calibration variance energy be:
$$E_{\text{full}} = \sum_{d=0}^{D-1} \sigma_{8,d}^2$$
For all controlled-rank and noise conditions, we mandate:
$$\mathbb{E}[\|\epsilon_t\|_2^2] \approx E_{\text{full}}$$
This ensures that different rank conditions receive identical total perturbation energy, isolating the causal effect of representation rank and diversity from noise magnitude. Realized perturbation energy across all $196,000$ evaluation patch tokens must match $E_{\text{full}}$ within $2\%$.

### 3.3 Controlled Rank Sweep via Calibration PCA
Using centered Block-8 calibration patch representations $x_t^{\text{calib}} - \mu_8$, compute the empirical covariance matrix $\Sigma_{\text{calib}} \in \mathbb{R}^{D \times D}$ and its eigendecomposition:
$$\Sigma_{\text{calib}} = V \Lambda V^\top, \quad \Lambda = \text{diag}(\lambda_1, \dots, \lambda_D), \quad \lambda_1 \ge \dots \ge \lambda_D$$
For target rank $R \in \{1, 2, 4, 8, 16, 32, 64, \text{FULL}\}$:
- Take top-$R$ eigenvectors $V_R = [v_1, \dots, v_R] \in \mathbb{R}^{D \times R}$.
- Let $S_R = \sum_{r=1}^R \lambda_r$. To match total variance energy $E_{\text{full}}$, set coefficient variance:
  $$\sigma_{a,r}^2 = \frac{E_{\text{full}}}{S_R} \lambda_r$$
- Sample token coefficients independently: $a_{t,r} \sim \mathcal{N}(0, \sigma_{a,r}^2)$ for $t=1\dots196, r=1\dots R$.
- Form replacement:
  $$h'_{8,t} = \mu_8 + \sum_{r=1}^R a_{t,r} v_r$$
This defines the **PCA-RANK-$R$** condition.

### 3.4 Matched Random-Subspace Control
For each target rank $R \in \{1, 2, 4, 8, 16, 32, 64\}$:
- Generate a random orthonormal basis $U_R \in \mathbb{R}^{D \times R}$ via QR decomposition of standard normal Gaussian matrices.
- Sample token coefficients independently: $a_{t} \sim \mathcal{N}(0, \frac{E_{\text{full}}}{R} I_R)$.
- Form replacement:
  $$h'_{8,t} = \mu_8 + U_R a_t$$
Evaluated with 3 frozen seeds: `12001, 12002, 12003`.
Directly tests whether token diversity must align with learned visual covariance or whether generic subspace dimensionality suffices.

### 3.5 Isotropic Token Noise Control
- $h'_{8,t} = \mu_8 + \epsilon_t$, where $\epsilon_t \sim \mathcal{N}(0, \frac{E_{\text{full}}}{D} I_D)$.
- Preserves full dimensional rank ($D$), but destroys feature-wise variance anisotropy.
- Evaluated with 3 frozen seeds: `13001, 13002, 13003`.
- Tests whether coordinate-wise variance structure is necessary beyond full dimensional rank.

### 3.6 Direct Token Diversity Test: Shared vs. Independent Noise
Directly separates perturbation energy from token-to-token independence:
- **SHARED-NOISE:** Sample a SINGLE noise vector $\epsilon_{\text{shared}} \sim \mathcal{N}(0, \text{diag}(\sigma_8^2))$ per evaluation image, and inject:
  $$h'_{8,t} = \mu_8 + \epsilon_{\text{shared}} \quad \forall t \in \{1, \dots, 196\}$$
  All 196 patch tokens remain mathematically identical at injection (zero spatial diversity, rank-1).
- **INDEPENDENT-NOISE:** Sample independent noise vectors $\epsilon_t \sim \mathcal{N}(0, \text{diag}(\sigma_8^2))$ for each patch $t$.
Both conditions have identical marginal distribution, identical expected token norm, and identical noise energy. Only spatial independence differs.
Evaluated with 5 frozen seeds: `14001, 14002, 14003, 14004, 14005`.

### 3.7 Grouped-Diversity Control (Unique Token Count $K$)
Construct replacement streams containing exactly $K$ unique vector prototypes:
$$K \in \{1, 2, 4, 8, 16, 32, 64, 196\}$$
- For each image, sample $K$ independent Gaussian vectors: $g_1, \dots, g_K \sim \mathcal{N}(\mu_8, \text{diag}(\sigma_8^2))$.
- Assign the 196 spatial positions deterministically to these $K$ vectors ($t \mapsto g_{(t-1) \bmod K + 1}$).
- At $K=1$, all 196 patches are identical (shared noise).
- At $K=196$, all 196 patches are independent.
Evaluated with 3 frozen seeds: `15001, 15002, 15003`.
Tests the minimum number of unique tokens required to rescue downstream attention.

### 3.8 Secondary 75% Confirmation
Repeat on a reduced condition set at 75% replacement (147/196 tokens):
- Clean, Zero, Static Centroid ($\mu_8$), Full Gaussian
- PCA Ranks $R \in \{1, 4, 16\}$
- Grouped Diversity $K \in \{1, 8, 196\}$
Tests whether diversity requirements only emerge under 100% replacement or scale continuously.

---

## 4. Quantitative Measurements & Diagnostics

### 4.1 Representation Rank Diagnostics
At injection (Depth 8 output) and after each downstream block (Block 9, 10, 11 outputs), extract the spatial patch representation matrix $H_{\text{patch}} \in \mathbb{R}^{196 \times D}$.
Compute centered representation: $H_c = H_{\text{patch}} - \text{mean}_t(H_{\text{patch}})$.
Compute singular values $s_1 \ge \dots \ge s_{\min(196, D)}$.
Record:
1. **Numerical Rank:** Count of $s_i$ such that $s_i / s_1 > 10^{-3}$.
2. **Stable Rank:** $r_{\text{stable}} = \frac{\|H_c\|_F^2}{\|H_c\|_2^2} = \frac{\sum_i s_i^2}{s_1^2}$.
3. **Participation Ratio (Effective Rank):**
   $$r_{\text{eff}} = \frac{(\sum_i s_i^2)^2}{\sum_i s_i^4}$$
4. **Mean Pairwise Cosine Similarity:** $\frac{1}{\binom{196}{2}} \sum_{i < j} \frac{h_i \cdot h_j}{\|h_i\| \|h_j\|}$.

### 4.2 Attention Diversity Diagnostics
For Blocks 9, 10, 11 and each attention head:
1. **Attention Matrix Stable Rank:** Stable rank of the softmax attention matrix $A \in \mathbb{R}^{197 \times 197}$.
2. **CLS-to-Patch Attention Entropy:**
   $$H(A_{0, 1:196}) = -\sum_{t=1}^{196} A_{0, t} \log (A_{0, t} + \epsilon)$$
3. **Patch Key Variance:** $\frac{1}{D_h} \sum_{d=1}^{D_h} \text{Var}_t(K_{t, d})$.
4. **Patch Value Variance:** $\frac{1}{D_h} \sum_{d=1}^{D_h} \text{Var}_t(V_{t, d})$.

### 4.3 Classification & Statistical Metrics
- Unit of independence: Evaluation image ($N_{\text{eval}} = 1,000$).
- Metrics: True-class logit margin, margin damage ($m_{\text{clean}} - m_{\text{cond}}$), Top-1 accuracy, Top-1 prediction flip rate.
- Paired statistical tests: Paired Student's $t$-test, Wilcoxon signed-rank test, 10,000 bootstrap CI for $\Delta m$, Cohen's $d_z$, exact McNemar test.
- Multiple comparisons: Benjamini-Hochberg FDR correction applied within logical families.
- Spearman correlation ($\rho$) between effective representation rank and mean accuracy / margin.
- Descriptive saturation metrics:
  - $R_{95}$: Smallest tested PCA rank reaching $\ge 95\%$ of full Gaussian accuracy relative to static centroid.
  - $K_{95}$: Smallest tested unique token count reaching $\ge 95\%$ of full Gaussian accuracy relative to static centroid.

---

## 5. Pre-Registered Decision Rules

### OUTCOME A — DIVERSITY DOES NOT EXPLAIN FAILURE
Declare **OUTCOME A** if:
- Independent noise does not materially outperform shared noise ($|d_z| < 0.20$ or $\Delta \text{Acc} < 2\%$),
- Controlled rank increases do not produce systematic recovery,
- Grouped unique-token count does not systematically improve performance.
- **Interpretation:** The 100% centroid failure cannot be explained primarily by token-level diversity or representation rank. The rank-collapse explanation is rejected.

### OUTCOME B — GENERIC TOKEN DIVERSITY
Declare **OUTCOME B** if:
- Shared Noise performs poorly, while Independent Noise strongly improves performance ($d_z \ge 0.20$, $\Delta \text{Acc} \ge 5\%$),
- Increasing rank / $K$ produces systematic monotonic recovery,
- BUT Random subspaces perform approximately as well as PCA subspaces ($|d_z| < 0.20$ and $\Delta \text{Acc} < 2\%$),
- AND Isotropic full-rank noise performs approximately as well as Diagonal Gaussian.
- **Interpretation:** Downstream self-attention requires token-level diversity to avoid rank collapse, but the diversity need not follow the learned activation geometry closely.

### OUTCOME C — STRUCTURED LOW-RANK DIVERSITY
Declare **OUTCOME C** only if **BOTH** DeiT-Tiny and DeiT-Small show:
1. Independent Noise substantially outperforms Shared Noise ($d_z \ge 0.20$, $p < 0.001$, $\Delta \text{Acc} \ge 5\%$);
2. Controlled PCA rank increase yields monotonic or near-monotonic performance recovery;
3. Performance saturates at a low effective rank ($R_{95} \le 16$ or $K_{95} \le 16$);
4. PCA subspaces substantially outperform matched Random subspaces at intermediate ranks ($d_z \ge 0.20$);
5. Diagonal Gaussian outperforms Isotropic noise OR PCA geometry clearly matters;
6. Results replicate across random seeds.
- **Interpretation:** Downstream Blocks 9–11 require token-level diversity, but only a low-dimensional structured subspace of late-layer patch variation is sufficient.

### OUTCOME D — HIGH-DIMENSIONAL STRUCTURE REQUIRED
Declare **OUTCOME D** if:
- Token diversity clearly matters (Independent >> Shared),
- BUT performance only recovers near full dimensional rank ($R \ge 64$),
- Low-rank PCA conditions remain substantially degraded.
- **Interpretation:** Exact image content is dispensable, but downstream computation depends on high-dimensional late-layer activation variation.

---

## 6. Interpretation Guardrails

1. **NO Claim of Whole-Model Disregard of Visual Content:**
   Do NOT write: *"The model does not need visual content."* The original image shaped CLS and patch representations through Blocks 0–8 before intervention.
2. **NO Unsupported Attention Claims:**
   Do NOT claim attention rank collapse unless attention diagnostics (Key/Value variance, attention entropy, attention stable rank) directly demonstrate it.
3. **Mandatory Permitted Phrasing:**
   *"After Block 8, downstream classification is largely insensitive to exact patch-specific content but remains sensitive to the diversity structure of the replacement patch stream."*

---

## 7. Pre-Registered Programmatic Validations

Before issuing any scientific verdict, the following 18 assertions must pass:
1. Backbone weights strictly frozen ($\nabla_\theta = 0$, hash invariant).
2. Exact V0.6/V0.7 calibration ($N=1,000$) and evaluation ($N=1,000$) splits reused.
3. Evaluation images never used for calibration moments, PCA bases, or prototypes.
4. PCA computed exclusively on calibration patch activations without class labels or loss gradients.
5. CLS token untouched at injection ($h'_{8,0} == h_{8,0}$).
6. Exactly all 196 patches replaced in primary experiment.
7. Low-rank bases match requested ranks $R \in \{1, 2, 4, 8, 16, 32, 64\}$.
8. PCA basis vectors are strictly orthonormal ($V_R^\top V_R = I_R$ within $10^{-5}$).
9. Random basis vectors are strictly orthonormal ($U_R^\top U_R = I_R$ within $10^{-5}$).
10. Total expected perturbation energy matched across all rank conditions within $2\%$ of $E_{\text{full}}$.
11. Shared-noise and independent-noise use identical noise distributions.
12. Grouped conditions contain exactly the requested $K$ unique vectors.
13. Seeds exactly match protocol (`11001-11005`, `12001-12003`, `13001-13003`, `14001-14005`, `15001-15003`).
14. All replacements are label-free and gradient-free.
15. Effective rank calculated from actual resulting tensors.
16. Attention diagnostics measured from actual Blocks 9–11 forward passes.
17. Statistics use exactly $N=1,000$ independent evaluation images.
18. Scientific report matches machine-readable parquet/CSV outputs.
