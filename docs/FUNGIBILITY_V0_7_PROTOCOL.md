# Patch Content Fungibility V0.7: Protocol Specification
## Prototype Specificity and Complete Patch-Stream Replacement Test

**Author:** DeepMind Antigravity Team  
**Date:** September 28, 2026  
**Repository:** [https://github.com/nhatminh-115/ResCancel](https://github.com/nhatminh-115/ResCancel)  
**Status:** FROZEN BEFORE EXECUTION  

---

## 1. Scientific Context & Motivation

In Patch Content Fungibility V0.6, we established three key empirical facts:
1. **Held-Out Generalization:** Activation statistics estimated on $N=1,000$ disjoint calibration images completely replicated the Block-8 replacement effect on $N=1,000$ independent evaluation images.
2. **Coherent Depth Transition:** Model sensitivity to patch replacement transitions smoothly from fragile intermediate blocks (5–6) through a highly tolerant mid-to-late window (7–9) to near-complete causally unperturbed terminal blocks (10).
3. **The Static Global Prototype Discovery:** Replacing masked patch representations with a single static calibration mean vector $\mu_l$ (with zero stochastic noise) matched or slightly outperformed stochastic Gaussian sampling:
   - DeiT-Tiny ($l=8, f=25\%$): Clean $67.9\%$, Zero $59.2\%$, Held-Out Gaussian $66.9\%$, Global Mean $\mu_8$ **$67.6\%$**.
   - DeiT-Small ($l=8, f=25\%$): Clean $76.1\%$, Zero $68.4\%$, Held-Out Gaussian $75.6\%$, Global Mean $\mu_8$ **$76.1\%$** (exact baseline retention).

This discovery fundamentally reframes the research question. Stochastic token diversity is unnecessary. A single static prototype vector preserves classification.
However, why does $\mu_8$ function? We must distinguish four competing hypotheses:
- **H1 — Depth-Specific Prototype:** Downstream Blocks 9–11 expect a specific feature-coordinate activation template aligned with Block 8's causal manifold.
- **H2 — Generic Centroid / Bias Effect:** Any centroid vector from the network or a coarse bias alignment suffices.
- **H3 — Norm / Low-Dimensional Property:** Only vector magnitude or low-dimensional projection properties matter.
- **H4 — Prototype Not Actually Necessary:** At late depth, patch content is already nearly irrelevant and almost any non-destructive vector works.

---

## 2. Experimental Design & Protocol Controls

This remains a strictly **mechanistic falsification experiment**.
- **No Training or Fine-Tuning:** Backbone weights are strictly frozen ($\nabla_\theta = 0$).
- **No Optimization:** No prototype vectors are derived via loss gradients or labels.
- **Models:**
  - `deit_tiny_patch16_224` ($D=192$, 12 blocks)
  - `deit_small_patch16_224` ($D=384$, 12 blocks)
  - Standard public checkpoints from `timm`, eval mode.
- **Disjoint Split Reuse:**
  - **Calibration Set ($N_{\text{calib}} = 1,000$):** Exactly reused from `outputs/fungibility_v0_6/calibration_split.csv` (seed `9101`).
  - **Evaluation Set ($N_{\text{eval}} = 1,000$):** Exactly reused from `outputs/fungibility_v0_6/evaluation_split.csv` (seed `9201`).
  - Strict zero overlap: $\text{IDs}_{\text{calib}} \cap \text{IDs}_{\text{eval}} = \emptyset$.
  - Evaluation image activations are strictly never used to construct prototypes.
- **Target Depth:**
  - Primary experiment: **Depth 8 only** (output of Block 8 / input to Block 9).
  - Secondary confirmation: Depths 7 and 9 (evaluated post-primary if required).
- **Intervention Fractions & Nested Masks:**
  - Tested fractions: **25% (49), 50% (98), 75% (147), 100% (196)**.
  - Deterministic spatial patch ordering generated with seed `9601`.
  - Nested mask invariant:
    $$\mathcal{M}_{49} \subset \mathcal{M}_{98} \subset \mathcal{M}_{147} \subset \mathcal{M}_{196}$$
  - Patch indices mapped to sequence positions: `idx + 1`.
  - CLS token (index 0) remains strictly **untouched** at injection during primary patch interventions ($h'_{8,0} = h_{8,0}$).
  - In the **100% condition**, all 196 spatial patch tokens are replaced simultaneously.

---

## 3. Tested Conditions & Interventions

At Depth 8 across each fraction $f \in \{25\%, 50\%, 75\%, 100\%\}$:

### 3.1 Primary Conditions
1. **Clean Baseline:** Untouched network evaluation.
2. **Zero Ablation:** $h'_{8,t} = 0$ for $t \in \mathcal{M}_f$.
3. **Correct-Depth Global Mean ($\mu_8$):**
   $h'_{8,t} = \mu_8$, where $\mu_8 = \frac{1}{N_{\text{calib}} \times 196} \sum_{i,s} h_{i,8,s}^{\text{calib}}$.
   Key reference prototype.
4. **Held-Out Diagonal Gaussian:**
   $h'_{8,t}[d] \sim \mathcal{N}(\mu_{8,d}, \sigma_{8,d}^2)$, evaluated across 3 frozen seeds: `9701, 9702, 9703`.

### 3.2 Specificity Controls
5. **Wrong-Depth Means (Raw):**
   Inject calibration mean vectors from mismatched depths:
   $$\mu_k \quad \text{for } k \in \{5, 6, 7, 9, 10\}$$
   Directly tests whether the prototype is depth-specific.
6. **Norm-Matched Wrong-Depth Means:**
   To isolate feature direction from activation magnitude differences across depth, define:
   $$\mu'_k = \frac{\|\mu_8\|_2}{\|\mu_k\|_2} \mu_k \quad \text{for } k \in \{5, 6, 7, 9, 10\}$$
   Guarantees $\|\mu'_k\|_2 = \|\mu_8\|_2$ within $10^{-5}$.
7. **Coordinate-Permuted Mean:**
   $\mu_{\text{perm}}[d] = \mu_8[\pi(d)]$, where $\pi$ is a frozen random permutation of $\{0, \dots, D-1\}$.
   Evaluated with 3 seeds: `9801, 9802, 9803`.
   Preserves scalar values, mean, variance, and $L_2$ norm, but destroys coordinate-channel alignment.
8. **Sign-Flipped Means:**
   - Full inversion: $-\mu_8$ (preserves norm and axis magnitudes, inverts direction).
   - Partial sign inversion: randomly invert signs of 25%, 50%, and 75% of coordinate dimensions (deterministic seed `9901`).
9. **Prototype Scale Sweep:**
   Scale the correct prototype magnitude without altering direction:
   $$s \cdot \mu_8 \quad \text{for } s \in \{0.25, 0.50, 1.00, 2.00, 4.00\}$$
   Tests downstream sensitivity to global activation norm.
10. **Cosine-Controlled Directional Sweep:**
    Construct vectors $r_\alpha$ with controlled cosine similarity $\alpha \in \{0.0, 0.25, 0.50, 0.75\}$ to $\mu_8$ and exact norm $\|r_\alpha\|_2 = \|\mu_8\|_2$:
    - Let $\hat{\mu} = \mu_8 / \|\mu_8\|_2$.
    - Sample $v \sim \mathcal{N}(0, I_D)$, orthogonalize against $\hat{\mu}$: $v_\perp = v - (v \cdot \hat{\mu}) \hat{\mu}$, normalize $\hat{v} = v_\perp / \|v_\perp\|_2$.
    - Construct: $u_\alpha = \alpha \hat{\mu} + \sqrt{1 - \alpha^2} \hat{v}$.
    - Scale: $r_\alpha = \|\mu_8\|_2 \cdot u_\alpha$.
    - Evaluated with 3 frozen random seeds: `10001, 10002, 10003`.
11. **Exploratory Variance-Normalized Prototype:**
    $z_{\mu}[d] = \mu_{8,d} / (\sigma_{8,d} + \epsilon)$, rescaled to $\|z_{\mu,\text{rescaled}}\|_2 = \|\mu_8\|_2$. Tests raw coordinate values vs. variance-standardized direction.

### 3.3 Complete 100% Patch Replacement Test
- At $f = 100\%$ (all 196 spatial tokens replaced):
  - CLS token remains untouched at injection ($h'_{8,0} = h_{8,0}$).
  - Tested conditions: Zero, $\mu_8$, Held-Out Gaussian, best wrong-depth mean, coordinate-permuted, sign-flipped, cosine controls.
  - Measures whether Blocks 9–11 require any original patch representation after Block 8.

### 3.4 Secondary CLS Replacement Asymmetry Control (Exploratory)
- Keep patch stream 100% original ($h'_{8,t} = h_{8,t}$ for $t=1\dots196$).
- Replace CLS token ($h'_{8,0}$):
  1. Calibration mean CLS vector $\mu_{8,\text{cls}}$
  2. Cross-image CLS token
  3. Zero CLS vector ($h'_{8,0} = 0$)
- Directly tests stream asymmetry: Does downstream classification depend on image-specific CLS while patch tokens are prototype-replaceable?

---

## 4. Quantitative Metrics & Statistical Testing

- **Unit of Independence:** The evaluation image ($N_{\text{eval}} = 1,000$).
- **Per-Image Metrics:**
  - True-class margin: $m_i = z_{y_i} - \max_{c \neq y_i} z_c$
  - Margin damage: $\text{Damage}_C = \text{BaselineMargin} - \text{Margin}_C$
  - Correctness indicator ($0/1$).
  - Transition indicators: Correct $\to$ Incorrect (failure flip), Incorrect $\to$ Correct.
- **Prototype Specificity Contrast:**
  $$\text{Advantage}(\mu_8, \text{control}_k, f) = \text{Margin}(\mu_8, f) - \text{Margin}(\text{control}_k, f)$$
- **Prototype Retention:**
  $$\text{Retention}(f) = \frac{\text{Damage}(\text{zero}, f) - \text{Damage}(\mu_8, f)}{\text{Damage}(\text{zero}, f)}$$
  Computed only when $\text{Damage}(\text{zero}, f) \ge 0.10$. Sub-threshold conditions reported as `N/A`.
- **Statistical Significance Testing:**
  - 10,000 paired percentile bootstrap 95% Confidence Intervals for $\Delta m$.
  - Paired two-sided Student's $t$-test and Wilcoxon signed-rank test.
  - Paired effect size Cohen's $d_z = \text{mean}(\Delta m) / \text{std}(\Delta m)$.
  - McNemar exact two-sided test for Top-1 accuracy differences.
- **Multiple Comparison Correction:**
  - Benjamini-Hochberg (BH-FDR) applied within logically related contrast families:
    1. Wrong-depth means family ($k \in \{5, 6, 7, 9, 10\}$)
    2. Norm-matched wrong-depth means family
    3. Scale sweep family
    4. Cosine sweep family
    5. Sign-flip sweep family
    Families are strictly kept separate and not conflated.

---

## 5. Pre-Registered Decision Rules

### OUTCOME A — GENERIC REPLACEMENT / PROTOTYPE NOT SPECIAL
Declare **OUTCOME A** if:
- Wrong-depth means, coordinate-permuted mean, or low-cosine matched vectors perform approximately as well as $\mu_8$ across both architectures.
- Operationally: $\mu_8$ advantage over multiple destructive controls yields:
  $$|d_z| < 0.20 \quad \text{and} \quad |\Delta \text{Acc}| < 1.0\%$$
- **Interpretation:** The static replacement phenomenon is real, but the specific Block-8 prototype is not functionally special; downstream layers simply tolerate non-zero static fill.

### OUTCOME B — DEPTH-SPECIFIC PROTOTYPE
Declare **OUTCOME B** if **BOTH** DeiT-Tiny and DeiT-Small show:
1. $\mu_8$ significantly outperforms wrong-depth means ($d_z \ge 0.20$, FDR $q < 0.05$);
2. Norm-matching does not eliminate the advantage (direction/coordinates matter beyond norm);
3. Coordinate permutation substantially degrades performance ($d_z \ge 0.20$);
4. Sign inversion substantially degrades performance;
5. Margin degrades monotonically as cosine similarity to $\mu_8$ decreases;
6. Results remain robust across 25% and 50% replacement fractions.
- **Interpretation:** Downstream Blocks 9–11 are selectively compatible with a depth-specific feature-coordinate prototype.

### OUTCOME C — STRONG PROTOTYPE SUFFICIENCY
Declare **OUTCOME C** only if **Outcome B holds AND**:
1. At 75% and/or 100% patch replacement:
   - $\mu_8$ preserves substantial classification accuracy;
   - Destructive controls (Zero, coordinate permutation, sign inversion, low cosine) fail materially;
2. The finding replicates across both DeiT-Tiny and DeiT-Small.
- **Interpretation:** After Block 8, downstream classification requires surprisingly little exact patch-token-specific information and can operate using a largely static, depth-specific prototype patch stream.

---

## 6. Interpretation Guardrails & Prohibited Claims

1. **NO Claim of Whole-Model Disregard of Visual Content:**
   Because the original image influenced the CLS token through Blocks 0–8, V0.7 tests only whether Blocks 9–11 require exact patch representations *after* Block 8.
2. **NO Unsupported "Information Movement" Language:**
   Do NOT write:
   - *"The model no longer needs visual content."*
   - *"All information moved to CLS."*
   - *"Patches are discarded by the transformer."*
3. **Mandatory Permitted Phrasing:**
   - *"Downstream Blocks 9–11 show limited dependence on exact patch-token content after Block 8, provided the remaining patch stream has an appropriate depth-specific prototype structure."*

---

## 7. Pre-Registered Programmatic Validations

Before issuing any scientific verdict, the following 14 programmatic assertions must pass:
1. Backbone weights strictly frozen ($\nabla_\theta = 0$).
2. Same V0.6 calibration ($N=1,000$) and evaluation ($N=1,000$) split manifests reused.
3. Evaluation images never used to estimate calibration moments or construct prototypes.
4. All prototype vectors generated without class labels or loss gradients.
5. Exact patch intervention counts: 49, 98, 147, 196.
6. Nested mask invariant confirmed ($\mathcal{M}_{49} \subset \mathcal{M}_{98} \subset \mathcal{M}_{147} \subset \mathcal{M}_{196}$).
7. CLS token strictly untouched at injection during primary patch interventions.
8. $\mu_8$ exactly equals persisted calibration Block-8 mean vector.
9. Wrong-depth means loaded from their correct source depths ($l \in \{5, 6, 7, 9, 10\}$).
10. Norm-matched means match $\|\mu_8\|_2$ within tolerance $10^{-5}$.
11. Coordinate permutation is a strict bijection of $\{0, \dots, D-1\}$.
12. Sign-flip masks match requested fractions (25%, 50%, 75%, 100%).
13. Cosine-controlled vectors achieve target cosine $\alpha$ within tolerance $10^{-3}$ and norm within $10^{-5}$.
14. Prototype scale sweep factors strictly match $\{0.25, 0.50, 1.00, 2.00, 4.00\}$.
