# DELTA TRANSPORT V0.5: Matched-Oracle Specificity Falsification Protocol

**Document Status:** FROZEN  
**Date:** 2026-09-28  
**Repository:** `https://github.com/nhatminh-115/ResCancel`  
**Research Line:** Delta Transport V0.5 — Matched-Oracle Specificity Falsification in Vision Transformers  

---

## 1. Scientific Objective & The Capacity Confound

In Delta Transport V0, the Tokenwise Historical Oracle demonstrated a large, statistically significant advantage over the Global Historical Oracle across both DeiT-Tiny ($\Delta m = +0.2521, d_z = 1.761$) and DeiT-Small ($\Delta m = +0.1833, d_z = 1.495$).

However, an independent audit identified a fundamental confound:
> **The Capacity Asymmetry:**
> The Tokenwise Oracle is granted **196 independent choices**, each selecting the best of **8 candidate directions** via the analytical true-label margin gradient $g_t$. In contrast, the Global Oracle is constrained to **1 decision per image** (choosing 1 of 8 layers shared globally).
> 
> Therefore, the observed V0 advantage may reflect **generic oracle optimization degrees-of-freedom** rather than meaningful, sample-specific or spatially specific information residing within historical residual deltas.

**DELTA TRANSPORT V0.5** resolves this confound through a **matched-capacity falsification test**. Every condition in V0.5 receives:
- Exactly 196 patch-level decisions ($t \in \{1, \dots, 196\}$);
- Exactly 8 candidate directions per patch;
- The exact same analytical true-label margin gradient $g_t = \frac{\partial m}{\partial h_{8,t}}$;
- The exact same first-order argmax selection rule: $k^*_t = \arg\max_{k \in \{0..7\}} g_t^T u_{k,t}$;
- The exact same transport perturbation magnitude: $h'_t = h_t + \gamma \|h_t\|_2 u_{k^*_t, t}$;
- The exact same injection point (input to Block 9) and downstream propagation (Blocks 9–11);
- CLS token strictly untouched at injection ($h'_{8,\text{CLS}} = h_{8,\text{CLS}}$);
- Fully frozen pretrained backbone weights.

**The ONLY factor that changes is the structural origin of the 8 candidate directions.**

---

## 2. Experimental Setup & Frozen Configurations

### 2.1 Models & Datasets
- **Models:** Standard pretrained checkpoints from `timm`:
  - `deit_tiny_patch16_224` (12 blocks, $d=192$, heads=3)
  - `deit_small_patch16_224` (12 blocks, $d=384$, heads=6)
  - Zero backbone parameter modifications ($\theta$ frozen).
- **Dataset:** Reproducible ImageNet-1k validation subset, $N = 1,000$ images (stratified 1 per class), seed $42$.
- **Injection Point:** Input to Block 9 (after Block 8), candidate historical source bank Blocks $0..7$.
- **Primary Budget:** $\gamma = 0.05$. Sensitivity budgets: $\gamma \in \{0.02, 0.10\}$ (labelled secondary).

### 2.2 Gradient & Selection Rule
For an image with label $y$, baseline logits $z$, and true-class margin $m = z_y - \max_{c \neq y} z_c$:
1. Analytical margin gradient at input to Block 9:
   $$g_t = \frac{\partial m}{\partial h_{8,t}} \in \mathbb{R}^d \quad \text{for } t \in \{1, \dots, 196\}$$
2. For any candidate bank $U_t = \{u_{0,t}, \dots, u_{7,t}\}$ where each $\|u_{k,t}\|_2 = 1$:
   $$s_{k,t} = g_t^T u_{k,t}$$
   $$k^*_t = \arg\max_{k \in \{0, \dots, 7\}} s_{k,t}$$
3. Perturbation:
   $$h'_{8,t} = h_{8,t} + \gamma \|h_{8,t}\|_2 u_{k^*_t, t} \quad (t \in \{1..196\}); \quad h'_{8,\text{CLS}} = h_{8,\text{CLS}}$$

---

## 3. Compared Conditions: Reference & Matched-Capacity Controls

### Condition 0: TRUE HISTORICAL TOKENWISE ORACLE (Reference)
- **Candidate Source:** Historical block residual deltas from the **same image** and **same spatial patch**:
  $$\Delta_{l,t} = h_{l+1,t} - h_{l,t} \quad \text{for } l \in \{0, \dots, 7\}$$
  $$u_{l,t} = \frac{\Delta_{l,t}}{\|\Delta_{l,t}\|_2 + 10^{-7}}$$
- Represents the hypothesis that past residual updates contain specific, reusable directional information.

---

### Condition 1: RANDOM-DIRECTION TOKENWISE ORACLE (Primary Control)
- **Scientific Purpose:** Tests whether the oracle's advantage is merely an artifact of having 8 random directions to project gradient $g_t$ onto.
- **Candidate Source:** For each patch $t$, draw 8 random vectors from standard Gaussian:
  $$r_{k,t} \sim \mathcal{N}(0, I_d) \quad \text{for } k \in \{0, \dots, 7\}$$
  $$u^{\text{rand}}_{k,t} = \frac{r_{k,t}}{\|r_{k,t}\|_2}$$
- The oracle independently selects the best random direction for every patch:
  $$k^*_t = \arg\max_{k \in \{0..7\}} g_t^T u^{\text{rand}}_{k,t}$$
- **Frozen Seeds:** Evaluated across 5 frozen seeds: `2501, 2502, 2503, 2504, 2505`.

---

### Condition 2: CROSS-IMAGE HISTORICAL TOKENWISE ORACLE (Secondary Control)
- **Scientific Purpose:** Tests whether the benefit requires historical deltas from the *same input image*, or whether residual deltas from unrelated images provide identical utility.
- **Candidate Source:** For each target image $i \in \{0, \dots, 999\}$, pair with a deterministic donor image $j \neq i$ via a fixed pseudo-random permutation of the 1,000 images using seed `3501`.
  $$u^{\text{cross}}_{l,t}(i) = \frac{\Delta_{l,t}(\text{donor } j)}{\|\Delta_{l,t}(\text{donor } j)\|_2 + 10^{-7}}$$
- The target image supplies $h_t(i)$, $g_t(i)$, and label $y_i$. The oracle chooses the best donor delta for each patch:
  $$l^*_t = \arg\max_{l \in \{0..7\}} g_t(i)^T u^{\text{cross}}_{l,t}(i)$$
- *Assertion:* Strictly verify donor image $j \neq i$ for all $i$.

---

### Condition 3: SPATIALLY SHUFFLED HISTORICAL TOKENWISE ORACLE (Secondary Control)
- **Scientific Purpose:** Tests whether the historical delta must correspond to the *same spatial patch location*, or whether spatial correspondence is irrelevant.
- **Candidate Source:** Within each image and historical layer $l \in \{0, \dots, 7\}$, randomly permute the 196 spatial patch positions:
  $$\pi_l \in \mathcal{S}_{196}$$
  $$u^{\text{shuffled}}_{l,t} = u_{l, \pi_l(t)}$$
- Target patch $t$ receives 8 historical candidates originating from scrambled spatial positions within the same image.
- **Frozen Seeds:** Evaluated across 3 frozen seeds: `4501, 4502, 4503`.

---

### Condition 4: UNCOUPLED POOLED DELTA BANK (Secondary Control)
- **Scientific Purpose:** Tests whether destroying layer depth identity while preserving the empirical delta distribution affects oracle utility.
- **Candidate Source:** For each image and patch, sample 8 real delta vectors uniformly at random from the pool of all $8 \times 196$ deltas of that image without layer alignment.
- **Frozen Seed:** `5501`.

---

## 4. Programmatic Validation Assertions

Before any scientific verdict is rendered, the pipeline must verify:
1. Backbone parameter weights $\theta$ are verified unchanged via SHA-256 hash.
2. Identical 1,000 image IDs are used in all conditions.
3. Every oracle receives exactly 8 candidate directions per patch.
4. Every oracle independently selects exactly 1 candidate for each of 196 patches.
5. Historical and control policies receive the **exact same gradient tensor $g_t$**.
6. All candidate vectors are strictly L2-normalized ($\|u_{k,t}\|_2 = 1$).
7. Perturbation magnitude strictly matches $\gamma \|h_t\|_2$ across all conditions.
8. CLS token is untouched at injection ($h'_{8,\text{CLS}} = h_{8,\text{CLS}}$).
9. Cross-image donor index $j \neq i$ strictly holds for all $i \in \{0..999\}$.
10. Spatial shuffle is a valid permutation of $\{1..196\}$ ($\pi_l$ is bijective).
11. Random controls use frozen seeds (`2501..2505`, `3501`, `4501..4503`, `5501`).
12. All statistical tests operate on $N = 1,000$ independent image-level observations.
13. Report numbers match saved machine-readable CSVs and manifest exactly.

---

## 5. Statistical Framework & Primary Estimands

The primary comparison is:
$$\text{TRUE HISTORICAL TOKENWISE} \quad \text{vs.} \quad \text{RANDOM-DIRECTION TOKENWISE ORACLE}$$

- **Unit of Independence:** The image ($N = 1,000$ independent units per architecture).
- **Multi-Seed Aggregation:** For multi-seed controls (Random and Shuffled), evaluate each seed independently, then report seed-wise effects, mean effect, and cross-seed variability. Do NOT pool seeds as independent observations.
- **Pairwise Estimands (Historical vs. Control):**
  - Paired mean margin difference: $\Delta m = m_{\text{hist}} - m_{\text{ctrl}}$
  - Median margin difference
  - 10,000-resample Bootstrap 95% Confidence Interval
  - Paired Student's $t$-test and Wilcoxon signed-rank test
  - Cohen's $d_z = \frac{\bar{D}}{s_D}$
  - Top-1 accuracy difference: $\Delta \text{acc} = \text{acc}_{\text{hist}} - \text{acc}_{\text{ctrl}}$
  - McNemar's exact binomial test for paired classification discordance.

### Re-evaluation of Spatial Entropy
For every condition, compute:
- Shannon Entropy: $H = -\sum_{k=0}^7 p(k) \log_2 (p(k) + \epsilon)$
- Dominant candidate fraction: $\max_k p(k)$
- Number of distinct candidate indices selected per image.

*Statistical Note:* If the Random-Direction Oracle also exhibits near-maximal entropy ($H \approx 2.7 - 3.0$), high entropy is formally established as an intrinsic property of independent argmax selection over 8 directions in high dimension, and CANNOT be cited as evidence of spatial specialization.

---

## 6. Pre-Registered Decision Rules

### OUTCOME A — CAPACITY ARTIFACT / KILL
Triggered if:
- True Historical Tokenwise does NOT materially outperform the matched Random-Direction Tokenwise Oracle;
- Operationally:
  1. Historical-minus-Random margin 95% CI includes zero on both architectures; OR
  2. Cohen's $d_z < 0.20$ on both architectures; OR
  3. Historical accuracy advantage is negligible or negative ($\Delta \text{acc} \le 0$); OR
  4. Random-Direction Oracle reproduces $\ge 80\%$ of the V0 Historical-over-Baseline margin gain.

*Scientific Conclusion:* "The V0 advantage was largely an artifact of gradient-guided per-token selection degrees-of-freedom rather than meaningful historical residual information."  
*Action:* **KILL the hypothesis.** Do not design or train a router.

---

### OUTCOME B — GENERIC DELTA DISTRIBUTION
Triggered if:
- Historical deltas beat pure random directions ($d_z \ge 0.20$), BUT cross-image and/or spatially shuffled historical deltas retain nearly all ($\ge 85\%$) of the benefit.

*Scientific Conclusion:* "Residual-delta-like directions are useful candidate perturbations, but evidence for sample-specific spatial historical retrieval is weak."  
*Action:* Reject sample-specific spatial routing. Do not proceed to V1 spatial router.

---

### OUTCOME C — HISTORICAL-SPECIFIC SIGNAL SURVIVES
Triggered **ONLY** if True Historical Tokenwise:
1. Beats Random-Direction Tokenwise Oracle on **BOTH** DeiT-Tiny and DeiT-Small;
2. Bootstrap 95% CI for Historical - Random is strictly above zero on both;
3. Effect size $d_z \ge 0.20$ on both;
4. Beats Cross-Image Oracle meaningfully on both architectures;
5. Beats Spatially Shuffled Oracle meaningfully on both architectures;
6. Maintains a clean Top-1 accuracy advantage over all matched controls;
7. Effect is consistent across all control seeds.

*Scientific Conclusion:* "Useful oracle headroom cannot be explained solely by per-token gradient selection capacity. Same-image, spatially aligned historical residual deltas contain specific, reusable information."  
*Action:* Justifies proceeding to **Delta Transport V1** (test-time learned router).
