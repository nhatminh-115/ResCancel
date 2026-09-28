# ResCancel V0: Pre-Registered Scientific Protocol & Freeze Document

**Project:** ResCancel — Mechanistic Falsification of Residual Cancellation in Pretrained Vision Transformers  
**Document Version:** 1.0.0 (Pre-Analysis Freeze)  
**Date:** September 2026  
**Status:** FROZEN BEFORE PRIMARY RESULT COLLECTION  

---

## 1. Executive Summary & Research Question

In pretrained Vision Transformers (ViTs), residual connections accumulate representations across self-attention and MLP sub-layers:
$$x_{\text{new}} = x + \delta$$

When the update vector $\delta$ points in an opposing direction to the incoming representation $x$, geometric cancellation and coordinate opposition occur. 

**Central Research Question:**  
*In pretrained Vision Transformers, are extreme residual-cancellation events merely a normal and useful part of standard computation, or is there a distinct subset that causally contributes to predictive fragility?*

This is a **mechanistic falsification experiment**, NOT a method-development experiment. We explicitly pre-register hypotheses, metrics, candidate event thresholds, confound controls, and causal intervention designs to prevent cherry-picking.

---

## 2. Models & Datasets

### 2.1 Pretrained Checkpoints
Standard public checkpoints from `timm`:
1. **DeiT-Tiny** (`deit_tiny_patch16_224`, 5.7M parameters, 12 layers, 3 heads, embed dim 192)
2. **DeiT-Small** (`deit_small_patch16_224`, 22.1M parameters, 12 layers, 6 heads, embed dim 384)

Both checkpoints are evaluated in `eval()` mode with standard ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`, resize 256, center crop 224).

### 2.2 Dataset Definition
- **Evaluation Dataset:** ImageNet-1k Validation set (reproducible subset of $N=1,000$ validation images sampled across classes with fixed seed `seed=42`).
- **Hardware & VRAM Budget:** Batch size $\le 32$ to maintain peak VRAM strictly $\le 8\text{ GB}$.

### 2.3 Perturbations for Fragility Testing
To measure predictive fragility beyond clean accuracy:
1. **Mild Gaussian Noise:** Additive zero-mean Gaussian noise with $\sigma = 0.08$ on normalized image tensors.
2. **Mild Gaussian Blur:** 2D Gaussian blur with kernel size $5 \times 5$ and $\sigma = 1.0$.
3. **Perturbation Metric:** Prediction flip rate:
   $$\text{Flip} = \mathbb{I}\left(\arg\max z_{\text{clean}} \neq \arg\max z_{\text{pert}}\right)$$

---

## 3. Residual Instrumentation & Metrics

For every transformer block $l \in \{0, \dots, 11\}$, instrument residual additions separately at:
1. **Self-Attention residual addition:** $x_1 = x_0 + \delta_{\text{attn}}$
2. **MLP residual addition:** $x_2 = x_1 + \delta_{\text{mlp}}$

Metrics are computed separately for:
- **CLS token** ($i = 0$)
- **Patch tokens** ($i \in \{1, \dots, 196\}$): summary statistics (mean, std, min, max, fraction of extreme events) per layer and site.

### 3.1 Primary Instrumentation Metrics
1. **Coordinate Cancellation ($C_{L1}$):**
   $$C_{L1} = \frac{\|x\|_1 + \|\delta\|_1}{\|x + \delta\|_1 + \epsilon}$$
   Measures elementwise sign opposition. $C_{L1} = 1.0$ indicates perfect sign agreement; $C_{L1} > 1.0$ indicates destructive coordinate cancellation.

2. **Geometric Anti-Alignment ($\cos(x, \delta)$):**
   $$\cos(x, \delta) = \frac{\langle x, \delta \rangle}{\|x\|_2 \|\delta\|_2 + \epsilon}$$
   Ranges from $-1$ (complete anti-alignment) to $+1$ (complete alignment).

3. **Relative Update Magnitude ($r$):**
   $$r = \frac{\|\delta\|_2}{\|x\|_2 + \epsilon}$$

4. **Residual Contraction ($q$):**
   $$q = \frac{\|x + \delta\|_2}{\|x\|_2 + \epsilon}$$
   Values of $q < 1.0$ indicate that the residual addition strictly reduces the Euclidean feature norm.

---

## 4. Frozen Candidate Event Definition

To prevent post-hoc threshold tuning, an extreme candidate cancellation event is **strictly pre-defined** by the conjunction of:
- **Strong negative cosine:** $\cos(x, \delta) \le -0.60$
- **Comparable update magnitude:** $r \ge 0.60$
- **Substantial contraction:** $q \le 0.85$
- **High coordinate cancellation:** $C_{L1} \ge 2.00$

Any residual event satisfying all 4 criteria is tagged as `EXTREME_CANCELLATION`.  
A secondary tier (`MODERATE_CANCELLATION`) is recorded for sensitivity checks ($\cos \le -0.40, r \ge 0.40, q \le 0.95$), but is explicitly designated as secondary.

---

## 5. Confound-Controlled & Matched Analysis

Cancellation may merely correlate with large updates or hard inputs. To eliminate confounds, we build matched control pairs for each candidate cancellation event:
- **Exact matching on discrete variables:**
  - Layer index $l \in \{0, \dots, 11\}$
  - Residual site (`attn` vs `mlp`)
  - Token type (`cls` vs `patch`)
- **Caliper matching on continuous confounds:**
  - Input norm $\|x\|_2$ (matched within $\pm 15\%$)
  - Delta norm $\|\delta\|_2$ (matched within $\pm 15\%$)
  - Clean logit margin $\Delta z = z_{\text{top1}} - z_{\text{top2}}$ (matched within $\pm 0.50$ logits)

**Statistical Test:**
Logistic regression modeling prediction flip ($\text{Flip} \in \{0, 1\}$) and linear regression modeling margin $\Delta z$:
$$\text{Logit}(\text{Flip}) = \beta_0 + \beta_1 \cdot \text{IsExtreme} + \beta_2 \cdot \|x\| + \beta_3 \cdot \|\delta\| + \beta_4 \cdot \Delta z_{\text{clean}} + \gamma_{\text{layer,site}}$$
We evaluate the odds ratio $\exp(\beta_1)$ and standardized effect size (Cohen's $d$).

---

## 6. Causal Intervention Design

For candidate residual events, we decompose the update vector $\delta$ into parallel and orthogonal components relative to $x$:
$$\delta_\parallel = \frac{\langle \delta, x \rangle}{\|x\|_2^2 + \epsilon} x, \quad \delta_\perp = \delta - \delta_\parallel$$

When $\delta_\parallel$ opposes $x$ ($\langle \delta, x \rangle < 0$), we selectively weaken the opposing component:
$$\delta' = \alpha \delta_\parallel + \delta_\perp$$

### 6.1 Pre-Registered Intervention Grid
- Weakening scale: $\alpha \in \{1.0, 0.75, 0.50, 0.25\}$
  ($\alpha = 1.0$ is the unperturbed original model).

### 6.2 Pre-Registered Control Baselines (Matched Perturbation Magnitude)
At each intervention site and $\alpha$, the total perturbation magnitude introduced is $\|\delta' - \delta\|_2 = (1 - \alpha)\|\delta_\parallel\|_2$. We evaluate against three matched controls:
1. **Random-Direction Perturbation:**  
   $\delta_{\text{rand}} = \delta + \eta$, where $\eta \sim \mathcal{N}(0, I)$ scaled such that $\|\eta\|_2 = (1 - \alpha)\|\delta_\parallel\|_2$.
2. **Low-Cancellation Site Control:**  
   Apply the identical $\alpha$-weakening to a matched layer/site with low cancellation ($\cos(x, \delta) > 0$).
3. **Uniform Residual Scaling Control:**  
   $\delta_{\text{uniform}} = (1 - \beta)\delta$, where $\beta = \frac{(1 - \alpha)\|\delta_\parallel\|_2}{\|\delta\|_2}$ matches the exact total norm shift.

---

## 7. Pre-Registered Decision Criteria

| Outcome | Quantitative Criteria | Scientific Conclusion & Action |
| :--- | :--- | :--- |
| **Outcome A: KILL** | Confound-controlled effect size $|d| < 0.10$ ($p > 0.05$), or targeted intervention shows no statistically significant benefit over matched random perturbations ($p > 0.05$). | **KILL PROJECT.** Cancellation is an epiphenomenon or generic noise. Stop all further exploration. |
| **Outcome B: KILL Training Hypothesis** | Targeted intervention consistently degrades clean accuracy/margin or worsens perturbation flip rates relative to $\alpha = 1.0$. | **KILL TRAINING HYPOTHESIS.** Residual cancellation is productive, required computation. Do not design training mitigations. |
| **Outcome C: INTERESTING** | Targeted intervention on extreme events improves margin or reduces flip rate significantly more than matched controls ($p < 0.01$, effect size $d > 0.2$), without collapsing clean accuracy. | **PROCEED TO V1.** Evidence supports a distinction between productive and harmful cancellation regimes. Justifies exploring training-time regularizers. |

---

## 8. Anti-Cherry-Picking Declaration

All parameters, metrics, thresholds, and controls in this document are frozen prior to inspecting final experimental tables. Any post-hoc analyses must be explicitly designated as exploratory / secondary.
