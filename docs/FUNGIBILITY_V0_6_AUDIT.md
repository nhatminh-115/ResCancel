# Patch Content Fungibility V0.6: Recovery Fraction Calculation Audit

**Author:** DeepMind Antigravity Team  
**Date:** September 28, 2026  
**Repository:** [https://github.com/nhatminh-115/ResCancel](https://github.com/nhatminh-115/ResCancel)  
**Status:** Audit Completed & Discrepancy Resolved  

---

## 1. Executive Summary & Audit Question

In the Patch Content Fungibility V0.6 Scientific Report ([`docs/FUNGIBILITY_V0_6_REPORT.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_V0_6_REPORT.md)), the pre-registered definition of Gaussian Recovery Fraction was:
$$\text{Recovery}(l,f) = \frac{\text{Damage}_{\text{zero}}(l,f) - \text{Damage}_{\text{gaussian}}(l,f)}{\text{Damage}_{\text{zero}}(l,f)}$$

However, an audit of Table 2.1 revealed several entries that appear mathematically inconsistent with this formula:
- **DeiT-Tiny ($l=6, f=10\%$):** Zero damage $= 0.062$, Gaussian damage $= 0.036$. The raw arithmetic gives:
  $$\frac{0.062 - 0.036}{0.062} \approx 41.9\%$$
  yet Table 2.1 reported **`0.0%`**.
- **DeiT-Small ($l=6, f=10\%$):** Zero damage $= 0.059$, Gaussian damage $= 0.038$. The raw arithmetic gives:
  $$\frac{0.059 - 0.038}{0.059} \approx 35.6\%$$
  yet Table 2.1 reported **`0.0%`**.
- **DeiT-Tiny ($l=10$, all budgets) and DeiT-Small ($l=10$, all budgets):** All reported **`0.0%`**.

The objective of this audit was to determine whether:
1. A hidden minimum-zero-damage threshold was intentionally applied, OR
2. This is a reporting / code bug.

---

## 2. Root Cause Analysis

### 2.1 Code & Protocol Investigation
Inspection of the implementation in [`patch_fungibility/v0_6_pipeline.py`](file:///d:/Study/ResCancel/patch_fungibility/v0_6_pipeline.py#L330-L335) shows the exact logic:
```python
# Gaussian Recovery Fraction
zero_dmg = cond_stats["zero"]["mean_damage"]
gauss_dmg = gauss_agg["mean_damage"]
if zero_dmg >= 0.10:
    recovery_frac = float((zero_dmg - gauss_dmg) / zero_dmg)
else:
    recovery_frac = 0.0
```

Cross-referencing with the pre-registered protocol in [`docs/FUNGIBILITY_V0_6_PROTOCOL.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_V0_6_PROTOCOL.md#L100-L102) reveals:
```markdown
- **Gaussian Recovery Fraction:**
  $$\text{Recovery}(l,f) = \frac{\text{Damage}_{\text{zero}}(l,f) - \text{Damage}_{\text{gaussian}}(l,f)}{\text{Damage}_{\text{zero}}(l,f)}$$
  Computed whenever $\text{Damage}_{\text{zero}}(l,f) \ge 0.10$.
```

### 2.2 Finding: An Intentionally Pre-Registered Threshold Combined with a Reporting Representation Bug

The audit reveals a compound situation:
1. **The Threshold Was Intentional and Pre-Registered:**
   The constraint $\text{Damage}_{\text{zero}}(l,f) \ge 0.10$ was explicitly pre-registered in Section 4 of `docs/FUNGIBILITY_V0_6_PROTOCOL.md`.
   - *Scientific Justification:* When zero damage is below $0.10$ logit margin units (or negative, as occurs at Block 10 for DeiT-Small: $-0.039$), the denominator is governed by measurement noise rather than a meaningful causal perturbation. Dividing by a tiny or negative denominator causes extreme numerical volatility (e.g., $1000\%$ recovery or negative recovery when both damages are effectively zero). Thus, bounding the ratio to regimes with meaningful baseline damage ($\ge 0.10$) is mathematically necessary and scientifically sound.
2. **The Reporting Bug:**
   In `patch_fungibility/v0_6_pipeline.py` line 335, the code executed `recovery_frac = 0.0` when `zero_dmg < 0.10` rather than assigning `None` or `np.nan`.
   When exported to markdown tables and CSV summaries, this numerical `0.0` was formatted as `0.0%`.
   This was a **reporting representation bug**: it falsely implied **$0\%$ recovery** (i.e. that Gaussian replacement failed completely and suffered identical damage to Zero ablation), whereas the true status was that the metric was **uncomputed / suppressed due to sub-threshold denominator**.

---

## 3. Audit of Sub-Threshold Conditions

The table below audits all conditions where $\text{Damage}_{\text{zero}} < 0.10$:

| Model | Depth $l$ | Budget $f$ | Zero Dmg | Gauss Dmg | Unthresholded Raw Recovery | Reported V0.6 Value | Correct Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DeiT-Tiny** | 6 | 10% (20) | 0.0618 | 0.0357 | **+42.27%** ($\approx 41.9\%$) | `0.0%` | Sub-threshold ($0.062 < 0.10$). Positive partial recovery. |
| | 10 | 10% (20) | 0.0190 | 0.0070 | **+63.24%** | `0.0%` | Sub-threshold ($0.019 < 0.10$). Negligible zero damage. |
| | 10 | 25% (49) | 0.0374 | 0.0140 | **+62.56%** | `0.0%` | Sub-threshold ($0.037 < 0.10$). Negligible zero damage. |
| | 10 | 50% (98) | 0.0455 | 0.0247 | **+45.62%** | `0.0%` | Sub-threshold ($0.045 < 0.10$). Negligible zero damage. |
| **DeiT-Small**| 6 | 10% (20) | 0.0587 | 0.0380 | **+35.16%** ($\approx 35.6\%$) | `0.0%` | Sub-threshold ($0.059 < 0.10$). Positive partial recovery. |
| | 10 | 10% (20) | -0.0390 | -0.0082 | *Ill-conditioned* | `0.0%` | Denominator negative (zero ablation slightly improved margin). |
| | 10 | 25% (49) | -0.0349 | -0.0119 | *Ill-conditioned* | `0.0%` | Denominator negative. |
| | 10 | 50% (98) | -0.0312 | -0.0146 | *Ill-conditioned* | `0.0%` | Denominator negative. |

---

## 4. Remediation Actions

1. **No Rerunning of Raw Experiments:**
   All underlying per-image margin outputs, accuracies, paired differences ($\Delta m$), Cohen's $d_z$, bootstrap confidence intervals, and BH-FDR $q$-values in [`outputs/fungibility_v0_6/`](file:///d:/Study/ResCancel/outputs/fungibility_v0_6/) are completely correct and untouched.
2. **Correction in Scientific Report ([`docs/FUNGIBILITY_V0_6_REPORT.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_V0_6_REPORT.md)):**
   - In Table 2.1, all 8 sub-threshold cells are updated from `0.0%` to **`N/A*`**.
   - An explanatory footnote is appended to Table 2.1:
     > `* N/A`: Condition has $\text{Damage}_{\text{zero}} < 0.10$, below the pre-registered threshold for stable recovery fraction calculation. For transparency: Tiny $l=6, 10\%$ unthresholded raw recovery is $+42.3\%$; Small $l=6, 10\%$ is $+35.2\%$; Tiny $l=10$ ranges from $+45.6\%$ to $+63.2\%$; Small $l=10$ has negative zero damage (ill-conditioned denominator).
3. **Protocol Carry-Over for V0.7:**
   In V0.7, any relative retention / recovery ratio will explicitly format sub-threshold conditions as `N/A` rather than numerical `0.0`, eliminating ambiguity between genuine zero recovery and sub-threshold denominator suppression.
