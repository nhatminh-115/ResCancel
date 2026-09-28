# ResCancel V0.1: Independent Implementation & Protocol Audit

**Project:** ResCancel — Mechanistic Falsification of Residual Cancellation in Pretrained Vision Transformers  
**Audit Target:** ResCancel V0 implementation against frozen protocol ([`docs/V0_PROTOCOL.md`](V0_PROTOCOL.md))  
**Protocol Freeze Commit:** `b1c3d82`  
**Initial Result Commit:** `8cd613e`  
**Audit Date:** September 2026  
**Auditor Status:** Independent Mechanistic Audit  

---

## Executive Summary of Audit Findings

An independent verification audit of the ResCancel V0 codebase identified 8 concrete implementation and statistical discrepancies between the pre-registered protocol in [`docs/V0_PROTOCOL.md`](V0_PROTOCOL.md) and the executed code in `8cd613e`. 

All 8 discrepancies have been cataloged, repaired, and programmatically validated in V0.1 without modifying the frozen core hypotheses, thresholds, or models.

| Issue | Protocol Component | Nature of Discrepancy | Scientific Severity | Estimand Changed? | Impact on Final Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Causal Intervention | `only_extreme=False` used in sweep; intervened on all anti-aligned updates rather than pre-registered extreme events | **High** | Yes (population widened to all opposing updates) | Re-evaluating on strict extreme events confirms whether suppression is benign or harmful |
| **B** | Confound Matching | Combined standardized distance used instead of strict simultaneous individual calipers | **High** | Yes (permitted caliper violations) | Strict matching reduces pair count but achieves 0 protocol violations |
| **C** | Token Scope | Aggregate patch rows assigned `is_extreme = (frac > 0.05)`, conflating patch burden with CLS events | **Medium** | Yes (mixed token definitions) | Primary analysis restricted to CLS tokens; patch burden separated as secondary |
| **D** | Regression Independence | 48,000 residual events treated as independent in regression, causing pseudo-replication | **High** | Yes (standard errors severely underestimated) | Repaired to image-level analysis ($N=1,000$ independent images) |
| **E** | Logistic Regression | Binary `is_extreme` standardized with continuous covariates; lack of exact p-values/CIs | **Medium** | Yes (odds ratios uninterpretable as 0→1 odds) | Repaired to unstandardized indicator via `statsmodels` with analytical SE, p-val, and 95% CI |
| **F** | Matched Fragility Test | Paired t-test run on binary flip differences | **Medium** | No (hypothesis test), but wrong distributional assumption | Repaired to McNemar's test and paired risk difference with asymptotic/Wilson CI |
| **G** | Causal Significance | Decision made on raw percentages without paired significance testing or multi-seed random controls | **Medium** | No | Added bootstrap 95% CIs, McNemar tests, and 3 fixed seeds (`2501, 2502, 2503`) |
| **H** | Decision Rule | Decision logic fell through to Outcome B even when suppression did not clearly harm prediction | **Medium** | No (decision rule clarification) | Restructured decision tree to require verified negative causal delta for Outcome B vs Outcome A |

---

## Detailed Audit & Repair Inventory

### Issue A — Causal Intervention Targeted the Wrong Population
- **Discovered Mismatch:** In `rescancel/intervention.py` (line 104), `only_extreme=False` was passed to `InterventionConfig`. Consequently, the causal intervention was applied to *every* CLS token where $\langle x, \delta \rangle < 0$, regardless of whether it met the pre-registered criteria ($r \ge 0.60, q \le 0.85, C_{L1} \ge 2.00$).
- **Scientific Impact:** The experiment evaluated general anti-aligned updates rather than the specific pre-registered "extreme residual cancellation" regime.
- **Code Repair:**
  1. Defaulted `only_extreme=True` in `rescancel/instrumentation.py`.
  2. In `rescancel/intervention.py`, set `only_extreme=True` across all runs (`weaken_opposing`, `random_direction`, `uniform_scaling`).
  3. Added multi-seed evaluation (`2501, 2502, 2503`) for random-direction controls.
  4. Added assertion verifying that random and uniform controls act on the exact same token mask as `weaken_opposing`.

### Issue B — Matching Implementation Violated Individual Calipers
- **Discovered Mismatch:** In `rescancel/confound_matching.py`, candidate controls were selected if a combined Euclidean distance $\sum ((\text{ctrl} - \text{ext})/\sigma)^2 \le \text{caliper\_std} \times K$. This allowed individual continuous variables (e.g. $\|x\|_2$ or margin) to deviate by more than $\pm 15\%$ or $\pm 0.50$ logits if compensated by other variables.
- **Scientific Impact:** Violated pre-registered protocol Section 5.
- **Code Repair:**
  Rewrote `match_controls` to enforce strict simultaneous satisfaction:
  $$\frac{|x_{\text{ctrl}} - x_{\text{ext}}|}{x_{\text{ext}}} \le 0.15 \quad \text{AND} \quad \frac{\|\delta_{\text{ctrl}} - \delta_{\text{ext}}\|}{\delta_{\text{ext}}} \le 0.15 \quad \text{AND} \quad |\Delta z_{\text{ctrl}} - \Delta z_{\text{ext}}| \le 0.50$$
  plus exact equality on layer, site, and token type. Programmatically asserted zero violations.

### Issue C — Conflation of Patch Burden with CLS Events
- **Discovered Mismatch:** In `rescancel/pipeline.py`, aggregate patch summary rows were assigned `is_extreme = (patch_extreme_frac > 0.05)`, while CLS rows used the four frozen criteria. Both were placed in the same `is_extreme` column.
- **Scientific Impact:** Created two conflicting definitions of "extreme event" in the same table.
- **Code Repair:**
  Primary analysis restricted strictly to CLS token events. Patch summaries renamed to `patch_extreme_burden` and separated into `patch_summary.csv` without `is_extreme` tagging.

### Issue D & E — Pseudo-Replication & Logistic Regression Standardization
- **Discovered Mismatch:** In `rescancel/confound_matching.py`, 48,000 event rows were passed into `LogisticRegression` to predict image-level flips, treating repeated observations from the same image as independent. Furthermore, `is_extreme` was standardized via `StandardScaler`, making $\exp(\beta)$ uninterpretable as a binary odds ratio.
- **Scientific Impact:** Artificially inflated degrees of freedom and produced invalid standard errors.
- **Code Repair:**
  Implemented image-level aggregation ($N=1,000$ independent images) using `statsmodels.api.Logit`:
  - Binary indicator `has_l0_attn_extreme` (unstandardized).
  - Controlled continuous covariates (`margin`, $\|x\|$, $\|\delta\|$) standardized.
  - Computed exact standard errors, z-statistics, p-values, 95% CIs, and Odds Ratios with 95% CIs.

### Issue F — Statistical Testing for Matched Pairs
- **Discovered Mismatch:** Paired t-tests were performed on binary prediction flip indicators.
- **Scientific Impact:** Binary paired differences take discrete values $\{-1, 0, +1\}$ and violate Gaussian assumptions.
- **Code Repair:**
  Implemented McNemar's test for paired binary outcomes, computing discordant pair counts ($n_{10}, n_{01}$), paired risk difference, asymptotic/Wilson 95% CIs, and paired odds ratios. For continuous margins, added 10,000-resample bootstrap 95% CIs.

### Issue G & H — Causal Testing Significance & Decision Tree
- **Discovered Mismatch:** Causal outcomes were decided from raw percentage deltas without confidence intervals, and the decision rule had fall-through paths labeling outcomes as Outcome B without verifying negative margin CIs.
- **Code Repair:**
  Rewrote causal analysis to compute paired image-level differences, 10,000-resample bootstrap CIs, and McNemar tests for both clean accuracy and flip rate. Rewrote decision tree to strictly require statistically verified harm for Outcome B, distinguishing null effects (Outcome A) from productive computation (Outcome B).

---

## Impact on Scientific Conclusions

The repairs in V0.1 test the pre-registered hypothesis with complete mathematical and protocol fidelity.
The corrected results are documented in [`docs/V0_1_REPORT.md`](V0_1_REPORT.md).
