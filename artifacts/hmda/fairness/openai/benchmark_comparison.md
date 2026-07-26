# Benchmark Comparison: HMDA Mortgage Lending (Georgia)

Deterministic pipeline vs. OpenAI qualitative benchmark over fixed self-refinement cycles.

## Napkin Math

| Metric | Value |
|---|---|
| Model | `o3` |
| Wall-clock time | 10.78s |
| API attempts | 1 |
| Input tokens | 4,271 |
| Output tokens | 1,898 |
| Total tokens | 6,169 |
| Input cost | $0.0427 (@ $10.0/1M tokens) |
| Output cost | $0.0759 (@ $40.0/1M tokens) |
| **Total est. cost** | **$0.1186** |

## Cost Guardrail

| Metric | Value |
|---|---|
| Projected input tokens | 2,109 |
| Max output tokens | 5,000 |
| Projected total cost ceiling | $0.2211 |
| Configured max cost | $15.00 USD (approx. $20.27 CAD) |

## Cycle Improvement

| Cycle | Total Score | Summary |
|---|---:|---|
| 1 | 85.0 | Total score: 85.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=0.0, Mitigation specificity=10.0, Research grounding=15.0. |
| 2 | 90.0 | Total score: 90.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=5.0, Mitigation specificity=10.0, Research grounding=15.0. |
| 3 | 95.0 | Total score: 95.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=10.0, Mitigation specificity=10.0, Research grounding=15.0. |

## Severity Comparison

| Attribute | Deterministic Baseline | OpenAI | Agreement? |
|---|---|---|---|
| race | MODERATE (borderline) | Moderate | Yes |
| sex | MODERATE (borderline) | Moderate | Yes |
| age_group | LOW (fair) | Low | Yes |

## Qualitative Narrative Comparison

### race

**Deterministic root causes:** Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.; Representation bias: severely underrepresented groups make the model less reliable for those populations and can amplify existing disparities.

**OpenAI:** Causal chain: (a) Historical label bias – past redlining and disparate underwriting are recorded as the ‘ground truth’ approvals (Bartlett et al., 2019); (b) Representation bias – Whites out-number Blacks 1.7×, so loss minimisation weighs White errors more heavily (Mehrabi et al., 2019); (c) Proxy leakage – geographic and wealth variables correlate with race and act as stand-in predictors (Feldman et al., 2014). Because the TPR/FPR parity is high, the disparity is driven mainly by biased labels and features, not model thresholds.

**Deterministic mitigations:** Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.; Data collection: gather more samples from underrepresented groups. In the interim, use class-weighted training (class_weight='balanced' in scikit-learn) to upweight minority-group errors.

**OpenAI:** Immediate (≤2 wks):
• Post-processing Threshold Optimiser (Equalized Odds) to lift Black selection rate until DI ≥ 0.95 while capping total FPR change at ±1 pp.
• Manual ‘second-look’ review for rejected applications from majority-Black census tracts.
Near-term (1-3 mo):
• Apply Kamiran-Calders re-weighing during retrain; target DI gap < 0.02 in cross-validation.
• Remove or regularise ZIP-code, tract-level variables with mutual information > 0.1 w.r.t race.
Long-term (≥3 mo):
• Expand dataset with additional non-White applications to reduce White:Other ratio below 5:1; bucket groups with <30 rows into “Other” until data grow.
• Adopt counterfactual fairness testing—swap race while holding financial features constant; require score change < 2 pp.

### sex

**Deterministic root causes:** 

**OpenAI:** Causal chain: (a) Data omission – HMDA stores only primary-applicant sex; household income contributed by a (often female) secondary earner is omitted, understating creditworthiness (Duarte et al., 2012). (b) Proxy leakage – employment-tenure and industry variables reflect gender-segregated labour markets. (c) Objective-first optimisation – threshold tuned for accuracy not parity allows residual disparity (Mehrabi et al., 2019).

**Deterministic mitigations:** Post-processing -- Calibrated Equalized Odds (aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing): a minimal intervention that adjusts thresholds to bring DI above the 0.8 threshold while preserving calibration.

**OpenAI:** Immediate (≤2 wks):
• Deploy Exponentiated-Gradient post-processor constrained to DI ≥ 0.95 for sex; validate ≤0.5 pp accuracy loss.
Near-term (1-3 mo):
• Collect and incorporate co-applicant income; retrain with combined household income.
• Audit top-10 SHAP features; bin or drop any with sex mutual information > 0.05.
• Introduce fairness-aware hyper-parameter search optimising Accuracy – λ·|DI-1| (λ = 0.2).
Governance:
• CI guardrail: block promotion if sex DI < 0.8 or Theil Index rises > 0.05 above baseline.

### age_group

**Deterministic root causes:** Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.

**OpenAI:** (a) Feature mis-specification – length-of-credit-history favours younger ‘credit-builder’ products while retired seniors often have frozen files (Pope & Sydnor 2011). (b) Income volatility – retirement income is discounted by underwriting rules, acting as a proxy for age. (c) Sampling noise – young group is only 21 % of data; differences may shrink with more observations.

**Deterministic mitigations:** Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.

**OpenAI:** Preventive actions:
• Remove raw ‘length-of-credit-history’; replace with stability-normalised version (e.g., ratio of open to closed accounts).
• Use age_group only as a fairness control variable; exclude it from prediction features.
• Quarterly fairness dashboard; trigger review if DI < 0.95 or EO diff > ±0.02.
• Counterfactual test: swap age group and require score change < 2 pp.
