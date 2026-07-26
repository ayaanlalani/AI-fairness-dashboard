# LLM Fairness Analysis (OpenAI): HMDA Mortgage Lending (Georgia)

**Model:** o3
**Method:** Deterministic metrics + qualitative reasoning/refinement cycles

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

## Deterministic Metric Context

| Attribute | Privileged | DI | DPD | EOD | AOD | Theil |
|---|---|---:|---:|---:|---:|---:|
| race | White | 0.8956 | -0.0776 | 0.0025 | 0.0117 | 0.4838 |
| sex | Male | 0.9291 | -0.0518 | -0.0076 | -0.0084 | 0.4838 |
| age_group | mid | 1.0138 | 0.0097 | 0.0008 | -0.0150 | 0.4838 |

## Cycle Scores

| Cycle | Total Score | Completeness | Severity | Cause Alignment | Mitigation | Research |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 85.0 | 40.0 | 20.0 | 0.0 | 10.0 | 15.0 |
| 2 | 90.0 | 40.0 | 20.0 | 5.0 | 10.0 | 15.0 |
| 3 | 95.0 | 40.0 | 20.0 | 10.0 | 10.0 | 15.0 |

## Final Reference Audit Specification

**Name:** HMDA-GA Fair-Lending Remediation Audit v3.0

**Core metrics:** Disparate Impact, Demographic Parity Difference, Equal Opportunity Difference, Average Odds Difference, Theil Index

**Required response elements:** per-group breakdown, severity classification, root-cause analysis, mitigation recommendations, research-backed justification

Georgia mortgage decisions must satisfy ECOA/FHA disparate-impact doctrine. A multi-metric audit plus qualitative diagnosis provides a remediation-ready hand-off for model, compliance, and business teams.

**Supporting research:**
- Mehrabi et al. 2019 – complementary metrics catch different harms [4]
- Feldman et al. 2014 – certify/remove disparate impact with legal-style thresholds [5]
- Kamiran & Calders 2011 – pre-processing to repair biased training labels [6]
- Bartlett et al. 2019 – historical underwriting discrimination embedded in HMDA data [3]
- Duarte et al. 2012 – gender-linked informational gaps in lending [1]
- Pope & Sydnor 2011 – age proxies and indirect bias in credit markets [2]

---
## race  (severity: Moderate)

### What is wrong

Black applicants (34 % of the sample) are approved 64.4 % of the time vs 74.3 % for White applicants (DI = 0.896 < 0.95). Equal-opportunity and error-rate gaps are small, so the harm manifests as lower access to credit rather than higher mis-classification risk. Very small groups (≤30 rows) have unstable metrics but do not drive the disparity.

### Why it is wrong

Causal chain: (a) Historical label bias – past redlining and disparate underwriting are recorded as the ‘ground truth’ approvals (Bartlett et al., 2019); (b) Representation bias – Whites out-number Blacks 1.7×, so loss minimisation weighs White errors more heavily (Mehrabi et al., 2019); (c) Proxy leakage – geographic and wealth variables correlate with race and act as stand-in predictors (Feldman et al., 2014). Because the TPR/FPR parity is high, the disparity is driven mainly by biased labels and features, not model thresholds.

### How to fix it

Immediate (≤2 wks):
• Post-processing Threshold Optimiser (Equalized Odds) to lift Black selection rate until DI ≥ 0.95 while capping total FPR change at ±1 pp.
• Manual ‘second-look’ review for rejected applications from majority-Black census tracts.
Near-term (1-3 mo):
• Apply Kamiran-Calders re-weighing during retrain; target DI gap < 0.02 in cross-validation.
• Remove or regularise ZIP-code, tract-level variables with mutual information > 0.1 w.r.t race.
Long-term (≥3 mo):
• Expand dataset with additional non-White applications to reduce White:Other ratio below 5:1; bucket groups with <30 rows into “Other” until data grow.
• Adopt counterfactual fairness testing—swap race while holding financial features constant; require score change < 2 pp.

### Supporting research

- Bartlett et al. 2019 – label bias from redlining [3]
- Feldman et al. 2014 – proxy variable removal lowers DI [5]
- Kamiran & Calders 2011 – efficacy of re-weighing [6]
- Mehrabi et al. 2019 – representation bias mechanisms [4]

---
## sex  (severity: Moderate)

### What is wrong

Female applicants receive loans 67.9 % of the time vs 73.1 % for males (DI = 0.929). Error-rate differences are negligible but the benefit shortfall can still trigger ECOA disparate-impact liability.

### Why it is wrong

Causal chain: (a) Data omission – HMDA stores only primary-applicant sex; household income contributed by a (often female) secondary earner is omitted, understating creditworthiness (Duarte et al., 2012). (b) Proxy leakage – employment-tenure and industry variables reflect gender-segregated labour markets. (c) Objective-first optimisation – threshold tuned for accuracy not parity allows residual disparity (Mehrabi et al., 2019).

### How to fix it

Immediate (≤2 wks):
• Deploy Exponentiated-Gradient post-processor constrained to DI ≥ 0.95 for sex; validate ≤0.5 pp accuracy loss.
Near-term (1-3 mo):
• Collect and incorporate co-applicant income; retrain with combined household income.
• Audit top-10 SHAP features; bin or drop any with sex mutual information > 0.05.
• Introduce fairness-aware hyper-parameter search optimising Accuracy – λ·|DI-1| (λ = 0.2).
Governance:
• CI guardrail: block promotion if sex DI < 0.8 or Theil Index rises > 0.05 above baseline.

### Supporting research

- Duarte et al. 2012 – information asymmetry by gender [1]
- Mehrabi et al. 2019 – optimisation trade-offs [4]
- Feldman et al. 2014 – DI thresholds in practice [5]
- Kamiran & Calders 2011 – feature sanitisation [6]

---
## age_group  (severity: Low)

### What is wrong

Overall parity holds (DI = 1.014), yet seniors’ approval rate (66.6 %) is 11 pp below young borrowers (77.8 %). The disparity is below the moderate threshold but merits monitoring because age is federally protected.

### Why it is wrong

(a) Feature mis-specification – length-of-credit-history favours younger ‘credit-builder’ products while retired seniors often have frozen files (Pope & Sydnor 2011). (b) Income volatility – retirement income is discounted by underwriting rules, acting as a proxy for age. (c) Sampling noise – young group is only 21 % of data; differences may shrink with more observations.

### How to fix it

Preventive actions:
• Remove raw ‘length-of-credit-history’; replace with stability-normalised version (e.g., ratio of open to closed accounts).
• Use age_group only as a fairness control variable; exclude it from prediction features.
• Quarterly fairness dashboard; trigger review if DI < 0.95 or EO diff > ±0.02.
• Counterfactual test: swap age group and require score change < 2 pp.

### Supporting research

- Pope & Sydnor 2011 – age proxies in credit [2]
- Mehrabi et al. 2019 – need for continuous monitoring [4]

## Cross-attribute summary

Race and sex each show Moderate severity disparate impact driven primarily by historical label bias, representation imbalance, and proxy variables; immediate threshold or post-processing fixes are required. Age currently poses Low risk but should be watched because feature design choices could quickly push DI below the 0.95 monitoring line. Governance proposal: DI < 0.95 triggers remediation, DI < 0.8 blocks deployment, and Theil Index drift > 0.05 mandates re-audit.
