# LLM Fairness Analysis (OpenAI): Lending Club P2P Loans

**Model:** o3
**Method:** Deterministic metrics + qualitative reasoning/refinement cycles

## Napkin Math

| Metric | Value |
|---|---|
| Model | `o3` |
| Wall-clock time | 11.81s |
| API attempts | 1 |
| Input tokens | 3,478 |
| Output tokens | 1,906 |
| Total tokens | 5,384 |
| Input cost | $0.0348 (@ $10.0/1M tokens) |
| Output cost | $0.0762 (@ $40.0/1M tokens) |
| **Total est. cost** | **$0.1110** |

## Cost Guardrail

| Metric | Value |
|---|---|
| Projected input tokens | 1,796 |
| Max output tokens | 5,000 |
| Projected total cost ceiling | $0.2180 |
| Configured max cost | $15.00 USD (approx. $20.27 CAD) |

## Deterministic Metric Context

| Attribute | Privileged | DI | DPD | EOD | AOD | Theil |
|---|---|---:|---:|---:|---:|---:|
| gender | male | 0.9931 | -0.0069 | 0.0000 | -0.0357 | -- |
| income_level | medium | 1.0072 | 0.0072 | 0.0000 | 0.0323 | -- |
| loan_amount_level | medium | 1.0063 | 0.0062 | 0.0000 | 0.0270 | -- |

## Cycle Scores

| Cycle | Total Score | Completeness | Severity | Cause Alignment | Mitigation | Research |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 65.0 | 40.0 | 0.0 | 0.0 | 10.0 | 15.0 |
| 2 | 70.0 | 40.0 | 0.0 | 0.0 | 15.0 | 15.0 |
| 3 | 65.0 | 40.0 | 0.0 | 0.0 | 10.0 | 15.0 |

## Final Reference Audit Specification

**Name:** Remediation-Ready LendingClub Audit v3

**Core metrics:** Disparate Impact, Demographic Parity Difference, Equal Opportunity Difference, Average Odds Difference, Theil Index, Sample Size Imbalance Ratio

**Required response elements:** per-group breakdown, severity classification, root-cause analysis, mitigation recommendations, research-backed justification

Fin-tech lenders must show both statistical neutrality and a concrete remediation plan under ECOA/Reg B. A spec that pairs classical fairness metrics with representation diagnostics (Sample-Size Imbalance Ratio) creates an audit trail that regulators, risk officers, and data scientists can simultaneously validate.

**Supporting research:**
- [3] Bartlett et al., 2019 – documents how seemingly neutral credit models hide disparate impact once thresholds tighten.
- [5] Feldman et al., 2014 – formalises disparate-impact certification and algorithmic repair frameworks.
- [6] Kamiran & Calders, 2011 – shows pre-processing (re-weighing, sampling) methods to reduce both class imbalance and inter-group unfairness.
- [2] Pope & Sydnor, 2011 – evidence of gender-linked approval gaps when credit standards increase.
- [1] Duarte et al., 2012 – highlights interaction between loan characteristics and appearance that can mask amount-related bias.

---
## gender  (severity: none)

### What is wrong

Current disparate-impact (0.99) and other parity metrics exceed the 0.95 safe threshold, so no immediate statistical violation. However, a confusion matrix with TPR = 1.0 and FPR ≈ 0.93–1.0 shows the model is granting loans to essentially every applicant—a near-trivial classifier. This masks any gender-specific denial risk that will appear once the lender adopts a realistic approval cut-off.

### Why it is wrong

Credit models must differentiate between good and bad risks. When the operating threshold is artificially lenient, fairness metrics lose diagnostic power (Bartlett et al., 2019). Historic evidence shows that tightening cut-offs disproportionately harms women (Pope & Sydnor, 2011). Therefore the apparent neutrality is likely an artefact of the lax threshold, not an assurance of gender fairness.

### How to fix it

1. Re-balance the label distribution by augmenting legitimately defaulting cases (e.g., SMOTE-NC for categorical/continuous mix) so that the classifier must learn a decision boundary.
2. Perform a threshold sweep: compute DI, ΔDP, and ΔEO at multiple prospective approval rates (e.g., 90 %, 75 %, 50 %). Use the resulting fairness–utility frontier to select a threshold.
3. If disparities emerge post-sweep, apply pre-processing re-weighing or post-processing equalised-odds (Fairlearn’s GridSearch) targeted to the gender attribute.
4. Document threshold-selection rationale and mitigation efficacy for compliance records.

### Supporting research

- [3] Bartlett et al., 2019 – threshold choice as a latent source of disparate impact.
- [2] Pope & Sydnor, 2011 – empirical proof that women face higher rejection odds when standards rise.
- [6] Kamiran & Calders, 2011 – re-weighing to correct class and group imbalance.

---
## income_level  (severity: none)

### What is wrong

Measured DI = 1.01 (> 0.95) at the single operating point, but the same trivial-classifier pattern (TPR/FPR ≈ 1) again nullifies the diagnostic value. Historical studies show that low-income borrowers are disproportionately screened out when capital supply tightens.

### Why it is wrong

Income correlates with repayment capacity, so any future credit-tightening interacts with income to create potential disparate impact. A point-in-time audit without stress testing therefore underestimates compliance risk (Bartlett et al., 2019; Butler et al., 2017).

### How to fix it

1. Conduct stress tests by simulating progressively lower approval rates and plotting DI and ΔEO across income tiers.
2. Use Theil-index decomposition to quantify within- vs. between-income inequality and identify if disparities are driven by systematic under-prediction for low-income applicants.
3. If stress tests reveal high disparity, train with exponentiated-gradient reduction (Fairlearn) using income as a protected feature to jointly optimise accuracy and fairness.
4. Establish ongoing monitoring triggers (e.g., DI < 0.95 for income) so production drift cannot silently introduce bias.

### Supporting research

- [4] Butler et al., 2017 – capital-market contractions disproportionately affect low-income consumers.
- [5] Feldman et al., 2014 – threshold-independent fairness diagnostics framework.

---
## loan_amount_level  (severity: moderate)

### What is wrong

Statistical parity appears fine (DI ≈ 1.006); however, representation is imbalanced: the ‘large’-loan segment (n = 99) is only 31 % of the ‘medium’ segment (n = 320), yielding a Sample-Size Imbalance Ratio of 3.2 ×, which exceeds the 3 × ‘moderate’ threshold. Under-represented large-loan applicants therefore contribute little to parameter estimation, inflating variance and hiding any amount-based disparate impact.

### Why it is wrong

Models trained on majority groups generalise poorly to minority segments; risk under-estimation for large-loan requests may create hidden bias in real-world deployment. Duarte et al. (2012) show that loan characteristics can interact with borrower attributes to exacerbate such hidden bias.

### How to fix it

1. Data collection: solicit or purchase additional historical data for large-loan applicants until each tier’s sample count is at least 50 % of the majority tier (target ratio ≤ 2 ×).
2. Interim mitigation: apply stratified bootstrap oversampling or importance weighting (Kamiran & Calders, 2011) so that the effective training weight of each loan-amount tier is equal.
3. Post-mitigation validation: recompute per-tier TPR/FPR and fairness metrics to confirm variance reduction; include confidence intervals to show statistical sufficiency.
4. Maintain separate monitoring dashboards for each loan-amount tier once deployed.

### Supporting research

- [1] Duarte et al., 2012 – interaction of loan characteristics and borrower traits can conceal bias.
- [6] Kamiran & Calders, 2011 – oversampling and re-weighing correct minority under-representation.

## Cross-attribute summary

Metric-wise, no protected attribute currently violates the disparate-impact threshold, but this stems from a near-trivial classifier that approves almost every application. Once realistic credit thresholds are enforced, literature suggests latent gender and income disparities are likely. Additionally, a moderate representation risk exists for the ‘large’ loan tier due to a 3.2 × sample-size imbalance. Priority remediation steps: (i) rebuild a discriminative model with balanced default examples; (ii) stress-test fairness across multiple operating points for gender and income; (iii) boost or re-weight the large-loan sample before final model freeze; and (iv) document threshold-selection and monitoring triggers to satisfy ECOA/Reg B compliance audits.
