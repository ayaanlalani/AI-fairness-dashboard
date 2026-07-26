# Benchmark Comparison: Lending Club P2P Loans

Deterministic pipeline vs. OpenAI qualitative benchmark over fixed self-refinement cycles.

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

## Cycle Improvement

| Cycle | Total Score | Summary |
|---|---:|---|
| 1 | 65.0 | Total score: 65.0/100. Completeness=40.0, Severity agreement=0.0, Cause alignment=0.0, Mitigation specificity=10.0, Research grounding=15.0. |
| 2 | 70.0 | Total score: 70.0/100. Completeness=40.0, Severity agreement=0.0, Cause alignment=0.0, Mitigation specificity=15.0, Research grounding=15.0. |
| 3 | 65.0 | Total score: 65.0/100. Completeness=40.0, Severity agreement=0.0, Cause alignment=0.0, Mitigation specificity=10.0, Research grounding=15.0. |

## Severity Comparison

| Attribute | Deterministic Baseline | OpenAI | Agreement? |
|---|---|---|---|
| gender | LOW (fair) | none | No |
| income_level | LOW (fair) | none | No |
| loan_amount_level | LOW (fair) | moderate | No |

## Qualitative Narrative Comparison

### gender

**Deterministic root causes:** No significant disparate impact detected. The model treats groups approximately equally on the measured metrics.

**OpenAI:** Credit models must differentiate between good and bad risks. When the operating threshold is artificially lenient, fairness metrics lose diagnostic power (Bartlett et al., 2019). Historic evidence shows that tightening cut-offs disproportionately harms women (Pope & Sydnor, 2011). Therefore the apparent neutrality is likely an artefact of the lax threshold, not an assurance of gender fairness.

**Deterministic mitigations:** Continue monitoring: fairness can drift as data distributions change. Re-run this analysis periodically and after any model retraining.

**OpenAI:** 1. Re-balance the label distribution by augmenting legitimately defaulting cases (e.g., SMOTE-NC for categorical/continuous mix) so that the classifier must learn a decision boundary.
2. Perform a threshold sweep: compute DI, ΔDP, and ΔEO at multiple prospective approval rates (e.g., 90 %, 75 %, 50 %). Use the resulting fairness–utility frontier to select a threshold.
3. If disparities emerge post-sweep, apply pre-processing re-weighing or post-processing equalised-odds (Fairlearn’s GridSearch) targeted to the gender attribute.
4. Document threshold-selection rationale and mitigation efficacy for compliance records.

### income_level

**Deterministic root causes:** Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: annual_inc (r=0.916), loan_amnt (r=0.458), installment (r=0.437), revol_bal (r=0.309).

**OpenAI:** Income correlates with repayment capacity, so any future credit-tightening interacts with income to create potential disparate impact. A point-in-time audit without stress testing therefore underestimates compliance risk (Bartlett et al., 2019; Butler et al., 2017).

**Deterministic mitigations:** Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.

**OpenAI:** 1. Conduct stress tests by simulating progressively lower approval rates and plotting DI and ΔEO across income tiers.
2. Use Theil-index decomposition to quantify within- vs. between-income inequality and identify if disparities are driven by systematic under-prediction for low-income applicants.
3. If stress tests reveal high disparity, train with exponentiated-gradient reduction (Fairlearn) using income as a protected feature to jointly optimise accuracy and fairness.
4. Establish ongoing monitoring triggers (e.g., DI < 0.95 for income) so production drift cannot silently introduce bias.

### loan_amount_level

**Deterministic root causes:** Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: loan_amnt (r=0.922), installment (r=0.876), annual_inc (r=0.424), term_60_months (r=0.4), revol_bal (r=0.321).

**OpenAI:** Models trained on majority groups generalise poorly to minority segments; risk under-estimation for large-loan requests may create hidden bias in real-world deployment. Duarte et al. (2012) show that loan characteristics can interact with borrower attributes to exacerbate such hidden bias.

**Deterministic mitigations:** Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.

**OpenAI:** 1. Data collection: solicit or purchase additional historical data for large-loan applicants until each tier’s sample count is at least 50 % of the majority tier (target ratio ≤ 2 ×).
2. Interim mitigation: apply stratified bootstrap oversampling or importance weighting (Kamiran & Calders, 2011) so that the effective training weight of each loan-amount tier is equal.
3. Post-mitigation validation: recompute per-tier TPR/FPR and fairness metrics to confirm variance reduction; include confidence intervals to show statistical sufficiency.
4. Maintain separate monitoring dashboards for each loan-amount tier once deployed.
