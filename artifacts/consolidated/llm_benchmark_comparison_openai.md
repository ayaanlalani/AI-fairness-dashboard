# Cross-Dataset Openai LLM Benchmark Comparison

# Benchmark Comparison: German Credit

Deterministic pipeline vs. OpenAI qualitative benchmark over fixed self-refinement cycles.

## Napkin Math

| Metric | Value |
|---|---|
| Model | `o3` |
| Wall-clock time | 13.96s |
| API attempts | 1 |
| Input tokens | 3,400 |
| Output tokens | 2,195 |
| Total tokens | 5,595 |
| Input cost | $0.0340 (@ $10.0/1M tokens) |
| Output cost | $0.0878 (@ $40.0/1M tokens) |
| **Total est. cost** | **$0.1218** |

## Cost Guardrail

| Metric | Value |
|---|---|
| Projected input tokens | 1,755 |
| Max output tokens | 5,000 |
| Projected total cost ceiling | $0.2175 |
| Configured max cost | $15.00 USD (approx. $20.27 CAD) |

## Cycle Improvement

| Cycle | Total Score | Summary |
|---|---:|---|
| 1 | 85.0 | Total score: 85.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=0.0, Mitigation specificity=10.0, Research grounding=15.0. |
| 2 | 85.0 | Total score: 85.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=0.0, Mitigation specificity=10.0, Research grounding=15.0. |
| 3 | 85.0 | Total score: 85.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=0.0, Mitigation specificity=10.0, Research grounding=15.0. |

## Severity Comparison

| Attribute | Deterministic Baseline | OpenAI | Agreement? |
|---|---|---|---|
| Sex_original | MODERATE (borderline) | moderate | Yes |
| AgeGroup_original | MODERATE (borderline) | moderate | Yes |
| foreign_worker_original | MODERATE (borderline) | moderate | Yes |

## Qualitative Narrative Comparison

### Sex_original

**Deterministic root causes:** Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: Sex (r=1.0).; Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0512). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.; Unequal odds: both true-positive and false-positive rates differ across groups (AOD = -0.1256), indicating the model's errors are systematically distributed along group lines.

**OpenAI:** The model ingests variables such as “personal_status”, “employment_since”, and “other_debtors”, each historically gender-skewed in Germany. These act as causal mediators of structural wage gaps rather than of a woman’s actual repayment ability. Research shows gendered visual and trust signals bias credit allocation even in tech-mediated settings ([1], [2], [3]). Thus the disparity is primarily proxy-driven, not risk-driven.

**Deterministic mitigations:** Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.; In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.; Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

**OpenAI:** 1. Immediate (no code change): Use a Calibrated-Equalized-Odds post-processor to cap TPR/FPR gaps ≤ 3 pp for sex. 2. Data-level (sprint): Apply Kamiran-Calders Re-weighing on the training set; empirically lifts DI by ≈0.1 with <1 pp AUC loss ([6]). 3. Feature surgery: Remove / bucket “personal_status” and “other_debtors” after verifying they add >0.1 mutual information with sex but <0.01 with the label—Feldman’s disparate-impact removal recipe ([5]). 4. Model-level: Retrain with Fairlearn’s Exponentiated-Gradient enforcing ε-EOD ≤ 0.03. 5. Governance: Add a “gender proxy detector” test to the feature-store CI so new features with MI > 0.15 against sex are quarantined.

### AgeGroup_original

**Deterministic root causes:** Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.; Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: AgeGroup (r=1.0), Age (r=-0.841).; Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0513). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.; Unequal odds: both true-positive and false-positive rates differ across groups (AOD = -0.1649), indicating the model's errors are systematically distributed along group lines.

**OpenAI:** Features like “credit_history_length”, “savings_account_since”, and “employment_since” correlate linearly with age (Pearson r ≈ 0.6 in this data). These lifecycle proxies pass age information into the model even if age itself were excluded. ECOA treats such indirect age discrimination as unlawful. Feldman et al. show that simply dropping the sensitive feature does not break the causal path when proxies remain.

**Deterministic mitigations:** Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.; Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.; In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.; Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

**OpenAI:** 1. Proxy quantification: Compute mutual information between each feature and age; mark any MI > 0.15. 2. Pre-processing: Apply Feldman’s disparate-impact removal (optimised orthogonalisation) to high-MI features; target DI↑ to ≥ 0.9. 3. In-training: Add Prejudice-Remover regulariser with λ = 0.4—documented to raise DI by ≥ 0.1 in credit tasks ([5]). 4. Monitoring: Deploy a guard-rail alert if DI < 0.80 (breaching ‘high’ threshold). 5. Product change: Collect cash-flow or utility-payment data that reflect near-term liquidity rather than life-cycle tenure, lowering the causal linkage to age.

### foreign_worker_original

**Deterministic root causes:** Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.; Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: foreign_worker (r=1.0).; Representation bias: severely underrepresented groups make the model less reliable for those populations and can amplify existing disparities.; Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0889). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.; Unequal odds: both true-positive and false-positive rates differ across groups (AOD = +0.2013), indicating the model's errors are systematically distributed along group lines.

**OpenAI:** Historical data collection focused on resident workers, starving the model of counter-examples for non-resident (‘0’) applicants. Extreme class imbalance (32×) inflates variance of all fairness metrics ([4]). The model therefore learns near-deterministic rules from noisy signals for the tiny group, increasing the chance of ungrounded decisions.

**Deterministic mitigations:** Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.; Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.; Data collection: gather more samples from underrepresented groups. In the interim, use class-weighted training (class_weight='balanced' in scikit-learn) to upweight minority-group errors.; In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.; Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

**OpenAI:** 1. Data acquisition: Prioritise collection of ≥ 30 additional ‘0’ records via data-share consortia or targeted outreach. 2. Until sufficient data: Fit a Bayesian hierarchical logistic model with group-level shrinkage; this guards against over-fitting to tiny samples. 3. Training phase: Use SMOTE-NC on categorical-numeric mix to synthetically up-sample the minority; Kamiran & Calders show this stabilises DI estimates. 4. Validation: Report bootstrapped 95 % CIs for all fairness metrics and flag any interval crossing high-severity thresholds. 5. Operational safeguard: Route decisions concerning group ‘0’ to manual review until N ≥ 30.


---

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


---

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


---
