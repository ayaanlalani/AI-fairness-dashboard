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
