# Cross-Dataset OpenAI LLM Benchmark Comparison

# Benchmark Comparison: German Credit

Deterministic pipeline vs. OpenAI qualitative benchmark over fixed self-refinement cycles.

## Napkin Math

| Metric | Value |
|---|---|
| Model | `o3` |
| Wall-clock time | 29.73s |
| API attempts | 1 |
| Input tokens | 3,862 |
| Output tokens | 2,182 |
| Total tokens | 6,044 |
| Input cost | $0.0386 (@ $10.0/1M tokens) |
| Output cost | $0.0873 (@ $40.0/1M tokens) |
| **Total est. cost** | **$0.1259** |

## Cost Guardrail

| Metric | Value |
|---|---|
| Projected input tokens | 1,973 |
| Max output tokens | 5,000 |
| Projected total cost ceiling | $0.2197 |
| Configured max cost | $37.00 USD (approx. $50.00 CAD) |

## Cycle Improvement

| Cycle | Total Score | Summary |
|---|---:|---|
| 1 | 90.0 | Total score: 90.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=0.0, Mitigation specificity=15.0, Research grounding=15.0. |
| 2 | 85.0 | Total score: 85.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=0.0, Mitigation specificity=10.0, Research grounding=15.0. |
| 3 | 85.0 | Total score: 85.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=0.0, Mitigation specificity=10.0, Research grounding=15.0. |

## Severity Comparison

| Attribute | Deterministic Baseline | OpenAI | Agreement? |
|---|---|---|---|
| Sex_original | MODERATE (borderline) | moderate | Yes |
| AgeGroup_original | MODERATE (borderline) | moderate | Yes |
| foreign_worker_original | MODERATE (borderline) | moderate (unstable) | Yes |

## Qualitative Narrative Comparison

### Sex_original

**Deterministic root causes:** Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: Sex (r=1.0).; Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0512). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.; Unequal odds: both true-positive and false-positive rates differ across groups (AOD = +0.1256), indicating the model's errors are systematically distributed along group lines.

**OpenAI:** Causal trace:  (1) Historic labels embed gender pay gaps and shorter employment histories → label bias.  (2) Correlated features such as ‘job_status’, ‘housing’, and ‘number_of_dependents’ act as proxy variables for sex → feature bias.  (3) The unconstrained learning algorithm minimises overall error; because male records are 2.3× more numerous, gradient updates prioritise male loss → optimisation bias.  Together these channels tilt decision boundaries against women.

**Deterministic mitigations:** Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.; In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.; Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

**OpenAI:** Priority 1 – Data re-balancing: Apply reweighing to increase the weight of correctly-paid female loans until sex–label mutual information ≈ 0 (Sucharita & Shaw [1]).  
Priority 2 – Causal feature pruning: Remove or transform features whose conditional mutual information with the target falls ≤ 1 % after conditioning on sex (Osborne & Usani [3]).  
Priority 3 – Constrained learning: Fine-tune an exponentiated-gradient (EG) fair classifier with a demographic-parity constraint λ selected to keep AUC loss < 3 % while pushing DI ≥ 0.90.  
Priority 4 – Post-hoc safeguard: Deploy calibrated equal-odds post-processing if a hot-fix is needed before full retrain.  
Governance: Add a CI-pipeline test that fails if DI < 0.80 or if the 95 % lower-bound of DI < 0.75.

### AgeGroup_original

**Deterministic root causes:** Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.; Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: AgeGroup (r=1.0), Age (r=-0.841).; Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0513). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.; Unequal odds: both true-positive and false-positive rates differ across groups (AOD = +0.1649), indicating the model's errors are systematically distributed along group lines.

**OpenAI:** Dual bias mechanism:  (1) Label bias – the ‘good credit’ label rewards longer payment history, inherently easier for older borrowers.  (2) Feature leakage – raw tenure, asset total, and account age are monotonic in chronological age, functioning as causal mediators.  Model optimisation then codifies these structural advantages, inflating selection odds for the 40 + cohort.

**Deterministic mitigations:** Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.; Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.; In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.; Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

**OpenAI:** Data: Perform stratified bootstrap so that within each age bin the favourable-label share approximates the global mean; keep the effective sample size constant via importance weighting (Rane et al. [2]).  
Representation: Replace absolute tenure variables with age-normalised ratios (e.g., credit_history_length / applicant_age) to break direct age signal.  
Algorithmic: Add a PrejudiceRemover regulariser with β tuned by grid-search; Sucharita & Shaw [1] report β = 25 raised DI from 0.78 → 0. Nine-fold CV to ensure < 2 pp AUC loss.  
Causal robustness: Train under Invariant Risk Minimisation across age-stratified environments so learned relations hold when age distribution shifts (Osborne & Usani [3]).  
Monitoring: Track ‘age-specific AUC’ and stop deployment if the metric for under-40 drops > 5 pp below 40 +.

### foreign_worker_original

**Deterministic root causes:** Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.; Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: foreign_worker (r=1.0).; Representation bias: severely underrepresented groups make the model less reliable for those populations and can amplify existing disparities.; Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0889). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.; Unequal odds: both true-positive and false-positive rates differ across groups (AOD = -0.2013), indicating the model's errors are systematically distributed along group lines.

**OpenAI:** Sampling bias: The German Credit dataset was historically assembled to study foreign worker risk; domestic workers were inadvertently under-sampled.  As a result, the model sees virtually no domestic examples, cannot learn a calibrated threshold, and yields variance-inflated disparity estimates.  Down-stream deployments with higher domestic proportions could therefore incur unseen harm (hidden fairness debt).

**Deterministic mitigations:** Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.; Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.; Data collection: gather more samples from underrepresented groups. In the interim, use class-weighted training (class_weight='balanced' in scikit-learn) to upweight minority-group errors.; In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.; Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

**OpenAI:** Data acquisition: Collect ≥ 25 additional domestic-worker records to surpass the 30-sample stability heuristic (Rane et al. [2]).  If impossible, use SMOTE-NC with k = 3 to synthetically expand the domestic group, followed by sample-weight correction to avoid mode collapse.  
Metric estimation: Report bootstrapped 95 % CIs; flag attribute if CI lower-bound DI < 0.80.  
Robust optimisation: Train with a worst-case group DRO objective (Jagadeesan [4]) so that even tiny groups receive protective margins.  
Deployment guardrail: Introduce a ‘min-support’ trigger that halts scoring for groups with < 30 observations and routes them to manual review until data sufficiency is achieved.


---

# Benchmark Comparison: HMDA Mortgage Lending (Georgia)

Deterministic pipeline vs. OpenAI qualitative benchmark over fixed self-refinement cycles.

## Napkin Math

| Metric | Value |
|---|---|
| Model | `o3` |
| Wall-clock time | 14.22s |
| API attempts | 1 |
| Input tokens | 4,032 |
| Output tokens | 1,666 |
| Total tokens | 5,698 |
| Input cost | $0.0403 (@ $10.0/1M tokens) |
| Output cost | $0.0666 (@ $40.0/1M tokens) |
| **Total est. cost** | **$0.1070** |

## Cost Guardrail

| Metric | Value |
|---|---|
| Projected input tokens | 2,327 |
| Max output tokens | 5,000 |
| Projected total cost ceiling | $0.2233 |
| Configured max cost | $37.00 USD (approx. $50.00 CAD) |

## Cycle Improvement

| Cycle | Total Score | Summary |
|---|---:|---|
| 1 | 90.0 | Total score: 90.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=0.0, Mitigation specificity=15.0, Research grounding=15.0. |
| 2 | 85.0 | Total score: 85.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=0.0, Mitigation specificity=10.0, Research grounding=15.0. |
| 3 | 90.0 | Total score: 90.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=5.0, Mitigation specificity=10.0, Research grounding=15.0. |

## Severity Comparison

| Attribute | Deterministic Baseline | OpenAI | Agreement? |
|---|---|---|---|
| race | MODERATE (borderline) | moderate | Yes |
| sex | MODERATE (borderline) | moderate | Yes |
| age_group | LOW (fair) | low | Yes |

## Qualitative Narrative Comparison

### race

**Deterministic root causes:** Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.; Representation bias: severely underrepresented groups make the model less reliable for those populations and can amplify existing disparities.

**OpenAI:** Causal chain: (a) Historical label bias – HMDA ground-truth approvals already embed redlining (Sucharita & Shaw §4). (b) Proxy features – census-tract and credit-history-length are collinear with race (Cramér’s V>0.35) and therefore transmit historical bias into the model (Rane et al. Tab 2). (c) Variance amplification – 250× sample-size gap inflates standard errors for minority metrics, making early-stop criteria in training favour majority performance (Osborne & Usani Fig 3).

**Deterministic mitigations:** Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.; Data collection: gather more samples from underrepresented groups. In the interim, use class-weighted training (class_weight='balanced' in scikit-learn) to upweight minority-group errors.

**OpenAI:** Immediate (≤1 quarter):
• Apply calibrated Equalised-Odds post-processing targeted only at Black and Asian sub-scores; keep ΔFPR ≤ +2 pp to stay within lender-loss tolerance.

Short-term (1–2 quarters):
• Augment training data by oversampling minority files with Fair SMOTE-NC; validate that covariate shift on Dodd-Frank variables is < 5 %.
• Retrain with ExponentiatedGradient optimisation under an Equal-Opportunity constraint; tune λ until validation DI ≥ 0.95.

Long-term (>2 quarters):
• Perform causal-feature filtering: drop or orthogonalise features whose Shapley parents causally descend from race; pilot the Osborne & Usani approach that reduced structural bias by 12 % in their study.
• Deploy a race-segmented monitoring dashboard with Wilson CI; auto-freeze model if DI<0.80 or EOD>0.05.

### sex

**Deterministic root causes:** 

**OpenAI:** (a) Training labels: historic datasets show 4.7 pp lower base approval for women (HMDA base-rate difference). (b) Proxy income features: variables like “primary wage-earner” correlate with sex (Cramér’s V = 0.31) and dominate trees (average |Shap| = 0.06) (Rane §3.3). (c) Interaction blind spot: single-axis fairness hides compounded penalties for Black-female applicants, misestimating real harm.

**Deterministic mitigations:** Post-processing -- Calibrated Equalized Odds (aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing): a minimal intervention that adjusts thresholds to bring DI above the 0.8 threshold while preserving calibration.

**OpenAI:** • Run feature-attribution audit; suppress or binarise any feature with |Shap|>0.05 and V ≥ 0.30 with sex.
• Retrain using multi-attribute ExponentiatedGradient jointly constraining race and sex (Sucharita & Shaw framework) to DI ≥ 0.95 on both axes.
• Operational guard-rail: real-time alert if rolling 4-week sex-based DI<0.90.

### age_group

**Deterministic root causes:** Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.

**OpenAI:** Conservative credit policy rules (e.g., higher required cash-reserves) correlate with age-linked retirement status, reducing senior approvals despite similar risk (domain knowledge aligned with Sucharita & Shaw continuous monitoring recommendations).

**Deterministic mitigations:** Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.

**OpenAI:** • Keep age in fairness dashboard; trigger review if DI<0.95.
• Simulate recession scenarios to ensure senior FPR does not spike >2 pp.
• Document age-related policy rationale with ADEA citations in the model-risk file.


---
