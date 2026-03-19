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
