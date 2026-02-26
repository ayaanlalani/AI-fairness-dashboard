# Benchmark Comparison: German Credit

Deterministic pipeline (AIF360) vs. Gemini LLM (fairlearn only -- no access to our calculations).

## Napkin Math

| Metric | Value |
|---|---|
| Model | `gemini-2.5-flash` |
| Wall-clock time | 75.57s |
| API attempts | 1 |
| Input tokens | 3,539 |
| Output tokens | 1,904 |
| Total tokens | 5,443 |
| Input cost | $0.0004 (@ $0.1/1M tokens) |
| Output cost | $0.0008 (@ $0.4/1M tokens) |
| **Total est. cost** | **$0.0011** |

## Metric Comparison

| Attribute | Metric | Our Pipeline | Gemini LLM | Delta |
|---|---|---:|---:|---:|
| Sex_original | Disparate Impact | 0.8595 | 0.8752 | +0.0157 |
| Sex_original | Demographic Parity Diff | -0.1152 | -0.0988 | +0.0164 |
| Sex_original | Equal Opportunity Diff | -0.0512 | -0.0658 | -0.0146 |
| Sex_original | Average Odds Diff | -0.1256 | -0.0381 | +0.0875 |
| Sex_original | Theil Index | 0.3084 | 0.0543 | -0.2541 |
| AgeGroup_original | Disparate Impact | 0.8212 | 0.8848 | +0.0636 |
| AgeGroup_original | Demographic Parity Diff | -0.1586 | -0.0950 | +0.0636 |
| AgeGroup_original | Equal Opportunity Diff | -0.0513 | -0.0811 | -0.0298 |
| AgeGroup_original | Average Odds Diff | -0.1649 | -0.0245 | +0.1404 |
| AgeGroup_original | Theil Index | 0.3084 | 0.0543 | -0.2541 |
| foreign_worker_original | Disparate Impact | 0.9402 | 0.9755 | +0.0353 |
| foreign_worker_original | Demographic Parity Diff | -0.0498 | -0.0195 | +0.0303 |
| foreign_worker_original | Equal Opportunity Diff | -0.0889 | -0.0336 | +0.0553 |
| foreign_worker_original | Average Odds Diff | 0.2013 | -0.0175 | -0.2188 |
| foreign_worker_original | Theil Index | 0.3084 | 0.0543 | -0.2541 |

## Severity Comparison

| Attribute | Our Pipeline | Gemini LLM | Agreement? |
|---|---|---|---|
| Sex_original | MODERATE (borderline | MODERATE | Yes |
| AgeGroup_original | MODERATE (borderline | MODERATE | Yes |
| foreign_worker_original | MODERATE (borderline | LOW | No |

## Qualitative Narrative Comparison

### Sex_original

#### What is wrong

**Our pipeline:** The model's favorable-outcome rate varies across Sex_original groups. The highest rate is for **male** (82.0%) and the lowest is for **female** (70.5%), a gap of 11.5%.

- Demographic Parity Difference = -0.1152: the unprivileged group is selected less often.
- Equal Opportunity Difference = -0.0512: among truly deserving candidates, the model misses more unprivileged individuals.
- Average Odds Difference = -0.1256: error rates differ systematically across groups.

**Gemini:** For the 'Sex_original' attribute, the unprivileged group ('female') experiences a lower selection rate compared to the privileged group ('male'). This disparity is reflected by a Disparate Impact of 0.8752 and a Demographic Parity Difference of -0.0988. Furthermore, the model exhibits a lower True Positive Rate (TPR) for the unprivileged group (Equal Opportunity Difference: -0.0658), meaning individuals in the unprivileged group who are actually favorable are less likely to be correctly identified as such. The Average Odds Difference of -0.0381 suggests broader disparities in overall error rates (false positives and false negatives) across the groups, indicating that the model's performance is not consistently fair across all sensitive groups.

#### Why it is wrong

**Our pipeline:** **Proxy features** (features correlated with the protected attribute):
- `Sex`: r = 1.0 (strong)
- `Housing`: r = -0.267 (weak)
- `Age`: r = 0.216 (weak)
- `Job`: r = 0.165 (weak)
- `AgeGroup`: r = -0.151 (weak)

**Root causes identified:**
1. Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: Sex (r=1.0).
2. Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0512). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.
3. Unequal odds: both true-positive and false-positive rates differ across groups (AOD = -0.1256), indicating the model's errors are systematically distributed along group lines.

**Gemini:** Potential root causes for such bias often include historical biases embedded in the training data, where past decisions might have disproportionately favored the privileged group, leading the model to learn and perpetuate these patterns. Proxy features highly correlated with the protected attribute could also inadvertently lead to discriminatory outcomes. Additionally, data imbalance, where certain groups are underrepresented or exhibit different underlying distributions, can exacerbate these issues, making it harder for the model to learn fair representations.

#### How to fix it

**Our pipeline:** 1. Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.
2. In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.
3. Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

**Gemini:** To mitigate the observed bias for the 'Sex_original' attribute, various fairlearn strategies can be employed. For post-processing, `fairlearn.postprocessing.ThresholdOptimizer` can be used to adjust the decision threshold for each group, aiming to satisfy fairness constraints like Demographic Parity or Equalized Odds without retraining the original model. For in-processing mitigation, `fairlearn.reductions.ExponentiatedGradient` or `fairlearn.reductions.GridSearch` can be integrated into the model training pipeline. These methods allow a standard estimator to be optimized for both accuracy and fairness (e.g., 'demographic_parity' or 'equalized_odds') by learning a reweighing of the training samples or by generating a sequence of reweighted models. Data preprocessing techniques like `fairlearn.reductions.Reweighing` could also be applied to adjust sample weights based on group membership and label before training, to achieve specific fairness criteria.

### AgeGroup_original

#### What is wrong

**Our pipeline:** The model's favorable-outcome rate varies across AgeGroup_original groups. The highest rate is for **40_plus** (88.7%) and the lowest is for **under_40** (72.9%), a gap of 15.9%.

- Demographic Parity Difference = -0.1586: the unprivileged group is selected less often.
- Equal Opportunity Difference = -0.0513: among truly deserving candidates, the model misses more unprivileged individuals.
- Average Odds Difference = -0.1649: error rates differ systematically across groups.

**Gemini:** For the 'AgeGroup_original' attribute, the unprivileged group ('under_40') experiences a lower selection rate compared to the privileged group ('40_plus'). This disparity is reflected by a Disparate Impact of 0.8848 and a Demographic Parity Difference of -0.0950. Furthermore, the model exhibits a lower True Positive Rate (TPR) for the unprivileged group (Equal Opportunity Difference: -0.0811), meaning individuals in the unprivileged group who are actually favorable are less likely to be correctly identified as such. The Average Odds Difference of -0.0245 suggests broader disparities in overall error rates (false positives and false negatives) across the groups, indicating that the model's performance is not consistently fair across all sensitive groups.

#### Why it is wrong

**Our pipeline:** **Proxy features** (features correlated with the protected attribute):
- `AgeGroup`: r = 1.0 (strong)
- `Age`: r = -0.841 (strong)
- `Housing`: r = 0.229 (weak)
- `Sex`: r = -0.151 (weak)

**Root causes identified:**
1. Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.
2. Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: AgeGroup (r=1.0), Age (r=-0.841).
3. Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0513). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.
4. Unequal odds: both true-positive and false-positive rates differ across groups (AOD = -0.1649), indicating the model's errors are systematically distributed along group lines.

**Gemini:** Potential root causes for such bias often include historical biases embedded in the training data, where past decisions might have disproportionately favored the privileged group, leading the model to learn and perpetuate these patterns. Proxy features highly correlated with the protected attribute could also inadvertently lead to discriminatory outcomes. Additionally, data imbalance, where certain groups are underrepresented or exhibit different underlying distributions, can exacerbate these issues, making it harder for the model to learn fair representations.

#### How to fix it

**Our pipeline:** 1. Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.
2. Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.
3. In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.
4. Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

**Gemini:** To mitigate the observed bias for the 'AgeGroup_original' attribute, various fairlearn strategies can be employed. For post-processing, `fairlearn.postprocessing.ThresholdOptimizer` can be used to adjust the decision threshold for each group, aiming to satisfy fairness constraints like Demographic Parity or Equalized Odds without retraining the original model. For in-processing mitigation, `fairlearn.reductions.ExponentiatedGradient` or `fairlearn.reductions.GridSearch` can be integrated into the model training pipeline. These methods allow a standard estimator to be optimized for both accuracy and fairness (e.g., 'demographic_parity' or 'equalized_odds') by learning a reweighing of the training samples or by generating a sequence of reweighted models. Data preprocessing techniques like `fairlearn.reductions.Reweighing` could also be applied to adjust sample weights based on group membership and label before training, to achieve specific fairness criteria.

### foreign_worker_original

#### What is wrong

**Our pipeline:** The model's favorable-outcome rate varies across foreign_worker_original groups. The highest rate is for **0** (83.3%) and the lowest is for **1** (78.3%), a gap of 5.0%.

- Demographic Parity Difference = -0.0498: the unprivileged group is selected less often.
- Equal Opportunity Difference = -0.0889: among truly deserving candidates, the model misses more unprivileged individuals.
- Average Odds Difference = +0.2013: error rates differ systematically across groups.

**Gemini:** For the 'foreign_worker_original' attribute, the unprivileged group ('1') experiences a lower selection rate compared to the privileged group ('0'). This disparity is reflected by a Disparate Impact of 0.9755 and a Demographic Parity Difference of -0.0195. Furthermore, the model exhibits a lower True Positive Rate (TPR) for the unprivileged group (Equal Opportunity Difference: -0.0336), meaning individuals in the unprivileged group who are actually favorable are less likely to be correctly identified as such. The Average Odds Difference of -0.0175 suggests broader disparities in overall error rates (false positives and false negatives) across the groups, indicating that the model's performance is not consistently fair across all sensitive groups.

#### Why it is wrong

**Our pipeline:** **Data imbalance:**
- Severe size imbalance: 1 has 194 records vs 0 with 6 (32x ratio). Metrics for the smaller group have wide confidence intervals.
- Groups with fewer than 30 records (unreliable metrics): 0.

**Proxy features** (features correlated with the protected attribute):
- `foreign_worker`: r = 1.0 (strong)

**Root causes identified:**
1. Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.
2. Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: foreign_worker (r=1.0).
3. Representation bias: severely underrepresented groups make the model less reliable for those populations and can amplify existing disparities.
4. Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0889). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.
5. Unequal odds: both true-positive and false-positive rates differ across groups (AOD = +0.2013), indicating the model's errors are systematically distributed along group lines.

**Gemini:** Potential root causes for such bias often include historical biases embedded in the training data, where past decisions might have disproportionately favored the privileged group, leading the model to learn and perpetuate these patterns. Proxy features highly correlated with the protected attribute could also inadvertently lead to discriminatory outcomes. Additionally, data imbalance, where certain groups are underrepresented or exhibit different underlying distributions, can exacerbate these issues, making it harder for the model to learn fair representations.

#### How to fix it

**Our pipeline:** 1. Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.
2. Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.
3. Data collection: gather more samples from underrepresented groups. In the interim, use class-weighted training (class_weight='balanced' in scikit-learn) to upweight minority-group errors.
4. In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.
5. Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

**Gemini:** To mitigate the observed bias for the 'foreign_worker_original' attribute, various fairlearn strategies can be employed. For post-processing, `fairlearn.postprocessing.ThresholdOptimizer` can be used to adjust the decision threshold for each group, aiming to satisfy fairness constraints like Demographic Parity or Equalized Odds without retraining the original model. For in-processing mitigation, `fairlearn.reductions.ExponentiatedGradient` or `fairlearn.reductions.GridSearch` can be integrated into the model training pipeline. These methods allow a standard estimator to be optimized for both accuracy and fairness (e.g., 'demographic_parity' or 'equalized_odds') by learning a reweighing of the training samples or by generating a sequence of reweighted models. Data preprocessing techniques like `fairlearn.reductions.Reweighing` could also be applied to adjust sample weights based on group membership and label before training, to achieve specific fairness criteria.
