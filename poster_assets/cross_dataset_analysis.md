# Cross-Dataset Qualitative Fairness Analysis

# Qualitative Fairness Analysis: German Credit

**Records analysed:** 200
**Favorable label:** 1
**Bias threshold (DI):** 0.8

**Overall selection rate:** 78.5%

## Reference Audit Specification

**Name:** Deterministic Remediation Audit

**Core metrics:** Disparate Impact, Demographic Parity Difference, Equal Opportunity Difference, Average Odds Difference, Theil Index

**Required response elements:** per-group breakdown, severity classification, root-cause analysis, mitigation recommendations, research-backed justification

This deterministic baseline pairs quantitative disparity metrics with interpretable causal diagnostics and concrete mitigation options so the output is directly usable for remediation planning and LLM benchmark scoring.

---
## Sex_original  (DI = 0.8595, severity: MODERATE (borderline))

### Per-group breakdown

| Group | N | % of total | Base rate | Selection rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| male | 139 | 69.5% | 0.7122 | 0.8201 | 0.9293 | 0.5500 |
| female | 61 | 30.5% | 0.6721 | 0.7049 | 0.8780 | 0.3500 |

### What is wrong

The model's favorable-outcome rate varies across Sex_original groups. The highest rate is for **male** (82.0%) and the lowest is for **female** (70.5%), a gap of 11.5%.

- Demographic Parity Difference = -0.1152: the unprivileged group is selected less often.
- Equal Opportunity Difference = -0.0512: among truly deserving candidates, the model misses more unprivileged individuals.
- Average Odds Difference = -0.1256: error rates differ systematically across groups.

### Why it is wrong

**Proxy features** (features correlated with the protected attribute):
- `Sex`: r = 1.0 (strong)
- `Housing`: r = -0.267 (weak)
- `Age`: r = 0.216 (weak)
- `Job`: r = 0.165 (weak)
- `AgeGroup`: r = -0.151 (weak)

**Root causes identified:**
1. Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: Sex (r=1.0).
2. Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0512). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.
3. Unequal odds: both true-positive and false-positive rates differ across groups (AOD = -0.1256), indicating the model's errors are systematically distributed along group lines.

### How to fix it

1. Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.
2. In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.
3. Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

---
## AgeGroup_original  (DI = 0.8212, severity: MODERATE (borderline))

### Per-group breakdown

| Group | N | % of total | Base rate | Selection rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| under_40 | 129 | 64.5% | 0.6589 | 0.7287 | 0.8941 | 0.4091 |
| 40_plus | 71 | 35.5% | 0.7746 | 0.8873 | 0.9455 | 0.6875 |

### What is wrong

The model's favorable-outcome rate varies across AgeGroup_original groups. The highest rate is for **40_plus** (88.7%) and the lowest is for **under_40** (72.9%), a gap of 15.9%.

- Demographic Parity Difference = -0.1586: the unprivileged group is selected less often.
- Equal Opportunity Difference = -0.0513: among truly deserving candidates, the model misses more unprivileged individuals.
- Average Odds Difference = -0.1649: error rates differ systematically across groups.

### Why it is wrong

**Proxy features** (features correlated with the protected attribute):
- `AgeGroup`: r = 1.0 (strong)
- `Age`: r = -0.841 (strong)
- `Housing`: r = 0.229 (weak)
- `Sex`: r = -0.151 (weak)

**Root causes identified:**
1. Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.
2. Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: AgeGroup (r=1.0), Age (r=-0.841).
3. Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0513). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.
4. Unequal odds: both true-positive and false-positive rates differ across groups (AOD = -0.1649), indicating the model's errors are systematically distributed along group lines.

### How to fix it

1. Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.
2. Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.
3. In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.
4. Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

---
## foreign_worker_original  (DI = 0.9402, severity: MODERATE (borderline))

### Per-group breakdown

| Group | N | % of total | Base rate | Selection rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 194 | 97.0% | 0.6959 | 0.7835 | 0.9111 | 0.4915 |
| 0 | 6 | 3.0% | 0.8333 | 0.8333 | 1.0000 | 0.0000 |

### What is wrong

The model's favorable-outcome rate varies across foreign_worker_original groups. The highest rate is for **0** (83.3%) and the lowest is for **1** (78.3%), a gap of 5.0%.

- Demographic Parity Difference = -0.0498: the unprivileged group is selected less often.
- Equal Opportunity Difference = -0.0889: among truly deserving candidates, the model misses more unprivileged individuals.
- Average Odds Difference = +0.2013: error rates differ systematically across groups.

### Why it is wrong

**Data imbalance:**
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

### How to fix it

1. Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.
2. Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.
3. Data collection: gather more samples from underrepresented groups. In the interim, use class-weighted training (class_weight='balanced' in scikit-learn) to upweight minority-group errors.
4. In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.
5. Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

---
## Summary

| Attribute | DI | Severity | Top root cause |
|---|---:|---|---|
| Sex_original | 0.8595 | MODERATE (borderline) | Proxy discrimination: features that correlate with the protected attribute allow... |
| AgeGroup_original | 0.8212 | MODERATE (borderline) | Historical / label bias: the ground-truth labels themselves show a significant d... |
| foreign_worker_original | 0.9402 | MODERATE (borderline) | Historical / label bias: the ground-truth labels themselves show a significant d... |


---

# Qualitative Fairness Analysis: HMDA Mortgage Lending (Georgia)

**Records analysed:** 2196
**Favorable label:** 1
**Bias threshold (DI):** 0.8

**Overall selection rate:** 70.9%

## Reference Audit Specification

**Name:** Deterministic Remediation Audit

**Core metrics:** Disparate Impact, Demographic Parity Difference, Equal Opportunity Difference, Average Odds Difference, Theil Index

**Required response elements:** per-group breakdown, severity classification, root-cause analysis, mitigation recommendations, research-backed justification

This deterministic baseline pairs quantitative disparity metrics with interpretable causal diagnostics and concrete mitigation options so the output is directly usable for remediation planning and LLM benchmark scoring.

---
## race  (DI = 0.8956, severity: MODERATE (borderline))

### Per-group breakdown

| Group | N | % of total | Base rate | Selection rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| White | 1252 | 57.0% | 0.7572 | 0.7428 | 0.9705 | 0.0329 |
| Black | 748 | 34.1% | 0.6417 | 0.6444 | 0.9750 | 0.0522 |
| Asian | 173 | 7.9% | 0.7977 | 0.7861 | 0.9638 | 0.0857 |
| American Indian | 10 | 0.5% | 0.4000 | 0.4000 | 1.0000 | 0.0000 |
| Multiracial | 8 | 0.4% | 0.3750 | 0.3750 | 1.0000 | 0.0000 |
| Pacific Islander | 5 | 0.2% | 0.6000 | 0.6000 | 1.0000 | 0.0000 |

### What is wrong

The model's favorable-outcome rate varies across race groups. The highest rate is for **Asian** (78.6%) and the lowest is for **Multiracial** (37.5%), a gap of 41.1%.

- Demographic Parity Difference = -0.0776: the unprivileged group is selected less often.
- Equal Opportunity Difference = +0.0025: among truly deserving candidates, the model catches more unprivileged individuals.
- Average Odds Difference = +0.0117: error rates differ systematically across groups.

### Why it is wrong

**Data imbalance:**
- Severe size imbalance: White has 1252 records vs Pacific Islander with 5 (250x ratio). Metrics for the smaller group have wide confidence intervals.
- Groups with fewer than 30 records (unreliable metrics): American Indian, Multiracial, Pacific Islander.
- Base-rate disparity in the ground truth: Asian has 79.8% favorable outcomes vs Multiracial at 37.5% (gap = 42.3%). The model may be correctly learning a real disparity in historical outcomes, but this historical pattern itself may reflect systemic bias.

**Root causes identified:**
1. Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.
2. Representation bias: severely underrepresented groups make the model less reliable for those populations and can amplify existing disparities.

### How to fix it

1. Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.
2. Data collection: gather more samples from underrepresented groups. In the interim, use class-weighted training (class_weight='balanced' in scikit-learn) to upweight minority-group errors.

---
## sex  (DI = 0.9291, severity: MODERATE (borderline))

### Per-group breakdown

| Group | N | % of total | Base rate | Selection rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Male | 1276 | 58.1% | 0.7375 | 0.7312 | 0.9745 | 0.0478 |
| Female | 920 | 41.9% | 0.6902 | 0.6793 | 0.9669 | 0.0386 |

### What is wrong

The model's favorable-outcome rate varies across sex groups. The highest rate is for **Male** (73.1%) and the lowest is for **Female** (67.9%), a gap of 5.2%.

- Demographic Parity Difference = -0.0518: the unprivileged group is selected less often.
- Equal Opportunity Difference = -0.0076: among truly deserving candidates, the model misses more unprivileged individuals.
- Average Odds Difference = -0.0084: error rates differ systematically across groups.

### Why it is wrong


### How to fix it

1. Post-processing -- Calibrated Equalized Odds (aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing): a minimal intervention that adjusts thresholds to bring DI above the 0.8 threshold while preserving calibration.

---
## age_group  (DI = 1.0138, severity: LOW (fair))

### Per-group breakdown

| Group | N | % of total | Base rate | Selection rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| mid | 1121 | 51.0% | 0.7083 | 0.7047 | 0.9710 | 0.0581 |
| senior | 611 | 27.8% | 0.6792 | 0.6661 | 0.9735 | 0.0153 |
| young | 464 | 21.1% | 0.7909 | 0.7780 | 0.9700 | 0.0515 |

### What is wrong

No significant disparity detected (DI = 1.0138). The model treats age_group groups approximately equally.

### Why it is wrong

**Root causes identified:**
1. Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.

### How to fix it

1. Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.

---
## Summary

| Attribute | DI | Severity | Top root cause |
|---|---:|---|---|
| race | 0.8956 | MODERATE (borderline) | Historical / label bias: the ground-truth labels themselves show a significant d... |
| sex | 0.9291 | MODERATE (borderline) | None identified |
| age_group | 1.0138 | LOW (fair) | Historical / label bias: the ground-truth labels themselves show a significant d... |


---

# Qualitative Fairness Analysis: Lending Club P2P Loans

**Records analysed:** 600
**Favorable label:** 0
**Bias threshold (DI):** 0.8

**Overall selection rate:** 99.7%

## Reference Audit Specification

**Name:** Deterministic Remediation Audit

**Core metrics:** Disparate Impact, Demographic Parity Difference, Equal Opportunity Difference, Average Odds Difference, Theil Index

**Required response elements:** per-group breakdown, severity classification, root-cause analysis, mitigation recommendations, research-backed justification

This deterministic baseline pairs quantitative disparity metrics with interpretable causal diagnostics and concrete mitigation options so the output is directly usable for remediation planning and LLM benchmark scoring.

---
## gender  (DI = 0.9931, severity: LOW (fair))

### Per-group breakdown

| Group | N | % of total | Base rate | Selection rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| male | 312 | 52.0% | 0.8590 | 1.0000 | 1.0000 | 1.0000 |
| female | 288 | 48.0% | 0.9028 | 0.9931 | 1.0000 | 0.9286 |

### What is wrong

No significant disparity detected (DI = 0.9931). The model treats gender groups approximately equally.

### Why it is wrong

**Root causes identified:**
1. No significant disparate impact detected. The model treats groups approximately equally on the measured metrics.

### How to fix it

1. Continue monitoring: fairness can drift as data distributions change. Re-run this analysis periodically and after any model retraining.

---
## income_level  (DI = 1.0072, severity: LOW (fair))

### Per-group breakdown

| Group | N | % of total | Base rate | Selection rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| medium | 278 | 46.3% | 0.8885 | 0.9928 | 1.0000 | 0.9355 |
| low | 183 | 30.5% | 0.8525 | 1.0000 | 1.0000 | 1.0000 |
| high | 139 | 23.2% | 0.8993 | 1.0000 | 1.0000 | 1.0000 |

### What is wrong

No significant disparity detected (DI = 1.0072). The model treats income_level groups approximately equally.

### Why it is wrong

**Proxy features** (features correlated with the protected attribute):
- `annual_inc`: r = 0.916 (strong)
- `loan_amnt`: r = 0.458 (moderate)
- `installment`: r = 0.437 (moderate)
- `revol_bal`: r = 0.309 (moderate)
- `total_acc`: r = 0.295 (weak)
- `home_ownership_mortgage`: r = 0.276 (weak)
- `open_acc`: r = 0.25 (weak)
- `home_ownership_rent`: r = 0.25 (weak)

**Root causes identified:**
1. Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: annual_inc (r=0.916), loan_amnt (r=0.458), installment (r=0.437), revol_bal (r=0.309).

### How to fix it

1. Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.

---
## loan_amount_level  (DI = 1.0063, severity: LOW (fair))

### Per-group breakdown

| Group | N | % of total | Base rate | Selection rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| medium | 320 | 53.3% | 0.8844 | 0.9938 | 1.0000 | 0.9459 |
| small | 181 | 30.2% | 0.8729 | 1.0000 | 1.0000 | 1.0000 |
| large | 99 | 16.5% | 0.8788 | 1.0000 | 1.0000 | 1.0000 |

### What is wrong

No significant disparity detected (DI = 1.0063). The model treats loan_amount_level groups approximately equally.

### Why it is wrong

**Data imbalance:**
- Moderate size imbalance: medium (320) is 3.2x larger than large (99).

**Proxy features** (features correlated with the protected attribute):
- `loan_amnt`: r = 0.922 (strong)
- `installment`: r = 0.876 (strong)
- `annual_inc`: r = 0.424 (moderate)
- `term_60_months`: r = 0.4 (moderate)
- `revol_bal`: r = 0.321 (moderate)
- `home_ownership_rent`: r = 0.189 (weak)
- `home_ownership_mortgage`: r = 0.169 (weak)
- `total_acc`: r = 0.167 (weak)

**Root causes identified:**
1. Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: loan_amnt (r=0.922), installment (r=0.876), annual_inc (r=0.424), term_60_months (r=0.4), revol_bal (r=0.321).

### How to fix it

1. Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.

---
## Summary

| Attribute | DI | Severity | Top root cause |
|---|---:|---|---|
| gender | 0.9931 | LOW (fair) | No significant disparate impact detected. The model treats groups approximately ... |
| income_level | 1.0072 | LOW (fair) | Proxy discrimination: features that correlate with the protected attribute allow... |
| loan_amount_level | 1.0063 | LOW (fair) | Proxy discrimination: features that correlate with the protected attribute allow... |


---
