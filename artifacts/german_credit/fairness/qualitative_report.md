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
- Average Odds Difference = +0.1256: error rates differ systematically across groups.

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
3. Unequal odds: both true-positive and false-positive rates differ across groups (AOD = +0.1256), indicating the model's errors are systematically distributed along group lines.

### How to fix it

1. Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.
2. In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.
3. Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

### Research-backed evidence

1. **A fairness-aware machine learning framework for maternal health in Ghana: integrating explainability, bias mitigation, and causal inference for ethical AI deployment** (2025), Augustus Osborne, Kobloobase O. Usani. *BioData Mining*. Citations: 0.
   Antenatal care (ANC) uptake in Ghana remains inequitable, with socioeconomic and geographic disparities limiting progress toward universal maternal health coverage (SDG 3). We present a novel, fairness-aware machine l...
   https://www.semanticscholar.org/paper/aeaffe41e8ee8adc7aa2b58c45202b8c930ebc19
2. **Bias Mitigation in Federated Healthcare Cloud Models** (2025), VijayaAshwin Jagadeesan. *Journal of Medical and Health Studies*. Citations: 0.
   Federated learning (FL) provides a privacy sensitive model training approach that can be applied to train predictive models in more than one hospital without having to share raw patient information. Nevertheless, hete...
   https://www.semanticscholar.org/paper/9a250df458d676629354ee0f609a26ae446a6060
3. **Evaluating Bias Mitigation Techniques in Credit and Marketing Models: Balancing Fairness and Performance** (2025), Sai Prashanth Pathi. *Journal of Advances in Developmental Research*. Citations: 0.
   Algorithmic decision making in financial services often amplifies existing societal biases due to imbalanced data and historical discrimination. Ensuring fairness in machine learning models, particularly within credit...
   https://www.semanticscholar.org/paper/2cfcf3d7567764d7bd76d1296b4e16a3054ff71a

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
- Average Odds Difference = +0.1649: error rates differ systematically across groups.

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
4. Unequal odds: both true-positive and false-positive rates differ across groups (AOD = +0.1649), indicating the model's errors are systematically distributed along group lines.

### How to fix it

1. Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.
2. Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.
3. In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.
4. Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

### Research-backed evidence

1. **Bias Detection and Mitigation in AI: A Framework for fair and inclusive Machine Learning Models** (2025), S. Sucharita, Prof. Kamalakanta Shaw. *INTERNATIONAL JOURNAL OF SCIENTIFIC RESEARCH IN ENGINEERING AND MANAGEMENT*. Citations: 0.
   As AI becomes increasingly integrated into sectors like healthcare, finance, and recruitment, concerns around algorithmic bias, fairness, and data privacy are rising. This study addresses these ethical issues by intro...
   https://www.semanticscholar.org/paper/c41f9c5c5fff18dd742908efdf7c2683f4b70ddd
2. **Ethical Considerations and Bias Detection in Artificial Intelligence/Machine Learning Applications** (2025), Jayesh Rane, Reshma Amol Chaudhari, N. Rane. *Unknown venue*. Citations: 0.
   At a time when artificial intelligence (AI) and machine learning (ML) are used to make sensitive societal decisions such as the ones related to criminal justice, healthcare, finance, education, employment, algorithmic...
   https://www.semanticscholar.org/paper/bf11c75b72d2e8e6a65fa653fedde256277f5397
3. **A fairness-aware machine learning framework for maternal health in Ghana: integrating explainability, bias mitigation, and causal inference for ethical AI deployment** (2025), Augustus Osborne, Kobloobase O. Usani. *BioData Mining*. Citations: 0.
   Antenatal care (ANC) uptake in Ghana remains inequitable, with socioeconomic and geographic disparities limiting progress toward universal maternal health coverage (SDG 3). We present a novel, fairness-aware machine l...
   https://www.semanticscholar.org/paper/aeaffe41e8ee8adc7aa2b58c45202b8c930ebc19

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
- Average Odds Difference = -0.2013: error rates differ systematically across groups.

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
5. Unequal odds: both true-positive and false-positive rates differ across groups (AOD = -0.2013), indicating the model's errors are systematically distributed along group lines.

### How to fix it

1. Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.
2. Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.
3. Data collection: gather more samples from underrepresented groups. In the interim, use class-weighted training (class_weight='balanced' in scikit-learn) to upweight minority-group errors.
4. In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.
5. Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

### Research-backed evidence

1. **Evaluating Bias Mitigation Techniques in Credit and Marketing Models: Balancing Fairness and Performance** (2025), Sai Prashanth Pathi. *Journal of Advances in Developmental Research*. Citations: 0.
   Algorithmic decision making in financial services often amplifies existing societal biases due to imbalanced data and historical discrimination. Ensuring fairness in machine learning models, particularly within credit...
   https://www.semanticscholar.org/paper/2cfcf3d7567764d7bd76d1296b4e16a3054ff71a
2. **FAIRNESS AND BIAS MITIGATION IN AI-BASED CREDIT SCORING USING ALTERNATIVE DATA: A FRAMEWORK FOR ETHICAL FINANCIAL INCLUSION** (2025), Derrick Atuobi Oware, Samuel Amfo Junior. *EPRA international journal of multidisciplinary research*. Citations: 0.
   There are a lot of opportunities to speed up financial inclusion through the use of artificial intelligence (AI) and alternative data in credit scoring, especially for underprivileged groups that have been shut out of...
   https://www.semanticscholar.org/paper/6882ca76d65014f2e26b31e7cc1b846de9b1da52
3. **Persistent Proxy Discrimination in HIV Testing Prediction Models: A National Fairness Audit of 386,775 US Adults** (2026), H. Farquhar. *medRxiv*. Citations: 0.
   https://www.semanticscholar.org/paper/cf0489c8f01a03aece54cdca0c5ac34ef952b70b

---
## Summary

| Attribute | DI | Severity | Top root cause |
|---|---:|---|---|
| Sex_original | 0.8595 | MODERATE (borderline) | Proxy discrimination: features that correlate with the protected attribute allow... |
| AgeGroup_original | 0.8212 | MODERATE (borderline) | Historical / label bias: the ground-truth labels themselves show a significant d... |
| foreign_worker_original | 0.9402 | MODERATE (borderline) | Historical / label bias: the ground-truth labels themselves show a significant d... |
