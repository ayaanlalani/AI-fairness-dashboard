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
- Average Odds Difference = -0.0117: error rates differ systematically across groups.

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

### Research-backed evidence

1. **Bias Detection and Mitigation in AI: A Framework for fair and inclusive Machine Learning Models** (2025), S. Sucharita, Prof. Kamalakanta Shaw. *INTERNATIONAL JOURNAL OF SCIENTIFIC RESEARCH IN ENGINEERING AND MANAGEMENT*. Citations: 0.
   As AI becomes increasingly integrated into sectors like healthcare, finance, and recruitment, concerns around algorithmic bias, fairness, and data privacy are rising. This study addresses these ethical issues by intro...
   https://www.semanticscholar.org/paper/c41f9c5c5fff18dd742908efdf7c2683f4b70ddd
2. **Ethical Considerations and Bias Detection in Artificial Intelligence/Machine Learning Applications** (2025), Jayesh Rane, Reshma Amol Chaudhari, N. Rane. *Unknown venue*. Citations: 0.
   At a time when artificial intelligence (AI) and machine learning (ML) are used to make sensitive societal decisions such as the ones related to criminal justice, healthcare, finance, education, employment, algorithmic...
   https://www.semanticscholar.org/paper/bf11c75b72d2e8e6a65fa653fedde256277f5397
3. **Algorithmic Lending Bias: Evaluating the Fairness of Historical Redlining in Loan Approvals** (2024), K. Sarwal, Sheikh Rabiul Islam. *BigData Congress [Services Society]*. Citations: 1.
   This paper investigates the persistent influence of historical redlining on modern AI algorithms used in real estate and loan approvals. Utilizing Home Mortgage Disclosure Act (HMDA) data, we uncover demographic biase...
   https://www.semanticscholar.org/paper/ce2a78ceae4b197a508c3c40451c0288d5bce0cd

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
- Average Odds Difference = +0.0084: error rates differ systematically across groups.

### Why it is wrong


### How to fix it

1. Post-processing -- Calibrated Equalized Odds (aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing): a minimal intervention that adjusts thresholds to bring DI above the 0.8 threshold while preserving calibration.

### Research-backed evidence

1. **A fairness-aware machine learning framework for maternal health in Ghana: integrating explainability, bias mitigation, and causal inference for ethical AI deployment** (2025), Augustus Osborne, Kobloobase O. Usani. *BioData Mining*. Citations: 0.
   Antenatal care (ANC) uptake in Ghana remains inequitable, with socioeconomic and geographic disparities limiting progress toward universal maternal health coverage (SDG 3). We present a novel, fairness-aware machine l...
   https://www.semanticscholar.org/paper/aeaffe41e8ee8adc7aa2b58c45202b8c930ebc19
2. **Bias Mitigation in Federated Healthcare Cloud Models** (2025), VijayaAshwin Jagadeesan. *Journal of Medical and Health Studies*. Citations: 0.
   Federated learning (FL) provides a privacy sensitive model training approach that can be applied to train predictive models in more than one hospital without having to share raw patient information. Nevertheless, hete...
   https://www.semanticscholar.org/paper/9a250df458d676629354ee0f609a26ae446a6060
3. **Algorithmic Lending Bias: Evaluating the Fairness of Historical Redlining in Loan Approvals** (2024), K. Sarwal, Sheikh Rabiul Islam. *BigData Congress [Services Society]*. Citations: 1.
   This paper investigates the persistent influence of historical redlining on modern AI algorithms used in real estate and loan approvals. Utilizing Home Mortgage Disclosure Act (HMDA) data, we uncover demographic biase...
   https://www.semanticscholar.org/paper/ce2a78ceae4b197a508c3c40451c0288d5bce0cd

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

### Research-backed evidence

1. **Bias Detection and Mitigation in AI: A Framework for fair and inclusive Machine Learning Models** (2025), S. Sucharita, Prof. Kamalakanta Shaw. *INTERNATIONAL JOURNAL OF SCIENTIFIC RESEARCH IN ENGINEERING AND MANAGEMENT*. Citations: 0.
   As AI becomes increasingly integrated into sectors like healthcare, finance, and recruitment, concerns around algorithmic bias, fairness, and data privacy are rising. This study addresses these ethical issues by intro...
   https://www.semanticscholar.org/paper/c41f9c5c5fff18dd742908efdf7c2683f4b70ddd
2. **Ethical Considerations and Bias Detection in Artificial Intelligence/Machine Learning Applications** (2025), Jayesh Rane, Reshma Amol Chaudhari, N. Rane. *Unknown venue*. Citations: 0.
   At a time when artificial intelligence (AI) and machine learning (ML) are used to make sensitive societal decisions such as the ones related to criminal justice, healthcare, finance, education, employment, algorithmic...
   https://www.semanticscholar.org/paper/bf11c75b72d2e8e6a65fa653fedde256277f5397
3. **Assessing Adherence to TRIPOD+AI Guidelines in Machine Learning Models for Predicting Small for Gestational Age and Fetal Growth Restriction: A Systematic Review.** (2025), G. Zamagni, C. Fregona, Moira Barbieri. *American Journal of Obstetrics & Gynecology MFM*. Citations: 0.
   OBJECTIVES Fetal growth restriction (FGR) significantly contribute to perinatal morbidity, mortality, and long-term adverse health outcomes. While small for gestational age (SGA) is often used as a proxy for FGR, it d...
   https://www.semanticscholar.org/paper/29a91395de9d92cd8b4afb017367b8cd649dbb1a

---
## Summary

| Attribute | DI | Severity | Top root cause |
|---|---:|---|---|
| race | 0.8956 | MODERATE (borderline) | Historical / label bias: the ground-truth labels themselves show a significant d... |
| sex | 0.9291 | MODERATE (borderline) | None identified |
| age_group | 1.0138 | LOW (fair) | Historical / label bias: the ground-truth labels themselves show a significant d... |
