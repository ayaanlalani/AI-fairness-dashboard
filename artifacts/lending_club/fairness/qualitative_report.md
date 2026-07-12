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
## gender  (DI = 0.0000, severity: CRITICAL)

### Per-group breakdown

| Group | N | % of total | Base rate | Selection rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| male | 312 | 52.0% | 0.8590 | 1.0000 | 1.0000 | 1.0000 |
| female | 288 | 48.0% | 0.9028 | 0.9931 | 1.0000 | 0.9286 |

### What is wrong

The model's favorable-outcome rate varies across gender groups. The highest rate is for **male** (100.0%) and the lowest is for **female** (99.3%), a gap of 0.7%.

- Demographic Parity Difference = -0.0069: the unprivileged group is selected less often.
- Equal Opportunity Difference = -0.0714: among truly deserving candidates, the model misses more unprivileged individuals.
- Average Odds Difference = -0.0357: error rates differ systematically across groups.

### Why it is wrong

**Root causes identified:**
1. Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0714). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.

### How to fix it

1. In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.

### Research-backed evidence

1. **Ethical Considerations and Bias Detection in Artificial Intelligence/Machine Learning Applications** (2025), Jayesh Rane, Reshma Amol Chaudhari, N. Rane. *Unknown venue*. Citations: 3.
   At a time when artificial intelligence (AI) and machine learning (ML) are used to make sensitive societal decisions such as the ones related to criminal justice, healthcare, finance, education, employment, algorithmic...
   https://www.semanticscholar.org/paper/bf11c75b72d2e8e6a65fa653fedde256277f5397
2. **Bias Detection and Mitigation in AI: A Framework for fair and inclusive Machine Learning Models** (2025), S. Sucharita, P. Shaw. *INTERNATIONAL JOURNAL OF SCIENTIFIC RESEARCH IN ENGINEERING AND MANAGEMENT*. Citations: 0.
   As AI becomes increasingly integrated into sectors like healthcare, finance, and recruitment, concerns around algorithmic bias, fairness, and data privacy are rising. This study addresses these ethical issues by intro...
   https://www.semanticscholar.org/paper/c41f9c5c5fff18dd742908efdf7c2683f4b70ddd

---
## income_level  (DI = --, severity: UNKNOWN)

### Per-group breakdown

| Group | N | % of total | Base rate | Selection rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| medium | 278 | 46.3% | 0.8885 | 0.9928 | 1.0000 | 0.9355 |
| low | 183 | 30.5% | 0.8525 | 1.0000 | 1.0000 | 1.0000 |
| high | 139 | 23.2% | 0.8993 | 1.0000 | 1.0000 | 1.0000 |

### What is wrong

Disparate Impact could not be computed for this attribute.

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
2. Unequal opportunity: the model is over-predicting favorable outcomes for the unprivileged group (EOD = +0.0645). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.

### How to fix it

1. Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.
2. In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.

---
## loan_amount_level  (DI = --, severity: UNKNOWN)

### Per-group breakdown

| Group | N | % of total | Base rate | Selection rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| medium | 320 | 53.3% | 0.8844 | 0.9938 | 1.0000 | 0.9459 |
| small | 181 | 30.2% | 0.8729 | 1.0000 | 1.0000 | 1.0000 |
| large | 99 | 16.5% | 0.8788 | 1.0000 | 1.0000 | 1.0000 |

### What is wrong

Disparate Impact could not be computed for this attribute.

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
2. Unequal opportunity: the model is over-predicting favorable outcomes for the unprivileged group (EOD = +0.0541). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.

### How to fix it

1. Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.
2. In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.

---
## Summary

| Attribute | DI | Severity | Top root cause |
|---|---:|---|---|
| gender | 0.0000 | CRITICAL | Unequal opportunity: the model is under-predicting favorable outcomes for the un... |
| income_level | -- | UNKNOWN | Proxy discrimination: features that correlate with the protected attribute allow... |
| loan_amount_level | -- | UNKNOWN | Proxy discrimination: features that correlate with the protected attribute allow... |
