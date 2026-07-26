# Qualitative Fairness Analysis: Lending Club P2P Loans

**Records analysed:** 600
**Favorable label:** 0
**Bias threshold (DI):** 0.8

**Overall selection rate:** 99.7%

## Reference Audit Specification

**Name:** Deterministic Remediation Audit

**Core metrics:** Disparate Impact, Demographic Parity Difference, Equal Opportunity Difference, Average Odds Difference, Theil Index

**Required response elements:** per-group breakdown, severity classification, root-cause analysis, mitigation recommendations, research-backed justification

This deterministic baseline pairs quantitative disparity metrics with interpretable root-cause diagnostics (rule-based inference from metrics and feature correlations, not causal identification) and concrete mitigation options so the output is directly usable for remediation planning and LLM benchmark scoring.

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

### Research-backed evidence

1. **Trust and Credit: The Role of Appearance in Peer-to-peer Lending** (2012), J. Duarte, Stephan Siegel, Lance A. Young. *Unknown venue*. Citations: 961.
   https://www.semanticscholar.org/paper/b9475170ca58eb83868376517d7ab50bb4f52d65
2. **What’s in a Picture?** (2011), Devin G. Pope, Justin R. Sydnor. *The Journal of human resources*. Citations: 235.
   https://www.semanticscholar.org/paper/6e57ef0eacc47268313df14dbe0448d1842481b4
3. **Consumer-lending discrimination in the FinTech Era** (2019), Robert P. Bartlett, Adair Morse, Richard Stanton. *Journal of Financial Economics*. Citations: 587.
   Abstract U.S. fair-lending law prohibits lenders from making credit determinations that disparately affect minority borrowers if those determinations are based on characteristics unrelated to creditworthiness. Using a...
   https://www.semanticscholar.org/paper/97a98065c1fec5c36104f586c93ca756caf5caaf

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

### Research-backed evidence

1. **Consumer-lending discrimination in the FinTech Era** (2019), Robert P. Bartlett, Adair Morse, Richard Stanton. *Journal of Financial Economics*. Citations: 587.
   Abstract U.S. fair-lending law prohibits lenders from making credit determinations that disparately affect minority borrowers if those determinations are based on characteristics unrelated to creditworthiness. Using a...
   https://www.semanticscholar.org/paper/97a98065c1fec5c36104f586c93ca756caf5caaf
2. **Do Local Capital Market Conditions Affect Consumers' Borrowing Decisions?** (2017), Alexander W. Butler, Jess Cornaggia, Umit G. Gurun. *Management Sciences*. Citations: 99.
   https://www.semanticscholar.org/paper/b39f69b46e868c55a81a2649d1dae6366b5038cd
3. **Certifying and Removing Disparate Impact** (2014), Michael Feldman, Sorelle A. Friedler, John Moeller. *Knowledge Discovery and Data Mining*. Citations: 2287.
   What does it mean for an algorithm to be biased? In U.S. law, unintentional bias is encoded via disparate impact, which occurs when a selection process has widely different outcomes for different groups, even as it ap...
   https://www.semanticscholar.org/paper/0fee3b6c72f7676b4934651e517d0a328048c600

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

### Research-backed evidence

1. **Do Local Capital Market Conditions Affect Consumers' Borrowing Decisions?** (2017), Alexander W. Butler, Jess Cornaggia, Umit G. Gurun. *Management Sciences*. Citations: 99.
   https://www.semanticscholar.org/paper/b39f69b46e868c55a81a2649d1dae6366b5038cd
2. **Certifying and Removing Disparate Impact** (2014), Michael Feldman, Sorelle A. Friedler, John Moeller. *Knowledge Discovery and Data Mining*. Citations: 2287.
   What does it mean for an algorithm to be biased? In U.S. law, unintentional bias is encoded via disparate impact, which occurs when a selection process has widely different outcomes for different groups, even as it ap...
   https://www.semanticscholar.org/paper/0fee3b6c72f7676b4934651e517d0a328048c600
3. **Data preprocessing techniques for classification without discrimination** (2011), F. Kamiran, T. Calders. *Knowledge and Information Systems*. Citations: 1610.
   Recently, the following Discrimination-Aware Classification Problem was introduced: Suppose we are given training data that exhibit unlawful discrimination; e.g., toward sensitive attributes such as gender or ethnicit...
   https://www.semanticscholar.org/paper/3ac3c11bf6cd8ccc657eb629148d6e346e52c8e0

---
## Summary

| Attribute | DI | Severity | Top root cause |
|---|---:|---|---|
| gender | 0.9931 | LOW (fair) | No significant disparate impact detected. The model treats groups approximately ... |
| income_level | 1.0072 | LOW (fair) | Proxy discrimination: features that correlate with the protected attribute allow... |
| loan_amount_level | 1.0063 | LOW (fair) | Proxy discrimination: features that correlate with the protected attribute allow... |
