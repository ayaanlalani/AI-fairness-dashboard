# Fairness Report: HMDA Mortgage Lending

**Records analysed:** 2196
**DI bias threshold:** 0.8
**Favorable label:** 1 (loan originated / approved)

## Protected Attributes (legally protected under ECOA & Fair Housing Act)

| Attribute | Privileged | Legal basis |
|---|---|---|
| race | White | ECOA, Fair Housing Act |
| sex | Male | ECOA, Fair Housing Act |
| age_group | mid (35-54) | ECOA (age >= 40 protected) |

## Fairness Metrics

| Attribute | Privileged | DI | DPD | EOD | AOD | Theil | Bias? |
|---|---|---:|---:|---:|---:|---:|:---:|
| race | White | 0.8956 | -0.0776 | 0.0025 | 0.0117 | 0.4838 | no |
| sex | Male | 0.9291 | -0.0518 | -0.0076 | -0.0084 | 0.4838 | no |
| age_group | mid | 1.0138 | 0.0097 | 0.0008 | -0.015 | 0.4838 | no |

## Interpretation

- **Disparate Impact < 0.8** signals potential bias (80% rule / four-fifths rule).
- **DPD / EOD / AOD** closer to 0 is fairer.
- **Theil Index** closer to 0 means more equal benefit distribution.
- HMDA data contains real demographic attributes collected under federal law,
  making this dataset the regulatory standard for fair-lending analysis.

