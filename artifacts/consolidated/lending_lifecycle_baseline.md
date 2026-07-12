# Lending-Lifecycle Deterministic Baseline (Stage 1)

Deterministic drill-down across the three lending use cases (docs/RESEARCH_STAGING_PROMPT.md §2). Selection rate = share of the group receiving the favorable prediction. Subgroups with n < 15 are reported but flagged as too small to interpret. Disparate impact is computed against the most-favored adequately-sized subgroup. No LLM was involved in producing this file.

## UC1 — Consumer installment-credit scoring

### Per-attribute fairness metrics

| Attribute | PrivilegedValue | DisparateImpact | DemographicParityDiff | EqualOpportunityDiff | AverageOddsDiff | TheilIndex | Severity |
|---|---|---|---|---|---|---|---|
| Sex | male | 0.8595 | -0.1152 | -0.0512 | -0.1256 | 0.3084 | MODERATE (borderline) |
| AgeGroup | 40_plus | 0.8212 | -0.1586 | -0.0513 | -0.1649 | 0.3084 | MODERATE (borderline) |
| foreign_worker | 0 | 0.9402 | -0.0498 | -0.0889 | 0.2013 | 0.3084 | MODERATE (borderline) |

### Groups: `Sex_original`

| Sex_original | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| male | 139 | 0.8201 |  | 1.0000 |
| female | 61 | 0.7049 |  | 0.8595 |

### Groups: `AgeGroup_original`

| AgeGroup_original | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| 40_plus | 71 | 0.8873 |  | 1.0000 |
| under_40 | 129 | 0.7287 |  | 0.8212 |

### Groups: `foreign_worker_original`

| foreign_worker_original | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| 0 | 6 | 0.8333 | yes (n<15) | 1.0636 |
| 1 | 194 | 0.7835 |  | 1.0000 |

> **Small-subgroup caveat:** 0 (n < 15) — metrics for these groups carry high variance and must not be interpreted as stable disparities.

### Intersectional: `AgeGroup_original` × `Sex_original`

| AgeGroup_original | Sex_original | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|---|
| 40_plus | female | 15 | 1.0000 |  | 1.0000 |
| 40_plus | male | 56 | 0.8571 |  | 0.8571 |
| under_40 | male | 83 | 0.7952 |  | 0.7952 |
| under_40 | female | 46 | 0.6087 |  | 0.6087 |

> Largest adequately-sized gap: ('40_plus', 'female') at 1.0000 vs ('under_40', 'female') at 0.6087 (gap 0.3913, intersectional DI 0.6087).

---

## UC2 — Mortgage underwriting (HMDA Georgia)

### Per-attribute fairness metrics

| Attribute | PrivilegedValue | DisparateImpact | DemographicParityDiff | EqualOpportunityDiff | AverageOddsDiff | TheilIndex | Severity |
|---|---|---|---|---|---|---|---|
| race | White | 0.8956 | -0.0776 | 0.0025 | 0.0117 | 0.4838 | MODERATE (borderline) |
| sex | Male | 0.9291 | -0.0518 | -0.0076 | -0.0084 | 0.4838 | MODERATE (borderline) |
| age_group | mid | 1.0138 | 0.0097 | 0.0008 | -0.0150 | 0.4838 | LOW (fair) |

### Groups: `race`

| race | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| Asian | 173 | 0.7861 |  | 1.0000 |
| White | 1252 | 0.7428 |  | 0.9449 |
| Black | 748 | 0.6444 |  | 0.8197 |
| Pacific Islander | 5 | 0.6000 | yes (n<15) | 0.7632 |
| American Indian | 10 | 0.4000 | yes (n<15) | 0.5088 |
| Multiracial | 8 | 0.3750 | yes (n<15) | 0.4770 |

> **Small-subgroup caveat:** Pacific Islander, American Indian, Multiracial (n < 15) — metrics for these groups carry high variance and must not be interpreted as stable disparities.

### Groups: `sex`

| sex | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| Male | 1276 | 0.7312 |  | 1.0000 |
| Female | 920 | 0.6793 |  | 0.9291 |

### Groups: `age_group`

| age_group | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| young | 464 | 0.7780 |  | 1.0000 |
| mid | 1121 | 0.7047 |  | 0.9058 |
| senior | 611 | 0.6661 |  | 0.8562 |

### Intersectional: `race` × `sex`

| race | sex | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|---|
| Asian | Male | 130 | 0.8077 |  | 1.0000 |
| White | Male | 794 | 0.7531 |  | 0.9325 |
| White | Female | 458 | 0.7249 |  | 0.8975 |
| Asian | Female | 43 | 0.7209 |  | 0.8926 |
| Black | Male | 338 | 0.6598 |  | 0.8168 |
| Black | Female | 410 | 0.6317 |  | 0.7821 |
| Pacific Islander | Male | 5 | 0.6000 | yes (n<15) | 0.7429 |
| American Indian | Male | 4 | 0.5000 | yes (n<15) | 0.6190 |
| Multiracial | Male | 5 | 0.4000 | yes (n<15) | 0.4952 |
| American Indian | Female | 6 | 0.3333 | yes (n<15) | 0.4127 |
| Multiracial | Female | 3 | 0.3333 | yes (n<15) | 0.4127 |

> Largest adequately-sized gap: ('Asian', 'Male') at 0.8077 vs ('Black', 'Female') at 0.6317 (gap 0.1760, intersectional DI 0.7821).

---

## UC3 — P2P personal-loan default risk (Lending Club)

### Per-attribute fairness metrics

| Attribute | PrivilegedValue | DisparateImpact | DemographicParityDiff | EqualOpportunityDiff | AverageOddsDiff | TheilIndex | Severity |
|---|---|---|---|---|---|---|---|
| gender | male | 0.0000 | -0.0069 | -0.0714 | -0.0357 | -- | CRITICAL |
| income_level | medium | -- | 0.0072 | 0.0645 | 0.0323 | -- | UNKNOWN |
| loan_amount_level | medium | -- | 0.0063 | 0.0541 | 0.0270 | -- | UNKNOWN |

### Groups: `gender`

| gender | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| male | 312 | 1.0000 |  | 1.0000 |
| female | 288 | 0.9931 |  | 0.9931 |

### Groups: `income_level`

| income_level | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| high | 139 | 1.0000 |  | 1.0000 |
| low | 183 | 1.0000 |  | 1.0000 |
| medium | 278 | 0.9928 |  | 0.9928 |

### Groups: `loan_amount_level`

| loan_amount_level | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| large | 99 | 1.0000 |  | 1.0000 |
| small | 181 | 1.0000 |  | 1.0000 |
| medium | 320 | 0.9938 |  | 0.9938 |

### Intersectional: `income_level` × `loan_amount_level`

| income_level | loan_amount_level | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|---|
| high | large | 53 | 1.0000 |  | 1.0000 |
| high | medium | 71 | 1.0000 |  | 1.0000 |
| high | small | 15 | 1.0000 |  | 1.0000 |
| low | large | 9 | 1.0000 | yes (n<15) | 1.0000 |
| low | medium | 82 | 1.0000 |  | 1.0000 |
| low | small | 92 | 1.0000 |  | 1.0000 |
| medium | large | 37 | 1.0000 |  | 1.0000 |
| medium | small | 74 | 1.0000 |  | 1.0000 |
| medium | medium | 167 | 0.9880 |  | 0.9880 |

> Largest adequately-sized gap: ('high', 'large') at 1.0000 vs ('medium', 'medium') at 0.9880 (gap 0.0120, intersectional DI 0.9880).

### Intersectional: `gender` × `income_level`

| gender | income_level | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|---|
| female | high | 69 | 1.0000 |  | 1.0000 |
| female | low | 87 | 1.0000 |  | 1.0000 |
| male | high | 70 | 1.0000 |  | 1.0000 |
| male | low | 96 | 1.0000 |  | 1.0000 |
| male | medium | 146 | 1.0000 |  | 1.0000 |
| female | medium | 132 | 0.9848 |  | 0.9848 |

> Largest adequately-sized gap: ('female', 'high') at 1.0000 vs ('female', 'medium') at 0.9848 (gap 0.0152, intersectional DI 0.9848).

---
