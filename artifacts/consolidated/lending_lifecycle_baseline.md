# Lending-Lifecycle Deterministic Baseline (Stage 1)

Deterministic drill-down across the three lending use cases (docs/RESEARCH_STAGING_PROMPT.md §2). Selection rate = share of the group receiving the favorable prediction. Subgroups with n < 15 are reported but flagged as too small to interpret. Disparate impact is computed against the most-favored adequately-sized subgroup. No LLM was involved in producing this file.

## UC1 — Consumer installment-credit scoring

### Per-attribute fairness metrics

| Attribute | PrivilegedValue | DisparateImpact | DemographicParityDiff | EqualOpportunityDiff | AverageOddsDiff | TheilIndex | Severity |
|---|---|---|---|---|---|---|---|
| Sex | male | 0.8888 | -0.0921 | -0.0650 | -0.0734 | 0.2808 | MODERATE (borderline) |
| AgeGroup | 40_plus | 0.9258 | -0.0625 | -0.0459 | -0.0465 | 0.2808 | MODERATE (borderline) |
| foreign_worker | 0 | 0.8387 | -0.1526 | -0.0866 | -0.1227 | 0.2808 | MODERATE (borderline) |

### Groups: `Sex_original`

| Sex_original | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| male | 690 | 0.8275 |  | 1.0000 |
| female | 310 | 0.7355 |  | 0.8888 |

### Groups: `AgeGroup_original`

| AgeGroup_original | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| 40_plus | 299 | 0.8428 |  | 1.0000 |
| under_40 | 701 | 0.7803 |  | 0.9258 |

### Groups: `foreign_worker_original`

| foreign_worker_original | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| 0 | 37 | 0.9459 |  | 1.0000 |
| 1 | 963 | 0.7934 |  | 0.8387 |

### Intersectional: `AgeGroup_original` × `Sex_original`

| AgeGroup_original | Sex_original | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|---|
| 40_plus | female | 69 | 0.8551 |  | 1.0000 |
| 40_plus | male | 230 | 0.8391 |  | 0.9814 |
| under_40 | male | 460 | 0.8217 |  | 0.9610 |
| under_40 | female | 241 | 0.7012 |  | 0.8201 |

> Largest adequately-sized gap: ('40_plus', 'female') at 0.8551 vs ('under_40', 'female') at 0.7012 (gap 0.1538, intersectional DI 0.8201).

---

## UC2 — Mortgage underwriting (HMDA Georgia)

### Per-attribute fairness metrics

| Attribute | PrivilegedValue | DisparateImpact | DemographicParityDiff | EqualOpportunityDiff | AverageOddsDiff | TheilIndex | Severity |
|---|---|---|---|---|---|---|---|
| race | White | 0.9371 | -0.0512 | 0.0032 | -0.0133 | 0.2960 | MODERATE (borderline) |
| sex | Male | 0.9572 | -0.0345 | -0.0128 | -0.0326 | 0.2960 | LOW (fair) |
| age_group | mid | 0.9791 | -0.0167 | -0.0060 | -0.0134 | 0.2960 | LOW (fair) |
| age_62_plus | under_62 | 0.8897 | -0.0889 | -0.0638 | -0.0545 | 0.2960 | MODERATE (borderline) |

### Groups: `race`

| race | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| Asian | 810 | 0.8556 |  | 1.0000 |
| White | 6337 | 0.8128 |  | 0.9501 |
| Black | 3701 | 0.7447 |  | 0.8704 |
| Pacific Islander | 27 | 0.7037 |  | 0.8225 |
| American Indian | 56 | 0.6607 |  | 0.7723 |
| Multiracial | 47 | 0.6383 |  | 0.7461 |

### Groups: `sex`

| sex | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| Male | 6357 | 0.8057 |  | 1.0000 |
| Female | 4621 | 0.7713 |  | 0.9572 |

### Groups: `age_group`

| age_group | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|
| young | 2304 | 0.8372 |  | 1.0000 |
| mid | 5507 | 0.7995 |  | 0.9550 |
| senior | 3167 | 0.7433 |  | 0.8878 |

### Intersectional: `race` × `sex`

| race | sex | n | selection_rate | small_subgroup | disparate_impact_vs_best |
|---|---|---|---|---|---|
| Asian | Female | 254 | 0.8622 |  | 1.0000 |
| Asian | Male | 556 | 0.8525 |  | 0.9888 |
| White | Male | 4019 | 0.8281 |  | 0.9604 |
| White | Female | 2318 | 0.7865 |  | 0.9121 |
| American Indian | Female | 25 | 0.7600 |  | 0.8815 |
| Black | Male | 1707 | 0.7452 |  | 0.8643 |
| Black | Female | 1994 | 0.7442 |  | 0.8632 |
| Pacific Islander | Male | 15 | 0.7333 |  | 0.8505 |
| Pacific Islander | Female | 12 | 0.6667 | yes (n<15) | 0.7732 |
| Multiracial | Male | 29 | 0.6552 |  | 0.7599 |
| Multiracial | Female | 18 | 0.6111 |  | 0.7088 |
| American Indian | Male | 31 | 0.5806 |  | 0.6734 |

> Largest adequately-sized gap: ('Asian', 'Female') at 0.8622 vs ('American Indian', 'Male') at 0.5806 (gap 0.2816, intersectional DI 0.6734).

---

## UC3 — P2P personal-loan default risk (Lending Club)

### Per-attribute fairness metrics

| Attribute | PrivilegedValue | DisparateImpact | DemographicParityDiff | EqualOpportunityDiff | AverageOddsDiff | TheilIndex | Severity |
|---|---|---|---|---|---|---|---|
| gender | male | 0.9931 | -0.0069 | 0.0000 | -0.0357 | -- | LOW (fair) |
| income_level | medium | 1.0072 | 0.0072 | 0.0000 | 0.0323 | -- | LOW (fair) |
| loan_amount_level | medium | 1.0063 | 0.0062 | 0.0000 | 0.0270 | -- | LOW (fair) |

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
