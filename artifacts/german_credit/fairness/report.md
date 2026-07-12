# Fairness Report: German Credit

**Records analysed:** 200
**DI bias threshold:** 0.8
**Favorable label:** 1 (good credit)

## Protected Attributes (domain-knowledge privileged groups)

| Attribute | Privileged | Rationale |
|---|---|---|
| Sex | male | Historically advantaged in credit markets |
| AgeGroup | 40_plus | Established credit history, higher income |
| foreign_worker | 0 (non-foreign) | National-origin discrimination risk |

## Fairness Metrics

| Attribute | Privileged | DI | DPD | EOD | AOD | Theil | Bias? |
|---|---|---:|---:|---:|---:|---:|:---:|
| Sex | male | 0.8595 | -0.1152 | -0.0512 | -0.1256 | 0.3084 | no |
| AgeGroup | 40_plus | 0.8212 | -0.1586 | -0.0513 | -0.1649 | 0.3084 | no |
| foreign_worker | 0 | 0.9402 | -0.0498 | -0.0889 | 0.2013 | 0.3084 | no |

## Interpretation

- **Disparate Impact < 0.8** signals potential bias (80 % rule).
- **DPD / EOD / AOD** closer to 0 is fairer.
- **Theil Index** closer to 0 means more equal benefit distribution.

