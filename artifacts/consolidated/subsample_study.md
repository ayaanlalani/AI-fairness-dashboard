# Subsampling study

Population: `hmda_dataset/metrics/classification_predictions.csv` (n=10978); B=2000 SRS draws per size; seed 20260808.

## Population ground truth per convention

- **max_rate**: reference Asian x Female; worst American Indian x Male; violating: American Indian x Male, Multiracial x Female, Multiracial x Male, Pacific Islander x Female
- **max_rate_floor15**: reference Asian x Female; worst American Indian x Male; violating: American Indian x Male, Multiracial x Female, Multiracial x Male, Pacific Islander x Female
- **control**: reference White x Male; worst American Indian x Male; violating: American Indian x Male, Multiracial x Female, Multiracial x Male

## Rates at m=2196 (primary convention: max_rate_floor15), % of draws

| policy | false clearance | worst cell wrong | abstains | sign inversion | false alarm | suppressed cells | coverage |
|---|---|---|---|---|---|---|---|
| floor n>=15 | 0.0 [0.0, 0.2] | 100.0 [99.7, 100.0] | 0.0 | 0.0 [0.0, 0.2] | 17.9 [16.3, 19.6] | 6.0 | -- |
| floor n>=30 | 0.0 [0.0, 0.2] | 100.0 [99.8, 100.0] | 0.0 | 0.0 [0.0, 0.2] | 17.9 [16.3, 19.6] | 6.0 | -- |
| point, no floor | 83.8 [82.1, 85.3] | 72.4 [70.4, 74.3] | 0.0 | 79.0 [77.1, 80.7] | 70.5 [68.5, 72.5] | 0.1 | -- |
| Wilson interval | 0.0 [0.0, 0.2] | 6.0 [5.1, 7.2] | 90.3 | 0.0 [0.0, 0.2] | 0.6 [0.3, 1.1] | 0.1 | 0.9929 |
| EB shrinkage | 100.0 [99.8, 100.0] | 93.2 [92.1, 94.3] | 0.0 | 0.0 [0.0, 0.2] | 0.4 [0.2, 0.8] | 0.1 | -- |
