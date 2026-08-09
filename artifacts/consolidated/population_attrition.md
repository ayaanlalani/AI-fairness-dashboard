# HMDA population construction (attrition)

Raw file: `hmda_dataset/data/hmda_raw.csv`. Steps replayed from `hmda_dataset/clean_hmda.py::load_and_clean` in order.

| step | rule | dropped | remaining |
|---|---|---|---|
| raw file | - | 0 | 20000 |
| race | derived_race not in RACE_MAP or mapped to None | 4423 | 15577 |
| sex | derived_sex not in {Male, Female} | 4598 | 10979 |
| age | applicant_age not a mapped band | 1 | 10978 |

Dropped by category:

- race: Race Not Available — 4140
- race: Joint — 280
- race: Free Form Text Only — 3
- sex: Joint — 4529
- sex: Sex Not Available — 69
- age: 8888 — 1

## Race DI on raw approval rates, dropped categories retained

| race | in pipeline | n | approval rate | DI vs White | DI vs best |
|---|---|---|---|---|---|
| Joint | DROPPED | 280 | 0.8036 | 1.0225 | 1.0 |
| Asian | yes | 1063 | 0.7959 | 1.0127 | 0.9904 |
| White | yes | 9822 | 0.7859 | 1.0 | 0.978 |
| Race Not Available | DROPPED | 4140 | 0.7234 | 0.9205 | 0.9003 |
| Black | yes | 4541 | 0.641 | 0.8157 | 0.7977 |
| Pacific Islander | yes | 32 | 0.5312 | 0.676 | 0.6611 |
| American Indian | yes | 63 | 0.5238 | 0.6665 | 0.6519 |
| Multiracial | yes | 56 | 0.5 | 0.6362 | 0.6222 |
| Free Form Text Only | DROPPED | 3 | 0.0 | 0.0 | 0.0 |
