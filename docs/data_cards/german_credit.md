# Data Card: German Credit

## Dataset Name and Source

German Credit, based on the UCI Statlog German Credit dataset and a local Kaggle-formatted CSV used in this repository for human-readable feature names.

Relevant local files:

- `german_credit_dataset/data/german_credit_kaggle.csv`
- `german_credit_dataset/data/german_credit_CLEANED_dataset.csv`
- `german_credit_dataset/clean-germanCreditData.py`

## Intended Use

This dataset is used as a small lending-domain benchmark for deterministic fairness auditing and OpenAI-based qualitative fairness analysis. The artifact uses it to test whether audit narratives remain grounded in fixed metric evidence and whether subgroup-size warnings are surfaced clearly.

## Out-of-Scope Use

Do not use this dataset or the trained models in this repository for real credit decisions, credit scoring, legal compliance determinations, or individual-level eligibility decisions. The dataset is small, historically dated, and not representative of current lending populations.

## Collection and Provenance

The UCI Statlog German Credit dataset contains 1,000 historical credit records with coded applicant and loan attributes. The repository combines a human-readable Kaggle-style CSV with UCI fields fetched by `ucimlrepo` to recover target and selected original encodings.

## Preprocessing Performed in This Repo

`german_credit_dataset/clean-germanCreditData.py`:

- Loads `german_credit_kaggle.csv`.
- Fetches UCI Statlog German Credit metadata/data with `ucimlrepo`.
- Fills missing `Checking account` and `Saving accounts` values with `Unknown`.
- Derives `AgeGroup` as `40_plus` when `Age >= 40`, otherwise `under_40`.
- Maps UCI credit history codes to a numeric `credit reliability` feature.
- Maps UCI foreign-worker codes `A201`/`A202` to binary `foreign_worker`.
- Writes `german_credit_CLEANED_dataset.csv`.

`german_credit_dataset/scripts/train_models.py`:

- Label-encodes categorical features.
- Standardizes numeric features.
- Uses an 80/20 stratified train/test split with seed `42`.
- Trains logistic regression and random forest models.
- Uses random forest predictions as `metrics/classification_predictions.csv`.

## Protected Attributes Used

- `Sex_original`, privileged value `male`.
- `AgeGroup_original`, privileged value `40_plus`.
- `foreign_worker_original`, privileged value `0` (non-foreign worker).

## Target Variable

`credit risk`, where `1` is treated as the favorable label (`good`) and `2` as unfavorable (`bad`).

## Train/Test Split or Evaluation Subset

The model script uses an 80/20 stratified split. The current fairness evaluation uses the 200-row test split exported in `german_credit_dataset/metrics/classification_predictions.csv` and copied under `artifacts/german_credit/models/`.

## Known Limitations

- Only 1,000 total records and 200 test records.
- The `foreign_worker` subgroup is extremely small in the current test split, so metrics for that attribute are statistically fragile.
- The dataset is historical and may not reflect present-day credit markets.
- Some features can act as direct or indirect proxies for protected attributes.
- Protected groups are simplified binary comparisons for the artifact protocol.

## Ethical and Fairness Risks

The dataset encodes historical lending outcomes and applicant attributes that may reflect past discrimination. Audit results should be used to study fairness methodology, not to justify real lending actions. Small-subgroup findings can overstate or understate harm if interpreted without uncertainty and support-count checks.

## Licensing and Access Notes

The code and documentation in this repository are covered by the repository license. German Credit data remains subject to the original UCI/Kaggle source terms. Users should verify source access and citation requirements before redistributing the raw dataset.

## Maintenance and Versioning Notes

The artifact records current processed outputs under `artifacts/german_credit/`. If source files, preprocessing scripts, package versions, or random seeds change, regenerate the cleaned data, model predictions, deterministic metrics, and OpenAI benchmark artifacts together.
