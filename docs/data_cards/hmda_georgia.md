# Data Card: HMDA Georgia

## Dataset Name and Source

Home Mortgage Disclosure Act (HMDA) mortgage application records for Georgia, obtained from the CFPB/FFIEC HMDA Data Browser API and stored locally for this artifact.

Relevant local files:

- `hmda_dataset/data/hmda_raw.csv`
- `hmda_dataset/download_hmda.py`
- `hmda_dataset/clean_hmda.py`
- `hmda_dataset/processed/data_report.md`

## Intended Use

This dataset is used as a real-world fair-lending benchmark for deterministic fairness auditing and OpenAI-based remediation-readiness evaluation. It supports testing whether an LLM audit remains grounded in fixed Python-computed metrics, handles multi-group protected attributes cautiously, and distinguishes statistical fragility from actionable disparity.

## Out-of-Scope Use

Do not use this artifact for real mortgage decisions, legal compliance certification, individual applicant assessment, lender ranking, or causal claims about discrimination. HMDA lacks many underwriting variables and does not by itself establish whether a given decision was lawful or unlawful.

## Collection and Provenance

HMDA data is reported by covered U.S. financial institutions under federal law and published by CFPB/FFIEC. The repository includes a downloader that queries the CFPB Data Browser CSV endpoint for a selected year, state, and action filter. The artifact scope is Georgia mortgage applications filtered to originated and denied applications.

## Preprocessing Performed in This Repo

`hmda_dataset/download_hmda.py`:

- Downloads HMDA records from the CFPB/FFIEC Data Browser API.
- Filters `action_taken` to originated (`1`) and denied (`3`) applications.
- Can sample to a requested maximum row count with seed `42`.

`hmda_dataset/clean_hmda.py`:

- Keeps selected target, demographic, and loan/application columns.
- Converts `action_taken` to binary `approved`, with `1` for originated and `0` for denied.
- Maps `derived_race` to `White`, `Black`, `Asian`, `American Indian`, `Pacific Islander`, and `Multiracial`; excludes joint/unavailable/free-text-only race rows.
- Keeps `derived_sex` values `Male` and `Female`.
- Maps applicant age buckets into `young`, `mid`, and `senior`.
- Converts debt-to-income ranges to numeric midpoint approximations.
- Coerces numeric loan fields and imputes/scales features through a scikit-learn preprocessing pipeline.
- Uses an 80/20 stratified train/test split with seed `42`.
- Preserves protected attributes in processed train/test files for downstream fairness analysis.

## Protected Attributes Used

- `race`, privileged value `White`.
- `sex`, privileged value `Male`.
- `age_group`, privileged value `mid`.

## Target Variable

`approved`, where `1` means loan originated/approved and `0` means application denied.

## Train/Test Split or Evaluation Subset

The current processed data report lists 10,978 rows after cleaning, 8,782 training rows, and 2,196 test rows. The fairness evaluation uses random forest predictions on the 2,196-row test split exported as `hmda_dataset/metrics/classification_predictions.csv` and copied under `artifacts/hmda/models/`.

## Known Limitations

- HMDA does not include every underwriting variable used by lenders.
- The artifact filters to originated and denied applications and excludes other action categories.
- Rare race categories have very small support, making subgroup metrics unstable.
- DTI range midpoint conversion is an approximation.
- The simplified `age_group` buckets are useful for benchmarking but do not capture all legal or demographic nuance.
- Existing downloader defaults in code may differ from the artifact's Georgia scope; use the documented artifact files and commands when reproducing this release.

## Ethical and Fairness Risks

HMDA contains sensitive demographic information. Audit outputs can identify disparities but should not be treated as proof of individual discrimination or as a substitute for legal/statistical fair-lending review. Particular caution is required for small racial subgroups, where selection-rate ratios can be volatile.

## Licensing and Access Notes

HMDA public data is distributed by CFPB/FFIEC under its public data access terms. The repository license covers code and documentation only. Users should follow CFPB/FFIEC terms and privacy guidance when redistributing or extending the data.

## Maintenance and Versioning Notes

The artifact records current processed outputs under `artifacts/hmda/`. If the HMDA year, state, action filters, preprocessing script, package versions, or random seeds change, regenerate raw/processed data, predictions, deterministic metrics, and OpenAI benchmark artifacts together.
