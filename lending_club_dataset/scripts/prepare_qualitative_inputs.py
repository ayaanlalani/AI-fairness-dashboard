"""
prepare_qualitative_inputs.py

Bridge Lending Club outputs into the shared qualitative/benchmark tooling.

The shared scripts (scripts/qualitative_analysis.py, scripts/llm_benchmark.py)
expect (a) a predictions CSV that contains decoded protected-attribute columns
and (b) a fairness CSV in the German Credit column schema. Lending Club stores
protected attributes one-hot encoded in processed/X_test_processed.csv and
writes a lowercase fairness schema, so this script produces:

  metrics/classification_predictions_enriched.csv
      predictions + decoded gender / income_level / loan_amount_level
      + non-protected feature columns (for proxy detection)

  metrics/fairness/fairness_metrics_normalized.csv
      fairness metrics renamed to Attribute / PrivilegedValue /
      DisparateImpact / DemographicParityDiff / EqualOpportunityDiff /
      AverageOddsDiff / TheilIndex / BiasFlag (TheilIndex left empty —
      this dataset's compute_fairness.py does not produce it)

Decoding rules mirror scripts/compute_fairness.py exactly.

Example run (from lending_club_dataset/):

python scripts/prepare_qualitative_inputs.py \
  --data_dir processed \
  --metrics_dir metrics
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROTECTED_ONEHOT_COLS = [
    "gender_male",
    "income_level_low",
    "income_level_medium",
    "loan_amount_level_small",
    "loan_amount_level_medium",
]

FAIRNESS_COLUMN_MAP = {
    "protected_attribute": "Attribute",
    "privileged_group": "PrivilegedValue",
    "disparate_impact": "DisparateImpact",
    "demographic_parity_difference": "DemographicParityDiff",
    "equal_opportunity_difference": "EqualOpportunityDiff",
    "average_odds_difference": "AverageOddsDiff",
    "bias_detected": "BiasFlag",
}


def decode_protected_attributes(X: pd.DataFrame) -> pd.DataFrame:
    """Decode one-hot protected columns back to categorical labels."""
    out = pd.DataFrame(index=X.index)

    if "gender_male" in X.columns:
        out["gender"] = np.where(X["gender_male"].astype(float) > 0.5, "male", "female")
    else:
        logger.warning("gender_male column not found; skipping 'gender'")

    if "income_level_low" in X.columns and "income_level_medium" in X.columns:
        out["income_level"] = np.select(
            [X["income_level_low"].astype(float) == 1, X["income_level_medium"].astype(float) == 1],
            ["low", "medium"],
            default="high",
        )
    else:
        logger.warning("income_level_* columns not found; skipping 'income_level'")

    if "loan_amount_level_small" in X.columns and "loan_amount_level_medium" in X.columns:
        out["loan_amount_level"] = np.select(
            [
                X["loan_amount_level_small"].astype(float) == 1,
                X["loan_amount_level_medium"].astype(float) == 1,
            ],
            ["small", "medium"],
            default="large",
        )
    else:
        logger.warning("loan_amount_level_* columns not found; skipping 'loan_amount_level'")

    return out


def enrich_predictions(data_dir: Path, metrics_dir: Path) -> Path:
    X_test = pd.read_csv(data_dir / "X_test_processed.csv")
    preds = pd.read_csv(metrics_dir / "classification_predictions.csv")

    if len(X_test) != len(preds):
        raise ValueError(
            f"Row count mismatch: X_test_processed.csv has {len(X_test)} rows, "
            f"classification_predictions.csv has {len(preds)} — cannot align."
        )

    decoded = decode_protected_attributes(X_test)
    features = X_test.drop(columns=[c for c in PROTECTED_ONEHOT_COLS if c in X_test.columns])
    enriched = pd.concat(
        [features.reset_index(drop=True), decoded.reset_index(drop=True), preds.reset_index(drop=True)],
        axis=1,
    )

    out_path = metrics_dir / "classification_predictions_enriched.csv"
    enriched.to_csv(out_path, index=False)
    logger.info(f"Enriched predictions ({len(enriched)} rows) -> {out_path}")
    return out_path


def normalize_fairness_csv(metrics_dir: Path) -> Path:
    src = metrics_dir / "fairness" / "fairness_metrics.csv"
    df = pd.read_csv(src)

    if "Attribute" in df.columns:
        logger.info(f"{src} already uses the shared schema; copying through")
        normalized = df
    else:
        normalized = df.rename(columns=FAIRNESS_COLUMN_MAP)
        if "TheilIndex" not in normalized.columns:
            normalized["TheilIndex"] = np.nan
        ordered = [
            "Attribute", "PrivilegedValue", "DisparateImpact", "DemographicParityDiff",
            "EqualOpportunityDiff", "AverageOddsDiff", "TheilIndex", "BiasFlag",
        ]
        normalized = normalized[[c for c in ordered if c in normalized.columns]]

    out_path = metrics_dir / "fairness" / "fairness_metrics_normalized.csv"
    normalized.to_csv(out_path, index=False)
    logger.info(f"Normalized fairness metrics -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Lending Club inputs for the shared qualitative/benchmark scripts"
    )
    parser.add_argument("--data_dir", type=str, default="processed", help="Directory with processed CSVs")
    parser.add_argument("--metrics_dir", type=str, default="metrics", help="Directory with model predictions")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    metrics_dir = Path(args.metrics_dir)

    enrich_predictions(data_dir, metrics_dir)
    normalize_fairness_csv(metrics_dir)


if __name__ == "__main__":
    main()
