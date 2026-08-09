"""
train_models.py  (HMDA)

Train Logistic Regression and Random Forest classifiers to predict
mortgage approval (1 = originated, 0 = denied).

Exports predictions CSV with protected attribute columns intact so
the fairness pipeline can consume them directly.

Two prediction populations are supported:

  default            predictions on the held-out test split only (~2.2k rows)
  --full-population  out-of-fold predictions for EVERY cleaned row (~11k rows)

The full-population mode exists because auditing a test split is what produced
this project's small-cell problems: at test-split size the HMDA race x sex cells
for American Indian, Multiracial and Pacific Islander applicants hold 3-6 people
and are suppressed by the reporting floor, which silently removes the largest
measured disparities from the audit. Those same groups have 27-56 records in the
cleaned population. Predictions are generated with stratified 5-fold
cross-validation, so every row is scored by a model that never saw it — the
honest way to obtain a prediction for each applicant without leaking labels.

Usage (from hmda_dataset/):
  python3 scripts/train_models.py
  python3 scripts/train_models.py --full-population
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("processed")
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")
SEED = 42
TARGET = "approved"
N_FOLDS = 5

# age_62_plus is the ECOA cut (Reg B Sec. 1002.2(o)); age_group is the banded
# view retained for continuity. See clean_hmda.py for why both are audited.
PROTECTED_ATTRS = ["race", "sex", "age_group", "age_62_plus"]


def _score(y_true, y_pred, y_prob) -> dict:
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:  # noqa: BLE001
        auc = None
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc": round(auc, 4) if auc is not None else None,
    }


def _build_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            random_state=SEED, max_iter=1000, class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=SEED, n_jobs=-1, class_weight="balanced",
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--full-population",
        action="store_true",
        help="write out-of-fold predictions for every cleaned row instead of the test split",
    )
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(PROCESSED_DIR / "X_train_processed.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_processed.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")[TARGET]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")[TARGET]

    with (PROCESSED_DIR / "feature_map.json").open() as f:
        feature_names: list[str] = json.load(f)

    log.info(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_names)}")

    present_attrs = [a for a in PROTECTED_ATTRS if a in X_test.columns]
    if len(present_attrs) < len(PROTECTED_ATTRS):
        log.warning(f"missing protected attrs: {set(PROTECTED_ATTRS) - set(present_attrs)}")

    X_train_model = X_train[feature_names].copy()
    X_test_model = X_test[feature_names].copy()

    models = _build_models()

    # ── fit on train, score on test (unchanged reporting path) ─────────
    for name, model in models.items():
        log.info(f"Training {name}…")
        model.fit(X_train_model, y_train)
        y_pred = model.predict(X_test_model)
        y_prob = model.predict_proba(X_test_model)[:, 1]

        metrics = _score(y_test, y_pred, y_prob)
        log.info(f"  {name} → {metrics}")

        with (METRICS_DIR / f"{name}_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)
        joblib.dump(model, MODELS_DIR / f"{name}_model.joblib")

        preds = pd.DataFrame(
            {"actual": y_test.values, "predicted": y_pred, "predicted_proba": y_prob}
        )
        for attr in present_attrs:
            preds[attr] = X_test[attr].values
        preds.to_csv(METRICS_DIR / f"{name}_predictions.csv", index=False)

    # ── canonical predictions ──────────────────────────────────────────
    best_name = "random_forest"
    test_canon = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": models[best_name].predict(X_test_model),
            "predicted_proba": models[best_name].predict_proba(X_test_model)[:, 1],
        }
    )
    for attr in present_attrs:
        test_canon[attr] = X_test[attr].values

    # The test-split file is always written, so the population-vs-split
    # comparison stays reproducible even in full-population mode.
    test_canon.to_csv(METRICS_DIR / "classification_predictions_testsplit.csv", index=False)

    if not args.full_population:
        test_canon.to_csv(METRICS_DIR / "classification_predictions.csv", index=False)
        log.info(f"Canonical predictions: test split, n={len(test_canon)}")
        log.info("Training complete. Artifacts saved.")
        return

    # ── full population via stratified K-fold out-of-fold prediction ───
    X_all = pd.concat([X_train, X_test], ignore_index=True)
    y_all = pd.concat([y_train, y_test], ignore_index=True)
    X_all_model = X_all[feature_names].copy()

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    log.info(
        f"Full-population mode: {len(X_all)} rows, {N_FOLDS}-fold out-of-fold predictions"
    )

    oof_metrics = {}
    for name in models:
        estimator = clone(_build_models()[name])
        oof_pred = cross_val_predict(estimator, X_all_model, y_all, cv=cv, n_jobs=-1)
        oof_prob = cross_val_predict(
            clone(_build_models()[name]), X_all_model, y_all, cv=cv,
            method="predict_proba", n_jobs=-1,
        )[:, 1]
        oof_metrics[name] = _score(y_all, oof_pred, oof_prob)
        log.info(f"  {name} (out-of-fold) → {oof_metrics[name]}")

        if name == best_name:
            canon = pd.DataFrame(
                {"actual": y_all.values, "predicted": oof_pred, "predicted_proba": oof_prob}
            )
            for attr in present_attrs:
                canon[attr] = X_all[attr].values
            canon.to_csv(METRICS_DIR / "classification_predictions.csv", index=False)

    with (METRICS_DIR / "out_of_fold_metrics.json").open("w") as f:
        json.dump(
            {
                "n_rows": int(len(X_all)),
                "n_folds": N_FOLDS,
                "seed": SEED,
                "note": (
                    "Out-of-fold predictions over the full cleaned population. Every "
                    "row is scored by a model fitted without it. Compare against the "
                    "*_metrics.json files, which are single train/test-split scores."
                ),
                "models": oof_metrics,
            },
            f,
            indent=2,
        )

    log.info(f"Canonical predictions: FULL POPULATION, n={len(X_all)}")
    log.info("Training complete. Artifacts saved.")


if __name__ == "__main__":
    main()
