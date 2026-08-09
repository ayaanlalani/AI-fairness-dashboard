"""
train_models.py  (German Credit)

Train Logistic Regression and Random Forest classifiers on the cleaned
German Credit dataset and export predictions + metrics.

Target:  credit risk  (1 = good, 2 = bad)
Protected attributes preserved in predictions CSV:
  Sex, AgeGroup, foreign_worker

Usage (from german_credit_dataset/):
  python scripts/train_models.py
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────
DATA_PATH = Path("data/german_credit_CLEANED_dataset.csv")
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")
SEED = 42

# ── column definitions ─────────────────────────────────────────────────
TARGET = "credit risk"

PROTECTED_ATTRS = ["Sex", "AgeGroup", "foreign_worker"]

CATEGORICAL_FEATURES = [
    "Sex", "Job", "Housing", "Saving accounts",
    "Checking account", "Purpose", "AgeGroup",
]
NUMERICAL_FEATURES = ["Age", "Credit amount", "Duration", "credit reliability"]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + ["foreign_worker"]


N_FOLDS = 5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--full-population",
        action="store_true",
        help="write out-of-fold predictions for all rows instead of the test split",
    )
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load ───────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    log.info(f"Loaded {len(df)} rows from {DATA_PATH}")

    # ── encode categoricals ────────────────────────────────────────────
    label_encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    # ── train / test split (stratified) ────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y,
    )
    log.info(f"Split: train={len(X_train)}, test={len(X_test)}")

    # ── scale numerical features ───────────────────────────────────────
    scaler = StandardScaler()
    X_train[NUMERICAL_FEATURES] = scaler.fit_transform(X_train[NUMERICAL_FEATURES])
    X_test[NUMERICAL_FEATURES] = scaler.transform(X_test[NUMERICAL_FEATURES])

    # ── train models ───────────────────────────────────────────────────
    models = {
        "logistic_regression": LogisticRegression(
            random_state=SEED, max_iter=1000,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=SEED, n_jobs=-1,
        ),
    }

    for name, model in models.items():
        log.info(f"Training {name}…")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # ── metrics ────────────────────────────────────────────────────
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        # The target is encoded 1 = good, 2 = bad, so roc_auc_score treats 2 as
        # the positive class. Scoring column 0 (P(good)) against that inverts the
        # curve: it is why the committed metrics carried AUC 0.2013 and 0.2756,
        # both below chance and both exactly 1 - the true value. Column 1 is
        # P(bad), which is what the positive label refers to.
        try:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        except Exception:
            auc = None

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4) if auc is not None else None,
        }
        log.info(f"  {name} → {metrics}")

        with (METRICS_DIR / f"{name}_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)

        # ── save model ─────────────────────────────────────────────────
        joblib.dump(model, MODELS_DIR / f"{name}_model.joblib")

        # ── predictions CSV (includes protected attrs for fairness) ────
        preds_df = X_test.copy()
        preds_df["actual"] = y_test.values
        preds_df["predicted"] = y_pred

        # Decode protected attributes back to original labels
        for attr in PROTECTED_ATTRS:
            if attr in label_encoders:
                preds_df[f"{attr}_original"] = label_encoders[attr].inverse_transform(
                    preds_df[attr].astype(int)
                )
            else:
                preds_df[f"{attr}_original"] = preds_df[attr]

        preds_df.to_csv(
            METRICS_DIR / f"{name}_predictions.csv", index=False,
        )

    # ── also export a canonical classification_predictions.csv ─────────
    # Uses the best model (random_forest) for the fairness pipeline
    def _decode(frame: pd.DataFrame) -> pd.DataFrame:
        for attr in PROTECTED_ATTRS:
            if attr in label_encoders:
                frame[f"{attr}_original"] = label_encoders[attr].inverse_transform(
                    frame[attr].astype(int)
                )
            else:
                frame[f"{attr}_original"] = frame[attr]
        return frame

    best = models["random_forest"]
    canon = X_test.copy()
    canon["actual"] = y_test.values
    canon["predicted"] = best.predict(X_test)
    canon["predicted_proba"] = best.predict_proba(X_test)[:, 1]
    canon = _decode(canon)

    # Always written, so the population-vs-split comparison stays reproducible.
    canon.to_csv(METRICS_DIR / "classification_predictions_testsplit.csv", index=False)

    if not args.full_population:
        canon.to_csv(METRICS_DIR / "classification_predictions.csv", index=False)
        log.info(f"Canonical predictions: test split, n={len(canon)}")
    else:
        # ── full population via stratified K-fold out-of-fold prediction ──
        # A 200-row test split leaves AgeGroup x Sex cells as small as n=15,
        # which is where this project's severity grades became unstable. Scoring
        # every row out-of-fold keeps the labels honest while giving the audit
        # the whole population.
        X_all = X.copy()
        X_all[NUMERICAL_FEATURES] = StandardScaler().fit_transform(
            X_all[NUMERICAL_FEATURES]
        )
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        log.info(
            f"Full-population mode: {len(X_all)} rows, {N_FOLDS}-fold out-of-fold predictions"
        )

        oof_metrics = {}
        for name in models:
            est = clone(models[name])
            oof_pred = cross_val_predict(est, X_all, y, cv=cv, n_jobs=-1)
            oof_prob = cross_val_predict(
                clone(models[name]), X_all, y, cv=cv,
                method="predict_proba", n_jobs=-1,
            )[:, 1]
            try:
                oof_auc = roc_auc_score(y, oof_prob)
            except Exception:  # noqa: BLE001
                oof_auc = None
            oof_metrics[name] = {
                "accuracy": round(accuracy_score(y, oof_pred), 4),
                "precision": round(precision_score(y, oof_pred, pos_label=1, zero_division=0), 4),
                "recall": round(recall_score(y, oof_pred, pos_label=1, zero_division=0), 4),
                "f1": round(f1_score(y, oof_pred, pos_label=1, zero_division=0), 4),
                "auc": round(oof_auc, 4) if oof_auc is not None else None,
            }
            log.info(f"  {name} (out-of-fold) → {oof_metrics[name]}")

            if name == "random_forest":
                full = X_all.copy()
                full["actual"] = y.values
                full["predicted"] = oof_pred
                full["predicted_proba"] = oof_prob
                full = _decode(full)
                full.to_csv(METRICS_DIR / "classification_predictions.csv", index=False)

        # The target is 1 = good / 2 = bad, not 0/1, so y.mean() is not a rate.
        majority = float(y.value_counts(normalize=True).max())
        with (METRICS_DIR / "out_of_fold_metrics.json").open("w") as f:
            json.dump(
                {
                    "n_rows": int(len(X_all)),
                    "n_folds": N_FOLDS,
                    "seed": SEED,
                    "majority_class_rate": round(majority, 4),
                    "note": (
                        "Out-of-fold predictions over all rows. Compare accuracy "
                        "against majority_class_rate: a model below that line is "
                        "worse than predicting one class for everyone."
                    ),
                    "models": oof_metrics,
                },
                f,
                indent=2,
            )
        log.info(
            f"Canonical predictions: FULL POPULATION, n={len(X_all)} "
            f"(majority-class rate {majority:.4f})"
        )

    # Save scaler + encoders for reproducibility
    joblib.dump(
        {"scaler": scaler, "label_encoders": label_encoders},
        MODELS_DIR / "preprocessing_artifacts.joblib",
    )

    log.info("Training complete. Artifacts saved to models/ and metrics/.")


if __name__ == "__main__":
    main()
