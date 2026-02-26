"""
train_models.py  (HMDA)

Train Logistic Regression and Random Forest classifiers to predict
mortgage approval (1 = originated, 0 = denied).

Exports predictions CSV with protected attribute columns intact so
the fairness pipeline can consume them directly.

Usage (from hmda_dataset/):
  python3 scripts/train_models.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

PROTECTED_ATTRS = ["race", "sex", "age_group"]


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load processed data ────────────────────────────────────────
    X_train = pd.read_csv(PROCESSED_DIR / "X_train_processed.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_processed.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")[TARGET]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")[TARGET]

    with (PROCESSED_DIR / "feature_map.json").open() as f:
        feature_names: list[str] = json.load(f)

    log.info(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_names)}")

    # Separate protected attrs from model features
    prot_train = X_train[PROTECTED_ATTRS].copy() if all(a in X_train.columns for a in PROTECTED_ATTRS) else pd.DataFrame()
    prot_test = X_test[PROTECTED_ATTRS].copy() if all(a in X_test.columns for a in PROTECTED_ATTRS) else pd.DataFrame()

    X_train_model = X_train[feature_names].copy()
    X_test_model = X_test[feature_names].copy()

    # ── train models ───────────────────────────────────────────────
    models = {
        "logistic_regression": LogisticRegression(
            random_state=SEED, max_iter=1000, class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=SEED, n_jobs=-1, class_weight="balanced",
        ),
    }

    for name, model in models.items():
        log.info(f"Training {name}…")
        model.fit(X_train_model, y_train)
        y_pred = model.predict(X_test_model)
        y_prob = model.predict_proba(X_test_model)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = None

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4) if auc else None,
        }
        log.info(f"  {name} → {metrics}")

        with (METRICS_DIR / f"{name}_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)
        joblib.dump(model, MODELS_DIR / f"{name}_model.joblib")

        # Predictions CSV
        preds = pd.DataFrame({"actual": y_test.values, "predicted": y_pred})
        for attr in PROTECTED_ATTRS:
            if attr in prot_test.columns:
                preds[attr] = prot_test[attr].values
        preds.to_csv(METRICS_DIR / f"{name}_predictions.csv", index=False)

    # ── canonical predictions (use random_forest for fairness) ─────
    best = models["random_forest"]
    y_pred_best = best.predict(X_test_model)
    canon = pd.DataFrame({"actual": y_test.values, "predicted": y_pred_best})
    for attr in PROTECTED_ATTRS:
        if attr in prot_test.columns:
            canon[attr] = prot_test[attr].values
    canon.to_csv(METRICS_DIR / "classification_predictions.csv", index=False)

    log.info("Training complete. Artifacts saved.")


if __name__ == "__main__":
    main()
