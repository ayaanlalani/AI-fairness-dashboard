"""
train_models.py

Train and evaluate a classifier for the Diabetes dataset to predict binary `Outcome`.

Outputs (under diabetes_dataset/ by default):
- models/classification_model.joblib
- metrics/classification_metrics.json
- metrics/classification_predictions.csv (includes protected attributes for fairness)
- metrics/plots/confusion_matrix.png

Example:

python scripts/train_models.py \
  --data_path data/diabetes_cleaned.csv \
  --out_dir metrics \
  --models_dir models \
  --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns


def get_logger() -> logging.Logger:
    logger = logging.getLogger("diabetes_train")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classifier for Diabetes Outcome")
    parser.add_argument("--data_path", type=str, default="data/diabetes_cleaned.csv", help="Path to cleaned diabetes CSV")
    parser.add_argument("--out_dir", type=str, default="metrics", help="Directory to write metrics and predictions")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save trained models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_estimators", type=int, default=300, help="Trees for RandomForestClassifier")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")

    args = parser.parse_args()
    logger = get_logger()

    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    models_dir = Path(args.models_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {data_path}")

    df = pd.read_csv(data_path)
    if "Outcome" not in df.columns:
        raise ValueError("Expected 'Outcome' column in cleaned dataset")

    # Features: use all columns except target
    feature_cols = [c for c in df.columns if c != "Outcome"]
    X = df[feature_cols].copy()
    y = df["Outcome"].astype(int)

    # Train/test split (stratified)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Train classifier
    logger.info("Training RandomForestClassifier on Outcome ...")
    clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics_dict: Dict[str, float] = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
    }

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "classification_model.joblib"
    joblib.dump(clf, model_path)

    # Save metrics and predictions
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "classification_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    # Include protected attributes in predictions for downstream fairness
    prot_cols = [c for c in ["older_age", "high_pregnancy_count"] if c in X_test.columns]
    preds_df = pd.DataFrame({"actual": y_test.reset_index(drop=True), "predicted": pd.Series(y_pred)})
    for c in prot_cols:
        preds_df[c] = X_test.reset_index(drop=True)[c].astype(int)

    # Persist predictions
    preds_path = out_dir / "classification_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    # Confusion matrix plot
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Diabetes Outcome: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    logger.info(
        f"Classification metrics: F1={metrics_dict['F1']:.4f} | Accuracy={metrics_dict['Accuracy']:.4f} | "
        f"Precision={metrics_dict['Precision']:.4f} | Recall={metrics_dict['Recall']:.4f}"
    )
    logger.info(f"📊 Metrics saved under {out_dir}/")
    logger.info(f"💾 Model saved to {model_path}")


if __name__ == "__main__":
    main()


