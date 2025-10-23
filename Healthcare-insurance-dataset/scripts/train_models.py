"""
train_models.py

Train and evaluate models for the cleaned Healthcare Cost Prediction dataset.

- Regression: Predict continuous `charges` using LinearRegression
- Classification: Predict binary `high_cost` (if available) using RandomForestClassifier

Example run:

python scripts/train_models.py \
  --data_dir data/processed \
  --out_dir metrics \
  --models_dir models \
  --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

import matplotlib
matplotlib.use("Agg")  # For headless environments
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# Logging
# -------------------------

def get_logger() -> logging.Logger:
    logger = logging.getLogger("train_models")
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


# -------------------------
# Data loading
# -------------------------

def load_processed_data(data_dir: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    """Load processed train/test features and labels.

    Returns: X_train, X_test, y_train, y_test, high_cost_train, high_cost_test
    """
    logger.info(f"Loading processed data from: {data_dir}")

    x_train_path = data_dir / "X_train_processed.csv"
    x_test_path = data_dir / "X_test_processed.csv"
    y_train_path = data_dir / "y_train.csv"
    y_test_path = data_dir / "y_test.csv"
    hc_train_path = data_dir / "high_cost_train.csv"
    hc_test_path = data_dir / "high_cost_test.csv"

    for p in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path)["charges"]
    y_test = pd.read_csv(y_test_path)["charges"]

    # Optional classification labels
    high_cost_train: Optional[pd.Series] = None
    high_cost_test: Optional[pd.Series] = None
    if hc_train_path.exists() and hc_test_path.exists():
        high_cost_train = pd.read_csv(hc_train_path)["high_cost"]
        high_cost_test = pd.read_csv(hc_test_path)["high_cost"]
        logger.info("Found high_cost labels for classification.")
    else:
        logger.warning("high_cost label not found — skipping classification.")

    # Validate shapes
    assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch between train and test"
    assert list(X_train.columns) == list(X_test.columns), "Feature columns mismatch order or names"
    assert len(X_train) == len(y_train), "X_train and y_train length mismatch"
    assert len(X_test) == len(y_test), "X_test and y_test length mismatch"
    if high_cost_train is not None and high_cost_test is not None:
        assert len(X_train) == len(high_cost_train), "X_train and high_cost_train length mismatch"
        assert len(X_test) == len(high_cost_test), "X_test and high_cost_test length mismatch"

    logger.info(
        f"Dataset summary: X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )

    return X_train, X_test, y_train, y_test, high_cost_train, high_cost_test


# -------------------------
# Regression training
# -------------------------

def train_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_dir: Path,
    out_dir: Path,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Train LinearRegression on continuous charges and export metrics and predictions."""
    logger.info("Training regression model (LinearRegression)...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    # Compute RMSE in a version-compatible way (older sklearn may not support squared=False)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    metrics_dict = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    reg_model_path = models_dir / "regression_model.joblib"
    joblib.dump(model, reg_model_path)

    # Save metrics and predictions
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "regression_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    preds_path = out_dir / "regression_predictions.csv"
    pd.DataFrame({"actual": y_test, "predicted": y_pred}).to_csv(preds_path, index=False)

    # Plot: Actual vs Predicted
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, s=20, alpha=0.7)
    min_val = float(min(np.min(y_test), np.min(y_pred)))
    max_val = float(max(np.max(y_test), np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1)
    plt.xlabel("Actual charges")
    plt.ylabel("Predicted charges")
    plt.title("Regression: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(plots_dir / "regression_fit.png", dpi=150)
    plt.close()

    logger.info(f"Regression metrics: R2={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
    return metrics_dict


# -------------------------
# Classification training
# -------------------------

def train_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_dir: Path,
    out_dir: Path,
    seed: int,
    n_estimators: int,
    logger: logging.Logger,
) -> Optional[Dict[str, float]]:
    """Train RandomForestClassifier on high_cost and export metrics and predictions."""
    if y_train is None or y_test is None:
        logger.warning("high_cost label not found — skipping classification.")
        return None

    logger.info("Training classification model (RandomForestClassifier)...")
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics_dict = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
    }

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    clf_model_path = models_dir / "classification_model.joblib"
    joblib.dump(clf, clf_model_path)

    # Save metrics and predictions
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "classification_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    preds_path = out_dir / "classification_predictions.csv"
    pd.DataFrame({"actual": y_test, "predicted": y_pred}).to_csv(preds_path, index=False)

    # Plot: Confusion Matrix
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Classification: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    logger.info(
        f"Classification metrics: F1={f1:.4f}, Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}"
    )
    return metrics_dict


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate regression/classification models")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with processed CSVs")
    parser.add_argument("--out_dir", type=str, default="metrics", help="Output directory for metrics and predictions")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save trained models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_estimators", type=int, default=200, help="Number of trees for RandomForestClassifier")

    args = parser.parse_args()

    logger = get_logger()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    models_dir = Path(args.models_dir)

    np.random.seed(args.seed)

    X_train, X_test, y_train, y_test, high_cost_train, high_cost_test = load_processed_data(data_dir, logger)

    # Regression
    reg_metrics = train_regression(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        models_dir=models_dir,
        out_dir=out_dir,
        logger=logger,
    )

    # Classification (optional)
    cls_metrics: Optional[Dict[str, float]] = None
    if high_cost_train is not None and high_cost_test is not None:
        cls_metrics = train_classification(
            X_train=X_train,
            y_train=high_cost_train,
            X_test=X_test,
            y_test=high_cost_test,
            models_dir=models_dir,
            out_dir=out_dir,
            seed=args.seed,
            n_estimators=args.n_estimators,
            logger=logger,
        )

    # Console summary
    reg_summary = (
        f"✅ Regression R²: {reg_metrics['R2']:.4f} | "
        f"RMSE: {reg_metrics['RMSE']:.2f} | MAE: {reg_metrics['MAE']:.2f}"
    )
    logger.info(reg_summary)
    if cls_metrics is not None:
        cls_summary = (
            f"✅ Classification F1-score: {cls_metrics['F1']:.4f} | "
            f"Accuracy: {cls_metrics['Accuracy']:.4f} | "
            f"Precision: {cls_metrics['Precision']:.4f} | Recall: {cls_metrics['Recall']:.4f}"
        )
        logger.info(cls_summary)
    else:
        logger.info("Classification skipped (no high_cost labels)")

    logger.info(f"📊 Metrics saved under {out_dir}/")
    logger.info(f"💾 Models saved under {models_dir}/")


if __name__ == "__main__":
    main()
