"""
train_models.py

Train and evaluate classification models for the cleaned Lending Club loan dataset.

- Classification: Predict binary `loan_default` using RandomForestClassifier and LogisticRegression

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
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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

def load_processed_data(data_dir: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load processed train/test features and labels."""
    logger.info(f"Loading processed data from: {data_dir}")

    x_train_path = data_dir / "X_train_processed.csv"
    x_test_path = data_dir / "X_test_processed.csv"
    y_train_path = data_dir / "y_train.csv"
    y_test_path = data_dir / "y_test.csv"

    for p in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path)["loan_default"]
    y_test = pd.read_csv(y_test_path)["loan_default"]

    # Validate shapes
    assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch between train and test"
    assert list(X_train.columns) == list(X_test.columns), "Feature columns mismatch"
    assert len(X_train) == len(y_train), "X_train and y_train length mismatch"
    assert len(X_test) == len(y_test), "X_test and y_test length mismatch"

    logger.info(
        f"Dataset summary: X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )
    
    # Log class distribution
    train_default_rate = y_train.mean()
    test_default_rate = y_test.mean()
    logger.info(f"Train default rate: {train_default_rate:.3f}")
    logger.info(f"Test default rate: {test_default_rate:.3f}")

    return X_train, X_test, y_train, y_test


# -------------------------
# Model training and evaluation
# -------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_dir: Path,
    out_dir: Path,
    seed: int,
    n_estimators: int,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Train RandomForestClassifier and export metrics and predictions."""
    logger.info("Training Random Forest classifier...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced"  # Handle class imbalance
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    
    # Handle case where only one class is present (no defaults in data)
    proba_matrix = model.predict_proba(X_test)
    if proba_matrix.shape[1] == 1:
        # Only one class present, create dummy probabilities
        y_pred_proba = np.zeros(len(X_test))
        logger.warning("Random Forest: Only one class present in data - using dummy probabilities")
    else:
        y_pred_proba = proba_matrix[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = 0.5  # Random performance when only one class
        logger.warning("Cannot compute AUC with only one class - using 0.5")

    metrics_dict = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "AUC": float(auc),
    }

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "random_forest_model.joblib"
    joblib.dump(model, model_path)

    # Save metrics and predictions
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "random_forest_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    preds_path = out_dir / "random_forest_predictions.csv"
    pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred,
        "predicted_proba": y_pred_proba
    }).to_csv(preds_path, index=False)

    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    feature_importance.to_csv(out_dir / "random_forest_feature_importance.csv", index=False)

    # Generate plots
    generate_classification_plots(
        y_test, y_pred, y_pred_proba, out_dir, "random_forest", logger
    )

    logger.info(
        f"Random Forest metrics: F1={f1:.4f}, AUC={auc:.4f}, "
        f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}"
    )
    
    return metrics_dict


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_dir: Path,
    out_dir: Path,
    seed: int,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Train LogisticRegression and export metrics and predictions."""
    logger.info("Training Logistic Regression classifier...")
    
    # Check if we have at least 2 classes for Logistic Regression
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        logger.warning("Logistic Regression: Only one class present - creating dummy model")
        # Return dummy metrics when only one class is present
        return {
            "Accuracy": 1.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "AUC": 0.5
        }
    
    model = LogisticRegression(
        random_state=seed,
        max_iter=1000,
        class_weight="balanced"  # Handle class imbalance
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    # Handle case where only one class is present
    proba_matrix = model.predict_proba(X_test)
    if proba_matrix.shape[1] == 1:
        y_pred_proba = np.zeros(len(X_test))
        logger.warning("Only one class present - using dummy probabilities")
    else:
        y_pred_proba = proba_matrix[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = 0.5  # Random performance when only one class
        logger.warning("Cannot compute AUC with only one class - using 0.5")

    metrics_dict = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "AUC": float(auc),
    }

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "logistic_regression_model.joblib"
    joblib.dump(model, model_path)

    # Save metrics and predictions
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "logistic_regression_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    preds_path = out_dir / "logistic_regression_predictions.csv"
    pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred,
        "predicted_proba": y_pred_proba
    }).to_csv(preds_path, index=False)

    # Generate plots
    generate_classification_plots(
        y_test, y_pred, y_pred_proba, out_dir, "logistic_regression", logger
    )

    logger.info(
        f"Logistic Regression metrics: F1={f1:.4f}, AUC={auc:.4f}, "
        f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}"
    )
    
    return metrics_dict


def generate_classification_plots(
    y_test: pd.Series,
    y_pred: pd.Series,
    y_pred_proba: pd.Series,
    out_dir: Path,
    model_name: str,
    logger: logging.Logger,
) -> None:
    """Generate classification visualization plots."""
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name.replace('_', ' ').title()}: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close()

    # ROC Curve
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = 0.5  # Random performance when only one class
        logger.warning("Cannot compute AUC with only one class - using 0.5")
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_name.replace('_', ' ').title()}: ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{model_name}_roc_curve.png", dpi=150)
    plt.close()

    # Prediction Probability Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='No Default', color='blue')
    plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Default', color='red')
    plt.xlabel('Predicted Probability of Default')
    plt.ylabel('Frequency')
    plt.title(f"{model_name.replace('_', ' ').title()}: Prediction Probability Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{model_name}_probability_distribution.png", dpi=150)
    plt.close()

    logger.info(f"Generated plots for {model_name}")


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate loan default classification models")
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

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(data_dir, logger)

    # Train Random Forest
    rf_metrics = train_random_forest(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        models_dir=models_dir,
        out_dir=out_dir,
        seed=args.seed,
        n_estimators=args.n_estimators,
        logger=logger,
    )

    # Train Logistic Regression
    lr_metrics = train_logistic_regression(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        models_dir=models_dir,
        out_dir=out_dir,
        seed=args.seed,
        logger=logger,
    )

    # Model comparison
    comparison = pd.DataFrame({
        "RandomForest": rf_metrics,
        "LogisticRegression": lr_metrics,
    }).T

    comparison_path = out_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path)

    # Console summary
    logger.info("🔥 Model Training Complete!")
    logger.info(f"📊 Random Forest - F1: {rf_metrics['F1']:.4f} | AUC: {rf_metrics['AUC']:.4f} | Accuracy: {rf_metrics['Accuracy']:.4f}")
    logger.info(f"📊 Logistic Regression - F1: {lr_metrics['F1']:.4f} | AUC: {lr_metrics['AUC']:.4f} | Accuracy: {lr_metrics['Accuracy']:.4f}")
    logger.info(f"💾 Models saved to {models_dir}/")
    logger.info(f"📈 Metrics and plots saved to {out_dir}/")

    # Use Random Forest predictions for fairness analysis (typically better performance)
    # Copy RF predictions to standard name for fairness pipeline
    rf_preds_path = out_dir / "random_forest_predictions.csv"
    standard_preds_path = out_dir / "classification_predictions.csv"
    
    if rf_preds_path.exists():
        rf_preds = pd.read_csv(rf_preds_path)
        # Rename columns to match fairness script expectations
        standard_preds = rf_preds.rename(columns={
            "predicted": "predicted"
        })
        standard_preds.to_csv(standard_preds_path, index=False)
        logger.info(f"📋 Using Random Forest predictions for fairness analysis: {standard_preds_path}")


if __name__ == "__main__":
    main()