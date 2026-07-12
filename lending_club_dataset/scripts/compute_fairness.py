"""
compute_fairness.py

Evaluate algorithmic fairness for the Lending Club loan default prediction dataset
using the trained classifier's predictions on the binary loan_default label.

- Primary metrics per protected attribute (via AIF360 when available):
  - Disparate Impact (DI)
  - Demographic Parity Difference (DPD)
  - Equal Opportunity Difference (EOD)
  - Average Odds Difference (AOD)
- Optional cross-check with Fairlearn (group-wise selection rate, TPR, FPR)

All metrics are oriented on the *favorable* outcome (default: predicted
non-default, loan_default = 0) and follow the AIF360 conventions:
DI = unprivileged rate / privileged rate; differences are
unprivileged - privileged. This matches the shared audit spec used by
scripts/qualitative_analysis.py (invoked with --favorable_label 0).

Example run:

python scripts/compute_fairness.py \
  --data_dir processed \
  --predictions_dir metrics \
  --out_dir metrics/fairness \
  --favorable_label 0 \
  --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional dependencies
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    AIF360_AVAILABLE = True
except Exception:
    AIF360_AVAILABLE = False

try:
    from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
    FAIRLEARN_AVAILABLE = True
except Exception:
    FAIRLEARN_AVAILABLE = False


def get_logger() -> logging.Logger:
    logger = logging.getLogger("compute_fairness")
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


def load_data(data_dir: Path, predictions_dir: Path, protected_attrs: List[str], logger: logging.Logger) -> pd.DataFrame:
    """Load processed test features, labels, and predictions; reconstruct protected attributes."""
    x_test_path = data_dir / "X_test_processed.csv"
    y_test_path = data_dir / "y_test.csv"
    preds_path = predictions_dir / "classification_predictions.csv"

    for p in [x_test_path, y_test_path, preds_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    X_test = pd.read_csv(x_test_path)
    loan_default = pd.read_csv(y_test_path)["loan_default"].astype(int)
    preds_df = pd.read_csv(preds_path)

    if "predicted" not in preds_df.columns:
        raise ValueError("classification_predictions.csv must contain a 'predicted' column")
    y_pred = preds_df["predicted"].astype(int)

    if len(X_test) != len(loan_default) or len(X_test) != len(y_pred):
        raise ValueError("Length mismatch between X_test, loan_default, and predictions")

    prot_df = reconstruct_protected_attributes(X_test, protected_attrs, logger)
    df = pd.DataFrame({"loan_default": loan_default, "y_pred": y_pred})
    merged = pd.concat([prot_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    default_rate = float(merged["loan_default"].mean())
    pred_default_rate = float(merged["y_pred"].mean())
    logger.info(f"Actual default rate: {default_rate:.3f}")
    logger.info(f"Predicted default rate: {pred_default_rate:.3f}")

    return merged


def reconstruct_protected_attributes(X: pd.DataFrame, protected_attrs: List[str], logger: logging.Logger) -> pd.DataFrame:
    """Reconstruct original protected attributes from processed (one-hot) features."""
    out: Dict[str, pd.Series] = {}

    if "gender" in protected_attrs:
        if "gender_male" in X.columns:
            out["gender"] = np.where(X["gender_male"].astype(float) > 0.5, "male", "female")
        else:
            logger.warning("gender_male column not found; skipping 'gender' attribute")

    if "income_level" in protected_attrs:
        income_cols = [c for c in X.columns if c.startswith("income_level_")]
        if "income_level_medium" in X.columns and "income_level_low" in X.columns:
            def infer_income(row: pd.Series) -> str:
                if row.get("income_level_low", 0) == 1:
                    return "low"
                if row.get("income_level_medium", 0) == 1:
                    return "medium"
                return "high"  # dropped baseline
            
            income_cols_available = [c for c in ["income_level_low", "income_level_medium"] if c in X.columns]
            if income_cols_available:
                out["income_level"] = X[income_cols_available].apply(infer_income, axis=1)
        else:
            logger.warning("income_level_* columns not found; skipping 'income_level' attribute")

    if "loan_amount_level" in protected_attrs:
        if "loan_amount_level_medium" in X.columns and "loan_amount_level_small" in X.columns:
            def infer_loan_amount(row: pd.Series) -> str:
                if row.get("loan_amount_level_small", 0) == 1:
                    return "small"
                if row.get("loan_amount_level_medium", 0) == 1:
                    return "medium"
                return "large"  # dropped baseline
            
            loan_cols_available = [c for c in ["loan_amount_level_small", "loan_amount_level_medium"] if c in X.columns]
            if loan_cols_available:
                out["loan_amount_level"] = X[loan_cols_available].apply(infer_loan_amount, axis=1)
        else:
            logger.warning("loan_amount_level_* columns not found; skipping 'loan_amount_level' attribute")

    if not out:
        raise ValueError("None of the requested protected attributes could be reconstructed from processed features.")

    return pd.DataFrame(out)


def _majority_group(values: pd.Series) -> str:
    """Return the most frequent value (privileged group)."""
    counts = values.value_counts(dropna=False)
    return str(counts.idxmax())


def _manual_group_rates(
    y_true: pd.Series,
    y_pred: pd.Series,
    group: pd.Series,
    privileged_value: str,
    favorable_label: int,
) -> Tuple[float, float, float, float]:
    """Compute DI, DPD, EOD, AOD oriented on the favorable label (AIF360 conventions)."""
    mask_priv = group == privileged_value
    mask_unpriv = group != privileged_value

    def rate_sel(mask: pd.Series) -> float:
        if mask.sum() == 0:
            return np.nan
        return float((y_pred[mask] == favorable_label).mean())

    def rate_tpr(mask: pd.Series) -> float:
        denom = (y_true[mask] == favorable_label).sum()
        if denom == 0:
            return np.nan
        return float(((y_true[mask] == favorable_label) & (y_pred[mask] == favorable_label)).sum() / denom)

    def rate_fpr(mask: pd.Series) -> float:
        denom = (y_true[mask] != favorable_label).sum()
        if denom == 0:
            return np.nan
        return float(((y_true[mask] != favorable_label) & (y_pred[mask] == favorable_label)).sum() / denom)

    sel_priv = rate_sel(mask_priv)
    sel_unpriv = rate_sel(mask_unpriv)

    # Disparate Impact: unprivileged favorable rate / privileged favorable rate
    di = np.nan
    if not (np.isnan(sel_priv) or np.isnan(sel_unpriv)) and sel_priv > 0:
        di = float(sel_unpriv / sel_priv)

    # Demographic Parity Difference: unprivileged - privileged
    dpd = float(sel_unpriv - sel_priv) if not (np.isnan(sel_priv) or np.isnan(sel_unpriv)) else np.nan

    # Equal Opportunity Difference: unprivileged - privileged
    tpr_priv = rate_tpr(mask_priv)
    tpr_unpriv = rate_tpr(mask_unpriv)
    eod = float(tpr_unpriv - tpr_priv) if not (np.isnan(tpr_priv) or np.isnan(tpr_unpriv)) else np.nan

    # Average Odds Difference: mean of FPR and TPR gaps (unprivileged - privileged)
    fpr_priv = rate_fpr(mask_priv)
    fpr_unpriv = rate_fpr(mask_unpriv)
    aod = np.nan
    if not any(np.isnan([fpr_priv, fpr_unpriv, tpr_priv, tpr_unpriv])):
        aod = float(((fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv)) / 2.0)

    return di, dpd, eod, aod


def parse_attr_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fairness metrics for loan default classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed data")
    parser.add_argument("--predictions_dir", type=str, default="metrics", help="Directory with classification_predictions.csv")
    parser.add_argument("--out_dir", type=str, default="metrics/fairness", help="Directory to write fairness outputs")
    parser.add_argument("--protected_attrs", type=str, default="gender,income_level,loan_amount_level", help="Comma-separated protected attribute names")
    parser.add_argument("--threshold", type=float, default=0.8, help="DI bias flag threshold")
    parser.add_argument("--favorable_label", type=int, default=0, help="Favorable outcome label (0 = predicted non-default)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    logger = get_logger()

    logger.info("Fairness analysis script started successfully")
    logger.info(f"AIF360 available: {AIF360_AVAILABLE}")
    logger.info(f"Fairlearn available: {FAIRLEARN_AVAILABLE}")
    logger.info(f"Favorable label: {args.favorable_label}")

    # Parse protected attributes
    protected_attrs = parse_attr_list(args.protected_attrs)
    logger.info(f"Protected attributes: {protected_attrs}")

    # Load data
    data_dir = Path(args.data_dir)
    predictions_dir = Path(args.predictions_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data and predictions...")
    df = load_data(data_dir, predictions_dir, protected_attrs, logger)
    
    # Compute fairness metrics for each protected attribute
    all_results = {}
    
    for attr in protected_attrs:
        if attr not in df.columns:
            logger.warning(f"Protected attribute '{attr}' not found in data")
            continue
            
        logger.info(f"Computing fairness metrics for: {attr}")
        
        # Get unique groups
        groups = df[attr].unique()
        privileged_group = _majority_group(df[attr])
        logger.info(f"  Groups: {list(groups)}")
        logger.info(f"  Privileged group: {privileged_group}")
        
        # Compute metrics manually
        di, dpd, eod, aod = _manual_group_rates(
            df['loan_default'],
            df['y_pred'],
            df[attr],
            privileged_group,
            args.favorable_label,
        )
        
        # Store results
        results = {
            'disparate_impact': float(di) if not np.isnan(di) else None,
            'demographic_parity_difference': float(dpd) if not np.isnan(dpd) else None,
            'equal_opportunity_difference': float(eod) if not np.isnan(eod) else None,
            'average_odds_difference': float(aod) if not np.isnan(aod) else None,
            'privileged_group': privileged_group,
            'groups': list(groups),
            'bias_detected': (di < args.threshold) if not np.isnan(di) else False
        }
        
        all_results[attr] = results
        
        # Log results
        logger.info(f"  Disparate Impact: {di:.3f}" if not np.isnan(di) else "  Disparate Impact: N/A")
        logger.info(f"  Demographic Parity Diff: {dpd:.3f}" if not np.isnan(dpd) else "  Demographic Parity Diff: N/A")
        logger.info(f"  Equal Opportunity Diff: {eod:.3f}" if not np.isnan(eod) else "  Equal Opportunity Diff: N/A")
        logger.info(f"  Average Odds Diff: {aod:.3f}" if not np.isnan(aod) else "  Average Odds Diff: N/A")
        logger.info(f"  Bias detected: {'⚠️ YES' if results['bias_detected'] else '✅ NO'}")
    
    # Save results
    summary_path = out_dir / "fairness_summary.json"
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    
    # Create detailed CSV
    rows = []
    for attr, results in all_results.items():
        row = {
            'protected_attribute': attr,
            'disparate_impact': results['disparate_impact'],
            'demographic_parity_difference': results['demographic_parity_difference'],
            'equal_opportunity_difference': results['equal_opportunity_difference'],
            'average_odds_difference': results['average_odds_difference'],
            'privileged_group': results['privileged_group'],
            'bias_detected': results['bias_detected']
        }
        rows.append(row)
    
    import pandas as pd
    fairness_df = pd.DataFrame(rows)
    fairness_df.to_csv(out_dir / "fairness_metrics.csv", index=False)
    
    # Generate markdown report
    report_lines = [
        "# AI Fairness Analysis Report",
        "",
        f"**Dataset**: Lending Club Loan Default Prediction",
        f"**Total Records**: {len(df):,}",
        f"**Default Rate**: {df['loan_default'].mean():.1%}",
        f"**Prediction Default Rate**: {df['y_pred'].mean():.1%}",
        f"**Favorable Label**: {args.favorable_label} (metrics oriented on predicted {'non-default' if args.favorable_label == 0 else 'default'})",
        "",
        "## Fairness Metrics Summary",
        "",
    ]
    
    for attr, results in all_results.items():
        report_lines.extend([
            f"### {attr.replace('_', ' ').title()}",
            "",
            f"- **Privileged Group**: {results['privileged_group']}",
            f"- **Groups**: {', '.join(results['groups'])}",
            f"- **Disparate Impact**: {results['disparate_impact']:.3f}" if results['disparate_impact'] is not None else "- **Disparate Impact**: N/A",
            f"- **Demographic Parity Difference**: {results['demographic_parity_difference']:.3f}" if results['demographic_parity_difference'] is not None else "- **Demographic Parity Difference**: N/A",
            f"- **Equal Opportunity Difference**: {results['equal_opportunity_difference']:.3f}" if results['equal_opportunity_difference'] is not None else "- **Equal Opportunity Difference**: N/A",
            f"- **Average Odds Difference**: {results['average_odds_difference']:.3f}" if results['average_odds_difference'] is not None else "- **Average Odds Difference**: N/A",
            f"- **Bias Status**: {'⚠️ BIAS DETECTED' if results['bias_detected'] else '✅ WITHIN THRESHOLD'}",
            "",
        ])
    
    report_lines.extend([
        "## Interpretation",
        "",
        "- **Disparate Impact < 0.8**: Potential bias (80% rule)",
        "- **Disparate Impact closer to 1.0**: Better fairness",
        "- **Large absolute differences**: Concerning for equity",
        "",
        "## Recommendations",
        "",
    ])
    
    # Add recommendations based on results
    bias_detected = any(r['bias_detected'] for r in all_results.values())
    if bias_detected:
        report_lines.extend([
            "⚠️ **Bias detected in one or more protected attributes:**",
            "1. Review model training data for historical bias",
            "2. Consider bias mitigation techniques (reweighting, adversarial debiasing)",
            "3. Implement fairness constraints during model training",
            "4. Monitor model performance across all groups regularly",
        ])
    else:
        report_lines.extend([
            "✅ **No significant bias detected:**",
            "1. Continue monitoring model fairness over time",
            "2. Validate with additional datasets if available",
            "3. Consider stakeholder feedback on fairness perception",
        ])
    
    with (out_dir / "report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    logger.info("🎉 Fairness analysis complete!")
    logger.info(f"📊 Results saved to: {out_dir}")
    logger.info(f"📋 Summary: {summary_path}")
    logger.info(f"📈 Detailed metrics: {out_dir / 'fairness_metrics.csv'}")
    logger.info(f"📝 Report: {out_dir / 'report.md'}")
    
    # Print summary
    bias_count = sum(1 for r in all_results.values() if r['bias_detected'])
    logger.info(f"🚨 Bias detected in {bias_count}/{len(all_results)} protected attributes")

if __name__ == "__main__":
    main()