"""
compute_fairness.py (Diabetes)

Evaluate fairness metrics for the Diabetes Outcome classifier using predictions
exported by scripts/train_models.py.

Outputs under diabetes_dataset/metrics/fairness/:
- fairness_metrics.csv
- fairness_summary.json
- report.md
- plots/*.png

Example:

python compute_fairness.py \
  --predictions_dir metrics \
  --out_dir metrics/fairness \
  --protected_attrs older_age,high_pregnancy_count \
  --threshold 0.8
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    AIF360_AVAILABLE = True
except Exception:
    AIF360_AVAILABLE = False


def get_logger() -> logging.Logger:
    logger = logging.getLogger("compute_fairness_diabetes")
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


def load_predictions(predictions_dir: Path, protected_attrs: List[str], logger: logging.Logger) -> pd.DataFrame:
    preds_path = predictions_dir / "classification_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {preds_path}. Run scripts/train_models.py first.")
    df = pd.read_csv(preds_path)
    required = ["actual", "predicted"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Predictions CSV must contain column: {c}")
    # Ensure protected attrs columns exist (if any missing, drop silently with warning)
    for a in list(protected_attrs):
        if a not in df.columns:
            logger.warning(f"Protected attribute '{a}' not found in predictions; it will be skipped.")
    return df


def _majority_group(values: pd.Series) -> int:
    counts = values.value_counts(dropna=False)
    return int(counts.idxmax())


def _manual_group_rates(y_true: pd.Series, y_pred: pd.Series, group: pd.Series, privileged_value: int) -> Tuple[float, float, float, float]:
    mask_priv = group == privileged_value
    mask_unpriv = group != privileged_value

    def rate_sel(mask: pd.Series) -> float:
        if mask.sum() == 0:
            return np.nan
        return float((y_pred[mask] == 1).mean())

    def rate_tpr(mask: pd.Series) -> float:
        denom = (y_true[mask] == 1).sum()
        if denom == 0:
            return np.nan
        return float(((y_true[mask] == 1) & (y_pred[mask] == 1)).sum() / denom)

    def rate_fpr(mask: pd.Series) -> float:
        denom = (y_true[mask] == 0).sum()
        if denom == 0:
            return np.nan
        return float(((y_true[mask] == 0) & (y_pred[mask] == 1)).sum() / denom)

    sel_priv = rate_sel(mask_priv)
    sel_unpriv = rate_sel(mask_unpriv)
    di = np.nan
    if sel_unpriv is not None and sel_unpriv and not np.isnan(sel_unpriv):
        di = float(sel_priv / sel_unpriv) if sel_unpriv > 0 else np.nan
    dpd = float(sel_priv - sel_unpriv) if not (np.isnan(sel_priv) or np.isnan(sel_unpriv)) else np.nan

    tpr_priv = rate_tpr(mask_priv)
    tpr_unpriv = rate_tpr(mask_unpriv)
    eod = float(tpr_priv - tpr_unpriv) if not (np.isnan(tpr_priv) or np.isnan(tpr_unpriv)) else np.nan

    fpr_priv = rate_fpr(mask_priv)
    fpr_unpriv = rate_fpr(mask_unpriv)
    aod = np.nan
    if not (np.isnan(fpr_priv) or np.isnan(fpr_unpriv) or np.isnan(tpr_priv) or np.isnan(tpr_unpriv)):
        aod = float(((fpr_priv - fpr_unpriv) + (tpr_priv - tpr_unpriv)) / 2.0)

    return di, dpd, eod, aod


def compute_metrics(df: pd.DataFrame, protected_attrs: List[str], threshold: float, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    results: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, float]] = {}

    y_true = df["actual"].astype(int)
    y_pred = df["predicted"].astype(int)

    for attr in protected_attrs:
        if attr not in df.columns:
            logger.warning(f"Attribute '{attr}' missing; skipping.")
            continue
        series = df[attr].astype(int)
        privileged_value = _majority_group(series)

        di = np.nan
        dpd = np.nan
        eod = np.nan
        aod = np.nan

        if AIF360_AVAILABLE:
            try:
                df_true = pd.DataFrame({attr: series, "Outcome": y_true})
                bld_true = BinaryLabelDataset(
                    df=df_true,
                    label_names=["Outcome"],
                    protected_attribute_names=[attr],
                    favorable_label=1,
                    unfavorable_label=0,
                )
                df_pred = pd.DataFrame({attr: series, "Outcome": y_pred})
                bld_pred = BinaryLabelDataset(
                    df=df_pred,
                    label_names=["Outcome"],
                    protected_attribute_names=[attr],
                    favorable_label=1,
                    unfavorable_label=0,
                )
                priv = [{attr: 1 if privileged_value == 1 else 0}]
                unpriv = [{attr: 0 if privileged_value == 1 else 1}]
                bld_metric = BinaryLabelDatasetMetric(bld_true, privileged_groups=priv, unprivileged_groups=unpriv)
                di = float(bld_metric.disparate_impact())
                dpd = float(bld_metric.mean_difference())

                cls_metric = ClassificationMetric(bld_true, bld_pred, privileged_groups=priv, unprivileged_groups=unpriv)
                eod = float(cls_metric.equal_opportunity_difference())
                aod = float(cls_metric.average_odds_difference())
            except Exception as e:
                logger.warning(f"AIF360 failed for '{attr}', fallback to manual: {e}")
                di, dpd, eod, aod = _manual_group_rates(y_true, y_pred, series, privileged_value)
        else:
            di, dpd, eod, aod = _manual_group_rates(y_true, y_pred, series, privileged_value)

        bias_flag = bool(di < threshold) if (isinstance(di, (float, int)) and not np.isnan(di)) else False
        results.append({
            "Attribute": attr,
            "PrivilegedValue": int(privileged_value),
            "DisparateImpact": round(float(di), 4) if isinstance(di, (float, int)) and not np.isnan(di) else None,
            "DemographicParityDiff": round(float(dpd), 4) if isinstance(dpd, (float, int)) and not np.isnan(dpd) else None,
            "EqualOpportunityDiff": round(float(eod), 4) if isinstance(eod, (float, int)) and not np.isnan(eod) else None,
            "AverageOddsDiff": round(float(aod), 4) if isinstance(aod, (float, int)) and not np.isnan(aod) else None,
            "BiasFlag": bias_flag,
        })
        summary[attr] = {
            "DisparateImpact": float(di) if isinstance(di, (float, int)) and not np.isnan(di) else None,
            "BiasFlag": bias_flag,
        }

    return pd.DataFrame(results), summary


def export_results(results_df: pd.DataFrame, summary: Dict[str, Dict[str, float]], out_dir: Path, threshold: float, logger: logging.Logger) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fairness_metrics.csv"
    json_path = out_dir / "fairness_summary.json"
    report_path = out_dir / "report.md"

    results_df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines: List[str] = []
    lines.append("# Fairness Report: Diabetes Outcome")
    lines.append("")
    lines.append(f"Threshold for disparate impact bias flag: DI < {threshold}")
    lines.append("")
    if not results_df.empty:
        lines.append("| Attribute | Privileged | DI | DPD | EOD | AOD | BiasFlag |")
        lines.append("|---|---|---:|---:|---:|---:|:---:|")
        for _, r in results_df.iterrows():
            lines.append(
                f"| {r['Attribute']} | {r['PrivilegedValue']} | "
                f"{r['DisparateImpact'] if pd.notna(r['DisparateImpact']) else ''} | "
                f"{r['DemographicParityDiff'] if pd.notna(r['DemographicParityDiff']) else ''} | "
                f"{r['EqualOpportunityDiff'] if pd.notna(r['EqualOpportunityDiff']) else ''} | "
                f"{r['AverageOddsDiff'] if pd.notna(r['AverageOddsDiff']) else ''} | "
                f"{'⚠️' if r['BiasFlag'] else '✅'} |"
            )

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Saved fairness metrics to {csv_path}")
    logger.info(f"Saved fairness summary to {json_path}")
    logger.info(f"Saved fairness report to {report_path}")

    # Simple DI bar plot
    try:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        if not results_df.empty and 'DisparateImpact' in results_df.columns:
            attrs = results_df['Attribute'].tolist()
            di_vals = [np.nan if pd.isna(v) else float(v) for v in results_df['DisparateImpact'].tolist()]
            plt.figure(figsize=(6, 4))
            bars = plt.bar(attrs, di_vals, color=['#2E86AB' for _ in attrs])
            plt.axhline(threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold {threshold}')
            plt.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='Parity (1.0)')
            plt.ylabel('Disparate Impact (DI)')
            plt.title('DI by Attribute')
            plt.xticks(rotation=15)
            plt.legend()
            for bar, val in zip(bars, di_vals):
                if not np.isnan(val):
                    plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=8)
            plt.tight_layout()
            plt.savefig(plots_dir / 'di_by_attribute.png', dpi=150)
            plt.close()
    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")


def parse_attr_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fairness metrics for Diabetes predictions")
    parser.add_argument("--predictions_dir", type=str, default="metrics")
    parser.add_argument("--out_dir", type=str, default="metrics/fairness")
    parser.add_argument("--protected_attrs", type=str, default="older_age,high_pregnancy_count")
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    logger = get_logger()
    predictions_dir = Path(args.predictions_dir)
    out_dir = Path(args.out_dir)
    protected_attrs = parse_attr_list(args.protected_attrs)

    logger.info(
        f"Starting with predictions_dir={predictions_dir}, out_dir={out_dir}, protected_attrs={protected_attrs}, threshold={args.threshold}"
    )

    df = load_predictions(predictions_dir=predictions_dir, protected_attrs=protected_attrs, logger=logger)
    results_df, summary = compute_metrics(df=df, protected_attrs=protected_attrs, threshold=args.threshold, logger=logger)
    export_results(results_df=results_df, summary=summary, out_dir=out_dir, threshold=args.threshold, logger=logger)

    # Console summary
    lines = ["⚖️ Fairness Results (Diabetes):"]
    for _, r in results_df.iterrows():
        di_val = r["DisparateImpact"]
        mark = "⚠️ bias detected" if r["BiasFlag"] else "✅"
        lines.append(f" - {r['Attribute']} → DI: {di_val if pd.notna(di_val) else 'NA'} {mark}")
    lines.append(f"📊 Full report saved to {out_dir}/report.md")
    for ln in lines:
        logger.info(ln)


if __name__ == "__main__":
    main()
