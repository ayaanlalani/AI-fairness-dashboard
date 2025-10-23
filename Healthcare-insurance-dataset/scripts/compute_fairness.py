"""
compute_fairness.py

Evaluate algorithmic fairness for the Healthcare Cost Prediction dataset (Kaggle Insurance)
using the trained classifier's predictions on the binary high_cost label.

- Primary metrics per protected attribute (via AIF360 when available):
  - Disparate Impact (DI)
  - Demographic Parity Difference (DPD)
  - Equal Opportunity Difference (EOD)
  - Average Odds Difference (AOD)
- Optional cross-check with Fairlearn (group-wise selection rate, TPR, FPR)

Example run (matching current project layout under Healthcare-insurance-dataset/):

python scripts/compute_fairness.py \
  --data_dir processed \
  --predictions_dir metrics \
  --out_dir metrics/fairness \
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
except Exception:  # pragma: no cover
    AIF360_AVAILABLE = False

try:
    from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
    FAIRLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    FAIRLEARN_AVAILABLE = False


# -------------------------
# Logging
# -------------------------

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


# -------------------------
# Loading and preparation
# -------------------------

def load_data(data_dir: Path, predictions_dir: Path, protected_attrs: List[str], logger: logging.Logger) -> pd.DataFrame:
    """Load processed test features, labels, and predictions; reconstruct protected attributes.

    Returns a DataFrame with columns: protected_attrs..., high_cost (true), y_pred (predicted)
    """
    x_test_path = data_dir / "X_test_processed.csv"
    y_test_path = data_dir / "high_cost_test.csv"
    preds_path = predictions_dir / "classification_predictions.csv"

    for p in [x_test_path, y_test_path, preds_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    X_test = pd.read_csv(x_test_path)
    high_cost = pd.read_csv(y_test_path)["high_cost"].astype(int)
    preds_df = pd.read_csv(preds_path)

    # prefer predictions from file; ensure integer labels
    if "predicted" not in preds_df.columns:
        raise ValueError("classification_predictions.csv must contain a 'predicted' column")
    y_pred = preds_df["predicted"].astype(int)

    if len(X_test) != len(high_cost) or len(X_test) != len(y_pred):
        raise ValueError("Length mismatch between X_test, high_cost_test, and predictions")

    prot_df = reconstruct_protected_attributes(X_test, protected_attrs, logger)
    df = pd.DataFrame({"high_cost": high_cost, "y_pred": y_pred})
    merged = pd.concat([prot_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    # Log quick distribution
    pos_rate = float(merged["high_cost"].mean())
    logger.info(f"high_cost test positive rate: {pos_rate:.3f}")

    return merged


def reconstruct_protected_attributes(X: pd.DataFrame, protected_attrs: List[str], logger: logging.Logger) -> pd.DataFrame:
    """Reconstruct original protected attributes from processed (one-hot) features.

    Assumes preprocessing used OneHotEncoder(drop='first') with feature names:
    - sex -> 'sex_male' (implies female when 0)
    - smoker -> 'smoker_yes' (implies no when 0)
    - region -> ['region_northwest','region_southeast','region_southwest'] (implies 'northeast' when all 0)
    - age_bucket -> ['age_bucket_gt_50','age_bucket_lt_30'] (implies '30_50' when both 0)
    """
    out: Dict[str, pd.Series] = {}

    if "sex" in protected_attrs:
        if "sex_male" in X.columns:
            out["sex"] = np.where(X["sex_male"].astype(float) > 0.5, "male", "female")
        else:
            logger.warning("sex_male column not found; skipping 'sex' attribute")

    if "smoker" in protected_attrs:
        if "smoker_yes" in X.columns:
            out["smoker"] = np.where(X["smoker_yes"].astype(float) > 0.5, "yes", "no")
        else:
            logger.warning("smoker_yes column not found; skipping 'smoker' attribute")

    if "region" in protected_attrs:
        region_cols = [c for c in X.columns if c.startswith("region_")]
        expected = ["region_northwest", "region_southeast", "region_southwest"]
        if all(c in X.columns for c in expected):
            def infer_region(row: pd.Series) -> str:
                if row["region_northwest"] == 1:
                    return "northwest"
                if row["region_southeast"] == 1:
                    return "southeast"
                if row["region_southwest"] == 1:
                    return "southwest"
                return "northeast"  # dropped baseline
            out["region"] = X[expected].apply(infer_region, axis=1)
        elif region_cols:
            # Fallback: pick the 1-valued region col name suffix; else baseline 'northeast'
            def fallback_region(row: pd.Series) -> str:
                for c in region_cols:
                    if row[c] == 1:
                        return c.split("region_")[-1]
                return "northeast"
            out["region"] = X[region_cols].apply(fallback_region, axis=1)
        else:
            logger.warning("region_* columns not found; skipping 'region' attribute")

    if "age_bucket" in protected_attrs:
        gt = "age_bucket_gt_50"
        lt = "age_bucket_lt_30"
        if gt in X.columns and lt in X.columns:
            out["age_bucket"] = np.select(
                [X[lt] == 1, X[gt] == 1],
                ["lt_30", "gt_50"],
                default="30_50",
            )
        else:
            logger.warning("age_bucket_* columns not found; skipping 'age_bucket' attribute")

    if not out:
        raise ValueError("None of the requested protected attributes could be reconstructed from processed features.")

    return pd.DataFrame(out)


# -------------------------
# Fairness metrics (AIF360 and manual fallbacks)
# -------------------------

def _majority_group(values: pd.Series) -> str:
    counts = values.value_counts(dropna=False)
    return str(counts.idxmax())


def _manual_group_rates(y_true: pd.Series, y_pred: pd.Series, group: pd.Series, privileged_value: str) -> Tuple[float, float, float, float]:
    """Compute selection rate, TPR, FPR for privileged and unprivileged groups, plus DI, DPD, EOD, AOD manually.
    Returns: (di, dpd, eod, aod)
    """
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


def compute_aif360_metrics(df: pd.DataFrame, protected_attrs: List[str], threshold: float, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Compute fairness metrics per protected attribute using AIF360 when available, with manual fallbacks.

    Returns: (results_df, summary_dict)
    """
    results: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, float]] = {}

    for attr in protected_attrs:
        if attr not in df.columns:
            logger.warning(f"Protected attribute '{attr}' missing in merged data; skipping.")
            continue

        series = df[attr].astype(str)
        y_true = df["high_cost"].astype(int)
        y_pred = df["y_pred"].astype(int)

        privileged_value = _majority_group(series)
        unprivileged_values = [v for v in series.unique().tolist() if v != privileged_value]

        di = np.nan
        dpd = np.nan
        eod = np.nan
        aod = np.nan

        if AIF360_AVAILABLE:
            # Build datasets for aif360
            try:
                # Encode protected attr as binary: 1 if privileged, else 0 (required numeric for AIF360)
                series_bin = (series == privileged_value).astype(int)

                # true labels dataset
                df_true = pd.DataFrame({attr: series_bin, "high_cost": y_true})
                bld_true = BinaryLabelDataset(
                    df=df_true,
                    label_names=["high_cost"],
                    protected_attribute_names=[attr],
                    favorable_label=1,
                    unfavorable_label=0,
                )
                # predicted labels dataset
                df_pred = pd.DataFrame({attr: series_bin, "high_cost": y_pred})
                bld_pred = BinaryLabelDataset(
                    df=df_pred,
                    label_names=["high_cost"],
                    protected_attribute_names=[attr],
                    favorable_label=1,
                    unfavorable_label=0,
                )

                priv = [{attr: 1}]
                unpriv = [{attr: 0}]

                bld_metric = BinaryLabelDatasetMetric(bld_true, privileged_groups=priv, unprivileged_groups=unpriv)
                di = float(bld_metric.disparate_impact())
                dpd = float(bld_metric.mean_difference())

                cls_metric = ClassificationMetric(bld_true, bld_pred, privileged_groups=priv, unprivileged_groups=unpriv)
                eod = float(cls_metric.equal_opportunity_difference())
                aod = float(cls_metric.average_odds_difference())
            except Exception as e:  # pragma: no cover
                logger.warning(f"AIF360 metric computation failed for '{attr}', falling back to manual. Error: {e}")
                di, dpd, eod, aod = _manual_group_rates(y_true, y_pred, series, privileged_value)
        else:
            # Manual fallback
            di, dpd, eod, aod = _manual_group_rates(y_true, y_pred, series, privileged_value)

        bias_flag = bool(di < threshold) if (isinstance(di, (float, int)) and not np.isnan(di)) else False

        results.append({
            "Attribute": attr,
            "PrivilegedValue": privileged_value,
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

    results_df = pd.DataFrame(results)
    return results_df, summary


# -------------------------
# Fairlearn group metrics (optional)
# -------------------------

def compute_fairlearn_metrics(df: pd.DataFrame, protected_attrs: List[str], logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """Compute group-wise metrics (selection rate, TPR, FPR) per attribute.

    If Fairlearn is available, use MetricFrame; otherwise compute manually.
    Returns a dict of DataFrames keyed by attribute.
    """
    tables: Dict[str, pd.DataFrame] = {}
    y_true = df["high_cost"].astype(int)
    y_pred = df["y_pred"].astype(int)

    use_fairlearn = FAIRLEARN_AVAILABLE
    if not FAIRLEARN_AVAILABLE:
        logger.warning("Fairlearn not available; computing group-wise metrics manually.")

    for attr in protected_attrs:
        if attr not in df.columns:
            continue
        sf = df[attr].astype(str)
        if use_fairlearn:
            try:
                mf = MetricFrame(
                    metrics={
                        "selection_rate": selection_rate,
                        "TPR": true_positive_rate,
                        "FPR": false_positive_rate,
                    },
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sf,
                )
                table = pd.DataFrame({
                    "selection_rate": mf.by_group["selection_rate"],
                    "TPR": mf.by_group["TPR"],
                    "FPR": mf.by_group["FPR"],
                })
                tables[attr] = table
                continue
            except Exception as e:  # pragma: no cover
                logger.warning(f"Fairlearn MetricFrame failed for '{attr}', falling back to manual: {e}")

        # Manual per-group computation
        groups = sf.unique().tolist()
        rows = []
        for g in groups:
            mask = sf == g
            sel = float((y_pred[mask] == 1).mean()) if mask.any() else np.nan
            pos = int((y_true[mask] == 1).sum())
            neg = int((y_true[mask] == 0).sum())
            tpr = float(((y_true[mask] == 1) & (y_pred[mask] == 1)).sum() / pos) if pos > 0 else np.nan
            fpr = float(((y_true[mask] == 0) & (y_pred[mask] == 1)).sum() / neg) if neg > 0 else np.nan
            rows.append({"Group": g, "selection_rate": sel, "TPR": tpr, "FPR": fpr})
        table = pd.DataFrame(rows).set_index("Group")
        tables[attr] = table

    return tables


# -------------------------
# Export
# -------------------------

def export_results(results_df: pd.DataFrame, summary: Dict[str, Dict[str, float]], fairlearn_tables: Dict[str, pd.DataFrame], out_dir: Path, threshold: float, logger: logging.Logger) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "fairness_metrics.csv"
    json_path = out_dir / "fairness_summary.json"
    report_path = out_dir / "report.md"

    results_df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Markdown report
    lines: List[str] = []
    lines.append("# Fairness Report: high_cost classification")
    lines.append("")
    lines.append(f"Threshold for disparate impact bias flag: DI < {threshold}")
    lines.append("")

    if not results_df.empty:
        lines.append("## Summary metrics (per attribute)")
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
        lines.append("")

    # Fairlearn group tables
    if fairlearn_tables:
        lines.append("## Fairlearn group-wise diagnostics (selection rate, TPR, FPR)")
        for attr, table in fairlearn_tables.items():
            lines.append(f"### {attr}")
            lines.append("| Group | selection_rate | TPR | FPR |")
            lines.append("|---|---:|---:|---:|")
            for grp, row in table.iterrows():
                lines.append(f"| {grp} | {row['selection_rate']:.4f} | {row['TPR']:.4f} | {row['FPR']:.4f} |")
            lines.append("")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved fairness metrics to {csv_path}")
    logger.info(f"Saved fairness summary to {json_path}")
    logger.info(f"Saved fairness report to {report_path}")

    # Generate visuals
    try:
        generate_visuals(results_df=results_df, fairlearn_tables=fairlearn_tables, out_dir=out_dir, threshold=threshold)
        logger.info(f"Saved fairness plots to {out_dir / 'plots'}")
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to generate fairness plots: {e}")

    # Append interpretation and plot links to report
    try:
        interp: List[str] = []
        interp.append("\n## Interpretation")
        interp.append(f"- We flag potential disparate impact bias when DI < {threshold}.")
        if not results_df.empty:
            for _, r in results_df.iterrows():
                di_val = r['DisparateImpact']
                mark = '⚠️ potential bias' if r['BiasFlag'] else '✅ within threshold'
                interp.append(f"- {r['Attribute']}: DI={di_val if pd.notna(di_val) else 'NA'} → {mark}")
        interp.append("\n## Plots")
        interp.append("- DI by attribute: ./plots/di_by_attribute.png")
        for attr in fairlearn_tables.keys():
            interp.append(f"- Selection rate by group ({attr}): ./plots/selection_rate_{attr}.png")
            interp.append(f"- TPR by group ({attr}): ./plots/tpr_{attr}.png")
            interp.append(f"- FPR by group ({attr}): ./plots/fpr_{attr}.png")
        with report_path.open("a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(interp) + "\n")
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to augment report with interpretation: {e}")


def generate_visuals(results_df: pd.DataFrame, fairlearn_tables: Dict[str, pd.DataFrame], out_dir: Path, threshold: float) -> None:
    """Create DI bar chart and per-attribute group-wise selection/TPR/FPR plots."""
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # DI by attribute
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

    # Group-wise plots per attribute
    for attr, table in fairlearn_tables.items():
        if table.empty:
            continue
        # Selection rate
        plt.figure(figsize=(6, 3.5))
        table['selection_rate'].plot(kind='bar', color='#6C5CE7')
        plt.ylabel('Selection rate (P(ŷ=1))')
        plt.title(f'Selection rate by group: {attr}')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / f'selection_rate_{attr}.png', dpi=150)
        plt.close()

        # TPR
        plt.figure(figsize=(6, 3.5))
        table['TPR'].plot(kind='bar', color='#00B894')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'TPR by group: {attr}')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / f'tpr_{attr}.png', dpi=150)
        plt.close()

        # FPR
        plt.figure(figsize=(6, 3.5))
        table['FPR'].plot(kind='bar', color='#E17055')
        plt.ylabel('False Positive Rate (FPR)')
        plt.title(f'FPR by group: {attr}')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / f'fpr_{attr}.png', dpi=150)
        plt.close()


# -------------------------
# Main
# -------------------------

def parse_attr_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fairness metrics for high_cost classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed data (X_test_processed.csv, high_cost_test.csv)")
    parser.add_argument("--predictions_dir", type=str, default="metrics", help="Directory with classification_predictions.csv")
    parser.add_argument("--out_dir", type=str, default="metrics/fairness", help="Directory to write fairness outputs")
    parser.add_argument("--protected_attrs", type=str, default="sex,smoker,region,age_bucket", help="Comma-separated protected attribute names")
    parser.add_argument("--threshold", type=float, default=0.8, help="DI bias flag threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    logger = get_logger()

    data_dir = Path(args.data_dir)
    predictions_dir = Path(args.predictions_dir)
    out_dir = Path(args.out_dir)
    protected_attrs = parse_attr_list(args.protected_attrs)

    logger.info(
        f"Starting fairness computation with data_dir={data_dir}, predictions_dir={predictions_dir}, "
        f"out_dir={out_dir}, protected_attrs={protected_attrs}, threshold={args.threshold}"
    )

    df = load_data(data_dir=data_dir, predictions_dir=predictions_dir, protected_attrs=protected_attrs, logger=logger)

    results_df, summary = compute_aif360_metrics(df=df, protected_attrs=protected_attrs, threshold=args.threshold, logger=logger)

    fairlearn_tables = compute_fairlearn_metrics(df=df, protected_attrs=protected_attrs, logger=logger)

    export_results(results_df=results_df, summary=summary, fairlearn_tables=fairlearn_tables, out_dir=out_dir, threshold=args.threshold, logger=logger)

    # Console summary
    lines = ["⚖️ Fairness Results:"]
    for _, r in results_df.iterrows():
        di_val = r["DisparateImpact"]
        mark = "⚠️ bias detected" if r["BiasFlag"] else "✅"
        lines.append(f" - {r['Attribute']} → DI: {di_val if pd.notna(di_val) else 'NA'} {mark}")
    lines.append(f"📊 Full report saved to {out_dir}/report.md")
    for ln in lines:
        logger.info(ln)


if __name__ == "__main__":
    main()
