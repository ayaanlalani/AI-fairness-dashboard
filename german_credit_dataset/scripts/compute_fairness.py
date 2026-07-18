"""
compute_fairness.py  (German Credit)

Evaluate algorithmic fairness using the classifier's predictions.

Protected attributes and privileged groups (domain-knowledge based):
  Sex:            male is privileged   (historically advantaged in credit)
  AgeGroup:       40_plus is privileged (established credit history)
  foreign_worker: 0 (non-foreign) is privileged (national origin bias)

Metrics computed:
  Disparate Impact (DI), Demographic Parity Difference (DPD),
  Equal Opportunity Difference (EOD), Average Odds Difference (AOD),
  Theil Index

Usage (from german_credit_dataset/):
  python scripts/compute_fairness.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── configuration ──────────────────────────────────────────────────────
TARGET = "credit risk"
FAVORABLE_LABEL = 1
UNFAVORABLE_LABEL = 2

PROTECTED_CONFIGS: dict[str, dict] = {
    "Sex": {
        "column": "Sex_original",
        "privileged": "male",
        "unprivileged_label": "female",
    },
    "AgeGroup": {
        "column": "AgeGroup_original",
        "privileged": "40_plus",
        "unprivileged_label": "under_40",
    },
    "foreign_worker": {
        "column": "foreign_worker_original",
        "privileged": 0,
        "unprivileged_label": 1,
    },
}


# ── manual metric computation ─────────────────────────────────────────
def _manual_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    group: pd.Series,
    priv_value,
) -> dict:
    mask_p = group == priv_value
    mask_u = group != priv_value

    def sel(m):
        return float((y_pred[m] == FAVORABLE_LABEL).mean()) if m.sum() else np.nan

    def tpr(m):
        d = (y_true[m] == FAVORABLE_LABEL).sum()
        return (
            float(
                ((y_true[m] == FAVORABLE_LABEL) & (y_pred[m] == FAVORABLE_LABEL)).sum()
                / d
            )
            if d
            else np.nan
        )

    def fpr(m):
        d = (y_true[m] == UNFAVORABLE_LABEL).sum()
        return (
            float(
                (
                    (y_true[m] == UNFAVORABLE_LABEL)
                    & (y_pred[m] == FAVORABLE_LABEL)
                ).sum()
                / d
            )
            if d
            else np.nan
        )

    sp, su = sel(mask_p), sel(mask_u)
    tp, tu = tpr(mask_p), tpr(mask_u)
    fp, fu = fpr(mask_p), fpr(mask_u)

    di = (su / sp) if (sp and not np.isnan(sp) and sp > 0) else np.nan
    dpd = float(su - sp) if not (np.isnan(sp) or np.isnan(su)) else np.nan
    eod = float(tu - tp) if not (np.isnan(tp) or np.isnan(tu)) else np.nan
    # Pinned to AIF360 orientation (unprivileged - privileged), matching
    # DI/DPD/EOD above and ClassificationMetric.average_odds_difference().
    aod = (
        float(((fu - fp) + (tu - tp)) / 2.0)
        if not any(np.isnan([fp, fu, tp, tu]))
        else np.nan
    )

    return {
        "DisparateImpact": di,
        "DemographicParityDiff": dpd,
        "EqualOpportunityDiff": eod,
        "AverageOddsDiff": aod,
    }


def _compute_theil_index(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute the Theil Index (generalized entropy with alpha=1)."""
    n = len(y_true)
    if n == 0:
        return np.nan
    benefit = (y_pred == FAVORABLE_LABEL).astype(float)
    b_mean = benefit.mean()
    if b_mean == 0 or b_mean == 1:
        return 0.0
    bi = benefit / b_mean
    bi_safe = bi[bi > 0]
    theil = float((bi_safe * np.log(bi_safe)).mean()) if len(bi_safe) > 0 else 0.0
    return theil


# ── AIF360-backed computation ─────────────────────────────────────────
def compute_aif360_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    group_binary: pd.Series,
    attr_name: str,
) -> dict:
    """Use AIF360 ClassificationMetric. group_binary: 1=privileged, 0=unprivileged."""
    df_true = pd.DataFrame({attr_name: group_binary.values, "label": y_true.values})
    df_pred = pd.DataFrame({attr_name: group_binary.values, "label": y_pred.values})

    bld_true = BinaryLabelDataset(
        df=df_true,
        label_names=["label"],
        protected_attribute_names=[attr_name],
        favorable_label=FAVORABLE_LABEL,
        unfavorable_label=UNFAVORABLE_LABEL,
    )
    bld_pred = BinaryLabelDataset(
        df=df_pred,
        label_names=["label"],
        protected_attribute_names=[attr_name],
        favorable_label=FAVORABLE_LABEL,
        unfavorable_label=UNFAVORABLE_LABEL,
    )

    priv = [{attr_name: 1}]
    unpriv = [{attr_name: 0}]

    ds_metric = BinaryLabelDatasetMetric(
        bld_pred, privileged_groups=priv, unprivileged_groups=unpriv
    )
    cls_metric = ClassificationMetric(
        bld_true, bld_pred, privileged_groups=priv, unprivileged_groups=unpriv
    )

    return {
        "DisparateImpact": float(ds_metric.disparate_impact()),
        "DemographicParityDiff": float(ds_metric.mean_difference()),
        "EqualOpportunityDiff": float(cls_metric.equal_opportunity_difference()),
        "AverageOddsDiff": float(cls_metric.average_odds_difference()),
    }


# ── main pipeline ─────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="metrics/classification_predictions.csv")
    parser.add_argument("--out_dir", default="metrics/fairness")
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    log.info(f"Loaded {len(df)} predictions")

    y_true = df["actual"].astype(int)
    y_pred = df["predicted"].astype(int)

    results: list[dict] = []
    summary: dict = {}

    for attr_name, cfg in PROTECTED_CONFIGS.items():
        col = cfg["column"]
        priv_val = cfg["privileged"]

        if col not in df.columns:
            log.warning(f"Column '{col}' missing — skipping {attr_name}")
            continue

        series = df[col]
        group_binary = (series == priv_val).astype(int)

        log.info(
            f"  {attr_name}: privileged={priv_val}, "
            f"priv_n={group_binary.sum()}, unpriv_n={(group_binary == 0).sum()}"
        )

        # Compute metrics
        if AIF360_AVAILABLE:
            try:
                m = compute_aif360_metrics(y_true, y_pred, group_binary, attr_name)
            except Exception as e:
                log.warning(f"AIF360 failed for {attr_name}: {e}; using manual")
                m = _manual_metrics(y_true, y_pred, series, priv_val)
        else:
            m = _manual_metrics(y_true, y_pred, series, priv_val)

        theil = _compute_theil_index(y_true, y_pred)

        di = m["DisparateImpact"]
        bias_flag = bool(di < args.threshold) if not np.isnan(di) else False

        row = {
            "Attribute": attr_name,
            "PrivilegedValue": str(priv_val),
            "DisparateImpact": round(di, 4) if not np.isnan(di) else None,
            "DemographicParityDiff": round(m["DemographicParityDiff"], 4)
            if not np.isnan(m["DemographicParityDiff"])
            else None,
            "EqualOpportunityDiff": round(m["EqualOpportunityDiff"], 4)
            if not np.isnan(m["EqualOpportunityDiff"])
            else None,
            "AverageOddsDiff": round(m["AverageOddsDiff"], 4)
            if not np.isnan(m["AverageOddsDiff"])
            else None,
            "TheilIndex": round(theil, 4) if not np.isnan(theil) else None,
            "BiasFlag": bias_flag,
        }
        results.append(row)
        summary[attr_name] = {
            "DisparateImpact": row["DisparateImpact"],
            "TheilIndex": row["TheilIndex"],
            "BiasFlag": bias_flag,
        }

        flag = "BIAS" if bias_flag else "OK"
        log.info(f"    DI={row['DisparateImpact']}  DPD={row['DemographicParityDiff']}  "
                 f"EOD={row['EqualOpportunityDiff']}  AOD={row['AverageOddsDiff']}  "
                 f"Theil={row['TheilIndex']}  [{flag}]")

    results_df = pd.DataFrame(results)

    # ── export artifacts ───────────────────────────────────────────────
    results_df.to_csv(out_dir / "fairness_metrics.csv", index=False)
    with (out_dir / "fairness_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # ── markdown report ────────────────────────────────────────────────
    lines = [
        "# Fairness Report: German Credit",
        "",
        f"**Records analysed:** {len(df)}",
        f"**DI bias threshold:** {args.threshold}",
        f"**Favorable label:** {FAVORABLE_LABEL} (good credit)",
        "",
        "## Protected Attributes (domain-knowledge privileged groups)",
        "",
        "| Attribute | Privileged | Rationale |",
        "|---|---|---|",
        "| Sex | male | Historically advantaged in credit markets |",
        "| AgeGroup | 40_plus | Established credit history, higher income |",
        "| foreign_worker | 0 (non-foreign) | National-origin discrimination risk |",
        "",
        "## Fairness Metrics",
        "",
        "| Attribute | Privileged | DI | DPD | EOD | AOD | Theil | Bias? |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for _, r in results_df.iterrows():
        mark = "YES" if r["BiasFlag"] else "no"
        lines.append(
            f"| {r['Attribute']} | {r['PrivilegedValue']} | "
            f"{r['DisparateImpact'] if pd.notna(r['DisparateImpact']) else ''} | "
            f"{r['DemographicParityDiff'] if pd.notna(r['DemographicParityDiff']) else ''} | "
            f"{r['EqualOpportunityDiff'] if pd.notna(r['EqualOpportunityDiff']) else ''} | "
            f"{r['AverageOddsDiff'] if pd.notna(r['AverageOddsDiff']) else ''} | "
            f"{r['TheilIndex'] if pd.notna(r['TheilIndex']) else ''} | "
            f"{mark} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **Disparate Impact < 0.8** signals potential bias (80 % rule).")
    lines.append("- **DPD / EOD / AOD** closer to 0 is fairer.")
    lines.append("- **Theil Index** closer to 0 means more equal benefit distribution.")
    lines.append("")

    with (out_dir / "report.md").open("w") as f:
        f.write("\n".join(lines) + "\n")

    # ── plots ──────────────────────────────────────────────────────────
    if not results_df.empty:
        attrs = results_df["Attribute"].tolist()
        di_vals = [
            float(v) if pd.notna(v) else 0 for v in results_df["DisparateImpact"]
        ]
        plt.figure(figsize=(7, 4))
        colors = ["#d63031" if v < args.threshold else "#00b894" for v in di_vals]
        bars = plt.bar(attrs, di_vals, color=colors)
        plt.axhline(args.threshold, color="red", ls="--", lw=1, label=f"Threshold {args.threshold}")
        plt.axhline(1.0, color="gray", ls=":", lw=1, label="Parity (1.0)")
        plt.ylabel("Disparate Impact")
        plt.title("Disparate Impact by Protected Attribute — German Credit")
        plt.legend()
        for bar, val in zip(bars, di_vals):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(plots_dir / "di_by_attribute.png", dpi=150)
        plt.close()

    log.info(f"Fairness report saved to {out_dir}")

    # ── qualitative analysis ───────────────────────────────────────
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
        from qualitative_analysis import run_qualitative_analysis

        run_qualitative_analysis(
            predictions_path=args.predictions,
            fairness_csv_path=out_dir / "fairness_metrics.csv",
            target_col="actual",
            pred_col="predicted",
            favorable_label=FAVORABLE_LABEL,
            protected_attrs=["Sex_original", "AgeGroup_original", "foreign_worker_original"],
            out_dir=out_dir,
            dataset_name="German Credit",
            threshold=args.threshold,
        )
    except Exception as e:
        log.warning(f"Qualitative analysis failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
