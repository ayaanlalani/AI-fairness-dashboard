"""
compute_fairness.py  (HMDA)

Evaluate algorithmic fairness for the HMDA mortgage-approval classifier.

Protected attributes and privileged groups (domain-knowledge based):
  race:      White is privileged  (historically advantaged in mortgage lending)
  sex:       Male is privileged   (historical gender gap in credit access)
  age_group: mid (35-54) is privileged  (prime earning years, established credit)

Metrics:
  Disparate Impact (DI), Demographic Parity Difference (DPD),
  Equal Opportunity Difference (EOD), Average Odds Difference (AOD),
  Theil Index

Usage (from hmda_dataset/):
  python3 scripts/compute_fairness.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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

FAVORABLE_LABEL = 1
UNFAVORABLE_LABEL = 0

PROTECTED_CONFIGS = {
    "race": {
        "column": "race",
        "privileged": "White",
    },
    "sex": {
        "column": "sex",
        "privileged": "Male",
    },
    "age_group": {
        "column": "age_group",
        "privileged": "mid",
    },
}


# ── manual metric helpers ──────────────────────────────────────────────
def _manual_metrics(y_true, y_pred, group, priv_value) -> dict:
    mask_p = group == priv_value
    mask_u = group != priv_value

    def sel(m):
        return float((y_pred[m] == FAVORABLE_LABEL).mean()) if m.sum() else np.nan

    def tpr(m):
        d = (y_true[m] == FAVORABLE_LABEL).sum()
        return float(((y_true[m] == FAVORABLE_LABEL) & (y_pred[m] == FAVORABLE_LABEL)).sum() / d) if d else np.nan

    def fpr(m):
        d = (y_true[m] == UNFAVORABLE_LABEL).sum()
        return float(((y_true[m] == UNFAVORABLE_LABEL) & (y_pred[m] == FAVORABLE_LABEL)).sum() / d) if d else np.nan

    sp, su = sel(mask_p), sel(mask_u)
    tp, tu = tpr(mask_p), tpr(mask_u)
    fp, fu = fpr(mask_p), fpr(mask_u)

    di = (su / sp) if (sp and not np.isnan(sp) and sp > 0) else np.nan
    dpd = float(su - sp) if not (np.isnan(sp) or np.isnan(su)) else np.nan
    eod = float(tu - tp) if not (np.isnan(tp) or np.isnan(tu)) else np.nan
    aod = float(((fp - fu) + (tp - tu)) / 2.0) if not any(np.isnan([fp, fu, tp, tu])) else np.nan

    return {"DisparateImpact": di, "DemographicParityDiff": dpd, "EqualOpportunityDiff": eod, "AverageOddsDiff": aod}


def _theil_index(y_pred) -> float:
    benefit = (y_pred == FAVORABLE_LABEL).astype(float)
    b_mean = benefit.mean()
    if b_mean == 0 or b_mean == 1:
        return 0.0
    bi = benefit / b_mean
    bi_safe = bi[bi > 0]
    return float((bi_safe * np.log(bi_safe)).mean()) if len(bi_safe) else 0.0


def _aif360_metrics(y_true, y_pred, group_binary, attr_name) -> dict:
    df_true = pd.DataFrame({attr_name: group_binary.values, "label": y_true.values})
    df_pred = pd.DataFrame({attr_name: group_binary.values, "label": y_pred.values})

    bld_true = BinaryLabelDataset(df=df_true, label_names=["label"],
                                   protected_attribute_names=[attr_name],
                                   favorable_label=1, unfavorable_label=0)
    bld_pred = BinaryLabelDataset(df=df_pred, label_names=["label"],
                                   protected_attribute_names=[attr_name],
                                   favorable_label=1, unfavorable_label=0)

    priv, unpriv = [{attr_name: 1}], [{attr_name: 0}]
    ds = BinaryLabelDatasetMetric(bld_pred, privileged_groups=priv, unprivileged_groups=unpriv)
    cls = ClassificationMetric(bld_true, bld_pred, privileged_groups=priv, unprivileged_groups=unpriv)

    return {
        "DisparateImpact": float(ds.disparate_impact()),
        "DemographicParityDiff": float(ds.mean_difference()),
        "EqualOpportunityDiff": float(cls.equal_opportunity_difference()),
        "AverageOddsDiff": float(cls.average_odds_difference()),
    }


# ── main ───────────────────────────────────────────────────────────────
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

        n_priv = int(group_binary.sum())
        n_unpriv = int((group_binary == 0).sum())
        log.info(f"  {attr_name}: privileged={priv_val} (n={n_priv}), unprivileged (n={n_unpriv})")

        if AIF360_AVAILABLE:
            try:
                m = _aif360_metrics(y_true, y_pred, group_binary, attr_name)
            except Exception as e:
                log.warning(f"AIF360 failed for {attr_name}: {e}; manual fallback")
                m = _manual_metrics(y_true, y_pred, series, priv_val)
        else:
            m = _manual_metrics(y_true, y_pred, series, priv_val)

        theil = _theil_index(y_pred)
        di = m["DisparateImpact"]
        bias_flag = bool(di < args.threshold) if not np.isnan(di) else False

        row = {
            "Attribute": attr_name,
            "PrivilegedValue": str(priv_val),
            "DisparateImpact": round(di, 4) if not np.isnan(di) else None,
            "DemographicParityDiff": round(m["DemographicParityDiff"], 4) if not np.isnan(m["DemographicParityDiff"]) else None,
            "EqualOpportunityDiff": round(m["EqualOpportunityDiff"], 4) if not np.isnan(m["EqualOpportunityDiff"]) else None,
            "AverageOddsDiff": round(m["AverageOddsDiff"], 4) if not np.isnan(m["AverageOddsDiff"]) else None,
            "TheilIndex": round(theil, 4) if not np.isnan(theil) else None,
            "BiasFlag": bias_flag,
        }
        results.append(row)
        summary[attr_name] = {"DisparateImpact": row["DisparateImpact"], "TheilIndex": row["TheilIndex"], "BiasFlag": bias_flag}

        flag = "BIAS" if bias_flag else "OK"
        log.info(f"    DI={row['DisparateImpact']}  DPD={row['DemographicParityDiff']}  "
                 f"EOD={row['EqualOpportunityDiff']}  AOD={row['AverageOddsDiff']}  "
                 f"Theil={row['TheilIndex']}  [{flag}]")

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "fairness_metrics.csv", index=False)
    with (out_dir / "fairness_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # ── markdown report ────────────────────────────────────────────
    lines = [
        "# Fairness Report: HMDA Mortgage Lending",
        "",
        f"**Records analysed:** {len(df)}",
        f"**DI bias threshold:** {args.threshold}",
        f"**Favorable label:** {FAVORABLE_LABEL} (loan originated / approved)",
        "",
        "## Protected Attributes (legally protected under ECOA & Fair Housing Act)",
        "",
        "| Attribute | Privileged | Legal basis |",
        "|---|---|---|",
        "| race | White | ECOA, Fair Housing Act |",
        "| sex | Male | ECOA, Fair Housing Act |",
        "| age_group | mid (35-54) | ECOA (age >= 40 protected) |",
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
    lines += [
        "",
        "## Interpretation",
        "",
        "- **Disparate Impact < 0.8** signals potential bias (80% rule / four-fifths rule).",
        "- **DPD / EOD / AOD** closer to 0 is fairer.",
        "- **Theil Index** closer to 0 means more equal benefit distribution.",
        "- HMDA data contains real demographic attributes collected under federal law,",
        "  making this dataset the regulatory standard for fair-lending analysis.",
        "",
    ]

    with (out_dir / "report.md").open("w") as f:
        f.write("\n".join(lines) + "\n")

    # ── plots ──────────────────────────────────────────────────────
    if not results_df.empty:
        attrs = results_df["Attribute"].tolist()
        di_vals = [float(v) if pd.notna(v) else 0 for v in results_df["DisparateImpact"]]

        plt.figure(figsize=(7, 4))
        colors = ["#d63031" if v < args.threshold else "#00b894" for v in di_vals]
        bars = plt.bar(attrs, di_vals, color=colors)
        plt.axhline(args.threshold, color="red", ls="--", lw=1, label=f"Threshold {args.threshold}")
        plt.axhline(1.0, color="gray", ls=":", lw=1, label="Parity (1.0)")
        plt.ylabel("Disparate Impact")
        plt.title("Disparate Impact by Protected Attribute — HMDA")
        plt.legend()
        for bar, val in zip(bars, di_vals):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                     ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(plots_dir / "di_by_attribute.png", dpi=150)
        plt.close()

        # Per-attribute group breakdown
        for attr_name, cfg in PROTECTED_CONFIGS.items():
            col = cfg["column"]
            if col not in df.columns:
                continue
            groups = df[col].unique()
            sel_rates = {}
            for g in groups:
                mask = df[col] == g
                sel_rates[str(g)] = float((y_pred[mask] == FAVORABLE_LABEL).mean()) if mask.sum() else 0
            plt.figure(figsize=(7, 4))
            gs = list(sel_rates.keys())
            vals = list(sel_rates.values())
            plt.bar(gs, vals, color="#6C5CE7")
            plt.ylabel("Selection Rate P(ŷ=1)")
            plt.title(f"Approval Rate by {attr_name} — HMDA")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(plots_dir / f"selection_rate_{attr_name}.png", dpi=150)
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
            protected_attrs=list(PROTECTED_CONFIGS.keys()),
            out_dir=out_dir,
            dataset_name="HMDA Mortgage Lending (Georgia)",
            threshold=args.threshold,
        )
    except Exception as e:
        log.warning(f"Qualitative analysis failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
