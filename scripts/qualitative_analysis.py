"""
qualitative_analysis.py

Deterministic qualitative fairness analysis module.

Takes fairness metrics + model predictions and produces a structured
"what is wrong / why / how to fix" report for each protected attribute.

Analyses performed:
  1. Per-group breakdown    -- approval/selection rates, TPR, FPR per group
  2. Data imbalance         -- group sizes, class balance within groups
  3. Proxy feature detection -- correlation between features and protected attrs
  4. Root cause + mitigation -- maps metric patterns to causes and AIF360 fixes

Usage (standalone):
  python3 scripts/qualitative_analysis.py \
    --predictions metrics/classification_predictions.csv \
    --fairness_csv metrics/fairness/fairness_metrics.csv \
    --target approved --favorable_label 1 \
    --protected_attrs race,sex,age_group \
    --out_dir metrics/fairness

Or import and call from another script:
  from qualitative_analysis import run_qualitative_analysis
  run_qualitative_analysis(predictions_df, fairness_df, config)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  1.  Per-group breakdown
# ═══════════════════════════════════════════════════════════════════════

def per_group_breakdown(
    df: pd.DataFrame,
    attr: str,
    target_col: str,
    pred_col: str,
    favorable_label: int,
) -> pd.DataFrame:
    """Compute selection rate, TPR, FPR, group size for every group value."""
    rows = []
    for group_val, grp in df.groupby(attr, dropna=False):
        n = len(grp)
        y_true = grp[target_col]
        y_pred = grp[pred_col]

        sel_rate = float((y_pred == favorable_label).mean())

        pos = (y_true == favorable_label).sum()
        neg = n - pos
        tp = ((y_true == favorable_label) & (y_pred == favorable_label)).sum()
        fp = ((y_true != favorable_label) & (y_pred == favorable_label)).sum()

        tpr = float(tp / pos) if pos > 0 else np.nan
        fpr = float(fp / neg) if neg > 0 else np.nan

        base_rate = float(pos / n) if n > 0 else np.nan

        rows.append({
            "group": str(group_val),
            "n": n,
            "pct_of_total": round(n / len(df) * 100, 1),
            "base_rate": round(base_rate, 4) if not np.isnan(base_rate) else None,
            "selection_rate": round(sel_rate, 4),
            "tpr": round(tpr, 4) if not np.isnan(tpr) else None,
            "fpr": round(fpr, 4) if not np.isnan(fpr) else None,
        })

    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
#  2.  Data imbalance diagnosis
# ═══════════════════════════════════════════════════════════════════════

MIN_GROUP_SIZE = 30  # below this, metrics are unreliable

def diagnose_imbalance(breakdown: pd.DataFrame, attr: str) -> list[str]:
    """Return list of plain-language imbalance findings."""
    findings: list[str] = []

    total = breakdown["n"].sum()
    largest = breakdown.iloc[0]
    smallest = breakdown.iloc[-1]

    ratio = largest["n"] / smallest["n"] if smallest["n"] > 0 else float("inf")
    if ratio > 10:
        findings.append(
            f"Severe size imbalance: {largest['group']} has {largest['n']} records "
            f"vs {smallest['group']} with {smallest['n']} ({ratio:.0f}x ratio). "
            f"Metrics for the smaller group have wide confidence intervals."
        )
    elif ratio > 3:
        findings.append(
            f"Moderate size imbalance: {largest['group']} ({largest['n']}) is "
            f"{ratio:.1f}x larger than {smallest['group']} ({smallest['n']})."
        )

    small_groups = breakdown[breakdown["n"] < MIN_GROUP_SIZE]
    if len(small_groups) > 0:
        names = ", ".join(small_groups["group"].tolist())
        findings.append(
            f"Groups with fewer than {MIN_GROUP_SIZE} records (unreliable metrics): {names}."
        )

    base_rates = breakdown.dropna(subset=["base_rate"])
    if len(base_rates) >= 2:
        br_max = base_rates["base_rate"].max()
        br_min = base_rates["base_rate"].min()
        gap = br_max - br_min
        if gap > 0.15:
            row_max = base_rates.loc[base_rates["base_rate"].idxmax()]
            row_min = base_rates.loc[base_rates["base_rate"].idxmin()]
            findings.append(
                f"Base-rate disparity in the ground truth: "
                f"{row_max['group']} has {row_max['base_rate']:.1%} favorable outcomes "
                f"vs {row_min['group']} at {row_min['base_rate']:.1%} "
                f"(gap = {gap:.1%}). The model may be correctly learning a real "
                f"disparity in historical outcomes, but this historical pattern "
                f"itself may reflect systemic bias."
            )

    return findings


# ═══════════════════════════════════════════════════════════════════════
#  3.  Proxy feature detection
# ═══════════════════════════════════════════════════════════════════════

def detect_proxy_features(
    df: pd.DataFrame,
    attr: str,
    feature_cols: list[str],
) -> list[dict]:
    """
    Compute correlation between each numeric feature and the protected
    attribute.  For binary attrs uses point-biserial; for multi-class
    uses eta-squared (ANOVA-like ratio of between-group variance).

    Returns list of {feature, correlation, strength} sorted descending.
    """
    unique_vals = df[attr].dropna().unique()
    results: list[dict] = []

    if len(unique_vals) <= 1:
        return results

    numeric_features = [c for c in feature_cols if df[c].dtype in ("float64", "float32", "int64", "int32")]

    if len(unique_vals) == 2:
        # Point-biserial: encode attr as 0/1
        binary = pd.Categorical(df[attr]).codes
        for feat in numeric_features:
            vals = df[feat].dropna()
            aligned_bin = binary[vals.index]
            if len(vals) < 10:
                continue
            try:
                r = float(np.corrcoef(vals, aligned_bin)[0, 1])
            except Exception:
                continue
            if np.isnan(r):
                continue
            abs_r = abs(r)
            strength = _corr_strength(abs_r)
            if abs_r >= 0.15:
                results.append({"feature": feat, "correlation": round(r, 3), "strength": strength})
    else:
        # Eta-squared for multi-class
        for feat in numeric_features:
            try:
                grand_mean = df[feat].mean()
                ss_between = 0.0
                ss_total = ((df[feat] - grand_mean) ** 2).sum()
                for _, grp in df.groupby(attr):
                    grp_mean = grp[feat].mean()
                    ss_between += len(grp) * (grp_mean - grand_mean) ** 2
                eta2 = ss_between / ss_total if ss_total > 0 else 0
                r_equiv = float(np.sqrt(eta2))
            except Exception:
                continue
            if np.isnan(r_equiv):
                continue
            strength = _corr_strength(r_equiv)
            if r_equiv >= 0.15:
                results.append({"feature": feat, "correlation": round(r_equiv, 3), "strength": strength})

    results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return results


def _corr_strength(abs_r: float) -> str:
    if abs_r >= 0.5:
        return "strong"
    if abs_r >= 0.3:
        return "moderate"
    if abs_r >= 0.15:
        return "weak"
    return "negligible"


# ═══════════════════════════════════════════════════════════════════════
#  4.  Root cause + mitigation mapping
# ═══════════════════════════════════════════════════════════════════════

def map_root_causes(
    di: float | None,
    dpd: float | None,
    eod: float | None,
    aod: float | None,
    imbalance_findings: list[str],
    proxy_features: list[dict],
    group_breakdown: pd.DataFrame,
    threshold: float = 0.8,
) -> dict[str, list[str]]:
    """
    Deterministic rule engine: examine metric patterns, imbalance, and
    proxy features to produce root-cause hypotheses and mitigation
    recommendations.
    """
    causes: list[str] = []
    fixes: list[str] = []

    di_val = di if di is not None else 1.0
    dpd_val = dpd if dpd is not None else 0.0
    eod_val = eod if eod is not None else 0.0
    aod_val = aod if aod is not None else 0.0

    is_biased = di_val < threshold
    is_borderline = threshold <= di_val < 0.95
    is_fair = di_val >= 0.95

    # ── cause: historical / label bias ─────────────────────────────
    base_rates = group_breakdown.dropna(subset=["base_rate"])
    if len(base_rates) >= 2:
        br_gap = base_rates["base_rate"].max() - base_rates["base_rate"].min()
        if br_gap > 0.10:
            causes.append(
                "Historical / label bias: the ground-truth labels themselves "
                "show a significant disparity across groups. The model is "
                "learning to replicate outcomes that may embed past "
                "discriminatory decisions."
            )
            fixes.append(
                "Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): "
                "assign sample weights that compensate for historical label "
                "imbalance across groups before training."
            )

    # ── cause: proxy features ──────────────────────────────────────
    strong_proxies = [p for p in proxy_features if p["strength"] in ("strong", "moderate")]
    if strong_proxies:
        names = ", ".join(f"{p['feature']} (r={p['correlation']})" for p in strong_proxies[:5])
        causes.append(
            f"Proxy discrimination: features that correlate with the protected "
            f"attribute allow the model to indirectly discriminate even without "
            f"direct access to the attribute. Proxy features detected: {names}."
        )
        fixes.append(
            "Pre-processing -- Disparate Impact Remover "
            "(aif360.algorithms.preprocessing.DisparateImpactRemover): "
            "transform feature distributions to reduce correlation with the "
            "protected attribute while preserving rank-ordering."
        )

    # ── cause: data imbalance ──────────────────────────────────────
    if any("Severe" in f or "fewer than" in f for f in imbalance_findings):
        causes.append(
            "Representation bias: severely underrepresented groups make the "
            "model less reliable for those populations and can amplify "
            "existing disparities."
        )
        fixes.append(
            "Data collection: gather more samples from underrepresented "
            "groups. In the interim, use class-weighted training "
            "(class_weight='balanced' in scikit-learn) to upweight "
            "minority-group errors."
        )

    # ── cause: unequal error rates ─────────────────────────────────
    if abs(eod_val) > 0.05:
        direction = "under-predicting favorable outcomes" if eod_val < 0 else "over-predicting favorable outcomes"
        causes.append(
            f"Unequal opportunity: the model is {direction} for the "
            f"unprivileged group (EOD = {eod_val:+.4f}). Deserving members "
            f"of the unprivileged group are disproportionately missed or "
            f"incorrectly classified."
        )
        fixes.append(
            "In-processing -- Prejudice Remover "
            "(aif360.algorithms.inprocessing.PrejudiceRemover): "
            "add a fairness regularization term during model training "
            "that penalizes dependence on the protected attribute."
        )

    if abs(aod_val) > 0.05:
        causes.append(
            f"Unequal odds: both true-positive and false-positive rates "
            f"differ across groups (AOD = {aod_val:+.4f}), indicating the "
            f"model's errors are systematically distributed along group lines."
        )
        fixes.append(
            "Post-processing -- Equalized Odds "
            "(aif360.algorithms.postprocessing.EqOddsPostprocessing): "
            "adjust per-group classification thresholds after training to "
            "equalize TPR and FPR across groups."
        )

    # ── general mitigation for borderline cases ────────────────────
    if is_borderline and not fixes:
        fixes.append(
            "Post-processing -- Calibrated Equalized Odds "
            "(aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing): "
            "a minimal intervention that adjusts thresholds to bring DI above "
            "the 0.8 threshold while preserving calibration."
        )

    if is_biased and not fixes:
        fixes.append(
            "In-processing -- Adversarial Debiasing "
            "(aif360.algorithms.inprocessing.AdversarialDebiasing): "
            "train with an adversary network that prevents the classifier "
            "from encoding protected-attribute information."
        )

    # ── if everything looks fair ───────────────────────────────────
    if is_fair and not causes:
        causes.append(
            "No significant disparate impact detected. The model treats "
            "groups approximately equally on the measured metrics."
        )
        fixes.append(
            "Continue monitoring: fairness can drift as data distributions "
            "change. Re-run this analysis periodically and after any "
            "model retraining."
        )

    return {"causes": causes, "fixes": fixes}


# ═══════════════════════════════════════════════════════════════════════
#  5.  Severity classification
# ═══════════════════════════════════════════════════════════════════════

def classify_severity(di: float | None, threshold: float = 0.8) -> str:
    if di is None:
        return "UNKNOWN"
    if di < threshold * 0.9:        # e.g. < 0.72
        return "CRITICAL"
    if di < threshold:              # e.g. < 0.80
        return "HIGH"
    if di < 0.95:
        return "MODERATE (borderline)"
    return "LOW (fair)"


# ═══════════════════════════════════════════════════════════════════════
#  Assemble full report
# ═══════════════════════════════════════════════════════════════════════

def generate_qualitative_report(
    predictions_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
    protected_attrs: list[str],
    target_col: str,
    pred_col: str,
    favorable_label: int,
    feature_cols: list[str],
    dataset_name: str = "Dataset",
    threshold: float = 0.8,
) -> str:
    """Build the full qualitative markdown report. Returns markdown string."""

    lines: list[str] = [
        f"# Qualitative Fairness Analysis: {dataset_name}",
        "",
        f"**Records analysed:** {len(predictions_df)}",
        f"**Favorable label:** {favorable_label}",
        f"**Bias threshold (DI):** {threshold}",
        "",
    ]

    # Overall model summary
    overall_sel = (predictions_df[pred_col] == favorable_label).mean()
    lines.append(f"**Overall selection rate:** {overall_sel:.1%}")
    lines.append("")

    for attr in protected_attrs:
        if attr not in predictions_df.columns:
            log.warning(f"Attribute '{attr}' not in predictions; skipping")
            continue

        # Fetch quantitative metrics for this attr
        attr_metrics = _get_attr_metrics(fairness_df, attr)
        di = attr_metrics.get("DisparateImpact")
        dpd = attr_metrics.get("DemographicParityDiff")
        eod = attr_metrics.get("EqualOpportunityDiff")
        aod = attr_metrics.get("AverageOddsDiff")
        theil = attr_metrics.get("TheilIndex")

        severity = classify_severity(di, threshold)

        lines.append("---")
        lines.append(f"## {attr}  (DI = {_fmt(di)}, severity: {severity})")
        lines.append("")

        # ── per-group breakdown ────────────────────────────────────
        breakdown = per_group_breakdown(predictions_df, attr, target_col, pred_col, favorable_label)

        lines.append("### Per-group breakdown")
        lines.append("")
        lines.append("| Group | N | % of total | Base rate | Selection rate | TPR | FPR |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, row in breakdown.iterrows():
            lines.append(
                f"| {row['group']} | {row['n']} | {row['pct_of_total']}% "
                f"| {_fmt(row['base_rate'])} | {_fmt(row['selection_rate'])} "
                f"| {_fmt(row['tpr'])} | {_fmt(row['fpr'])} |"
            )
        lines.append("")

        # ── what is wrong ──────────────────────────────────────────
        lines.append("### What is wrong")
        lines.append("")

        if di is not None and di < 0.95:
            priv_val = attr_metrics.get("PrivilegedValue", "privileged")
            priv_row = breakdown[breakdown["group"] == str(priv_val)]
            priv_rate = priv_row["selection_rate"].iloc[0] if len(priv_row) else None

            worst_row = breakdown.loc[breakdown["selection_rate"].idxmin()]
            best_row = breakdown.loc[breakdown["selection_rate"].idxmax()]

            lines.append(
                f"The model's favorable-outcome rate varies across {attr} groups. "
                f"The highest rate is for **{best_row['group']}** "
                f"({best_row['selection_rate']:.1%}) and the lowest is for "
                f"**{worst_row['group']}** ({worst_row['selection_rate']:.1%}), "
                f"a gap of {abs(best_row['selection_rate'] - worst_row['selection_rate']):.1%}."
            )
            lines.append("")
            if dpd is not None:
                lines.append(
                    f"- Demographic Parity Difference = {dpd:+.4f}: "
                    f"the unprivileged group is selected "
                    f"{'less' if dpd < 0 else 'more'} often."
                )
            if eod is not None:
                lines.append(
                    f"- Equal Opportunity Difference = {eod:+.4f}: "
                    f"among truly deserving candidates, the model "
                    f"{'misses more unprivileged' if eod < 0 else 'catches more unprivileged'} individuals."
                )
            if aod is not None:
                lines.append(
                    f"- Average Odds Difference = {aod:+.4f}: "
                    f"error rates differ systematically across groups."
                )
        elif di is not None:
            lines.append(
                f"No significant disparity detected (DI = {di:.4f}). "
                f"The model treats {attr} groups approximately equally."
            )
        else:
            lines.append("Disparate Impact could not be computed for this attribute.")
        lines.append("")

        # ── data imbalance ─────────────────────────────────────────
        imbalance = diagnose_imbalance(breakdown, attr)

        # ── proxy features ─────────────────────────────────────────
        proxies = detect_proxy_features(predictions_df, attr, feature_cols)

        # ── why is it wrong ────────────────────────────────────────
        root = map_root_causes(di, dpd, eod, aod, imbalance, proxies, breakdown, threshold)

        lines.append("### Why it is wrong")
        lines.append("")

        if imbalance:
            lines.append("**Data imbalance:**")
            for finding in imbalance:
                lines.append(f"- {finding}")
            lines.append("")

        if proxies:
            lines.append("**Proxy features** (features correlated with the protected attribute):")
            for p in proxies[:8]:
                lines.append(f"- `{p['feature']}`: r = {p['correlation']} ({p['strength']})")
            lines.append("")

        if root["causes"]:
            lines.append("**Root causes identified:**")
            for i, cause in enumerate(root["causes"], 1):
                lines.append(f"{i}. {cause}")
        lines.append("")

        # ── how to fix it ──────────────────────────────────────────
        lines.append("### How to fix it")
        lines.append("")
        if root["fixes"]:
            for i, fix in enumerate(root["fixes"], 1):
                lines.append(f"{i}. {fix}")
        else:
            lines.append("No specific mitigations required at this time.")
        lines.append("")

    # ── cross-attribute summary ────────────────────────────────────
    lines.append("---")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Attribute | DI | Severity | Top root cause |")
    lines.append("|---|---:|---|---|")
    for attr in protected_attrs:
        if attr not in predictions_df.columns:
            continue
        m = _get_attr_metrics(fairness_df, attr)
        di = m.get("DisparateImpact")
        sev = classify_severity(di, threshold)

        breakdown = per_group_breakdown(predictions_df, attr, target_col, pred_col, favorable_label)
        imbalance = diagnose_imbalance(breakdown, attr)
        proxies = detect_proxy_features(predictions_df, attr, feature_cols)
        root = map_root_causes(di, m.get("DemographicParityDiff"), m.get("EqualOpportunityDiff"),
                               m.get("AverageOddsDiff"), imbalance, proxies, breakdown, threshold)
        top_cause = root["causes"][0][:80] + "..." if root["causes"] else "None identified"
        lines.append(f"| {attr} | {_fmt(di)} | {sev} | {top_cause} |")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _fmt(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _get_attr_metrics(fairness_df: pd.DataFrame, attr: str) -> dict:
    """Look up a row in the fairness CSV by attribute name.

    Tries exact match first, then strips common suffixes like '_original'
    to handle column naming differences between predictions and fairness CSVs.
    """
    col = "Attribute" if "Attribute" in fairness_df.columns else "protected_attribute"
    candidates = [attr, attr.replace("_original", ""), attr.split("_original")[0]]
    mask = pd.Series([False] * len(fairness_df))
    for candidate in candidates:
        mask = fairness_df[col].astype(str) == candidate
        if mask.any():
            break
    if not mask.any():
        return {}
    row = fairness_df[mask].iloc[0]
    result: dict[str, Any] = {}
    for key in ["DisparateImpact", "DemographicParityDiff", "EqualOpportunityDiff",
                "AverageOddsDiff", "TheilIndex", "PrivilegedValue"]:
        if key in row.index:
            val = row[key]
            if pd.notna(val):
                try:
                    result[key] = float(val)
                except (ValueError, TypeError):
                    result[key] = val
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Public API  (for importing from other scripts)
# ═══════════════════════════════════════════════════════════════════════

def run_qualitative_analysis(
    predictions_path: str | Path,
    fairness_csv_path: str | Path,
    target_col: str,
    pred_col: str,
    favorable_label: int,
    protected_attrs: list[str],
    out_dir: str | Path,
    dataset_name: str = "Dataset",
    threshold: float = 0.8,
) -> Path:
    """Run qualitative analysis and write report. Returns path to report."""
    predictions_df = pd.read_csv(predictions_path)
    fairness_df = pd.read_csv(fairness_csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine feature columns (everything that isn't target, pred, or protected)
    non_feature = {target_col, pred_col, "actual", "predicted"} | set(protected_attrs)
    feature_cols = [c for c in predictions_df.columns if c not in non_feature]

    report = generate_qualitative_report(
        predictions_df=predictions_df,
        fairness_df=fairness_df,
        protected_attrs=protected_attrs,
        target_col=target_col,
        pred_col=pred_col,
        favorable_label=favorable_label,
        feature_cols=feature_cols,
        dataset_name=dataset_name,
        threshold=threshold,
    )

    report_path = out_dir / "qualitative_report.md"
    report_path.write_text(report, encoding="utf-8")
    log.info(f"Qualitative report saved to {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative fairness analysis")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV")
    parser.add_argument("--fairness_csv", required=True, help="Path to fairness_metrics.csv")
    parser.add_argument("--target", required=True, help="Target column name in predictions CSV")
    parser.add_argument("--pred_col", default="predicted", help="Prediction column name")
    parser.add_argument("--favorable_label", type=int, default=1, help="Favorable label value")
    parser.add_argument("--protected_attrs", required=True, help="Comma-separated protected attribute names")
    parser.add_argument("--out_dir", default=".", help="Output directory")
    parser.add_argument("--dataset_name", default="Dataset", help="Name for the report header")
    parser.add_argument("--threshold", type=float, default=0.8, help="DI bias threshold")
    args = parser.parse_args()

    attrs = [a.strip() for a in args.protected_attrs.split(",") if a.strip()]

    run_qualitative_analysis(
        predictions_path=args.predictions,
        fairness_csv_path=args.fairness_csv,
        target_col=args.target,
        pred_col=args.pred_col,
        favorable_label=args.favorable_label,
        protected_attrs=attrs,
        out_dir=args.out_dir,
        dataset_name=args.dataset_name,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
