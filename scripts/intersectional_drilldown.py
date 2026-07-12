"""
intersectional_drilldown.py

Stage 1 deterministic drill-down for the lending use cases
(docs/RESEARCH_STAGING_PROMPT.md §2). No LLM involvement.

For each registered lending use case this script:
  1. reads the predictions CSV (with decoded protected attributes) and the
     fairness metrics CSV,
  2. builds per-attribute group tables (n, selection rate) — surfacing
     small-subgroup uncertainty (Q1.2, Q3.1),
  3. builds intersectional subgroup tables for the configured attribute
     pairs (Q1.1, Q2.1, Q3.2): n, selection rate, and disparate impact
     relative to the most-favored subgroup with n >= MIN_SUBGROUP_N,
  4. writes CSVs into <dataset>/metrics/fairness/ and mirrors them into
     artifacts/<dataset>/fairness/,
  5. writes the consolidated lending-lifecycle baseline to
     artifacts/consolidated/lending_lifecycle_baseline.md.

Usage (from repo root):
  python3 scripts/intersectional_drilldown.py
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from qualitative_analysis import classify_severity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
MIN_SUBGROUP_N = 15

USE_CASES: dict[str, dict] = {
    "german_credit": {
        "use_case": "UC1 — Consumer installment-credit scoring",
        "dir": "german_credit_dataset",
        "predictions": "metrics/classification_predictions.csv",
        "fairness_csv": "metrics/fairness/fairness_metrics.csv",
        "target": "actual",
        "pred": "predicted",
        "favorable_label": 1,
        "attrs": ["Sex_original", "AgeGroup_original", "foreign_worker_original"],
        "pairs": [("AgeGroup_original", "Sex_original")],
    },
    "hmda": {
        "use_case": "UC2 — Mortgage underwriting (HMDA Georgia)",
        "dir": "hmda_dataset",
        "predictions": "metrics/classification_predictions.csv",
        "fairness_csv": "metrics/fairness/fairness_metrics.csv",
        "target": "actual",
        "pred": "predicted",
        "favorable_label": 1,
        "attrs": ["race", "sex", "age_group"],
        "pairs": [("race", "sex")],
    },
    "lending_club": {
        "use_case": "UC3 — P2P personal-loan default risk (Lending Club)",
        "dir": "lending_club_dataset",
        "predictions": "metrics/classification_predictions_enriched.csv",
        "fairness_csv": "metrics/fairness/fairness_metrics_normalized.csv",
        "target": "actual",
        "pred": "predicted",
        "favorable_label": 0,
        "attrs": ["gender", "income_level", "loan_amount_level"],
        "pairs": [("income_level", "loan_amount_level"), ("gender", "income_level")],
    },
}


def group_table(df: pd.DataFrame, attrs: list[str], pred: str, favorable: int) -> pd.DataFrame:
    """n and favorable-prediction selection rate per (intersectional) group."""
    grouped = df.groupby(attrs, dropna=False)
    out = grouped.apply(
        lambda g: pd.Series({
            "n": len(g),
            "selection_rate": (g[pred] == favorable).mean(),
        }),
        include_groups=False,
    ).reset_index()
    out["n"] = out["n"].astype(int)
    out["small_subgroup"] = out["n"] < MIN_SUBGROUP_N
    # DI relative to the most-favored adequately-sized subgroup
    sized = out[~out["small_subgroup"]]
    ref = sized["selection_rate"].max() if not sized.empty else out["selection_rate"].max()
    out["disparate_impact_vs_best"] = (out["selection_rate"] / ref) if ref and ref > 0 else None
    return out.sort_values("selection_rate", ascending=False).reset_index(drop=True)


def fmt_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "--")
    if "small_subgroup" in show.columns:
        show["small_subgroup"] = show["small_subgroup"].map(lambda b: "yes (n<15)" if b == True else "")  # noqa: E712
    header = "| " + " | ".join(show.columns) + " |"
    sep = "|" + "|".join("---" for _ in show.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in show.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def per_attribute_summary(fairness_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in fairness_df.iterrows():
        di = r.get("DisparateImpact")
        rows.append({
            "Attribute": r.get("Attribute"),
            "PrivilegedValue": r.get("PrivilegedValue"),
            "DisparateImpact": di,
            "DemographicParityDiff": r.get("DemographicParityDiff"),
            "EqualOpportunityDiff": r.get("EqualOpportunityDiff"),
            "AverageOddsDiff": r.get("AverageOddsDiff"),
            "TheilIndex": r.get("TheilIndex"),
            "Severity": classify_severity(float(di) if pd.notna(di) else None),
        })
    return pd.DataFrame(rows)


def main() -> None:
    cons_lines = [
        "# Lending-Lifecycle Deterministic Baseline (Stage 1)",
        "",
        "Deterministic drill-down across the three lending use cases "
        "(docs/RESEARCH_STAGING_PROMPT.md §2). Selection rate = share of the "
        "group receiving the favorable prediction. Subgroups with n < "
        f"{MIN_SUBGROUP_N} are reported but flagged as too small to interpret. "
        "Disparate impact is computed against the most-favored adequately-sized "
        "subgroup. No LLM was involved in producing this file.",
        "",
    ]

    for key, cfg in USE_CASES.items():
        ds_dir = ROOT / cfg["dir"]
        preds_path = ds_dir / cfg["predictions"]
        fairness_path = ds_dir / cfg["fairness_csv"]
        if not preds_path.exists() or not fairness_path.exists():
            log.warning(f"[{key}] missing inputs ({preds_path.name}, {fairness_path.name}); skipping")
            continue

        df = pd.read_csv(preds_path)
        fairness_df = pd.read_csv(fairness_path)
        out_dir = ds_dir / "metrics" / "fairness"
        art_dir = ARTIFACTS / key / "fairness"
        art_dir.mkdir(parents=True, exist_ok=True)

        cons_lines += [f"## {cfg['use_case']}", ""]

        # ── per-attribute metric summary with severity ─────────────────
        summary = per_attribute_summary(fairness_df)
        cons_lines += ["### Per-attribute fairness metrics", "",
                       fmt_table(summary), ""]

        # ── per-attribute group sizes (small-subgroup caveats) ─────────
        for attr in cfg["attrs"]:
            if attr not in df.columns:
                log.warning(f"[{key}] attribute '{attr}' not in predictions; skipping")
                continue
            table = group_table(df, [attr], cfg["pred"], cfg["favorable_label"])
            csv_path = out_dir / f"drilldown_{attr}.csv"
            table.to_csv(csv_path, index=False)
            shutil.copy2(csv_path, art_dir / csv_path.name)
            small = table[table["small_subgroup"]]
            cons_lines += [f"### Groups: `{attr}`", "", fmt_table(table), ""]
            if not small.empty:
                groups = ", ".join(str(v) for v in small[attr])
                cons_lines += [
                    f"> **Small-subgroup caveat:** {groups} (n < {MIN_SUBGROUP_N}) — "
                    "metrics for these groups carry high variance and must not be "
                    "interpreted as stable disparities.", "",
                ]

        # ── intersectional pairs ───────────────────────────────────────
        for a, b in cfg["pairs"]:
            if a not in df.columns or b not in df.columns:
                log.warning(f"[{key}] pair ({a}, {b}) not available; skipping")
                continue
            table = group_table(df, [a, b], cfg["pred"], cfg["favorable_label"])
            csv_path = out_dir / f"intersectional_{a}_x_{b}.csv"
            table.to_csv(csv_path, index=False)
            shutil.copy2(csv_path, art_dir / csv_path.name)
            log.info(f"[{key}] intersectional {a} × {b} -> {csv_path}")

            sized = table[~table["small_subgroup"]]
            cons_lines += [f"### Intersectional: `{a}` × `{b}`", "", fmt_table(table), ""]
            if len(sized) >= 2:
                worst = sized.iloc[-1]
                best = sized.iloc[0]
                gap = best["selection_rate"] - worst["selection_rate"]
                cons_lines += [
                    f"> Largest adequately-sized gap: {tuple(best[[a, b]])} at "
                    f"{best['selection_rate']:.4f} vs {tuple(worst[[a, b]])} at "
                    f"{worst['selection_rate']:.4f} (gap {gap:.4f}, intersectional "
                    f"DI {worst['disparate_impact_vs_best']:.4f}).", "",
                ]

        cons_lines += ["---", ""]

    cons_dir = ARTIFACTS / "consolidated"
    cons_dir.mkdir(parents=True, exist_ok=True)
    out_md = cons_dir / "lending_lifecycle_baseline.md"
    out_md.write_text("\n".join(cons_lines), encoding="utf-8")
    log.info(f"Consolidated baseline -> {out_md}")


if __name__ == "__main__":
    main()
