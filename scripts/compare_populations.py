#!/usr/bin/env python3
"""
compare_populations.py

Quantify what auditing a test split costs, relative to auditing the whole
cleaned population.

This exists because the project's headline intersectional findings turned out to
be substantially artifacts of sample size:

  * HMDA race x sex on the test split suppressed five cells under the n floor.
    Every one of them had a worse disparate impact than the worst *reported*
    cell, so the audit named the wrong group as most disadvantaged and
    understated the severity band.
  * German Credit's `under_40 x female` scored DI 0.6087 (CRITICAL) against a
    comparator of 15 people with a saturated 15/15 selection rate. At full
    population the same comparison is 0.8201, which clears the four-fifths
    screen.

Both prediction files are written by the dataset train_models.py scripts:
`classification_predictions.csv` (full population, out-of-fold) and
`classification_predictions_testsplit.csv`. Nothing here recomputes a model.

Usage:
  python3.11 scripts/compare_populations.py
  python3.11 scripts/compare_populations.py --floor 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "artifacts" / "consolidated" / "population_comparison.json"

# Default floor matches scripts/intersectional_drilldown.py.
DEFAULT_FLOOR = 15

CASES = {
    "german_credit": {
        "dir": "german_credit_dataset",
        "attrs": ("AgeGroup_original", "Sex_original"),
        "favorable": 1,
        "pred_col": "predicted",
        "label": "UC1 German Credit — AgeGroup x Sex",
    },
    "hmda": {
        "dir": "hmda_dataset",
        "attrs": ("race", "sex"),
        "favorable": 1,
        "pred_col": "predicted",
        "label": "UC2 HMDA — race x sex",
    },
}


def cells(df: pd.DataFrame, attrs: tuple[str, str], favorable: int,
          pred_col: str, floor: int) -> pd.DataFrame:
    """Selection rate and DI per intersectional cell, mirroring the drilldown."""
    a, b = attrs
    grp = df.groupby([a, b], dropna=True)
    out = grp.apply(
        lambda g: pd.Series(
            {"n": len(g), "selection_rate": float((g[pred_col] == favorable).mean())}
        ),
        include_groups=False,
    ).reset_index()
    out["n"] = out["n"].astype(int)
    out["small_subgroup"] = out["n"] < floor

    sized = out[~out["small_subgroup"]]
    ref = sized["selection_rate"].max() if len(sized) else None
    out["di_vs_best"] = (out["selection_rate"] / ref) if ref else None
    return out.sort_values("di_vs_best").reset_index(drop=True)


def summarise(tab: pd.DataFrame) -> dict:
    visible = tab[~tab["small_subgroup"]]
    hidden = tab[tab["small_subgroup"]]
    worst_visible = visible.iloc[0] if len(visible) else None
    worst_any = tab.iloc[0] if len(tab) else None
    return {
        "n_rows": int(tab["n"].sum()),
        "cells_total": int(len(tab)),
        "cells_suppressed": int(len(hidden)),
        "worst_visible_di": float(worst_visible["di_vs_best"]) if worst_visible is not None else None,
        "worst_visible_cell": (
            " x ".join(str(worst_visible[c]) for c in tab.columns[:2]) if worst_visible is not None else None
        ),
        "worst_visible_n": int(worst_visible["n"]) if worst_visible is not None else None,
        "worst_actual_di": float(worst_any["di_vs_best"]) if worst_any is not None else None,
        "worst_actual_cell": (
            " x ".join(str(worst_any[c]) for c in tab.columns[:2]) if worst_any is not None else None
        ),
        "suppressed_worse_than_reported": int(
            (hidden["di_vs_best"] < worst_visible["di_vs_best"]).sum()
        ) if worst_visible is not None and len(hidden) else 0,
    }


def band(di: float | None) -> str:
    """Severity band from scripts/qualitative_analysis.py classify_severity."""
    if di is None:
        return "UNKNOWN"
    if di < 0.72:
        return "CRITICAL"
    if di < 0.80:
        return "HIGH"
    if di < 0.95:
        return "MODERATE"
    return "LOW"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--floor", type=int, default=DEFAULT_FLOOR)
    args = ap.parse_args()

    report: dict = {"floor": args.floor, "cases": {}}
    lines: list[str] = []

    for key, cfg in CASES.items():
        base = REPO_ROOT / cfg["dir"] / "metrics"
        full_p = base / "classification_predictions.csv"
        split_p = base / "classification_predictions_testsplit.csv"
        if not full_p.exists() or not split_p.exists():
            print(f"skip {key}: run train_models.py --full-population first")
            continue

        tabs = {}
        for name, path in (("test_split", split_p), ("full_population", full_p)):
            df = pd.read_csv(path)
            tabs[name] = cells(df, cfg["attrs"], cfg["favorable"], cfg["pred_col"], args.floor)

        report["cases"][key] = {
            "label": cfg["label"],
            "test_split": summarise(tabs["test_split"]),
            "full_population": summarise(tabs["full_population"]),
        }

        lines.append(f"\n### {cfg['label']}\n")
        for name in ("test_split", "full_population"):
            s = report["cases"][key][name]
            lines.append(
                f"**{name.replace('_',' ')}** — n={s['n_rows']}, "
                f"{s['cells_suppressed']}/{s['cells_total']} cells suppressed at n<{args.floor}. "
                f"Worst *reported* {s['worst_visible_cell']} DI {s['worst_visible_di']:.4f} "
                f"({band(s['worst_visible_di'])}, n={s['worst_visible_n']}); "
                f"worst *actual* {s['worst_actual_cell']} DI {s['worst_actual_di']:.4f} "
                f"({band(s['worst_actual_di'])})."
            )
            if s["suppressed_worse_than_reported"]:
                lines.append(
                    f"  → {s['suppressed_worse_than_reported']} suppressed cell(s) are worse "
                    f"than the worst cell the audit reported."
                )
        lines.append("")
        lines.append(tabs["full_population"].to_string(index=False))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
