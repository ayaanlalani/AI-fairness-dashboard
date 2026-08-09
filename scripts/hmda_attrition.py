#!/usr/bin/env python3.11
"""Population construction for the HMDA audit, made visible.

`hmda_dataset/clean_hmda.py` reduces 20,000 raw LAR rows to the 10,978 the
audit scores. This script replays the row-dropping steps in their exact order
and emits the attrition table, then recomputes the race disparate-impact table
on raw approval rates with the dropped race categories — `Race Not Available`,
`Joint`, `Free Form Text Only` — retained as categories instead of removed.

The point is disclosure: dropping `Race Not Available` (a fifth of the file)
is selection on the protected attribute, and it happens before any metric
runs. Approval-rate DI here is a property of the data, not of the model — the
dropped rows never receive model predictions, which is exactly why this table
cannot be produced anywhere downstream.

Writes artifacts/consolidated/population_attrition.{json,md} and
report/generated/attrition_table.tex + race_not_available_di.tex.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "hmda_dataset" / "data" / "hmda_raw.csv"
OUT_JSON = REPO_ROOT / "artifacts" / "consolidated" / "population_attrition.json"
OUT_MD = REPO_ROOT / "artifacts" / "consolidated" / "population_attrition.md"
GENERATED = REPO_ROOT / "report" / "generated"

sys.path.insert(0, str(REPO_ROOT / "hmda_dataset"))
from clean_hmda import AGE_BUCKETS, RACE_MAP, SEX_KEEP  # noqa: E402


def replay_attrition(df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """Replay clean_hmda.load_and_clean()'s row-dropping steps in order."""
    steps = [{"step": "raw file", "rule": "-", "dropped": 0, "remaining": len(df)}]

    race = df["derived_race"].map(RACE_MAP)
    dropped_by_cat = (
        df.loc[race.isna(), "derived_race"].value_counts().to_dict()
    )
    kept = race.notna()
    steps.append(
        {
            "step": "race",
            "rule": "derived_race not in RACE_MAP or mapped to None",
            "dropped": int((~kept).sum()),
            "remaining": int(kept.sum()),
            "dropped_by_category": {k: int(v) for k, v in dropped_by_cat.items()},
        }
    )
    df = df[kept].copy()
    df["race"] = race[kept]

    kept = df["derived_sex"].isin(SEX_KEEP)
    steps.append(
        {
            "step": "sex",
            "rule": "derived_sex not in {Male, Female}",
            "dropped": int((~kept).sum()),
            "remaining": int(kept.sum()),
            "dropped_by_category": {
                k: int(v)
                for k, v in df.loc[~kept, "derived_sex"].value_counts().items()
            },
        }
    )
    df = df[kept].copy()

    age = df["applicant_age"].map(AGE_BUCKETS)
    kept = age.notna()
    steps.append(
        {
            "step": "age",
            "rule": "applicant_age not a mapped band",
            "dropped": int((~kept).sum()),
            "remaining": int(kept.sum()),
            "dropped_by_category": {
                k: int(v)
                for k, v in df.loc[~kept, "applicant_age"].value_counts().items()
            },
        }
    )
    df = df[kept].copy()
    return steps, df


def race_di_with_dropped_categories(df: pd.DataFrame) -> dict:
    """Raw-file approval-rate DI by race, dropped categories retained."""
    label = df["derived_race"].map(lambda r: RACE_MAP.get(r) or r)
    approved = (df["action_taken"] == 1).astype(int)
    table = (
        pd.DataFrame({"race": label, "approved": approved})
        .groupby("race")
        .agg(n=("approved", "size"), rate=("approved", "mean"))
        .sort_values("rate", ascending=False)
    )
    white_rate = table.loc["White", "rate"]
    best_rate = table["rate"].max()
    rows = []
    for race, r in table.iterrows():
        rows.append(
            {
                "race": race,
                "retained_by_pipeline": race in set(RACE_MAP[k] for k in RACE_MAP if RACE_MAP[k]),
                "n": int(r.n),
                "approval_rate": round(float(r.rate), 4),
                "di_vs_white": round(float(r.rate / white_rate), 4),
                "di_vs_best": round(float(r.rate / best_rate), 4),
            }
        )
    return {
        "outcome": "raw approval (action_taken == 1), not model predictions",
        "population": "all 20,000 raw rows",
        "rows": rows,
    }


def render_md(report: dict) -> str:
    lines = [
        "# HMDA population construction (attrition)",
        "",
        f"Raw file: `{report['raw_file']}`. Steps replayed from "
        "`hmda_dataset/clean_hmda.py::load_and_clean` in order.",
        "",
        "| step | rule | dropped | remaining |",
        "|---|---|---|---|",
    ]
    for s in report["attrition"]:
        lines.append(
            f"| {s['step']} | {s['rule']} | {s['dropped']} | {s['remaining']} |"
        )
    lines += ["", "Dropped by category:", ""]
    for s in report["attrition"]:
        for cat, cnt in (s.get("dropped_by_category") or {}).items():
            lines.append(f"- {s['step']}: {cat} — {cnt}")
    lines += [
        "",
        "## Race DI on raw approval rates, dropped categories retained",
        "",
        "| race | in pipeline | n | approval rate | DI vs White | DI vs best |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["race_di_with_dropped_categories"]["rows"]:
        lines.append(
            f"| {r['race']} | {'yes' if r['retained_by_pipeline'] else 'DROPPED'} | {r['n']} | "
            f"{r['approval_rate']} | {r['di_vs_white']} | {r['di_vs_best']} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_attrition_tex(report: dict) -> str:
    rows = []
    for s in report["attrition"]:
        cats = s.get("dropped_by_category") or {}
        detail = "; ".join(f"{k} ({v})" for k, v in cats.items()) or "--"
        detail = detail.replace("&", "\\&")
        rows.append(
            f"  {s['step']} & {detail} & {s['dropped']:,} & {s['remaining']:,} \\\\"
        )
    body = "\n".join(rows)
    return (
        "% Generated by scripts/hmda_attrition.py -- do not edit.\n"
        "\\begin{tabular}{llrr}\n"
        "  \\toprule\n"
        "  step & categories dropped (count) & dropped & remaining \\\\\n"
        "  \\midrule\n"
        f"{body}\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )


def render_di_tex(report: dict) -> str:
    rows = []
    for r in report["race_di_with_dropped_categories"]["rows"]:
        kept = "" if r["retained_by_pipeline"] else " (dropped)"
        rows.append(
            f"  {r['race']}{kept} & {r['n']:,} & {r['approval_rate']:.4f} & "
            f"{r['di_vs_white']:.4f} & {r['di_vs_best']:.4f} \\\\"
        )
    body = "\n".join(rows)
    return (
        "% Generated by scripts/hmda_attrition.py -- do not edit.\n"
        "% Raw approval rates over all 20,000 rows; not model predictions.\n"
        "\\begin{tabular}{lrrrr}\n"
        "  \\toprule\n"
        "  race & $n$ & approval rate & DI vs White & DI vs best \\\\\n"
        "  \\midrule\n"
        f"{body}\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-artifacts", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(RAW, low_memory=False)
    steps, cleaned = replay_attrition(df)
    report = {
        "generator": "scripts/hmda_attrition.py",
        "raw_file": str(RAW.relative_to(REPO_ROOT)),
        "raw_rows": len(df),
        "final_rows": len(cleaned),
        "attrition": steps,
        "race_di_with_dropped_categories": race_di_with_dropped_categories(df),
    }

    md = render_md(report)
    if args.no_artifacts:
        print(md)
        return 0

    GENERATED.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    OUT_MD.write_text(md)
    (GENERATED / "attrition_table.tex").write_text(render_attrition_tex(report))
    (GENERATED / "race_not_available_di.tex").write_text(render_di_tex(report))
    print(md)
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}, {OUT_MD.relative_to(REPO_ROOT)}, "
          f"and 2 tables under {GENERATED.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
