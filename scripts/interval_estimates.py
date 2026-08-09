#!/usr/bin/env python3
"""
interval_estimates.py

Wilson score intervals on subgroup selection rates, propagated to disparate
impact, for every intersectional cell.

This exists because a disparate impact ratio is a point estimate and the audit
reported it to four decimal places regardless of whether it rested on 31 people
or 4,019. Two consequences follow, and both change what the audit may claim:

  * Ranking small cells is not supported. American Indian x Male has the lowest
    point estimate in the HMDA data (0.6734) but a 95% interval of roughly
    [0.45, 0.90], which spans the four-fifths screen. It cannot be called the
    worst-treated group, and it cannot be assigned a severity band.
  * The precisely estimated disparities are the large ones. Black x Female
    (n=1,994) sits at 0.8632 with an interval of about [0.81, 0.94] -- entirely
    below parity. That is the finding the audit can actually defend, and a
    point-estimate ranking pushes it to sixth place behind five cells whose
    ordering is noise.

The interval on the ratio is deliberately conservative: the disadvantaged cell's
lower bound over the reference cell's upper bound, and vice versa. This ignores
the correlation between the two estimates and is therefore wider than a delta
method or bootstrap interval would be. For an audit that is the right direction
to err -- it makes the claim harder to assert, not easier.

Every table is produced under three reference-group conventions, because the
choice of denominator moves every ratio and nothing in a bare DI records which
convention was in force:

  max_rate_floor   highest-rate cell with n >= floor (the pipeline's rule)
  max_rate         highest-rate cell, no floor
  control          a designated control cell (White x Male for HMDA;
                   40_plus x male for German Credit)

Usage:
  python3.11 scripts/interval_estimates.py
  python3.11 scripts/interval_estimates.py --dataset hmda
  python3.11 scripts/interval_estimates.py --emit-tex   # appendix tables
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "artifacts" / "consolidated" / "interval_estimates.json"

# Matches classify_severity() in scripts/qualitative_analysis.py.
FOUR_FIFTHS = 0.80
CRITICAL = 0.72

CASES = {
    "german_credit": {
        "dir": "german_credit_dataset",
        "attrs": ("AgeGroup_original", "Sex_original"),
        "favorable": 1,
        "control_cell": "40_plus x male",
    },
    "hmda": {
        "dir": "hmda_dataset",
        "attrs": ("race", "sex"),
        "favorable": 1,
        "control_cell": "White x Male",
    },
}

CONVENTIONS = ("max_rate_floor", "max_rate", "control")
DEFAULT_CONVENTION = "max_rate_floor"
GENERATED = REPO_ROOT / "report" / "generated"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Correct at small n, where the normal approximation
    produces bounds outside [0, 1] and understates uncertainty."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cell_table(cfg: dict, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    a, b = cfg["attrs"]
    fav = cfg["favorable"]
    g = (
        df.assign(_fav=(df["predicted"] == fav).astype(int))
        .groupby([a, b])
        .agg(n=("_fav", "size"), k=("_fav", "sum"))
        .reset_index()
    )
    g["rate"] = g.k / g.n
    g["cell"] = g[a].astype(str) + " x " + g[b].astype(str)
    return g


def analyse(dataset: str, cfg: dict, floor: int, convention: str = DEFAULT_CONVENTION) -> dict | None:
    path = REPO_ROOT / cfg["dir"] / "metrics" / "classification_predictions.csv"
    if not path.exists():
        return None
    g = cell_table(cfg, path)

    if convention == "control":
        match = g[g.cell == cfg["control_cell"]]
        if match.empty:
            return None
        ref = match.iloc[0]
    elif convention == "max_rate":
        ref = g.loc[g.rate.idxmax()]
    else:  # max_rate_floor -- the pipeline's convention
        sized = g[g.n >= floor]
        if sized.empty:
            return None
        ref = sized.loc[sized.rate.idxmax()]
    ref_lo, ref_hi = wilson(int(ref.k), int(ref.n))

    cells = []
    for _, r in g.sort_values("rate").iterrows():
        lo, hi = wilson(int(r.k), int(r.n))
        di = r.rate / ref.rate if ref.rate else float("nan")
        di_lo = lo / ref_hi if ref_hi else float("nan")
        di_hi = hi / ref_lo if ref_lo else float("nan")
        cells.append(
            {
                "cell": r.cell,
                "n": int(r.n),
                "selection_rate": round(float(r.rate), 4),
                "di": round(float(di), 4),
                "di_ci_low": round(float(di_lo), 4),
                "di_ci_high": round(float(di_hi), 4),
                # A severity band is only assignable if the whole interval sits
                # inside one band. Otherwise the audit does not know the band.
                "band_determinate": bool(di_hi < CRITICAL or di_lo >= FOUR_FIFTHS
                                         or (di_lo >= CRITICAL and di_hi < FOUR_FIFTHS)),
                "interval_spans_four_fifths": bool(di_lo < FOUR_FIFTHS <= di_hi),
                "below_floor": bool(r.n < floor),
            }
        )

    determinate = [c for c in cells if c["band_determinate"]]
    return {
        "reference_cell": ref.cell,
        "reference_n": int(ref.n),
        "reference_convention": convention,
        "floor": floor,
        "cells": cells,
        "n_cells": len(cells),
        "n_band_determinate": len(determinate),
        "defensible_disparities": [
            c["cell"] for c in cells
            if c["band_determinate"] and c["di_ci_high"] < 1.0
        ],
    }


def emit_tex(name: str, convention: str, res: dict) -> Path:
    rows = []
    for c in res["cells"]:
        det = "yes" if c["band_determinate"] else "--"
        cell = c["cell"].replace("_", "\\_")
        rows.append(
            f"  {cell} & {c['n']:,} & {c['selection_rate']:.4f} & "
            f"{c['di']:.4f} & [{c['di_ci_low']:.3f}, {c['di_ci_high']:.3f}] & {det} \\\\"
        )
    body = "\n".join(rows)
    ref = res["reference_cell"]
    tex = (
        "% Generated by scripts/interval_estimates.py -- do not edit.\n"
        f"% dataset {name}; convention {convention}; reference {ref} "
        f"(n={res['reference_n']}); floor {res['floor']}.\n"
        "\\begin{tabular}{lrrrlc}\n"
        "  \\toprule\n"
        "  cell & $n$ & rate & DI & 95\\% CI & band det. \\\\\n"
        "  \\midrule\n"
        f"{body}\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / f"appendix_a_{name}_{convention}.tex"
    path.write_text(tex)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(CASES) + ["all"], default="all")
    ap.add_argument("--floor", type=int, default=15)
    ap.add_argument("--emit-tex", action="store_true",
                    help="write appendix tables to report/generated/")
    args = ap.parse_args()

    targets = CASES if args.dataset == "all" else {args.dataset: CASES[args.dataset]}
    report = {}
    for name, cfg in targets.items():
        conventions = {}
        for convention in CONVENTIONS:
            res = analyse(name, cfg, args.floor, convention)
            if res is not None:
                conventions[convention] = res
        if DEFAULT_CONVENTION not in conventions:
            print(f"skip {name}: predictions not found")
            continue
        # The default convention stays at the top level (the pipeline's rule);
        # the full set sits under "conventions".
        report[name] = dict(conventions[DEFAULT_CONVENTION])
        report[name]["conventions"] = conventions

        res = conventions[DEFAULT_CONVENTION]
        print(f"\n=== {name}  (reference: {res['reference_cell']}, n={res['reference_n']}) ===")
        print(f"{'cell':32s} {'n':>6s} {'DI':>8s}   {'95% CI':>16s}  band")
        print("-" * 78)
        for c in res["cells"]:
            band = "determinate" if c["band_determinate"] else "NOT DETERMINATE"
            print(f"{c['cell']:32s} {c['n']:6d} {c['di']:8.4f}   "
                  f"[{c['di_ci_low']:.3f}, {c['di_ci_high']:.3f}]  {band}")
        print(f"\n  {res['n_band_determinate']}/{res['n_cells']} cells have a determinate severity band.")
        print(f"  Defensible disparities (interval entirely below parity): "
              f"{res['defensible_disparities'] or 'none'}")
        others = {c: r["reference_cell"] for c, r in conventions.items()}
        print(f"  Reference by convention: {others}")

        if args.emit_tex:
            for convention, r in conventions.items():
                path = emit_tex(name, convention, r)
                print(f"  wrote {path.relative_to(REPO_ROOT)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
