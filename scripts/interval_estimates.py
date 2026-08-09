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

Usage:
  python3.11 scripts/interval_estimates.py
  python3.11 scripts/interval_estimates.py --dataset hmda --attrs race sex
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
    },
    "hmda": {"dir": "hmda_dataset", "attrs": ("race", "sex"), "favorable": 1},
}


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


def analyse(dataset: str, cfg: dict, floor: int) -> dict | None:
    path = REPO_ROOT / cfg["dir"] / "metrics" / "classification_predictions.csv"
    if not path.exists():
        return None
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
                "cell": f"{r[a]} x {r[b]}",
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
        "reference_cell": f"{ref[a]} x {ref[b]}",
        "reference_n": int(ref.n),
        "floor": floor,
        "cells": cells,
        "n_cells": len(cells),
        "n_band_determinate": len(determinate),
        "defensible_disparities": [
            c["cell"] for c in cells
            if c["band_determinate"] and c["di_ci_high"] < 1.0
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(CASES) + ["all"], default="all")
    ap.add_argument("--floor", type=int, default=15)
    args = ap.parse_args()

    targets = CASES if args.dataset == "all" else {args.dataset: CASES[args.dataset]}
    report = {}
    for name, cfg in targets.items():
        res = analyse(name, cfg, args.floor)
        if res is None:
            print(f"skip {name}: predictions not found")
            continue
        report[name] = res
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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
