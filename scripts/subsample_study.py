#!/usr/bin/env python3.11
"""Subsampling study: what does an audit of size m conclude, and how often is it wrong?

Treats the full-population HMDA race x sex table (n=10,978, out-of-fold
predictions) as ground truth. For each audit size m, draws B simple random
subsamples without replacement, runs five reporting policies under three
reference-group conventions on each draw, and records how often each policy
misleads:

  policies     floor15   suppress cells with n < 15, point estimates otherwise
               floor30   suppress cells with n < 30
               point     no floor, point estimates for every observed cell
               wilson    no floor; conservative DI interval per cell; a verdict
                         is asserted only when the interval clears or fails the
                         0.80 screen entirely (spanning it = indeterminate)
               eb        empirical-Bayes shrinkage of cell rates toward the
                         pooled rate (beta prior by method of moments), point
                         estimates, no floor

  conventions  max_rate         reference = highest-rate observed cell
               max_rate_floor15 reference = highest-rate cell with n >= 15
                                (the pipeline's convention; falls back to
                                max_rate when no cell reaches the floor)
               control          reference = White x Male, fixed

  events (per draw, relative to the population DI under the same convention)
    sign_inversion    an asserted cell sits on the opposite side of parity
    false_clearance   a population-violating cell (DI < 0.80) asserted clear
    false_alarm       a population-clearing cell asserted violating
    worst_wrong       the asserted worst cell is not the population worst
    worst_abstain     the policy declines to name a worst cell
    suppressed        cells (of 12) not reported at all
    coverage          wilson only: DI interval contains the population DI

The DI interval reuses interval_estimates.py's deliberately conservative form
(cell lower bound / reference upper bound); it over-covers by construction,
which the coverage column makes visible rather than hides.

Usage:
  python3.11 scripts/subsample_study.py                  # full run (seeded)
  python3.11 scripts/subsample_study.py --quick          # small B, for tests
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = REPO_ROOT / "hmda_dataset" / "metrics" / "classification_predictions.csv"
OUT_JSON = REPO_ROOT / "artifacts" / "consolidated" / "subsample_study.json"
OUT_MD = REPO_ROOT / "artifacts" / "consolidated" / "subsample_study.md"
GENERATED = REPO_ROOT / "report" / "generated"

FOUR_FIFTHS = 0.80
DEFAULT_SIZES = (500, 1000, 2196, 4000, 8000)
DEFAULT_B = 2000
DEFAULT_SEED = 20260808
BODY_M = 2196  # the audit size the original test-split audit actually used

POLICIES = ("floor15", "floor30", "point", "wilson", "eb")
CONVENTIONS = ("max_rate", "max_rate_floor15", "control")
PRIMARY_CONVENTION = "max_rate_floor15"
CONTROL_CELL = ("White", "Male")

POLICY_LABELS = {
    "floor15": "floor n>=15",
    "floor30": "floor n>=30",
    "point": "point, no floor",
    "wilson": "Wilson interval",
    "eb": "EB shrinkage",
}


def wilson_bounds(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Wilson score interval; NaN where n == 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = np.clip(centre - half, 0.0, 1.0)
    hi = np.clip(centre + half, 0.0, 1.0)
    lo[n == 0] = np.nan
    hi[n == 0] = np.nan
    return lo, hi


def rate_ci(successes: int, trials: int) -> dict:
    lo, hi = wilson_bounds(np.array([float(successes)]), np.array([float(trials)]))
    return {
        "rate": round(successes / trials, 4),
        "ci_low": round(float(lo[0]), 4),
        "ci_high": round(float(hi[0]), 4),
    }


def load_population() -> tuple[np.ndarray, np.ndarray, list[str], int]:
    df = pd.read_csv(PREDICTIONS)
    labels = sorted(
        f"{r} x {s}" for r, s in df.groupby(["race", "sex"]).size().index
    )
    code_of = {lab: i for i, lab in enumerate(labels)}
    codes = np.array(
        [code_of[f"{r} x {s}"] for r, s in zip(df["race"], df["sex"])], dtype=np.int64
    )
    fav = (df["predicted"] == 1).to_numpy(dtype=np.float64)
    control_idx = code_of[f"{CONTROL_CELL[0]} x {CONTROL_CELL[1]}"]
    return codes, fav, labels, control_idx


def cell_counts(codes: np.ndarray, fav: np.ndarray, idx: np.ndarray, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    sub = codes[idx]
    n = np.bincount(sub, minlength=n_cells).astype(np.float64)
    k = np.bincount(sub, weights=fav[idx], minlength=n_cells)
    return k, n


def pick_reference(p: np.ndarray, n: np.ndarray, convention: str, control_idx: int) -> np.ndarray:
    """Reference cell index per draw. p, n are [B, C]."""
    if convention == "control":
        return np.full(p.shape[0], control_idx, dtype=np.int64)
    masked = np.where(n >= 1, p, -np.inf)
    ref = np.argmax(masked, axis=1)
    if convention == "max_rate_floor15":
        floored = np.where(n >= 15, p, -np.inf)
        has_floored = np.isfinite(floored).any(axis=1) & (floored.max(axis=1) > -np.inf)
        ref = np.where(has_floored, np.argmax(floored, axis=1), ref)
    return ref


def eb_shrink(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Empirical-Bayes cell rates: beta prior with mean = pooled rate and
    strength tau from the weighted (Kleinman) beta-binomial moment estimator
    of the intraclass correlation rho; tau = (1 - rho) / rho."""
    total_n = n.sum(axis=1)
    pooled = k.sum(axis=1) / total_n  # [B]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
    observed = n >= 1
    c = observed.sum(axis=1)  # observed cells per draw
    pq = pooled * (1 - pooled)
    s = np.where(observed, n * (p - pooled[:, None]) ** 2, 0.0).sum(axis=1)
    denom = pq * (total_n - (n**2).sum(axis=1) / total_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (s - pq * (c - 1)) / denom
    rho = np.clip(np.nan_to_num(rho, nan=1e-4), 1e-4, 0.99)
    tau = (1 - rho) / rho  # prior strength
    shrunk = (k + (pooled * tau)[:, None]) / (n + tau[:, None])
    shrunk[~observed] = np.nan
    return shrunk


def population_truth(codes: np.ndarray, fav: np.ndarray, labels: list[str], control_idx: int) -> dict:
    n_cells = len(labels)
    k, n = cell_counts(codes, fav, np.arange(codes.size), n_cells)
    p = k / n
    truth = {}
    for convention in CONVENTIONS:
        ref = int(pick_reference(p[None, :], n[None, :], convention, control_idx)[0])
        di = p / p[ref]
        non_ref = np.arange(n_cells) != ref
        violating = non_ref & (di < FOUR_FIFTHS)
        truth[convention] = {
            "reference_cell": labels[ref],
            "reference_idx": ref,
            "di": {labels[i]: round(float(di[i]), 4) for i in range(n_cells)},
            "violating_cells": [labels[i] for i in np.where(violating)[0]],
            "worst_cell": labels[int(np.argmin(di))],
            "worst_idx": int(np.argmin(di)),
        }
    return truth


def assertions_for_policy(
    policy: str,
    k: np.ndarray,
    n: np.ndarray,
    ref: np.ndarray,
    ref_from: np.ndarray,
) -> dict:
    """What the policy asserts on each draw.

    Returns dict with [B, C] arrays: `di` (NaN where nothing is asserted for
    ranking), `asserted` (a clear/violate verdict exists), `clear`, `violate`,
    `reported`, and for wilson also `di_lo`/`di_hi`/`indeterminate`.
    """
    B, C = k.shape
    rows = np.arange(B)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
    est = ref_from  # rate estimates the policy ranks with ([B, C])
    ref_rate = est[rows, ref]
    with np.errstate(divide="ignore", invalid="ignore"):
        di = est / ref_rate[:, None]

    observed = n >= 1
    if policy == "floor15":
        reported = n >= 15
    elif policy == "floor30":
        reported = n >= 30
    else:
        reported = observed

    out = {"reported": reported, "di": np.where(reported, di, np.nan)}
    if policy == "wilson":
        lo, hi = wilson_bounds(k, n)
        ref_lo = lo[rows, ref]
        ref_hi = hi[rows, ref]
        with np.errstate(divide="ignore", invalid="ignore"):
            di_lo = lo / ref_hi[:, None]
            di_hi = hi / ref_lo[:, None]
        clear = reported & (di_lo >= FOUR_FIFTHS)
        violate = reported & (di_hi < FOUR_FIFTHS)
        out.update(
            di_lo=di_lo,
            di_hi=di_hi,
            clear=clear,
            violate=violate,
            asserted=clear | violate,
            indeterminate=reported & ~(clear | violate),
        )
    else:
        clear = reported & (out["di"] >= FOUR_FIFTHS)
        violate = reported & (out["di"] < FOUR_FIFTHS)
        out.update(clear=clear, violate=violate, asserted=reported)
    return out


def summarise_draws(
    policy: str,
    assertion: dict,
    truth_conv: dict,
    ref: np.ndarray,
    n_cells: int,
) -> dict:
    B = ref.shape[0]
    rows = np.arange(B)
    di_theta = np.array([truth_conv["di"][lab] for lab in truth_conv["_labels"]])
    theta_ref = truth_conv["reference_idx"]
    theta_worst = truth_conv["worst_idx"]
    violating = np.zeros(n_cells, dtype=bool)
    for lab in truth_conv["violating_cells"]:
        violating[truth_conv["_labels"].index(lab)] = True
    clearing = ~violating
    clearing[theta_ref] = False  # the reference itself is not an alarm candidate

    asserted = assertion["asserted"].copy()
    asserted[rows, ref] = False  # the sample reference (DI=1) asserts nothing

    di = assertion["di"]
    # Sign inversion: asserted estimate on the wrong side of parity.
    with np.errstate(invalid="ignore"):
        inverted = asserted & (di_theta[None, :] != 1.0) & ((di - 1.0) * (di_theta[None, :] - 1.0) < 0)
    sign_inversion = inverted.any(axis=1)

    false_clearance = (assertion["clear"] & violating[None, :]).any(axis=1)
    fc_worst = assertion["clear"][:, theta_worst]
    false_alarm = (assertion["violate"] & clearing[None, :]).any(axis=1)

    # Worst-cell identification among cells the policy is willing to rank.
    rankable = assertion["violate"] if policy == "wilson" else asserted
    di_rank = np.where(rankable, di, np.inf)
    has_rankable = rankable.any(axis=1)
    named_worst = np.argmin(di_rank, axis=1)
    worst_wrong = has_rankable & (named_worst != theta_worst)
    worst_abstain = ~has_rankable

    suppressed = n_cells - assertion["reported"].sum(axis=1)

    result = {
        "sign_inversion": rate_ci(int(sign_inversion.sum()), B),
        "false_clearance": rate_ci(int(false_clearance.sum()), B),
        "false_clearance_worst_cell": rate_ci(int(fc_worst.sum()), B),
        "false_alarm": rate_ci(int(false_alarm.sum()), B),
        "worst_cell_wrong": rate_ci(int(worst_wrong.sum()), B),
        "worst_cell_abstain": rate_ci(int(worst_abstain.sum()), B),
        "mean_suppressed_cells": round(float(suppressed.mean()), 4),
    }
    if policy == "wilson":
        contains = (
            (assertion["di_lo"] <= di_theta[None, :])
            & (di_theta[None, :] <= assertion["di_hi"])
            & assertion["reported"]
        )
        contains[rows, ref] = False
        reported_nonref = assertion["reported"].copy()
        reported_nonref[rows, ref] = False
        per_draw = contains.sum(axis=1) / np.maximum(reported_nonref.sum(axis=1), 1)
        result["coverage"] = round(float(per_draw.mean()), 4)
        result["mean_indeterminate_cells"] = round(
            float(assertion["indeterminate"].sum(axis=1).mean()), 4
        )
    return result


def run_study(sizes, B, seed) -> dict:
    codes, fav, labels, control_idx = load_population()
    n_cells = len(labels)
    truth = population_truth(codes, fav, labels, control_idx)
    for conv in truth.values():
        conv["_labels"] = labels

    rng = np.random.default_rng(seed)
    results: dict = {c: {p: {} for p in POLICIES} for c in CONVENTIONS}
    for m in sizes:
        K = np.empty((B, n_cells))
        N = np.empty((B, n_cells))
        for b in range(B):
            idx = rng.choice(codes.size, size=m, replace=False)
            K[b], N[b] = cell_counts(codes, fav, idx, n_cells)
        with np.errstate(divide="ignore", invalid="ignore"):
            P = K / N
        shrunk = eb_shrink(K, N)
        for convention in CONVENTIONS:
            for policy in POLICIES:
                est = shrunk if policy == "eb" else P
                ref = pick_reference(est, N, convention, control_idx)
                assertion = assertions_for_policy(policy, K, N, ref, est)
                results[convention][policy][str(m)] = summarise_draws(
                    policy, assertion, truth[convention], ref, n_cells
                )

    for conv in truth.values():
        del conv["_labels"]
    return {
        "generator": "scripts/subsample_study.py",
        "population_file": str(PREDICTIONS.relative_to(REPO_ROOT)),
        "population_n": int(codes.size),
        "sizes": list(sizes),
        "draws_per_size": B,
        "seed": seed,
        "sampling": "simple random, without replacement",
        "four_fifths_screen": FOUR_FIFTHS,
        "primary_convention": PRIMARY_CONVENTION,
        "eb_estimator": (
            "beta prior, mean = pooled sample rate, strength tau = (1-rho)/rho "
            "with rho from the weighted (Kleinman) beta-binomial moment "
            "estimator of the intraclass correlation, rho clipped to "
            "[1e-4, 0.99]"
        ),
        "di_interval": (
            "conservative: cell Wilson bound over opposite reference Wilson "
            "bound, as in scripts/interval_estimates.py; over-covers by design"
        ),
        "theta": truth,
        "results": results,
    }


def render_body_table(study: dict) -> str:
    rows = []
    res = study["results"][PRIMARY_CONVENTION]
    for policy in POLICIES:
        r = res[policy][str(BODY_M)]
        def pct(metric):
            v = r[metric]
            return f"{100*v['rate']:.1f} [{100*v['ci_low']:.1f}, {100*v['ci_high']:.1f}]"
        abstain = r["worst_cell_abstain"]["rate"]
        label = POLICY_LABELS[policy].replace(">=", "$\\geq$")
        rows.append(
            f"  {label} & "
            f"{pct('false_clearance')} & {pct('worst_cell_wrong')} & "
            f"{100*abstain:.1f} & {pct('sign_inversion')} & "
            f"{pct('false_alarm')} & {r['mean_suppressed_cells']:.1f} \\\\"
        )
    body = "\n".join(rows)
    return (
        "% Generated by scripts/subsample_study.py -- do not edit.\n"
        f"% m = {BODY_M}, B = {study['draws_per_size']}, convention = {PRIMARY_CONVENTION}.\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "  \\toprule\n"
        "  & false clearance & worst cell wrong & abstains & sign inversion & false alarm & cells \\\\\n"
        "  policy & \\% of draws & \\% of draws & \\% & \\% of draws & \\% of draws & suppressed \\\\\n"
        "  \\midrule\n"
        f"{body}\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )


def render_grid_table(study: dict) -> str:
    """One tabular per reference convention, so no single table exceeds a
    column's height (a plain tabular cannot break across pages)."""
    lines = [
        "% Generated by scripts/subsample_study.py -- do not edit.",
        "% Full grid: percent of draws.",
    ]
    for convention in CONVENTIONS:
        pretty_conv = convention.replace("_", " ")
        lines += [
            f"\\subsubsection*{{Reference convention: {pretty_conv}}}",
            "\\begin{tabular}{lrrrrrr}",
            "  \\toprule",
            "  policy & $m$ & false clr. & worst wrong & abstain & sign inv. & false alarm \\\\",
            "  \\midrule",
        ]
        for policy in POLICIES:
            for m in study["sizes"]:
                r = study["results"][convention][policy][str(m)]
                def pct(metric):
                    v = r[metric]
                    return f"{100*v['rate']:.1f}"
                pretty = POLICY_LABELS[policy].replace(">=", "$\\geq$")
                label = pretty if m == study["sizes"][0] else ""
                lines.append(
                    f"  {label} & {m} & {pct('false_clearance')} & "
                    f"{pct('worst_cell_wrong')} & {100*r['worst_cell_abstain']['rate']:.1f} & "
                    f"{pct('sign_inversion')} & {pct('false_alarm')} \\\\"
                )
        lines += ["  \\bottomrule", "\\end{tabular}", "\\par\\medskip", ""]
    return "\n".join(lines)


def render_figure(study: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Validated categorical palette (dataviz reference, slots 1-5, light mode);
    # low-contrast slots are relieved by direct labels and distinct markers.
    colors = {
        "floor15": "#2a78d6",
        "floor30": "#eb6834",
        "point": "#1baf7a",
        "wilson": "#eda100",
        "eb": "#e87ba4",
    }
    markers = {"floor15": "o", "floor30": "s", "point": "^", "wilson": "D", "eb": "v"}
    sizes = study["sizes"]
    res = study["results"][PRIMARY_CONVENTION]
    panels = [
        ("false_clearance", "False clearance"),
        ("worst_cell_wrong", "Worst cell misidentified"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, (metric, title) in zip(axes, panels):
        for policy in POLICIES:
            ys = [100 * res[policy][str(m)][metric]["rate"] for m in sizes]
            los = [100 * res[policy][str(m)][metric]["ci_low"] for m in sizes]
            his = [100 * res[policy][str(m)][metric]["ci_high"] for m in sizes]
            ax.plot(
                sizes, ys,
                color=colors[policy], marker=markers[policy], markersize=4,
                linewidth=1.6, label=POLICY_LABELS[policy],
                clip_on=False, zorder=3,
            )
            ax.fill_between(sizes, los, his, color=colors[policy], alpha=0.12, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(m) for m in sizes], fontsize=7)
        ax.minorticks_off()
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("audit size $m$", fontsize=8)
        ax.tick_params(labelsize=7, length=2.5)
        ax.grid(axis="y", color="#e6e6e3", linewidth=0.7, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c9c8c2")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("% of draws", fontsize=8)
    # Direct labels at the right edge of the second panel (relief for the
    # low-contrast slots), plus a compact legend on the first.
    axes[0].legend(fontsize=6.5, frameon=False, handlelength=1.6, loc="center left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_md(study: dict) -> str:
    lines = [
        "# Subsampling study",
        "",
        f"Population: `{study['population_file']}` (n={study['population_n']}); "
        f"B={study['draws_per_size']} SRS draws per size; seed {study['seed']}.",
        "",
        "## Population ground truth per convention",
        "",
    ]
    for convention, t in study["theta"].items():
        lines.append(
            f"- **{convention}**: reference {t['reference_cell']}; "
            f"worst {t['worst_cell']}; violating: {', '.join(t['violating_cells'])}"
        )
    lines += [
        "",
        f"## Rates at m={BODY_M} (primary convention: {PRIMARY_CONVENTION}), % of draws",
        "",
        "| policy | false clearance | worst cell wrong | abstains | sign inversion | false alarm | suppressed cells | coverage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    res = study["results"][PRIMARY_CONVENTION]
    for policy in POLICIES:
        r = res[policy][str(BODY_M)]
        def pct(metric):
            v = r[metric]
            return f"{100*v['rate']:.1f} [{100*v['ci_low']:.1f}, {100*v['ci_high']:.1f}]"
        cov = r.get("coverage")
        lines.append(
            f"| {POLICY_LABELS[policy]} | {pct('false_clearance')} | {pct('worst_cell_wrong')} | "
            f"{100*r['worst_cell_abstain']['rate']:.1f} | {pct('sign_inversion')} | {pct('false_alarm')} | "
            f"{r['mean_suppressed_cells']:.1f} | {cov if cov is not None else '--'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    ap.add_argument("--draws", type=int, default=DEFAULT_B)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--quick", action="store_true", help="B=50 for fast checks")
    ap.add_argument("--no-artifacts", action="store_true", help="print only")
    args = ap.parse_args()
    B = 50 if args.quick else args.draws

    study = run_study(tuple(args.sizes), B, args.seed)

    if args.no_artifacts:
        print(json.dumps(study["results"][PRIMARY_CONVENTION], indent=2))
        return 0

    GENERATED.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(study, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text(render_md(study))
    (GENERATED / "subsample_body_table.tex").write_text(render_body_table(study))
    (GENERATED / "subsample_full_grid.tex").write_text(render_grid_table(study))
    render_figure(study, GENERATED / "subsample_rates.pdf")
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}, {OUT_MD.relative_to(REPO_ROOT)}, "
          f"and 3 files under {GENERATED.relative_to(REPO_ROOT)}/")
    print()
    print(render_md(study))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
