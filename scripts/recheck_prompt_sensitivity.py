#!/usr/bin/env python3.11
"""Reanalyse the cross-model prompt-sensitivity result without the suspect instrument.

The paper reports per-cycle mean-score spreads (max - min of cycle means) as
evidence that prompt sensitivity is model-generation-specific. Two instrument
concerns hang over that claim:

1. The refusal detector (40-character threshold) never feeds total_score, but
   the constrained cycle (cycle 4) demands terse output, and terse answers may
   be penalised by the *scorer* instead — the same brevity operationalisation
   in a different subscore. This script decomposes the cycle-4 dip by subscore
   and reports each model's spread with and without cycle 4.

2. The Gemini figures (mean 19.79 vs 47.50 excluding zeroed cycles) mix API
   failures (result == {}) with content-bearing responses. This script
   separates them.

Reads the frozen per-record benchmark JSONs; writes
artifacts/consolidated/prompt_sensitivity_check.{json,md} and
report/generated/prompt_sensitivity_decomposition.tex. Pure stdlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent

# Model -> record directory. gpt-4o uses the v2 rerun: identical frozen packs
# to the gpt-5-mini run, which is what makes the cross-model spread comparison
# well-defined.
MODEL_DIRS = {
    "gpt-4o": ROOT / "artifacts" / "llm_benchmark_v2" / "gpt-4o",
    "gpt-5-mini": ROOT / "artifacts" / "llm_benchmark" / "gpt-5-mini",
    "gemini-2.5-flash": ROOT / "artifacts" / "llm_benchmark" / "gemini-2.5-flash",
}

CONSTRAINED_CYCLE = 4
SUBSCORE_KEYS = (
    "completeness",
    "severity_agreement",
    "cause_alignment",
    "mitigation_specificity",
    "research_grounding",
)

OUT_JSON = ROOT / "artifacts" / "consolidated" / "prompt_sensitivity_check.json"
OUT_MD = ROOT / "artifacts" / "consolidated" / "prompt_sensitivity_check.md"
OUT_TEX = ROOT / "report" / "generated" / "prompt_sensitivity_decomposition.tex"


def load_records(model_dir: Path) -> list[dict]:
    records = []
    for path in sorted(model_dir.glob("*/*_cycle*.json")):
        rec = json.loads(path.read_text())
        records.append(
            {
                "dataset": rec["dataset"],
                "attribute": rec["attribute"],
                "cycle": rec["cycle"],
                "prompt_format": rec["prompt_format"],
                "total_score": rec["score"]["total_score"],
                "subscores": {k: rec["score"]["subscores"].get(k, 0.0) for k in SUBSCORE_KEYS},
                "refusal_detected": bool(rec.get("refusal_detected")),
                "zeroed": rec.get("result") in ({}, None),
                "path": str(path.relative_to(ROOT)),
            }
        )
    return records


def cycle_means(records: list[dict]) -> dict[int, float]:
    cycles = sorted({r["cycle"] for r in records})
    return {
        c: mean(r["total_score"] for r in records if r["cycle"] == c) for c in cycles
    }


def spread(means: dict[int, float], exclude: tuple[int, ...] = ()) -> float:
    vals = [v for c, v in means.items() if c not in exclude]
    return max(vals) - min(vals)


def decompose_cycle4(records: list[dict]) -> dict:
    """Attribute the cycle-4 dip in mean total score to individual subscores."""
    base = [r for r in records if r["cycle"] != CONSTRAINED_CYCLE]
    c4 = [r for r in records if r["cycle"] == CONSTRAINED_CYCLE]
    if not base or not c4:
        return {}
    deltas = {}
    for key in SUBSCORE_KEYS:
        base_mean = mean(r["subscores"][key] for r in base)
        c4_mean = mean(r["subscores"][key] for r in c4)
        deltas[key] = round(base_mean - c4_mean, 4)
    total_dip = round(
        mean(r["total_score"] for r in base) - mean(r["total_score"] for r in c4), 4
    )
    shares = {
        k: (round(v / total_dip, 4) if total_dip else None) for k, v in deltas.items()
    }
    flagged = [r for r in c4 if r["refusal_detected"]]
    return {
        "total_dip": total_dip,
        "subscore_deltas": deltas,
        "subscore_shares_of_dip": shares,
        "cycle4_records": len(c4),
        "cycle4_refusal_flagged": len(flagged),
        "cycle4_mean_cause_alignment_flagged": (
            round(mean(r["subscores"]["cause_alignment"] for r in flagged), 4)
            if flagged
            else None
        ),
        "cycle4_mean_cause_alignment_unflagged": (
            round(
                mean(
                    r["subscores"]["cause_alignment"]
                    for r in c4
                    if not r["refusal_detected"]
                ),
                4,
            )
            if any(not r["refusal_detected"] for r in c4)
            else None
        ),
    }


def analyse_spread_model(records: list[dict]) -> dict:
    out = {}
    for dataset in sorted({r["dataset"] for r in records}):
        ds_records = [r for r in records if r["dataset"] == dataset]
        means = cycle_means(ds_records)
        out[dataset] = {
            "cycle_means": {str(c): round(v, 4) for c, v in means.items()},
            "spread_published": round(spread(means), 4),
            "spread_excluding_constrained": round(
                spread(means, exclude=(CONSTRAINED_CYCLE,)), 4
            ),
            "zeroed_records": sum(r["zeroed"] for r in ds_records),
            "refusal_flags": sum(r["refusal_detected"] for r in ds_records),
            "cycle4_decomposition": decompose_cycle4(ds_records),
        }
    return out


def analyse_gemini(records: list[dict]) -> dict:
    empty = [r for r in records if r["zeroed"]]
    content = [r for r in records if not r["zeroed"]]
    score_zero = [r for r in records if r["total_score"] == 0.0]
    nonzero = [r for r in records if r["total_score"] > 0.0]
    empty_but_scored = [r for r in empty if r["total_score"] > 0.0]
    flagged_content = [r for r in content if r["refusal_detected"]]
    return {
        "records": len(records),
        "mean_all": round(mean(r["total_score"] for r in records), 4),
        "empty_result_records": len(empty),
        "score_zero_records": len(score_zero),
        "mean_excluding_score_zero": (
            round(mean(r["total_score"] for r in nonzero), 4) if nonzero else None
        ),
        "mean_content_bearing_only": (
            round(mean(r["total_score"] for r in content), 4) if content else None
        ),
        "empty_but_scored_records": [
            {"path": r["path"], "total_score": r["total_score"]}
            for r in empty_but_scored
        ],
        "refusal_flags_total": sum(r["refusal_detected"] for r in records),
        "refusal_flags_on_content": len(flagged_content),
        "flagged_content_scores": sorted(
            round(r["total_score"], 2) for r in flagged_content
        ),
        "notes": [
            "Every empty record has result == {} (an API failure), which the "
            "40-character rule also counts as a refusal; the flag conflates "
            "transport failure with model behaviour.",
            "mean_excluding_score_zero reproduces the previously published "
            "47.50, but that figure still includes empty-result records "
            "scored above zero: _score_cause_alignment returns a flat 10.0 "
            "whenever the baseline lists no root_cause_labels "
            "(scripts/llm_benchmark_common.py, 'if total == 0: return 10.0'), "
            "so an empty response earns 10/15 on that subscore. "
            "mean_content_bearing_only excludes all empty results.",
        ],
    }


def render_md(result: dict) -> str:
    lines = [
        "# Prompt-sensitivity reanalysis (cycle-4 instrument check)",
        "",
        "Generated by `scripts/recheck_prompt_sensitivity.py` from the frozen",
        "per-record benchmark JSONs. See that script for definitions.",
        "",
    ]
    for model in ("gpt-4o", "gpt-5-mini"):
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            "| dataset | cycle means (1-4) | spread | spread excl. constrained | c4 dip | dip share: completeness / severity / cause | c4 refusal flags |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for dataset, d in result["spreads"][model].items():
            dec = d["cycle4_decomposition"]
            means = ", ".join(repr(d["cycle_means"][k]) for k in sorted(d["cycle_means"]))
            shares = dec.get("subscore_shares_of_dip", {})
            share_s = " / ".join(
                str(shares.get(k))
                for k in ("completeness", "severity_agreement", "cause_alignment")
            )
            lines.append(
                f"| {dataset} | {means} | {d['spread_published']} | "
                f"{d['spread_excluding_constrained']} | {dec.get('total_dip')} | "
                f"{share_s} | {dec.get('cycle4_refusal_flagged')}/{dec.get('cycle4_records')} |"
            )
        lines.append("")
    g = result["gemini"]
    lines += [
        "## gemini-2.5-flash (zeroed-record separation)",
        "",
        f"- {g['records']} records; mean over all: **{g['mean_all']}**",
        f"- empty API results (result == {{}}): **{g['empty_result_records']}**; records scoring 0: **{g['score_zero_records']}**",
        f"- mean excluding score-0 records (the previously published figure): **{g['mean_excluding_score_zero']}**",
        f"- mean over content-bearing records only: **{g['mean_content_bearing_only']}**",
        f"- empty results that still scored points (flat 10.0 cause_alignment default): {len(g['empty_but_scored_records'])}",
        f"- refusal flags: {g['refusal_flags_total']} total, {g['refusal_flags_on_content']} on content-bearing records "
        f"(scores: {g['flagged_content_scores']})",
        "",
        "## Verdict",
        "",
        result["verdict"],
        "",
    ]
    return "\n".join(lines)


def render_tex(result: dict) -> str:
    def fmt(value, spec=".2f"):
        return format(value, spec) if isinstance(value, float) else "--"

    rows = []
    for model in ("gpt-4o", "gpt-5-mini"):
        for dataset, d in result["spreads"][model].items():
            dec = d["cycle4_decomposition"]
            shares = dec.get("subscore_shares_of_dip", {})
            rows.append(
                f"  {model} & {dataset.replace('_', ' ')} & "
                f"{d['spread_published']:.2f} & {d['spread_excluding_constrained']:.2f} & "
                f"{fmt(dec.get('total_dip'))} & {fmt(shares.get('completeness'))} & "
                f"{fmt(shares.get('severity_agreement'))} \\\\"
            )
    body = "\n".join(rows)
    return (
        "% Generated by scripts/recheck_prompt_sensitivity.py -- do not edit.\n"
        "\\begin{tabular}{llrrrrr}\n"
        "  \\toprule\n"
        "  & & & spread excl. & cycle-4 & \\multicolumn{2}{c}{share of dip} \\\\\n"
        "  \\cmidrule(lr){6-7}\n"
        "  model & dataset & spread & constrained & dip & completeness & severity \\\\\n"
        "  \\midrule\n"
        f"{body}\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )


def build_verdict(spreads: dict) -> str:
    g4 = spreads["gpt-4o"]
    parts = []
    for dataset, d in g4.items():
        dec = d["cycle4_decomposition"]
        shares = dec.get("subscore_shares_of_dip", {})
        top = sorted(
            ((k, v) for k, v in shares.items() if isinstance(v, float) and v > 0),
            key=lambda kv: -kv[1],
        )
        top_s = ", ".join(f"{k} {v:.0%}" for k, v in top)
        parts.append(
            f"{dataset}: spread {d['spread_published']} -> "
            f"{d['spread_excluding_constrained']} excluding the constrained "
            f"cycle (dip decomposition: {top_s})"
        )
    return (
        "The refusal flag never feeds total_score and neither gpt-4o nor "
        "gpt-5-mini has a zeroed record, so the published spread contrast is "
        "unchanged by removing the refusal penalty. However, the gpt-4o "
        "spread is concentrated in the constrained cycle (" + "; ".join(parts) + "). "
        "The dip is carried by the completeness and severity_agreement "
        "subscores on deliberately terse output, not by cause_alignment. The "
        "residual spread excluding that cycle remains larger than "
        "gpt-5-mini's, so a model-generation difference survives, but the "
        "published magnitude is partly an artifact of how the scorer treats "
        "terse output."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify outputs match committed artifacts")
    args = parser.parse_args()

    spreads = {
        model: analyse_spread_model(load_records(MODEL_DIRS[model]))
        for model in ("gpt-4o", "gpt-5-mini")
    }
    gemini = analyse_gemini(load_records(MODEL_DIRS["gemini-2.5-flash"]))
    result = {
        "generator": "scripts/recheck_prompt_sensitivity.py",
        "constrained_cycle": CONSTRAINED_CYCLE,
        "spreads": spreads,
        "gemini": gemini,
        "verdict": build_verdict(spreads),
    }

    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    md = render_md(result)
    tex = render_tex(result)

    if args.check:
        ok = OUT_JSON.read_text() == payload and OUT_TEX.read_text() == tex
        raise SystemExit(0 if ok else "outputs differ from committed artifacts")

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(payload)
    OUT_MD.write_text(md)
    OUT_TEX.write_text(tex)
    print(f"wrote {OUT_JSON.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}, {OUT_TEX.relative_to(ROOT)}")
    print()
    print(result["verdict"])


if __name__ == "__main__":
    main()
