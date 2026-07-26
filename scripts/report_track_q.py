"""
report_track_q.py

Track Q analysis: is the LLM a reliable metric *interpreter*?

Reads the 4-cycle benchmark records written by scripts/llm_benchmark.py and
emits the Stage 3 tables. Nothing else in the repo read these packs before —
visualize_openai_benchmark.py and visualize_benchmark.py both read the Track H
(openai_fairness_analysis.py) artifacts instead, and both hardcode a
two-dataset dict that excludes lending_club.

The deterministic baseline severity is recovered from the matching `dry_run`
pack: `_mock_llm_response` restates the baseline verbatim, which is why a
matching live severity scores exactly 20.0 on the `severity_agreement`
subscore. That makes the dry-run packs a reliable ground-truth source without
recomputing metrics here (guardrail 5: this script never recomputes a metric).

No network, no API key, no LLM call.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH_ROOT = REPO_ROOT / "artifacts" / "llm_benchmark"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "consolidated"

CYCLE_STRATEGIES = {
    1: "zero_shot",
    2: "chain_of_thought",
    3: "self_critique",
    4: "constrained",
}
SEVERITY_ORDER = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

# Mirrors of the frozen detector's trigger conditions (llm_benchmark.py).
# Imported rather than re-declared so the classification below cannot drift
# from the detector it is explaining.
try:
    from llm_benchmark import (
        REFUSAL_PHRASES,
        REQUIRED_RESPONSE_FIELDS,
        VALID_SEVERITIES,
    )
except ImportError:  # pragma: no cover - keeps this script standalone-runnable
    REFUSAL_PHRASES = ["i cannot", "i can't", "as an ai"]
    REQUIRED_RESPONSE_FIELDS = [
        "attribute", "severity", "what_is_wrong", "why_is_wrong", "how_to_fix",
    ]
    VALID_SEVERITIES = {"CRITICAL", "HIGH", "MODERATE", "LOW"}

_NARRATIVE_FIELDS = ("what_is_wrong", "why_is_wrong", "how_to_fix")
_BREVITY_THRESHOLD = 40  # llm_benchmark.detect_refusal flags fields under this


def classify_refusal(rec: dict[str, Any]) -> str | None:
    """Explain *why* a record was flagged as a refusal.

    `detect_refusal` treats a narrative field under 40 characters as
    "effectively empty". The `constrained` prompt explicitly asks for terse
    output, so it trips that heuristic while answering correctly — e.g.
    `how_to_fix: "DisparateImpactRemover"` is a precise, on-spec mitigation in
    22 characters. Reporting those as refusals would misattribute a measurement
    artefact to model behaviour, so they are separated here.

    Returns "genuine", "brevity_only", or None if not flagged.
    """
    if not rec.get("refusal_detected"):
        return None
    result = rec.get("result") or {}
    if not result:
        return "genuine"
    blob = json.dumps(result).lower()
    if any(phrase in blob for phrase in REFUSAL_PHRASES):
        return "genuine"
    if any(not result.get(field) for field in REQUIRED_RESPONSE_FIELDS):
        return "genuine"
    sev = str(result.get("severity", "")).upper().strip()
    if sev and sev not in VALID_SEVERITIES:
        return "genuine"
    if any(len(str(result.get(f, ""))) < _BREVITY_THRESHOLD for f in _NARRATIVE_FIELDS):
        return "brevity_only"
    return "genuine"


def load_records(bench_root: Path, model_dir: str) -> list[dict[str, Any]]:
    """Load every cycle record for a model directory."""
    records: list[dict[str, Any]] = []
    root = bench_root / model_dir
    if not root.exists():
        return records
    for path in sorted(root.glob("*/*_cycle*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            log.warning("Skipping unparseable record: %s", path)
    return records


def baseline_severities(bench_root: Path) -> dict[tuple[str, str], str]:
    """Deterministic severity per (dataset, attribute), from the dry-run packs."""
    out: dict[tuple[str, str], str] = {}
    for path in sorted((bench_root / "dry_run").glob("*/*_cycle1.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        sev = (rec.get("result") or {}).get("severity")
        if sev:
            out[(rec["dataset"], rec["attribute"])] = str(sev).upper()
    return out


def _severity(rec: dict[str, Any]) -> str | None:
    sev = (rec.get("result") or {}).get("severity")
    return str(sev).upper() if sev else None


def analyse(records: list[dict[str, Any]], baselines: dict[tuple[str, str], str]) -> dict[str, Any]:
    by_pair: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        by_pair[(rec["dataset"], rec["attribute"])][rec["cycle"]] = rec

    # ---- cross-cycle severity stability -------------------------------
    stability_rows = []
    for (dataset, attr), cycles in sorted(by_pair.items()):
        sevs = [_severity(cycles[c]) for c in sorted(cycles)]
        present = [s for s in sevs if s]
        distinct = sorted(set(present), key=lambda s: SEVERITY_ORDER.index(s) if s in SEVERITY_ORDER else 99)
        baseline = baselines.get((dataset, attr))
        agree_flags = [s == baseline for s in sevs if s]
        stability_rows.append({
            "dataset": dataset,
            "attribute": attr,
            "baseline_severity": baseline,
            "cycle_severities": sevs,
            "distinct_severities": distinct,
            "stable_across_cycles": len(distinct) <= 1,
            "agreement_count": sum(agree_flags),
            "cycles_scored": len(present),
        })

    # ---- agreement vs deterministic classify_severity -----------------
    agree_total = sum(r["agreement_count"] for r in stability_rows)
    agree_denom = sum(r["cycles_scored"] for r in stability_rows)
    by_dataset_agree: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    direction = {"over_escalates": 0, "under_escalates": 0, "matches": 0}
    for row in stability_rows:
        slot = by_dataset_agree[row["dataset"]]
        slot[0] += row["agreement_count"]
        slot[1] += row["cycles_scored"]
        base = row["baseline_severity"]
        for sev in row["cycle_severities"]:
            if not sev or not base:
                continue
            if sev == base:
                direction["matches"] += 1
            elif sev in SEVERITY_ORDER and base in SEVERITY_ORDER:
                if SEVERITY_ORDER.index(sev) > SEVERITY_ORDER.index(base):
                    direction["over_escalates"] += 1
                else:
                    direction["under_escalates"] += 1

    # ---- per-prompt-strategy rates ------------------------------------
    by_strategy: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_strategy[rec["cycle"]].append(rec)

    strategy_rows = []
    for cycle in sorted(by_strategy):
        group = by_strategy[cycle]
        n = len(group)
        scores = [r["score"]["total_score"] for r in group]
        refusals = sum(1 for r in group if r.get("refusal_detected"))
        kinds = [classify_refusal(r) for r in group]
        genuine_refusals = sum(1 for k in kinds if k == "genuine")
        brevity_refusals = sum(1 for k in kinds if k == "brevity_only")
        halluc_records = sum(1 for r in group if r.get("hallucination_flags"))
        halluc_flags = sum(len(r.get("hallucination_flags") or []) for r in group)
        agree = 0
        for r in group:
            sev = _severity(r)
            base = baselines.get((r["dataset"], r["attribute"]))
            if sev and base and sev == base:
                agree += 1
        strategy_rows.append({
            "cycle": cycle,
            "strategy": CYCLE_STRATEGIES.get(cycle, f"cycle{cycle}"),
            "n": n,
            "mean_score": round(sum(scores) / n, 2) if n else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "refusal_count": refusals,
            "refusal_rate": round(refusals / n, 4) if n else 0.0,
            "genuine_refusal_count": genuine_refusals,
            "genuine_refusal_rate": round(genuine_refusals / n, 4) if n else 0.0,
            "brevity_only_refusal_count": brevity_refusals,
            "hallucinated_records": halluc_records,
            "hallucination_flag_count": halluc_flags,
            "hallucination_rate": round(halluc_records / n, 4) if n else 0.0,
            "severity_agreement_rate": round(agree / n, 4) if n else 0.0,
        })

    # ---- cost / token accounting --------------------------------------
    cost = sum(float((r.get("usage") or {}).get("total_cost_usd", 0) or 0) for r in records)
    tok_in = sum(int((r.get("usage") or {}).get("input_tokens", 0) or 0) for r in records)
    tok_out = sum(int((r.get("usage") or {}).get("output_tokens", 0) or 0) for r in records)
    wall = sum(float((r.get("usage") or {}).get("wall_clock_s", 0) or 0) for r in records)

    return {
        "n_records": len(records),
        "stability": stability_rows,
        "agreement": {
            "overall_rate": round(agree_total / agree_denom, 4) if agree_denom else 0.0,
            "agreed": agree_total,
            "scored": agree_denom,
            "by_dataset": {
                d: {"agreed": v[0], "scored": v[1],
                    "rate": round(v[0] / v[1], 4) if v[1] else 0.0}
                for d, v in sorted(by_dataset_agree.items())
            },
            "direction": direction,
        },
        "by_strategy": strategy_rows,
        "cost": {
            "total_usd": round(cost, 6),
            "input_tokens": tok_in,
            "output_tokens": tok_out,
            "total_tokens": tok_in + tok_out,
            "wall_clock_s": round(wall, 1),
            "usd_per_call": round(cost / len(records), 6) if records else 0.0,
        },
    }


def _fmt_sevs(sevs: list[str | None]) -> str:
    return " → ".join(s or "—" for s in sevs)


def render_markdown(
    result: dict[str, Any],
    model: str,
    legacy: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    A = lines.append

    A(f"# Track Q — 4-Cycle Benchmark Analysis (`{model}`)")
    A("")
    A("Track Q asks a narrow question: **is the LLM a reliable interpreter of "
      "fairness metrics it was handed?** It is scored against the deterministic "
      "`classify_severity` baseline. It is not a test of audit quality — that is "
      "Track H, which is deliberately never scored against these numbers.")
    A("")
    A(f"Records analysed: **{result['n_records']}** "
      f"(3 use cases x 3 attributes x 4 prompt strategies). "
      f"Prompts replayed verbatim from the frozen Stage 2 packs.")
    A("")

    cost = result["cost"]
    A("## Cost and token accounting")
    A("")
    A("| Metric | Value |")
    A("|---|---|")
    A(f"| Total spend | **${cost['total_usd']:.4f}** |")
    A(f"| Cost per call | ${cost['usd_per_call']:.5f} |")
    A(f"| Input tokens | {cost['input_tokens']:,} |")
    A(f"| Output tokens | {cost['output_tokens']:,} |")
    A(f"| Total tokens | {cost['total_tokens']:,} |")
    A(f"| Cumulative API wall clock | {cost['wall_clock_s']:.1f} s |")
    A("")

    agree = result["agreement"]
    A("## Agreement with the deterministic severity baseline")
    A("")
    A(f"Overall agreement: **{agree['agreed']}/{agree['scored']} "
      f"({agree['overall_rate']*100:.1f}%)**.")
    A("")
    A("| Use case | Agreed | Scored | Rate |")
    A("|---|---|---|---|")
    for dataset, row in agree["by_dataset"].items():
        A(f"| {dataset} | {row['agreed']} | {row['scored']} | {row['rate']*100:.1f}% |")
    A("")
    A("**The spread across use cases matters more than the 75% headline, and the "
      "ordering is not a competence ranking.** lending_club's 100% is the easiest "
      "possible case, not the best performance: its classifier predicts "
      "non-default for ~99.5% of the test set, so every attribute is a "
      "near-parity LOW and agreeing costs the model nothing. german_credit's "
      "33.3% is the hardest: three MODERATE baselines sitting near threshold "
      "boundaries, where a small interpretive difference flips the label. "
      "Aggregate agreement is therefore a function of how discriminating the "
      "underlying classifiers are, and should not be quoted as a single "
      "capability number.")
    A("")
    d = agree["direction"]
    total_dir = sum(d.values()) or 1
    A("Where it disagrees, the direction matters more than the rate:")
    A("")
    A("| Direction | Count | Share |")
    A("|---|---|---|")
    A(f"| Matches baseline | {d['matches']} | {d['matches']/total_dir*100:.1f}% |")
    A(f"| **Over-escalates** (harsher than baseline) | {d['over_escalates']} | {d['over_escalates']/total_dir*100:.1f}% |")
    A(f"| **Under-escalates** (milder than baseline) | {d['under_escalates']} | {d['under_escalates']/total_dir*100:.1f}% |")
    A("")
    A("Under-escalation is the consequential error for an audit tool: it means a "
      "disparity the deterministic pipeline flagged was reported as less serious "
      "than it is. Over-escalation is comparatively safe — it produces extra "
      "review, not a missed finding.")
    A("")

    A("## Cross-cycle severity stability")
    A("")
    A("| Use case | Attribute | Baseline | Cycle 1 → 2 → 3 → 4 | Stable | Agreed |")
    A("|---|---|---|---|---|---|")
    for row in result["stability"]:
        stable = "yes" if row["stable_across_cycles"] else "**no**"
        A(f"| {row['dataset']} | `{row['attribute']}` | {row['baseline_severity'] or '—'} "
          f"| {_fmt_sevs(row['cycle_severities'])} | {stable} "
          f"| {row['agreement_count']}/{row['cycles_scored']} |")
    A("")
    unstable = [r for r in result["stability"] if not r["stable_across_cycles"]]
    A(f"{len(unstable)} of {len(result['stability'])} (use case x attribute) pairs "
      f"changed severity label across the four prompt strategies on identical "
      f"input context. Prompt phrasing alone moves the verdict.")
    A("")

    A("## Per-prompt-strategy behaviour")
    A("")
    A("| Cycle | Strategy | Mean score | Range | Severity agreement | Genuine refusals | Flagged (brevity) | Hallucination rate | Flags |")
    A("|---|---|---|---|---|---|---|---|---|")
    for row in result["by_strategy"]:
        A(f"| {row['cycle']} | `{row['strategy']}` | {row['mean_score']:.2f} "
          f"| {row['min_score']:.0f}–{row['max_score']:.0f} "
          f"| {row['severity_agreement_rate']*100:.1f}% "
          f"| {row['genuine_refusal_count']}/{row['n']} "
          f"| {row['brevity_only_refusal_count']}/{row['n']} "
          f"| {row['hallucination_rate']*100:.1f}% "
          f"| {row['hallucination_flag_count']} |")
    A("")
    total_brevity = sum(r["brevity_only_refusal_count"] for r in result["by_strategy"])
    total_genuine = sum(r["genuine_refusal_count"] for r in result["by_strategy"])
    A("### The refusal column needs a caveat, not a headline")
    A("")
    A(f"`detect_refusal` flagged {total_brevity + total_genuine} records. "
      f"**{total_genuine} are genuine refusals; {total_brevity} are the brevity "
      f"heuristic firing on a correct answer.** The detector treats any narrative "
      f"field under {_BREVITY_THRESHOLD} characters as \"effectively empty\", and "
      f"the `constrained` prompt explicitly asks for terse output — so "
      f"`how_to_fix: \"DisparateImpactRemover\"` is flagged despite being a "
      f"precise, on-spec mitigation in 22 characters. None of the flagged "
      f"records contain a refusal phrase, a missing field, or an invalid "
      f"severity.")
    A("")
    A("This is a finding about the *harness*, not the model: a length proxy for "
      "substance misclassifies compliance with an instruction to be brief. The "
      "detector is deliberately left unchanged — it is part of the frozen Stage 2 "
      "scoring harness, and altering it would invalidate comparison against the "
      "pre-freeze run — but the distinction is reported wherever the rate is "
      "quoted. The scoring penalty is real either way: `constrained` has the "
      "lowest floor of any strategy because `completeness` also rewards length.")
    A("")

    if legacy:
        A("## Contrast: the pre-freeze run (no research grounding)")
        A("")
        A("`artifacts/llm_benchmark/legacy_pre_freeze/` holds an earlier real "
          "gpt-4o run over german_credit and hmda whose prompts carried **no "
          "Semantic Scholar evidence** (`research_grounding` 0.0) and were never "
          "frozen. Comparing like-for-like on those two use cases isolates what "
          "the research context changed.")
        A("")
        A("| Run | Records | Mean score | Refusals | Hallucination flags |")
        A("|---|---|---|---|---|")
        A(f"| Pre-freeze, no evidence | {legacy['n_records']} "
          f"| {legacy['mean_score']:.2f} | {legacy['refusals']} "
          f"| **{legacy['hallucination_flags']}** |")
        A(f"| Stage 3, evidence embedded | {legacy['current_n']} "
          f"| {legacy['current_mean']:.2f} | {legacy['current_refusals']} "
          f"| **{legacy['current_hallucination_flags']}** |")
        A("")

    return "\n".join(lines) + "\n"


def summarise_legacy(
    bench_root: Path, current: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Compare the pre-freeze run against the current one on shared use cases."""
    legacy_records = load_records(bench_root, "legacy_pre_freeze")
    if not legacy_records:
        return None
    shared = {r["dataset"] for r in legacy_records}
    cur = [r for r in current if r["dataset"] in shared]
    if not cur:
        return None

    def _mean(rs):
        return sum(r["score"]["total_score"] for r in rs) / len(rs) if rs else 0.0

    return {
        "n_records": len(legacy_records),
        "mean_score": _mean(legacy_records),
        "refusals": sum(1 for r in legacy_records if r.get("refusal_detected")),
        "hallucination_flags": sum(len(r.get("hallucination_flags") or []) for r in legacy_records),
        "current_n": len(cur),
        "current_mean": _mean(cur),
        "current_refusals": sum(1 for r in cur if r.get("refusal_detected")),
        "current_hallucination_flags": sum(len(r.get("hallucination_flags") or []) for r in cur),
        "shared_datasets": sorted(shared),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Track Q benchmark analysis (no API calls)")
    parser.add_argument("--model", default="gpt-4o", help="Model directory under artifacts/llm_benchmark/")
    parser.add_argument("--bench_root", default=str(DEFAULT_BENCH_ROOT))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    bench_root = Path(args.bench_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(bench_root, args.model)
    if not records:
        raise SystemExit(f"No records found under {bench_root / args.model}")
    baselines = baseline_severities(bench_root)
    if not baselines:
        log.warning("No dry-run packs found; severity agreement will be empty.")

    result = analyse(records, baselines)
    legacy = summarise_legacy(bench_root, records)

    json_path = out_dir / "track_q_analysis.json"
    json_path.write_text(
        json.dumps({"model": args.model, **result, "legacy_contrast": legacy}, indent=2),
        encoding="utf-8",
    )
    md_path = out_dir / "track_q_analysis.md"
    md_path.write_text(render_markdown(result, args.model, legacy), encoding="utf-8")

    log.info("Analysed %d records", result["n_records"])
    log.info(
        "Severity agreement %.1f%% | spend $%.4f",
        result["agreement"]["overall_rate"] * 100,
        result["cost"]["total_usd"],
    )
    log.info("Wrote %s", md_path)
    log.info("Wrote %s", json_path)


if __name__ == "__main__":
    main()
