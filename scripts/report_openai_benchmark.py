"""
report_openai_benchmark.py

Generate a markdown report comparing OpenAI to the deterministic model.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
REPORT_PATH = ARTIFACTS / "consolidated" / "openai_vs_model_report.md"

DATASETS = {
    "german_credit": "German Credit",
    "hmda": "HMDA Mortgage Lending (Georgia)",
}


def _normalize_severity(value: str) -> str:
    value = (value or "").upper()
    for label in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        if label in value:
            return label
    return value.strip() or "UNKNOWN"


def _first_fix_line(text: str) -> str:
    for part in text.replace("\r", "\n").split("\n"):
        part = part.strip(" -•")
        if part:
            return part
    return text.strip()


def _load_usage(openai_dir: Path, raw: dict) -> dict:
    usage = dict(raw.get("usage", {}))
    napkin_path = openai_dir / "llm_napkin_math.json"
    if napkin_path.exists():
        napkin_usage = json.loads(napkin_path.read_text(encoding="utf-8")).get("usage", {})
        usage = {**napkin_usage, **usage}
    return usage


def _fmt_delta(value: float) -> str:
    return f"{value:+.1f}"


def _load(ds_key: str) -> dict:
    fairness = ARTIFACTS / ds_key / "fairness"
    openai_dir = fairness / "openai"
    raw = json.loads((openai_dir / "llm_raw_response.json").read_text(encoding="utf-8"))
    return {
        "name": DATASETS[ds_key],
        "raw": raw,
        "usage": _load_usage(openai_dir, raw),
        "baseline": json.loads((openai_dir / "llm_baseline_payload.json").read_text(encoding="utf-8")),
        "context": json.loads((openai_dir / "llm_context_payload.json").read_text(encoding="utf-8")),
        "research": json.loads((fairness / "qualitative_research_evidence.json").read_text(encoding="utf-8"))
        if (fairness / "qualitative_research_evidence.json").exists()
        else {},
    }


def build_report() -> str:
    bundles = {key: _load(key) for key in DATASETS}
    lines = [
        "# OpenAI vs Deterministic Model Snapshot",
        "",
        "This is a short results snapshot comparing the OpenAI qualitative benchmark against the deterministic fairness model. "
        "The deterministic baseline remains the system of record for quantitative metrics, and it now includes "
        "Semantic Scholar-backed qualitative evidence and mitigation guidance.",
        "",
        "## Summary",
        "",
        "| Dataset | Final Score | Cycle Gain | Severity Agreement | Cost (USD) | Latency (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for ds_key, bundle in bundles.items():
        usage = bundle["usage"]
        final_cycle = bundle["raw"]["cycles"][-1]
        first_cycle = bundle["raw"]["cycles"][0]
        baseline_by_attr = {item["attribute"]: item["severity"] for item in bundle["baseline"]["attributes"]}
        agree = 0
        total = 0
        for item in bundle["raw"]["result"]["qualitative"]:
            total += 1
            if _normalize_severity(item["severity"]) == _normalize_severity(baseline_by_attr.get(item["attribute"], "")):
                agree += 1
        latency = usage.get("stage_timings_s", {}).get("api_call", usage.get("wall_clock_s", 0.0))
        delta = final_cycle["score"]["total_score"] - first_cycle["score"]["total_score"]
        lines.append(
            f"| {bundle['name']} | {final_cycle['score']['total_score']:.1f} | {_fmt_delta(delta)} | {agree}/{total} | {usage.get('total_cost_usd', 0.0):.4f} | {latency:.1f} |"
        )

    lines.extend([
        "",
        "## Key Takeaways",
        "",
        "- The deterministic model remains stronger as the quantitative source of truth and as the audit baseline grounded in Semantic Scholar evidence.",
        "- Self-refinement changed the output meaningfully, but it did not reliably improve final benchmark score; the cycle charts should be treated as an evaluation signal, not assumed progress.",
        "- OpenAI's strongest output is operational mitigation language; our model is stronger on consistency, metric framing, and stable cross-attribute audit structure.",
        "",
        "## Visuals",
        "",
        "![OpenAI Cycle Scores](../visualizations/openai_cycle_scores.png)",
        "",
        "![OpenAI Final Subscores](../visualizations/openai_final_subscores.png)",
        "",
        "![OpenAI Severity Agreement](../visualizations/openai_severity_agreement.png)",
        "",
        "![OpenAI Cost and Latency](../visualizations/openai_cost_latency.png)",
        "",
        "![OpenAI DI Context](../visualizations/openai_di_context.png)",
        "",
    ])

    for ds_key, bundle in bundles.items():
        usage = bundle["usage"]
        lines.extend([
            f"## {bundle['name']}",
            "",
        ])
        final_cycle = bundle["raw"]["cycles"][-1]
        first_cycle = bundle["raw"]["cycles"][0]
        baseline_by_attr = {item["attribute"]: item["severity"] for item in bundle["baseline"]["attributes"]}
        agree = sum(
            1
            for item in bundle["raw"]["result"]["qualitative"]
            if _normalize_severity(item["severity"]) == _normalize_severity(baseline_by_attr.get(item["attribute"], ""))
        )
        total = len(bundle["raw"]["result"]["qualitative"])
        latency = usage.get("stage_timings_s", {}).get("api_call", usage.get("wall_clock_s", 0.0))
        delta = final_cycle["score"]["total_score"] - first_cycle["score"]["total_score"]
        lines.extend([
            f"- OpenAI final score: `{final_cycle['score']['total_score']:.1f}/100`",
            f"- OpenAI cycle gain: `{_fmt_delta(delta)}` points",
            f"- Severity agreement with our model: `{agree}/{total}` attributes",
            f"- OpenAI cost: `${usage.get('total_cost_usd', 0.0):.4f}`",
            f"- OpenAI API latency: `{latency:.1f}s`",
            f"- Deterministic model research packets: `{sum(len(v) for v in bundle['research'].values())}` paper entries across attributes",
            "",
            "### Deterministic Model",
            "",
        ])

        for attr in bundle["baseline"]["attributes"][:2]:
            lines.append(f"- `{attr['attribute']}`: {_first_fix_line(attr['mitigations'][0]) if attr['mitigations'] else 'No mitigation listed.'}")

        lines.extend([
            "",
            "### OpenAI",
            "",
        ])
        for item in bundle["raw"]["result"]["qualitative"][:2]:
            lines.append(f"- `{item['attribute']}`: {_first_fix_line(item['how_to_fix'])}")

        lines.extend([
            "",
            "### Read",
            "",
            "Keep the deterministic model as the benchmark backbone and quantitative audit record. Use OpenAI as a second-pass "
            "qualitative planner that turns the metric findings into prioritised remediation steps.",
            "",
        ])

    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(), encoding="utf-8")


if __name__ == "__main__":
    main()
