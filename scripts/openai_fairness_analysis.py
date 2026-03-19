"""
openai_fairness_analysis.py

OpenAI-powered LLM fairness benchmark.

This script intentionally mirrors the Gemini benchmark methodology:
- the LLM receives raw prediction data,
- fairness toolkit context,
- Semantic Scholar research evidence,
- and protected attribute configuration.

It does NOT receive our pre-computed fairness metrics or qualitative report.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Any

import pandas as pd

from scholarly_evidence import gather_research_context, format_evidence_for_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "o3"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 20
PRICE_INPUT_PER_M = 10.00
PRICE_OUTPUT_PER_M = 40.00
DEFAULT_MAX_COST_USD = 50.0
DEFAULT_MAX_OUTPUT_TOKENS = 5000

FAIRNESS_CONTEXT = """\
Reference fairness toolkit context:
- AIF360 common mitigations: Reweighing, DisparateImpactRemover, PrejudiceRemover,
  EqOddsPostprocessing, CalibratedEqOddsPostprocessing.
- Fairlearn common mitigations: ExponentiatedGradient, GridSearch, ThresholdOptimizer.
- Use these only as reference concepts. You still need to reason from the dataset provided.
"""

OUTPUT_SCHEMA = {
    "name": "fairness_audit",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reference_audit_spec": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "core_metrics": {"type": "array", "items": {"type": "string"}},
                    "required_response_elements": {"type": "array", "items": {"type": "string"}},
                    "why_this_spec": {"type": "string"},
                    "supporting_research": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "name",
                    "core_metrics",
                    "required_response_elements",
                    "why_this_spec",
                    "supporting_research",
                ],
                "additionalProperties": False,
            },
            "metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "attribute": {"type": "string"},
                        "privileged_value": {"type": "string"},
                        "disparate_impact": {"type": "number"},
                        "demographic_parity_diff": {"type": "number"},
                        "equal_opportunity_diff": {"type": "number"},
                        "average_odds_diff": {"type": "number"},
                        "theil_index": {"type": "number"},
                        "bias_flag": {"type": "boolean"},
                    },
                    "required": [
                        "attribute",
                        "privileged_value",
                        "disparate_impact",
                        "demographic_parity_diff",
                        "equal_opportunity_diff",
                        "average_odds_diff",
                        "theil_index",
                        "bias_flag",
                    ],
                    "additionalProperties": False,
                },
            },
            "qualitative": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "attribute": {"type": "string"},
                        "severity": {"type": "string"},
                        "what_is_wrong": {"type": "string"},
                        "why_is_wrong": {"type": "string"},
                        "how_to_fix": {"type": "string"},
                        "supporting_research": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "attribute",
                        "severity",
                        "what_is_wrong",
                        "why_is_wrong",
                        "how_to_fix",
                        "supporting_research",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["reference_audit_spec", "metrics", "qualitative"],
        "additionalProperties": False,
    },
}


def _csv_for_prompt(df: pd.DataFrame, protected_attrs: list[str], target_col: str, pred_col: str) -> str:
    cols = [target_col, pred_col] + protected_attrs
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_csv(index=False)


def build_prompt(
    csv_text: str,
    protected_configs: dict[str, str],
    dataset_name: str,
    target_col: str,
    pred_col: str,
    favorable_label: int,
    research_context: str,
) -> str:
    attr_desc = "\n".join(
        f"  - `{attr}` (privileged value: `{priv}`)" for attr, priv in protected_configs.items()
    )

    return textwrap.dedent(f"""\
    You are an AI fairness auditor.

    You are given raw model prediction data, fairness toolkit context, and research
    evidence retrieved from Semantic Scholar. Your job is to independently analyse
    this dataset for bias. You do NOT have access to our pre-computed fairness
    metrics or qualitative report.

    ## Dataset
    Name: {dataset_name}
    Target column: `{target_col}`
    Prediction column: `{pred_col}`
    Favorable label: {favorable_label}

    Protected attributes and privileged values:
    {attr_desc}

    ## Fairness toolkit context
    {FAIRNESS_CONTEXT}

    ## Research evidence (Semantic Scholar)
    Use this as supporting evidence when defining the remediation-ready audit
    standard and recommending mitigations.
    {research_context}

    ## Raw prediction data (CSV)
    ```
    {csv_text}
    ```

    ## Your task
    First, define a remediation-ready reference audit specification: the set of
    metrics and response elements an audit should include to support future
    remediation.

    Then, for each protected attribute, compute these metrics from the raw data:
    1. Disparate Impact (DI)
    2. Demographic Parity Difference (DPD)
    3. Equal Opportunity Difference (EOD)
    4. Average Odds Difference (AOD)
    5. Theil Index

    For multi-group attributes, use the group with the LOWEST selection rate as the
    unprivileged group. Round all floats to 4 decimal places. bias_flag should be
    true if DI < 0.8, false otherwise.

    Then provide, for each attribute:
    - What is wrong
    - Why it is wrong
    - How to fix it
    - Severity: CRITICAL (DI < 0.72), HIGH (0.72 <= DI < 0.80), MODERATE (0.80 <= DI < 0.95), LOW (DI >= 0.95)
    - Supporting research citations based on the evidence above

    Return ONLY valid JSON matching the required schema.
    """)


def estimate_cost(prompt: str, max_output_tokens: int) -> dict[str, float]:
    """Project upper-bound request cost using a simple chars->tokens heuristic."""
    input_tokens = math.ceil(len(prompt) / 4)
    output_tokens = max_output_tokens
    input_cost = (input_tokens / 1_000_000) * PRICE_INPUT_PER_M
    output_cost = (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_M
    return {
        "projected_input_tokens": input_tokens,
        "projected_output_tokens": output_tokens,
        "projected_total_tokens": input_tokens + output_tokens,
        "projected_input_cost_usd": round(input_cost, 6),
        "projected_output_cost_usd": round(output_cost, 6),
        "projected_total_cost_usd": round(input_cost + output_cost, 6),
    }


def call_openai(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> tuple[dict, dict]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    t0 = time.time()
    response = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"Calling OpenAI ({model}) ... [attempt {attempt}/{MAX_RETRIES}]")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a rigorous AI fairness auditor. Follow the JSON schema exactly."},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=max_output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": OUTPUT_SCHEMA,
                },
            )
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = RETRY_BACKOFF_BASE * attempt
                log.warning(f"Rate limited. Retrying in {wait}s ...")
                time.sleep(wait)
                if attempt == MAX_RETRIES:
                    raise
            else:
                raise

    if response is None:
        raise RuntimeError("OpenAI response was never created")

    elapsed = time.time() - t0
    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = input_tokens + output_tokens
    input_cost = (input_tokens / 1_000_000) * PRICE_INPUT_PER_M
    output_cost = (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_M
    total_cost = input_cost + output_cost

    usage_stats = {
        "model": model,
        "attempts": attempt,
        "wall_clock_s": round(elapsed, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "price_input_per_m": PRICE_INPUT_PER_M,
        "price_output_per_m": PRICE_OUTPUT_PER_M,
    }

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)
    return json.loads(raw), usage_stats


def _format_usage_block(usage: dict) -> list[str]:
    lines = [
        "## Napkin Math",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Model | `{usage['model']}` |",
        f"| Wall-clock time | {usage['wall_clock_s']}s |",
        f"| API attempts | {usage['attempts']} |",
        f"| Input tokens | {usage['input_tokens']:,} |",
        f"| Output tokens | {usage['output_tokens']:,} |",
        f"| Total tokens | {usage['total_tokens']:,} |",
        f"| Input cost | ${usage['input_cost_usd']:.4f} (@ ${usage['price_input_per_m']}/1M tokens) |",
        f"| Output cost | ${usage['output_cost_usd']:.4f} (@ ${usage['price_output_per_m']}/1M tokens) |",
        f"| **Total est. cost** | **${usage['total_cost_usd']:.4f}** |",
        "",
    ]
    projection = usage.get("projection")
    if projection:
        lines.extend([
            "## Cost Guardrail",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Projected input tokens | {projection['projected_input_tokens']:,} |",
            f"| Max output tokens | {projection['projected_output_tokens']:,} |",
            f"| Projected total cost ceiling | ${projection['projected_total_cost_usd']:.4f} |",
            f"| Configured max cost | ${usage['max_cost_usd']:.2f} |",
            "",
        ])
    stage_timings = usage.get("stage_timings_s")
    if stage_timings:
        lines.extend([
            "## Stage Timings",
            "",
            "| Stage | Seconds |",
            "|---|---:|",
            f"| Load predictions | {stage_timings.get('load_predictions', 0):.2f} |",
            f"| Research retrieval | {stage_timings.get('research_retrieval', 0):.2f} |",
            f"| Prompt build | {stage_timings.get('prompt_build', 0):.2f} |",
            f"| API call | {stage_timings.get('api_call', 0):.2f} |",
            f"| Report generation | {stage_timings.get('report_generation', 0):.2f} |",
            f"| Comparison generation | {stage_timings.get('comparison_generation', 0):.2f} |",
            f"| Total benchmark | {stage_timings.get('total_benchmark', 0):.2f} |",
            "",
        ])
    return lines


def generate_llm_report(result: dict, dataset_name: str, model: str, usage_stats: dict | None = None) -> str:
    lines = [
        f"# LLM Fairness Analysis (OpenAI): {dataset_name}",
        "",
        f"**Model:** {model}",
        "**Method:** Independent analysis using raw data, fairness context, and Semantic Scholar evidence",
        "",
    ]
    if usage_stats:
        lines.extend(_format_usage_block(usage_stats))

    ref_spec = result.get("reference_audit_spec")
    if ref_spec:
        lines += [
            "## Reference Audit Specification",
            "",
            f"**Name:** {ref_spec.get('name', 'N/A')}",
            "",
            "**Core metrics:** " + ", ".join(ref_spec.get("core_metrics", [])),
            "",
            "**Required response elements:** " + ", ".join(ref_spec.get("required_response_elements", [])),
            "",
            ref_spec.get("why_this_spec", ""),
            "",
        ]
        if ref_spec.get("supporting_research"):
            lines.append("**Supporting research:**")
            for citation in ref_spec["supporting_research"]:
                lines.append(f"- {citation}")
            lines.append("")

    lines.append("## Computed Metrics")
    lines.append("")
    lines.append("| Attribute | Privileged | DI | DPD | EOD | AOD | Theil | Bias? |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for m in result.get("metrics", []):
        lines.append(
            f"| {m['attribute']} | {m['privileged_value']} | {m['disparate_impact']:.4f} "
            f"| {m['demographic_parity_diff']:.4f} | {m['equal_opportunity_diff']:.4f} "
            f"| {m['average_odds_diff']:.4f} | {m['theil_index']:.4f} "
            f"| {'Yes' if m['bias_flag'] else 'No'} |"
        )
    lines.append("")

    for q in result.get("qualitative", []):
        lines += [
            "---",
            f"## {q['attribute']}  (severity: {q['severity']})",
            "",
            "### What is wrong",
            "",
            q["what_is_wrong"],
            "",
            "### Why it is wrong",
            "",
            q["why_is_wrong"],
            "",
            "### How to fix it",
            "",
            q["how_to_fix"],
            "",
        ]
        if q.get("supporting_research"):
            lines.append("### Supporting research")
            lines.append("")
            for citation in q["supporting_research"]:
                lines.append(f"- {citation}")
            lines.append("")

    return "\n".join(lines)


def _find_our_row(our_df: pd.DataFrame, attr: str) -> dict[str, Any] | None:
    col = "Attribute"
    candidates = [attr, attr.replace("_original", ""), attr.split("_original")[0]]
    for candidate in candidates:
        mask = our_df[col].astype(str) == candidate
        if mask.any():
            return our_df[mask].iloc[0].to_dict()
    return None


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _normalize_severity(value: str) -> str:
    value = value.upper().strip()
    for label in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        if label in value:
            return label
    return value


def _extract_severities(report_md: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in re.finditer(r"##\s+(\S+)\s+\(DI\s*=.*?severity:\s*(.+?)\)", report_md):
        result[match.group(1)] = match.group(2).strip().rstrip(")")
    return result


def _extract_qualitative_sections(report_md: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    attr_blocks = re.split(r"\n---\n## ", report_md)
    for block in attr_blocks:
        attr_match = re.match(r"(\S+)\s+\(", block)
        if not attr_match:
            continue
        attr = attr_match.group(1)
        sec: dict[str, str] = {}
        for key, pattern in {
            "what_is_wrong": r"### What is wrong\n\n(.*?)(?=\n### |\n---|\Z)",
            "why_is_wrong": r"### Why it is wrong\n\n(.*?)(?=\n### |\n---|\Z)",
            "how_to_fix": r"### How to fix it\n\n(.*?)(?=\n### |\n---|\Z)",
        }.items():
            match = re.search(pattern, block, re.DOTALL)
            if match:
                sec[key] = match.group(1).strip()
        sections[attr] = sec
    return sections


def generate_comparison(
    result: dict,
    our_fairness_csv: Path,
    our_qualitative_report: Path,
    dataset_name: str,
    usage_stats: dict | None = None,
) -> str:
    our_df = pd.read_csv(our_fairness_csv)
    our_qual = our_qualitative_report.read_text(encoding="utf-8") if our_qualitative_report.exists() else ""

    lines = [
        f"# Benchmark Comparison: {dataset_name}",
        "",
        "Deterministic pipeline (AIF360) vs. OpenAI LLM (raw data + fairness context + Semantic Scholar; no access to our calculations).",
        "",
    ]
    if usage_stats:
        lines.extend(_format_usage_block(usage_stats))

    lines += [
        "## Metric Comparison",
        "",
        "| Attribute | Metric | Our Pipeline | OpenAI LLM | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    metric_map = {
        "disparate_impact": "DisparateImpact",
        "demographic_parity_diff": "DemographicParityDiff",
        "equal_opportunity_diff": "EqualOpportunityDiff",
        "average_odds_diff": "AverageOddsDiff",
        "theil_index": "TheilIndex",
    }
    display_names = {
        "disparate_impact": "Disparate Impact",
        "demographic_parity_diff": "Demographic Parity Diff",
        "equal_opportunity_diff": "Equal Opportunity Diff",
        "average_odds_diff": "Average Odds Diff",
        "theil_index": "Theil Index",
    }
    for item in result.get("metrics", []):
        attr = item["attribute"]
        our_row = _find_our_row(our_df, attr)
        for llm_key, our_key in metric_map.items():
            llm_val = item.get(llm_key)
            our_val = _safe_float(our_row.get(our_key)) if our_row is not None else None
            llm_str = f"{llm_val:.4f}" if llm_val is not None else "--"
            our_str = f"{our_val:.4f}" if our_val is not None else "--"
            delta_str = f"{(llm_val - our_val):+.4f}" if llm_val is not None and our_val is not None else "--"
            lines.append(f"| {attr} | {display_names[llm_key]} | {our_str} | {llm_str} | {delta_str} |")
    lines.append("")

    lines += [
        "## Severity Comparison",
        "",
        "| Attribute | Our Pipeline | OpenAI LLM | Agreement? |",
        "|---|---|---|---|",
    ]
    our_severities = _extract_severities(our_qual)
    for q in result.get("qualitative", []):
        attr = q["attribute"]
        llm_sev = q["severity"]
        our_sev = our_severities.get(attr, our_severities.get(attr + "_original", "--"))
        agree = "Yes" if _normalize_severity(llm_sev) == _normalize_severity(our_sev) else "No"
        lines.append(f"| {attr} | {our_sev} | {llm_sev} | {agree} |")
    lines.append("")

    lines += ["## Qualitative Narrative Comparison", ""]
    our_sections = _extract_qualitative_sections(our_qual)
    for q in result.get("qualitative", []):
        attr = q["attribute"]
        our_sec = our_sections.get(attr, our_sections.get(attr + "_original", {}))
        lines += [
            f"### {attr}",
            "",
            "#### What is wrong",
            "",
            f"**Our pipeline:** {our_sec.get('what_is_wrong', 'N/A')}",
            "",
            f"**OpenAI:** {q['what_is_wrong']}",
            "",
            "#### Why it is wrong",
            "",
            f"**Our pipeline:** {our_sec.get('why_is_wrong', 'N/A')}",
            "",
            f"**OpenAI:** {q['why_is_wrong']}",
            "",
            "#### How to fix it",
            "",
            f"**Our pipeline:** {our_sec.get('how_to_fix', 'N/A')}",
            "",
            f"**OpenAI:** {q['how_to_fix']}",
            "",
        ]
    return "\n".join(lines)


def run_llm_benchmark(
    predictions_path: str | Path,
    fairness_csv_path: str | Path,
    qualitative_report_path: str | Path,
    target_col: str,
    pred_col: str,
    favorable_label: int,
    protected_configs: dict[str, str],
    out_dir: str | Path,
    dataset_name: str = "Dataset",
    model: str = DEFAULT_MODEL,
    max_cost_usd: float = DEFAULT_MAX_COST_USD,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> Path:
    benchmark_t0 = time.time()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Export it or add to a .env file.")

    t0 = time.time()
    predictions_df = pd.read_csv(predictions_path)
    load_predictions_s = time.time() - t0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_text = _csv_for_prompt(predictions_df, list(protected_configs.keys()), target_col, pred_col)
    t0 = time.time()
    research_evidence = gather_research_context(
        dataset_name=dataset_name,
        protected_attrs=list(protected_configs.keys()),
        max_papers=6,
    )
    research_retrieval_s = time.time() - t0
    t0 = time.time()
    prompt = build_prompt(
        csv_text=csv_text,
        protected_configs=protected_configs,
        dataset_name=dataset_name,
        target_col=target_col,
        pred_col=pred_col,
        favorable_label=favorable_label,
        research_context=format_evidence_for_prompt(research_evidence),
    )
    prompt_build_s = time.time() - t0
    projection = estimate_cost(prompt, max_output_tokens=max_output_tokens)
    log.info(
        "OpenAI preflight: "
        f"{len(predictions_df):,} rows | {len(prompt):,} prompt chars | "
        f"{projection['projected_input_tokens']:,} input tok + "
        f"{projection['projected_output_tokens']:,} output tok = "
        f"${projection['projected_total_cost_usd']:.4f}"
    )
    if projection["projected_total_cost_usd"] > max_cost_usd:
        raise RuntimeError(
            "Projected OpenAI request cost "
            f"${projection['projected_total_cost_usd']:.4f} exceeds cap ${max_cost_usd:.2f}. "
            "Reduce prompt size or max_output_tokens."
        )

    (out_dir / "llm_prompt.txt").write_text(prompt, encoding="utf-8")
    (out_dir / "semantic_scholar_context.json").write_text(json.dumps(research_evidence, indent=2), encoding="utf-8")
    (out_dir / "llm_napkin_math.json").write_text(
        json.dumps({"projection": projection, "max_cost_usd": max_cost_usd}, indent=2),
        encoding="utf-8",
    )

    t0 = time.time()
    result, usage_stats = call_openai(
        prompt,
        api_key,
        model,
        max_output_tokens=max_output_tokens,
    )
    api_call_s = time.time() - t0
    usage_stats["projection"] = projection
    usage_stats["max_cost_usd"] = max_cost_usd

    (out_dir / "llm_raw_response.json").write_text(
        json.dumps({"result": result, "usage": usage_stats}, indent=2),
        encoding="utf-8",
    )

    t0 = time.time()
    llm_report = generate_llm_report(result, dataset_name, model, usage_stats)
    llm_report_path = out_dir / "llm_fairness_report.md"
    llm_report_path.write_text(llm_report, encoding="utf-8")
    report_generation_s = time.time() - t0

    t0 = time.time()
    comparison = generate_comparison(
        result=result,
        our_fairness_csv=Path(fairness_csv_path),
        our_qualitative_report=Path(qualitative_report_path),
        dataset_name=dataset_name,
        usage_stats=usage_stats,
    )
    comparison_path = out_dir / "benchmark_comparison.md"
    comparison_path.write_text(comparison, encoding="utf-8")
    comparison_generation_s = time.time() - t0
    usage_stats["stage_timings_s"] = {
        "load_predictions": round(load_predictions_s, 2),
        "research_retrieval": round(research_retrieval_s, 2),
        "prompt_build": round(prompt_build_s, 2),
        "api_call": round(api_call_s, 2),
        "report_generation": round(report_generation_s, 2),
        "comparison_generation": round(comparison_generation_s, 2),
        "total_benchmark": round(time.time() - benchmark_t0, 2),
    }
    (out_dir / "llm_napkin_math.json").write_text(
        json.dumps({"projection": projection, "usage": usage_stats}, indent=2),
        encoding="utf-8",
    )
    log.info(
        "OpenAI benchmark complete: "
        f"api={api_call_s:.1f}s total={usage_stats['stage_timings_s']['total_benchmark']:.1f}s "
        f"actual_cost=${usage_stats['total_cost_usd']:.4f}"
    )
    log.info(f"Benchmark comparison saved to {comparison_path}")
    return comparison_path


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI fairness benchmark")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV")
    parser.add_argument("--fairness_csv", required=True, help="Path to our fairness_metrics.csv (for comparison only)")
    parser.add_argument("--qualitative_report", required=True, help="Path to our qualitative_report.md (for comparison only)")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--pred_col", default="predicted", help="Prediction column name")
    parser.add_argument("--favorable_label", type=int, default=1, help="Favorable label value")
    parser.add_argument("--protected_attrs", required=True, help="Comma-separated attr:privileged pairs")
    parser.add_argument("--out_dir", default=".", help="Output directory")
    parser.add_argument("--dataset_name", default="Dataset", help="Name for the report header")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--max_cost_usd", type=float, default=DEFAULT_MAX_COST_USD, help="Abort if projected request cost exceeds this amount")
    parser.add_argument("--max_output_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Maximum completion tokens for the OpenAI response")
    args = parser.parse_args()

    configs: dict[str, str] = {}
    for pair in args.protected_attrs.split(","):
        pair = pair.strip()
        if ":" not in pair:
            parser.error(f"Invalid attr:privileged pair: '{pair}'")
        attr, priv = pair.split(":", 1)
        configs[attr.strip()] = priv.strip()

    run_llm_benchmark(
        predictions_path=args.predictions,
        fairness_csv_path=args.fairness_csv,
        qualitative_report_path=args.qualitative_report,
        target_col=args.target,
        pred_col=args.pred_col,
        favorable_label=args.favorable_label,
        protected_configs=configs,
        out_dir=args.out_dir,
        dataset_name=args.dataset_name,
        model=args.model,
        max_cost_usd=args.max_cost_usd,
        max_output_tokens=args.max_output_tokens,
    )


if __name__ == "__main__":
    main()
