"""
llm_fairness_analysis.py

Gemini-powered LLM fairness benchmark.

Sends the raw predictions CSV to Gemini with instructions to independently
compute fairness metrics (using fairlearn definitions) and produce qualitative
analysis.  Then compares Gemini's results against our deterministic pipeline.

The LLM receives ONLY:
  - The raw predictions data (actual labels, predicted labels, protected attrs)
  - Protected attribute names and their privileged values
  - Instructions referencing fairlearn metric definitions

It does NOT receive our pre-computed metrics or qualitative report.

Usage (standalone):
  python3 scripts/llm_fairness_analysis.py \
    --predictions german_credit_dataset/metrics/classification_predictions.csv \
    --fairness_csv german_credit_dataset/metrics/fairness/fairness_metrics.csv \
    --qualitative_report german_credit_dataset/metrics/fairness/qualitative_report.md \
    --target actual --pred_col predicted --favorable_label 1 \
    --protected_attrs Sex_original:male,AgeGroup_original:40_plus,foreign_worker_original:0 \
    --out_dir german_credit_dataset/metrics/fairness \
    --dataset_name "German Credit"
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

import pandas as pd

from llm_benchmark_common import build_context_and_baseline, score_llm_output
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

GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_MAX_OUTPUT_TOKENS = 5000

AIF360_CONTEXT = """\
Reference fairness toolkit context:
- AIF360 common mitigations: Reweighing, DisparateImpactRemover, PrejudiceRemover,
  EqOddsPostprocessing, CalibratedEqOddsPostprocessing.
- Fairlearn common mitigations: ExponentiatedGradient, GridSearch, ThresholdOptimizer.
- Use these only as reference concepts. You still need to reason from the dataset provided.
"""


# ═══════════════════════════════════════════════════════════════════════
#  Prompt construction
# ═══════════════════════════════════════════════════════════════════════

def _csv_for_prompt(df: pd.DataFrame, protected_attrs: list[str],
                    target_col: str, pred_col: str) -> str:
    """Extract only the columns Gemini needs: target, prediction, protected attrs."""
    cols = [target_col, pred_col] + protected_attrs
    cols = [c for c in cols if c in df.columns]
    subset = df[cols]
    return subset.to_csv(index=False)


def build_prompt(
    context_payload: dict[str, object],
    dataset_name: str,
    research_context: str = "",
) -> str:
    """Build the initial Gemini prompt using deterministic metrics context."""
    context_json = json.dumps(context_payload, indent=2)
    return textwrap.dedent(f"""\
    You are an AI fairness auditor. Python has already computed the fairness
    metrics and quantitative group breakdowns. Do NOT recompute the math. Your
    job is to provide the strongest possible qualitative analysis, define a
    remediation-ready reference audit specification, and give mitigation advice.

    ## Dataset
    Name: {dataset_name}

    ## Fairness toolkit context
    {AIF360_CONTEXT}

    ## Research evidence (Semantic Scholar)
    Use this as supporting evidence. Do not just list papers; use them to justify
    the audit structure and mitigation recommendations.
    {research_context}

    ## Deterministic fairness context (Python-computed)
    ```json
    {context_json}
    ```

    ## Your task
    1. Define a remediation-ready **reference audit specification**.
    2. For each protected attribute, explain:
       - what is wrong
       - why it is wrong
       - how to fix it
       - severity, consistent with the provided DI thresholds
    3. Use named mitigation algorithms where possible.
    4. Support the audit specification and attribute-level recommendations with research citations.

    ## Required output format
    Return ONLY valid JSON with this structure:
    {{
      "reference_audit_spec": {{
        "name": "<short name>",
        "core_metrics": ["<metric>", "..."],
        "required_response_elements": ["<element>", "..."],
        "why_this_spec": "<paragraph>",
        "supporting_research": ["<paper citation>", "..."]
      }},
      "qualitative": [
        {{
          "attribute": "<attr name>",
          "severity": "<CRITICAL|HIGH|MODERATE|LOW>",
          "what_is_wrong": "<paragraph>",
          "why_is_wrong": "<paragraph>",
          "how_to_fix": "<paragraph>",
          "supporting_research": ["<paper citation>", "..."]
        }}
      ],
      "cross_attribute_summary": "<paragraph>"
    }}
    """)


def build_refinement_prompt(
    context_payload: dict[str, object],
    previous_output: dict[str, object],
    dataset_name: str,
    cycle_idx: int,
    research_context: str = "",
) -> str:
    context_json = json.dumps(context_payload, indent=2)
    previous_json = json.dumps(previous_output, indent=2)
    return textwrap.dedent(f"""\
    You are refining a previous AI fairness audit for cycle {cycle_idx}.
    Improve the previous output using the same deterministic fairness context and
    research evidence. Do not recompute metrics. Focus on:
    - clearer causal reasoning tied to the provided metric context
    - stronger, more specific mitigation advice
    - better use of research support
    - consistent severity labels

    ## Dataset
    Name: {dataset_name}

    ## Research evidence (Semantic Scholar)
    {research_context}

    ## Deterministic fairness context
    ```json
    {context_json}
    ```

    ## Previous output to improve
    ```json
    {previous_json}
    ```

    Return ONLY valid JSON in the exact same structure as before.
    """)


# ═══════════════════════════════════════════════════════════════════════
#  Gemini API call
# ═══════════════════════════════════════════════════════════════════════

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 30  # seconds

# Gemini 2.0 Flash pricing (per 1M tokens, as of Feb 2025)
PRICE_INPUT_PER_M = 0.10   # $0.10 per 1M input tokens
PRICE_OUTPUT_PER_M = 0.40  # $0.40 per 1M output tokens


def estimate_cost(prompt: str, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> dict[str, float]:
    """Project request cost using a simple chars-to-tokens heuristic."""
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


def _extract_json_candidate(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw


def _sanitize_json_candidate(raw: str) -> str:
    candidate = _extract_json_candidate(raw)
    candidate = candidate.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    candidate = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", candidate)
    candidate = re.sub(r"\s{2,}", " ", candidate)
    return candidate.strip()


def _parse_gemini_json(raw: str) -> dict:
    candidate = _extract_json_candidate(raw)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        sanitized = _sanitize_json_candidate(raw)
        return json.loads(sanitized)


def call_gemini(prompt: str, api_key: str, model: str = GEMINI_MODEL) -> tuple[dict, dict]:
    """Send prompt to Gemini and parse the JSON response.

    Uses the google-genai SDK with retry + exponential backoff for rate limits.
    Returns (parsed_json, usage_stats) where usage_stats contains token counts,
    cost estimates, and wall-clock timing.
    """
    from google import genai

    client = genai.Client(api_key=api_key)

    t0 = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"Calling Gemini ({model}) ... [attempt {attempt}/{MAX_RETRIES}]")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            break
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "ResourceExhausted" in err_str:
                wait = RETRY_BACKOFF_BASE * attempt
                log.warning(f"Rate limited. Retrying in {wait}s ...")
                time.sleep(wait)
                if attempt == MAX_RETRIES:
                    raise
            else:
                raise
    elapsed = time.time() - t0

    # ── extract usage stats ───────────────────────────────────────────
    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
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
    log.info(
        f"Gemini responded in {elapsed:.1f}s | "
        f"{input_tokens:,} in + {output_tokens:,} out = {total_tokens:,} tokens | "
        f"est. cost ${total_cost:.4f}"
    )

    try:
        raw = response.text.strip()
        return _parse_gemini_json(raw), usage_stats
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse Gemini response as JSON: {e}")
        log.error(f"Raw response (first 500 chars): {raw[:500]}")
        raise


# ═══════════════════════════════════════════════════════════════════════
#  Report generators
# ═══════════════════════════════════════════════════════════════════════

def _format_usage_block(usage: dict) -> list[str]:
    """Format usage stats into markdown lines."""
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
            "## Preflight Estimate",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Projected input tokens | {projection['projected_input_tokens']:,} |",
            f"| Assumed max output tokens | {projection['projected_output_tokens']:,} |",
            f"| Projected total cost ceiling | ${projection['projected_total_cost_usd']:.4f} |",
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
            f"| Build metric context | {stage_timings.get('build_metric_context', 0):.2f} |",
            f"| Prompt build | {stage_timings.get('prompt_build', 0):.2f} |",
            f"| API call | {stage_timings.get('api_call', 0):.2f} |",
            f"| Report generation | {stage_timings.get('report_generation', 0):.2f} |",
            f"| Comparison generation | {stage_timings.get('comparison_generation', 0):.2f} |",
            f"| Total benchmark | {stage_timings.get('total_benchmark', 0):.2f} |",
            "",
        ])
    return lines


def generate_llm_report(
    final_result: dict,
    context_payload: dict,
    cycle_scores: list[dict],
    dataset_name: str,
    usage_stats: dict | None = None,
) -> str:
    lines = [
        f"# LLM Fairness Analysis (Gemini): {dataset_name}",
        "",
        f"**Model:** {GEMINI_MODEL}",
        "**Method:** Deterministic metrics + qualitative reasoning/refinement cycles",
        "",
    ]
    if usage_stats:
        lines.extend(_format_usage_block(usage_stats))

    lines.extend([
        "## Deterministic Metric Context",
        "",
        "| Attribute | Privileged | DI | DPD | EOD | AOD | Theil |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for attr in context_payload.get("attributes", []):
        metrics = attr["metrics"]
        lines.append(
            f"| {attr['attribute']} | {attr['privileged_value']} "
            f"| {_fmt_num(metrics.get('disparate_impact'))} "
            f"| {_fmt_num(metrics.get('demographic_parity_diff'))} "
            f"| {_fmt_num(metrics.get('equal_opportunity_diff'))} "
            f"| {_fmt_num(metrics.get('average_odds_diff'))} "
            f"| {_fmt_num(metrics.get('theil_index'))} |"
        )
    lines.append("")

    lines.extend([
        "## Cycle Scores",
        "",
        "| Cycle | Total Score | Completeness | Severity | Cause Alignment | Mitigation | Research |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for score in cycle_scores:
        subs = score["score"]["subscores"]
        lines.append(
            f"| {score['cycle']} | {score['score']['total_score']:.1f} | "
            f"{subs['completeness']:.1f} | {subs['severity_agreement']:.1f} | "
            f"{subs['cause_alignment']:.1f} | {subs['mitigation_specificity']:.1f} | "
            f"{subs['research_grounding']:.1f} |"
        )
    lines.append("")

    ref_spec = final_result.get("reference_audit_spec", {})
    lines.extend([
        "## Final Reference Audit Specification",
        "",
        f"**Name:** {ref_spec.get('name', 'N/A')}",
        "",
        "**Core metrics:** " + ", ".join(ref_spec.get("core_metrics", [])),
        "",
        "**Required response elements:** " + ", ".join(ref_spec.get("required_response_elements", [])),
        "",
        ref_spec.get("why_this_spec", ""),
        "",
    ])
    if ref_spec.get("supporting_research"):
        lines.append("**Supporting research:**")
        for citation in ref_spec["supporting_research"]:
            lines.append(f"- {citation}")
        lines.append("")

    for q in final_result.get("qualitative", []):
        lines.extend([
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
        ])
        if q.get("supporting_research"):
            lines.append("### Supporting research")
            lines.append("")
            for citation in q["supporting_research"]:
                lines.append(f"- {citation}")
            lines.append("")
    if final_result.get("cross_attribute_summary"):
        lines.extend([
            "## Cross-attribute summary",
            "",
            final_result["cross_attribute_summary"],
            "",
        ])
    return "\n".join(lines)


def generate_comparison(
    final_result: dict,
    baseline_payload: dict,
    our_qualitative_report: Path,
    dataset_name: str,
    cycle_scores: list[dict],
    usage_stats: dict | None = None,
) -> str:
    our_qual = our_qualitative_report.read_text(encoding="utf-8") if our_qualitative_report.exists() else ""
    lines = [
        f"# Benchmark Comparison: {dataset_name}",
        "",
        "Deterministic pipeline vs. Gemini qualitative benchmark over fixed self-refinement cycles.",
        "",
    ]
    if usage_stats:
        lines.extend(_format_usage_block(usage_stats))

    lines.extend([
        "## Cycle Improvement",
        "",
        "| Cycle | Total Score | Summary |",
        "|---|---:|---|",
    ])
    for score in cycle_scores:
        lines.append(f"| {score['cycle']} | {score['score']['total_score']:.1f} | {score['score']['summary']} |")
    lines.append("")

    baseline_by_attr = {row["attribute"]: row for row in baseline_payload.get("attributes", [])}
    lines.extend([
        "## Severity Comparison",
        "",
        "| Attribute | Deterministic Baseline | Gemini | Agreement? |",
        "|---|---|---|---|",
    ])
    for q in final_result.get("qualitative", []):
        attr = q["attribute"]
        baseline = baseline_by_attr.get(attr, {})
        our_sev = baseline.get("severity", "--")
        gem_sev = q["severity"]
        agree = "Yes" if _normalize_severity(gem_sev) == _normalize_severity(our_sev) else "No"
        lines.append(f"| {attr} | {our_sev} | {gem_sev} | {agree} |")
    lines.append("")

    our_sections = _extract_qualitative_sections(our_qual)
    lines.append("## Qualitative Narrative Comparison")
    lines.append("")
    for q in final_result.get("qualitative", []):
        attr = q["attribute"]
        our_sec = our_sections.get(attr, our_sections.get(attr + "_original", {}))
        baseline = baseline_by_attr.get(attr, {})
        lines.extend([
            f"### {attr}",
            "",
            f"**Deterministic root causes:** {'; '.join(baseline.get('root_causes', [])) or 'N/A'}",
            "",
            f"**Gemini why it is wrong:** {q['why_is_wrong']}",
            "",
            f"**Deterministic mitigations:** {'; '.join(baseline.get('mitigations', [])) or our_sec.get('how_to_fix', 'N/A')}",
            "",
            f"**Gemini how to fix it:** {q['how_to_fix']}",
            "",
        ])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _find_our_row(our_df: pd.DataFrame, attr: str) -> dict | None:
    """Match a Gemini attribute name to a row in our fairness CSV."""
    col = "Attribute"
    candidates = [attr, attr.replace("_original", ""), attr.split("_original")[0]]
    for c in candidates:
        mask = our_df[col].astype(str) == c
        if mask.any():
            return our_df[mask].iloc[0].to_dict()
    return None


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _fmt_num(val) -> str:
    safe = _safe_float(val)
    return f"{safe:.4f}" if safe is not None else "--"


def _normalize_severity(s: str) -> str:
    s = s.upper().strip()
    for label in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        if label in s:
            return label
    return s


def _extract_severities(report_md: str) -> dict[str, str]:
    """Pull severity labels from our qualitative report markdown."""
    result = {}
    for match in re.finditer(r"##\s+(\S+)\s+\(DI\s*=.*?severity:\s*(.+?)\)", report_md):
        attr = match.group(1)
        sev = match.group(2).strip().rstrip(")")
        result[attr] = sev
    return result


def _extract_qualitative_sections(report_md: str) -> dict[str, dict]:
    """Parse our qualitative report into per-attribute sections."""
    sections: dict[str, dict] = {}
    attr_blocks = re.split(r"\n---\n## ", report_md)

    for block in attr_blocks:
        attr_match = re.match(r"(\S+)\s+\(", block)
        if not attr_match:
            continue
        attr = attr_match.group(1)
        sec: dict[str, str] = {}

        what_match = re.search(r"### What is wrong\n\n(.*?)(?=\n### |\n---|\Z)", block, re.DOTALL)
        if what_match:
            sec["what_is_wrong"] = what_match.group(1).strip()

        why_match = re.search(r"### Why it is wrong\n\n(.*?)(?=\n### |\n---|\Z)", block, re.DOTALL)
        if why_match:
            sec["why_is_wrong"] = why_match.group(1).strip()

        fix_match = re.search(r"### How to fix it\n\n(.*?)(?=\n### |\n---|\Z)", block, re.DOTALL)
        if fix_match:
            sec["how_to_fix"] = fix_match.group(1).strip()

        sections[attr] = sec

    return sections


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

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
    cycles: int = 3,
) -> Path:
    """Run the full LLM benchmark and write reports. Returns comparison path."""
    benchmark_t0 = time.time()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Export it or add to a .env file."
        )

    t0 = time.time()
    predictions_df = pd.read_csv(predictions_path)
    load_predictions_s = time.time() - t0
    fairness_df = pd.read_csv(fairness_csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    context_payload, baseline_payload, research_evidence = build_context_and_baseline(
        predictions_df=predictions_df,
        fairness_df=fairness_df,
        protected_configs=protected_configs,
        target_col=target_col,
        pred_col=pred_col,
        favorable_label=favorable_label,
        dataset_name=dataset_name,
    )
    metric_context_s = time.time() - t0

    t0 = time.time()
    prompt = build_prompt(
        context_payload=context_payload,
        dataset_name=dataset_name,
        research_context=format_evidence_for_prompt(research_evidence),
    )
    prompt_build_s = time.time() - t0
    projection = estimate_cost(prompt)
    log.info(
        "Gemini preflight: "
        f"{len(predictions_df):,} rows | {len(prompt):,} prompt chars | "
        f"{projection['projected_input_tokens']:,} projected input tok | "
        f"{projection['projected_output_tokens']:,} assumed max output tok | "
        f"${projection['projected_total_cost_usd']:.4f} projected cost ceiling | "
        f"{cycles} cycles"
    )

    # Save prompt for reproducibility
    (out_dir / "llm_prompt.txt").write_text(prompt, encoding="utf-8")
    log.info(f"Prompt saved to {out_dir / 'llm_prompt.txt'}")
    (out_dir / "semantic_scholar_context.json").write_text(
        json.dumps(research_evidence, indent=2),
        encoding="utf-8",
    )
    log.info(f"Research context saved to {out_dir / 'semantic_scholar_context.json'}")
    (out_dir / "llm_context_payload.json").write_text(
        json.dumps(context_payload, indent=2),
        encoding="utf-8",
    )
    (out_dir / "llm_baseline_payload.json").write_text(
        json.dumps(baseline_payload, indent=2),
        encoding="utf-8",
    )
    (out_dir / "llm_napkin_math.json").write_text(
        json.dumps({"projection": projection}, indent=2),
        encoding="utf-8",
    )

    prompt_log = [prompt]
    cycle_outputs: list[dict] = []
    cycle_scores: list[dict] = []
    total_api_call_s = 0.0
    usage_stats: dict | None = None
    gemini_result: dict = {}
    current_prompt = prompt
    for cycle_idx in range(1, cycles + 1):
        t0 = time.time()
        gemini_result, cycle_usage = call_gemini(current_prompt, api_key)
        cycle_api_call_s = time.time() - t0
        total_api_call_s += cycle_api_call_s
        usage_stats = cycle_usage
        score = score_llm_output(gemini_result, baseline_payload)
        cycle_outputs.append({
            "cycle": cycle_idx,
            "result": gemini_result,
            "usage": cycle_usage,
            "score": score,
        })
        cycle_scores.append({
            "cycle": cycle_idx,
            "score": score,
        })
        log.info(f"Gemini cycle {cycle_idx}/{cycles}: score={score['total_score']:.1f}/100")
        if cycle_idx < cycles:
            current_prompt = build_refinement_prompt(
                context_payload=context_payload,
                previous_output=gemini_result,
                dataset_name=dataset_name,
                cycle_idx=cycle_idx + 1,
                research_context=format_evidence_for_prompt(research_evidence),
            )
            prompt_log.append(current_prompt)

    api_call_s = total_api_call_s
    usage_stats = usage_stats or {}
    usage_stats["projection"] = projection
    (out_dir / "llm_prompt.txt").write_text(
        "\n\n".join(prompt_log),
        encoding="utf-8",
    )

    # Save raw JSON response + usage stats
    (out_dir / "llm_raw_response.json").write_text(
        json.dumps({"result": gemini_result, "cycles": cycle_outputs, "usage": usage_stats}, indent=2),
        encoding="utf-8",
    )

    # Generate standalone LLM report
    t0 = time.time()
    llm_report = generate_llm_report(
        final_result=gemini_result,
        context_payload=context_payload,
        cycle_scores=cycle_scores,
        dataset_name=dataset_name,
        usage_stats=usage_stats,
    )
    llm_report_path = out_dir / "llm_fairness_report.md"
    llm_report_path.write_text(llm_report, encoding="utf-8")
    report_generation_s = time.time() - t0
    log.info(f"LLM report saved to {llm_report_path}")

    # Generate comparison
    t0 = time.time()
    comparison = generate_comparison(
        final_result=gemini_result,
        baseline_payload=baseline_payload,
        our_qualitative_report=Path(qualitative_report_path),
        dataset_name=dataset_name,
        cycle_scores=cycle_scores,
        usage_stats=usage_stats,
    )
    comparison_path = out_dir / "benchmark_comparison.md"
    comparison_path.write_text(comparison, encoding="utf-8")
    comparison_generation_s = time.time() - t0
    usage_stats["stage_timings_s"] = {
        "load_predictions": round(load_predictions_s, 2),
        "build_metric_context": round(metric_context_s, 2),
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
        "Gemini benchmark complete: "
        f"api={api_call_s:.1f}s total={usage_stats['stage_timings_s']['total_benchmark']:.1f}s "
        f"actual_cost=${usage_stats['total_cost_usd']:.4f}"
    )
    log.info(f"Benchmark comparison saved to {comparison_path}")

    return comparison_path


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini LLM fairness benchmark",
    )
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions CSV")
    parser.add_argument("--fairness_csv", required=True,
                        help="Path to our fairness_metrics.csv (for comparison only)")
    parser.add_argument("--qualitative_report", required=True,
                        help="Path to our qualitative_report.md (for comparison only)")
    parser.add_argument("--target", required=True,
                        help="Target column name")
    parser.add_argument("--pred_col", default="predicted",
                        help="Prediction column name")
    parser.add_argument("--favorable_label", type=int, default=1,
                        help="Favorable label value")
    parser.add_argument("--protected_attrs", required=True,
                        help="Comma-separated attr:privileged pairs, "
                             "e.g. 'Sex_original:male,AgeGroup_original:40_plus'")
    parser.add_argument("--out_dir", default=".",
                        help="Output directory")
    parser.add_argument("--dataset_name", default="Dataset",
                        help="Name for the report header")
    parser.add_argument("--cycles", type=int, default=3,
                        help="Number of fixed self-refinement cycles")
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
        cycles=args.cycles,
    )


if __name__ == "__main__":
    main()
