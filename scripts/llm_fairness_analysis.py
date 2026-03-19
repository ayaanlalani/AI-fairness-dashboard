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
    csv_text: str,
    protected_configs: dict[str, str],
    dataset_name: str,
    target_col: str,
    pred_col: str,
    favorable_label: int,
    research_context: str = "",
) -> str:
    """Build the Gemini prompt.

    The LLM receives raw data only and is told to use fairlearn definitions.
    """
    attr_desc = "\n".join(
        f"  - `{attr}` (privileged value: `{priv}`)"
        for attr, priv in protected_configs.items()
    )

    return textwrap.dedent(f"""\
    You are an AI fairness auditor. You are given a CSV of model predictions with
    ground-truth labels and protected attributes. Your job is to independently
    analyse this data for bias using the dataset itself, the fairness toolkit
    context below, and the research evidence provided. You do NOT have access to
    our pre-computed fairness metrics or qualitative analysis.

    ## Dataset
    Name: {dataset_name}
    Target column: `{target_col}` (ground truth)
    Prediction column: `{pred_col}` (model output)
    Favorable label: {favorable_label}

    Protected attributes and their privileged values:
    {attr_desc}

    ## Fairness toolkit context
    {AIF360_CONTEXT}

    ## Research evidence (Semantic Scholar)
    Use this as supporting evidence when you define the audit standard and justify
    mitigations. Do not just repeat paper titles; use them to support your reasoning.
    {research_context}

    ## Raw prediction data (CSV)
    ```
    {csv_text}
    ```

    ## Your task

    First, define a remediation-ready **reference audit specification**: the set
    of metrics and response elements an audit should include to support future
    remediation. Ground this in the research evidence above and your reasoning.

    Then, for EACH protected attribute listed above, compute the following fairness
    metrics (compare unprivileged group(s) vs the privileged group):

    1. **Disparate Impact (DI)** = selection_rate(unprivileged) / selection_rate(privileged)
       Equivalent to fairlearn.metrics.selection_rate ratio.
       For multi-group attributes, use the group with the LOWEST selection rate as the unprivileged group.
    2. **Demographic Parity Difference (DPD)** = selection_rate(unprivileged) - selection_rate(privileged)
       Equivalent to fairlearn.metrics.demographic_parity_difference.
    3. **Equal Opportunity Difference (EOD)** = TPR(unprivileged) - TPR(privileged)
       Equivalent to fairlearn.metrics.equalized_odds_difference restricted to the positive class.
    4. **Average Odds Difference (AOD)** = 0.5 * ((FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv))
       Equivalent to the average of TPR and FPR gaps from fairlearn.metrics.equalized_odds_difference.
    5. **Theil Index** = (1/N) * sum( (y_pred_i / mean(y_pred)) * ln(y_pred_i / mean(y_pred)) )
       where y_pred are the binary predictions. If mean is 0 or all predictions
       are the same, report 0.

    Then produce a qualitative fairness analysis covering:
    - **What is wrong**: describe the disparities you found in the data
    - **Why it is wrong**: root causes (data imbalance, historical bias, proxy features, etc.)
    - **How to fix it**: specific mitigation strategies referencing **fairlearn** algorithms
      (e.g. fairlearn.reductions.ExponentiatedGradient, fairlearn.postprocessing.ThresholdOptimizer,
       fairlearn.reductions.GridSearch, etc.)
    - **Severity**: classify each attribute as CRITICAL (DI < 0.72), HIGH (0.72 <= DI < 0.80),
      MODERATE (0.80 <= DI < 0.95), or LOW (DI >= 0.95)

    ## Required output format

    Return ONLY valid JSON (no markdown fences, no commentary outside the JSON) with this structure:
    {{
      "reference_audit_spec": {{
        "name": "<short name>",
        "core_metrics": ["<metric>", "..."],
        "required_response_elements": ["<element>", "..."],
        "why_this_spec": "<paragraph>",
        "supporting_research": ["<paper citation>", "..."]
      }},
      "metrics": [
        {{
          "attribute": "<attr name>",
          "privileged_value": "<value>",
          "disparate_impact": <float>,
          "demographic_parity_diff": <float>,
          "equal_opportunity_diff": <float>,
          "average_odds_diff": <float>,
          "theil_index": <float>,
          "bias_flag": <bool>
        }}
      ],
      "qualitative": [
        {{
          "attribute": "<attr name>",
          "severity": "<CRITICAL|HIGH|MODERATE|LOW>",
          "what_is_wrong": "<paragraph>",
          "why_is_wrong": "<paragraph>",
          "how_to_fix": "<paragraph>",
          "supporting_research": ["<paper citation>", "..."]
        }}
      ]
    }}

    bias_flag should be true if DI < 0.8, false otherwise.
    Round all floats to 4 decimal places.
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

    raw = response.text.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)

    try:
        return json.loads(raw), usage_stats
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
            f"| Research retrieval | {stage_timings.get('research_retrieval', 0):.2f} |",
            f"| Prompt build | {stage_timings.get('prompt_build', 0):.2f} |",
            f"| API call | {stage_timings.get('api_call', 0):.2f} |",
            f"| Report generation | {stage_timings.get('report_generation', 0):.2f} |",
            f"| Comparison generation | {stage_timings.get('comparison_generation', 0):.2f} |",
            f"| Total benchmark | {stage_timings.get('total_benchmark', 0):.2f} |",
            "",
        ])
    return lines


def generate_llm_report(gemini_result: dict, dataset_name: str,
                        usage_stats: dict | None = None) -> str:
    """Convert Gemini's JSON response into a standalone markdown report."""
    lines = [
        f"# LLM Fairness Analysis (Gemini): {dataset_name}",
        "",
        f"**Model:** {GEMINI_MODEL}",
        f"**Method:** Independent analysis using fairlearn (metrics and mitigations)",
        "",
    ]

    if usage_stats:
        lines.extend(_format_usage_block(usage_stats))

    ref_spec = gemini_result.get("reference_audit_spec")
    if ref_spec:
        lines.append("## Reference Audit Specification")
        lines.append("")
        lines.append(f"**Name:** {ref_spec.get('name', 'N/A')}")
        lines.append("")
        if ref_spec.get("core_metrics"):
            lines.append("**Core metrics:** " + ", ".join(ref_spec["core_metrics"]))
            lines.append("")
        if ref_spec.get("required_response_elements"):
            lines.append(
                "**Required response elements:** "
                + ", ".join(ref_spec["required_response_elements"])
            )
            lines.append("")
        if ref_spec.get("why_this_spec"):
            lines.append(ref_spec["why_this_spec"])
            lines.append("")
        if ref_spec.get("supporting_research"):
            lines.append("**Supporting research:**")
            for citation in ref_spec["supporting_research"]:
                lines.append(f"- {citation}")
            lines.append("")

    # Metrics table
    lines.append("## Computed Metrics")
    lines.append("")
    lines.append("| Attribute | Privileged | DI | DPD | EOD | AOD | Theil | Bias? |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")

    for m in gemini_result.get("metrics", []):
        lines.append(
            f"| {m['attribute']} | {m['privileged_value']} "
            f"| {m['disparate_impact']:.4f} "
            f"| {m['demographic_parity_diff']:.4f} "
            f"| {m['equal_opportunity_diff']:.4f} "
            f"| {m['average_odds_diff']:.4f} "
            f"| {m['theil_index']:.4f} "
            f"| {'Yes' if m['bias_flag'] else 'No'} |"
        )
    lines.append("")

    # Qualitative sections
    for q in gemini_result.get("qualitative", []):
        lines.append("---")
        lines.append(f"## {q['attribute']}  (severity: {q['severity']})")
        lines.append("")
        lines.append("### What is wrong")
        lines.append("")
        lines.append(q["what_is_wrong"])
        lines.append("")
        lines.append("### Why it is wrong")
        lines.append("")
        lines.append(q["why_is_wrong"])
        lines.append("")
        lines.append("### How to fix it")
        lines.append("")
        lines.append(q["how_to_fix"])
        lines.append("")
        if q.get("supporting_research"):
            lines.append("### Supporting research")
            lines.append("")
            for citation in q["supporting_research"]:
                lines.append(f"- {citation}")
            lines.append("")

    return "\n".join(lines)


def generate_comparison(
    gemini_result: dict,
    our_fairness_csv: Path,
    our_qualitative_report: Path,
    dataset_name: str,
    usage_stats: dict | None = None,
) -> str:
    """Build a side-by-side comparison markdown report."""
    our_df = pd.read_csv(our_fairness_csv)
    our_qual = our_qualitative_report.read_text(encoding="utf-8") if our_qualitative_report.exists() else ""

    lines = [
        f"# Benchmark Comparison: {dataset_name}",
        "",
        "Deterministic pipeline (AIF360) vs. Gemini LLM (fairlearn only -- no access to our calculations).",
        "",
    ]

    if usage_stats:
        lines.extend(_format_usage_block(usage_stats))

    # ── metric comparison table ───────────────────────────────────────
    lines.append("## Metric Comparison")
    lines.append("")
    lines.append("| Attribute | Metric | Our Pipeline | Gemini LLM | Delta |")
    lines.append("|---|---|---:|---:|---:|")

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

    for gem_m in gemini_result.get("metrics", []):
        attr = gem_m["attribute"]
        our_row = _find_our_row(our_df, attr)

        for gem_key, our_key in metric_map.items():
            gem_val = gem_m.get(gem_key)
            our_val = _safe_float(our_row.get(our_key)) if our_row is not None else None

            gem_str = f"{gem_val:.4f}" if gem_val is not None else "--"
            our_str = f"{our_val:.4f}" if our_val is not None else "--"

            if gem_val is not None and our_val is not None:
                delta = gem_val - our_val
                delta_str = f"{delta:+.4f}"
            else:
                delta_str = "--"

            lines.append(f"| {attr} | {display_names[gem_key]} | {our_str} | {gem_str} | {delta_str} |")

    lines.append("")

    # ── severity comparison ───────────────────────────────────────────
    lines.append("## Severity Comparison")
    lines.append("")
    lines.append("| Attribute | Our Pipeline | Gemini LLM | Agreement? |")
    lines.append("|---|---|---|---|")

    our_severities = _extract_severities(our_qual)

    for q in gemini_result.get("qualitative", []):
        attr = q["attribute"]
        gem_sev = q["severity"]
        our_sev = our_severities.get(attr, our_severities.get(attr + "_original", "--"))
        agree = "Yes" if _normalize_severity(gem_sev) == _normalize_severity(our_sev) else "No"
        lines.append(f"| {attr} | {our_sev} | {gem_sev} | {agree} |")

    lines.append("")

    # ── qualitative narrative comparison ──────────────────────────────
    lines.append("## Qualitative Narrative Comparison")
    lines.append("")

    our_sections = _extract_qualitative_sections(our_qual)

    for q in gemini_result.get("qualitative", []):
        attr = q["attribute"]
        lines.append(f"### {attr}")
        lines.append("")

        our_sec = our_sections.get(attr, our_sections.get(attr + "_original", {}))

        lines.append("#### What is wrong")
        lines.append("")
        lines.append(f"**Our pipeline:** {our_sec.get('what_is_wrong', 'N/A')}")
        lines.append("")
        lines.append(f"**Gemini:** {q['what_is_wrong']}")
        lines.append("")

        lines.append("#### Why it is wrong")
        lines.append("")
        lines.append(f"**Our pipeline:** {our_sec.get('why_is_wrong', 'N/A')}")
        lines.append("")
        lines.append(f"**Gemini:** {q['why_is_wrong']}")
        lines.append("")

        lines.append("#### How to fix it")
        lines.append("")
        lines.append(f"**Our pipeline:** {our_sec.get('how_to_fix', 'N/A')}")
        lines.append("")
        lines.append(f"**Gemini:** {q['how_to_fix']}")
        lines.append("")

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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_text = _csv_for_prompt(
        predictions_df,
        list(protected_configs.keys()),
        target_col,
        pred_col,
    )
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
    projection = estimate_cost(prompt)
    log.info(
        "Gemini preflight: "
        f"{len(predictions_df):,} rows | {len(prompt):,} prompt chars | "
        f"{projection['projected_input_tokens']:,} projected input tok | "
        f"{projection['projected_output_tokens']:,} assumed max output tok | "
        f"${projection['projected_total_cost_usd']:.4f} projected cost ceiling"
    )

    # Save prompt for reproducibility
    (out_dir / "llm_prompt.txt").write_text(prompt, encoding="utf-8")
    log.info(f"Prompt saved to {out_dir / 'llm_prompt.txt'}")
    (out_dir / "semantic_scholar_context.json").write_text(
        json.dumps(research_evidence, indent=2),
        encoding="utf-8",
    )
    log.info(f"Research context saved to {out_dir / 'semantic_scholar_context.json'}")
    (out_dir / "llm_napkin_math.json").write_text(
        json.dumps({"projection": projection}, indent=2),
        encoding="utf-8",
    )

    t0 = time.time()
    gemini_result, usage_stats = call_gemini(prompt, api_key)
    api_call_s = time.time() - t0
    usage_stats["projection"] = projection

    # Save raw JSON response + usage stats
    (out_dir / "llm_raw_response.json").write_text(
        json.dumps({"result": gemini_result, "usage": usage_stats}, indent=2),
        encoding="utf-8",
    )

    # Generate standalone LLM report
    t0 = time.time()
    llm_report = generate_llm_report(gemini_result, dataset_name, usage_stats)
    llm_report_path = out_dir / "llm_fairness_report.md"
    llm_report_path.write_text(llm_report, encoding="utf-8")
    report_generation_s = time.time() - t0
    log.info(f"LLM report saved to {llm_report_path}")

    # Generate comparison
    t0 = time.time()
    comparison = generate_comparison(
        gemini_result=gemini_result,
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
    )


if __name__ == "__main__":
    main()
