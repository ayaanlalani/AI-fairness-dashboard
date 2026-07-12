"""
llm_benchmark.py  —  Section C benchmark (NeurIPS 2026).

4-cycle OpenAI-primary LLM fairness benchmark with distinct prompt strategies:
  1. zero_shot       — direct audit request, no special framing
  2. chain_of_thought — explicit step-by-step reasoning scaffold
  3. self_critique   — LLM refines its own cycle-2 output
  4. constrained     — hard output constraints (must quote metric values,
                       name a specific algorithm, ≥2 citations)

For each (dataset, attribute), writes:
  <out_root>/<model>/<dataset>/<attr>_cycle<N>.json

Each JSON file contains:
  dataset, attribute, cycle, prompt_format, model,
  result, score (5 sub-scores), refusal_detected,
  hallucination_flags, usage

Usage:
  python3 scripts/llm_benchmark.py \\
    --predictions german_credit_dataset/metrics/classification_predictions.csv \\
    --fairness_csv german_credit_dataset/metrics/fairness/fairness_metrics.csv \\
    --qualitative_report german_credit_dataset/metrics/fairness/qualitative_report.md \\
    --target actual --pred_col predicted --favorable_label 1 \\
    --protected_attrs 'Sex_original:male,AgeGroup_original:40_plus,foreign_worker_original:0' \\
    --dataset_name 'German Credit' \\
    --dataset_key german_credit \\
    --out_root artifacts/llm_benchmark \
    --model gpt-4o

Gemini remains available by passing --model gemini-2.5-flash for legacy
comparison runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Any

import pandas as pd

from llm_benchmark_common import build_context_and_baseline, score_llm_output
from scholarly_evidence import format_evidence_for_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True) or ".env")
except ImportError:
    pass

DEFAULT_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-2.5-flash"
PRICE_INPUT_PER_M = 0.10
PRICE_OUTPUT_PER_M = 0.40
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 20

REFUSAL_PHRASES = [
    "i cannot", "i can't", "i'm not able", "unable to",
    "i refuse", "i won't", "not appropriate", "as an ai",
    "i'm sorry, but", "i apologize, but",
]

REQUIRED_RESPONSE_FIELDS = [
    "attribute", "severity", "what_is_wrong", "why_is_wrong", "how_to_fix",
]

VALID_SEVERITIES = {"CRITICAL", "HIGH", "MODERATE", "LOW"}

REQUIRED_ALGORITHMS = [
    "reweighing", "disparateimpactremover", "prejudiceremover",
    "eqoddspostprocessing", "calibratedeqoddspostprocessing",
    "exponentiatedgradient", "gridsearch", "thresholdoptimizer",
    "equalized odds",
]

GUARDRAILS_PATH = Path(__file__).resolve().parent.parent / "configs" / "research_guardrails.json"

AIF360_CONTEXT = (
    "Reference fairness toolkit context:\n"
    "- AIF360: Reweighing, DisparateImpactRemover, PrejudiceRemover, "
    "EqOddsPostprocessing, CalibratedEqOddsPostprocessing.\n"
    "- Fairlearn: ExponentiatedGradient, GridSearch, ThresholdOptimizer.\n"
    "Reason from the provided metric values; do not invent new data."
)


# ═══════════════════════════════════════════════════════════════════════
#  Prompt builders
# ═══════════════════════════════════════════════════════════════════════

def _single_attr_context(context_payload: dict, attr: str) -> str:
    for a in context_payload.get("attributes", []):
        if a["attribute"] == attr:
            payload = {k: v for k, v in context_payload.items() if k != "attributes"}
            payload["attributes"] = [a]
            return json.dumps(payload, indent=2)
    return json.dumps(context_payload, indent=2)


_OUTPUT_SCHEMA = textwrap.dedent("""\
    Return ONLY valid JSON with this exact structure — no markdown, no prose outside:
    {{
      "attribute": "{attr}",
      "severity": "<CRITICAL|HIGH|MODERATE|LOW>",
      "what_is_wrong": "<paragraph>",
      "why_is_wrong": "<paragraph>",
      "how_to_fix": "<paragraph>",
      "supporting_research": ["<citation>", "..."]
    }}""")


def build_zero_shot_prompt(
    context_payload: dict, attr: str, dataset_name: str, research_ctx: str,
    **_kwargs,
) -> str:
    ctx = _single_attr_context(context_payload, attr)
    schema = _OUTPUT_SCHEMA.format(attr=attr)
    return textwrap.dedent(f"""\
    You are an AI fairness auditor. Python has pre-computed the fairness metrics
    below for the "{dataset_name}" dataset. Do NOT recompute the math.
    Provide qualitative analysis for the protected attribute "{attr}".

    {AIF360_CONTEXT}

    ## Research evidence (Semantic Scholar)
    {research_ctx}

    ## Fairness context (pre-computed)
    ```json
    {ctx}
    ```

    {schema}
    """)


def build_chain_of_thought_prompt(
    context_payload: dict, attr: str, dataset_name: str, research_ctx: str,
    **_kwargs,
) -> str:
    ctx = _single_attr_context(context_payload, attr)
    schema = _OUTPUT_SCHEMA.format(attr=attr)
    return textwrap.dedent(f"""\
    You are an AI fairness auditor. Reason step by step before writing your answer.

    Step 1 — Read each metric value for "{attr}" in "{dataset_name}".
    Step 2 — Identify which metric(s) signal the most severe disparity and why.
    Step 3 — Trace the disparity back to its structural or historical cause.
    Step 4 — Select a mitigation algorithm from the AIF360/Fairlearn list that
              specifically addresses that cause. Explain why it fits.
    Step 5 — Assign severity using the DI threshold rules in the context.
    Step 6 — Output your final answer as JSON only (no step-by-step prose in output).

    {AIF360_CONTEXT}

    ## Research evidence (Semantic Scholar)
    {research_ctx}

    ## Fairness context (pre-computed)
    ```json
    {ctx}
    ```

    {schema}
    """)


def build_self_critique_prompt(
    context_payload: dict, attr: str, dataset_name: str, research_ctx: str,
    previous_output: dict,
    **_kwargs,
) -> str:
    ctx = _single_attr_context(context_payload, attr)
    prev_json = json.dumps(previous_output, indent=2)
    schema = _OUTPUT_SCHEMA.format(attr=attr)
    return textwrap.dedent(f"""\
    You are an AI fairness auditor reviewing your own prior analysis of "{attr}"
    in the "{dataset_name}" dataset.

    ## Your previous output
    ```json
    {prev_json}
    ```

    Self-critique checklist — assess each point, then improve:
    1. Severity: is it consistent with the DI threshold rules in the context?
    2. Cause: is "why_is_wrong" specific (cites a mechanism) or generic (says "bias")?
    3. Fix: does "how_to_fix" name a specific algorithm from AIF360 or Fairlearn?
    4. Citations: are the research entries real and directly relevant?

    {AIF360_CONTEXT}

    ## Research evidence (Semantic Scholar)
    {research_ctx}

    ## Fairness context (pre-computed)
    ```json
    {ctx}
    ```

    {schema}
    """)


def build_constrained_prompt(
    context_payload: dict, attr: str, dataset_name: str, research_ctx: str,
    **_kwargs,
) -> str:
    ctx = _single_attr_context(context_payload, attr)
    alg_list = "Reweighing, DisparateImpactRemover, ExponentiatedGradient, ThresholdOptimizer"
    schema = _OUTPUT_SCHEMA.format(attr=attr)
    return textwrap.dedent(f"""\
    You are an AI fairness auditor operating under strict output constraints.

    ## Hard constraints (violations invalidate the response)
    1. "what_is_wrong" MUST quote the exact numerical DI value from the context.
    2. "how_to_fix" MUST name exactly one algorithm from: {alg_list}.
    3. "severity" MUST be exactly one of: CRITICAL, HIGH, MODERATE, LOW — no other text.
    4. "supporting_research" MUST contain at least 2 entries.
    5. Do NOT speculate beyond what the provided metric values support.

    ## Attribute: "{attr}"  |  Dataset: "{dataset_name}"

    {AIF360_CONTEXT}

    ## Research evidence (Semantic Scholar)
    {research_ctx}

    ## Fairness context (pre-computed)
    ```json
    {ctx}
    ```

    {schema}
    """)


# Maps cycle index (1-4) to prompt builder
CYCLE_CONFIGS: list[tuple[str, Any]] = [
    ("zero_shot", build_zero_shot_prompt),
    ("chain_of_thought", build_chain_of_thought_prompt),
    ("self_critique", build_self_critique_prompt),  # receives previous_output
    ("constrained", build_constrained_prompt),
]


# ═══════════════════════════════════════════════════════════════════════
#  Gemini API
# ═══════════════════════════════════════════════════════════════════════

def _extract_json_candidate(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw


def _parse_gemini_json(raw: str) -> dict:
    candidate = _extract_json_candidate(raw)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        sanitized = (
            candidate.replace("\r", " ")
            .replace("\n", " ")
            .replace("\t", " ")
        )
        sanitized = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", sanitized)
        sanitized = re.sub(r"\s{2,}", " ", sanitized)
        return json.loads(sanitized.strip())


def call_gemini(
    prompt: str, api_key: str, model: str = GEMINI_MODEL
) -> tuple[dict, dict, str]:
    """Call Gemini, return (parsed_json, usage_stats, raw_text)."""
    from google import genai

    client = genai.Client(api_key=api_key)
    t0 = time.time()
    raw_text = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"Calling {model} [attempt {attempt}/{MAX_RETRIES}]")
            response = client.models.generate_content(model=model, contents=prompt)
            break
        except Exception as exc:
            err = str(exc)
            retryable = (
                "429" in err or "ResourceExhausted" in err or
                "503" in err or "UNAVAILABLE" in err or
                "500" in err or "INTERNAL" in err
            )
            if retryable and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                log.warning(f"Gemini {err[:80].strip()} — retrying in {wait}s …")
                time.sleep(wait)
            else:
                raise
    elapsed = time.time() - t0

    usage = getattr(response, "usage_metadata", None)
    in_tok = getattr(usage, "prompt_token_count", 0) or 0
    out_tok = getattr(usage, "candidates_token_count", 0) or 0
    in_cost = (in_tok / 1_000_000) * PRICE_INPUT_PER_M
    out_cost = (out_tok / 1_000_000) * PRICE_OUTPUT_PER_M

    usage_stats = {
        "model": model,
        "attempts": attempt,
        "wall_clock_s": round(elapsed, 2),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "input_cost_usd": round(in_cost, 6),
        "output_cost_usd": round(out_cost, 6),
        "total_cost_usd": round(in_cost + out_cost, 6),
    }
    log.info(
        f"  {elapsed:.1f}s | {in_tok:,} in + {out_tok:,} out "
        f"| ${in_cost + out_cost:.5f}"
    )

    raw_text = response.text.strip()
    try:
        result = _parse_gemini_json(raw_text)
    except json.JSONDecodeError as exc:
        log.error(f"JSON parse error: {exc}. Raw (first 500): {raw_text[:500]}")
        raise
    return result, usage_stats, raw_text


OPENAI_PRICE_INPUT_PER_M = 2.50   # gpt-4o
OPENAI_PRICE_OUTPUT_PER_M = 10.00


def call_openai(
    prompt: str, api_key: str, model: str = "gpt-4o"
) -> tuple[dict, dict, str]:
    """Call OpenAI chat completions, return (parsed_json, usage_stats, raw_text)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    t0 = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"Calling {model} [attempt {attempt}/{MAX_RETRIES}]")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a rigorous AI fairness auditor. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            break
        except Exception as exc:
            err = str(exc)
            if ("429" in err or "rate" in err.lower()) and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE * attempt
                log.warning(f"OpenAI rate-limited. Retrying in {wait}s …")
                time.sleep(wait)
                if attempt == MAX_RETRIES:
                    raise
            else:
                raise
    elapsed = time.time() - t0

    usage = response.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    in_cost = (in_tok / 1_000_000) * OPENAI_PRICE_INPUT_PER_M
    out_cost = (out_tok / 1_000_000) * OPENAI_PRICE_OUTPUT_PER_M

    usage_stats = {
        "model": model,
        "attempts": attempt,
        "wall_clock_s": round(elapsed, 2),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "input_cost_usd": round(in_cost, 6),
        "output_cost_usd": round(out_cost, 6),
        "total_cost_usd": round(in_cost + out_cost, 6),
    }
    log.info(
        f"  {elapsed:.1f}s | {in_tok:,} in + {out_tok:,} out | ${in_cost + out_cost:.5f}"
    )

    raw_text = response.choices[0].message.content or ""
    try:
        result = _parse_gemini_json(raw_text)  # same JSON extraction logic
    except json.JSONDecodeError as exc:
        log.error(f"JSON parse error: {exc}. Raw (first 500): {raw_text[:500]}")
        raise
    return result, usage_stats, raw_text


def _provider_for_model(model: str) -> str:
    return "openai" if model.startswith(("gpt", "o1", "o3")) else "gemini"


def enforce_llm_gate(model: str) -> None:
    """Refuse live LLM calls unless configs/research_guardrails.json approves the provider.

    Gate policy: docs/RESEARCH_STAGING_PROMPT.md §0. Missing or unparseable
    guardrail files fail closed.
    """
    provider = _provider_for_model(model)
    status = "blocked"
    if GUARDRAILS_PATH.exists():
        try:
            gates = json.loads(GUARDRAILS_PATH.read_text(encoding="utf-8"))
            status = gates.get("llm_providers", {}).get(provider, "blocked")
        except (json.JSONDecodeError, OSError):
            status = "blocked"
    if status != "approved":
        raise RuntimeError(
            f"Guardrail gate: provider '{provider}' is '{status}' in {GUARDRAILS_PATH}. "
            "Live LLM calls require the user to flip it to 'approved'. "
            "Use --dry-run to build prompt packs without any API call."
        )


def call_model(
    prompt: str, model: str
) -> tuple[dict, dict, str]:
    """Route to the correct LLM backend based on model name."""
    enforce_llm_gate(model)
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        return call_openai(prompt, api_key, model)
    else:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        return call_gemini(prompt, api_key, model)


# ═══════════════════════════════════════════════════════════════════════
#  Refusal + hallucination detection
# ═══════════════════════════════════════════════════════════════════════

def detect_refusal(result: dict | None, raw_text: str = "") -> bool:
    """Return True if the LLM refused or gave an unusably short answer."""
    if result is None:
        return True
    lower = raw_text.lower()
    if any(phrase in lower for phrase in REFUSAL_PHRASES):
        return True
    for field in REQUIRED_RESPONSE_FIELDS:
        if not result.get(field):
            return True
    # Flag very short paragraphs (< 40 chars) as effectively empty
    for field in ("what_is_wrong", "why_is_wrong", "how_to_fix"):
        if len(str(result.get(field, ""))) < 40:
            return True
    sev = str(result.get("severity", "")).upper().strip()
    if sev and sev not in VALID_SEVERITIES:
        return True
    return False


def detect_hallucinations(result: dict, known_titles: list[str]) -> list[str]:
    """
    Flag citations not traceable to our Semantic Scholar context.

    Heuristic: a citation is suspicious if no 3-gram from it appears in
    any known paper title (case-insensitive). Short single-word entries are
    also flagged.
    """
    if not result:
        return []
    cited = result.get("supporting_research", [])
    if not cited:
        return []

    known_blob = " ".join(t.lower() for t in known_titles)
    flags: list[str] = []
    for citation in cited:
        words = citation.lower().split()
        if len(words) < 3:
            flags.append(citation)
            continue
        trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        if not any(tg in known_blob for tg in trigrams):
            flags.append(citation)
    return flags


# ═══════════════════════════════════════════════════════════════════════
#  Per-attribute benchmark runner
# ═══════════════════════════════════════════════════════════════════════

def _mock_llm_response(
    attr: str,
    attr_baseline: dict,
    research_evidence: list[dict],
    context_payload: dict,
) -> tuple[dict, dict, str]:
    """Deterministic stand-in for an LLM response (dry-run mode).

    Restates the deterministic baseline so the scoring harness, refusal
    detector, and hallucination detector all execute on realistic input
    without any network call.
    """
    rows = attr_baseline.get("attributes", [])
    base = rows[0] if rows else {}
    di = None
    for a in context_payload.get("attributes", []):
        if a["attribute"] == attr:
            di = a.get("metrics", {}).get("disparate_impact")
            break
    causes = base.get("root_cause_labels") or ["representation_bias"]
    cause_text = ", ".join(label.replace("_", " ") for label in causes)
    titles = [p.get("title", "") for p in research_evidence[:2] if p.get("title")]
    # Baseline severity strings can carry qualifiers ("MODERATE (borderline)");
    # the LLM is instructed to emit a bare label, so the mock does too.
    severity = next(
        (label for label in VALID_SEVERITIES if label in str(base.get("severity", "")).upper()),
        "MODERATE",
    )
    result = {
        "attribute": attr,
        "severity": severity,
        "what_is_wrong": (
            f"[DRY RUN] The pre-computed context reports disparate impact = {di} for "
            f"'{attr}', alongside the demographic parity, equal opportunity, and "
            "average odds differences supplied by the deterministic pipeline."
        ),
        "why_is_wrong": (
            f"[DRY RUN] The deterministic root-cause analysis attributes this disparity "
            f"to {cause_text}; this mock restates that finding so the cause-alignment "
            "scorer runs end-to-end without an LLM call."
        ),
        "how_to_fix": (
            "[DRY RUN] Apply Reweighing preprocessing, then validate with a "
            "ThresholdOptimizer postprocessing pass, targeting disparate impact "
            ">= 0.8 on the held-out test split."
        ),
        # Only cite titles actually present in the research context; inventing
        # one here would (correctly) trip the hallucination detector.
        "supporting_research": titles,
    }
    usage = {
        "model": "dry_run",
        "attempts": 0,
        "wall_clock_s": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "total_cost_usd": 0.0,
        "dry_run": True,
    }
    return result, usage, json.dumps(result)


def _make_single_attr_baseline(baseline_payload: dict, attr: str) -> dict:
    """Extract a single-attribute baseline payload for scoring."""
    for row in baseline_payload.get("attributes", []):
        if row["attribute"] == attr:
            return {"attributes": [row]}
    return {"attributes": []}


def run_attribute_benchmark(
    attr: str,
    context_payload: dict,
    baseline_payload: dict,
    research_evidence: list[dict],
    dataset_name: str,
    dataset_key: str,
    out_root: Path,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
) -> list[dict]:
    """Run all 4 cycles for a single protected attribute. Returns list of result dicts."""
    known_titles = [p.get("title", "") for p in research_evidence]
    research_ctx = format_evidence_for_prompt(research_evidence)
    attr_baseline = _make_single_attr_baseline(baseline_payload, attr)

    model_dir = "dry_run" if dry_run else model.replace("/", "-")
    attr_dir = out_root / model_dir / dataset_key
    attr_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    prev_output: dict = {}

    for cycle_idx, (fmt_name, builder) in enumerate(CYCLE_CONFIGS, start=1):
        log.info(f"  [{dataset_key}/{attr}] Cycle {cycle_idx}/4 — {fmt_name}")

        prompt = builder(
            context_payload=context_payload,
            attr=attr,
            dataset_name=dataset_name,
            research_ctx=research_ctx,
            previous_output=prev_output,
        )

        if dry_run:
            llm_result, usage, raw_text = _mock_llm_response(
                attr, attr_baseline, research_evidence, context_payload
            )
        else:
            if cycle_idx > 1:
                time.sleep(7)  # stay under 10 RPM (1 req / 6s with margin)
            try:
                llm_result, usage, raw_text = call_model(prompt, model)
            except Exception as exc:
                log.error(f"  Cycle {cycle_idx} failed: {exc}")
                llm_result, usage, raw_text = {}, {}, str(exc)

        refusal = detect_refusal(llm_result if llm_result else None, raw_text)
        hallucinations = detect_hallucinations(llm_result or {}, known_titles)

        # Wrap result for score_llm_output (expects {"qualitative": [...]})
        wrapped = {"qualitative": [llm_result]} if llm_result else {"qualitative": []}
        score = score_llm_output(wrapped, attr_baseline)

        record: dict = {
            "dataset": dataset_key,
            "dataset_name": dataset_name,
            "attribute": attr,
            "cycle": cycle_idx,
            "prompt_format": fmt_name,
            "model": model,
            "result": llm_result,
            "score": score,
            "refusal_detected": refusal,
            "hallucination_flags": hallucinations,
            "usage": usage,
            "dry_run": dry_run,
        }
        if dry_run:
            record["prompt"] = prompt  # frozen prompt pack for post-approval replay

        out_path = attr_dir / f"{attr}_cycle{cycle_idx}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        log.info(
            f"    score={score['total_score']:.1f}/100 | "
            f"refusal={refusal} | hallucinations={len(hallucinations)} | "
            f"→ {out_path.name}"
        )

        results.append(record)
        # Cycle 3 (self_critique) refines cycle 2 output
        if fmt_name == "chain_of_thought" and llm_result:
            prev_output = llm_result

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Dataset-level runner
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark(
    predictions_path: str | Path,
    fairness_csv_path: str | Path,
    qualitative_report_path: str | Path,
    target_col: str,
    pred_col: str,
    favorable_label: int,
    protected_configs: dict[str, str],
    dataset_name: str,
    dataset_key: str,
    out_root: str | Path = "artifacts/llm_benchmark",
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
) -> Path:
    """Run the 4-cycle benchmark for every protected attribute in the dataset."""
    t0_total = time.time()

    out_root = Path(out_root)
    predictions_df = pd.read_csv(predictions_path)
    fairness_df = pd.read_csv(fairness_csv_path)

    log.info(
        f"Building context for {dataset_name} "
        f"({len(predictions_df):,} rows, {len(protected_configs)} attributes)"
    )
    context_payload, baseline_payload, research_evidence = build_context_and_baseline(
        predictions_df=predictions_df,
        fairness_df=fairness_df,
        protected_configs=protected_configs,
        target_col=target_col,
        pred_col=pred_col,
        favorable_label=favorable_label,
        dataset_name=dataset_name,
    )

    all_results: list[dict] = []
    for attr in protected_configs:
        attr_results = run_attribute_benchmark(
            attr=attr,
            context_payload=context_payload,
            baseline_payload=baseline_payload,
            research_evidence=research_evidence,
            dataset_name=dataset_name,
            dataset_key=dataset_key,
            out_root=out_root,
            model=model,
            dry_run=dry_run,
        )
        all_results.extend(attr_results)

    # Write dataset summary
    summary_dir = out_root / ("dry_run" if dry_run else model.replace("/", "-")) / dataset_key
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_summary(all_results, dataset_name, dataset_key, model, time.time() - t0_total)
    summary_path = summary_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"Summary → {summary_path}")
    return summary_path


def _build_summary(
    results: list[dict],
    dataset_name: str,
    dataset_key: str,
    model: str,
    elapsed_s: float,
) -> dict:
    total_cost = sum(r.get("usage", {}).get("total_cost_usd", 0) for r in results)
    refusals = [r for r in results if r["refusal_detected"]]
    hallucinated = [r for r in results if r["hallucination_flags"]]

    # Score progression by cycle (averaged across attributes)
    by_cycle: dict[int, list[float]] = {}
    for r in results:
        c = r["cycle"]
        by_cycle.setdefault(c, []).append(r["score"]["total_score"])
    cycle_avg = {c: round(sum(v) / len(v), 2) for c, v in by_cycle.items()}

    return {
        "dataset": dataset_key,
        "dataset_name": dataset_name,
        "model": model,
        "total_results": len(results),
        "total_cost_usd": round(total_cost, 4),
        "elapsed_s": round(elapsed_s, 1),
        "refusal_count": len(refusals),
        "hallucination_count": len(hallucinated),
        "cycle_avg_score": cycle_avg,
        "attributes": sorted({r["attribute"] for r in results}),
        "cycles": [
            {
                "cycle": r["cycle"],
                "prompt_format": r["prompt_format"],
                "attribute": r["attribute"],
                "score": r["score"]["total_score"],
                "refusal_detected": r["refusal_detected"],
                "hallucination_count": len(r["hallucination_flags"]),
            }
            for r in results
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="4-cycle OpenAI-primary LLM fairness benchmark (NeurIPS Section C)",
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--fairness_csv", required=True)
    parser.add_argument("--qualitative_report", required=True)
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--pred_col", default="predicted")
    parser.add_argument("--favorable_label", type=int, default=1)
    parser.add_argument(
        "--protected_attrs", required=True,
        help="Comma-separated attr:privileged pairs, e.g. 'Sex_original:male,race:White'",
    )
    parser.add_argument("--dataset_name", required=True, help="Human-readable dataset name")
    parser.add_argument("--dataset_key", required=True, help="Short key for output paths, e.g. german_credit")
    parser.add_argument("--out_root", default="artifacts/llm_benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build prompt packs and score a deterministic mock response; "
             "no API client is constructed and no key is read.",
    )
    args = parser.parse_args()

    if not args.dry_run:
        # Fail fast (before any Semantic Scholar traffic) if the provider is gated.
        enforce_llm_gate(args.model)

    configs: dict[str, str] = {}
    for pair in args.protected_attrs.split(","):
        pair = pair.strip()
        if ":" not in pair:
            parser.error(f"Invalid attr:privileged pair: '{pair}'")
        attr, priv = pair.split(":", 1)
        configs[attr.strip()] = priv.strip()

    run_benchmark(
        predictions_path=args.predictions,
        fairness_csv_path=args.fairness_csv,
        qualitative_report_path=args.qualitative_report,
        target_col=args.target,
        pred_col=args.pred_col,
        favorable_label=args.favorable_label,
        protected_configs=configs,
        dataset_name=args.dataset_name,
        dataset_key=args.dataset_key,
        out_root=args.out_root,
        model=args.model,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
