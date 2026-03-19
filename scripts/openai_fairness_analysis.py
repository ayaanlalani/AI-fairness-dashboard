"""
llm_fairness_analysis_openai.py

OpenAI o3-powered LLM fairness benchmark — a "blind" second-opinion auditor.

Architecture (3-pass pipeline):
  Pass 1 — Python deterministically computes ALL metrics and hands verified
            numbers to o3 via a tool-call result. o3 never does raw math.
  Pass 2 — o3 receives the verified metric payload and writes deep qualitative
            narrative (proxy detection, sample-size warnings, regulatory framing).
  Pass 3 — o3 receives the Pass-2 narrative and writes the prioritised
            mitigation roadmap with explicit tradeoffs.

Metrics computed (beyond the Gemini baseline):
  Standard:      DI, DPD, EOD, AOD, Theil Index
  Extended:      Predictive Parity Difference (PPD)
                 False Discovery Rate Difference (FDRD)
                 Intersectional analysis (all pairwise attr combinations)
  Structural:    Per-group sample sizes + statistical reliability flags
                 (groups with n < 30 are flagged as unreliable)

Structured Outputs: every o3 response is validated against a strict JSON schema
using OpenAI's response_format={"type": "json_schema", ...} — no regex cleanup.

Usage (standalone):
  python3 scripts/llm_fairness_analysis_openai.py \
    --predictions german_credit_dataset/metrics/classification_predictions.csv \
    --fairness_csv german_credit_dataset/metrics/fairness/fairness_metrics.csv \
    --qualitative_report german_credit_dataset/metrics/fairness/qualitative_report.md \
    --target actual --pred_col predicted --favorable_label 1 \
    --protected_attrs Sex_original:male,AgeGroup_original:40_plus,foreign_worker_original:0 \
    --out_dir german_credit_dataset/metrics/fairness \
    --dataset_name "German Credit" \
    --model o3

Drop-in compatible: produces the same output filenames as the Gemini version
so run_pipeline.py can call either auditor interchangeably.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

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

# ── model config ──────────────────────────────────────────────────────
DEFAULT_MODEL = "o3"

# o3 pricing (per 1M tokens, as of 2025)
PRICE_INPUT_PER_M  = 10.00   # $10 per 1M input tokens
PRICE_OUTPUT_PER_M = 40.00   # $40 per 1M output tokens

# Reliability threshold: groups below this sample size get a warning flag
MIN_RELIABLE_N = 30

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 20  # seconds


# ═══════════════════════════════════════════════════════════════════════
#  PASS 1 — Deterministic metric computation (Python, not LLM)
# ═══════════════════════════════════════════════════════════════════════

def _rates(df: pd.DataFrame, pred_col: str, target_col: str,
           favorable_label: int) -> dict[str, float]:
    """Compute selection rate, TPR, FPR, precision for a group slice."""
    n = len(df)
    if n == 0:
        return {"n": 0, "selection_rate": float("nan"),
                "tpr": float("nan"), "fpr": float("nan"),
                "precision": float("nan"), "fdr": float("nan")}

    y     = df[target_col]
    y_hat = df[pred_col]

    pos_pred = (y_hat == favorable_label)
    actual_pos = (y == favorable_label)
    actual_neg = ~actual_pos

    tp = (pos_pred & actual_pos).sum()
    fp = (pos_pred & actual_neg).sum()
    fn = (~pos_pred & actual_pos).sum()
    tn = (~pos_pred & actual_neg).sum()

    selection_rate = pos_pred.sum() / n
    tpr  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    # Predictive parity: precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    # False discovery rate = FP / (TP + FP)
    fdr = fp / (tp + fp) if (tp + fp) > 0 else float("nan")

    return {
        "n": int(n),
        "selection_rate": round(selection_rate, 6),
        "tpr":  round(tpr, 6)  if not math.isnan(tpr)  else None,
        "fpr":  round(fpr, 6)  if not math.isnan(fpr)  else None,
        "precision": round(precision, 6) if not math.isnan(precision) else None,
        "fdr": round(fdr, 6)   if not math.isnan(fdr)   else None,
    }


def _theil_index(y_pred_series: pd.Series, favorable_label: int) -> float:
    """Compute Theil Index on binary predictions."""
    vals = (y_pred_series == favorable_label).astype(float)
    mean = vals.mean()
    if mean == 0 or vals.std() == 0:
        return 0.0
    terms = []
    for v in vals:
        if v > 0 and mean > 0:
            ratio = v / mean
            terms.append(ratio * math.log(ratio))
    return round(sum(terms) / len(vals), 6) if terms else 0.0


def compute_attribute_metrics(
    df: pd.DataFrame,
    attr: str,
    privileged_value: str,
    target_col: str,
    pred_col: str,
    favorable_label: int,
) -> dict[str, Any]:
    """
    Compute all standard + extended metrics for one protected attribute.
    Uses the group with the LOWEST selection rate as the unprivileged group
    when multiple unprivileged groups exist (consistent with fairlearn convention).
    """
    # Normalise privileged value for comparison
    col_vals = df[attr].astype(str)
    priv_mask = col_vals == str(privileged_value)
    priv_df   = df[priv_mask]
    unpriv_df = df[~priv_mask]

    # Per-group breakdown (for multi-group attrs)
    unique_vals = col_vals.unique().tolist()
    group_stats: dict[str, dict] = {}
    for v in unique_vals:
        group_stats[v] = _rates(df[col_vals == v], pred_col, target_col, favorable_label)

    priv_rates   = _rates(priv_df,   pred_col, target_col, favorable_label)
    # For unprivileged: pick group with lowest selection rate
    unpriv_vals  = [v for v in unique_vals if str(v) != str(privileged_value)]
    if not unpriv_vals:
        unpriv_rates = priv_rates  # degenerate case
        worst_unpriv = str(privileged_value)
    else:
        worst_unpriv = min(
            unpriv_vals,
            key=lambda v: group_stats[v]["selection_rate"] or float("inf")
        )
        unpriv_rates = group_stats[worst_unpriv]

    sr_priv   = priv_rates["selection_rate"]
    sr_unpriv = unpriv_rates["selection_rate"]

    # Disparate Impact
    di = round(sr_unpriv / sr_priv, 6) if sr_priv and sr_priv > 0 else None

    # Demographic Parity Difference
    dpd = round(sr_unpriv - sr_priv, 6) if sr_priv is not None else None

    # Equal Opportunity Difference (TPR gap)
    tpr_priv   = priv_rates["tpr"]
    tpr_unpriv = unpriv_rates["tpr"]
    eod = round(tpr_unpriv - tpr_priv, 6) \
          if tpr_priv is not None and tpr_unpriv is not None else None

    # Average Odds Difference
    fpr_priv   = priv_rates["fpr"]
    fpr_unpriv = unpriv_rates["fpr"]
    if all(v is not None for v in [tpr_priv, tpr_unpriv, fpr_priv, fpr_unpriv]):
        aod = round(0.5 * ((fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv)), 6)
    else:
        aod = None

    # Theil Index
    theil = _theil_index(df[pred_col], favorable_label)

    # Predictive Parity Difference (precision gap)
    prec_priv   = priv_rates["precision"]
    prec_unpriv = unpriv_rates["precision"]
    ppd = round(prec_unpriv - prec_priv, 6) \
          if prec_priv is not None and prec_unpriv is not None else None

    # False Discovery Rate Difference
    fdr_priv   = priv_rates["fdr"]
    fdr_unpriv = unpriv_rates["fdr"]
    fdrd = round(fdr_unpriv - fdr_priv, 6) \
           if fdr_priv is not None and fdr_unpriv is not None else None

    # Statistical reliability flags
    reliability_warnings: list[str] = []
    for v, stats in group_stats.items():
        if stats["n"] < MIN_RELIABLE_N:
            reliability_warnings.append(
                f"Group '{v}' has only {stats['n']} samples — metrics unreliable (n < {MIN_RELIABLE_N})"
            )

    # Bias flag (DI < 0.8 or DI > 1.25 for both directions)
    bias_flag = (di is not None) and (di < 0.8 or di > 1.25)

    # Severity
    severity = _severity(di)

    return {
        "attribute": attr,
        "privileged_value": str(privileged_value),
        "worst_unprivileged_group": str(worst_unpriv),
        "group_stats": group_stats,
        # Core metrics
        "disparate_impact":          di,
        "demographic_parity_diff":   dpd,
        "equal_opportunity_diff":    eod,
        "average_odds_diff":         aod,
        "theil_index":               theil,
        # Extended metrics
        "predictive_parity_diff":    ppd,
        "false_discovery_rate_diff": fdrd,
        # Meta
        "bias_flag":                 bias_flag,
        "severity":                  severity,
        "reliability_warnings":      reliability_warnings,
        "n_privileged":              int(priv_rates["n"]),
        "n_unprivileged":            int(unpriv_rates["n"]),
    }


def compute_intersectional_metrics(
    df: pd.DataFrame,
    protected_configs: dict[str, str],
    target_col: str,
    pred_col: str,
    favorable_label: int,
) -> list[dict[str, Any]]:
    """
    Compute metrics for all pairwise combinations of protected attributes.
    Each intersectional group is compared against the joint privileged group
    (all attributes at their privileged value simultaneously).
    """
    attrs = list(protected_configs.keys())
    results = []

    for a1, a2 in itertools.combinations(attrs, 2):
        priv1 = str(protected_configs[a1])
        priv2 = str(protected_configs[a2])

        col1 = df[a1].astype(str)
        col2 = df[a2].astype(str)

        priv_mask  = (col1 == priv1) & (col2 == priv2)
        priv_df    = df[priv_mask]
        priv_rates = _rates(priv_df, pred_col, target_col, favorable_label)

        # Enumerate all unique (a1, a2) group pairs that aren't fully privileged
        pairs = df.groupby([a1, a2]).size().reset_index(name="n")
        group_results = []

        for _, row in pairs.iterrows():
            v1, v2, n = str(row[a1]), str(row[a2]), int(row["n"])
            if v1 == priv1 and v2 == priv2:
                continue  # skip privileged reference group
            group_df    = df[(col1 == v1) & (col2 == v2)]
            group_rates = _rates(group_df, pred_col, target_col, favorable_label)
            sr_g   = group_rates["selection_rate"]
            sr_p   = priv_rates["selection_rate"]
            di_g   = round(sr_g / sr_p, 6) if sr_p and sr_p > 0 else None
            dpd_g  = round(sr_g - sr_p, 6) if sr_p is not None else None
            group_results.append({
                "group": f"{a1}={v1} & {a2}={v2}",
                "n": n,
                "selection_rate": sr_g,
                "disparate_impact": di_g,
                "demographic_parity_diff": dpd_g,
                "unreliable": n < MIN_RELIABLE_N,
            })

        # Worst intersectional group (lowest DI among reliable groups)
        reliable = [g for g in group_results if not g["unreliable"] and g["disparate_impact"] is not None]
        worst = min(reliable, key=lambda g: g["disparate_impact"]) if reliable else None

        results.append({
            "pair": f"{a1} × {a2}",
            "attr1": a1, "attr2": a2,
            "privileged_reference": f"{a1}={priv1} & {a2}={priv2}",
            "n_privileged_joint": int(priv_rates["n"]),
            "groups": group_results,
            "worst_group": worst,
            "intersectional_bias_detected": worst is not None and worst["disparate_impact"] < 0.8,
        })

    return results


def _severity(di: float | None) -> str:
    if di is None:
        return "UNKNOWN"
    if di < 0.72:
        return "CRITICAL"
    if di < 0.80:
        return "HIGH"
    if di < 0.95:
        return "MODERATE"
    return "LOW"


def compute_all_metrics(
    df: pd.DataFrame,
    protected_configs: dict[str, str],
    target_col: str,
    pred_col: str,
    favorable_label: int,
) -> dict[str, Any]:
    """Top-level metric computation — returns the full payload for Pass 2."""
    per_attr = [
        compute_attribute_metrics(df, attr, priv, target_col, pred_col, favorable_label)
        for attr, priv in protected_configs.items()
    ]
    intersectional = compute_intersectional_metrics(
        df, protected_configs, target_col, pred_col, favorable_label
    )

    # Global Theil (across all predictions, not per-attr)
    global_theil = _theil_index(df[pred_col], favorable_label)

    return {
        "dataset_summary": {
            "n_total": len(df),
            "overall_selection_rate": round((df[pred_col] == favorable_label).mean(), 6),
            "global_theil_index": global_theil,
            "favorable_label": favorable_label,
        },
        "per_attribute": per_attr,
        "intersectional": intersectional,
    }


# ═══════════════════════════════════════════════════════════════════════
#  JSON Schemas for Structured Outputs
# ═══════════════════════════════════════════════════════════════════════

NARRATIVE_SCHEMA = {
    "name": "fairness_narrative",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "narrative": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "attribute":           {"type": "string"},
                        "severity":            {"type": "string", "enum": ["CRITICAL", "HIGH", "MODERATE", "LOW", "UNKNOWN"]},
                        "what_is_wrong":       {"type": "string"},
                        "why_is_wrong":        {"type": "string"},
                        "proxy_features":      {"type": "array", "items": {"type": "string"}},
                        "sample_size_warnings":{"type": "array", "items": {"type": "string"}},
                        "regulatory_flags": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "regulation": {"type": "string"},
                                    "relevance":  {"type": "string"},
                                    "risk_level": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}
                                },
                                "required": ["regulation", "relevance", "risk_level"],
                                "additionalProperties": False
                            }
                        },
                        "intersectional_findings": {"type": "string"}
                    },
                    "required": [
                        "attribute", "severity", "what_is_wrong", "why_is_wrong",
                        "proxy_features", "sample_size_warnings",
                        "regulatory_flags", "intersectional_findings"
                    ],
                    "additionalProperties": False
                }
            },
            "cross_attribute_patterns": {"type": "string"},
            "overall_risk_assessment":  {"type": "string"}
        },
        "required": ["narrative", "cross_attribute_patterns", "overall_risk_assessment"],
        "additionalProperties": False
    }
}

MITIGATION_SCHEMA = {
    "name": "fairness_mitigations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "mitigations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "attribute":   {"type": "string"},
                        "priority":    {"type": "integer"},
                        "strategies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name":        {"type": "string"},
                                    "phase":       {"type": "string", "enum": ["pre-processing", "in-processing", "post-processing"]},
                                    "algorithm":   {"type": "string"},
                                    "description": {"type": "string"},
                                    "tradeoffs":   {"type": "string"},
                                    "effort":      {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                                    "impact":      {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]}
                                },
                                "required": ["name", "phase", "algorithm", "description", "tradeoffs", "effort", "impact"],
                                "additionalProperties": False
                            }
                        },
                        "recommended_first_step": {"type": "string"}
                    },
                    "required": ["attribute", "priority", "strategies", "recommended_first_step"],
                    "additionalProperties": False
                }
            },
            "intersectional_mitigations": {"type": "string"},
            "implementation_roadmap":     {"type": "string"},
            "monitoring_recommendations": {"type": "string"}
        },
        "required": [
            "mitigations", "intersectional_mitigations",
            "implementation_roadmap", "monitoring_recommendations"
        ],
        "additionalProperties": False
    }
}


# ═══════════════════════════════════════════════════════════════════════
#  Prompts
# ═══════════════════════════════════════════════════════════════════════

def build_narrative_prompt(
    metrics_payload: dict[str, Any],
    dataset_name: str,
    protected_configs: dict[str, str],
) -> str:
    attr_desc = "\n".join(
        f"  - `{attr}` (privileged value: `{priv}`)"
        for attr, priv in protected_configs.items()
    )
    metrics_json = json.dumps(metrics_payload, indent=2)

    return f"""You are a senior AI fairness auditor conducting a rigorous blind audit.
You have been given PRE-COMPUTED, VERIFIED metrics from a deterministic Python
pipeline. Do NOT recompute or second-guess the numbers — your job is deep
qualitative analysis only.

Dataset: {dataset_name}
Protected attributes and their privileged values:
{attr_desc}

## Verified Metrics (Python-computed, trust these)
```json
{metrics_json}
```

## Your task — deep qualitative narrative

For EACH protected attribute, produce:

1. **what_is_wrong** — Describe the specific disparities visible in the metrics.
   Reference exact numbers. Be precise about which groups are disadvantaged
   and by how much.

2. **why_is_wrong** — Root causes. Go deep:
   - Historical bias and societal context for this specific attribute
   - Data collection biases
   - Feedback loops in the training process
   - Structural factors in the domain ({dataset_name})

3. **proxy_features** — List any features in a typical {dataset_name} dataset
   that could act as proxies for this protected attribute (e.g. zip code → race,
   income → age). Be specific and explain the proxy mechanism.

4. **sample_size_warnings** — Critically assess the statistical reliability
   of every group. Flag any group with n < {MIN_RELIABLE_N} as unreliable.
   Explain what conclusions CANNOT be drawn and why the pipeline's reliability
   flags for these groups should be respected. This is critical — do not let
   small-sample metrics drive severity classifications.

5. **regulatory_flags** — For each relevant regulation (ECOA, Fair Housing Act,
   Equal Credit Opportunity, EU AI Act Article 10, GDPR Article 22, CFPB
   guidance), state its relevance to this attribute in this domain and whether
   the observed disparity constitutes a HIGH, MEDIUM, or LOW legal risk.

6. **intersectional_findings** — Synthesise the intersectional analysis.
   Which joint-group combinations show the most severe compounded disadvantage?
   Which intersectional findings are reliable vs. small-sample noise?

Also provide:
- **cross_attribute_patterns** — Are there patterns across attributes suggesting
  a systemic model bias rather than isolated per-attribute issues?
- **overall_risk_assessment** — A brief executive summary of the model's overall
  fairness posture, suitable for a non-technical stakeholder.

Be rigorous, specific, and honest. Do not soften findings for political comfort.
"""


def build_mitigation_prompt(
    metrics_payload: dict[str, Any],
    narrative_result: dict[str, Any],
    dataset_name: str,
) -> str:
    metrics_json  = json.dumps(metrics_payload, indent=2)
    narrative_json = json.dumps(narrative_result, indent=2)

    return f"""You are a senior ML fairness engineer. You have:
1. Verified fairness metrics (Python-computed)
2. A deep qualitative audit from a fairness auditor

Your job: produce a PRIORITISED, ACTIONABLE mitigation roadmap.

Dataset: {dataset_name}

## Verified Metrics
```json
{metrics_json}
```

## Qualitative Audit
```json
{narrative_json}
```

## Your task — mitigation roadmap

For EACH protected attribute (ordered by severity, most critical first):

For each attribute, provide 2-4 mitigation strategies covering at least one
strategy from each phase where applicable:
- **pre-processing**: fix the training data before model training
- **in-processing**: constrain the model during training
- **post-processing**: adjust predictions after training

For each strategy:
- **name**: short name (e.g. "Reweighing", "Exponentiated Gradient")
- **algorithm**: specific library + class (e.g. "fairlearn.reductions.ExponentiatedGradient
  with DemographicParity constraint", "aif360.algorithms.preprocessing.Reweighing")
- **description**: what it does and why it helps for THIS specific attribute/disparity
- **tradeoffs**: be honest — what accuracy/performance is sacrificed? What
  fairness metrics improve vs. which may worsen? Are there any unintended
  consequences for other groups?
- **effort**: LOW / MEDIUM / HIGH (engineering effort to implement)
- **impact**: LOW / MEDIUM / HIGH (expected fairness improvement)
- **recommended_first_step**: which single strategy to try first and why

Also provide:
- **intersectional_mitigations**: specific strategies for the intersectional
  disparities found. Standard per-attribute fixes often worsen intersectional
  bias — address this explicitly.
- **implementation_roadmap**: ordered sequence of steps across the full pipeline
  (data → training → evaluation → deployment → monitoring).
- **monitoring_recommendations**: what to measure in production to detect
  fairness regression. Include specific metrics, cadence, and alert thresholds.

Prioritise by: severity first, then ease of implementation, then breadth of
impact across multiple attributes simultaneously.
"""


# ═══════════════════════════════════════════════════════════════════════
#  OpenAI API calls
# ═══════════════════════════════════════════════════════════════════════

def _call_openai(
    client,
    model: str,
    system_prompt: str,
    user_content: str,
    json_schema: dict,
    pass_name: str,
) -> tuple[dict, dict]:
    """
    Single OpenAI call with Structured Outputs + retry logic.
    Returns (parsed_json, usage_stats).
    """
    t0 = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"[{pass_name}] Calling {model} (attempt {attempt}/{MAX_RETRIES}) ...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema,
                },
            )
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = RETRY_BACKOFF_BASE * attempt
                log.warning(f"[{pass_name}] Rate limited. Retrying in {wait}s ...")
                time.sleep(wait)
                if attempt == MAX_RETRIES:
                    raise
            else:
                raise

    elapsed = time.time() - t0
    usage   = response.usage
    in_tok  = usage.prompt_tokens
    out_tok = usage.completion_tokens
    total   = in_tok + out_tok
    cost    = (in_tok / 1_000_000) * PRICE_INPUT_PER_M + \
              (out_tok / 1_000_000) * PRICE_OUTPUT_PER_M

    usage_stats = {
        "pass":            pass_name,
        "model":           model,
        "wall_clock_s":    round(elapsed, 2),
        "input_tokens":    in_tok,
        "output_tokens":   out_tok,
        "total_tokens":    total,
        "cost_usd":        round(cost, 6),
    }
    log.info(
        f"[{pass_name}] Done in {elapsed:.1f}s | "
        f"{in_tok:,} in + {out_tok:,} out | est. ${cost:.4f}"
    )

    raw = response.choices[0].message.content
    # Structured Outputs should never need cleanup, but be defensive
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    raw = re.sub(r"\n?```\s*$", "", raw)
    return json.loads(raw), usage_stats


def run_three_pass_pipeline(
    metrics_payload: dict[str, Any],
    protected_configs: dict[str, str],
    dataset_name: str,
    model: str,
    api_key: str,
) -> tuple[dict, dict, list[dict]]:
    """
    Run the full 3-pass o3 pipeline.
    Returns (narrative_result, mitigation_result, usage_stats_list).
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    all_usage: list[dict] = []

    # ── Pass 2: Narrative ─────────────────────────────────────────────
    narrative_prompt = build_narrative_prompt(
        metrics_payload, dataset_name, protected_configs
    )
    narrative_result, usage2 = _call_openai(
        client, model,
        system_prompt="You are a rigorous AI fairness auditor. Follow the schema exactly.",
        user_content=narrative_prompt,
        json_schema=NARRATIVE_SCHEMA,
        pass_name="Pass2-Narrative",
    )
    all_usage.append(usage2)

    # ── Pass 3: Mitigations ───────────────────────────────────────────
    mitigation_prompt = build_mitigation_prompt(
        metrics_payload, narrative_result, dataset_name
    )
    mitigation_result, usage3 = _call_openai(
        client, model,
        system_prompt="You are a senior ML fairness engineer. Follow the schema exactly.",
        user_content=mitigation_prompt,
        json_schema=MITIGATION_SCHEMA,
        pass_name="Pass3-Mitigations",
    )
    all_usage.append(usage3)

    return narrative_result, mitigation_result, all_usage


# ═══════════════════════════════════════════════════════════════════════
#  Report generators
# ═══════════════════════════════════════════════════════════════════════

def _usage_block(usage_list: list[dict]) -> list[str]:
    total_cost = sum(u["cost_usd"] for u in usage_list)
    total_tok  = sum(u["total_tokens"] for u in usage_list)
    lines = [
        "## API Usage",
        "",
        "| Pass | Model | Time | In tokens | Out tokens | Cost |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for u in usage_list:
        lines.append(
            f"| {u['pass']} | `{u['model']}` | {u['wall_clock_s']}s "
            f"| {u['input_tokens']:,} | {u['output_tokens']:,} | ${u['cost_usd']:.4f} |"
        )
    lines += [
        f"| **Total** | | | | {total_tok:,} | **${total_cost:.4f}** |",
        "",
        f"> Pricing: ${PRICE_INPUT_PER_M}/1M input tokens, "
        f"${PRICE_OUTPUT_PER_M}/1M output tokens",
        "",
    ]
    return lines


def generate_llm_report(
    metrics_payload: dict[str, Any],
    narrative_result: dict[str, Any],
    mitigation_result: dict[str, Any],
    dataset_name: str,
    model: str,
    usage_list: list[dict],
) -> str:
    """Generate the standalone LLM audit report markdown."""
    lines = [
        f"# OpenAI Fairness Audit: {dataset_name}",
        "",
        f"**Model:** `{model}` (3-pass pipeline)",
        "**Method:** Python-verified metrics → o3 narrative → o3 mitigations",
        "**Framework references:** fairlearn, aif360",
        "",
    ]
    lines += _usage_block(usage_list)

    # ── Metrics summary table ─────────────────────────────────────────
    lines += [
        "## Verified Metrics (Python-computed)",
        "",
        f"**Total samples:** {metrics_payload['dataset_summary']['n_total']:,}  ",
        f"**Overall selection rate:** {metrics_payload['dataset_summary']['overall_selection_rate']:.4f}  ",
        f"**Global Theil Index:** {metrics_payload['dataset_summary']['global_theil_index']:.4f}",
        "",
        "| Attribute | Severity | DI | DPD | EOD | AOD | Theil | PPD | FDRD | Bias? |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in metrics_payload["per_attribute"]:
        def _f(v): return f"{v:.4f}" if v is not None else "N/A"
        lines.append(
            f"| {m['attribute']} | {m['severity']} "
            f"| {_f(m['disparate_impact'])} "
            f"| {_f(m['demographic_parity_diff'])} "
            f"| {_f(m['equal_opportunity_diff'])} "
            f"| {_f(m['average_odds_diff'])} "
            f"| {_f(m['theil_index'])} "
            f"| {_f(m['predictive_parity_diff'])} "
            f"| {_f(m['false_discovery_rate_diff'])} "
            f"| {'⚠ Yes' if m['bias_flag'] else 'No'} |"
        )
    lines.append("")

    # Reliability warnings
    all_warnings = []
    for m in metrics_payload["per_attribute"]:
        for w in m.get("reliability_warnings", []):
            all_warnings.append(f"- **{m['attribute']}**: {w}")
    if all_warnings:
        lines += ["### ⚠ Statistical Reliability Warnings", ""] + all_warnings + [""]

    # Intersectional summary
    lines += ["## Intersectional Analysis", ""]
    for inter in metrics_payload["intersectional"]:
        lines.append(f"### {inter['pair']}")
        lines.append(f"Privileged reference: `{inter['privileged_reference']}` (n={inter['n_privileged_joint']})")
        lines.append("")
        lines.append("| Group | n | Selection Rate | DI | DPD | Reliable? |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for g in inter["groups"]:
            lines.append(
                f"| {g['group']} | {g['n']} "
                f"| {g['selection_rate']:.4f} "
                f"| {g['disparate_impact']:.4f} if g['disparate_impact'] is not None else 'N/A'} "
                f"| {g['demographic_parity_diff']:.4f} if g['demographic_parity_diff'] is not None else 'N/A'} "
                f"| {'No ⚠' if g['unreliable'] else 'Yes'} |"
            )
        if inter["worst_group"]:
            wg = inter["worst_group"]
            lines.append(f"\n**Worst reliable group:** `{wg['group']}` — DI={wg['disparate_impact']:.4f}")
        lines.append(f"**Intersectional bias detected:** {'Yes ⚠' if inter['intersectional_bias_detected'] else 'No'}")
        lines.append("")

    # ── Qualitative narrative ─────────────────────────────────────────
    lines += ["## Qualitative Analysis (o3 Pass 2)", ""]
    for n_item in narrative_result.get("narrative", []):
        lines += [
            "---",
            f"## {n_item['attribute']}  (severity: {n_item['severity']})",
            "",
            "### What is wrong", "",
            n_item["what_is_wrong"], "",
            "### Why it is wrong", "",
            n_item["why_is_wrong"], "",
        ]
        if n_item.get("proxy_features"):
            lines += ["### Proxy Features Detected", ""]
            for pf in n_item["proxy_features"]:
                lines.append(f"- {pf}")
            lines.append("")
        if n_item.get("sample_size_warnings"):
            lines += ["### Sample Size Warnings", ""]
            for w in n_item["sample_size_warnings"]:
                lines.append(f"- ⚠ {w}")
            lines.append("")
        if n_item.get("regulatory_flags"):
            lines += ["### Regulatory Risk", "",
                      "| Regulation | Relevance | Risk Level |",
                      "|---|---|---|"]
            for rf in n_item["regulatory_flags"]:
                lines.append(f"| {rf['regulation']} | {rf['relevance']} | {rf['risk_level']} |")
            lines.append("")
        if n_item.get("intersectional_findings"):
            lines += ["### Intersectional Findings", "",
                      n_item["intersectional_findings"], ""]

    lines += [
        "### Cross-Attribute Patterns", "",
        narrative_result.get("cross_attribute_patterns", ""), "",
        "### Overall Risk Assessment", "",
        narrative_result.get("overall_risk_assessment", ""), "",
    ]

    # ── Mitigations ───────────────────────────────────────────────────
    lines += ["## Mitigation Roadmap (o3 Pass 3)", ""]
    for mit in sorted(
        mitigation_result.get("mitigations", []),
        key=lambda x: x.get("priority", 99)
    ):
        lines += [f"### Priority {mit['priority']}: {mit['attribute']}", ""]
        for strat in mit.get("strategies", []):
            lines += [
                f"#### {strat['name']} ({strat['phase']})",
                "",
                f"**Algorithm:** `{strat['algorithm']}`  ",
                f"**Effort:** {strat['effort']} | **Impact:** {strat['impact']}",
                "",
                strat["description"],
                "",
                f"**Tradeoffs:** {strat['tradeoffs']}",
                "",
            ]
        lines += [
            f"**Recommended first step:** {mit['recommended_first_step']}", ""
        ]

    lines += [
        "### Intersectional Mitigations", "",
        mitigation_result.get("intersectional_mitigations", ""), "",
        "### Implementation Roadmap", "",
        mitigation_result.get("implementation_roadmap", ""), "",
        "### Monitoring Recommendations", "",
        mitigation_result.get("monitoring_recommendations", ""), "",
    ]

    return "\n".join(lines)


def generate_comparison(
    metrics_payload: dict[str, Any],
    narrative_result: dict[str, Any],
    mitigation_result: dict[str, Any],
    our_fairness_csv: Path,
    our_qualitative_report: Path,
    dataset_name: str,
    model: str,
    usage_list: list[dict],
) -> str:
    """Build a side-by-side comparison markdown report (pipeline-compatible)."""
    our_df   = pd.read_csv(our_fairness_csv)
    our_qual = our_qualitative_report.read_text(encoding="utf-8") \
               if our_qualitative_report.exists() else ""

    lines = [
        f"# Benchmark Comparison: {dataset_name}",
        "",
        "Deterministic pipeline (AIF360) vs. OpenAI o3 (fairlearn + extended metrics).",
        f"**Auditor model:** `{model}` | **Architecture:** 3-pass (Python metrics → o3 narrative → o3 mitigations)",
        "",
    ]
    lines += _usage_block(usage_list)

    # ── Metric comparison ─────────────────────────────────────────────
    lines += [
        "## Metric Comparison",
        "",
        "| Attribute | Metric | Our Pipeline | OpenAI Auditor | Delta |",
        "|---|---|---:|---:|---:|",
    ]

    metric_map = {
        "disparate_impact":          "DisparateImpact",
        "demographic_parity_diff":   "DemographicParityDiff",
        "equal_opportunity_diff":    "EqualOpportunityDiff",
        "average_odds_diff":         "AverageOddsDiff",
        "theil_index":               "TheilIndex",
    }
    display_names = {
        "disparate_impact":          "Disparate Impact",
        "demographic_parity_diff":   "Demographic Parity Diff",
        "equal_opportunity_diff":    "Equal Opportunity Diff",
        "average_odds_diff":         "Average Odds Diff",
        "theil_index":               "Theil Index",
    }

    for m in metrics_payload["per_attribute"]:
        attr     = m["attribute"]
        our_row  = _find_our_row(our_df, attr)
        for key, csv_key in metric_map.items():
            our_val  = _safe_float(our_row.get(csv_key)) if our_row else None
            our_val_computed = m.get(key)
            our_str  = f"{our_val:.4f}" if our_val is not None else "--"
            aud_str  = f"{our_val_computed:.4f}" if our_val_computed is not None else "--"
            delta_str = (
                f"{our_val_computed - our_val:+.4f}"
                if our_val is not None and our_val_computed is not None else "--"
            )
            lines.append(
                f"| {attr} | {display_names[key]} | {our_str} | {aud_str} | {delta_str} |"
            )

    # Extended metrics (new — not in Gemini version)
    lines += [
        "",
        "### Extended Metrics (OpenAI Auditor only)",
        "",
        "| Attribute | PPD | FDRD | Worst Unprivileged Group |",
        "|---|---:|---:|---|",
    ]
    for m in metrics_payload["per_attribute"]:
        def _f(v): return f"{v:.4f}" if v is not None else "N/A"
        lines.append(
            f"| {m['attribute']} | {_f(m['predictive_parity_diff'])} "
            f"| {_f(m['false_discovery_rate_diff'])} "
            f"| {m['worst_unprivileged_group']} |"
        )
    lines.append("")

    # ── Severity comparison ───────────────────────────────────────────
    lines += [
        "## Severity Comparison",
        "",
        "| Attribute | Our Pipeline | OpenAI Auditor | Agreement? |",
        "|---|---|---|---|",
    ]
    our_severities = _extract_severities(our_qual)
    nar_by_attr    = {n["attribute"]: n for n in narrative_result.get("narrative", [])}
    for m in metrics_payload["per_attribute"]:
        attr    = m["attribute"]
        aud_sev = m["severity"]  # Python-computed severity (canonical)
        our_sev = our_severities.get(attr, our_severities.get(attr + "_original", "--"))
        agree   = "✓" if _normalize_severity(aud_sev) == _normalize_severity(our_sev) else "✗"
        lines.append(f"| {attr} | {our_sev} | {aud_sev} | {agree} |")
    lines.append("")

    # ── Reliability flags new to this auditor ────────────────────────
    all_warnings = []
    for m in metrics_payload["per_attribute"]:
        for w in m.get("reliability_warnings", []):
            all_warnings.append(f"- **{m['attribute']}**: {w}")
    if all_warnings:
        lines += [
            "## Statistical Reliability Flags (new in OpenAI auditor)",
            "",
            "These flags did not exist in the Gemini version:",
            "",
        ] + all_warnings + [""]

    # ── Qualitative narrative comparison ─────────────────────────────
    lines += ["## Qualitative Narrative Comparison", ""]
    our_sections = _extract_qualitative_sections(our_qual)

    for n_item in narrative_result.get("narrative", []):
        attr    = n_item["attribute"]
        our_sec = our_sections.get(attr, our_sections.get(attr + "_original", {}))

        lines += [f"### {attr}", ""]

        for field, label in [
            ("what_is_wrong", "What is wrong"),
            ("why_is_wrong",  "Why it is wrong"),
        ]:
            lines += [
                f"#### {label}", "",
                f"**Our pipeline:** {our_sec.get(field, 'N/A')}", "",
                f"**OpenAI Auditor:** {n_item[field]}", "",
            ]

        # Proxy features — new vs Gemini
        if n_item.get("proxy_features"):
            lines += ["#### Proxy Features (new — not in Gemini version)", ""]
            for pf in n_item["proxy_features"]:
                lines.append(f"- {pf}")
            lines.append("")

        # Regulatory framing — new vs Gemini
        if n_item.get("regulatory_flags"):
            lines += [
                "#### Regulatory Framing (new — not in Gemini version)", "",
                "| Regulation | Relevance | Risk |",
                "|---|---|---|",
            ]
            for rf in n_item["regulatory_flags"]:
                lines.append(f"| {rf['regulation']} | {rf['relevance']} | {rf['risk_level']} |")
            lines.append("")

    lines += [
        "## Overall Risk Assessment (o3)", "",
        narrative_result.get("overall_risk_assessment", ""), "",
        "## Implementation Roadmap (o3)", "",
        mitigation_result.get("implementation_roadmap", ""), "",
    ]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers (shared with Gemini version)
# ═══════════════════════════════════════════════════════════════════════

def _find_our_row(our_df: pd.DataFrame, attr: str) -> dict | None:
    col        = "Attribute"
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
    result = {}
    for match in re.finditer(r"##\s+(\S+)\s+\(DI\s*=.*?severity:\s*(.+?)\)", report_md):
        result[match.group(1)] = match.group(2).strip().rstrip(")")
    return result


def _extract_qualitative_sections(report_md: str) -> dict[str, dict]:
    sections: dict[str, dict] = {}
    for block in re.split(r"\n---\n## ", report_md):
        attr_match = re.match(r"(\S+)\s+\(", block)
        if not attr_match:
            continue
        attr = attr_match.group(1)
        sec: dict[str, str] = {}
        for field, pattern in [
            ("what_is_wrong", r"### What is wrong\n\n(.*?)(?=\n### |\n---|\Z)"),
            ("why_is_wrong",  r"### Why it is wrong\n\n(.*?)(?=\n### |\n---|\Z)"),
            ("how_to_fix",    r"### How to fix it\n\n(.*?)(?=\n### |\n---|\Z)"),
        ]:
            m = re.search(pattern, block, re.DOTALL)
            if m:
                sec[field] = m.group(1).strip()
        sections[attr] = sec
    return sections


# ═══════════════════════════════════════════════════════════════════════
#  Public API (drop-in compatible with Gemini version)
# ═══════════════════════════════════════════════════════════════════════

def run_llm_benchmark(
    predictions_path:       str | Path,
    fairness_csv_path:      str | Path,
    qualitative_report_path: str | Path,
    target_col:             str,
    pred_col:               str,
    favorable_label:        int,
    protected_configs:      dict[str, str],
    out_dir:                str | Path,
    dataset_name:           str = "Dataset",
    model:                  str = DEFAULT_MODEL,
) -> Path:
    """
    Run the full OpenAI 3-pass benchmark and write reports.
    Returns the comparison report path (drop-in compatible with Gemini version).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Export it or add to a .env file."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_path)

    # ── Pass 1: Python computes all metrics ───────────────────────────
    log.info("Pass 1: Computing all metrics deterministically ...")
    t0 = time.time()
    metrics_payload = compute_all_metrics(
        df, protected_configs, target_col, pred_col, favorable_label
    )
    log.info(f"Pass 1 complete in {time.time()-t0:.2f}s")

    # Save metrics payload
    metrics_path = out_dir / "llm_verified_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    log.info(f"Verified metrics saved to {metrics_path}")

    # ── Passes 2 & 3: o3 narrative + mitigations ─────────────────────
    narrative_result, mitigation_result, usage_list = run_three_pass_pipeline(
        metrics_payload, protected_configs, dataset_name, model, api_key
    )

    # Save raw responses
    raw_path = out_dir / "llm_raw_response.json"
    raw_path.write_text(
        json.dumps({
            "metrics":    metrics_payload,
            "narrative":  narrative_result,
            "mitigations": mitigation_result,
            "usage":      usage_list,
        }, indent=2),
        encoding="utf-8",
    )
    log.info(f"Raw responses saved to {raw_path}")

    # Save prompts for reproducibility
    prompts = {
        "pass2_narrative":   build_narrative_prompt(metrics_payload, dataset_name, protected_configs),
        "pass3_mitigations": build_mitigation_prompt(metrics_payload, narrative_result, dataset_name),
    }
    (out_dir / "llm_prompt.txt").write_text(
        "\n\n" + "="*80 + "\n\n".join(
            f"=== {k.upper()} ===\n\n{v}" for k, v in prompts.items()
        ),
        encoding="utf-8",
    )

    # ── Generate reports ──────────────────────────────────────────────
    llm_report = generate_llm_report(
        metrics_payload, narrative_result, mitigation_result,
        dataset_name, model, usage_list,
    )
    llm_path = out_dir / "llm_fairness_report.md"
    llm_path.write_text(llm_report, encoding="utf-8")
    log.info(f"LLM report saved to {llm_path}")

    comparison = generate_comparison(
        metrics_payload, narrative_result, mitigation_result,
        Path(fairness_csv_path), Path(qualitative_report_path),
        dataset_name, model, usage_list,
    )
    comparison_path = out_dir / "benchmark_comparison.md"
    comparison_path.write_text(comparison, encoding="utf-8")
    log.info(f"Benchmark comparison saved to {comparison_path}")

    # ── Print cost summary ────────────────────────────────────────────
    total_cost = sum(u["cost_usd"] for u in usage_list)
    log.info(f"Total estimated cost: ${total_cost:.4f} across {len(usage_list)} passes")

    return comparison_path


# ═══════════════════════════════════════════════════════════════════════
#  CLI (drop-in compatible with Gemini version)
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI o3 fairness benchmark (3-pass blind auditor)",
    )
    parser.add_argument("--predictions",        required=True,
                        help="Path to predictions CSV")
    parser.add_argument("--fairness_csv",        required=True,
                        help="Path to our fairness_metrics.csv (for comparison only)")
    parser.add_argument("--qualitative_report",  required=True,
                        help="Path to our qualitative_report.md (for comparison only)")
    parser.add_argument("--target",              required=True,
                        help="Target column name")
    parser.add_argument("--pred_col",            default="predicted",
                        help="Prediction column name")
    parser.add_argument("--favorable_label",     type=int, default=1,
                        help="Favorable label value")
    parser.add_argument("--protected_attrs",     required=True,
                        help="Comma-separated attr:privileged pairs, "
                             "e.g. 'Sex_original:male,AgeGroup_original:40_plus'")
    parser.add_argument("--out_dir",             default=".",
                        help="Output directory")
    parser.add_argument("--dataset_name",        default="Dataset",
                        help="Name for the report header")
    parser.add_argument("--model",               default=DEFAULT_MODEL,
                        help="OpenAI model to use (default: o3)")
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
    )


if __name__ == "__main__":
    main()