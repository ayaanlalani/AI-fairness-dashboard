"""
llm_benchmark_common.py

Shared helpers for qualitative-only LLM fairness benchmarking.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from qualitative_analysis import (
    _get_attr_metrics,
    classify_severity,
    detect_proxy_features,
    diagnose_imbalance,
    map_root_causes,
    per_group_breakdown,
)
from scholarly_evidence import build_attribute_queries, gather_research_context

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  Guardrail gate + cost enforcement
#
#  Both live here rather than in llm_benchmark.py because every script that
#  reads an LLM key already imports this module. Before this, the gate existed
#  only in llm_benchmark.py while openai_fairness_analysis.py,
#  openai_hybrid_self_improve.py and llm_fairness_analysis.py read
#  OPENAI_API_KEY/GEMINI_API_KEY with no gate at all — so the standing
#  verification claim in docs/RESEARCH_STAGING_PROMPT.md §4 ("no new file
#  reads an LLM key outside the gated client-construction path") was not true.
# ═══════════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).resolve().parent.parent
GUARDRAILS_PATH = REPO_ROOT / "configs" / "research_guardrails.json"
SPEND_LEDGER_PATH = REPO_ROOT / "artifacts" / "llm_benchmark" / "spend_ledger.json"

# Per-1M-token list prices, USD. Keep in one place so cost estimates and
# post-hoc accounting cannot drift apart.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3": (2.00, 8.00),
    "gemini-2.5-flash": (0.10, 0.40),
}
_DEFAULT_PRICE = (2.50, 10.00)


class CostCapExceeded(RuntimeError):
    """Raised when a call would push cumulative spend past the configured cap."""


def load_guardrails() -> dict[str, Any]:
    """Read the guardrail config. Missing or unparseable files fail closed."""
    if not GUARDRAILS_PATH.exists():
        return {}
    try:
        return json.loads(GUARDRAILS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def provider_for_model(model: str) -> str:
    return "openai" if model.startswith(("gpt", "o1", "o3")) else "gemini"


def enforce_llm_gate(model: str) -> None:
    """Refuse live LLM calls unless the guardrail config approves the provider.

    Gate policy: docs/RESEARCH_STAGING_PROMPT.md §0. Fails closed.
    """
    provider = provider_for_model(model)
    status = load_guardrails().get("llm_providers", {}).get(provider, "blocked")
    if status != "approved":
        raise RuntimeError(
            f"Guardrail gate: provider '{provider}' is '{status}' in {GUARDRAILS_PATH}. "
            "Live LLM calls require the user to flip it to 'approved'. "
            "Use --dry-run to build prompt packs without any API call."
        )


def guardrail_max_cost_usd(default: float | None = None) -> float | None:
    """Return `max_cost_usd_per_run` from the guardrail config, if set."""
    raw = load_guardrails().get("max_cost_usd_per_run", default)
    try:
        return None if raw is None else float(raw)
    except (TypeError, ValueError):
        return default


def model_prices(model: str) -> tuple[float, float]:
    return MODEL_PRICES.get(model, _DEFAULT_PRICE)


def cost_for_tokens(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = model_prices(model)
    return (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out


def estimate_call_cost_usd(
    prompt: str, model: str, expected_output_tokens: int = 1200
) -> float:
    """Conservative pre-call cost estimate.

    Input tokens are approximated at 4 chars/token, then padded 15% so the cap
    trips *before* an overrun rather than after it.
    """
    approx_input = max(1, len(prompt) // 4)
    est = cost_for_tokens(model, approx_input, expected_output_tokens)
    return est * 1.15


class SpendLedger:
    """Cumulative spend tracker enforcing a hard USD cap.

    The cap must hold across the whole program, not per process: Stage 3 runs
    three separate `llm_benchmark.py` invocations, so a purely in-memory
    counter would let 3 x cap through. Spend is therefore persisted to
    `artifacts/llm_benchmark/spend_ledger.json` and re-read on construction.
    """

    def __init__(
        self,
        max_cost_usd: float | None,
        path: Path | None = None,
        run_label: str = "",
    ) -> None:
        self.max_cost_usd = max_cost_usd
        self.path = path or SPEND_LEDGER_PATH
        self.run_label = run_label
        self.entries: list[dict[str, Any]] = []
        self._prior_total = 0.0
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._prior_total = float(data.get("total_usd", 0.0) or 0.0)
                self.entries = list(data.get("entries", []))
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                log.warning("Spend ledger at %s unreadable; starting a fresh one.", self.path)

    @property
    def spent_usd(self) -> float:
        return round(self._prior_total, 6)

    @property
    def remaining_usd(self) -> float | None:
        if self.max_cost_usd is None:
            return None
        return round(max(0.0, self.max_cost_usd - self.spent_usd), 6)

    def assert_headroom(self, projected_usd: float, label: str = "") -> None:
        """Abort *before* a call that would breach the cap."""
        if self.max_cost_usd is None:
            return
        if self.spent_usd + projected_usd > self.max_cost_usd:
            raise CostCapExceeded(
                f"Cost cap reached: ${self.spent_usd:.4f} already spent, next call "
                f"({label or 'unlabelled'}) projected at ${projected_usd:.4f}, cap is "
                f"${self.max_cost_usd:.2f}. Aborting before the call. "
                f"Raise --max_cost_usd or max_cost_usd_per_run to continue."
            )

    def record(self, usage: dict[str, Any], **meta: Any) -> None:
        cost = float(usage.get("total_cost_usd", 0.0) or 0.0)
        if cost <= 0:
            return  # dry-run / failed call: nothing was billed
        self._prior_total += cost
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run": self.run_label,
            "cost_usd": round(cost, 6),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "model": usage.get("model", ""),
        }
        entry.update(meta)
        self.entries.append(entry)
        self.flush()

    def flush(self) -> None:
        if not self.entries:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "max_cost_usd": self.max_cost_usd,
                    "total_usd": round(self._prior_total, 6),
                    "entries": self.entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

REFERENCE_AUDIT_SPEC = {
    "name": "Reference Audit Specification",
    "core_metrics": [
        "Disparate Impact",
        "Demographic Parity Difference",
        "Equal Opportunity Difference",
        "Average Odds Difference",
        "Theil Index",
    ],
    "required_response_elements": [
        "per-group breakdown",
        "severity classification",
        "root-cause analysis",
        "mitigation recommendations",
        "research-backed justification",
    ],
    "why_this_spec": (
        "A remediation-ready fairness audit should combine quantitative disparity "
        "measurement with interpretable causes, explicit severity, and concrete "
        "recommended interventions."
    ),
}

ALGORITHM_KEYWORDS = [
    "reweighing",
    "disparateimpactremover",
    "prejudiceremover",
    "eqoddspostprocessing",
    "calibratedeqoddspostprocessing",
    "exponentiatedgradient",
    "gridsearch",
    "thresholdoptimizer",
    "equalized odds",
]

CAUSE_LABELS = {
    "historical / label bias": "historical_label_bias",
    "proxy discrimination": "proxy_discrimination",
    "representation bias": "representation_bias",
    "unequal opportunity": "unequal_opportunity",
    "unequal odds": "unequal_odds",
}


def build_context_and_baseline(
    predictions_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
    protected_configs: dict[str, str],
    target_col: str,
    pred_col: str,
    favorable_label: int,
    dataset_name: str,
    threshold: float = 0.8,
    preloaded_research: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Build shared quantitative context and deterministic qualitative baseline."""
    protected_attrs = list(protected_configs.keys())
    non_feature = {target_col, pred_col, "actual", "predicted"} | set(protected_attrs)
    feature_cols = [c for c in predictions_df.columns if c not in non_feature]

    if preloaded_research is not None:
        dataset_research = preloaded_research
    else:
        dataset_research = gather_research_context(
            dataset_name=dataset_name,
            protected_attrs=protected_attrs,
            max_papers=6,
        )

    context_attrs: list[dict[str, Any]] = []
    baseline_attrs: list[dict[str, Any]] = []

    for attr, privileged_value in protected_configs.items():
        metrics = _get_attr_metrics(fairness_df, attr)
        breakdown = per_group_breakdown(predictions_df, attr, target_col, pred_col, favorable_label)
        imbalance = diagnose_imbalance(breakdown, attr)
        di = metrics.get("DisparateImpact")
        dpd = metrics.get("DemographicParityDiff")
        eod = metrics.get("EqualOpportunityDiff")
        aod = metrics.get("AverageOddsDiff")
        theil = metrics.get("TheilIndex")
        severity = classify_severity(di, threshold)

        context_attrs.append({
            "attribute": attr,
            "privileged_value": str(privileged_value),
            "metrics": {
                "disparate_impact": _safe_num(di),
                "demographic_parity_diff": _safe_num(dpd),
                "equal_opportunity_diff": _safe_num(eod),
                "average_odds_diff": _safe_num(aod),
                "theil_index": _safe_num(theil),
            },
            "group_breakdown": breakdown.to_dict(orient="records"),
            "imbalance_findings": imbalance,
            "severity_thresholds": {
                "critical_if_di_below": round(threshold * 0.9, 2),
                "high_if_di_below": threshold,
                "moderate_if_di_below": 0.95,
            },
        })

        proxies = detect_proxy_features(predictions_df, attr, feature_cols)
        root = map_root_causes(di, dpd, eod, aod, imbalance, proxies, breakdown, threshold)
        if preloaded_research is not None:
            attr_research = preloaded_research
        else:
            attr_research = gather_research_context(
                dataset_name=dataset_name,
                protected_attrs=[attr],
                extra_queries=build_attribute_queries(dataset_name, attr, root["causes"], root["fixes"]),
                max_papers=3,
            )
        baseline_attrs.append({
            "attribute": attr,
            "severity": severity,
            "root_cause_labels": _extract_cause_labels(root["causes"]),
            "root_causes": root["causes"],
            "mitigations": root["fixes"],
            "research_titles": [paper["title"] for paper in attr_research],
        })

    context_payload = {
        "dataset_name": dataset_name,
        "favorable_label": favorable_label,
        "reference_audit_spec": REFERENCE_AUDIT_SPEC,
        "attributes": context_attrs,
    }
    baseline_payload = {
        "reference_audit_spec": REFERENCE_AUDIT_SPEC,
        "attributes": baseline_attrs,
    }
    return context_payload, baseline_payload, dataset_research


def build_score_summary(score: dict[str, Any]) -> str:
    return (
        f"Total score: {score['total_score']:.1f}/100. "
        f"Completeness={score['subscores']['completeness']:.1f}, "
        f"Severity agreement={score['subscores']['severity_agreement']:.1f}, "
        f"Cause alignment={score['subscores']['cause_alignment']:.1f}, "
        f"Mitigation specificity={score['subscores']['mitigation_specificity']:.1f}, "
        f"Research grounding={score['subscores']['research_grounding']:.1f}."
    )


def score_llm_output(result: dict[str, Any], baseline_payload: dict[str, Any]) -> dict[str, Any]:
    """Score a qualitative LLM output against the deterministic baseline."""
    baseline_by_attr = {
        row["attribute"]: row for row in baseline_payload.get("attributes", [])
    }
    result_by_attr = {
        row["attribute"]: row for row in result.get("qualitative", [])
    }

    completeness = _score_completeness(result, expected_attrs=list(baseline_by_attr.keys()))
    severity_agreement = _score_severity(result_by_attr, baseline_by_attr)
    cause_alignment = _score_cause_alignment(result_by_attr, baseline_by_attr)
    mitigation_specificity = _score_mitigation_specificity(result_by_attr)
    research_grounding = _score_research_grounding(result, result_by_attr)

    total = completeness + severity_agreement + cause_alignment + mitigation_specificity + research_grounding
    return {
        "total_score": round(total, 2),
        "subscores": {
            "completeness": round(completeness, 2),
            "severity_agreement": round(severity_agreement, 2),
            "cause_alignment": round(cause_alignment, 2),
            "mitigation_specificity": round(mitigation_specificity, 2),
            "research_grounding": round(research_grounding, 2),
        },
        "summary": build_score_summary({
            "total_score": total,
            "subscores": {
                "completeness": completeness,
                "severity_agreement": severity_agreement,
                "cause_alignment": cause_alignment,
                "mitigation_specificity": mitigation_specificity,
                "research_grounding": research_grounding,
            },
        }),
    }


def _extract_cause_labels(causes: list[str]) -> list[str]:
    labels: list[str] = []
    for cause in causes:
        lower = cause.lower()
        for phrase, label in CAUSE_LABELS.items():
            if phrase in lower:
                labels.append(label)
    return sorted(set(labels))


def _normalize_severity(value: str) -> str:
    value = (value or "").upper()
    for label in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        if label in value:
            return label
    return value.strip() or "UNKNOWN"


def _score_completeness(result: dict[str, Any], expected_attrs: list[str]) -> float:
    """Score structural completeness (40 pts max).

    Supports two output formats:
    - Multi-attribute: result has 'reference_audit_spec' + 'qualitative' list.
    - Single-attribute: result has 'qualitative' list with one item; no audit spec.

    Both formats award points for the same semantic qualities — identification,
    diagnosis depth, element coverage, justification depth, research backing,
    and attribute coverage — using format-appropriate field checks.
    """
    score = 0.0
    qualitative = result.get("qualitative", [])
    spec = result.get("reference_audit_spec")

    if spec:
        # ── multi-attribute format ────────────────────────────────────
        if spec.get("name"):
            score += 4
        if len(spec.get("core_metrics", [])) >= 4:
            score += 8
        if len(spec.get("required_response_elements", [])) >= 4:
            score += 8
        if spec.get("why_this_spec"):
            score += 5
        if len(spec.get("supporting_research", [])) >= 2:
            score += 5
    else:
        # ── single-attribute format ───────────────────────────────────
        # Score against the first qualitative entry (there should be exactly one).
        item = qualitative[0] if qualitative else {}
        # 4 pts: attribute identified
        if item.get("attribute"):
            score += 4
        # 8 pts: what_is_wrong is substantive (≥100 chars ≈ "≥4 metrics discussed")
        if len(str(item.get("what_is_wrong", ""))) >= 100:
            score += 8
        # 8 pts: all four narrative fields non-empty
        if all(item.get(f) for f in ("what_is_wrong", "why_is_wrong", "how_to_fix", "severity")):
            score += 8
        # 5 pts: why_is_wrong is substantive (≥100 chars ≈ justification paragraph)
        if len(str(item.get("why_is_wrong", ""))) >= 100:
            score += 5
        # 5 pts: ≥2 supporting research entries
        if len(item.get("supporting_research", [])) >= 2:
            score += 5

    # ── attribute coverage (both formats) ────────────────────────────
    result_attrs = {row.get("attribute") for row in qualitative}
    if set(expected_attrs).issubset(result_attrs):
        score += 10

    return min(score, 40.0)


def _score_severity(result_by_attr: dict[str, dict[str, Any]], baseline_by_attr: dict[str, dict[str, Any]]) -> float:
    if not baseline_by_attr:
        return 0.0
    matches = 0
    for attr, baseline in baseline_by_attr.items():
        got = result_by_attr.get(attr, {})
        if _normalize_severity(got.get("severity", "")) == _normalize_severity(baseline.get("severity", "")):
            matches += 1
    return 20.0 * (matches / len(baseline_by_attr))


def _score_cause_alignment(result_by_attr: dict[str, dict[str, Any]], baseline_by_attr: dict[str, dict[str, Any]]) -> float:
    if not baseline_by_attr:
        return 0.0
    matched = 0
    total = 0
    for attr, baseline in baseline_by_attr.items():
        why = (result_by_attr.get(attr, {}).get("why_is_wrong", "") or "").lower()
        labels = baseline.get("root_cause_labels", [])
        for label in labels:
            total += 1
            plain = label.replace("_", " ")
            if plain in why:
                matched += 1
    if total == 0:
        return 10.0
    return 15.0 * (matched / total)


def _score_mitigation_specificity(result_by_attr: dict[str, dict[str, Any]]) -> float:
    if not result_by_attr:
        return 0.0
    hits = 0
    for row in result_by_attr.values():
        text = (row.get("how_to_fix", "") or "").lower().replace("-", "").replace("_", "")
        if any(keyword.replace("-", "").replace("_", "") in text for keyword in ALGORITHM_KEYWORDS):
            hits += 1
    return 15.0 * (hits / max(len(result_by_attr), 1))


def _score_research_grounding(result: dict[str, Any], result_by_attr: dict[str, dict[str, Any]]) -> float:
    score = 0.0
    if len(result.get("reference_audit_spec", {}).get("supporting_research", [])) >= 2:
        score += 5
    attr_hits = 0
    for row in result_by_attr.values():
        if len(row.get("supporting_research", [])) >= 1:
            attr_hits += 1
    if result_by_attr:
        score += 10.0 * (attr_hits / len(result_by_attr))
    return score


def _safe_num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

