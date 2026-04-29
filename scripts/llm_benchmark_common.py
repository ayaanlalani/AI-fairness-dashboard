"""
llm_benchmark_common.py

Shared helpers for qualitative-only LLM fairness benchmarking.
"""
from __future__ import annotations

from collections import Counter
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

