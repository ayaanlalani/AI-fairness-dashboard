"""Stage 2 validation: scoring harness + refusal/hallucination detectors.

Exercises the exact code paths the dry-run benchmark uses, on mock outputs —
including one deliberately hallucinated metric value (staging prompt §3,
Stage 2 task 2). No network, no LLM client, no API key.
"""
from __future__ import annotations

import pytest

from llm_benchmark import (
    detect_hallucinations,
    detect_refusal,
    known_metric_values,
)
from llm_benchmark_common import score_llm_output

KNOWN_TITLES = [
    "Fairness Definitions Explained",
    "A Survey on Bias and Fairness in Machine Learning",
]

CONTEXT_PAYLOAD = {
    "dataset": "German Credit",
    "attributes": [
        {
            "attribute": "Sex_original",
            "metrics": {
                "disparate_impact": 0.8595,
                "demographic_parity_difference": -0.1152,
                "equal_opportunity_difference": -0.0512,
                "average_odds_difference": -0.1256,
            },
        }
    ],
}

BASELINE_PAYLOAD = {
    "attributes": [
        {
            "attribute": "Sex_original",
            "severity": "MODERATE",
            "root_cause_labels": ["representation_bias"],
        }
    ],
}


def good_mock() -> dict:
    return {
        "attribute": "Sex_original",
        "severity": "MODERATE",
        "what_is_wrong": (
            "The pre-computed context reports disparate impact = 0.8595 for "
            "Sex_original, just above the 0.8 four-fifths threshold."
        ),
        "why_is_wrong": (
            "The deterministic root-cause analysis attributes the gap to "
            "representation bias in the training data for this attribute."
        ),
        "how_to_fix": (
            "Apply Reweighing preprocessing and validate with a "
            "ThresholdOptimizer pass, targeting disparate impact >= 0.8."
        ),
        "supporting_research": [KNOWN_TITLES[0]],
    }


class TestScorerRoundTrip:
    def test_good_mock_scores_high(self):
        score = score_llm_output({"qualitative": [good_mock()]}, BASELINE_PAYLOAD)
        assert score["total_score"] >= 60
        assert score["subscores"]["severity_agreement"] == pytest.approx(20.0)

    def test_empty_output_scores_zero_completeness(self):
        score = score_llm_output({"qualitative": []}, BASELINE_PAYLOAD)
        assert score["subscores"]["completeness"] == 0


class TestRefusalDetector:
    def test_good_mock_is_not_refusal(self):
        assert detect_refusal(good_mock(), "") is False

    def test_none_result_is_refusal(self):
        assert detect_refusal(None, "") is True

    def test_refusal_phrase_flagged(self):
        assert detect_refusal(good_mock(), "I'm sorry, but I cannot help") is True

    def test_missing_required_field_flagged(self):
        broken = good_mock()
        broken["how_to_fix"] = ""
        assert detect_refusal(broken, "") is True

    def test_invalid_severity_flagged(self):
        broken = good_mock()
        broken["severity"] = "APOCALYPTIC"
        assert detect_refusal(broken, "") is True


class TestHallucinationDetector:
    def test_clean_output_has_no_flags(self):
        flags = detect_hallucinations(
            good_mock(), KNOWN_TITLES, known_metric_values(CONTEXT_PAYLOAD)
        )
        assert flags == []

    def test_fabricated_citation_flagged(self):
        mock = good_mock()
        mock["supporting_research"] = ["Entirely Invented Paper About Loans 2031"]
        flags = detect_hallucinations(mock, KNOWN_TITLES)
        assert flags == ["Entirely Invented Paper About Loans 2031"]

    def test_deliberately_hallucinated_metric_value_flagged(self):
        # Guardrail 5: an asserted metric value absent from the provided
        # context is a hallucination flag, not a result.
        mock = good_mock()
        mock["what_is_wrong"] = (
            "The context reports disparate impact = 0.4321 for Sex_original, "
            "a severe violation."
        )
        flags = detect_hallucinations(
            mock, KNOWN_TITLES, known_metric_values(CONTEXT_PAYLOAD)
        )
        assert any("0.4321" in f for f in flags)

    def test_context_values_and_canonical_thresholds_not_flagged(self):
        mock = good_mock()
        mock["why_is_wrong"] = (
            "Demographic parity difference is -0.1152 while the canonical "
            "0.8 threshold and a 0.05 tolerance are the usual anchors here."
        )
        flags = detect_hallucinations(
            mock, KNOWN_TITLES, known_metric_values(CONTEXT_PAYLOAD)
        )
        assert flags == []

    def test_no_context_values_skips_metric_check(self):
        mock = good_mock()
        mock["what_is_wrong"] = "Disparate impact = 0.4321 asserted without context."
        assert detect_hallucinations(mock, KNOWN_TITLES) == []
