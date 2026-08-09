"""The prompt-sensitivity reanalysis must reproduce the published figures and
stay in sync with its committed artifacts."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "scripts"))

import recheck_prompt_sensitivity as rps  # noqa: E402


def test_gpt4o_spreads_match_published_figures():
    spreads = rps.analyse_spread_model(rps.load_records(rps.MODEL_DIRS["gpt-4o"]))
    assert spreads["german_credit"]["spread_published"] == 14.6667
    assert spreads["hmda"]["spread_published"] == 9.0
    # The spread collapses once the constrained cycle is excluded.
    assert spreads["german_credit"]["spread_excluding_constrained"] == 6.6667
    assert spreads["hmda"]["spread_excluding_constrained"] == 1.25


def test_gpt5_mini_spreads_match_published_figures():
    spreads = rps.analyse_spread_model(rps.load_records(rps.MODEL_DIRS["gpt-5-mini"]))
    assert spreads["german_credit"]["spread_published"] == 0.0
    assert spreads["hmda"]["spread_published"] == 1.875


def test_cycle4_dip_is_completeness_and_severity_not_cause_alignment():
    spreads = rps.analyse_spread_model(rps.load_records(rps.MODEL_DIRS["gpt-4o"]))
    for dataset in ("german_credit", "hmda"):
        shares = spreads[dataset]["cycle4_decomposition"]["subscore_shares_of_dip"]
        assert shares["cause_alignment"] == 0.0
        assert shares["completeness"] + shares["severity_agreement"] > 0.99


def test_gemini_three_means_are_distinct_and_reproduce_published_value():
    gemini = rps.analyse_gemini(rps.load_records(rps.MODEL_DIRS["gemini-2.5-flash"]))
    assert gemini["mean_all"] == 19.7917
    assert gemini["mean_excluding_score_zero"] == 47.5  # previously published
    assert gemini["mean_content_bearing_only"] == 72.5
    assert gemini["empty_result_records"] == 18
    assert gemini["score_zero_records"] == 14
    # Four empty API results still earned points via the flat cause_alignment
    # default — the reason the published 47.50 is not a content-only mean.
    assert len(gemini["empty_but_scored_records"]) == 4


def test_committed_artifacts_are_current():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "recheck_prompt_sensitivity.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_json_artifact_verdict_present():
    payload = json.loads(
        (ROOT / "artifacts" / "consolidated" / "prompt_sensitivity_check.json").read_text()
    )
    assert "constrained cycle" in payload["verdict"]
