"""Stage B validation: the guardrail gate and the USD cost cap actually bind.

Before Stage B neither was true:
  - `llm_benchmark.py` had no cost flag and no cap logic; `max_cost_usd_per_run`
    in configs/research_guardrails.json had zero code readers.
  - The gate existed only in `llm_benchmark.py`. Three other scripts read an
    LLM key with no gate at all.

No network, no API key, no LLM client. The gate is asserted to fail closed, so
these tests are safe to run with a real key in the environment.
"""
from __future__ import annotations

import json

import pytest

import llm_benchmark_common as common
from llm_benchmark_common import (
    CostCapExceeded,
    SpendLedger,
    cost_for_tokens,
    enforce_llm_gate,
    estimate_call_cost_usd,
    guardrail_max_cost_usd,
    provider_for_model,
)


def _write_guardrails(tmp_path, monkeypatch, **overrides):
    payload = {
        "llm_providers": {"openai": "blocked", "gemini": "blocked"},
        "max_cost_usd_per_run": 15.0,
    }
    payload.update(overrides)
    path = tmp_path / "research_guardrails.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(common, "GUARDRAILS_PATH", path)
    return path


class TestGateFailsClosed:
    def test_blocked_provider_raises(self, tmp_path, monkeypatch):
        _write_guardrails(tmp_path, monkeypatch)
        with pytest.raises(RuntimeError, match="is 'blocked'"):
            enforce_llm_gate("gpt-4o")

    def test_approved_provider_passes(self, tmp_path, monkeypatch):
        _write_guardrails(
            tmp_path, monkeypatch, llm_providers={"openai": "approved", "gemini": "blocked"}
        )
        enforce_llm_gate("gpt-4o")  # must not raise
        with pytest.raises(RuntimeError):
            enforce_llm_gate("gemini-2.5-flash")  # gemini stays blocked

    def test_missing_config_fails_closed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(common, "GUARDRAILS_PATH", tmp_path / "does_not_exist.json")
        with pytest.raises(RuntimeError):
            enforce_llm_gate("gpt-4o")

    def test_unparseable_config_fails_closed(self, tmp_path, monkeypatch):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(common, "GUARDRAILS_PATH", path)
        with pytest.raises(RuntimeError):
            enforce_llm_gate("gpt-4o")

    @pytest.mark.parametrize(
        "model,provider",
        [
            ("gpt-4o", "openai"),
            ("o3", "openai"),
            ("gemini-2.5-flash", "gemini"),
        ],
    )
    def test_provider_routing(self, model, provider):
        assert provider_for_model(model) == provider


class TestEveryKeyReadIsGated:
    """The staging prompt's standing verification claim, asserted in code."""

    def test_all_key_reading_scripts_import_the_gate(self):
        from pathlib import Path

        scripts = Path(common.__file__).parent
        offenders = []
        for path in sorted(scripts.glob("*.py")):
            src = path.read_text(encoding="utf-8")
            reads_key = "OPENAI_API_KEY" in src or "GEMINI_API_KEY" in src
            if reads_key and "enforce_llm_gate" not in src:
                offenders.append(path.name)
        assert offenders == [], f"ungated LLM key reads in: {offenders}"


class TestGuardrailCostCap:
    def test_cap_is_read_from_config(self, tmp_path, monkeypatch):
        _write_guardrails(tmp_path, monkeypatch, max_cost_usd_per_run=15.0)
        assert guardrail_max_cost_usd() == 15.0

    def test_missing_cap_returns_default(self, tmp_path, monkeypatch):
        path = tmp_path / "g.json"
        path.write_text(json.dumps({"llm_providers": {}}), encoding="utf-8")
        monkeypatch.setattr(common, "GUARDRAILS_PATH", path)
        assert guardrail_max_cost_usd(default=None) is None


class TestSpendLedger:
    def test_gpt4o_pricing(self):
        # 1M in + 1M out at $2.50/$10.00
        assert cost_for_tokens("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.50)

    def test_estimate_is_padded_above_naive_cost(self):
        prompt = "x" * 4000  # ~1000 input tokens
        est = estimate_call_cost_usd(prompt, "gpt-4o", expected_output_tokens=1000)
        naive = cost_for_tokens("gpt-4o", 1000, 1000)
        assert est > naive  # conservative, so the cap trips before an overrun

    def test_headroom_check_aborts_before_breach(self, tmp_path):
        ledger = SpendLedger(max_cost_usd=1.0, path=tmp_path / "ledger.json")
        ledger.record({"total_cost_usd": 0.95, "model": "gpt-4o"})
        assert ledger.spent_usd == pytest.approx(0.95)
        with pytest.raises(CostCapExceeded, match="Cost cap reached"):
            ledger.assert_headroom(0.10, label="cycle 4")

    def test_headroom_allows_calls_under_cap(self, tmp_path):
        ledger = SpendLedger(max_cost_usd=1.0, path=tmp_path / "ledger.json")
        ledger.record({"total_cost_usd": 0.5, "model": "gpt-4o"})
        ledger.assert_headroom(0.4)  # must not raise

    def test_no_cap_means_no_enforcement(self, tmp_path):
        ledger = SpendLedger(max_cost_usd=None, path=tmp_path / "ledger.json")
        ledger.assert_headroom(10_000.0)  # must not raise
        assert ledger.remaining_usd is None

    def test_spend_persists_across_instances(self, tmp_path):
        """The cap must hold across the three Stage 3 invocations, not per process."""
        path = tmp_path / "ledger.json"
        first = SpendLedger(max_cost_usd=1.0, path=path, run_label="german_credit")
        first.record({"total_cost_usd": 0.6, "model": "gpt-4o"}, dataset="german_credit")

        second = SpendLedger(max_cost_usd=1.0, path=path, run_label="hmda")
        assert second.spent_usd == pytest.approx(0.6)
        assert second.remaining_usd == pytest.approx(0.4)
        with pytest.raises(CostCapExceeded):
            second.assert_headroom(0.5)

    def test_zero_cost_calls_are_not_recorded(self, tmp_path):
        """Dry runs and failed calls bill nothing and must not pollute the ledger."""
        ledger = SpendLedger(max_cost_usd=1.0, path=tmp_path / "ledger.json")
        ledger.record({"total_cost_usd": 0.0, "model": "dry_run"})
        ledger.record({})
        assert ledger.spent_usd == 0.0
        assert ledger.entries == []

    def test_unreadable_ledger_starts_fresh(self, tmp_path):
        path = tmp_path / "ledger.json"
        path.write_text("{corrupt", encoding="utf-8")
        ledger = SpendLedger(max_cost_usd=1.0, path=path)
        assert ledger.spent_usd == 0.0
