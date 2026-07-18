"""Regression tests for lending_club fairness-metric orientation.

Guards the fix for stage1_findings.md Q3.1: metrics must be oriented on the
favorable outcome (predicted non-default, label 0) using AIF360 conventions
(DI = unprivileged/privileged, differences = unprivileged - privileged).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "lending_club_dataset" / "scripts" / "compute_fairness.py"

spec = importlib.util.spec_from_file_location("lc_compute_fairness", MODULE_PATH)
lc_compute_fairness = importlib.util.module_from_spec(spec)
sys.modules["lc_compute_fairness"] = lc_compute_fairness
spec.loader.exec_module(lc_compute_fairness)

_manual_group_rates = lc_compute_fairness._manual_group_rates


def q31_frame() -> pd.DataFrame:
    """Reproduce the Q3.1 gender scenario in miniature.

    Privileged (male): predicted non-default for everyone.
    Unprivileged (female): ~99.31% predicted non-default; the sole default
    prediction lands on a true defaulter (so favorable-TPR is equal at 1.0).
    """
    males = pd.DataFrame({
        "group": ["male"] * 100,
        "y_true": [1] * 12 + [0] * 88,
        "y_pred": [0] * 100,
    })
    # 144 females, exactly one default prediction on a true defaulter
    females = pd.DataFrame({
        "group": ["female"] * 144,
        "y_true": [1] * 18 + [0] * 126,
        "y_pred": [1] + [0] * 143,
    })
    return pd.concat([males, females], ignore_index=True)


class TestFavorableOrientation:
    def test_gender_di_is_near_parity_not_zero(self):
        df = q31_frame()
        di, dpd, eod, aod = _manual_group_rates(
            df["y_true"], df["y_pred"], df["group"], "male", favorable_label=0
        )
        assert di == pytest.approx(143 / 144, abs=1e-9)  # 0.99306, not 0.0
        assert dpd == pytest.approx(143 / 144 - 1.0, abs=1e-9)  # unpriv - priv
        assert eod == pytest.approx(0.0, abs=1e-9)

    def test_default_oriented_computation_would_have_read_zero(self):
        # The pre-fix behavior: orienting on label 1 with priv/unpriv gives
        # 0/rate = 0.0. Ensure favorable_label=1 still computes coherently
        # (it is a supported argument), but is NOT the lending_club default.
        df = q31_frame()
        di, _, _, _ = _manual_group_rates(
            df["y_true"], df["y_pred"], df["group"], "male", favorable_label=1
        )
        # AIF360 orientation on the default label: unpriv/priv with priv=0 -> nan
        assert di != di  # nan: no privileged selection to ratio against

    def test_argparse_default_favorable_label_is_zero(self):
        text = MODULE_PATH.read_text(encoding="utf-8")
        assert '"--favorable_label", type=int, default=0' in text

    def test_perfect_parity(self):
        df = pd.DataFrame({
            "group": ["a"] * 10 + ["b"] * 10,
            "y_true": [0, 1] * 10,
            "y_pred": [0, 1] * 10,
        })
        di, dpd, eod, aod = _manual_group_rates(
            df["y_true"], df["y_pred"], df["group"], "a", favorable_label=0
        )
        assert di == pytest.approx(1.0)
        assert dpd == pytest.approx(0.0)
        assert eod == pytest.approx(0.0)
        assert aod == pytest.approx(0.0)
