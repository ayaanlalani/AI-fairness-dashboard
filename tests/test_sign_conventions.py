"""Cross-dataset sign-convention pinning (stage1_findings.md cross-cutting #4).

Every dataset-local fairness script must report DI/DPD/EOD/AOD in the AIF360
orientation: DI = unprivileged/privileged, differences = unprivileged −
privileged. A synthetic frame where the unprivileged group receives strictly
fewer favorable predictions (lower selection rate, TPR, and FPR) must
therefore yield DI < 1 and negative DPD/EOD/AOD everywhere.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(module_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def unpriv_disadvantaged_frame(favorable: int, unfavorable: int) -> pd.DataFrame:
    """Unprivileged group strictly worse off on the favorable class."""
    rng = np.random.default_rng(42)
    n = 400
    group = np.array(["priv"] * (n // 2) + ["unpriv"] * (n // 2))
    y_true = np.where(rng.integers(0, 2, size=n) == 1, favorable, unfavorable)
    y_pred = y_true.copy()
    # Every 3rd unprivileged true-favorable is denied (lowers unpriv TPR/selection).
    unpriv_fav = np.where((group == "unpriv") & (y_true == favorable))[0]
    y_pred[unpriv_fav[::3]] = unfavorable
    # Every 4th privileged true-unfavorable is approved anyway (raises priv FPR).
    priv_unfav = np.where((group == "priv") & (y_true == unfavorable))[0]
    y_pred[priv_unfav[::4]] = favorable
    return pd.DataFrame({"group": group, "y_true": y_true, "y_pred": y_pred})


def fpr_gap(df: pd.DataFrame, favorable: int) -> float:
    def fpr(mask):
        denom = ((df["y_true"] != favorable) & mask).sum()
        return (
            ((df["y_true"] != favorable) & (df["y_pred"] == favorable) & mask).sum()
            / denom
        )

    return float(fpr(df["group"] == "unpriv") - fpr(df["group"] == "priv"))


def assert_aif360_orientation(metrics: dict, df: pd.DataFrame, favorable: int):
    assert metrics["DisparateImpact"] < 1.0
    assert metrics["DemographicParityDiff"] < 0
    assert metrics["EqualOpportunityDiff"] < 0
    assert metrics["AverageOddsDiff"] < 0
    # AOD must average the TPR and FPR gaps in the same direction as EOD.
    assert metrics["AverageOddsDiff"] == pytest.approx(
        (metrics["EqualOpportunityDiff"] + fpr_gap(df, favorable)) / 2.0
    )


class TestGermanCredit:
    def test_manual_metrics_orientation(self):
        mod = _load("gc_fairness", "german_credit_dataset/scripts/compute_fairness.py")
        df = unpriv_disadvantaged_frame(mod.FAVORABLE_LABEL, mod.UNFAVORABLE_LABEL)
        m = mod._manual_metrics(df["y_true"], df["y_pred"], df["group"], "priv")
        assert_aif360_orientation(m, df, mod.FAVORABLE_LABEL)


class TestHMDA:
    def test_manual_metrics_orientation(self):
        mod = _load("hmda_fairness", "hmda_dataset/scripts/compute_fairness.py")
        df = unpriv_disadvantaged_frame(mod.FAVORABLE_LABEL, mod.UNFAVORABLE_LABEL)
        m = mod._manual_metrics(df["y_true"], df["y_pred"], df["group"], "priv")
        assert_aif360_orientation(m, df, mod.FAVORABLE_LABEL)


class TestLendingClub:
    def test_manual_group_rates_orientation(self):
        mod = _load("lc_fairness_sign", "lending_club_dataset/scripts/compute_fairness.py")
        # Lending Club: favorable outcome is predicted non-default (label 0).
        df = unpriv_disadvantaged_frame(favorable=0, unfavorable=1)
        di, dpd, eod, aod = mod._manual_group_rates(
            df["y_true"], df["y_pred"], df["group"], "priv", favorable_label=0
        )
        m = {
            "DisparateImpact": di,
            "DemographicParityDiff": dpd,
            "EqualOpportunityDiff": eod,
            "AverageOddsDiff": aod,
        }
        assert_aif360_orientation(m, df, favorable=0)
