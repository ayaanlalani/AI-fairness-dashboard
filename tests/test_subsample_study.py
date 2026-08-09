"""Structural invariants of the subsampling study. A small B keeps this fast;
the assertions are about ground truth, estimator behaviour, and event logic
rather than the published rates themselves."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import subsample_study as ss  # noqa: E402

_STUDY = ss.run_study(sizes=(500, 2196), B=40, seed=7)


def test_population_ground_truth_matches_published_table():
    theta = _STUDY["theta"]
    assert theta["max_rate_floor15"]["reference_cell"] == "Asian x Female"
    assert theta["control"]["reference_cell"] == "White x Male"
    for convention in ss.CONVENTIONS:
        assert theta[convention]["worst_cell"] == "American Indian x Male"
    di = theta["max_rate_floor15"]["di"]
    assert di["American Indian x Male"] == 0.6734
    assert di["Black x Female"] == 0.8632
    assert set(theta["max_rate_floor15"]["violating_cells"]) == {
        "American Indian x Male",
        "Multiracial x Female",
        "Multiracial x Male",
        "Pacific Islander x Female",
    }


def test_wilson_bounds_match_scalar_reference():
    from interval_estimates import wilson

    k = np.array([18.0, 745.0, 0.0])
    n = np.array([31.0, 1000.0, 0.0])
    lo, hi = ss.wilson_bounds(k, n)
    for i in range(2):
        ref_lo, ref_hi = wilson(int(k[i]), int(n[i]))
        assert abs(lo[i] - ref_lo) < 1e-12
        assert abs(hi[i] - ref_hi) < 1e-12
    assert np.isnan(lo[2]) and np.isnan(hi[2])


def test_eb_shrinks_toward_pooled_rate():
    k = np.array([[2.0, 700.0, 80.0]])
    n = np.array([[4.0, 900.0, 100.0]])
    pooled = k.sum() / n.sum()
    shrunk = ss.eb_shrink(k, n)[0]
    raw = k[0] / n[0]
    for c in range(3):
        lo, hi = sorted((raw[c], pooled))
        assert lo - 1e-9 <= shrunk[c] <= hi + 1e-9


def test_floor_policies_trade_clearance_for_silence_at_audit_size():
    res = _STUDY["results"]["max_rate_floor15"]
    at = "2196"
    # Floors never assert a violating cell clear at this size; the point
    # policy frequently does.
    assert res["floor15"][at]["false_clearance"]["rate"] == 0.0
    assert res["point"][at]["false_clearance"]["rate"] > 0.5
    # The silence shows up as suppressed cells and a wrong worst cell.
    assert res["floor15"][at]["mean_suppressed_cells"] > 4
    assert res["floor15"][at]["worst_cell_wrong"]["rate"] == 1.0


def test_wilson_policy_abstains_rather_than_asserts():
    res = _STUDY["results"]["max_rate_floor15"]["wilson"]["2196"]
    assert res["false_clearance"]["rate"] == 0.0
    assert res["worst_cell_abstain"]["rate"] > 0.5
    # Deliberately conservative interval must at least reach nominal coverage.
    assert res["coverage"] >= 0.95


def test_same_seed_reproduces():
    again = ss.run_study(sizes=(500,), B=10, seed=123)
    once_more = ss.run_study(sizes=(500,), B=10, seed=123)
    assert again["results"] == once_more["results"]
