"""The reference-convention axis of interval_estimates.py must reproduce the
paper's sensitivity claim: moving from the best adequately-sized cell to the
designated control moves every HMDA ratio by three to four points."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import interval_estimates as ie  # noqa: E402


def _cells(dataset, convention):
    res = ie.analyse(dataset, ie.CASES[dataset], floor=15, convention=convention)
    assert res is not None
    return res, {c["cell"]: c for c in res["cells"]}


def test_hmda_control_convention_reproduces_paper_numbers():
    res, cells = _cells("hmda", "control")
    assert res["reference_cell"] == "White x Male"
    assert cells["American Indian x Male"]["di"] == 0.7012
    assert cells["Black x Female"]["di"] == 0.8988


def test_hmda_default_convention_unchanged():
    res, cells = _cells("hmda", "max_rate_floor")
    assert res["reference_cell"] == "Asian x Female"
    assert cells["American Indian x Male"]["di"] == 0.6734
    assert cells["Black x Female"]["di"] == 0.8632


def test_convention_moves_every_ratio_three_to_four_points():
    _, best = _cells("hmda", "max_rate_floor")
    _, ctrl = _cells("hmda", "control")
    for cell, c in best.items():
        if cell in ("White x Male", "Asian x Female"):
            continue  # a reference in one convention or the other
        shift = ctrl[cell]["di"] - c["di"]
        assert 0.02 <= shift <= 0.05, (cell, shift)


def test_german_credit_present_with_control_cell():
    res, cells = _cells("german_credit", "control")
    assert res["reference_cell"] == "40_plus x male"
    assert set(cells) == {
        "under_40 x female", "under_40 x male", "40_plus x male", "40_plus x female",
    }
