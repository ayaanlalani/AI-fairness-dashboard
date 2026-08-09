"""The attrition replay must reconcile exactly with the pipeline's population
and disclose the dropped race categories."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import hmda_attrition as ha  # noqa: E402

_DF = pd.read_csv(ha.RAW, low_memory=False)
_STEPS, _CLEANED = ha.replay_attrition(_DF)


def test_attrition_reconciles_to_pipeline_population():
    assert len(_DF) == 20000
    assert len(_CLEANED) == 10978
    assert _STEPS[-1]["remaining"] == 10978
    # Each step's remaining = previous remaining - dropped.
    for prev, step in zip(_STEPS, _STEPS[1:]):
        assert step["remaining"] == prev["remaining"] - step["dropped"]


def test_race_not_available_is_the_largest_race_exclusion():
    race_step = next(s for s in _STEPS if s["step"] == "race")
    by_cat = race_step["dropped_by_category"]
    assert by_cat["Race Not Available"] == 4140
    assert set(by_cat) == {"Race Not Available", "Joint", "Free Form Text Only"}


def test_dropped_categories_appear_in_di_table():
    table = ha.race_di_with_dropped_categories(_DF)
    by_race = {r["race"]: r for r in table["rows"]}
    rna = by_race["Race Not Available"]
    assert rna["retained_by_pipeline"] is False
    assert rna["n"] == 4140
    assert 0 < rna["di_vs_white"] < 1
    # Sanity: retained groups are flagged as retained.
    assert by_race["White"]["retained_by_pipeline"] is True
    assert by_race["Black"]["retained_by_pipeline"] is True
