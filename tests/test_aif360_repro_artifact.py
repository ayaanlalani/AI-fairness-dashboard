"""The AIF360 repro's committed output must carry the numbers the paper
quotes, with pinned versions. The live run needs aif360 installed
(requirements-repro.txt) and is exercised only when available."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "artifacts" / "consolidated" / "aif360_default_repro.txt"


def test_artifact_quotes_both_orientations_and_versions():
    text = ARTIFACT.read_text()
    assert "aif360==0.6.1" in text
    assert "fairlearn==0.13.0" in text
    assert "DI = 1.2072" in text
    assert "DI = 0.9862" in text
    assert "0.8284" in text  # fairlearn on the column as encoded


def test_repro_pins_are_declared():
    pins = (ROOT / "requirements-repro.txt").read_text()
    assert "aif360==0.6.1" in pins
    assert "fairlearn==0.13.0" in pins


@pytest.mark.skipif(
    importlib.util.find_spec("aif360") is None,
    reason="aif360 not installed (pip install -r requirements-repro.txt)",
)
def test_live_repro_matches_artifact():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "aif360_default_repro.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ARTIFACT.read_text()
