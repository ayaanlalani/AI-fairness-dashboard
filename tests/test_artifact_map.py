"""Every claim in appendix G must point at an artifact that exists, and the
emitted LaTeX must be current. This is the control the paper commits to after
two headline numbers were published with nothing behind them."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import artifact_map as am  # noqa: E402


def test_every_artifact_path_exists():
    missing = [path for _, path, _ in am.MAP if not (ROOT / path).exists()]
    assert not missing, f"artifact map references missing paths: {missing}"


def test_emitted_table_is_current():
    assert am.OUT.read_text() == am.render(), (
        "report/generated/artifact_map.tex is stale; run scripts/artifact_map.py"
    )


def test_map_is_nontrivial():
    assert len(am.MAP) >= 15


def test_appendix_f_is_current():
    import emit_appendix_f as ef

    assert ef.OUT.read_text() == ef.render(), (
        "report/generated/appendix_f_code.tex is stale; run scripts/emit_appendix_f.py"
    )
