"""
Bibliography integrity, offline.

The point of these tests is that "every citation is real" stops being a claim in
a commit message and becomes a build gate. Network resolution lives in
scripts/verify_bibliography.py and is cached to
artifacts/consolidated/bib_verification.json; these tests read the cache and the
.bib, so they run in CI with no key and no network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_bibliography import (  # noqa: E402
    check_fields,
    check_structure,
    normalise_title,
    parse_bib,
    parse_citations,
    title_similarity,
)

BIB = REPO_ROOT / "report" / "refs.bib"
TEX_FILES = [
    REPO_ROOT / "report" / "report_facct.tex",
    REPO_ROOT / "report" / "report.tex",
]
CACHE = REPO_ROOT / "artifacts" / "consolidated" / "bib_verification.json"


@pytest.fixture(scope="module")
def entries() -> dict:
    return parse_bib(BIB)


@pytest.fixture(scope="module")
def cited() -> set[str]:
    keys: set[str] = set()
    for tex in TEX_FILES:
        keys |= parse_citations(tex)
    return keys


def test_bib_parses_with_expected_scale(entries):
    assert len(entries) >= 40, f"expected 40+ entries, parsed {len(entries)}"


def test_no_undefined_or_uncited_across_both_papers(entries, cited):
    """report.tex and report_facct.tex share refs.bib, so coverage is a union."""
    problems = check_structure(entries, cited)
    assert problems == [], "\n".join(problems)


def test_required_fields_and_no_placeholders(entries):
    problems = check_fields(entries)
    assert problems == [], "\n".join(problems)


def test_every_entry_is_verified_or_attested(entries):
    """Each entry resolved against a registrar, or carries explicit provenance."""
    assert CACHE.exists(), (
        f"{CACHE.relative_to(REPO_ROOT)} missing — run "
        "scripts/verify_bibliography.py --tex report/report_facct.tex report/report.tex"
    )
    cache = json.loads(CACHE.read_text())["entries"]
    unverified = [
        key
        for key in entries
        if not cache.get(key, {}).get("verified")
    ]
    assert unverified == [], f"not verified: {unverified}"


def test_attested_entries_state_their_provenance(entries):
    """An attestation without a reason is indistinguishable from a stub."""
    for key, entry in entries.items():
        note = entry["fields"].get("verifynote")
        if note is not None:
            assert len(note) > 25, f"{key}: verifynote too vague to check: {note!r}"


def test_dropped_citations_stay_dropped(entries):
    """Guards the three that failed verification during construction.

    Crenshaw 1989 is the important one: the only CrossRef hit for that title is
    a 2024 German chapter *about* Crenshaw by a different author, which would
    read as a genuine citation in a reference list.
    """
    for key in ("crenshaw1989", "hardt2016", "kearns2018"):
        assert key not in entries, (
            f"{key} was added back without a resolvable record; see the header "
            "of report/refs.bib before re-adding"
        )


class TestTitleMatching:
    """normalise_title / title_similarity previously scored a correct title 0.00."""

    def test_bibtex_capital_protection_does_not_split_tokens(self):
        # "{S}tatlog" must not become the tokens "s" and "tatlog".
        assert normalise_title("{S}tatlog ({G}erman {C}redit {D}ata)") == (
            "statlog german credit data"
        )

    def test_protected_title_matches_its_plain_form(self):
        sim = title_similarity(
            "{S}tatlog ({G}erman {C}redit {D}ata)", "Statlog (German Credit Data)"
        )
        assert sim == pytest.approx(1.0)

    def test_truncated_subtitle_still_matches(self):
        # CrossRef drops subtitles; a prefix match must not be a mismatch.
        sim = title_similarity(
            "Improving Fairness in Machine Learning Systems: What Do Industry "
            "Practitioners Need?",
            "Improving Fairness in Machine Learning Systems",
        )
        assert sim == pytest.approx(1.0)

    def test_unrelated_titles_do_not_match(self):
        sim = title_similarity(
            "Demarginalizing the Intersection of Race and Sex",
            "Model Cards for Model Reporting",
        )
        assert sim < 0.3


class TestCitationParsing:
    def test_citestyle_is_not_a_citation(self, tmp_path):
        """\\citestyle{acmnumeric} is a format declaration, not a \\cite."""
        tex = tmp_path / "t.tex"
        tex.write_text("\\citestyle{acmnumeric}\n\\cite{real2020}\n")
        assert parse_citations(tex) == {"real2020"}

    def test_multi_key_and_optional_args(self, tmp_path):
        tex = tmp_path / "t.tex"
        tex.write_text("\\citep[see][]{a2020,b2021}\\citet{c2022}\n")
        assert parse_citations(tex) == {"a2020", "b2021", "c2022"}

    def test_commented_citations_are_ignored(self, tmp_path):
        tex = tmp_path / "t.tex"
        tex.write_text("% \\cite{ghost2020}\n\\cite{real2020}\n")
        assert parse_citations(tex) == {"real2020"}

    def test_escaped_percent_does_not_start_a_comment(self, tmp_path):
        tex = tmp_path / "t.tex"
        tex.write_text("A 75\\% rate \\cite{real2020}\n")
        assert parse_citations(tex) == {"real2020"}
