#!/usr/bin/env python3
"""
verify_bibliography.py

Mechanical gate over a LaTeX bibliography. Three independent checks:

  1. STRUCTURE  — every \\cite key used in the .tex exists in the .bib, and
                  every .bib entry is actually cited. Offline, always runs.
  2. FIELDS     — required fields present per entry type; no placeholder text.
                  Offline, always runs.
  3. RESOLUTION — every DOI resolves to a real record whose title matches the
                  .bib title. Network; results cached so later runs and CI are
                  offline.

Written because "every bibkey is backed by a fetched record" was previously an
assertion in a commit message with no way to re-check it. A fabricated citation
should fail a build, not read as well-supported prose.

Two DOI registrars are consulted, because neither alone covers this
bibliography: CrossRef (ACM/Elsevier/IEEE) and DataCite via doi.org content
negotiation (LIPIcs, arXiv). Semantic Scholar is used only as a last resort for
entries with no DOI at all, since its year field reports the preprint rather
than the publication.

Usage:
    python3.11 scripts/verify_bibliography.py --tex report/report_facct.tex
    python3.11 scripts/verify_bibliography.py --tex report/report.tex --offline

Exit status is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIB = REPO_ROOT / "report" / "refs.bib"
CACHE_PATH = REPO_ROOT / "artifacts" / "consolidated" / "bib_verification.json"

CROSSREF = "https://api.crossref.org/works"
DOI_ORG = "https://doi.org"
S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"

REQUEST_INTERVAL_S = 1.0
_LAST_REQUEST_TS = 0.0

# Minimum fields we insist on, by entry type. Anything not listed falls back to
# author/title/year, which is the floor for a citation being checkable at all.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "article": ("author", "title", "journal", "year"),
    "inproceedings": ("author", "title", "booktitle", "year"),
    "techreport": ("author", "title", "institution", "year"),
    "misc": ("author", "title", "year"),
}
_FALLBACK_REQUIRED = ("author", "title", "year")

# Text that means someone left a stub behind.
PLACEHOLDER_PATTERNS = (
    r"\bTODO\b",
    r"\bTBD\b",
    r"\bFIXME\b",
    r"\bXXX\b",
    r"\?\?\?",
)

# Title-similarity floor for calling a DOI resolution a match. Publishers
# routinely truncate subtitles (CrossRef drops "What Do Industry Practitioners
# Need?"), so this compares the leading portion rather than demanding equality.
TITLE_MATCH_THRESHOLD = 0.72


# --------------------------------------------------------------------------- #
#  Parsing
# --------------------------------------------------------------------------- #

def strip_tex_comments(text: str) -> str:
    """Drop % comments so a commented-out \\cite does not count as a citation."""
    out = []
    for line in text.splitlines():
        idx = 0
        while True:
            idx = line.find("%", idx)
            if idx == -1:
                out.append(line)
                break
            if idx > 0 and line[idx - 1] == "\\":
                idx += 1
                continue
            out.append(line[:idx])
            break
    return "\n".join(out)


def parse_bib(path: Path) -> dict[str, dict[str, Any]]:
    """Minimal BibTeX reader: entry type, key, and brace-balanced fields."""
    text = path.read_text(encoding="utf-8")
    entries: dict[str, dict[str, Any]] = {}

    for match in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        etype, key = match.group(1).lower(), match.group(2)
        start = match.end()
        depth, idx = 1, match.start() + match.group(0).index("{")
        # Walk from the opening brace to its partner to find the entry body.
        idx += 1
        depth = 1
        while idx < len(text) and depth:
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
            idx += 1
        body = text[start : idx - 1]

        fields: dict[str, str] = {}
        for fmatch in re.finditer(r"(\w+)\s*=\s*", body):
            fname = fmatch.group(1).lower()
            rest = body[fmatch.end() :].lstrip()
            if not rest:
                continue
            if rest[0] == "{":
                d, j = 1, 1
                while j < len(rest) and d:
                    if rest[j] == "{":
                        d += 1
                    elif rest[j] == "}":
                        d -= 1
                    j += 1
                value = rest[1 : j - 1]
            else:
                value = rest.split(",")[0]
            fields[fname] = " ".join(value.split())

        if key in entries:
            entries[key]["_duplicate"] = True
        else:
            entries[key] = {"type": etype, "fields": fields}

    return entries


# \citestyle{acmnumeric} is a formatting declaration, not a citation. Any other
# cite-prefixed command that takes a style rather than keys belongs here too.
NON_CITATION_COMMANDS = {"citestyle"}


def parse_citations(tex_path: Path) -> set[str]:
    """Collect keys from every \\cite variant, including multi-key forms."""
    text = strip_tex_comments(tex_path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    pattern = r"\\((?:no)?cite[a-zA-Z]*)\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}"
    for match in re.finditer(pattern, text):
        if match.group(1).lower() in NON_CITATION_COMMANDS:
            continue
        for raw in match.group(2).split(","):
            key = raw.strip()
            if key:
                keys.add(key)
    return keys


# --------------------------------------------------------------------------- #
#  Resolution
# --------------------------------------------------------------------------- #

def _throttle() -> None:
    global _LAST_REQUEST_TS
    wait = REQUEST_INTERVAL_S - (time.monotonic() - _LAST_REQUEST_TS)
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST_TS = time.monotonic()


def _fetch_json(url: str, accept: str | None = None) -> dict[str, Any] | None:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "AI-Fairness-Dashboard/1.0 (bibliography verifier)")
    if accept:
        req.add_header("Accept", accept)
    for attempt in range(3):
        try:
            _throttle()
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return None
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return None


def normalise_title(title: str) -> str:
    """Lowercase, strip TeX braces/escapes and punctuation for comparison.

    Braces are deleted rather than replaced with a space: BibTeX capital
    protection writes {S}tatlog, and spacing the braces would split that into
    the tokens "s" and "tatlog", making a correct title score 0.00 against its
    own resolved record.
    """
    t = title.lower()
    t = re.sub(r"\\[a-zA-Z]+\s*", " ", t)  # TeX macros -> space
    t = t.replace("{", "").replace("}", "")  # capital protection -> delete
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return " ".join(t.split())


def title_similarity(a: str, b: str) -> float:
    """Prefix-tolerant token overlap: |shared| / |tokens of the shorter title|."""
    ta, tb = normalise_title(a).split(), normalise_title(b).split()
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    return len(sa & sb) / min(len(sa), len(sb))


def resolve_doi(doi: str) -> dict[str, Any] | None:
    """CrossRef first, then DataCite/other registrars via doi.org negotiation."""
    rec = _fetch_json(f"{CROSSREF}/{urllib.parse.quote(doi, safe='./')}")
    if rec and rec.get("status") == "ok":
        msg = rec["message"]
        return {
            "registrar": "crossref",
            "title": (msg.get("title") or [""])[0],
            "doi": msg.get("DOI"),
        }

    rec = _fetch_json(
        f"{DOI_ORG}/{urllib.parse.quote(doi, safe='./')}",
        accept="application/vnd.citationstyles.csl+json",
    )
    if rec and rec.get("title"):
        title = rec["title"]
        return {
            "registrar": rec.get("publisher") or "doi.org",
            "title": title if isinstance(title, str) else title[0],
            "doi": rec.get("DOI") or doi,
        }
    return None


def _detex(title: str) -> str:
    """Readable title with TeX artifacts removed but punctuation intact."""
    t = re.sub(r"\\[a-zA-Z]+\s*", "", title)
    return " ".join(t.replace("{", "").replace("}", "").split())


def resolve_by_title(title: str) -> dict[str, Any] | None:
    """Last resort for DOI-less entries (PMLR, arXiv-only preprints).

    Two query forms are tried: the de-TeX'd title with punctuation, then the
    fully normalised token string. Semantic Scholar's ranking is sensitive to
    both subtitle colons and stray punctuation, and neither form wins reliably.
    """
    for query in (_detex(title), normalise_title(title)):
        url = (
            f"{S2_SEARCH}?query={urllib.parse.quote(query)}"
            "&limit=5&fields=title,year,externalIds"
        )
        rec = _fetch_json(url)
        if not rec:
            continue
        for cand in rec.get("data", []):
            if title_similarity(title, cand.get("title") or "") >= TITLE_MATCH_THRESHOLD:
                return {
                    "registrar": "semantic_scholar",
                    "title": cand.get("title"),
                    "doi": (cand.get("externalIds") or {}).get("DOI"),
                }
    return None


# --------------------------------------------------------------------------- #
#  Checks
# --------------------------------------------------------------------------- #

def check_structure(entries: dict, cited: set[str]) -> list[str]:
    problems = []
    for key in sorted(cited - entries.keys()):
        problems.append(f"undefined citation: \\cite{{{key}}} has no entry in the .bib")
    for key in sorted(entries.keys() - cited):
        problems.append(f"uncited entry: {key} is defined but never cited")
    for key, entry in sorted(entries.items()):
        if entry.get("_duplicate"):
            problems.append(f"duplicate key: {key} is defined more than once")
    return problems


def check_fields(entries: dict) -> list[str]:
    problems = []
    for key, entry in sorted(entries.items()):
        required = REQUIRED_FIELDS.get(entry["type"], _FALLBACK_REQUIRED)
        for field in required:
            if not entry["fields"].get(field):
                problems.append(f"{key}: missing required field '{field}' for @{entry['type']}")
        blob = " ".join(entry["fields"].values())
        for pattern in PLACEHOLDER_PATTERNS:
            if re.search(pattern, blob, re.IGNORECASE):
                problems.append(f"{key}: placeholder text matching /{pattern}/")
    return problems


def check_resolution(
    entries: dict, cache: dict, offline: bool
) -> tuple[list[str], dict, list[str]]:
    """Resolve every entry. Returns (problems, updated_cache, attested).

    Entries carrying a `verifynote` field are *attested* rather than resolved:
    tool and dataset citations (AIF360's Fairlearn tech report, the HMDA data
    release) are not papers and have no registrar record, and one harvested
    paper has no DOI but is recorded verbatim in the pipeline's evidence JSON.
    The note must say where the provenance lives, and attested entries are
    reported separately rather than counted as verified.
    """
    problems: list[str] = []
    attested: list[str] = []
    updated = dict(cache)

    for key, entry in sorted(entries.items()):
        title = entry["fields"].get("title", "")
        doi = entry["fields"].get("doi", "")
        note = entry["fields"].get("verifynote", "")

        if note:
            attested.append(f"{key}: {note}")
            updated[key] = {"verified": True, "doi": doi, "registrar": "attested", "note": note}
            continue

        cached = cache.get(key)
        if cached and cached.get("verified") and cached.get("doi", "") == doi:
            continue

        if offline:
            problems.append(f"{key}: no cached verification (re-run without --offline)")
            continue

        record = resolve_doi(doi) if doi else resolve_by_title(title)

        if record is None:
            kind = f"DOI {doi}" if doi else "title search"
            problems.append(f"{key}: did not resolve via {kind} — do not cite until it does")
            updated[key] = {"verified": False, "doi": doi, "reason": "unresolved"}
            continue

        sim = title_similarity(title, record.get("title") or "")
        if sim < TITLE_MATCH_THRESHOLD:
            problems.append(
                f"{key}: resolved record title does not match "
                f"(similarity {sim:.2f})\n"
                f"        .bib:     {title}\n"
                f"        resolved: {record.get('title')}"
            )
            updated[key] = {"verified": False, "doi": doi, "reason": "title_mismatch"}
            continue

        updated[key] = {
            "verified": True,
            "doi": doi,
            "registrar": record.get("registrar"),
            "resolved_title": record.get("title"),
            "similarity": round(sim, 3),
        }

    return problems, updated, attested


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Repeatable: report.tex and report_facct.tex share one refs.bib, so
    # "every entry is cited" only holds across their union. Checking a single
    # file would report the other paper's citations as dead weight.
    ap.add_argument(
        "--tex",
        required=True,
        nargs="+",
        help="LaTeX file(s) whose citations to check; coverage is checked against the union",
    )
    ap.add_argument("--bib", default=str(DEFAULT_BIB))
    ap.add_argument("--cache", default=str(CACHE_PATH))
    ap.add_argument(
        "--offline",
        action="store_true",
        help="structure and field checks only; require a cached resolution",
    )
    args = ap.parse_args()

    def _resolve(raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else REPO_ROOT / path

    tex_paths = [_resolve(t) for t in args.tex]
    bib_path = _resolve(args.bib)

    for path in [*tex_paths, bib_path]:
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 2

    entries = parse_bib(bib_path)
    cited: set[str] = set()
    per_file: list[tuple[Path, int]] = []
    for path in tex_paths:
        keys = parse_citations(path)
        per_file.append((path, len(keys)))
        cited |= keys

    cache_path = Path(args.cache)
    cache: dict[str, Any] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text()).get("entries", {})
        except Exception:  # noqa: BLE001
            cache = {}

    print(f"bib   : {bib_path.relative_to(REPO_ROOT)}  ({len(entries)} entries)")
    for path, count in per_file:
        print(f"tex   : {path.relative_to(REPO_ROOT)}  ({count} citations)")
    if len(per_file) > 1:
        print(f"union : {len(cited)} distinct citations")

    structure = check_structure(entries, cited)
    fields = check_fields(entries)
    resolution, updated, attested = check_resolution(entries, cache, args.offline)

    if not args.offline:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        verified = sum(1 for v in updated.values() if v.get("verified"))
        cache_path.write_text(
            json.dumps(
                {
                    "note": (
                        "Written by scripts/verify_bibliography.py. Each entry was "
                        "resolved against CrossRef, a DataCite-style registrar via "
                        "doi.org content negotiation, or Semantic Scholar title "
                        "search. Lets the check run offline and in CI."
                    ),
                    "verified_count": verified,
                    "total": len(updated),
                    "entries": updated,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    for label, problems in (
        ("STRUCTURE", structure),
        ("FIELDS", fields),
        ("RESOLUTION", resolution),
    ):
        if problems:
            print(f"\n{label}: {len(problems)} problem(s)")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"{label}: ok")

    if attested:
        print(f"\nATTESTED (not auto-resolved): {len(attested)}")
        for line in attested:
            print(f"  - {line}")

    total = len(structure) + len(fields) + len(resolution)
    if total:
        print(f"\nFAIL — {total} problem(s)")
        return 1
    resolved = len(entries) - len(attested)
    print(
        f"\nOK — {len(entries)} entries, all cited; "
        f"{resolved} resolved against a registrar, {len(attested)} attested"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
