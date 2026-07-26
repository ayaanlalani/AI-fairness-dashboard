"""
scholarly_evidence.py

Shared Semantic Scholar helpers for fairness benchmarking and reporting.

These helpers intentionally use only the Python standard library so they can be
used from any pipeline step without adding new dependencies.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_PAPER_API = "https://api.semanticscholar.org/graph/v1/paper"
DEFAULT_FIELDS = (
    "title,year,authors,abstract,url,venue,citationCount,externalIds,paperId"
)
REQUEST_INTERVAL_S = 2.0
MAX_RETRIES = 2
_RATE_LIMIT_BACKOFF_S = 15  # base wait on 429; doubles each retry
_CIRCUIT_OPEN = False  # once tripped, skip all Semantic Scholar calls this run
_CONSECUTIVE_ERRORS = 0  # trip circuit after 2 consecutive non-429 failures
_CIRCUIT_ERROR_THRESHOLD = 2
_KEY_DISABLED = False  # set when an API key is rejected; fall back to keyless
_DOI_CIRCUIT_OPEN = False  # separate breaker for the /paper/DOI: endpoint
MIN_CITATION_COUNT = 5
_LAST_REQUEST_TS = 0.0


def _respect_rate_limit() -> None:
    global _LAST_REQUEST_TS
    now = time.monotonic()
    wait_s = REQUEST_INTERVAL_S - (now - _LAST_REQUEST_TS)
    if wait_s > 0:
        time.sleep(wait_s)
    _LAST_REQUEST_TS = time.monotonic()


def _trip_circuit() -> None:
    global _CIRCUIT_OPEN
    _CIRCUIT_OPEN = True
    log.warning("Semantic Scholar circuit breaker OPEN — skipping all remaining queries this run.")


def _record_error() -> None:
    global _CONSECUTIVE_ERRORS
    _CONSECUTIVE_ERRORS += 1
    if _CONSECUTIVE_ERRORS >= _CIRCUIT_ERROR_THRESHOLD:
        _trip_circuit()


def _record_success() -> None:
    global _CONSECUTIVE_ERRORS
    _CONSECUTIVE_ERRORS = 0


def _active_api_key() -> str | None:
    """Return the configured key, unless it has already been rejected upstream."""
    if _KEY_DISABLED:
        return None
    return os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None


def _disable_api_key(reason: str) -> None:
    """Stop sending the API key for the rest of this process.

    A rejected key is strictly worse than no key: the keyless tier still serves
    requests (rate-limited), while an invalid key yields 403 on every call.
    """
    global _KEY_DISABLED
    if not _KEY_DISABLED:
        _KEY_DISABLED = True
        log.warning(
            "SEMANTIC_SCHOLAR_API_KEY rejected (%s) — falling back to the keyless "
            "tier for the rest of this run.",
            reason,
        )


def search_semantic_scholar(
    query: str,
    limit: int = 3,
    fields: str = DEFAULT_FIELDS,
    timeout_s: int = 20,
) -> list[dict[str, Any]]:
    """Return simplified Semantic Scholar search results for a query."""
    if _CIRCUIT_OPEN:
        return []

    params = urllib.parse.urlencode({
        "query": query,
        "limit": limit,
        "fields": fields,
    })

    def _build_request() -> urllib.request.Request:
        req = urllib.request.Request(f"{SEMANTIC_SCHOLAR_API}?{params}")
        req.add_header("User-Agent", "AI-Fairness-Dashboard/1.0")
        api_key = _active_api_key()
        if api_key:
            req.add_header("x-api-key", api_key)
        return req

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _respect_rate_limit()
            with urllib.request.urlopen(_build_request(), timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            _record_success()
            break
        except urllib.error.HTTPError as exc:
            # A 403 means the key itself was rejected. Drop it and retry keyless
            # rather than burning the remaining attempts on the same rejection.
            if exc.code in (401, 403) and _active_api_key():
                _disable_api_key(f"HTTP {exc.code}")
                continue
            if exc.code == 429 and attempt < MAX_RETRIES:
                wait_s = _RATE_LIMIT_BACKOFF_S * (2 ** (attempt - 1))
                log.warning(
                    f"Semantic Scholar rate limited for '{query}'. Retrying in {wait_s:.0f}s ..."
                )
                time.sleep(wait_s)
                continue
            if exc.code == 429:
                _trip_circuit()
            else:
                log.warning(f"Semantic Scholar query failed for '{query}': {exc}")
                _record_error()
            return []
        except Exception as exc:
            log.warning(f"Semantic Scholar query failed for '{query}': {exc}")
            _record_error()
            return []
    else:
        # Every attempt was consumed by key-rejection retries.
        return []

    papers: list[dict[str, Any]] = []
    for item in payload.get("data", []):
        papers.append({
            "paper_id": item.get("paperId"),
            "title": item.get("title") or "Untitled",
            "year": item.get("year"),
            "venue": item.get("venue"),
            "citation_count": item.get("citationCount", 0),
            "url": item.get("url"),
            "abstract": item.get("abstract") or "",
            "authors": [a.get("name", "") for a in item.get("authors", [])[:5]],
            "query": query,
        })
    return papers


# ---------------------------------------------------------------------------
# DOI-seed fallback
#
# The keyless /paper/search endpoint is frequently 429-throttled, but the
# keyless /paper/DOI: lookup still serves requests. When search yields nothing
# we resolve this curated list of *real* DOIs instead, so an attribute still
# gets verifiable citations. Every DOI below was confirmed to resolve against
# the live API; nothing here is synthesised. Records are fetched from Semantic
# Scholar, never hand-written, so titles/authors/years stay authoritative and
# the downstream hallucination detector keeps working.
# ---------------------------------------------------------------------------
DOI_SEEDS_GENERAL: tuple[str, ...] = (
    "10.1145/3457607",           # Mehrabi et al., A Survey on Bias and Fairness in ML
    "10.1145/2783258.2783311",   # Feldman et al., Certifying and Removing Disparate Impact
    "10.1007/s10115-011-0463-8", # Kamiran & Calders, preprocessing without discrimination
    "10.1145/3287560.3287589",   # Friedler et al., comparative study of fairness interventions
    "10.1145/2090236.2090255",   # Dwork et al., Fairness through Awareness
    "10.1089/big.2016.0047",     # Chouldechova, Fair Prediction with Disparate Impact
    "10.1613/jair.1.12814",      # Cheng et al., Socially Responsible AI Algorithms
    "10.1145/3306618.3314287",   # Kim et al., Multiaccuracy post-processing
)

# gendered disparity in peer-to-peer and consumer lending
_DOI_SEEDS_GENDER: tuple[str, ...] = (
    "10.1093/rfs/hhs071",            # Duarte et al., Trust and Credit (P2P appearance)
    "10.3368/jhr.46.1.53",           # Pope & Sydnor, What's in a Picture? (Prosper.com)
    "10.1016/j.jfineco.2021.05.047", # Bartlett et al., Consumer-lending discrimination
)

DOI_SEEDS_BY_TOPIC: dict[str, tuple[str, ...]] = {
    "gender": _DOI_SEEDS_GENDER,
    # "sex" is the same construct under the column name German Credit and HMDA use
    "sex": _DOI_SEEDS_GENDER,
    # economic-proxy attributes: income bands are not a protected class, so the
    # relevant literature is fintech pricing disparity + proxy discrimination
    "income": (
        "10.1016/j.jfineco.2021.05.047", # Bartlett et al., FinTech-era lending discrimination
        "10.1287/mnsc.2016.2560",        # Butler & Cornaggia, local capital market conditions
        "10.1145/2783258.2783311",       # Feldman et al., disparate impact / proxy removal
    ),
    # loan size as a proxy-bearing economic stratum
    "loan_amount": (
        "10.1287/mnsc.2016.2560",        # Butler & Cornaggia, borrowing decisions
        "10.1145/2783258.2783311",       # Feldman et al., proxy removal
        "10.1007/s10115-011-0463-8",     # Kamiran & Calders, reweighing
    ),
}


def fetch_paper_by_doi(
    doi: str,
    fields: str = DEFAULT_FIELDS,
    timeout_s: int = 20,
) -> dict[str, Any] | None:
    """Resolve a single DOI to a simplified Semantic Scholar record."""
    global _DOI_CIRCUIT_OPEN
    if _DOI_CIRCUIT_OPEN:
        return None

    url = (
        f"{SEMANTIC_SCHOLAR_PAPER_API}/DOI:"
        f"{urllib.parse.quote(doi, safe='./')}?fields={fields}"
    )
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "AI-Fairness-Dashboard/1.0")
    api_key = _active_api_key()
    if api_key:
        req.add_header("x-api-key", api_key)

    try:
        _respect_rate_limit()
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            item = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403) and api_key:
            _disable_api_key(f"HTTP {exc.code}")
            return fetch_paper_by_doi(doi, fields=fields, timeout_s=timeout_s)
        if exc.code == 429:
            _DOI_CIRCUIT_OPEN = True
            log.warning("Semantic Scholar DOI lookup rate limited — disabling DOI seeds this run.")
        else:
            log.warning(f"Semantic Scholar DOI lookup failed for {doi}: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning(f"Semantic Scholar DOI lookup failed for {doi}: {exc}")
        return None

    if not item.get("title"):
        return None

    return {
        "paper_id": item.get("paperId"),
        "title": item.get("title"),
        "year": item.get("year"),
        "venue": item.get("venue"),
        "citation_count": item.get("citationCount", 0),
        "url": item.get("url"),
        "abstract": item.get("abstract") or "",
        "authors": [a.get("name", "") for a in item.get("authors", [])[:5]],
        "query": f"DOI seed: {doi}",
        "doi": doi,
        "source": "doi_seed",
    }


def seed_dois_for_attrs(protected_attrs: list[str] | tuple[str, ...]) -> list[str]:
    """Pick topical DOI seeds for the given attributes, then general fairness work."""
    dois: list[str] = []
    for attr in protected_attrs or ():
        key = str(attr).lower()
        for topic, topic_dois in DOI_SEEDS_BY_TOPIC.items():
            if topic in key:
                dois.extend(topic_dois)
    dois.extend(DOI_SEEDS_GENERAL)
    return list(dict.fromkeys(dois))


def gather_doi_seed_papers(
    protected_attrs: list[str] | tuple[str, ...],
    max_papers: int = 3,
    min_citation_count: int = MIN_CITATION_COUNT,
) -> list[dict[str, Any]]:
    """Resolve curated DOI seeds when live search returns nothing."""
    papers: list[dict[str, Any]] = []
    for doi in seed_dois_for_attrs(protected_attrs):
        if len(papers) >= max_papers:
            break
        paper = fetch_paper_by_doi(doi)
        if paper is None:
            if _DOI_CIRCUIT_OPEN:
                break
            continue
        cites = int(paper.get("citation_count", 0) or 0)
        paper["meets_citation_threshold"] = cites >= min_citation_count
        papers.append(paper)
    if papers:
        log.info(
            "Resolved %d DOI-seed papers for %s (keyless search unavailable).",
            len(papers),
            ", ".join(str(a) for a in protected_attrs) or "dataset",
        )
    return papers


def build_default_queries(
    dataset_name: str,
    protected_attrs: list[str] | tuple[str, ...],
) -> list[str]:
    attrs = ", ".join(protected_attrs[:3]) if protected_attrs else "protected attributes"
    return [
        "algorithmic fairness bias detection mitigation machine learning",
        "AIF360 fairlearn bias mitigation demographic parity equal opportunity",
        "proxy discrimination machine learning fairness sample size subgroup reliability",
        f"{dataset_name} fairness bias mitigation {attrs}",
    ]


def build_attribute_queries(
    dataset_name: str,
    attr: str,
    causes: list[str],
    fixes: list[str],
) -> list[str]:
    queries = [
        f"{dataset_name} fairness {attr} bias mitigation",
        f"{attr} proxy discrimination machine learning fairness",
    ]

    joined_causes = " ".join(causes).lower()
    joined_fixes = " ".join(fixes).lower()

    if "historical / label bias" in joined_causes or "historical bias" in joined_causes:
        queries.append("historical label bias fairness machine learning mitigation")
    if "proxy discrimination" in joined_causes or "proxy" in joined_causes:
        queries.append("proxy discrimination fairness machine learning mitigation")
    if "representation bias" in joined_causes or "imbalance" in joined_causes:
        queries.append("subgroup sample size reliability fairness machine learning")

    if "reweigh" in joined_fixes:
        queries.append("reweighing fairness machine learning")
    if "thresholdoptimizer" in joined_fixes or "equalized odds" in joined_fixes:
        queries.append("equalized odds postprocessing fairness machine learning")
    if "prejudice remover" in joined_fixes:
        queries.append("prejudice remover fair classification")
    if "disparate impact remover" in joined_fixes:
        queries.append("disparate impact remover preprocessing fairness")

    return queries


def gather_research_context(
    dataset_name: str,
    protected_attrs: list[str] | tuple[str, ...],
    extra_queries: list[str] | None = None,
    max_papers: int = 6,
    per_query_limit: int = 2,
    min_citation_count: int = MIN_CITATION_COUNT,
) -> list[dict[str, Any]]:
    """Collect papers across fairness queries with a citation-quality preference.

    We first keep papers meeting the citation threshold to improve credibility.
    If there are not enough, we backfill with lower-citation papers so recent but
    relevant work is not excluded entirely.
    """
    queries = build_default_queries(dataset_name, protected_attrs)
    if extra_queries:
        queries.extend(extra_queries)
    queries = list(dict.fromkeys(queries))

    seen: set[str] = set()
    preferred: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    consecutive_failures = 0

    for query in queries:
        if len(preferred) >= max_papers:
            break
        if consecutive_failures >= 2:
            log.warning(
                "Semantic Scholar circuit breaker: %d consecutive failures — "
                "skipping remaining queries for this context.",
                consecutive_failures,
            )
            break
        results = search_semantic_scholar(query, limit=per_query_limit)
        if not results:
            consecutive_failures += 1
            continue
        consecutive_failures = 0
        for paper in results:
            key = paper.get("paper_id") or paper.get("title", "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            cites = int(paper.get("citation_count", 0) or 0)
            paper["meets_citation_threshold"] = cites >= min_citation_count
            if paper["meets_citation_threshold"]:
                preferred.append(paper)
            else:
                fallback.append(paper)
            if len(preferred) >= max_papers:
                break

    preferred.sort(key=lambda p: (int(p.get("citation_count", 0) or 0), int(p.get("year") or 0)), reverse=True)
    fallback.sort(key=lambda p: (int(p.get("year") or 0), int(p.get("citation_count", 0) or 0)), reverse=True)

    papers = preferred[:max_papers]
    if len(papers) < max_papers:
        papers.extend(fallback[: max_papers - len(papers)])

    # Live search exhausted without filling the quota (usually keyless 429s).
    # Backfill from curated real DOIs via the lookup endpoint, which stays
    # available when search does not.
    if len(papers) < max_papers:
        seen_titles = {str(p.get("title", "")).lower() for p in papers}
        for paper in gather_doi_seed_papers(
            protected_attrs,
            max_papers=max_papers - len(papers),
            min_citation_count=min_citation_count,
        ):
            if str(paper.get("title", "")).lower() in seen_titles:
                continue
            seen_titles.add(str(paper.get("title", "")).lower())
            papers.append(paper)

    if papers:
        preferred_count = sum(1 for paper in papers if paper.get("meets_citation_threshold"))
        if preferred_count < len(papers):
            log.info(
                "Semantic Scholar context for %s used %d cited papers and %d fallback papers (<%d citations).",
                dataset_name,
                preferred_count,
                len(papers) - preferred_count,
                min_citation_count,
            )

    return papers


def _trim(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def format_evidence_for_prompt(papers: list[dict[str, Any]]) -> str:
    """Render papers as a compact context block for an LLM prompt."""
    if not papers:
        return "No external research evidence was retrieved."

    lines = []
    for idx, paper in enumerate(papers, 1):
        authors = ", ".join(paper.get("authors", [])[:3]) or "Unknown authors"
        year = paper.get("year") or "n.d."
        venue = paper.get("venue") or "Unknown venue"
        cites = paper.get("citation_count", 0)
        abstract = _trim(paper.get("abstract", ""))
        lines.append(
            f"[{idx}] {paper.get('title', 'Untitled')} ({year}) | {authors} | "
            f"{venue} | citations={cites}\n"
            f"    Summary: {abstract}"
        )
    return "\n".join(lines)


def format_evidence_markdown(
    papers: list[dict[str, Any]],
    heading: str = "### Research-backed evidence",
) -> list[str]:
    """Render papers as markdown lines for deterministic reports."""
    lines = [heading, ""]
    if not papers:
        lines.append("No external research evidence was retrieved for this section.")
        lines.append("")
        return lines

    for idx, paper in enumerate(papers, 1):
        authors = ", ".join(paper.get("authors", [])[:3]) or "Unknown authors"
        year = paper.get("year") or "n.d."
        venue = paper.get("venue") or "Unknown venue"
        cites = paper.get("citation_count", 0)
        lines.append(
            f"{idx}. **{paper.get('title', 'Untitled')}** ({year}), {authors}. "
            f"*{venue}*. Citations: {cites}."
        )
        if paper.get("abstract"):
            lines.append(f"   {_trim(paper['abstract'])}")
        if paper.get("url"):
            lines.append(f"   {paper['url']}")
    lines.append("")
    return lines
