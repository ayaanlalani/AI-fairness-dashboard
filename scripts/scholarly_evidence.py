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
DEFAULT_FIELDS = (
    "title,year,authors,abstract,url,venue,citationCount,externalIds,paperId"
)
REQUEST_INTERVAL_S = 1.1
MAX_RETRIES = 3
_LAST_REQUEST_TS = 0.0


def _respect_rate_limit() -> None:
    global _LAST_REQUEST_TS
    now = time.monotonic()
    wait_s = REQUEST_INTERVAL_S - (now - _LAST_REQUEST_TS)
    if wait_s > 0:
        time.sleep(wait_s)
    _LAST_REQUEST_TS = time.monotonic()


def search_semantic_scholar(
    query: str,
    limit: int = 3,
    fields: str = DEFAULT_FIELDS,
    timeout_s: int = 20,
) -> list[dict[str, Any]]:
    """Return simplified Semantic Scholar search results for a query."""
    params = urllib.parse.urlencode({
        "query": query,
        "limit": limit,
        "fields": fields,
    })
    req = urllib.request.Request(f"{SEMANTIC_SCHOLAR_API}?{params}")
    req.add_header("User-Agent", "AI-Fairness-Dashboard/1.0")

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        req.add_header("x-api-key", api_key)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _respect_rate_limit()
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < MAX_RETRIES:
                wait_s = REQUEST_INTERVAL_S * (2 ** attempt)
                log.warning(
                    f"Semantic Scholar rate limited for '{query}'. Retrying in {wait_s:.1f}s ..."
                )
                time.sleep(wait_s)
                continue
            log.warning(f"Semantic Scholar query failed for '{query}': {exc}")
            return []
        except Exception as exc:
            log.warning(f"Semantic Scholar query failed for '{query}': {exc}")
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
) -> list[dict[str, Any]]:
    """Collect a deduplicated set of papers across several fairness queries."""
    queries = build_default_queries(dataset_name, protected_attrs)
    if extra_queries:
        queries.extend(extra_queries)
    queries = list(dict.fromkeys(queries))

    seen: set[str] = set()
    papers: list[dict[str, Any]] = []

    for query in queries:
        if len(papers) >= max_papers:
            break
        for paper in search_semantic_scholar(query, limit=per_query_limit):
            key = paper.get("paper_id") or paper.get("title", "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            papers.append(paper)
            if len(papers) >= max_papers:
                break

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
