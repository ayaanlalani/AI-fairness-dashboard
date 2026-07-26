"""Stage A validation: Semantic Scholar resilience under a rejected key.

The key shipped in .env returns HTTP 403 on every endpoint. A rejected key is
strictly worse than no key, because the keyless tier still serves requests, so
the client must drop it and retry rather than fail the harvest. When search is
throttled entirely, curated real DOIs are resolved via the /paper/DOI: endpoint
instead. All offline — urlopen is stubbed, no network, no API key.
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

import scholarly_evidence as se


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.semanticscholar.org/", code=code, msg="stub", hdrs=None, fp=None
    )


def _json_response(payload: dict):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Resp(json.dumps(payload).encode("utf-8"))


SEARCH_PAYLOAD = {
    "data": [
        {
            "paperId": "abc123",
            "title": "Live Search Result On Lending Fairness",
            "year": 2021,
            "venue": "FAccT",
            "citationCount": 42,
            "abstract": "An abstract.",
            "authors": [{"name": "A. Author"}],
        }
    ]
}

DOI_PAYLOAD = {
    "paperId": "doi789",
    "title": "Trust and Credit: The Role of Appearance in Peer-to-peer Lending",
    "year": 2012,
    "venue": "Review of Financial Studies",
    "citationCount": 961,
    "abstract": "A real abstract.",
    "authors": [{"name": "J. Duarte"}],
}


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Circuit breakers and the key-disabled flag are process-global."""
    monkeypatch.setattr(se, "_CIRCUIT_OPEN", False)
    monkeypatch.setattr(se, "_DOI_CIRCUIT_OPEN", False)
    monkeypatch.setattr(se, "_KEY_DISABLED", False)
    monkeypatch.setattr(se, "_CONSECUTIVE_ERRORS", 0)
    monkeypatch.setattr(se, "_LAST_REQUEST_TS", 0.0)
    monkeypatch.setattr(se, "_respect_rate_limit", lambda: None)
    monkeypatch.setattr(se.time, "sleep", lambda _s: None)


class TestRejectedKeyFallback:
    def test_403_drops_key_and_retries_keyless(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "invalid-key")
        seen_keys: list[str | None] = []

        def fake_urlopen(req, timeout=None):
            seen_keys.append(req.get_header("X-api-key"))
            if len(seen_keys) == 1:
                raise _http_error(403)
            return _json_response(SEARCH_PAYLOAD)

        monkeypatch.setattr(se.urllib.request, "urlopen", fake_urlopen)

        papers = se.search_semantic_scholar("lending fairness", limit=1)

        assert [p["title"] for p in papers] == ["Live Search Result On Lending Fairness"]
        # First attempt carried the key; the retry carried none.
        assert seen_keys[0] == "invalid-key"
        assert seen_keys[1] is None
        assert se._KEY_DISABLED is True

    def test_disabled_key_is_not_resent_on_later_calls(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "invalid-key")
        monkeypatch.setattr(se, "_KEY_DISABLED", True)
        seen_keys: list[str | None] = []

        def fake_urlopen(req, timeout=None):
            seen_keys.append(req.get_header("X-api-key"))
            return _json_response(SEARCH_PAYLOAD)

        monkeypatch.setattr(se.urllib.request, "urlopen", fake_urlopen)
        se.search_semantic_scholar("lending fairness", limit=1)

        assert seen_keys == [None]


class TestErrorCircuitBreaker:
    def test_two_consecutive_non_429_errors_trip_circuit(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        monkeypatch.setattr(
            se.urllib.request, "urlopen", lambda req, timeout=None: (_ for _ in ()).throw(_http_error(500))
        )

        se.search_semantic_scholar("q1")
        assert se._CIRCUIT_OPEN is False
        se.search_semantic_scholar("q2")
        assert se._CIRCUIT_OPEN is True

    def test_success_resets_error_counter(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 2:
                return _json_response(SEARCH_PAYLOAD)
            raise _http_error(500)

        monkeypatch.setattr(se.urllib.request, "urlopen", fake_urlopen)

        se.search_semantic_scholar("q1")  # error
        se.search_semantic_scholar("q2")  # success -> counter reset
        se.search_semantic_scholar("q3")  # error, but only the first since reset
        assert se._CIRCUIT_OPEN is False


class TestDoiSeedFallback:
    def test_seed_dois_are_topic_matched(self):
        gender = se.seed_dois_for_attrs(["gender"])
        assert gender[0] == "10.1093/rfs/hhs071"  # P2P appearance study leads
        # General fairness work is appended, never dropped.
        assert set(se.DOI_SEEDS_GENERAL).issubset(set(gender))

    def test_unknown_attribute_falls_back_to_general_only(self):
        assert se.seed_dois_for_attrs(["some_unmapped_attr"]) == list(se.DOI_SEEDS_GENERAL)

    def test_doi_records_are_fetched_not_synthesised(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        monkeypatch.setattr(
            se.urllib.request, "urlopen", lambda req, timeout=None: _json_response(DOI_PAYLOAD)
        )

        paper = se.fetch_paper_by_doi("10.1093/rfs/hhs071")

        assert paper is not None
        assert paper["title"] == DOI_PAYLOAD["title"]
        assert paper["citation_count"] == 961
        assert paper["source"] == "doi_seed"

    def test_search_failure_backfills_from_doi_seeds(self, monkeypatch):
        """The Stage A path: search is throttled, DOI lookup still works."""
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)

        def fake_urlopen(req, timeout=None):
            if "/paper/search" in req.full_url:
                raise _http_error(429)
            return _json_response(DOI_PAYLOAD)

        monkeypatch.setattr(se.urllib.request, "urlopen", fake_urlopen)

        papers = se.gather_research_context(
            dataset_name="Lending Club P2P Loans",
            protected_attrs=["gender"],
            max_papers=1,
        )

        assert len(papers) == 1
        assert papers[0]["source"] == "doi_seed"
        assert papers[0]["meets_citation_threshold"] is True

    def test_doi_429_disables_seeds_without_raising(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        monkeypatch.setattr(
            se.urllib.request,
            "urlopen",
            lambda req, timeout=None: (_ for _ in ()).throw(_http_error(429)),
        )

        assert se.gather_doi_seed_papers(["gender"], max_papers=3) == []
        assert se._DOI_CIRCUIT_OPEN is True
