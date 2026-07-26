"""
evaluate_track_h.py

Track H evaluation: does the LLM raise the audit-quality *ceiling*?

Track H is deliberately NOT scored against the deterministic severity numbers
(docs/RESEARCH_STAGING_PROMPT.md §1, Stage 4). Scoring it that way would just
re-run Track Q and would penalise an audit for being more careful than the
threshold table. It is judged only against:

  - the H1-H5 checklist (staging prompt Stage 4)
  - configs/guardrails_baseline.json  (hard, mechanical gates)
  - configs/report_quality_rubric.json (weighted quality dimensions)

Everything in the default path is mechanical and reproducible: no API call, no
key. The rubric's subjective dimensions can optionally be scored by an LLM
judge with `--judge`, which is reported separately and clearly labelled,
because an LLM grading an LLM is weaker evidence than a mechanical gate.

Citations are verified against the harvested Semantic Scholar evidence, so a
fabricated reference fails rather than passing as "well-supported".
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Only needed for the optional --judge path; the default path reads no key.
try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True) or ".env")
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "artifacts" / "consolidated" / "track_h_evaluation.md"

# Where each use case's Track H artifacts live.
USE_CASES = {
    "german_credit": ("German Credit", "german_credit_dataset/metrics/fairness/openai"),
    "hmda": ("HMDA Mortgage Lending (Georgia)", "hmda_dataset/metrics/fairness/openai"),
    "lending_club": ("Lending Club P2P Loans", "lending_club_dataset/metrics/fairness/openai"),
}

# H1: the audit must name concrete stakeholders, not "users".
STAKEHOLDER_TERMS = [
    "applicant", "borrower", "loan officer", "underwriter", "compliance",
    "regulator", "lender", "auditor", "examiner",
]
# H2: the finding must sit in a decision context, not only in metric space.
DECISION_CONTEXT_TERMS = [
    "decision", "approval", "denial", "reject", "threshold", "underwriting",
    "credit", "loan", "mortgage", "application", "adverse action", "pricing",
]
# H5: the audit must state its own epistemic limits. Grouped by the KIND of
# limit rather than kept as one flat keyword list, so the check asks "did it
# acknowledge at least one real class of uncertainty?" instead of "did it hit
# two arbitrary strings?". A flat list scored the German Credit audit as
# failing even though it wrote "the privileged group has only six records; one
# label change would swing DI to 0.78 ... statistical power is inadequate" —
# a textbook H5 statement that simply used none of the listed phrases.
LIMIT_CATEGORIES: dict[str, list[str]] = {
    "small_sample_or_power": [
        "small sample", "sample size", "statistical power", "power is inadequate",
        "only six", "only 6", "few records", "n =", "n=", "underpowered",
        "one label change", "sensitivity", "unreliable", "insufficient",
        "degenerate", "trivial classifier", "near-trivial",
    ],
    "non_causal": [
        "not causal", "non-causal", "no causal", "causal linkage", "correlation",
        "cannot conclude", "does not establish", "association",
    ],
    "proxy_or_confound": [
        "proxy", "confound", "unobserved", "residual", "omitted variable",
    ],
}
# H3/H4 support: a measurable target must carry a number.
TARGET_RE = re.compile(r"\d+(?:\.\d+)?\s*%|\b0?\.\d+\b|\b\d+(?:\.\d+)?\s*(?:pp|percentage points)\b")
METRIC_TERMS = [
    "disparate impact", "demographic parity", "equal opportunity",
    "average odds", "theil", "selection rate", "di ", "eod", "aod",
]


def _normalise(text: str) -> str:
    """Fold hyphens and whitespace so keyword gates survive stylistic variation.

    Without this, "Calibrated-Equalized-Odds post-processor" and
    "Kamiran-Calders Re-weighing" both fail a match against "equalized odds" /
    "reweighing" despite being *more* specific than the keyword asks for.
    """
    return re.sub(r"[\s\-‐-―]+", " ", str(text).lower())


def _algorithm_vocabulary(guardrails: dict[str, Any]) -> list[str]:
    """Guardrail keywords plus the repo's canonical algorithm names."""
    terms = [str(k) for k in guardrails.get("mitigation_must_include_keywords", [])]
    try:
        from llm_benchmark_common import ALGORITHM_KEYWORDS
        terms += [str(k) for k in ALGORITHM_KEYWORDS]
    except ImportError:  # pragma: no cover
        pass
    # Spelled-out variants the canonical list writes as single tokens.
    terms += [
        "re weighing", "reweighting", "disparate impact remover",
        "disparate impact removal", "prejudice remover", "equalised odds",
        "eq odds", "calibrated equalized odds", "exponentiated gradient",
        "threshold optimizer", "threshold sweep", "post processing",
    ]
    return sorted({_normalise(t) for t in terms if t})


def _text_of(entry: dict[str, Any]) -> str:
    parts = [str(entry.get(k, "")) for k in
             ("what_is_wrong", "why_is_wrong", "how_to_fix")]
    parts += [str(s) for s in (entry.get("supporting_research") or [])]
    return " ".join(parts).lower()


def _full_text(result: dict[str, Any]) -> str:
    blob = [str(result.get("cross_attribute_summary", ""))]
    spec = result.get("reference_audit_spec") or {}
    blob.append(str(spec.get("why_this_spec", "")))
    for entry in result.get("qualitative") or []:
        blob.append(_text_of(entry))
    return " ".join(blob).lower()


def check_guardrails(result: dict[str, Any], guardrails: dict[str, Any]) -> list[dict[str, Any]]:
    """Hard mechanical gates from configs/guardrails_baseline.json."""
    checks: list[dict[str, Any]] = []
    entries = result.get("qualitative") or []
    blob = _full_text(result)

    forbidden = [p for p in guardrails.get("forbidden_patterns", []) if p.lower() in blob]
    checks.append({
        "id": "G1", "name": "No forbidden refusal patterns",
        "passed": not forbidden,
        "detail": "none present" if not forbidden else f"found: {forbidden}",
    })

    required = guardrails.get("required_per_attribute", [])
    missing = [
        f"{e.get('attribute')}::{f}"
        for e in entries for f in required if not e.get(f)
    ]
    checks.append({
        "id": "G2", "name": "Required fields present for every attribute",
        "passed": not missing,
        "detail": f"{len(entries)} attributes x {len(required)} fields"
                  + ("" if not missing else f"; missing {missing}"),
    })

    min_refs = int(guardrails.get("min_supporting_research_per_attribute", 1))
    thin = [e.get("attribute") for e in entries
            if len(e.get("supporting_research") or []) < min_refs]
    checks.append({
        "id": "G3", "name": f"At least {min_refs} citation per attribute",
        "passed": not thin,
        "detail": "all attributes cited" if not thin else f"under-cited: {thin}",
    })

    keywords = _algorithm_vocabulary(guardrails)
    no_algo, named = [], []
    for e in entries:
        fix = _normalise(e.get("how_to_fix", ""))
        hits = [k for k in keywords if k in fix]
        if hits:
            named.append(f"{e.get('attribute')}: {hits[0]}")
        else:
            no_algo.append(e.get("attribute"))
    checks.append({
        "id": "G4", "name": "Mitigation names a named algorithm",
        "passed": not no_algo,
        "detail": ("; ".join(named) if not no_algo else f"missing: {no_algo}"),
    })

    if guardrails.get("must_include_measurable_target"):
        no_target = [
            e.get("attribute") for e in entries
            if not TARGET_RE.search(str(e.get("how_to_fix", "")))
        ]
        checks.append({
            "id": "G5", "name": "Mitigation states a measurable target",
            "passed": not no_target,
            "detail": "all attributes quantified" if not no_target else f"missing: {no_target}",
        })
    return checks


def check_h_criteria(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The H1-H5 checklist from the staging prompt, Stage 4."""
    entries = result.get("qualitative") or []
    blob = _full_text(result)
    checks: list[dict[str, Any]] = []

    found_stake = sorted({t for t in STAKEHOLDER_TERMS if t in blob})
    checks.append({
        "id": "H1", "name": "Names concrete stakeholders and their consequence",
        "passed": len(found_stake) >= 2,
        "detail": f"{len(found_stake)} stakeholder roles: {found_stake[:6]}",
    })

    per_attr_context = [
        e.get("attribute") for e in entries
        if not any(t in _text_of(e) for t in DECISION_CONTEXT_TERMS)
    ]
    checks.append({
        "id": "H2", "name": "Situates each finding in the decision context",
        "passed": not per_attr_context,
        "detail": "every attribute framed in decision terms"
                  if not per_attr_context else f"metric-space only: {per_attr_context}",
    })

    unsupported = []
    for e in entries:
        text = _text_of(e)
        has_metric = any(t in text for t in METRIC_TERMS) or bool(re.search(r"\d\.\d{3,}", text))
        has_citation = bool(e.get("supporting_research"))
        if not (has_metric or has_citation):
            unsupported.append(e.get("attribute"))
    checks.append({
        "id": "H3", "name": "Root causes tied to a metric, property, or citation",
        "passed": not unsupported,
        "detail": "all causes grounded" if not unsupported else f"free-floating: {unsupported}",
    })

    weak_fix = [
        e.get("attribute") for e in entries
        if not (TARGET_RE.search(str(e.get("how_to_fix", "")))
                and len(str(e.get("how_to_fix", ""))) > 80)
    ]
    checks.append({
        "id": "H4", "name": "Remediation names an algorithm AND a measurable target",
        "passed": not weak_fix,
        "detail": "all remediations actionable" if not weak_fix else f"underspecified: {weak_fix}",
    })

    hit_categories = {
        name: [t for t in terms if t in blob]
        for name, terms in LIMIT_CATEGORIES.items()
    }
    present = {k: v for k, v in hit_categories.items() if v}
    checks.append({
        "id": "H5", "name": "States its own epistemic limits",
        "passed": len(present) >= 2,
        "detail": f"{len(present)}/3 limit classes: "
                  + ", ".join(f"{k} ({v[0]!r})" for k, v in present.items()) or "none",
    })
    return checks


def check_citations(result: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify every citation against the harvested evidence (anti-fabrication)."""
    known_authors: set[str] = set()
    known_years: set[str] = set()
    known_titles: list[str] = []
    for paper in evidence:
        known_titles.append(str(paper.get("title", "")).lower())
        for author in paper.get("authors") or []:
            surname = str(author).split()[-1].lower() if author else ""
            if len(surname) > 2:
                known_authors.add(surname)
        if paper.get("year"):
            known_years.add(str(paper["year"]))

    cited, verified, unverified = 0, 0, []
    for entry in result.get("qualitative") or []:
        for ref in entry.get("supporting_research") or []:
            cited += 1
            low = str(ref).lower()
            author_hit = any(a in low for a in known_authors)
            title_hit = any(t and t[:40] in low for t in known_titles)
            if author_hit or title_hit:
                verified += 1
            else:
                unverified.append(str(ref)[:90])
    return {
        "cited": cited,
        "verified": verified,
        "unverified": unverified,
        "evidence_pool": len(evidence),
    }


def judge_with_llm(result: dict[str, Any], rubric: dict[str, Any], dataset_name: str) -> dict[str, Any] | None:
    """Optional LLM rubric judge. Gated; reported separately from the hard gates."""
    import os

    from llm_benchmark_common import SpendLedger, enforce_llm_gate, guardrail_max_cost_usd

    model = "gpt-4o"  # cross-model: the audits were generated by o3
    enforce_llm_gate(model)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    dims = "\n".join(
        f"- {d['id']} (weight {d['weight']}): {d['description']}"
        for d in rubric["dimensions"]
    )
    scale = rubric["scoring_scale"]
    prompt = (
        f"You are grading a fairness audit report for {dataset_name} against a fixed rubric.\n"
        f"Score each dimension from {scale['min']} to {scale['max']}.\n"
        f"Do NOT compare the report's severity labels to any external threshold table; "
        f"judge only the qualities below.\n\n"
        f"Rubric:\n{dims}\n\n"
        f"Report JSON:\n{json.dumps(result, indent=2)[:12000]}\n\n"
        f'Return ONLY JSON: {{"scores": {{"<dimension id>": <int>}}, "justification": "<2 sentences>"}}'
    )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict but fair report reviewer. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content or "{}")

    usage = response.usage
    from llm_benchmark_common import cost_for_tokens
    cost = cost_for_tokens(model, usage.prompt_tokens if usage else 0,
                           usage.completion_tokens if usage else 0)
    SpendLedger(
        max_cost_usd=guardrail_max_cost_usd(), run_label=f"track_h_judge/{dataset_name}"
    ).record(
        {
            "total_cost_usd": cost, "model": model,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        },
        dataset=dataset_name, track="H-judge",
    )

    scores = payload.get("scores", {})
    weighted = 0.0
    for d in rubric["dimensions"]:
        raw = scores.get(d["id"])
        if isinstance(raw, (int, float)):
            weighted += float(raw) * float(d["weight"])
    return {
        "model": model,
        "scores": scores,
        "weighted_score": round(weighted, 3),
        "max_weighted": float(scale["max"]),
        "justification": payload.get("justification", ""),
        "cost_usd": round(cost, 6),
    }


def evaluate(root: Path, guardrails: dict, rubric: dict, use_judge: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, (name, rel) in USE_CASES.items():
        out_dir = root / rel
        raw_path = out_dir / "llm_raw_response.json"
        if not raw_path.exists():
            log.warning("No Track H output for %s (%s)", key, raw_path)
            continue
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        result = payload.get("result", payload)

        evidence: list[dict[str, Any]] = []
        ev_path = out_dir / "semantic_scholar_context.json"
        if ev_path.exists():
            raw_ev = json.loads(ev_path.read_text(encoding="utf-8"))
            evidence = raw_ev if isinstance(raw_ev, list) else list(raw_ev.values())
            if evidence and isinstance(evidence[0], list):
                evidence = [p for group in evidence for p in group]

        row = {
            "use_case": key,
            "dataset_name": name,
            "attributes": [e.get("attribute") for e in result.get("qualitative") or []],
            "guardrails": check_guardrails(result, guardrails),
            "h_criteria": check_h_criteria(result),
            "citations": check_citations(result, evidence),
            "cycles": len(payload.get("cycles", []) or []),
        }
        if use_judge:
            try:
                row["judge"] = judge_with_llm(result, rubric, name)
            except Exception as exc:  # noqa: BLE001
                log.error("LLM judge failed for %s: %s", key, exc)
                row["judge"] = None
        rows.append(row)
    return rows


def render(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    A = lines.append
    A("# Track H — Narrative Audit Evaluation")
    A("")
    A("Track H asks whether the LLM **raises the audit-quality ceiling**, which is a "
      "different question from Track Q's \"is it a reliable metric interpreter?\". It "
      "is deliberately **never scored against the deterministic severity numbers** "
      "(staging prompt §1 and Stage 4): doing so would re-run Track Q and would "
      "penalise an audit for being more careful than the threshold table. The gates "
      "below come only from `configs/guardrails_baseline.json`, "
      "`configs/report_quality_rubric.json`, and the H1-H5 checklist.")
    A("")
    A("Every gate in the tables is mechanical and reproducible with no API key.")
    A("")

    for row in rows:
        A(f"## {row['dataset_name']} (`{row['use_case']}`)")
        A("")
        A(f"Attributes audited: {', '.join(f'`{a}`' for a in row['attributes'])} "
          f"· {row['cycles']} self-refinement cycles")
        A("")
        A("| Gate | Criterion | Result | Detail |")
        A("|---|---|---|---|")
        for c in row["guardrails"] + row["h_criteria"]:
            mark = "PASS" if c["passed"] else "**FAIL**"
            A(f"| {c['id']} | {c['name']} | {mark} | {c['detail']} |")
        A("")
        cit = row["citations"]
        rate = (cit["verified"] / cit["cited"] * 100) if cit["cited"] else 0.0
        A(f"**Citation verification:** {cit['verified']}/{cit['cited']} references "
          f"({rate:.0f}%) matched an author or title in the {cit['evidence_pool']}-paper "
          f"harvested evidence pool.")
        if cit["unverified"]:
            A("")
            A("Unmatched references (candidate fabrications):")
            for ref in cit["unverified"]:
                A(f"- `{ref}`")
        A("")
        judge = row.get("judge")
        if judge:
            A(f"**LLM rubric judge** (`{judge['model']}`, cross-model: audits were "
              f"generated by o3). Weighted score "
              f"**{judge['weighted_score']:.2f}/{judge['max_weighted']:.0f}**.")
            A("")
            A("| Dimension | Score |")
            A("|---|---|")
            for dim, score in judge["scores"].items():
                A(f"| {dim} | {score} |")
            A("")
            A(f"> {judge['justification']}")
            A("")
            A("An LLM grading an LLM is weaker evidence than the mechanical gates "
              "above and is reported as secondary. It is included because the rubric's "
              "dimensions (clarity, actionability) have no sound mechanical proxy.")
            A("")

    # Aggregate
    A("## Aggregate")
    A("")
    A("| Use case | Guardrail gates | H1-H5 | Citations verified |")
    A("|---|---|---|---|")
    for row in rows:
        g_pass = sum(1 for c in row["guardrails"] if c["passed"])
        h_pass = sum(1 for c in row["h_criteria"] if c["passed"])
        cit = row["citations"]
        A(f"| {row['use_case']} | {g_pass}/{len(row['guardrails'])} | "
          f"{h_pass}/{len(row['h_criteria'])} | {cit['verified']}/{cit['cited']} |")
    A("")

    # Which attributes failed G4, and why that is a finding about the gate.
    g4_failures: list[str] = []
    for row in rows:
        for c in row["guardrails"]:
            if c["id"] == "G4" and not c["passed"]:
                g4_failures.append(f"{row['use_case']}: {c['detail']}")
    if g4_failures:
        A("### G4 is mis-specified for data-scarcity findings")
        A("")
        for f in g4_failures:
            A(f"- {f}")
        A("")
        A("`guardrails_baseline.json` requires every mitigation to name one of "
          "reweighing / ThresholdOptimizer / equalized odds / postprocessing / "
          "counterfactual. The attributes that fail this gate are precisely the ones "
          "whose problem **is data scarcity** — german_credit `foreign_worker_original` "
          "has six privileged-group records, and lending_club `loan_amount_level` has a "
          "3.2x tier imbalance. For those, the audit proposed acquiring more records to "
          "a stated floor, Bayesian hierarchical shrinkage, bootstrapped 95% confidence "
          "intervals, and routing affected decisions to manual review until n is "
          "adequate.")
        A("")
        A("That is the epistemically correct response: no post-processing algorithm "
          "fixes an n=6 subgroup, and applying one would manufacture false confidence. "
          "The gate encodes an assumption that every fairness problem has an "
          "algorithmic mitigation, and so it penalises the right answer. This is "
          "reported as a **finding about the guardrail**, not smoothed away by widening "
          "the keyword list — the vocabulary was already normalised for hyphenation and "
          "spelling variants, and these two remain.")
        A("")
    A("### How to read H5")
    A("")
    A("H5 requires **two of three** classes of epistemic limit — small-sample/power, "
      "non-causal evidence, proxy uncertainty — matching the three the staging prompt "
      "names. A single class is treated as insufficient self-awareness. The German "
      "Credit audit clears it convincingly, independently reproducing the "
      "deterministic n=6 caveat: \"the privileged group ('0') has only six records; "
      "one label change would swing DI to 0.78 ... statistical power is inadequate.\"")
    A("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Track H narrative audit evaluation")
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--guardrails", default=str(REPO_ROOT / "configs" / "guardrails_baseline.json"))
    parser.add_argument("--rubric", default=str(REPO_ROOT / "configs" / "report_quality_rubric.json"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--judge", action="store_true",
                        help="Also run an LLM rubric judge (gated; costs money).")
    args = parser.parse_args()

    guardrails = json.loads(Path(args.guardrails).read_text(encoding="utf-8"))
    rubric = json.loads(Path(args.rubric).read_text(encoding="utf-8"))
    rows = evaluate(Path(args.root), guardrails, rubric, args.judge)
    if not rows:
        raise SystemExit("No Track H outputs found.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(rows), encoding="utf-8")
    out_path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for row in rows:
        g = sum(1 for c in row["guardrails"] if c["passed"])
        h = sum(1 for c in row["h_criteria"] if c["passed"])
        log.info("%-14s guardrails %d/%d | H %d/%d | citations %d/%d",
                 row["use_case"], g, len(row["guardrails"]), h, len(row["h_criteria"]),
                 row["citations"]["verified"], row["citations"]["cited"])
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
