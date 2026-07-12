# Staged Master Research Prompt — Lending-Industry Fairness Audit Benchmark

> **How to use this document.** This is the execution prompt for the research program. Work proceeds stage by stage; each stage has entry conditions, tasks, acceptance criteria, and ends with a **local git commit**. Do not begin a stage until its gate is satisfied. The guardrail preamble is binding for every stage and overrides anything else in this document.

---

## 0. Guardrail Preamble (binding, all stages)

1. **LLM key gate.** No stage may read, export, or pass `OPENAI_API_KEY` or `GEMINI_API_KEY` (or construct any OpenAI/Gemini/other LLM API client) unless `configs/research_guardrails.json` shows that provider as `"approved"`. Flipping a provider to `"approved"` requires the user's explicit, in-session consent — never infer it, never flip it yourself.
2. **Dry-run first.** All benchmark development uses `scripts/llm_benchmark.py --dry-run`, which builds prompt packs and exercises the scoring harness on mock output with **zero network calls to LLM providers**.
3. **Git policy.** Local commits only, on `feat/*` branches within this worktree. No `git push`, no `git remote` mutations, no PR creation, no force operations. Do not commit secrets, `.env` files, or raw PII.
4. **Semantic Scholar.** Keyless public API only, via `scripts/scholarly_evidence.py` (rate-limited, circuit-breaker protected). This is the one permitted network dependency and is available in every stage.
5. **Metric grounding.** LLMs never compute or recompute fairness metrics. They interpret deterministic, Python-computed metric context (per `docs/PROBLEM_STATEMENT.md`). Any LLM output that asserts a metric value not present in the provided context is a hallucination flag, not a result.
6. **Cost cap.** Any approved LLM run respects `max_cost_usd_per_run` in `configs/research_guardrails.json` and the existing `--max_cost_usd` plumbing.
7. **No overclaiming.** No legal-compliance claims (ECOA/Reg B framing is *context*, not a compliance finding), no causal claims beyond the observed data, no deletion of legacy artifacts.

---

## 1. Research Framing (CHI-EA-inspired)

Style anchor: Lee et al., *Human-Centered Approaches to Fair and Responsible AI* (CHI EA 2020, DOI 10.1145/3334480.3375158) and its citation graph. The through-line: **statistical fairness alone is not an audit** — fairness work must connect metrics to the human decision context in which a model operates. We operationalize that as two deliberately **disjoint** tracks:

- **Track Q (quantitative):** Can an LLM auditor, given fixed deterministic metric evidence, reproduce stable, correctly-calibrated audit judgments? Benchmarked head-to-head against the deterministic pipeline (severity agreement, cross-cycle stability, hallucination rate, refusal rate, cost).
- **Track H (human-centered / qualitative):** Where LLMs are *not* judged against deterministic numbers, but used for what they are good at — synthesizing scholarly evidence (Semantic Scholar), articulating stakeholder-specific root causes, and producing remediation narratives with measurable targets. Judged only by `configs/report_quality_rubric.json` + `configs/guardrails_baseline.json` + the human-centered criteria below.

The tracks intentionally do not share success criteria. Track Q asks "is the LLM a reliable metric interpreter?"; Track H asks "does the LLM raise the ceiling of audit *quality* beyond what the deterministic engine can write?"

**Human-centered quality criteria for Track H** (derived from the CHI EA lens):
- H1. Names concrete stakeholders (applicant, loan officer, compliance team, regulator) and the consequence of the disparity for each.
- H2. Situates each finding in the decision context of the use case (below), not in metric space alone.
- H3. Every root-cause claim is tied to either a computed metric, a dataset property, or a cited paper — no free-floating speculation.
- H4. Remediation includes a named algorithm (reweighing, ThresholdOptimizer, equalized-odds postprocessing, counterfactual analysis) **and** a measurable target (e.g., "DI ≥ 0.8 for age groups").
- H5. States its own epistemic limits (small subgroups, proxy uncertainty, non-causal evidence).

---

## 2. Use-Case Drill-Down (the specific lending decisions)

Each dataset is pinned to one concrete decision point in the lending lifecycle. All analysis, prompts, and write-ups reference the use case, not just the dataset.

### UC1 — Consumer installment-credit scoring (German Credit)
- **Decision:** accept/reject an applicant for installment credit; favorable outcome = creditworthy (`favorable_label 1`).
- **Regulatory context (framing only):** ECOA-style protected classes; age and sex are explicitly protected bases.
- **Protected attributes:** `Sex_original`, `AgeGroup_original` (privileged: 40_plus), `foreign_worker_original` — foreign-worker status is a *national-origin proxy*, the distinctive angle of this use case.
- **Drill-down questions:**
  - Q1.1 Does the under-35 age group face compounding disadvantage when intersected with sex (age×sex subgroup selection rates)?
  - Q1.2 Is `foreign_worker` disparity a data-scarcity artifact (n≈37 foreign workers) or a genuine selection-rate gap? Quantify subgroup-size uncertainty.
  - Q1.3 Which features act as proxies for age (correlation-based proxy detection from `scripts/qualitative_analysis.py`)?

### UC2 — Mortgage underwriting (HMDA, Georgia)
- **Decision:** originate/deny a home-purchase mortgage; favorable outcome = origination (`favorable_label 1`).
- **Regulatory context (framing only):** HMDA reporting and fair-lending review; race/sex/age are the reported bases.
- **Protected attributes:** `race`, `sex`, `age_group`.
- **Drill-down questions:**
  - Q2.1 Race×sex intersectional denial-rate gaps: is the joint disparity larger than either marginal disparity?
  - Q2.2 Do disparities persist within income/loan-amount strata available in the processed features (proxy screen)?
  - Q2.3 How do the five metrics disagree (e.g., DI passes while EOD fails), and what does each disagreement mean for an underwriting audit?

### UC3 — P2P personal-loan default risk (Lending Club)
- **Decision:** platform risk-screening of a personal loan; target is `loan_default` (1 = default), so **favorable outcome = predicted non-default (`favorable_label 0`)**.
- **Regulatory context (framing only):** platform lending sits outside branch-based underwriting but inside ECOA scope; income and loan-size tiers are *economic proxies*, not protected classes — this use case probes proxy-tier fairness.
- **Protected attributes (decoded from one-hot processed features):** `gender` (privileged: male), `income_level` (privileged: medium), `loan_amount_level` (privileged: medium).
- **Drill-down questions:**
  - Q3.1 Gender gap in predicted-non-default rates: the existing artifact shows DI = 0.0 for gender — validate whether that is a tiny-subgroup artifact (test-set n≈600) before narrating it.
  - Q3.2 Do income-tier and loan-size-tier disparities compound for low-income × small-loan applicants?
  - Q3.3 Contrast with UC1/UC2: what changes in audit conclusions when the "protected" attributes are economic proxies rather than legal classes?

**Cross-cutting lifecycle claim:** UC1 (credit scoring) → UC2 (mortgage underwriting) → UC3 (platform risk screening) span the lending lifecycle; the consolidated analysis compares how the *same* audit specification behaves across all three decision points.

---

## 3. Stages

### Stage 0 — Scaffolding (no LLM, no keys)
**Entry:** none. **Tasks:**
1. Register `lending_club` in `run_pipeline.py` `DATASETS` reusing `lending_club_dataset/scripts/{train_models,compute_fairness}.py`, with a prediction-enrichment step that decodes `gender`/`income_level`/`loan_amount_level` from `processed/X_test_processed.csv` into an enriched predictions CSV for the qualitative engine.
2. Add `--dry-run` to `scripts/llm_benchmark.py`: builds all 4-cycle prompts, writes prompt-pack JSONs under `artifacts/llm_benchmark/dry_run/<dataset>/`, runs the scorer on a canned mock response, imports no API client, reads no key.
3. Validate `configs/research_guardrails.json` parses and is referenced by the dry-run path (hard-fail if a provider is `"blocked"` and `--dry-run` is absent).

**Acceptance:** `make smoke` passes; `python3 run_pipeline.py --datasets lending_club --steps fairness` produces `artifacts/lending_club/fairness/*`; dry-run works with `OPENAI_API_KEY`/`GEMINI_API_KEY` unset. **Commit:** `Stage 0: lending_club registration + LLM dry-run + guardrail gate`.

### Stage 1 — Deterministic quantitative baseline (no LLM)
**Entry:** Stage 0 accepted. **Tasks:**
1. Run fairness + qualitative + visualize for `german_credit`, `hmda`, `lending_club` (known risk: `compute_fairness.py` AIF360 hang — diagnose or use the Fairlearn/manual fallback and document which path ran).
2. Produce per-use-case drill-down artifacts answering Q*.1–Q*.3 (per-attribute metrics + intersectional selection-rate tables where subgroup n ≥ 15; report n for every subgroup).
3. Consolidate: `artifacts/consolidated/lending_lifecycle_baseline.md` — one table row per (use case × attribute × metric) plus intersectional highlights, with severity labels from `classify_severity`.

**Acceptance:** fairness CSVs + qualitative reports exist for all three; consolidated baseline exists; every drill-down question has either an answer or a documented blocker. **Commit:** `Stage 1: deterministic lending-lifecycle baseline (3 use cases)`.

### Stage 2 — Prompt packs + dry-run benchmark (no API calls)
**Entry:** Stage 1 accepted. **Tasks:**
1. Build frozen 4-cycle prompt packs per (use case × protected attribute) via `--dry-run`, embedding: deterministic metric context, use-case framing from §2, and Semantic Scholar evidence (`gather_research_context` — keyless, rate-limited).
2. Validate the scoring harness (`score_llm_output`) and hallucination/refusal detectors against mock outputs, including at least one deliberately hallucinated metric value.
3. Write the exact post-approval run commands into `artifacts/llm_benchmark/RUN_MANIFEST.md` (model, attrs, expected call count, estimated cost).

**Acceptance:** prompt packs on disk for all (use case × attribute) pairs; scorer round-trips mocks; manifest states expected cost < cap. **Commit:** `Stage 2: frozen prompt packs + dry-run validation`.

### Stage 3 — Quantitative LLM benchmark (Track Q) — **GATE: user approval**
**Entry:** user explicitly approves OpenAI usage AND `configs/research_guardrails.json` flipped to `"openai": "approved"` by/with the user. **Tasks:**
1. Run the 4-cycle benchmark (`gpt-4o`) for all three use cases from the frozen Stage 2 packs; honor the cost cap; persist raw JSON per (dataset, attr, cycle).
2. Analyze: cross-cycle stability of severity labels; severity agreement vs deterministic `classify_severity`; hallucination and refusal rates per prompt strategy; cost/token accounting.
3. Consolidate into `artifacts/consolidated/` tables + `scripts/visualize_benchmark.py` plots; note where the constrained prompt (cycle 4) changes agreement.

**Acceptance:** all cycles complete or failures documented; agreement/stability tables exist; spend ≤ cap. **Commit:** `Stage 3: Track Q — 4-cycle OpenAI benchmark across lending lifecycle`.

### Stage 4 — Human-centered qualitative track (Track H, disjoint) — **GATE: same approval as Stage 3**
**Entry:** LLM provider approved (Semantic Scholar portion may be prepared earlier). **Tasks:**
1. For each use case, generate a publication-quality narrative audit: LLM synthesis over deterministic context + ≥1 Semantic Scholar citation per attribute (per `configs/guardrails_baseline.json`), explicitly addressing H1–H5.
2. Do **not** score these against deterministic numbers; evaluate only with `configs/report_quality_rubric.json` + H1–H5 checklist.
3. Contrast (qualitatively, in prose) the deterministic engine's template report vs the LLM narrative for the same use case — this contrast is the Track H finding.

**Acceptance:** three narrative audits passing guardrails_baseline checks; rubric scores recorded; contrast section drafted. **Commit:** `Stage 4: Track H — human-centered narrative audits`.

### Stage 5 — Paper artifacts
**Entry:** Stages 3–4 accepted (or explicitly descoped). **Tasks:**
1. Fold results into `report/report.tex`: use-case-first Results restructure, Track Q tables, Track H contrast, limitations (subgroup size, LLM nondeterminism, API versioning, metric grounding, no causal/legal claims).
2. Add a short CHI-EA-style framing pass to Introduction/Discussion: pair every statistical finding with its decision-context interpretation; cite Lee et al. 2020 and 2–3 works from its citation graph (via Semantic Scholar).
3. Regenerate visuals (`scripts/generate_consolidated_visuals.py`, `scripts/visualize_benchmark.py`); `make paper`.

**Acceptance:** `make paper` builds (or missing LaTeX dependency documented); all figure refs resolve; overclaim grep (`rg -i "compliant|causal|recompute"`) shows only intentional uses. **Commit:** `Stage 5: paper artifacts + CHI-EA framing pass`.

---

## 4. Standing Verification (every stage)

- `make smoke` passes; all JSON configs parse (`python3 -m json.tool`).
- With `OPENAI_API_KEY`/`GEMINI_API_KEY` unset, everything up to Stage 2 runs to completion.
- `git log --oneline origin/main..HEAD` shows staged local commits only; no push has occurred.
- No new file reads an LLM key outside the gated client-construction path in `scripts/llm_benchmark.py` / `scripts/openai_fairness_analysis.py`.
