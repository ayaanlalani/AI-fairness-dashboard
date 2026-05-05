# PRD: Remaining Work for NeurIPS 2026 E&D Artifact Readiness

## 1. Product Summary

Prepare the fairness-audit lending benchmark as a complete NeurIPS 2026 Evaluations & Datasets submission artifact.

The artifact should present a clear evaluation protocol for testing whether LLM-based fairness auditors produce stable, statistically grounded, remediation-ready audits on lending datasets. German Credit and HMDA Georgia remain the primary datasets. OpenAI is the primary LLM benchmark. Gemini is retained only as secondary/legacy provenance.

## 2. Current State

Already started or added in the current working tree:

- README rewritten toward a NeurIPS artifact README.
- Data cards added under `docs/data_cards/`.
- `pyproject.toml`, `Makefile`, `Dockerfile`, GitHub Actions smoke workflow, `LICENSE`, and `CITATION.cff` added.
- `docs/PROBLEM_STATEMENT.md` reframed around the evaluation protocol.
- `report/report.tex` partially updated to make OpenAI primary and avoid metric-recomputation overclaims.
- `run_pipeline.py` updated so OpenAI is the primary `llm_benchmark` path.
- `scripts/openai_fairness_analysis.py` clarified as metric-grounded qualitative benchmarking.
- `scripts/llm_benchmark.py` default changed toward OpenAI, with Gemini available explicitly.

Known verification status:

- `make smoke` passed after making the Makefile prefer `.venv/bin/python`.
- `make -n smoke`, `make -n fairness`, `make -n benchmark`, and `make -n paper` produced the intended commands.
- `make fairness` began running with `.venv/bin/python` but appeared to hang silently inside the first fairness subprocess. This needs follow-up before claiming deterministic reproducibility.

## 3. Goals

1. Make the artifact executable by reviewers with minimal setup.
2. Make the OpenAI-primary benchmark story consistent across README, report, configs, scripts, and artifacts.
3. Ensure data documentation meets responsible release expectations.
4. Provide cheap CI coverage that catches missing files, broken imports, invalid configs, and obvious pipeline regressions.
5. Avoid overclaiming novelty, legal compliance, metric independence, or causal conclusions.

## 4. Non-Goals

- Do not implement end-to-end mitigation algorithms for the submission unless time remains after artifact reproducibility is stable.
- Do not make Gemini the primary benchmark.
- Do not run expensive LLM benchmark jobs during normal verification.
- Do not delete existing legacy artifacts solely for aesthetic cleanup.
- Do not claim the LLM independently computes fairness metrics in the OpenAI workflow.

## 5. Remaining Work

### P0: Fix Deterministic Verification

Problem: `make fairness` appeared to hang after launching `german_credit_dataset/scripts/compute_fairness.py`.

Requirements:

- Run the German Credit fairness script directly with `.venv/bin/python` and diagnose the hang.
- Run the HMDA fairness script directly with `.venv/bin/python`.
- If AIF360 import or execution is the blocker, add a documented fast/smoke path or make the fallback path explicit and reliable.
- Ensure `make fairness` completes locally or document the exact blocker with a reproducible error.
- Prevent `run_pipeline.py` from consolidating failed datasets as if successful.

Acceptance criteria:

- `make smoke` passes.
- `make fairness` either passes end-to-end or exits quickly with a clear actionable error.
- Verification notes in README accurately reflect what reviewers should expect.

### P0: Finish Report Consistency Pass

Requirements:

- Read `report/report.tex` end-to-end after the partial edits.
- Remove remaining Gemini-primary language except clearly labeled legacy comparison discussion.
- Ensure all figure references point to existing OpenAI or deterministic visualizations.
- Ensure cost/token tables match current OpenAI artifacts or are explicitly labeled as an example run.
- Add or verify limitations around:
  - small subgroup size,
  - LLM nondeterminism,
  - OpenAI model/API versioning,
  - metric grounding,
  - lack of causal/legal compliance claims.

Acceptance criteria:

- `rg "Gemini|gemini|independent|recompute|Openai"` over README/docs/report/scripts shows only intentional references.
- `make paper` succeeds or the missing LaTeX dependency is documented.

### P0: Validate Reproducibility Scaffold

Requirements:

- Validate `pyproject.toml` syntax and dependency parity with `requirements.txt`.
- Generate `uv.lock` if `uv` is available locally without network-heavy surprises.
- Build or at least syntax-check the Dockerfile.
- Confirm GitHub Actions workflow runs only cheap local checks.

Acceptance criteria:

- `python3 -m tomllib` or equivalent can parse `pyproject.toml`.
- `python3 -m json.tool` validates all JSON configs.
- CI workflow is present at `.github/workflows/smoke.yml`.

### P1: Artifact Output Hygiene

Requirements:

- Decide whether consolidated generic paths such as `artifacts/consolidated/llm_benchmark_comparison.md` should be redirect stubs or regenerated OpenAI-primary reports.
- Keep legacy Gemini reports clearly labeled as legacy.
- Avoid modifying unrelated pre-existing dirty Gemini JSON artifacts unless they are intentionally part of this work.
- Remove or ignore local system files such as `artifacts/.DS_Store` without destructive cleanup.

Acceptance criteria:

- Artifact paths listed in README exist.
- OpenAI-primary report paths are easy to find.
- Legacy Gemini paths do not contradict the primary framing.

### P1: Data Cards Final Review

Requirements:

- Verify German Credit row counts, protected attribute values, target labels, and split sizes against scripts/artifacts.
- Verify HMDA Georgia source parameters and reconcile any mismatch between `download_hmda.py` defaults and the artifact’s Georgia scope.
- Add exact source URLs where appropriate.
- Confirm licensing/access notes are factual and conservative.

Acceptance criteria:

- `docs/data_cards/german_credit.md` and `docs/data_cards/hmda_georgia.md` are internally consistent with scripts and current artifacts.
- README links to both data cards.

### P1: OpenAI Benchmark Configuration

Requirements:

- Ensure config files name OpenAI as primary provider.
- Decide whether `scripts/llm_benchmark.py` should default to `gpt-4o`, `o3`, or the same model used in current OpenAI artifacts.
- If current artifacts use `o3`, document why scripts/configs may use a cheaper or different default.
- Add a cheap no-API benchmark dry-run if feasible.

Acceptance criteria:

- README, configs, and scripts agree on the default/recommended OpenAI benchmark model or explicitly document the difference.
- Running benchmark requires `OPENAI_API_KEY` and does not silently fall back to Gemini.

### P2: Reviewer Polish

Requirements:

- Add a short artifact checklist to README:
  - setup,
  - smoke,
  - deterministic fairness,
  - OpenAI benchmark,
  - paper build,
  - expected outputs.
- Add a troubleshooting section for dependency, LaTeX, API key, and AIF360 issues.
- Add runtime/cost estimates for reviewer planning.

Acceptance criteria:

- A reviewer can run the documented commands without reading the source first.
- Known limitations are visible before someone runs expensive API calls.

## 6. User Stories

- As a NeurIPS reviewer, I can run a cheap smoke test and know the artifact is structurally intact.
- As a NeurIPS reviewer, I can reproduce deterministic fairness metrics for German Credit and HMDA Georgia.
- As a NeurIPS reviewer with an OpenAI key, I can run or inspect the OpenAI benchmark and understand that it uses Python-computed metrics.
- As a researcher, I can inspect data cards and understand provenance, preprocessing, protected attributes, targets, limitations, and licensing notes.
- As a maintainer, I can distinguish primary OpenAI artifacts from legacy Gemini outputs.

## 7. Risks

- AIF360 or package compatibility may make fairness verification fragile under newer Python versions.
- OpenAI model outputs may drift as API models change.
- HMDA rare subgroup metrics may be misread without uncertainty language.
- Existing dirty Gemini artifacts may confuse review if not clearly labeled.
- Paper figures may break if visualization filenames or paths do not match current OpenAI-primary outputs.

## 8. Milestones

### Milestone 1: Executable Local Artifact

- Fix `make fairness`.
- Validate `make smoke`.
- Validate config and pyproject parsing.

### Milestone 2: Consistent Submission Narrative

- Finish report pass.
- Re-scan README/docs/report/scripts for contradictory Gemini or metric recomputation claims.
- Confirm artifact paths in README.

### Milestone 3: Reviewer-Ready Release

- Run paper build or document blocker.
- Optionally generate `uv.lock`.
- Final git status review and list changed files.
- Prepare remaining gaps section for final submission notes.

## 9. Definition of Done

The artifact is ready for submission when:

- `make smoke` passes.
- `make fairness` passes or has a documented environment-specific blocker with a reproducible fallback.
- `make paper` passes or has a documented missing system dependency.
- OpenAI-primary framing is consistent across README, report, configs, scripts, and primary artifacts.
- Data cards exist and are factual.
- LICENSE and CITATION metadata exist.
- CI runs a cheap smoke test.
- Remaining gaps are documented honestly without overclaiming.
