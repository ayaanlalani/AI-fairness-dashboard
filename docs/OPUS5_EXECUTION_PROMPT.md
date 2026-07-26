# Opus 5 Execution Prompt — Finish the Lending Fairness Benchmark

> **How to run:** open a Claude Code session with Opus 5 in this repository and
> paste everything below the line into `/loop` (no interval — self-paced), e.g.
> `/loop <contents of §Prompt>`. One coherent stage per iteration, each ending
> in a git commit and standing verification.

---

## Prompt

Work through the remaining stages of this repository's lending-fairness
research program, one coherent stage per iteration, in the order below. This
is the continuation of docs/RESEARCH_STAGING_PROMPT.md: Stages 0–2 are done
(frozen dry-run prompt packs in artifacts/llm_benchmark/dry_run/, validated
scorer + refusal/hallucination detectors in tests/, RUN_MANIFEST.md written,
AOD sign pinned, lending_club metrics oriented on favorable_label 0, CI green).

### Environment facts (read first, do not rediscover)

- Interpreter: Homebrew `python3.11` only. NEVER use or create repo-local
  venvs — they hang under iCloud Drive eviction (repo lives under Desktop/).
  The Makefile already prefers python3.11; `make smoke` and `make test` must
  stay green after every iteration.
- Secrets: `.env` at the repo root contains `SEMANTIC_SCHOLAR_API_KEY` and
  `OPENAI_API_KEY`. Scripts load it via python-dotenv automatically. NEVER
  print, echo, commit, or copy key values anywhere; `.env` stays untracked.
- Guardrails file: `configs/research_guardrails.json`. Gemini stays
  `"blocked"` permanently. OpenAI stays `"blocked"` until the approval gate in
  Stage C below — never flip it yourself before that gate.
- Consolidation footgun: `run_pipeline.py`'s consolidate step rebuilds the
  cross-dataset files in artifacts/consolidated/ and poster_assets/ from ONLY
  the datasets passed via --datasets. Either run all three datasets
  (`german_credit hmda lending_club`) or repair the consolidated files from
  git HEAD afterward. Never leave them holding a subset.
- Evidence retention: scripts/qualitative_analysis.py keeps non-empty
  research-evidence JSONs when a Semantic Scholar harvest returns empty. Do
  not "fix" this; it is deliberate.
- Verification after every iteration: `make smoke`, `make test` (19+ tests
  must pass), `python3.11 -m json.tool` over configs/*.json, and no LLM key
  read outside the gated client paths in scripts/llm_benchmark.py and
  scripts/openai_fairness_analysis.py.

### Stage A — Semantic Scholar harvest for lending_club (no LLM)

With SEMANTIC_SCHOLAR_API_KEY now available via .env:
1. `python3.11 run_pipeline.py --datasets german_credit hmda lending_club
   --steps qualitative` — lending_club should gain 3 papers/attribute in
   metrics/fairness/qualitative_research_evidence.json and inline citations
   in its qualitative_report.md; GC/HMDA retained evidence must survive.
2. Regenerate the lending_club dry-run packs with the refreshed evidence
   (exact command in artifacts/llm_benchmark/RUN_MANIFEST.md, "Exact
   post-approval commands", lending_club block, but WITH `--dry-run` — this
   stage makes no LLM calls). Its dry-run scores should rise to 100/100.
3. Update RUN_MANIFEST.md's "Research evidence provenance" section to reflect
   the successful harvest. Commit.

### Stage B — pre-flight

1. `make test && make smoke`; re-read RUN_MANIFEST.md; confirm all 36 packs
   have prompts and non-empty research context for all three datasets.
2. Set `max_cost_usd_per_run` in configs/research_guardrails.json to `15.0`
   (the user's stated budget for this program — this field is not the
   provider gate and may be set now). Confirm every benchmark command passes
   `--max_cost_usd` ≤ 15 or inherits the config cap. Commit.

### Stage C — GATE: paid OpenAI runs (Track Q + Track H)

STOP and ask the user in chat: "Ready to run the OpenAI benchmark: 36 gpt-4o
calls (Track Q) plus the Track H narrative synthesis, estimated well under
$5 total against your $15 budget. Approve?" Proceed ONLY on an explicit yes
in this session. On approval:
1. Flip `"openai"` to `"approved"` in configs/research_guardrails.json
   (record in the commit message that the user approved in-session with a
   $15 budget).
2. Track Q: run the three RUN_MANIFEST commands (gpt-4o, frozen packs,
   `--research_json`). Persist raw JSON per (dataset, attr, cycle). Track
   cumulative spend from the usage blocks; ABORT remaining calls if
   projected total exceeds $15.
3. Analyze: cross-cycle severity stability, agreement vs deterministic
   classify_severity, hallucination and refusal rates per prompt strategy,
   cost/token accounting. Consolidate tables into artifacts/consolidated/
   and regenerate plots (scripts/visualize_openai_benchmark.py /
   visualize_benchmark.py as available). Commit (Stage 3).
4. Track H: for each use case, one narrative audit per
   docs/RESEARCH_STAGING_PROMPT.md Stage 4 (H1–H5 criteria, ≥1 Semantic
   Scholar citation per attribute, judged only by
   configs/report_quality_rubric.json + guardrails_baseline.json — never
   scored against deterministic numbers). Draft the deterministic-vs-LLM
   contrast prose. Commit (Stage 4). Total spend across Q+H stays ≤ $15;
   report actual spend in both commit messages.

### Stage D — merge to main via PR

The user has authorized remote operations for this step (this supersedes the
earlier local-only policy):
1. From the worktree branch (`claude/relaxed-kepler-768cdc` or its successor),
   ensure a clean tree, then push the branch and open a PR to `main` with
   `gh pr create` — title "Lending-lifecycle fairness benchmark: Stages 2–4 +
   hardening", body summarizing the five hardening commits and the Stage 3/4
   results, ending with the standard Claude Code attribution.
2. Do not force-push and do not merge the PR yourself unless the user says
   to; post the PR URL in chat. If the user says merge, use a regular merge
   (no squash unless they ask).
3. After merge, in the main checkout run `make fairness` once so its
   gitignored metrics/ dirs regenerate, and verify `make test` there.

### Stage E — polish + paper-artifact cleanup

1. Sweep the repo for leftovers: duplicate files with " 2" suffixes
   (e.g. "README 2.md", "benchmark_comparison 2.md"), stale references to
   removed paths, dead config keys. Delete duplicates only after diffing
   against the canonical file; never delete legacy artifacts named in the
   guardrail policy.
2. Annotate artifacts/consolidated/stage1_findings.md with a short
   "Resolution log" appendix (Q3.1 fixed, cross-cutting #3/#4 resolved, with
   commit hashes) — append, do not rewrite history.
3. Overclaim grep per Stage 5 spec: `rg -i "compliant|causal|recompute"`
   over report/ and artifacts/consolidated/ — only intentional uses remain.
   Commit.

### Stage F — findings report tied to NeurIPS D&B + FAccT

Write artifacts/consolidated/FINDINGS.md: the complete results story —
per-use-case deterministic findings (incl. the corrected lending_club gender
DI 0.9931 vs the 0.0 artifact and why the near-degenerate UC3 classifier
limits metric informativeness), intersectional results (UC1 0.61 vs 0.82
marginal; UC2 0.78 vs 0.90), Track Q stability/agreement/hallucination
results, Track H rubric outcomes and the contrast finding. Then two explicit
positioning sections: (a) NeurIPS Datasets & Benchmarks relevance — a frozen,
reproducible, cost-capped benchmark artifact with deterministic ground truth,
released prompts/packs, and an anti-hallucination scoring harness; (b) FAccT
relevance — the human-centered audit framing (Lee et al. CHI EA 2020 lens),
metric-orientation pitfalls as a fairness-practice finding, intersectional
vs marginal audit gaps, and economic-proxy vs protected-class contrasts.
Commit.

### Stage G — first-draft academic paper

Rewrite report/report.tex into a NeurIPS-format first draft (use the NeurIPS
D&B track style; keep the existing report.tex as report_course.tex for
provenance). The user-designated exemplar for framing, voice, and
human-centered argument structure is:

  Lee et al., "Human-Centered Approaches to Fair and Responsible AI",
  CHI EA 2020, DOI 10.1145/3334480.3375158
  (https://dl.acm.org/doi/abs/10.1145/3334480.3375158)

Match its framing style — every fairness claim grounded in the human
decision context — while using the NeurIPS D&B format for structure and
length. Source 2–3 additional related-work citations from that paper's
citation graph (Google Scholar cluster 7844550987917920838; resolvable via
the Semantic Scholar API citations endpoint for the DOI, using the key in
.env). Requirements: use-case-first Results; Track Q tables and Track H
contrast; Limitations (subgroup n, LLM nondeterminism, API versioning, metric
grounding, no causal/legal claims); CHI-EA framing pass in Intro/Discussion
pairing every statistical finding with its decision-context interpretation;
cite Lee et al. 2020 plus the citation-graph picks and the harvested
Semantic Scholar papers (real citations only — verify every bibkey against
the evidence JSONs or the fetched citation records). `make paper` must build
(or document the missing LaTeX dependency). Commit.

### Stage H — explanatory Jupyter notebook

Create notebooks/pipeline_walkthrough.ipynb: a narrated, runnable tour —
(1) architecture map of run_pipeline.py and the dataset registry; (2) data
cleaning/encoding decisions per dataset incl. German Credit's 1/2 label
encoding and lending_club's one-hot protected-attribute decoding; (3) the
fairness metric layer with the favorable-label orientation lesson (show the
0.0-vs-0.9931 gender DI story numerically) and the AIF360 sign conventions;
(4) qualitative engine: severity classification, root-cause mapping, proxy
detection, evidence retention; (5) LLM benchmark: 4-cycle design, frozen
packs, scoring subscores, refusal/hallucination detectors (demo the
fabricated-metric flag on a mock — no API calls in the notebook); (6) results
gallery from artifacts/. Every cell must run top-to-bottom with python3.11
on committed/regenerable files only, no keys required. Execute it fully
(`jupyter nbconvert --execute`) before committing.

### Loop mechanics

Self-pace with ScheduleWakeup between iterations; one stage (or one coherent
sub-chunk of a stage) per iteration; never start Stage C without the explicit
in-session approval; stop the loop after Stage H with a final summary of
total OpenAI spend, PR URL/state, and any items left for the user.
