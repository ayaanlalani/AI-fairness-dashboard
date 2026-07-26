# Track Q Run Manifest — 4-Cycle OpenAI Benchmark (Stage 3)

**Status: BLOCKED — awaiting user approval.** `configs/research_guardrails.json`
shows `"openai": "blocked"`. Flipping it to `"approved"` requires the user's
explicit, in-session consent (guardrail preamble §0.1). Do not run any command
below until that gate is satisfied.

## What will run

| Item | Value |
|---|---|
| Model | `gpt-4o` |
| Use cases | UC1 german_credit, UC2 hmda, UC3 lending_club |
| Attributes | 3 per use case → 9 (use case × attribute) pairs |
| Cycles per attribute | 4 (zero_shot, chain_of_thought, self_critique, constrained) |
| **Total API calls** | **36** |
| Frozen prompt packs | `artifacts/llm_benchmark/dry_run/<dataset>/<attr>_cycle<N>.json` (`prompt` field) |
| Rate limiting | 7 s sleep between cycles (≤10 RPM) |
| Retry policy | up to 5 attempts, exponential backoff base 20 s |
| Cost cap | **$15.00**, enforced (see below) |

### Cost cap enforcement (added Stage B)

Stage B's pre-flight was specified as "confirm every benchmark command passes
`--max_cost_usd` ≤ 15 or inherits the config cap". Neither was possible:
`llm_benchmark.py` had **no cost flag and no cap logic at all** — it only summed
spend post hoc — and `max_cost_usd_per_run` had **zero code readers** anywhere in
the repo. The confirmation would have been vacuous, so the enforcement was built:

- `--max_cost_usd` on `llm_benchmark.py`, defaulting to `max_cost_usd_per_run`
  (now `15.0`). A live run with neither available is a hard argparse error.
- Spend is checked **before** each call using a padded estimate, so the cap
  trips ahead of an overrun rather than after it. `CostCapExceeded` stops the
  remaining cycles and attributes, then still writes `_summary.json` with
  `aborted_on_cost_cap`, `cumulative_spend_usd` and `attributes_completed`.
- The cap is **cumulative across invocations**, persisted to
  `artifacts/llm_benchmark/spend_ledger.json`. Stage 3 runs three separate
  processes; a per-process counter would have allowed 3 × $15.
- Prices live in one table (`llm_benchmark_common.MODEL_PRICES`) so the pre-call
  estimate and the post-hoc accounting cannot drift. gpt-4o is $2.50/$10.00 per
  1M tokens.

Dry runs bill nothing, construct no ledger, and need no cap.

## Cost estimate

Measured from the frozen Stage 2 packs (2026-07-17):

| Quantity | Estimate |
|---|---|
| Total input tokens (36 prompts, chars/4) | ≈ 47,600 (largest single prompt ≈ 2,000 tokens) |
| Total output tokens (700/call assumed) | ≈ 25,200 |
| **Cost @ gpt-4o list price ($2.50/M in, $10.00/M out)** | **≈ $0.37** |
| Cost cap (`max_cost_usd_per_run`) | $37.00 |
| Headroom | ~100× under cap |

Note: `scripts/llm_benchmark.py` prices runs with its `PRICE_INPUT_PER_M`/
`PRICE_OUTPUT_PER_M` constants ($0.10/$0.40 per M), which understate gpt-4o
list price; the $0.37 figure above uses list price and is the conservative
planning number. Either way the run is far below the cap.

## Research evidence provenance

All three datasets now embed 3 papers/attribute via
`--research_json artifacts/<ds>/fairness/qualitative_research_evidence.json`.
Resolved 2026-07-25 (Stage A).

- `german_credit`, `hmda`: unchanged. Packs embed the retained live-search
  Semantic Scholar harvest from the original Stage 1 run. The Stage A re-harvest
  left both evidence JSONs **byte-identical** (verified by SHA-256:
  `b8484281d763908a` and `376ad8b87c8b23f1` before and after), so the frozen
  prompts below are still exactly what Stage 3 will replay.
- `lending_club`: evidence went from 0 to 3 papers/attribute, raising the
  dry-run cycle average from **83.33 to 98.33** (gender 95.0, income_level
  100.0, loan_amount_level 100.0 — the `research_grounding` subscore moved
  0.0 → 10.0). The 95.0 on gender is the scorer's ceiling for that attribute,
  matching `hmda/sex`; it is not a missing-evidence signal.

### How the 429 blocker was actually resolved

The Stage 2 note above attributed the failure to keyless-tier throttling and
prescribed a `SEMANTIC_SCHOLAR_API_KEY`. That prescription does not work:

- The key now present in `.env` returns **HTTP 403 Forbidden on every
  endpoint** — it is invalid/inactive, not rate-limited. Sending it is
  strictly worse than sending nothing.
- `scripts/qualitative_analysis.py` never calls `load_dotenv()`, so the key was
  never picked up during harvest anyway; the 429s were genuine keyless
  throttling of `/paper/search`, which remains throttled.
- The keyless **`/paper/DOI:<doi>` lookup endpoint still serves requests** even
  while `/paper/search` returns 429.

Stage A therefore changed `scripts/scholarly_evidence.py` to:

1. Drop a rejected key (401/403) and retry keyless within the same call, so a
   bad key can never degrade the harvest below the keyless baseline.
2. Backfill from a curated list of **real, verified DOIs** through the lookup
   endpoint when search yields nothing. Records are fetched from Semantic
   Scholar, never hand-written, so titles/authors/years stay authoritative and
   the `detect_hallucinations` 3-gram check keeps working against them.

`scripts/qualitative_analysis.py` retention was tightened at the same time:
retained evidence now also wins when a fresh harvest returns **DOI seeds
only**, not just when it returns nothing. Without this, the seed fallback would
have overwritten the german_credit/hmda live-search evidence and silently
changed the prompts the frozen packs were built from.

lending_club's papers are the topical seeds for its attributes — Duarte et al.
(2012) *Trust and Credit* and Pope & Sydnor (2011) *What's in a Picture?* on
P2P appearance-based disparity, Bartlett et al. (2019) on FinTech-era
consumer-lending discrimination, plus Butler & Cornaggia (2017), Feldman et al.
(2014) and Kamiran & Calders (2011) for the economic-proxy attributes. Coverage
is honest but generic relative to a live topical search; it is recorded here as
a provenance caveat, not presented as a targeted harvest.

## Exact post-approval commands

Run from the repo root. Requires `OPENAI_API_KEY` in the environment/.env and
`"openai": "approved"` in `configs/research_guardrails.json` (user-flipped).
These are the Stage 2 dry-run commands with `--dry-run` removed.

```bash
cd german_credit_dataset && python3.11 ../scripts/llm_benchmark.py \
  --predictions metrics/classification_predictions.csv \
  --fairness_csv metrics/fairness/fairness_metrics.csv \
  --qualitative_report metrics/fairness/qualitative_report.md \
  --target actual --pred_col predicted --favorable_label 1 \
  --protected_attrs 'Sex_original:male,AgeGroup_original:40_plus,foreign_worker_original:0' \
  --dataset_name 'German Credit' --dataset_key german_credit \
  --out_root ../artifacts/llm_benchmark --model gpt-4o \
  --research_json ../artifacts/german_credit/fairness/qualitative_research_evidence.json

cd hmda_dataset && python3.11 ../scripts/llm_benchmark.py \
  --predictions metrics/classification_predictions.csv \
  --fairness_csv metrics/fairness/fairness_metrics.csv \
  --qualitative_report metrics/fairness/qualitative_report.md \
  --target actual --pred_col predicted --favorable_label 1 \
  --protected_attrs 'race:White,sex:Male,age_group:mid' \
  --dataset_name 'HMDA Mortgage Lending (Georgia)' --dataset_key hmda \
  --out_root ../artifacts/llm_benchmark --model gpt-4o \
  --research_json ../artifacts/hmda/fairness/qualitative_research_evidence.json

cd lending_club_dataset && python3.11 ../scripts/llm_benchmark.py \
  --predictions metrics/classification_predictions_enriched.csv \
  --fairness_csv metrics/fairness/fairness_metrics_normalized.csv \
  --qualitative_report metrics/fairness/qualitative_report.md \
  --target actual --pred_col predicted --favorable_label 0 \
  --protected_attrs 'gender:male,income_level:medium,loan_amount_level:medium' \
  --dataset_name 'Lending Club P2P Loans' --dataset_key lending_club \
  --out_root ../artifacts/llm_benchmark --model gpt-4o \
  --research_json ../artifacts/lending_club/fairness/qualitative_research_evidence.json
```

## Validation performed (Stage 2, no API calls)

- 36/36 frozen packs built via `--dry-run`; every pack carries its full prompt
  and non-empty research context (re-verified after the Stage A harvest).
- Scoring harness round-trips the deterministic mock: german_credit 100/100,
  hmda 98.33, lending_club 98.33 (was 83.33 before the Stage A harvest).
- Semantic Scholar resilience (rejected-key fallback, DOI seeds, retention
  precedence) covered by `tests/test_scholarly_evidence_resilience.py`.
  Full suite: 28 tests passing.
- Refusal detector and hallucination detector validated in
  `tests/test_llm_benchmark_detectors.py`, including one deliberately
  hallucinated metric value (DI = 0.4321 absent from context → flagged) and a
  fabricated citation (→ flagged). Run: `python3.11 -m pytest tests/ -q`.
