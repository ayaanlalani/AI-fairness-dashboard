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

- `german_credit`, `hmda`: packs embed the retained Semantic Scholar harvest
  (3 papers/attribute) via `--research_json artifacts/<ds>/fairness/qualitative_research_evidence.json`.
- `lending_club`: evidence list is empty — the keyless Semantic Scholar API
  returned HTTP 429 during Stage 1, Stage 2 pack generation, and repeated
  retries spaced over ~70 minutes on 2026-07-17. This appears to be the
  keyless tier's steady state, not a transient. The reliable refresh path is
  `SEMANTIC_SCHOLAR_API_KEY` (a free key; the only permitted network
  dependency, not gated by the LLM guardrail): with it set, rerun
  `python3.11 run_pipeline.py --datasets lending_club --steps qualitative`
  then regenerate the lending_club packs with the dry-run command. A later
  successful harvest is now retained across subsequent 429s
  (`scripts/qualitative_analysis.py` keeps non-empty evidence on disk), and
  the run commands below already pass `--research_json` so the refreshed
  JSON is picked up.

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

- 36/36 frozen packs built via `--dry-run`; every pack carries its full prompt.
- Scoring harness round-trips the deterministic mock (100/100 with research
  context for GC/HMDA; 80–85/100 for lending_club pending evidence).
- Refusal detector and hallucination detector validated in
  `tests/test_llm_benchmark_detectors.py`, including one deliberately
  hallucinated metric value (DI = 0.4321 absent from context → flagged) and a
  fabricated citation (→ flagged). Run: `python3.11 -m pytest tests/ -q`.
