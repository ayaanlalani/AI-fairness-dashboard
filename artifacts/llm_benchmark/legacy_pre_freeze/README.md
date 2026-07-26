# Legacy pre-freeze gpt-4o run — NOT the Stage 3 deliverable

These 24 result files (german_credit and hmda, 3 attributes x 4 cycles each)
are a **real, billed gpt-4o run** carried over from the earlier Section C work.
They are preserved for contrast, not presented as Track Q results.

## Why they are not Stage 3 output

| Signal | This run | Stage 3 requirement |
|---|---|---|
| `prompt` field in each record | **absent** | present (replayed from the frozen packs) |
| `research_grounding` subscore | **0.0** | 10.0 (Semantic Scholar evidence embedded) |
| Guardrail state when it ran | `"openai": "blocked"` | `"approved"` after explicit in-session consent |
| Provenance | pre-Stage-2, prompts not frozen | replays `artifacts/llm_benchmark/dry_run/` verbatim |
| Coverage | 2 of 3 use cases (no lending_club) | all 3 |

Because the prompts were never frozen, these records cannot be reproduced
exactly, which is precisely the gap Stage 2's frozen packs exist to close.

## Recorded spend

| Dataset | Calls | Cost | Wall clock | Refusals | Hallucination flags |
|---|---|---|---|---|---|
| german_credit | 12 | $0.0549 | 121.3 s | 1 | 5 |
| hmda | 12 | $0.0576 | 109.3 s | 2 | 5 |
| **Total** | **24** | **$0.1125** | | **3** | **10** |

This $0.1125 was spent before the Stage B cost ledger existed, so it is not in
`artifacts/llm_benchmark/spend_ledger.json`. It must be added to any
program-level spend total reported for the paper.

## Why it is still useful

Cycle-average scores rise across the prompt strategies while hallucination
flags cluster in the *later* cycles:

| Dataset | c1 zero_shot | c2 chain_of_thought | c3 self_critique | c4 constrained |
|---|---|---|---|---|
| german_credit | 63.33 | 53.33 | 73.33 | 75.67 |
| hmda | 60.00 | 76.67 | 76.67 | 73.67 |

All 10 hallucination flags fall in cycles 2-4; in german_credit every cycle-4
attribute carries exactly 2, and the cycle-4 `Sex_original` record is flagged
as a refusal *and* carries 2 hallucinations. That yields a testable hypothesis
for Stage 3 on the frozen, evidence-grounded packs: **the constrained and
self-critique prompts raise scores while increasing metric fabrication.** If
Stage 3 reproduces it with research grounding present, it is a Track Q finding
about prompt strategy rather than an artefact of missing context.

See also `artifacts/section_f/gemini_hallucination_summary.json` for the Gemini
provider record (19 of 24 cycles refused, mean score 19.79, 14 zero-score
cycles), which is why Gemini is permanently blocked in the guardrail config.
