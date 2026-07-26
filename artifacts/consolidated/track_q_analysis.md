# Track Q — 4-Cycle Benchmark Analysis (`gpt-4o`)

Track Q asks a narrow question: **is the LLM a reliable interpreter of fairness metrics it was handed?** It is scored against the deterministic `classify_severity` baseline. It is not a test of audit quality — that is Track H, which is deliberately never scored against these numbers.

Records analysed: **36** (3 use cases x 3 attributes x 4 prompt strategies). Prompts replayed verbatim from the frozen Stage 2 packs.

## Cost and token accounting

| Metric | Value |
|---|---|
| Total spend | **$0.2274** |
| Cost per call | $0.00632 |
| Input tokens | 52,294 |
| Output tokens | 9,664 |
| Total tokens | 61,958 |
| Cumulative API wall clock | 94.8 s |

## Agreement with the deterministic severity baseline

Overall agreement: **27/36 (75.0%)**.

| Use case | Agreed | Scored | Rate |
|---|---|---|---|
| german_credit | 4 | 12 | 33.3% |
| hmda | 11 | 12 | 91.7% |
| lending_club | 12 | 12 | 100.0% |

**The spread across use cases matters more than the 75% headline, and the ordering is not a competence ranking.** lending_club's 100% is the easiest possible case, not the best performance: its classifier predicts non-default for ~99.5% of the test set, so every attribute is a near-parity LOW and agreeing costs the model nothing. german_credit's 33.3% is the hardest: three MODERATE baselines sitting near threshold boundaries, where a small interpretive difference flips the label. Aggregate agreement is therefore a function of how discriminating the underlying classifiers are, and should not be quoted as a single capability number.

Where it disagrees, the direction matters more than the rate:

| Direction | Count | Share |
|---|---|---|
| Matches baseline | 27 | 75.0% |
| **Over-escalates** (harsher than baseline) | 7 | 19.4% |
| **Under-escalates** (milder than baseline) | 2 | 5.6% |

Under-escalation is the consequential error for an audit tool: it means a disparity the deterministic pipeline flagged was reported as less serious than it is. Over-escalation is comparatively safe — it produces extra review, not a missed finding.

## Cross-cycle severity stability

| Use case | Attribute | Baseline | Cycle 1 → 2 → 3 → 4 | Stable | Agreed |
|---|---|---|---|---|---|
| german_credit | `AgeGroup_original` | MODERATE | HIGH → HIGH → HIGH → HIGH | yes | 0/4 |
| german_credit | `Sex_original` | MODERATE | HIGH → HIGH → MODERATE → HIGH | **no** | 1/4 |
| german_credit | `foreign_worker_original` | MODERATE | LOW → MODERATE → MODERATE → MODERATE | **no** | 3/4 |
| hmda | `age_group` | LOW | LOW → LOW → LOW → LOW | yes | 4/4 |
| hmda | `race` | MODERATE | LOW → MODERATE → MODERATE → MODERATE | **no** | 3/4 |
| hmda | `sex` | MODERATE | MODERATE → MODERATE → MODERATE → MODERATE | yes | 4/4 |
| lending_club | `gender` | LOW | LOW → LOW → LOW → LOW | yes | 4/4 |
| lending_club | `income_level` | LOW | LOW → LOW → LOW → LOW | yes | 4/4 |
| lending_club | `loan_amount_level` | LOW | LOW → LOW → LOW → LOW | yes | 4/4 |

3 of 9 (use case x attribute) pairs changed severity label across the four prompt strategies on identical input context. Prompt phrasing alone moves the verdict.

## Per-prompt-strategy behaviour

| Cycle | Strategy | Mean score | Range | Severity agreement | Genuine refusals | Flagged (brevity) | Hallucination rate | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 | `zero_shot` | 76.67 | 65–95 | 55.6% | 0/9 | 0/9 | 33.3% | 5 |
| 2 | `chain_of_thought` | 80.00 | 65–95 | 77.8% | 0/9 | 0/9 | 11.1% | 1 |
| 3 | `self_critique` | 83.89 | 65–95 | 88.9% | 0/9 | 0/9 | 11.1% | 1 |
| 4 | `constrained` | 76.56 | 57–87 | 77.8% | 0/9 | 6/9 | 11.1% | 1 |

### The refusal column needs a caveat, not a headline

`detect_refusal` flagged 6 records. **0 are genuine refusals; 6 are the brevity heuristic firing on a correct answer.** The detector treats any narrative field under 40 characters as "effectively empty", and the `constrained` prompt explicitly asks for terse output — so `how_to_fix: "DisparateImpactRemover"` is flagged despite being a precise, on-spec mitigation in 22 characters. None of the flagged records contain a refusal phrase, a missing field, or an invalid severity.

This is a finding about the *harness*, not the model: a length proxy for substance misclassifies compliance with an instruction to be brief. The detector is deliberately left unchanged — it is part of the frozen Stage 2 scoring harness, and altering it would invalidate comparison against the pre-freeze run — but the distinction is reported wherever the rate is quoted. The scoring penalty is real either way: `constrained` has the lowest floor of any strategy because `completeness` also rewards length.

## Contrast: the pre-freeze run (no research grounding)

`artifacts/llm_benchmark/legacy_pre_freeze/` holds an earlier real gpt-4o run over german_credit and hmda whose prompts carried **no Semantic Scholar evidence** (`research_grounding` 0.0) and were never frozen. Comparing like-for-like on those two use cases isolates what the research context changed.

| Run | Records | Mean score | Refusals | Hallucination flags |
|---|---|---|---|---|
| Pre-freeze, no evidence | 24 | 69.08 | 3 | **16** |
| Stage 3, evidence embedded | 24 | 76.67 | 3 | **2** |

