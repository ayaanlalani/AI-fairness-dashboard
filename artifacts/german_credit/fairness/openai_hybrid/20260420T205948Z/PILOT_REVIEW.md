# OpenAI Hybrid Self-Improvement Pilot — Review Document
**Dataset:** German Credit  
**Run ID:** `20260420T205948Z`  
**Date:** April 20, 2026  
**Total Spend:** $0.9684 USD (~$1.31 CAD) of $50.00 CAD budget  

---

## 1. What We Were Testing

The goal of this pilot was to answer one question: **can a hybrid LLM self-improvement loop produce better fairness audit reports than a single-pass LLM call?**

We define "better" as a higher hybrid score — a weighted blend of:
- **45%** deterministic score (keyword/severity/completeness checks against our Python-computed baseline)
- **45%** judge score (a second o3 call evaluating the output against a quality rubric)
- **10%** guardrail compliance (did the output avoid forbidden patterns and meet structural requirements?)

The pilot also tested whether judge feedback could be used to derive tighter guardrails for future runs.

---

## 2. Experiment Setup

### Model
| Role | Model | Max Output Tokens |
|---|---|---|
| Generator | o3 | 2,200 |
| Judge | o3 | 1,500 |

### Budget
| Parameter | Value |
|---|---|
| Total budget | $50.00 CAD |
| Reserve (held back) | $10.00 CAD |
| Effective budget | $40.00 CAD (~$29.60 USD) |
| Actual spend | $0.97 USD |

### Self-Improvement Loop
Each experiment runs for **2 cycles**. In each cycle:
1. **Generator** produces a structured fairness audit (JSON schema enforced)
2. **Deterministic scorer** checks the output against our Python-computed fairness baseline
3. **Guardrail checker** validates against forbidden patterns, required fields, citation minimums
4. **Judge** scores the output against a 5-dimension quality rubric and returns `prompt_deltas` and `guardrail_updates`
5. **Reflection step** injects a capped subset of the judge's feedback into the next cycle's prompt (controlled by `reflection_strength`)

After all experiments, judge failures are aggregated to derive `guardrails_refined_v2.json`. A final **validation run** re-runs the best-scoring configuration with the refined guardrails.

### Quality Rubric (Judge Scoring Dimensions)
| Dimension | Weight | What It Measures |
|---|---|---|
| Actionability | 30% | Recommendations are concrete, prioritized, and measurable |
| Clarity | 20% | Understandable by non-technical stakeholders |
| Evidence Quality | 20% | Claims grounded in context and credible research |
| Consistency | 15% | Severity labels and logic consistent across attributes |
| Non-contradiction | 15% | No internal contradictions between diagnosis and mitigation |

### Baseline Guardrails
The generator is penalized for:
- Forbidden hedging phrases: `"cannot determine"`, `"insufficient information"`, `"as an ai language model"`
- Missing required fields per attribute: `what_is_wrong`, `why_is_wrong`, `how_to_fix`, `supporting_research`
- Fewer than 1 supporting research citation per attribute
- Mitigations that don't reference at least one algorithm keyword (`reweighing`, `thresholdoptimizer`, `equalized odds`, `postprocessing`, `counterfactual`)
- Mitigations that contain no measurable target language (`%`, `kpi`, `target`, `threshold`, `reduce`, `increase`)

### Experiment Matrix
| Experiment ID | Prompt Style | Citation Min | Reflection Strength |
|---|---|---|---|
| `baseline_v1` | baseline | 5 | medium |
| `actionability_v1` | actionability | 5 | high |
| `evidence_strict_v1` | evidence_strict | 8 | medium |
| `concise_ops_v1` | concise_ops | 5 | medium |
| `consistency_guarded_v1` | baseline | 8 | high |

**Prompt style effects:**
- `baseline` — stable, complete structure
- `actionability` — prioritize measurable remediation steps with owners and thresholds
- `evidence_strict` — only make claims tied to supplied evidence
- `concise_ops` — concise and operationally focused; avoid generic prose

**Reflection strength effects:**
- `high` — all judge `prompt_deltas` and `guardrail_updates` injected into next cycle
- `medium` — first 3 of each injected
- `low` — first 1 of each injected

---

## 3. Results

### Experiment Scores
| Experiment | Hybrid Score | Cycle Gain | Actual Cost (USD) |
|---|---:|---:|---:|
| `baseline_v1` | 81.26 | 0.0 | $0.1462 |
| `actionability_v1` | 79.90 | 0.0 | $0.1669 |
| `evidence_strict_v1` | **81.70** | 0.0 | $0.1590 |
| `concise_ops_v1` | 78.00 | 0.0 | $0.1527 |
| `consistency_guarded_v1` | 79.90 | 0.0 | $0.1705 |

**Winner:** `evidence_strict_v1` (score: 81.70)

### Validation Result
| | Score |
|---|---:|
| Best matrix experiment (`evidence_strict_v1`) | 81.70 |
| Validation with `guardrails_refined_v2` | **86.05** |
| Delta | **+4.35** |
| Promotion decision | ✅ **Promote** |

The validation run — which re-ran the `evidence_strict_v1` configuration with the refined guardrails — scored 4.35 points higher, triggering the promotion recommendation.

### Judge Failure Patterns (from `guardrails_refined_v2`)
The following failures were flagged by the judge across experiments. All appeared with `count: 1` (i.e. no failure recurred across two or more experiments with identical wording):

| Failure | Count |
|---|---|
| Missing per-group breakdown tables required by spec | 1 |
| Citations appear fabricated / unverifiable; no DOIs or links | 1 |
| Ownership, timeline, and cost of mitigations not specified | 1 |
| `foreign_worker` section labels the minority as "privileged" | 1 |
| Hallucinated quantitative metrics and research citations | 1 |
| Missing required per-group metric tables (only narrative summaries given) | 1 |
| No explicit Theil Index discussion even though listed in spec | 1 |
| `foreign_worker` sets numeric DI target despite unreliable metrics | 1 |
| Technical acronyms left unexplained for non-technical readers | 1 |
| Hallucinated or unverifiable research citations reduce credibility | 1 |

---

## 4. Problems Identified

### P1 — Zero cycle gain across all experiments (highest priority)
Every experiment shows `cycle_gain: 0.0`. With 2 cycles and a reflection step, we expected at least some experiments to improve between cycle 1 and cycle 2. This means the self-improvement loop is not iterating in a meaningful way yet. Possible causes:
- o3 is already near ceiling on this scoring function in a single pass
- 2 cycles is not enough for feedback to compound
- The scoring function may be too coarse to detect incremental improvements (e.g. severity matching is binary, keyword presence is binary)

### P2 — Guardrail refinement produced no new forbidden patterns
The `_derive_refined_guardrails` logic promotes failures to `forbidden_patterns` only when `count >= 2`. Since every failure appeared exactly once, nothing was promoted. The refined guardrails are identical to the baseline except `min_supporting_research` was bumped from 1 to 2. This means the guardrail learning loop effectively didn't fire.

Contributing factor: semantically identical failures were logged with slightly different wording by the judge and therefore didn't aggregate. For example:
- `"citations appear fabricated / unverifiable..."` (count: 1)
- `"hallucinated or unverifiable research citations..."` (count: 1)

These describe the same problem but are treated as distinct strings.

### P3 — Citation hallucination is a recurring real failure
The citation problem appeared in both the smoke run and across multiple pilot experiments. It is currently not enforced by any guardrail rule (the baseline only checks that at least 1 citation exists, not that it is verifiable). This is a genuine output quality risk that the automated loop has not self-corrected.

### P4 — Semantic Scholar rate limiting degraded research context
During both runs, Semantic Scholar returned HTTP 429 on most queries and fell back to cached/fallback papers. The research context injected into prompts was thin (0–2 cited papers, 2–4 fallback papers per attribute). This likely contributed to citation hallucination — the model had weak grounding material to cite from.

### P5 — Promotion decision driven by a single variable
The `+4.35` validation gain appears to come primarily from `min_supporting_research` being bumped from 1 to 2 in the refined guardrails, not from new forbidden patterns or prompt evolution. This is a narrow basis for promotion — it reflects one guardrail threshold change, not a qualitatively better output.

---

## 5. Recommendations

### R1 — Manually add citation guardrails now (do not wait for count threshold)
Add the following to `configs/guardrails_baseline.json` immediately:
```json
"forbidden_patterns": [
  "cannot determine",
  "insufficient information",
  "as an ai language model",
  "et al.",
  "forthcoming",
  "unpublished"
],
"require_citation_format": true
```
And consider adding a `required_citation_fields` check (title + year minimum) to `_apply_guardrail_checks`. Citation hallucination is the most consistent failure across all runs and is a credibility risk for stakeholder-facing reports.

### R2 — Increase cycles to 3–4 before concluding the loop doesn't work
With only 2 cycles, there is not enough evidence that the refinement loop is broken. Feedback from cycle 1 needs at least one more cycle to compound. Recommend re-running with `cycles: 3` or `cycles: 4` before drawing conclusions about loop effectiveness.

### R3 — Add fuzzy/semantic deduplication to failure aggregation
The current aggregation in `_derive_refined_guardrails` uses exact string match after `.strip().lower()`. Add simple substring deduplication or a keyword overlap check so that `"hallucinated citations"` and `"fabricated citations"` map to the same failure bucket. This would likely have promoted the citation failure to `forbidden_patterns` in this run.

### R4 — Investigate scoring function sensitivity
All 5 experiments scored in the 78–82 range with zero cycle gain. Before running more experiments, check whether the hybrid scoring function can distinguish between genuinely different outputs. If severity matching and keyword presence are both binary, two outputs that differ substantially in quality might score identically. Consider adding partial credit or weighted keyword matching to the deterministic scorer.

### R5 — Address Semantic Scholar rate limiting before the next run
Either obtain a Semantic Scholar API key (rate limits are much higher with auth) or add a delay between queries. Thin research context is likely contributing to hallucinated citations. This is a cheap fix with a direct impact on output quality.

---

## 6. Next Steps

| Priority | Step | Owner |
|---|---|---|
| 1 | Add citation guardrails manually to `guardrails_baseline.json` | — |
| 2 | Add Semantic Scholar API key to Deepnote environment secrets | — |
| 3 | Re-run pilot with `cycles: 4` and updated guardrails | — |
| 4 | Implement fuzzy deduplication in `_derive_refined_guardrails` | — |
| 5 | Review scoring function sensitivity (can it detect incremental gains?) | — |
| 6 | If cycle gain remains 0 after 4 cycles, review whether o3 needs a harder task (e.g. longer reports, stricter rubric) | — |

---

## 7. Files for Review

All artifacts are in `artifacts/german_credit/fairness/openai_hybrid/20260420T205948Z/`:

| File | Contents |
|---|---|
| `run_registry.json` / `.csv` | Per-experiment scores, costs, cycle gains |
| `promotion_decision.json` | Final promote/reject decision with scores |
| `validation_summary.json` | Validation run details |
| `guardrails_refined_v2.json` | Derived guardrails (judge failure aggregation) |
| `<experiment_id>/cycles.json` | Per-cycle scores, judge feedback, guardrail violations |
| `<experiment_id>/llm_fairness_report.md` | Final generated fairness audit report |
| `<experiment_id>/benchmark_comparison.md` | Side-by-side: deterministic pipeline vs OpenAI output |
| `<experiment_id>/llm_prompt.txt` | Full prompt evolution across cycles |
