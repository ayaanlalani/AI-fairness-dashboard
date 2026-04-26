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

## 7. Research Insights and Implications

This section interprets the results as research findings rather than operational issues. The pilot was not just a system test — it produced evidence relevant to the core research questions around LLM-assisted fairness auditing.

---

### Finding 1 — Evidence-constrained prompting outperforms actionability-focused prompting for fairness auditing

`evidence_strict_v1` won the matrix (81.70) while `actionability_v1` (79.90) and `concise_ops_v1` (78.00) underperformed. The style that instructed the model to *only make claims grounded in supplied evidence* consistently outperformed styles that asked it to *be more actionable* or *more concise*.

**Implication:** For high-stakes analytical tasks where hallucination risk is real, constraining the model's epistemic scope (what it is allowed to claim) matters more than shaping its rhetorical style (how it should express things). This has direct relevance to fairness auditing: a report that is slightly less actionable but grounded in real evidence is safer than one that sounds authoritative but cites fabricated research.

**For the deterministic model:** The qualitative output stage of your pipeline could benefit from an analogous constraint — explicitly marking which claims are derived from computed metrics versus inferred, and rejecting outputs that introduce ungrounded causal claims.

---

### Finding 2 — Self-reflective prompting did not improve output quality; structural guardrails did

The self-improvement loop (generator → judge → prompt deltas → next cycle) produced zero gain across all 5 experiments. But the validation run, which applied structurally tighter guardrails (`min_supporting_research: 2`), gained +4.35 points.

This is a meaningful research signal: **what you prohibit is more effective than what you encourage** in this context. Telling the model "you must include 2 citations per attribute" moved the needle; feeding it a list of qualitative critique bullets did not.

This connects to a broader debate in the LLM literature. Huang et al. (2023) showed that LLMs cannot reliably self-correct reasoning errors without external signals. Our result extends this to a structured analytical task: even with an external judge providing explicit feedback, the model does not meaningfully revise between cycles. The feedback is acknowledged but not acted upon in a way the scoring function can detect.

**For the research question ("can a hybrid loop outperform a single pass?"):** The answer from this pilot is: *not via reflective prompting, but potentially via guardrail evolution*. The loop's value is in accumulating failure patterns to tighten constraints for future runs — not in within-run self-correction.

**For the deterministic model:** This suggests that investing in tighter output specifications (what the model must and must not produce) will yield more reliable quality improvement than iterative prompting. If you are considering adding a self-reflection step to the deterministic pipeline, the evidence here suggests it should be guardrail-driven rather than free-form critique injection.

---

### Finding 3 — The scoring function may be measuring compliance, not quality

All 5 experiments scored in the 78–82 range despite meaningfully different prompt styles and judge failure patterns. The judge flagged distinct problems per experiment — `evidence_strict_v1` had citation issues, `concise_ops_v1` had missing structure, `actionability_v1` had ownership gaps — but the hybrid scores were nearly indistinguishable.

This suggests the hybrid scoring function (45% deterministic + 45% judge overall score + 10% guardrail compliance) is measuring *structural compliance* rather than *output quality*. An output that includes all required fields, mentions the right algorithm keywords, and avoids forbidden phrases will score ~80 regardless of whether its reasoning is sound.

**Implication for the research design:** The scoring function needs to be able to distinguish between a report that *contains* a mitigation recommendation and one that *correctly derives* a mitigation recommendation from the evidence. Right now it cannot. Before running more cycles or more experiments, it is worth asking: would a significantly worse report score materially lower? If not, the optimisation target is wrong.

**For the deterministic model:** This is the same problem your deterministic pipeline faces — keyword presence and severity label matching are necessary but not sufficient quality signals. The judge failure patterns (missing per-group tables, Theil Index not discussed, mislabelled privilege direction) are exactly the kinds of errors the deterministic scorer would also miss. Incorporating these as explicit checks would improve both pipelines.

---

### Finding 4 — Citation hallucination is a structural problem, not a prompting problem

Citation fabrication appeared in both the smoke run and across the pilot, despite o3 being given explicit research evidence via Semantic Scholar. The model cited sources that were either unverifiable or not in the supplied context.

This is not a prompt design failure — it is a known behaviour of LLMs when asked to produce research-grounded outputs with thin or rate-limited evidence context. The model fills evidence gaps with plausible-sounding but fabricated citations.

**Implication:** Any LLM-based fairness auditing system that asks the model to produce research citations without grounding verification is inherently unreliable on that dimension. The fix is architectural: either restrict citations to only what was provided in the prompt (enforced via guardrail), or add a post-hoc verification step that checks citations against the Semantic Scholar evidence actually supplied.

**For the deterministic model:** Your deterministic pipeline already has a stronger story here — it uses real Semantic Scholar evidence gathered before the LLM call and formats it for injection. The LLM's citation problem is partly a Semantic Scholar rate-limiting problem (thin context → hallucination). An authenticated API key would materially reduce this. But even with full context, a guardrail that cross-checks citations against the supplied evidence block would be a meaningful addition.

---

### Finding 5 — Prompt style ordering reveals a cost-quality tradeoff worth studying

Ranked by score: `evidence_strict` (81.70) > `baseline` (81.26) > `actionability` = `consistency_guarded` (79.90) > `concise_ops` (78.00).

The ordering is counterintuitive in two ways:
1. `actionability` — the style most aligned with the rubric's top-weighted dimension (30% weight) — did not win. Optimising directly for the top rubric dimension did not maximise the overall rubric score.
2. `concise_ops` scored lowest despite being the most "stakeholder-friendly" style. Operational conciseness and audit completeness are in tension — the rubric rewards thoroughness more than brevity.

**For the research question:** This suggests that prompt style and rubric alignment do not have a simple linear relationship. A model explicitly told to maximise the highest-weighted rubric dimension will trade off against other dimensions in ways that reduce the overall score. This is relevant to any future rubric design — if you weight actionability at 30%, you should expect models instructed to prioritise actionability to sacrifice evidence quality and completeness.

---

### Open Research Questions From This Pilot

1. **Does the self-improvement loop add value over more cycles?** The pilot used 2 cycles. If gain remains zero at 4–6 cycles, the reflective prompting approach should be deprioritised in favour of guardrail evolution as the primary improvement mechanism.

2. **Is `evidence_strict` the winning style across datasets, or is it specific to German Credit?** Running the same matrix on HMDA would reveal whether evidence-constrained prompting is a robust finding or an artefact of this dataset's characteristics.

3. **Can the scoring function be made sensitive enough to detect the differences the judge detects?** The judge flagged distinct failure patterns per experiment that the hybrid score did not reflect. Closing this gap is a prerequisite for the optimisation loop to function as intended.

4. **What is the minimum guardrail set that produces consistently acceptable output?** The pilot suggests structural constraints matter more than prompting style. A systematic ablation of guardrail rules (turn them on/off independently) would identify which constraints drive the most quality improvement per unit of constraint cost.

5. **Can the deterministic pipeline's qualitative output stage be improved by applying the guardrail approach learned here?** The failures the judge identified — mislabelled privilege direction, Theil Index not discussed, missing per-group tables — are all errors the deterministic pipeline could in principle catch. Translating judge failures into deterministic checks is a concrete path to improving the non-LLM side of the system.

---

## 8. Files for Review

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
