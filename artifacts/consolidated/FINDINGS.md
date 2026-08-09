# Findings — A Lending Fairness Benchmark

> **SUPERSEDED IN PART.** This file records the original test-split analysis.
> The audit has since been re-run on the full cleaned population with the HMDA
> target leak removed, and several figures below have changed or been withdrawn.
> **Read `RESULTS.md` alongside this file** — it lists every changed number and
> every retracted claim. Inline corrections are marked **CORRECTION:**.

Three lending use cases, a deterministic AIF360 baseline, and a frozen,
cost-capped LLM benchmark over the same computed context. This is the complete
results story, including the results that undercut the tidy version.

**Total LLM spend: $0.7094** — Track Q $0.2274 (36 gpt-4o calls), Track H
$0.3514 (3 o3 narrative audits), rubric judge $0.0181, plus $0.1125 from a
pre-freeze run retained for contrast. Cap $15.00, enforced cumulatively with a
pre-call abort.

---

## 1. The three use cases

The three datasets are not three copies of one experiment. They span the
lending lifecycle and differ in the thing that matters most for a fairness
audit: **what kind of attribute is protected, and how discriminating the
underlying classifier is.**

| | UC1 German Credit | UC2 HMDA (Georgia) | UC3 Lending Club |
|---|---|---|---|
| Decision | consumer credit approval | mortgage origination | P2P loan default risk |
| Protected attributes | sex, age group, foreign worker | race, sex, age group | gender, income level, loan amount level |
| Attribute type | protected class | protected class | **one protected class, two economic proxies** |
| Favorable outcome | good credit (label 1) | approved (label 1) | **non-default (label 0)** |
| Classifier behaviour | discriminating | discriminating | **near-degenerate: ~99.67% predicted non-default** |

That last row governs how everything else reads.

> **Disclosure — UC3 `gender` is synthetic.** Lending Club does not collect
> applicant sex. `clean_lending_club.py` assigns it with
> `np.random.random(len(df)) > 0.5`, independent of every feature and of the
> target. UC3 therefore contains **no protected class**, and its near-parity
> gender DI is a property of the construction rather than a finding. The
> protected-class-versus-economic-proxy contrast drawn below does not hold.

## 2. Deterministic results

### UC1 German Credit — the marginal view hides the finding

| Attribute | DI | DPD | EOD | AOD |
|---|---|---|---|---|
| `Sex` (priv. male) | 0.8595 | −0.1152 | −0.0512 | −0.1256 |
| `AgeGroup` (priv. 40+) | 0.8212 | −0.1586 | −0.0513 | −0.1649 |
| `foreign_worker` (priv. 0) | 0.9402 | −0.0498 | −0.0889 | 0.2013 |

Every marginal DI clears the four-fifths screen. The intersection does not:
**`under_40 × female` has DI 0.6087** (selection rate 0.6087 vs 1.0000 for
`40_plus × female`, n=46) — below the 0.72 CRITICAL threshold that **neither
marginal attribute approaches**. Auditing sex and age separately would have
returned three passes and missed the group actually being disadvantaged.

`foreign_worker`'s DI 0.9402 is **not** a finding. The privileged group has
**six records**, under the n ≥ 15 reporting floor, and the marginal is
dominated by the n=194 side. It is reported as a data-scarcity limitation.

### UC2 HMDA — the same lesson, and a different mechanism

| Attribute | DI | DPD | EOD | AOD |
|---|---|---|---|---|
| `race` (priv. White) | 0.8956 | −0.0776 | 0.0025 | 0.0117 |
| `sex` (priv. Male) | 0.9291 | −0.0518 | −0.0076 | −0.0084 |
| `age_group` (priv. mid) | 1.0138 | 0.0097 | 0.0008 | −0.0150 |

`Black × Female` (n=410) reaches **DI 0.7821** against `Asian × Male` (n=130),
exceeding both marginals and moving the conclusion from borderline to the HIGH
range. Both adequately sized — this is a finding, not an artifact.

The error profile looked diagnostic. **CORRECTION:** race DI 0.8956 does *not*
fail the four-fifths screen — 0.8956 > 0.80, `BiasFlag` is `false`, and
`classify_severity` returns MODERATE. The claim was wrong here and propagated to
report.tex. Worse, the near-zero EOD/AOD it was paired with came from a model
with target leakage (`interest_rate`, 99.3% missing on denials, AUC 0.9942).
The label-bias inference is **withdrawn**. See RESULTS.md §1.1. The model is **not** making differentially
worse errors by race; it is reproducing a base-rate difference already in the
labels. That is **label/base-rate bias, not differential model error** — and
the two call for different interventions. A post-processing equal-odds fix
targets a problem this model does not have.

### UC3 Lending Club — a corrected artifact, and an honest null

| Attribute | DI | DPD | EOD | AOD |
|---|---|---|---|---|
| `gender` (priv. male) | **0.9931** | −0.0069 | 0.0 | −0.0357 |
| `income_level` (priv. medium) | 1.0072 | 0.0072 | 0.0 | 0.0323 |
| `loan_amount_level` (priv. medium) | 1.0063 | 0.0062 | 0.0 | 0.0270 |

**Gender DI was originally computed as 0.0 and read as CRITICAL. That was a
metric-orientation artifact, not a disparity.** The target is `loan_default`, so
label 1 is the *unfavorable* outcome. Computing selection rates on predicted
default gave male 0.0000 vs female 0.0069 → DI = 0/0.0069 = 0.0. Oriented on
the favorable outcome (predicted non-default): 1.0000 vs 0.9931 → **DI 0.9931**,
near parity. A factor-of-infinity error in the headline metric, from one
convention.

The near-parity numbers are not evidence of fairness. The classifier predicts
non-default for **~99.67% of the test set** (2 default predictions in 600 → 99.67%), so
every group looks equal because the model barely discriminates at all. The
honest finding is **insufficient model discrimination to measure disparity**.
All intersectional rates are ≥ 0.9848 for the same reason.

UC3 also contributes the only **economic-proxy vs protected-class** contrast in
the set: `income_level` and `loan_amount_level` are not protected classes, and
near-parity on them means something different from near-parity on `gender`.

## 3. Track Q — is the LLM a reliable metric interpreter?

36 gpt-4o calls, prompts replayed verbatim from the frozen Stage 2 packs.
Deterministic ground truth from `classify_severity`.

### Agreement is 75%, and the number should not be quoted alone

| Use case | Agreed | Rate |
|---|---|---|
| german_credit | 4/12 | 33.3% |
| hmda | 11/12 | 91.7% |
| lending_club | 12/12 | 100.0% |
| **Overall** | **27/36** | **75.0%** |

**The ordering is not a competence ranking.** lending_club's 100% is the
easiest possible case: a near-degenerate classifier makes every attribute a
trivial near-parity LOW, and agreeing costs nothing. german_credit's 33.3% is
the hardest: three MODERATE baselines near threshold boundaries, where a small
interpretive difference flips the label. Aggregate agreement is a function of
how discriminating the underlying classifiers are. **A single headline
agreement number for an LLM fairness auditor is not a meaningful quantity.**

Errors are asymmetric in the safe direction: **7 over-escalations, 2
under-escalations**. Under-escalation is the consequential failure — a flagged
disparity reported as milder than it is. Over-escalation costs extra review.

### Prompt phrasing moves the verdict

**3 of 9 attribute pairs changed severity label across the four strategies on
identical input context.** `german_credit/Sex_original` went
HIGH → HIGH → MODERATE → HIGH; `hmda/race` went LOW → MODERATE → MODERATE →
MODERATE. This is the core reliability result: the same model, same numbers,
different wording, different answer.

| Cycle | Strategy | Mean score | Agreement | Hallucination flags |
|---|---|---|---|---|
| 1 | `zero_shot` | 76.67 | 55.6% | 5 |
| 2 | `chain_of_thought` | 80.00 | 77.8% | 1 |
| 3 | `self_critique` | **83.89** | **88.9%** | 1 |
| 4 | `constrained` | 76.56 | 77.8% | 1 |

`self_critique` is best on both axes. `zero_shot` is worst on agreement and
carries 5 of the 8 hallucination flags.

### Research grounding cuts fabrication 8-fold

Like-for-like on the two use cases the pre-freeze run also covered — same
model, same attributes, difference is whether Semantic Scholar evidence was
embedded in the prompt:

| | Hallucination flags | Mean score |
|---|---|---|
| No evidence (pre-freeze) | **16** | 69.08 |
| Evidence embedded (Stage 3) | **2** | 76.67 |

This is the strongest result in the benchmark and it is a *design* result:
grounding the prompt in real retrieved literature suppresses fabricated
citations, and it also raises quality.

### The provider matters more than the prompt

Gemini 2.5 Flash on the same harness: **19 of 24 cycles refused**, mean score
19.79, 14 zero-score cycles (`artifacts/section_f/`). Gemini is permanently
blocked in the guardrail config on that evidence. A benchmark like this
measures provider capability at least as much as it measures method.

## 4. Track H — does the LLM raise the audit-quality ceiling?

Three o3 narrative audits, judged **only** against H1–H5,
`guardrails_baseline.json` and `report_quality_rubric.json` — never against the
deterministic severity numbers, which would just re-run Track Q.

| Use case | Guardrail gates | H1–H5 | Citations verified | Rubric (gpt-4o judge) |
|---|---|---|---|---|
| german_credit | 4/5 | 5/5 | 10/10 | 4.8/5 |
| hmda | 5/5 | 4/5 | 10/10 | 4.7/5 |
| lending_club | 4/5 | 4/5 | 7/7 | 5.0/5 |

**27/27 citations verified** against the harvested evidence pool. The rubric
score is cross-model (o3 generated, gpt-4o graded) and is reported as secondary
— an LLM grading an LLM is weaker than a mechanical gate.

What the narrative form added that the template cannot:

- **Metric fragility.** On `foreign_worker_original` it independently
  reproduced the n=6 caveat *and* quantified it: "one label change would swing
  DI to 0.78 (high severity). Thus, statistical power is inadequate."
- **Refusing the flattering null.** On lending_club it identified the
  near-degenerate classifier unprompted and converted the null into a
  conditional prediction with a mechanism and a citation: "this stems from a
  near-trivial classifier that approves almost every application. Once
  realistic credit thresholds are enforced, literature suggests latent gender
  disparities may surface."
- **Sequenced remediation** with hyperparameters and numeric targets.

Full contrast: `deterministic_vs_llm_contrast.md`. **The two are not competing
implementations. The deterministic pipeline establishes what is true; the
narrative audit establishes what it means for a decision.**

## 5. Two findings about the measuring instruments

These matter for anyone reusing the harness, and both are cases of a mechanical
proxy quietly penalising the better answer.

**The refusal detector conflates brevity with refusal.** It flagged 6 of 9
`constrained`-prompt records; **zero were genuine refusals**. Any narrative
field under 40 characters counts as "effectively empty", so
`how_to_fix: "DisparateImpactRemover"` — precise and on-spec in 22 characters —
reads as a refusal, under a prompt that explicitly asked for terse output. No
flagged record contained a refusal phrase, a missing field, or an invalid
severity. Reporting a "66.7% refusal rate" would have blamed the model for a
measurement artifact. The detector is deliberately unchanged (it is part of the
frozen scoring harness) and the distinction is reported instead.

**Guardrail G4 penalises the epistemically correct answer.** It requires every
mitigation to name a canonical algorithm. The two attributes that fail are
exactly the two whose problem *is* data scarcity — `foreign_worker_original`
(n=6) and `loan_amount_level` (3.2× tier imbalance) — where the audit proposed
acquiring records to a stated floor, Bayesian hierarchical shrinkage,
bootstrapped confidence intervals, and manual review until n is adequate. **No
post-processor repairs an n=6 subgroup**, and applying one would manufacture
false confidence. G4 assumes every fairness problem has an algorithmic fix. The
keyword vocabulary was first normalised for hyphenation (o3 writes
"Calibrated-Equalized-Odds", "Kamiran-Calders Re-weighing"), which was masking
the real signal; these two survive that fix.

## 6. Limitations

- **Subgroup sizes.** `foreign_worker_original` privileged n=6; several HMDA
  race categories are severely sparse. Intersectional cells are reported with n
  and suppressed below n ≥ 15.
- **UC3 is barely informative.** A classifier predicting one class ~99.67% of the
  time cannot exhibit measurable disparity. UC3 tests the *pipeline*, not the
  fairness of a working model.
- **LLM nondeterminism.** No temperature pinning, no seed. The 4-cycle design
  measures across prompt strategies, not repeated sampling of one strategy, so
  cycle-to-cycle differences confound strategy with sampling noise.
- **API versioning.** `gpt-4o` and `o3` are moving targets. The frozen prompt
  packs make the *inputs* reproducible; the outputs are not.
- **Metric grounding.** All metrics are associational. `map_root_causes()` is
  rule-based inference from metric values and feature correlations, not causal
  identification.
- **o3 cost is an upper bound.** Priced at $10/$40 per 1M tokens, the figure
  this repo has always used, which may overstate current list price.
- **No causal and no legal claim.** ECOA, Regulation B and the four-fifths rule
  appear as decision context. A DI below 0.8 is a screening signal, not a
  finding of unlawful discrimination.
- **Known artifact defect.** `consolidated_fairness_metrics.csv` carries two
  disjoint column schemas: german_credit/hmda use `DisparateImpact`-style names,
  lending_club uses `disparate_impact`-style, because consolidation concatenates
  per-dataset CSVs without harmonising. The per-dataset files are authoritative.
- **Single geography, single era.** HMDA is Georgia only. German Credit is 1990s
  German data used as a benchmark, not a current population.

---

## 7. Positioning: NeurIPS Datasets & Benchmarks

The contribution is an **artifact**, not a model result.

1. **Deterministic ground truth.** Severity labels come from a rule-based
   classifier over computed AIF360 metrics, so LLM output is scored against
   something reproducible rather than against another model's opinion.
2. **Frozen, released prompts.** All 36 prompts ship in
   `artifacts/llm_benchmark/dry_run/` with the full text embedded per record.
   The benchmark is replayable without re-deriving context, and a `--dry-run`
   mode reproduces the entire scoring path with **no API key**.
3. **An anti-hallucination scoring harness.** Two detectors: fabricated
   citations (3-gram match against retrieved titles) and asserted metric values
   absent from the provided context. The 16→2 fabrication drop is measured, not
   asserted, and both detectors are unit-tested including a deliberately
   hallucinated DI = 0.4321.
4. **Enforced cost cap.** $15 cumulative across invocations with a pre-call
   abort and a persisted ledger. Total reproduction cost $0.71.
5. **Honest negative results.** A trivially-agreeing use case, an aggregate
   agreement number we argue against quoting, and two findings against our own
   instruments. Benchmarks that only report favourable numbers are less useful.
6. **Reproducibility.** 46 tests, CI, pinned metric conventions
   (favorable-label orientation and AOD sign), documented dataset cards.

Honest gap: three datasets and one primary model is small. This is a benchmark
*design* with a worked demonstration, not a large-scale evaluation.

## 8. Positioning: FAccT

The framing follows Lee et al., *Human-Centered Approaches to Fair and
Responsible AI* (CHI EA 2020): a fairness claim is only meaningful relative to
a human decision context.

1. **Metric orientation as a fairness-practice finding.** UC3's gender DI read
   0.0 (CRITICAL) instead of 0.9931 (near parity) because the favorable label
   was inverted. Not an exotic bug — an ordinary consequence of a target named
   `loan_default` rather than `approved`. **A practitioner using a fairness
   toolkit correctly, on clean data, with a sound metric, can produce a
   maximally wrong conclusion from one convention.** This is an argument for
   orientation to be explicit in tooling rather than inferred, and it is now
   pinned by tests in this repo.
2. **Intersectional vs marginal audit gaps.** In UC1 every marginal passes the
   four-fifths screen while `under_40 × female` sits at 0.6087. In UC2 the joint
   group exceeds both marginals. **Marginal-only auditing — the common practice
   — systematically misses the disadvantaged group.**
3. **Economic proxies are not protected classes.** UC3 audits `income_level` and
   `loan_amount_level` beside `gender`. Near-parity means different things for
   each, and the deterministic pipeline cannot tell them apart. Which attributes
   deserve protection is a normative question the tooling silently defers.
4. **What the audit is *for* decides what counts as a finding.** Race DI 0.8956
   with EOD ≈ 0 in UC2 means label bias, not model error; the fix is upstream of
   the model. UC3's null result means the model is not deciding anything, so
   there is nothing to be fair about yet. **Both are invisible if a DI threshold
   is read as a pass/fail gate** — which is precisely how it tends to get used.
5. **Mechanical proxies for judgement encode assumptions.** Our refusal detector
   punished terseness; our guardrail punished "collect more data" — the correct
   answer for an n=6 subgroup. Both were written in good faith by people who
   knew the domain. **Audit infrastructure carries normative commitments, and
   they surface as penalties on the right answer.**
6. **Stakeholder framing was measurably achievable.** H1 required naming
   concrete stakeholders and the consequence for each; all three audits passed.
   The gap between metric output and decision-relevant narrative is closable,
   and it is closable with grounding rather than scale.
