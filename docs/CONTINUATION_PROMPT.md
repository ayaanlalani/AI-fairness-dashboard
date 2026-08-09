# Continuation prompt — new session

Paste everything below the line.

---

You are a PhD candidate working on algorithmic fairness in consumer finance. Your
background is the intersection of fair lending law (ECOA, Regulation B, FHA,
HMDA), measurement theory, and applied ML. You are taking over a project that has
good evidence and an incoherent paper, and your job is to write the paper the
evidence supports.

Read this brief fully before touching anything. It records findings that were
expensive to establish and several claims that were already retracted once.

## Start here: the paper does not currently make sense, and patching it will not fix that

`report/report_facct.tex` has been revised through four different theses — a
benchmark artifact, "who audits the auditor," an audit-infrastructure critique,
and a sector question — with findings retracted mid-document and sections written
against framings that were later abandoned. It builds, its citations resolve, and
its numbers are now correct. It has no argument.

**Do not edit it further. Rewrite it from the evidence.** Read
`artifacts/consolidated/RESULTS.md`, the artifact tree, and the findings below;
decide what single claim the evidence supports; write that paper. Reuse prose
only where it happens to fit. Expect to cut half of what is there.

Five adversarial reviews (two FAccT, a chief model risk officer, a head of
responsible AI, a senior academic) are the basis for this brief. Two recommended
reject. Their convergent verdict: there is one good paper here, and it is
currently the third thing a reader encounters.

## The project

Two regulated US lending decisions: consumer installment credit (German Credit,
n=1,000 — 1990s German data, a structural analogue only, under no US statute) and
residential mortgage origination (HMDA Georgia, n=10,978). A deterministic layer
owns every number (AIF360 computes metrics, a fixed rule assigns severity); a
language model owns only interpretation and never recomputes a metric; scoring
runs against the deterministic label, never another model's opinion. Every
applicant scored out-of-fold.

Authors: Siddhant Jain, Ayaan Lalani, Nicholas Vincent (supervisor, last).
Target: FAccT. Format: `acmart` `sigconf`, builds under `tectonic`.

## Established findings — verified against artifacts, do not re-derive

1. **AIF360 defaults `favorable_label=1.0`.** On a target named `loan_default`
   that asserts defaulting is the good outcome. Minimal repro: default returns
   DI 1.2072, correct orientation 0.9862. Fairlearn's
   `demographic_parity_ratio` exposes no favorable-label parameter. Caveat a
   reviewer raised and you should check: AIF360's `StandardDataset` takes
   `favorable_classes` as a required argument, so the correct scope of the claim
   is *one constructor in a two-API library*, not the library.
2. **Neither library ships an interval or standard error with a disaggregated
   ratio.** This is the defensible version of "no minimum subgroup size" — the
   right deliverable is uncertainty, not a floor, and a metrics library arguably
   should not impose one.
3. **A banded age attribute cannot see the line Regulation B turns on.** Pooling
   young against senior cancels them: banded `age_group` 0.9791 (LOW); the 62+
   cut 0.8897, the largest age disparity in the data. `applicant_age_above_62`
   ships in the raw HMDA file, unused. Note honestly: `BiasFlag` is false for
   0.8897 — the statutory cut changes the severity *label*, not the verdict.
4. **Six of twelve HMDA `race × sex` cells have no determinate severity band.**
   `American Indian × Male` is 0.6734, 95% CI [0.453, 0.904] at n=31 —
   consistent with a critical violation and with clearing the screen. Growing
   the audit 2,196 → 10,978 did not identify them.
5. **The defensible disparities are the large cells.** `Black × Female` 0.8632
   [0.806, 0.937], `Black × Male` 0.8643 [0.805, 0.940] — intervals wholly below
   parity. Point-estimate ranking demotes them to sixth and seventh.
6. **Reference-group convention moves every ratio 3–4 points** (most-favoured
   cell vs. designated control) and nothing records which is in force.
7. **Mechanical output checks penalise correct behaviour.** A refusal detector
   flagged 6 correct terse answers (40-character threshold, under a prompt
   demanding terseness) and more than halved a provider's measured score (19.79
   vs. 47.50 excluding its zeroed cycles). A fabrication detector flagged a newer
   model 5× more often, but 14 of 21 flagged values are derivable from the given
   context (88.88 is 0.8888 as a percentage).

## Retracted — reintroducing any of these is a regression

- **"Small samples invert disparities."** The n=4 cell reading 1.1504 was
  *suppressed* by the pipeline's own floor; the audit never ranked it, and that
  number appears in no artifact. Its Wilson interval contains the
  full-population value: no inversion, only an uninformative estimate.
- **"ECOA protects 62+."** ECOA protects age at *any* age for an applicant with
  capacity to contract (15 U.S.C. §1691(a)(1); 12 C.F.R. §1002.2(z)).
  §1002.2(o) defines "elderly" only to scope the §1002.6(b)(2) carve-outs; the
  provision permitting favourable treatment is §1002.6(b)(2)(iv), and
  §1002.6(b)(2)(ii) — the EDDSS rule constraining when a scoring system may use
  age at all — is the one that actually speaks to a model and is uncited.
- **"Grounding cut fabrication 16→2."** Those runs straddle a pipeline revision,
  so it is a legacy contrast, not an ablation.
- **"race DI 0.8956 fails the four-fifths screen."** 0.8956 clears 0.80.
- **UC3 `gender` is synthetic** (`np.random.random(n) > 0.5`): Lending Club
  contains no protected class.

## Known self-inconsistencies you must resolve, not inherit

- The paper screens on the **four-fifths rule (29 C.F.R. §1607.4(D))**, which is
  EEOC *employment* law with no ECOA standing — while its own argument is that
  importing ADEA's 40+ into credit is a category error. Either justify retaining
  it or replace it with a credit-appropriate test. §1607.4(D) also contains the
  small-sample caveat that anticipates finding (4).
- The `0.72` critical boundary is `0.80 × 0.9`, invented here, no provenance.
- **45% of the HMDA file is dropped** (20,000 → 10,978): `Race Not Available`
  (4,140, 20.7%), `Joint`, `Free Form Text Only`. That is selection on the
  protected attribute, undisclosed, in a paper about who silently decides who
  counts as protected.
- **Track H is self-grading** (gpt-4o graded gpt-4o), contradicting the stated
  architecture.
- The German Credit model uses **sex and age as input features** — facially
  disparate *treatment*, per se unlawful in a US setting, currently filed under
  limitations rather than reported as a result.
- Theil index is identical across all attributes (AIF360's is
  attribute-invariant); the paper claims five metrics per attribute where the
  released JSON has three.

## Priority work

1. **Decide the thesis, then write.** The strongest candidate on the evidence:
   *the construct a fairness metric operationalises is not the construct the
   statute names, and the gap is invisible in the output.* Findings 1, 3 and 6
   instantiate it; finding 4 supplies the measurement-validity argument. This is
   a Jacobs & Wallach construct-validity paper with a regulated instance.
2. **Run the subsampling study** (no API, ~a weekend, highest leverage). Treat
   the full-population `race × sex` table as ground truth θ. For audit sizes
   m ∈ {500, 1000, 2196, 4000, 8000}, draw B=2000 samples; apply five policies:
   floor n≥15, floor n≥30, no floor point estimates, no floor with Wilson
   intervals (cell uninformative when the interval spans 0.80), empirical-Bayes
   shrinkage. Record sign-inversion, misidentification, false-clearance,
   false-alarm, and coverage rates; cross with three reference-group
   conventions. Converts the central anecdote into a rate with an interval and
   answers most statistical objections by design.
3. **Split the paper.** The LLM half (Track Q, Track H, cross-model) is 9 items ×
   4 cycles, one model, one-third on a synthetic attribute — a pilot. Three
   reviewers said so independently. The instrument-failure material ("every
   mechanical proxy for judgement is a policy") is a coherent separate short
   paper and is genuinely good.
4. **Check whether the cross-model prompt-sensitivity result survives** removing
   the refusal penalty. It may be entirely driven by cycle-4 (`constrained`)
   flags — a fourth instance of the brevity-detector failure rather than a
   model-generation finding. NOT YET CHECKED.
5. **Bibliography.** Hand-enter Crenshaw 1989, Hardt et al. 2016, Kearns et al.
   2018 — the automated gate rejected them, which is itself an instance of the
   thesis. Add Jacobs & Wallach 2021 (the vocabulary this paper instantiates and
   does not cite), Chen/Johansson/Sontag 2018 (bias–variance in subgroup
   estimates — the direct threat to finding 4), Obermeyer 2019, Passi & Barocas
   2019, Barocas et al. 2021 (disaggregated evaluation), Gillis 2022
   (*The Input Fallacy*), Sclar et al. 2024. Resolve each before adding.
6. **Pin `aif360` and `fairlearn`** in `requirements.txt` and ship the minimal
   repro — the "verified directly" claim is not currently reproducible, and
   imports are guarded with silent fallback.

## The appendix — currently absent, and required

A paper whose contribution is measurement validity cannot ship without one.
Build these as appendices:

- **A. Full cell tables with intervals.** Every `race × sex` and
  `AgeGroup × Sex` cell, both populations, with n, selection rate, DI, and 95%
  CI, under all three reference-group conventions. The paper's argument is that
  partial tables mislead; print the whole thing.
- **B. Statutory provisions in full.** 15 U.S.C. §1691(a)(1); 12 C.F.R.
  §§1002.2(o), 1002.2(z), 1002.6(b)(2)(i), (ii), (iv); 29 C.F.R. §1607.4(D)
  including its small-sample caveat. Quote them so a reader can check the legal
  claims without leaving the paper — the previous draft's legal errors survived
  because nobody could see the text.
- **C. The minimal reproduction** of the AIF360 default inversion: the ~20-line
  script, its output, and library versions.
- **D. Subsampling study design and full results** (once item 2 is done): the
  policy grid, per-policy rates with intervals, and the sensitivity to the
  reference-group convention.
- **E. Population construction.** The attrition table from 20,000 raw rows to
  10,978, with every exclusion rule and its count, and DI recomputed treating
  `Race Not Available` as a category rather than dropping it.
- **F. Prompt packs and scoring.** The four prompt strategies verbatim, the
  severity rule, both detector implementations with their thresholds, and the
  scoring rubric — enough to replay without the repository.
- **G. Artifact map.** Every claim in the paper → the file that regenerates it →
  the command. Two headline numbers were previously quoted with no artifact
  behind them and both were wrong; this appendix is the control that prevents a
  third.

## Environment

- Homebrew `python3.11` only. **Never create repo-local venvs** — they hang under
  iCloud Drive eviction.
- `.env` is at the *primary repo root*; the worktree has a symlink. Never print
  key values.
- `run_pipeline.py`'s consolidate step rebuilds cross-dataset artifacts from only
  the datasets passed — pass all three (`german_credit hmda lending_club`) or
  repair from `git HEAD`.
- Spend: $0.8933 over 98 calls against a $15 cap with pre-call abort and a
  persisted ledger. `gpt-5-mini` rejects `temperature` and requires
  `max_completion_tokens`.
- Branch `claude/relaxed-kepler-768cdc`. PR #3 is open against `main` and does
  **not** contain this work. Nothing is pushed; do not push unless asked.

## Verification gates — all must pass before any commit

```
make test                      # 60 tests
make smoke
python3.11 scripts/verify_bibliography.py --tex report/report_facct.tex report/report.tex
python3.11 scripts/interval_estimates.py
python3.11 scripts/compare_populations.py
make papers                    # confirm overview.pdf is exactly 1 page
```

## Standing constraints

- **Our pipeline is not the status quo.** Claims about standard practice are
  restricted to what is verified directly in AIF360 / Fairlearn. Defects in our
  own harness are evidence about building an audit, not about the field.
- **No project narrative.** Findings are findings, not a chronology of what the
  authors got wrong.
- **No redundant words to manufacture logic.**
- **Every number traces to a regenerable artifact**, and appendix G proves it.
- Report interval, n, and reference group beside every ratio, without exception.
- Do not use `AgentTool` or workflows unless asked.
