# Results — full-population re-analysis, leakage repair, and cross-model benchmark

Every number below is regenerated from the current pipeline. Where a figure
replaces a previously published one, both are shown. Nothing here is carried
over from the earlier run.

**Headline:** three of this project's previously reported findings were
substantially artifacts of the measuring apparatus rather than properties of the
data. Correcting them did not weaken the project's thesis — it produced four
cleaner instances of it.

---

## 1. What changed in the pipeline

| Change | Before | After |
|---|---|---|
| HMDA feature set | included `interest_rate` | leaked column removed |
| HMDA audited population | 2,196 (test split) | **10,978** (full cleaned population, out-of-fold) |
| German Credit audited population | 200 (test split) | **1,000** (full population, out-of-fold) |
| HMDA age attribute | `age_group` only (banded) | `age_group` **+ `age_62_plus`** (the ECOA cut) |
| Prediction method | fit on train, score test split | stratified 5-fold **out-of-fold**: every row scored by a model that never saw it |

### 1.1 The HMDA leak

`interest_rate` is populated only where a loan was originated. Missingness by
outcome, measured on `hmda_raw.csv`:

| Outcome | n | `interest_rate` missing |
|---|---|---|
| Approved (`action_taken=1`) | 14,774 | **0.0%** |
| Denied (`action_taken=3`) | 5,226 | **99.3%** |

Median-imputing that and handing it to a classifier leaks the target. Removing it:

| HMDA random forest | With leak | Without leak |
|---|---|---|
| Accuracy | 0.9672 | 0.8024 |
| **AUC** | **0.9942** | **0.7915** |

AUC 0.9942 is not a plausible mortgage approval model. The corrected
out-of-fold AUC over the full population is **0.8018**.

Consequence for the previously published claim: the near-zero EOD/AOD that
supported *"the model is not making differentially worse errors by race; it is
reproducing a base-rate difference already in the labels"* were a property of a
model that made almost no errors in any group. That inference is withdrawn.

### 1.2 A second, independent defect: German Credit AUC was inverted

The target is encoded `1 = good, 2 = bad`, so `roc_auc_score` treats 2 as the
positive class. The code scored `predict_proba[:, 0]` — the probability of
*good* — against it.

| German Credit | Published | Corrected |
|---|---|---|
| Logistic regression AUC | 0.2756 | **0.7244** |
| Random forest AUC | 0.2013 | **0.7987** |

Both published values were below chance and each is exactly `1 −` the true
value. Not caught previously because AUC was never read.

At full population the model also clears the baseline it had appeared to fail:
out-of-fold accuracy **0.743** against a majority-class rate of **0.700**.

---

## 2. Deterministic results: before and after

### 2.1 Marginal disparate impact

| Dataset | Attribute | Published (test split) | Corrected (full population) |
|---|---|---|---|
| German Credit | `Sex` | 0.8595 | 0.8888 |
| German Credit | `AgeGroup` | 0.8212 | 0.9258 |
| German Credit | `foreign_worker` | 0.9402 | **0.8387** |
| HMDA | `race` | 0.8956 | 0.9371 |
| HMDA | `sex` | 0.9291 | 0.9572 |
| HMDA | `age_group` (banded) | 1.0138 | 0.9791 |
| HMDA | **`age_62_plus`** (ECOA) | not measured | **0.8897** |

Note also a correction of arithmetic, independent of any re-run: the previously
published claim that HMDA `race` DI 0.8956 *"fails the four-fifths screen"* is
false. 0.8956 clears 0.80, the pipeline's own `BiasFlag` is `false`, and
`classify_severity` returns `MODERATE`. The error originated in `FINDINGS.md`
and propagated to `report.tex`.

### 2.2 The ECOA age finding

The banded `age_group` attribute pools young and senior applicants against the
middle band. They move in opposite directions, so they cancel:

| HMDA `age_group` cell | n | selection rate |
|---|---|---|
| young | 2,304 | higher |
| mid (reference) | 5,507 | — |
| senior | 3,167 | lower |

Pooled result: **DI 0.9791, severity LOW.** Audited on the cut the statute
actually specifies — Regulation B §1002.2(o) defines "elderly" as **62 or
older**, not 40+, which is the ADEA employment threshold — the same model gives
**DI 0.8897**, the largest age disparity in the dataset and the lowest of all
four HMDA attributes.

The `applicant_age_above_62` flag was present in the raw HMDA file and unused.
The public LAR age bands straddle 62 (`55-64`), so the banded attribute *cannot*
isolate the protected class; HMDA supplies the flag for exactly this reason.

A limitation this exposes rather than solves: Reg B §1002.6(b)(2) expressly
permits *favouring* an elderly applicant, so protection is asymmetric while
disparate impact is symmetric. A ratio above 1 here is lawful and is not a
reverse-disparity finding. The metric cannot express the rule it is being used
to check.

---

## 3. Small samples invert disparities

This is the strongest result in the re-analysis. Both rows use the **same
leak-free model**; only the audited population differs, so the comparison
isolates sample size.

### HMDA `race × sex`, floor n ≥ 15

| Cell | Test split (n=2,196) | Full population (n=10,978) |
|---|---|---|
| Cells suppressed at n<15 | **5 of 11** | 1 of 12 |
| Worst cell the audit **reports** | `Black × Male` 0.8169 (MODERATE) | `American Indian × Male` **0.6734** (CRITICAL) |
| `American Indian × Male` | **1.1504** — *best cell in the table* (n=4) | **0.6734** — *worst cell in the data* (n=31) |
| `American Indian × Female` | 0.1917 (n=6) — noise | 0.8815 (n=25) |

**The failure is inversion, not suppression.** On the split, `American Indian ×
Male` reads 1.1504 — above parity, the best-treated cell — because four
applicants were sampled and all four were approved. At full population that group
is the worst in the dataset.

**The floor was right about what it suppressed.** `American Indian × Female` at
0.1917 (one approval in six) looks like the most severe disparity in the study
and is noise: the full-population value is 0.8815. A reader who overrode the
floor to surface that number would have reported a catastrophe that does not
exist.

So the floor is not the error, and removing it is not the fix. At this sample
size the estimates for small groups are unreliable **in both directions** — one
group looked catastrophic, another looked flawless, and both were wrong. Every
affected cell belonged to a racial minority, which is mechanism rather than
coincidence: a floor on group size is a floor on group population.

Full-population cell table:

| race | sex | n | selection rate | DI vs best |
|---|---|---|---|---|
| American Indian | Male | 31 | 0.5806 | **0.6734** |
| Multiracial | Female | 18 | 0.6111 | 0.7088 |
| Multiracial | Male | 29 | 0.6552 | 0.7599 |
| Pacific Islander | Female | 12 | 0.6667 | 0.7732 *(suppressed)* |
| Pacific Islander | Male | 15 | 0.7333 | 0.8505 |
| Black | Female | 1,994 | 0.7442 | 0.8632 |
| Black | Male | 1,707 | 0.7452 | 0.8643 |
| American Indian | Female | 25 | 0.7600 | 0.8815 |
| White | Female | 2,318 | 0.7865 | 0.9121 |
| White | Male | 4,019 | 0.8281 | 0.9604 |
| Asian | Male | 556 | 0.8525 | 0.9888 |
| Asian | Female | 254 | 0.8622 | 1.0000 |

### German Credit `AgeGroup × Sex`

The previously published flagship number does not survive.

| | Published (test split) | Corrected (full population) |
|---|---|---|
| `under_40 × female` DI | **0.6087 (CRITICAL)** | **0.8201 (MODERATE)** |
| Its n | 46 | 241 |
| Comparator | `40_plus × female`, **n=15, rate 1.0000** | `40_plus × female`, n=69, rate 0.8551 |

The critical grade came from a comparator of fifteen people all of whom were
approved. At full population the disparity is real and in the same direction —
`under_40 × female` remains the worst cell — but it **clears the four-fifths
screen**. The published claim that it sat "below the 0.72 CRITICAL threshold" is
withdrawn.

Two further corrections to the published account of this cell: the text
attributed `n=46` to the *privileged* group (46 is the disadvantaged group), and
`under_40 × male` at 0.7952 also failed the screen on the test split, so the
marginal audit was not blind to the age effect as claimed.

---

## 4. Cross-model benchmark: gpt-4o vs gpt-5-mini

Both models were run on the **same corrected packs**, so this is like-for-like.
`gpt-5-mini` was confirmed against the live model list (`gpt-5-mini-2025-08-07`).

| Model | Dataset | n | Cycle means (zero-shot / CoT / self-critique / constrained) | **Spread** | Refusal flags | Fabrication flags |
|---|---|---|---|---|---|---|
| gpt-5-mini | german_credit | 12 | 85.0 / 85.0 / 85.0 / 85.0 | **0.0** | 1 | 10 |
| gpt-5-mini | hmda | 16 | 91.9 / 90.0 / 90.0 / 90.0 | **1.9** | 1 | 7 |
| gpt-4o | german_credit | 12 | 71.7 / 71.7 / 78.3 / 63.7 | **14.7** | 3 | 2 |
| gpt-4o | hmda | 16 | 83.8 / 85.0 / 85.0 / 76.0 | **9.0** | 4 | 2 |

**Prompt sensitivity is model-generation-specific.** The project's headline
reliability claim — that prompt phrasing moves the verdict — reproduces on
gpt-4o (spread 9.0–14.7) and very nearly vanishes on gpt-5-mini (0.0–1.9) on
identical input. The claim should be qualified accordingly rather than stated as
a general property of LLM auditing.

**API contract, verified not assumed:** `gpt-5-mini` rejects `temperature`
("Only the default (1) value is supported") and requires
`max_completion_tokens`. A temperature sweep is therefore impossible on this
model, and any repeated-sampling study must either use gpt-4o or measure
variance at the fixed default.

### 4.1 The fabrication detector penalises arithmetic

gpt-5-mini trips five times as many fabrication flags. Checking each flagged
value against the metric context:

**14 of 21 flagged values (67%) are derivable from the provided context.**

| Flagged value | What it actually is |
|---|---|
| `88.88` | `0.8888` (Sex DI) as a percentage |
| `83.87` | `0.8387` (foreign_worker DI) as a percentage |
| `9.2`, `6.5`, `15.3`, `3.45`, `1.28`, `6.25` | gaps between two context values, in percentage points |
| `0.0681` | difference of two context values |

The detector matches raw float literals. A model that converts a ratio to a
percentage, or reports a gap in points, is recorded as fabricating a metric.
gpt-5-mini does more of this arithmetic than gpt-4o, so **the better-reasoning
model scores worse on the fabrication measure.** Seven values (`19.9` ×3,
`5.3` ×4) were not derivable by this test and remain candidates for genuine
fabrication; their repetition across cycles suggests a systematic derivation
rather than random invention.

This is the same failure mode as the refusal detector, found independently in a
second instrument: a mechanical proxy scoring a correct behaviour as a fault.

---

## 5. Spend

| Item | Calls | Cost |
|---|---|---|
| Prior work (Track Q + Track H + rubric judge) | 42 | $0.5969 |
| gpt-5-mini, corrected packs | 28 | $0.1236 |
| gpt-4o control, same packs | 28 | $0.1729 |
| **Ledger total** | **98** | **$0.8933** |
| Cap | | $15.00 |

Reported honestly: calls and dollars are distinct. The previously published
"$0.7094 across 42 calls" conflated them — the ledger's 42 calls total $0.5969,
and the remaining $0.1125 came from a pre-freeze run whose calls are not in the
ledger. o3 and gpt-5-mini output prices are held deliberately high, so their
costs are upper bounds rather than billed amounts.

---

## 6. What this means for the paper's claims

| Claim | Status |
|---|---|
| Metric orientation inverted the headline number (DI 0.0 vs 0.9931) | **Holds.** Independent of population size. Magnitude remains a property of the degenerate classifier. |
| Refusal detector measures brevity (6 flags, 0 refusals) | **Holds.** |
| Refusal detector produced a governance decision (Gemini blocked) | **Holds.** Excluding detector-zeroed cycles moves the mean from 19.79 to 47.50. |
| Small samples invert disparities | **Corrected and strengthened.** The earlier framing (floor hides the worst cell) was itself a small-sample artifact; the real result is that a group ranked best-treated at n=4 is worst-treated at n=31. |
| ECOA age asymmetry and pooling | **Newly measured.** `age_62_plus` 0.8897 vs pooled `age_group` 0.9791. |
| Fabrication detector penalises arithmetic | **New.** 67% of flags derivable from context. |
| `under_40 × female` is CRITICAL at 0.6087 | **Withdrawn.** 0.8201 at full population; clears the screen. |
| HMDA `race` "fails the four-fifths screen" | **Withdrawn.** Arithmetically false. |
| Label bias not model error, from EOD ≈ 0 | **Withdrawn.** Derived from a leaked model. |
| Prompt phrasing moves the verdict | **Qualified.** Reproduces on gpt-4o, nearly absent on gpt-5-mini. |

Still outstanding and unaffected by this re-analysis: UC3's `gender` attribute is
synthetic (`np.random.random(n) > 0.5`), so Lending Club contains no protected
class and the protected-class-versus-economic-proxy contrast does not exist.
That disclosure belongs in the paper regardless of population size.
