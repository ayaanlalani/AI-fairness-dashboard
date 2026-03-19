# Deterministic Model vs OpenAI

This report compares our deterministic fairness audit pipeline against the OpenAI qualitative benchmark on the current lending datasets. The structure below follows the process in chronological order so it is clear what happens first, what each stage produces, and where the two systems differ.

## Executive Summary

Our deterministic model is the stronger **benchmark backbone** because it computes the actual fairness metrics, applies fixed severity rules, and produces reproducible audits at minimal cost. OpenAI performs best as a **second-pass qualitative layer**: it does not own the metrics, but it turns the same metric evidence into clearer and more operational remediation plans.

## Chronological Flow

### Step 1. Start with the dataset

Both systems begin from the same cleaned lending dataset.

**Inputs at this stage**

- cleaned feature table
- ground-truth labels
- protected-attribute columns such as sex, race, age, or foreign-worker status

**What happens**

- the baseline classifier is trained
- the selected model writes a canonical prediction file: `classification_predictions.csv`

**Output of Step 1**

- one shared prediction file containing actual labels, predicted labels, and protected attributes

### Step 2. Our deterministic pipeline computes the fairness audit

This is where our model does the work that OpenAI does not do.

**Inputs at this stage**

- `classification_predictions.csv`
- protected-group definitions
- fairness threshold rules

**What our model computes**

- `DI`
- `DPD`
- `EOD`
- `AOD`
- `Theil`
- per-group selection rate, TPR, FPR, and support counts

**What our model then derives**

- severity labels from fixed thresholds
- imbalance findings from group size and base-rate gaps
- proxy-feature findings from correlation-based checks
- root causes from deterministic rules
- mitigation recommendations from fixed AIF360-style mappings

**Outputs of Step 2**

- `fairness_metrics.csv`
- per-group breakdown tables
- deterministic severity, root-cause, and mitigation results

### Step 3. Semantic Scholar evidence is retrieved

After the deterministic audit identifies the fairness issues, Python gathers supporting literature.

**How the evidence is grabbed**

- endpoint used: `https://api.semanticscholar.org/graph/v1/paper/search`
- fields requested: `title,year,authors,abstract,url,venue,citationCount,externalIds,paperId`
- authentication: if `SEMANTIC_SCHOLAR_API_KEY` exists, it is sent as the `x-api-key` header
- rate limiting: the helper waits about `1.1s` between requests
- retries: `429` responses trigger exponential backoff, up to `3` attempts
- query strategy: the code first builds default dataset-level fairness queries, then adds attribute-specific queries based on detected causes and recommended fixes
- deduplication: papers are deduplicated by `paperId` or title

**Output shape of each paper**

- `paper_id`
- `title`
- `year`
- `venue`
- `citation_count`
- `url`
- `abstract`
- `authors`
- `query`

**Outputs of Step 3**

- `qualitative_research_evidence.json`
- evidence sections appended into the deterministic report

### Step 4. Our deterministic report is assembled

At this point, our model has enough information to produce a full standalone audit.

**Inputs at this stage**

- predictions
- fairness metrics
- per-group breakdowns
- deterministic rule outputs
- Semantic Scholar evidence

**Outputs of Step 4**

- `qualitative_report.md`
- a reproducible fairness audit that already includes what is wrong, why it is wrong, and how to fix it

### Step 5. OpenAI is invoked

Only after Python has already computed the fairness context do we call OpenAI.

**Important difference**

- OpenAI does **not** compute the fairness metrics
- OpenAI receives the metrics and evidence produced by Python

**Inputs sent to OpenAI**

- Python-computed fairness metrics
- per-group breakdowns
- deterministic severity context
- reference audit specification
- dataset metadata
- Semantic Scholar evidence packets

**What OpenAI produces**

- qualitative audit response
- severity labels per attribute
- root-cause narrative
- mitigation plan
- research-backed justification
- self-refinement cycle outputs

**Outputs of Step 5**

- `llm_raw_response.json`
- `llm_fairness_report.md`
- `benchmark_comparison.md`
- cost and latency tracking files such as `llm_napkin_math.json`

### Step 6. Final comparison is made

The final benchmark compares OpenAI's qualitative audit back against our deterministic baseline.

**What is compared**

- severity agreement
- qualitative completeness
- cause alignment
- mitigation specificity
- research grounding

**Purpose of this final step**

- determine whether OpenAI matches our baseline audit
- identify where OpenAI is stronger or weaker
- decide whether OpenAI can replace or only complement our model

## Side-by-Side Role Split

| Stage | Our deterministic model | OpenAI |
|---|---|---|
| Training and predictions | Owns the baseline pipeline | Not involved |
| Fairness metrics | Computes them deterministically | Does not compute them |
| Per-group breakdowns | Computes them deterministically | Consumes them |
| Severity | Fixed threshold rules | Inferred from provided context |
| Root-cause analysis | Rule-based mapping | Free-form reasoning |
| Mitigations | Fixed intervention mapping | More operational action plans |
| Research use | Retrieves and stores evidence deterministically | Uses provided evidence in qualitative output |
| Cost profile | Minimal local compute | API cost and latency |
| Reproducibility | High | Lower than deterministic baseline |

## Benchmark Results

| Dataset | OpenAI Final Score | Severity Agreement vs Our Model | OpenAI Cost | OpenAI API Latency |
|---|---:|---:|---:|---:|
| German Credit | 85.0/100 | 3/3 | $0.1259 | 75.4s |
| HMDA Mortgage Lending (Georgia) | 90.0/100 | 3/3 | $0.1070 | 62.6s |

## Interpretation of Results

### Where our deterministic model is better

- it is the quantitative source of truth
- it always follows the same audit logic
- it is cheaper and more reproducible
- it exposes fixed, inspectable links between observed metrics and selected mitigations

### Where OpenAI is better

- it writes more natural remediation language
- it prioritizes next steps more clearly
- it turns the same evidence into stronger policy and implementation framing

### Overall judgment

For the overall benchmark backbone, **our deterministic model is better**.

For qualitative remediation phrasing, **OpenAI is better**.

## Best Combined Use

The best workflow, in chronological order, is:

1. Use our deterministic model as the system of record for fairness metrics, severity, and baseline remediation logic.
2. Use OpenAI as a second-pass qualitative planner that rewrites those findings into clearer, more actionable mitigation guidance.

## Bottom Line

OpenAI is competitive as a qualitative auditor, but it is not a replacement for our deterministic model. The strongest setup is to keep our deterministic pipeline as the audit engine and use OpenAI to improve explanation quality, prioritisation, and remediation framing.

## Appendix

### Appendix A. Our Deterministic Model Outputs

The deterministic pipeline produces the baseline audit artifacts that act as the system of record.

| Output | Purpose |
|---|---|
| `classification_predictions.csv` | Canonical predictions used for downstream fairness analysis |
| `fairness_metrics.csv` | Deterministic fairness metrics by protected attribute |
| `fairness_summary.json` | Compact machine-readable summary of fairness status |
| `report.md` | Quantitative fairness summary report |
| `qualitative_report.md` | Deterministic narrative audit with severity, causes, and mitigations |
| `qualitative_research_evidence.json` | Semantic Scholar evidence packets attached to the deterministic audit |
| `plots/*.png` | Visual summaries such as disparate impact and group selection-rate plots |

**Main deterministic outputs used in this comparison**

- `artifacts/german_credit/fairness/qualitative_report.md`
- `artifacts/german_credit/fairness/qualitative_research_evidence.json`
- `artifacts/hmda/fairness/qualitative_report.md`
- `artifacts/hmda/fairness/qualitative_research_evidence.json`

### Appendix B. OpenAI Benchmark Outputs

The OpenAI benchmark consumes Python-computed fairness context and produces qualitative benchmark artifacts.

| Output | Purpose |
|---|---|
| `llm_context_payload.json` | Metric context, per-group breakdowns, and reference audit specification sent into the benchmark |
| `llm_baseline_payload.json` | Deterministic baseline used to score OpenAI's qualitative output |
| `llm_prompt.txt` | Full benchmark prompt assembled for OpenAI |
| `semantic_scholar_context.json` | Research context passed to the OpenAI run |
| `llm_raw_response.json` | Full raw structured response including refinement cycles |
| `llm_fairness_report.md` | Human-readable OpenAI qualitative audit |
| `benchmark_comparison.md` | Side-by-side OpenAI vs deterministic comparison for the dataset |
| `llm_napkin_math.json` | Token, latency, and cost tracking |

**Main OpenAI outputs used in this comparison**

- `artifacts/german_credit/fairness/openai/llm_fairness_report.md`
- `artifacts/german_credit/fairness/openai/benchmark_comparison.md`
- `artifacts/hmda/fairness/openai/llm_fairness_report.md`
- `artifacts/hmda/fairness/openai/benchmark_comparison.md`

### Appendix C. Consolidated Comparison Outputs

These are the cross-dataset artifacts used to summarize the final benchmark results.

| Output | Purpose |
|---|---|
| `artifacts/consolidated/openai_vs_model_report.md` | Cross-dataset short write-up for OpenAI vs deterministic model |
| `artifacts/consolidated/llm_benchmark_comparison_openai.md` | Consolidated OpenAI benchmark comparison |
| `artifacts/visualizations/openai_cycle_scores.png` | OpenAI self-refinement score progression |
| `artifacts/visualizations/openai_final_subscores.png` | Final OpenAI subscore breakdown |
| `artifacts/visualizations/openai_severity_agreement.png` | Severity agreement with deterministic baseline |
| `artifacts/visualizations/openai_cost_latency.png` | Cost and API latency summary |

### Appendix D. Mitigations Provided by Both Systems

This appendix compares the actual mitigation recommendations produced by our deterministic pipeline and the OpenAI benchmark.

#### German Credit

| Attribute | Our deterministic model | OpenAI |
|---|---|---|
| `Sex_original` | Disparate Impact Remover; Prejudice Remover; Equalized Odds post-processing | Reweigh female-positive loans; causal feature pruning; Exponentiated Gradient with demographic parity constraint; calibrated equal-odds hot-fix; CI guardrail on DI |
| `AgeGroup_original` | Reweighing; Disparate Impact Remover; Prejudice Remover; Equalized Odds post-processing | Stratified bootstrap / importance weighting; age-normalized feature ratios; PrejudiceRemover tuning; Invariant Risk Minimisation; age-specific AUC monitoring |
| `foreign_worker_original` | Reweighing; Disparate Impact Remover; more minority-group data; class-weighted training; Prejudice Remover; Equalized Odds post-processing | Collect more domestic-worker records; SMOTE-NC if data is scarce; bootstrap confidence intervals; group DRO; minimum-support manual-review trigger |

#### HMDA Mortgage Lending (Georgia)

| Attribute | Our deterministic model | OpenAI |
|---|---|---|
| `race` | Reweighing; collect more underrepresented-group data; class-weighted training | Calibrated Equalized Odds for Black and Asian groups; Fair SMOTE-NC; ExponentiatedGradient with equal-opportunity constraint; causal feature filtering; race-segmented monitoring dashboard |
| `sex` | Calibrated Equalized Odds post-processing | Feature-attribution audit for sex proxies; multi-attribute ExponentiatedGradient on race and sex; rolling DI operational alert |
| `age_group` | Reweighing | Keep age on dashboard; recession stress testing for senior FPR; document age-policy rationale with ADEA support |

#### Mitigation Pattern Summary

| Pattern | Our deterministic model | OpenAI |
|---|---|---|
| Style | Standardized fairness interventions from fixed rules | More phased, operational, and policy-style action plans |
| Typical first move | Apply known fairness pre/in/post-processing methods | Combine immediate controls, medium-term retraining, and long-term governance |
| Data recommendations | Reweighing, more samples, class weighting | Resampling plans, targeted data acquisition, support thresholds |
| Model recommendations | Disparate Impact Remover, Prejudice Remover, EqOdds, Calibrated EqOdds | ExponentiatedGradient, causal pruning, DRO, IRM, monitoring constraints |
| Governance recommendations | Limited, mostly implicit in the mitigation type | Explicit dashboards, CI alerts, freeze triggers, documentation, and compliance framing |
