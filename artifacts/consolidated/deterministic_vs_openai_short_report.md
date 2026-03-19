# Deterministic Model vs OpenAI

This report compares our deterministic fairness audit pipeline against the OpenAI qualitative benchmark on the current lending datasets.

## What Each System Does

| System | What it does exactly |
|---|---|
| Our deterministic model | Trains the baseline classifier, saves canonical predictions, computes fairness metrics deterministically in Python (`DI`, `DPD`, `EOD`, `AOD`, `Theil`), builds per-group breakdowns, detects imbalance and proxy features, maps metric patterns to fixed root-cause labels and AIF360-style mitigations, and attaches Semantic Scholar evidence to the report. |
| OpenAI benchmark | Does **not** compute fairness metrics. It receives the Python-computed metric context, per-group breakdowns, and Semantic Scholar context, then produces a qualitative audit: reference audit specification, severity labels, root-cause narrative, mitigation plan, and research-backed justification over fixed self-refinement cycles. |

## Main Difference

Our model is a deterministic audit engine. OpenAI is a qualitative reasoning layer on top of deterministic metric computation.

In practice:

- Our model is responsible for the numbers, thresholds, reproducibility, and rule-based remediation logic.
- OpenAI is responsible for interpretation, prioritisation, and turning the fairness findings into more natural remediation language.

## Exact Comparison

| Category | Our deterministic model | OpenAI |
|---|---|---|
| Metric computation | Deterministic Python/AIF360 | None |
| Per-group breakdowns | Deterministic | Consumes provided breakdowns |
| Severity classification | Fixed threshold rules from DI | Inferred from provided context |
| Root-cause analysis | Rule-based mapping from metrics, imbalance, and proxies | Free-form qualitative reasoning |
| Mitigation recommendations | Fixed AIF360-style intervention mapping | More operational, policy-oriented action plans |
| Research support | Semantic Scholar evidence attached deterministically | Uses provided research context in output |
| Reproducibility | High | Lower than deterministic baseline |
| Cost | Minimal local compute | API cost and latency |

## Benchmark Results

| Dataset | OpenAI Final Score | Severity Agreement vs Our Model | OpenAI Cost | OpenAI API Latency |
|---|---:|---:|---:|---:|
| German Credit | 85.0/100 | 3/3 | $0.1259 | 75.4s |
| HMDA Mortgage Lending (Georgia) | 90.0/100 | 3/3 | $0.1070 | 62.6s |

## What OpenAI Did Better

- Produced stronger mitigation wording for implementation and governance.
- Wrote more action-oriented remediation plans.
- Turned the same fairness evidence into clearer policy-style recommendations.

## What Our Model Did Better

- Remained the quantitative source of truth.
- Applied the same rules consistently across datasets and attributes.
- Kept the audit structure stable and reproducible.
- Linked observed metric patterns to fixed, inspectable causes and mitigation families.

## Which Is Better?

For the overall benchmark backbone, **our deterministic model is better**.

Reason:

- it is reproducible,
- it owns the actual fairness measurements,
- it is cheaper,
- and it gives a stable reference audit for comparing LLM outputs.

For qualitative remediation phrasing, **OpenAI is better**.

Reason:

- it is better at writing operational next steps,
- it adds stronger prioritisation language,
- and it reads more like an analyst or policy memo.

## Best Combined Use

The best workflow is:

1. Use our deterministic model as the system of record for fairness metrics, severity, and baseline remediation logic.
2. Use OpenAI as a second-pass qualitative planner that rewrites those findings into clearer, more actionable mitigation guidance.

## Bottom Line

OpenAI is competitive as a qualitative auditor, but it is not a replacement for our deterministic model. It performs best as a layer on top of our metric pipeline, not instead of it.
