# Problem Statement

**Project:** Fairness Audit Benchmark for Lending Datasets

---

## Statement

We frame this artifact as **an evaluation protocol and benchmark for testing whether LLM-based fairness auditors produce stable, statistically grounded, remediation-ready audits on lending datasets**.

The primary scope is German Credit and HMDA Georgia. The core comparison is between a deterministic Python fairness audit pipeline and an OpenAI LLM benchmark that interprets Python-computed metric context. Gemini remains only as a secondary legacy comparison where existing artifacts already contain it.

Concretely, the artifact addresses:

1. **Reference audit specification**
   Define the set of **analysis metrics** and **response elements** (root causes, severity, concrete mitigation recommendations) that a fairness audit must provide so that remediations can be carried out. This specification is the target for "remediation-ready" analysis: what to measure, how to interpret it, and what to recommend.

2. **Deterministic bias evaluation**
   Assess lending model predictions reproducibly against that specification: multiple dimensions (Disparate Impact, Demographic Parity Difference, Equal Opportunity Difference, Average Odds Difference, Theil Index), multiple protected attributes, fixed thresholds, and documented outputs.

3. **From evaluation to recommendations**  
   How to move from raw metrics to interpretable root causes and to concrete, actionable recommendations so that biases can be addressed before or in future deployments.

4. **OpenAI LLM benchmarking**
   Test whether an OpenAI fairness auditor, given deterministic metric context rather than permission to invent or recompute metrics, produces stable severity labels, statistically cautious narratives, and actionable remediation guidance. Legacy Gemini results are useful only as a secondary comparison, not as the main benchmark claim.

Lending is the application domain because credit and mortgage decisions have direct material consequences and established fair-lending regulatory context.

---

## Scope

- **In scope:** German Credit and HMDA Georgia data documentation; deterministic model/fairness outputs; OpenAI qualitative benchmark over fixed metric evidence; artifact README, data cards, citation metadata, Docker/Make/CI reproducibility scaffolding; limitations around subgroup size, LLM nondeterminism, API versioning, and metric grounding.
- **Out of scope:** Deploying lending models for real decisions; claiming legal compliance; implementing mitigation end-to-end; making causal claims that exceed the observed data; using LLM output as the source of record for metric computation.

The intended artifact contribution is an evaluation protocol and benchmark, not a new fairness metric or a production lending decision system.
