# Interim Problem Statement

**Project:** AI Fairness Dashboard (CMPT 415)

---

## Statement

We focus on **evaluating biases in datasets**, and over time moving towards the data used by **AI models**, in order to **provide recommendations on how to fix those biases so that future systems do not exhibit them**.

Concretely, we address:

1. **Reference audit specification**  
   Define the set of **analysis metrics** and **response elements** (root causes, severity, concrete mitigation recommendations) that a fairness audit must provide so that remediations can be carried out. This specification is the target for "remediation-ready" analysis: what to measure, how to interpret it, and what to recommend.

2. **Bias evaluation in data**  
   How to assess datasets in a reproducible way against that specification: multiple dimensions (e.g., Disparate Impact, Demographic Parity, Equal Opportunity), multiple protected attributes, and clear severity and interpretation.

3. **From evaluation to recommendations**  
   How to move from raw metrics to interpretable root causes and to concrete, actionable recommendations so that biases can be addressed before or in future deployments.

4. **Multi-LLM benchmarking**  
   How our deterministic pipeline and various LLM-based auditors compare on the same data (metrics, severity, recommendations), so we can see where our approach stands in the general landscape and when to prefer or combine rule-based vs. LLM-based analysis.

Lending (e.g., credit, mortgage) is the current application domain; the problem framing applies to any domain where biased data or models can lead to unfair outcomes.

---

## Scope (interim)

- **In scope:** Defining a reference audit specification (metrics + response elements for remediation); evaluating biases in specific datasets and model outputs; deterministic qualitative analysis; benchmarking against one or more LLMs (Gemini initially; additional models planned) to position our approach in the broader landscape.
- **Out of scope (for now):** Implementing mitigations end-to-end, intersectional fairness, longitudinal monitoring.

This statement may be refined as the project progresses.
