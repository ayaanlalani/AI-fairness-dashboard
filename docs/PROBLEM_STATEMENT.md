# Interim Problem Statement

**Project:** AI Fairness Dashboard (CMPT 415)

---

## Statement

We focus on **evaluating biases in datasets**, and over time moving towards the data used by **AI models**, in order to **provide recommendations on how to fix those biases so that future systems do not exhibit them**.

Concretely, we address:

1. **Bias evaluation in data**  
   How to assess datasets in a reproducible way for fairness-related harms: multiple dimensions (e.g., Disparate Impact, Demographic Parity, Equal Opportunity), multiple protected attributes (e.g., sex, age, race), and clear severity/interpretation.

2. **From evaluation to recommendations**  
   How to move from raw metrics to interpretable root causes (e.g., data imbalance, proxy features, historical label bias) and to concrete, actionable recommendations (e.g., preprocessing, in-processing, or post-processing mitigations) so that biases can be addressed before or in future deployments.

3. **Comparison of evaluation approaches**  
   How rule-based bias evaluation compares to LLM-based evaluation on the same data, so that practitioners can choose or combine approaches based on cost, reliability, and automation.

Lending (e.g., credit, mortgage) is the current application domain; the problem framing applies to any domain where biased data or models can lead to unfair outcomes.

---

## Scope (interim)

- **In scope:** Evaluating biases in specific datasets and in model outputs (predictions) derived from them; group fairness metrics; deterministic qualitative analysis and mitigation recommendations; benchmarking against one LLM (Gemini) for comparison.
- **Out of scope (for now):** Implementing mitigations end-to-end, intersectional fairness, longitudinal monitoring.

This statement may be refined as the project progresses.
