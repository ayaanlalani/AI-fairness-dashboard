# Benchmark Comparison: HMDA Mortgage Lending (Georgia)

Deterministic pipeline vs. Gemini qualitative benchmark over fixed self-refinement cycles.

## Napkin Math

| Metric | Value |
|---|---|
| Model | `gemini-2.5-flash` |
| Wall-clock time | 44.52s |
| API attempts | 1 |
| Input tokens | 5,781 |
| Output tokens | 3,392 |
| Total tokens | 9,173 |
| Input cost | $0.0006 (@ $0.1/1M tokens) |
| Output cost | $0.0014 (@ $0.4/1M tokens) |
| **Total est. cost** | **$0.0019** |

## Preflight Estimate

| Metric | Value |
|---|---|
| Projected input tokens | 2,534 |
| Assumed max output tokens | 5,000 |
| Projected total cost ceiling | $0.0023 |

## Cycle Improvement

| Cycle | Total Score | Summary |
|---|---:|---|
| 1 | 78.3 | Total score: 78.3/100. Completeness=40.0, Severity agreement=13.3, Cause alignment=0.0, Mitigation specificity=10.0, Research grounding=15.0. |
| 2 | 78.3 | Total score: 78.3/100. Completeness=40.0, Severity agreement=13.3, Cause alignment=0.0, Mitigation specificity=10.0, Research grounding=15.0. |
| 3 | 78.3 | Total score: 78.3/100. Completeness=40.0, Severity agreement=13.3, Cause alignment=0.0, Mitigation specificity=10.0, Research grounding=15.0. |

## Severity Comparison

| Attribute | Deterministic Baseline | Gemini | Agreement? |
|---|---|---|---|
| race | MODERATE (borderline) | CRITICAL | No |
| sex | MODERATE (borderline) | MODERATE | Yes |
| age_group | LOW (fair) | LOW | Yes |

## Qualitative Narrative Comparison

### race

**Deterministic root causes:** Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.; Representation bias: severely underrepresented groups make the model less reliable for those populations and can amplify existing disparities.

**Gemini why it is wrong:** The moderate Disparate Impact for the 'Black' group causally implies that Black applicants receive favorable mortgage outcomes at a measurably lower rate than White applicants. This directly limits access to financial capital, exacerbating existing socioeconomic disparities. More critically, the pronounced base-rate disparity in the historical data for several racial groups signifies that the ground truth labels themselves are a product of systemic inequities. When the model is trained on such biased historical data, it **causally learns and replicates these disparities**, leading to algorithmic outcomes that reflect past discrimination rather than objective merit [1, 2]. The severe data sparsity for 'American Indian', 'Multiracial', and 'Pacific Islander' groups makes it impossible to reliably assess or mitigate bias for these highly vulnerable populations, undermining the core objective of equitable AI deployment [3].

**Deterministic mitigations:** Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.; Data collection: gather more samples from underrepresented groups. In the interim, use class-weighted training (class_weight='balanced' in scikit-learn) to upweight minority-group errors.

**Gemini how to fix it:** Addressing the critical fairness issues for the 'race' attribute demands a comprehensive strategy:
1.  **For Severe Data Sparsity:** Given the extremely low counts for 'American Indian', 'Multiracial', and 'Pacific Islander' groups, **targeted data collection** should be the primary and most robust approach to acquire reliable, representative data. If immediate data collection is infeasible, **advanced synthetic data generation methods** (e.g., CTGAN or SMOTE for tabular data) can be explored, but with extreme caution and rigorous validation to avoid introducing further biases or misrepresenting rare groups. These techniques aim to increase statistical power for meaningful fairness metric computation.
2.  **For Disparate Impact & Base-Rate Disparity:** To directly mitigate the Disparate Impact and address the underlying base-rate disparities, a combination of algorithmic and systemic interventions is required.
    *   **Algorithmic Mitigation:** Employ **in-processing fairness-aware algorithms** such as Fairlearn's `ExponentiatedGradient` or `GridSearchCV` during model training, explicitly optimizing for metrics like `DemographicParityDifference` to equalize selection rates across racial groups. Alternatively, **pre-processing techniques** like AIF360's `Reweighing` can adjust training sample weights to promote parity, ensuring the model learns a more equitable distribution from the outset.
    *   **Causal & Systemic Intervention:** Conduct a **rigorous causality-focused review of historical lending policies and practices** to identify the true root causes of the observed base-rate disparities. This involves applying causal inference methods to pinpoint features that may act as proxies for race or reflect historical discrimination. Concurrently, **engage with affected communities** and domain experts to collaboratively define equitable lending outcomes and refine policy, ensuring that the 'favorable' label itself reflects fairness rather than historical bias [3].
3.  **Continuous Monitoring:** Implement continuous monitoring of all fairness metrics, especially Disparate Impact and base rates, with alerts for any significant deviations, alongside regular audits [1, 2].

### sex

**Deterministic root causes:** N/A

**Gemini why it is wrong:** The moderate Disparate Impact for the 'Female' group causally implies that women are proportionally less likely to receive a favorable mortgage outcome compared to men. This disparity, despite similar predictive accuracy (low EOD/AOD), directly translates to **unequal access** to a crucial financial service. The causal mechanism is that the model, by learning from historical data that may reflect pre-existing gender-based economic inequalities (e.g., income disparities, wealth gaps), perpetuates these outcome disparities. Therefore, even if the model is not exhibiting disparate *treatment* in terms of misclassification rates, it is *causally contributing to disparate outcomes* for female applicants, potentially exacerbating long-term socioeconomic inequalities [2]. This highlights a critical distinction between predictive fairness and outcome fairness, where the model's objective impact is unequal.

**Deterministic mitigations:** Post-processing -- Calibrated Equalized Odds (aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing): a minimal intervention that adjusts thresholds to bring DI above the 0.8 threshold while preserving calibration.

**Gemini how to fix it:** To mitigate the moderate Disparate Impact for the 'sex' attribute, the following specific interventions are recommended:
1.  **Targeted Post-processing:** Given the near-zero Equal Opportunity Difference and Average Odds Difference, **post-processing techniques** are highly effective and efficient as they can directly address `DemographicParityDifference` without compromising the model's equitable predictive accuracy. Specifically, Fairlearn's `ThresholdOptimizer` can be used to identify a distinct classification threshold for the 'Female' group, recalibrating outcomes to equalize selection rates (i.e., achieve demographic parity) with the 'Male' group.
2.  **Causal Proxy Investigation:** Conduct a **causal analysis** to identify and understand any features strongly correlated with 'sex' that may be acting as proxies, implicitly driving the observed Disparate Impact. This involves explainable AI (XAI) techniques and domain expert consultation to uncover and potentially neutralize these indirect sources of bias.
3.  **Algorithmic Refinement (Optional):** If model retraining is feasible, **in-processing methods** like Fairlearn's `ExponentiatedGradient` can be utilized, explicitly optimizing the model to satisfy `DemographicParityDifference` during training. Alternatively, **pre-processing techniques** like AIF360's `Reweighing` can adjust training data sample weights for the 'Female' group to promote a more equitable distribution prior to model training.
4.  **Continuous Monitoring:** Establish a robust continuous monitoring system for `Disparate Impact` and `Demographic Parity Difference` for the 'sex' attribute to ensure the sustained effectiveness of mitigation efforts and to detect any emergent bias [1, 2].

### age_group

**Deterministic root causes:** Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.

**Gemini why it is wrong:** The consistent performance across fairness metrics for 'age_group' causally demonstrates that the model's decision-making process does not systematically disadvantage any specific age group. The Disparate Impact ratio near 1, combined with negligible differences in demographic parity, equal opportunity, and average odds, indicates that mortgage lending outcomes are distributed equitably across 'young', 'mid', and 'senior' applicants. This applies to both overall access to favorable outcomes and the accuracy of predictions for eligible and ineligible individuals. Thus, the model is not contributing to age-based discrimination in this context.

**Deterministic mitigations:** Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.

**Gemini how to fix it:** Given the current equitable performance, no immediate algorithmic mitigation is required for the 'age_group' attribute. The primary recommendation is to establish a robust **proactive and continuous monitoring plan** focusing on detecting potential fairness degradation over time. This plan should include:
1.  **Automated Drift Detection:** Implement automated alerts for significant 'drift' in data distributions (e.g., shifts in applicant age demographics or feature values impacting age groups) or concept drift in ground truth labels, as these can causally lead to emergent biases.
2.  **Regular Fairness Re-evaluation:** Integrate periodic, ideally quarterly or bi-annual, re-evaluation of all fairness metrics for 'age_group' into the model's MLOps lifecycle.
3.  **Benchmarking & Guidelines Adherence:** Adhere to established guidelines for model development and validation, such as those emphasizing rigorous testing and transparency [5], to inform monitoring strategies and ensure sustained equitable performance.
This proactive posture, supported by comprehensive frameworks for fair machine learning [1], ensures that any future shifts in data or model behavior that might introduce bias are promptly identified and addressed before they can escalate.
