# LLM Fairness Analysis (Gemini): HMDA Mortgage Lending (Georgia)

**Model:** gemini-2.5-flash
**Method:** Deterministic metrics + qualitative reasoning/refinement cycles

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

## Deterministic Metric Context

| Attribute | Privileged | DI | DPD | EOD | AOD | Theil |
|---|---|---:|---:|---:|---:|---:|
| race | White | 0.8956 | -0.0776 | 0.0025 | 0.0117 | 0.4838 |
| sex | Male | 0.9291 | -0.0518 | -0.0076 | -0.0084 | 0.4838 |
| age_group | mid | 1.0138 | 0.0097 | 0.0008 | -0.0150 | 0.4838 |

## Cycle Scores

| Cycle | Total Score | Completeness | Severity | Cause Alignment | Mitigation | Research |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 78.3 | 40.0 | 13.3 | 0.0 | 10.0 | 15.0 |
| 2 | 78.3 | 40.0 | 13.3 | 0.0 | 10.0 | 15.0 |
| 3 | 78.3 | 40.0 | 13.3 | 0.0 | 10.0 | 15.0 |

## Final Reference Audit Specification

**Name:** Remediation-Ready AI Fairness Audit Specification (Cycle 3)

**Core metrics:** Disparate Impact, Demographic Parity Difference, Equal Opportunity Difference, Average Odds Difference, Theil Index

**Required response elements:** per-group breakdown, severity classification, root-cause analysis, mitigation recommendations, research-backed justification, continuous monitoring plan

A comprehensive, remediation-ready fairness audit specification is paramount for proactively identifying, thoroughly understanding, and effectively addressing algorithmic bias. This specification integrates quantitative disparity measurements with qualitative elements like interpretable causal analysis, explicit severity classifications, and concrete, actionable mitigation recommendations. The inclusion of root-cause analysis and robust research-backed justifications ensures that interventions are precisely targeted and grounded in established fairness principles. Furthermore, mandating a continuous monitoring plan is crucial for maintaining a sustained fairness posture beyond initial deployment, adapting to data shifts and evolving ethical landscapes. This holistic approach aligns directly with modern frameworks for fair and inclusive machine learning models [1] and is essential for navigating the complex ethical considerations in AI deployment, particularly in sensitive high-stakes domains such as finance [2]. The emphasis on causal analysis and concrete mitigation strategies is further supported by emerging fairness-aware machine learning frameworks [3].

**Supporting research:**
- Bias Detection and Mitigation in AI: A Framework for fair and inclusive Machine Learning Models (2025) | S. Sucharita, Prof. Kamalakanta Shaw | INTERNATIONAL JOURNAL OF SCIENTIFIC RESEARCH IN ENGINEERING AND MANAGEMENT
- Ethical Considerations and Bias Detection in Artificial Intelligence/Machine Learning Applications (2025) | Jayesh Rane, Reshma Amol Chaudhari, N. Rane | Unknown venue
- A fairness-aware machine learning framework for maternal health in Ghana: integrating explainability, bias mitigation, and causal inference for ethical AI deployment (2025) | Augustus Osborne, Kobloobase O. Usani | BioData Mining

---
## race  (severity: CRITICAL)

### What is wrong

The 'race' attribute presents critical fairness challenges. For the 'Black' group, the Disparate Impact (DI) relative to 'White' is 0.8675 (Black selection rate 0.6444 / White selection rate 0.7428), which falls into the 'moderate' severity range (0.8 < DI < 0.95). However, this moderate DI is compounded by severe data sparsity for 'American Indian' (10 records), 'Multiracial' (8 records), and 'Pacific Islander' (5 records), rendering computed metrics for these groups statistically unreliable and making effective bias detection impossible. Furthermore, a substantial base-rate disparity in the ground truth is observed: 'Asian' individuals have a 79.8% favorable outcome rate compared to 'Multiracial' at 37.5% (a 42.3% gap). This indicates deeply embedded systemic biases within the historical lending data itself, which the model is inherently learning and perpetuating.

### Why it is wrong

The moderate Disparate Impact for the 'Black' group causally implies that Black applicants receive favorable mortgage outcomes at a measurably lower rate than White applicants. This directly limits access to financial capital, exacerbating existing socioeconomic disparities. More critically, the pronounced base-rate disparity in the historical data for several racial groups signifies that the ground truth labels themselves are a product of systemic inequities. When the model is trained on such biased historical data, it **causally learns and replicates these disparities**, leading to algorithmic outcomes that reflect past discrimination rather than objective merit [1, 2]. The severe data sparsity for 'American Indian', 'Multiracial', and 'Pacific Islander' groups makes it impossible to reliably assess or mitigate bias for these highly vulnerable populations, undermining the core objective of equitable AI deployment [3].

### How to fix it

Addressing the critical fairness issues for the 'race' attribute demands a comprehensive strategy:
1.  **For Severe Data Sparsity:** Given the extremely low counts for 'American Indian', 'Multiracial', and 'Pacific Islander' groups, **targeted data collection** should be the primary and most robust approach to acquire reliable, representative data. If immediate data collection is infeasible, **advanced synthetic data generation methods** (e.g., CTGAN or SMOTE for tabular data) can be explored, but with extreme caution and rigorous validation to avoid introducing further biases or misrepresenting rare groups. These techniques aim to increase statistical power for meaningful fairness metric computation.
2.  **For Disparate Impact & Base-Rate Disparity:** To directly mitigate the Disparate Impact and address the underlying base-rate disparities, a combination of algorithmic and systemic interventions is required.
    *   **Algorithmic Mitigation:** Employ **in-processing fairness-aware algorithms** such as Fairlearn's `ExponentiatedGradient` or `GridSearchCV` during model training, explicitly optimizing for metrics like `DemographicParityDifference` to equalize selection rates across racial groups. Alternatively, **pre-processing techniques** like AIF360's `Reweighing` can adjust training sample weights to promote parity, ensuring the model learns a more equitable distribution from the outset.
    *   **Causal & Systemic Intervention:** Conduct a **rigorous causality-focused review of historical lending policies and practices** to identify the true root causes of the observed base-rate disparities. This involves applying causal inference methods to pinpoint features that may act as proxies for race or reflect historical discrimination. Concurrently, **engage with affected communities** and domain experts to collaboratively define equitable lending outcomes and refine policy, ensuring that the 'favorable' label itself reflects fairness rather than historical bias [3].
3.  **Continuous Monitoring:** Implement continuous monitoring of all fairness metrics, especially Disparate Impact and base rates, with alerts for any significant deviations, alongside regular audits [1, 2].

### Supporting research

- Bias Detection and Mitigation in AI: A Framework for fair and inclusive Machine Learning Models (2025) | S. Sucharita, Prof. Kamalakanta Shaw | INTERNATIONAL JOURNAL OF SCIENTIFIC RESEARCH IN ENGINEERING AND MANAGEMENT
- Ethical Considerations and Bias Detection in Artificial Intelligence/Machine Learning Applications (2025) | Jayesh Rane, Reshma Amol Chaudhari, N. Rane | Unknown venue
- A fairness-aware machine learning framework for maternal health in Ghana: integrating explainability, bias mitigation, and causal inference for ethical AI deployment (2025) | Augustus Osborne, Kobloobase O. Usani | BioData Mining

---
## sex  (severity: MODERATE)

### What is wrong

The 'sex' attribute presents a moderate fairness concern with a Disparate Impact (DI) of 0.9291 (Female selection rate 0.6793 / Male selection rate 0.7312), falling into the 'moderate' severity category (0.8 < DI < 0.95). This indicates that 'Female' applicants receive favorable mortgage outcomes at a moderately lower rate than 'Male' applicants. While the Equal Opportunity Difference (-0.0076) and Average Odds Difference (-0.0084) are near zero, implying similar predictive accuracy (true positive and false positive rates) for both sexes, the overall disparity in favorable selection rates, reflected by a Demographic Parity Difference of -0.0518, raises concerns regarding equitable access to financial services.

### Why it is wrong

The moderate Disparate Impact for the 'Female' group causally implies that women are proportionally less likely to receive a favorable mortgage outcome compared to men. This disparity, despite similar predictive accuracy (low EOD/AOD), directly translates to **unequal access** to a crucial financial service. The causal mechanism is that the model, by learning from historical data that may reflect pre-existing gender-based economic inequalities (e.g., income disparities, wealth gaps), perpetuates these outcome disparities. Therefore, even if the model is not exhibiting disparate *treatment* in terms of misclassification rates, it is *causally contributing to disparate outcomes* for female applicants, potentially exacerbating long-term socioeconomic inequalities [2]. This highlights a critical distinction between predictive fairness and outcome fairness, where the model's objective impact is unequal.

### How to fix it

To mitigate the moderate Disparate Impact for the 'sex' attribute, the following specific interventions are recommended:
1.  **Targeted Post-processing:** Given the near-zero Equal Opportunity Difference and Average Odds Difference, **post-processing techniques** are highly effective and efficient as they can directly address `DemographicParityDifference` without compromising the model's equitable predictive accuracy. Specifically, Fairlearn's `ThresholdOptimizer` can be used to identify a distinct classification threshold for the 'Female' group, recalibrating outcomes to equalize selection rates (i.e., achieve demographic parity) with the 'Male' group.
2.  **Causal Proxy Investigation:** Conduct a **causal analysis** to identify and understand any features strongly correlated with 'sex' that may be acting as proxies, implicitly driving the observed Disparate Impact. This involves explainable AI (XAI) techniques and domain expert consultation to uncover and potentially neutralize these indirect sources of bias.
3.  **Algorithmic Refinement (Optional):** If model retraining is feasible, **in-processing methods** like Fairlearn's `ExponentiatedGradient` can be utilized, explicitly optimizing the model to satisfy `DemographicParityDifference` during training. Alternatively, **pre-processing techniques** like AIF360's `Reweighing` can adjust training data sample weights for the 'Female' group to promote a more equitable distribution prior to model training.
4.  **Continuous Monitoring:** Establish a robust continuous monitoring system for `Disparate Impact` and `Demographic Parity Difference` for the 'sex' attribute to ensure the sustained effectiveness of mitigation efforts and to detect any emergent bias [1, 2].

### Supporting research

- Bias Detection and Mitigation in AI: A Framework for fair and inclusive Machine Learning Models (2025) | S. Sucharita, Prof. Kamalakanta Shaw | INTERNATIONAL JOURNAL OF SCIENTIFIC RESEARCH IN ENGINEERING AND MANAGEMENT
- Ethical Considerations and Bias Detection in Artificial Intelligence/Machine Learning Applications (2025) | Jayesh Rane, Reshma Amol Chaudhari, N. Rane | Unknown venue

---
## age_group  (severity: LOW)

### What is wrong

The 'age_group' attribute exhibits a low severity fairness risk, as indicated by its metrics. The Disparate Impact (DI) is 1.0138, which is above the 'moderate' threshold of 0.95. A DI greater than 1 suggests a slightly favorable impact for the unprivileged 'senior' and 'young' groups relative to the privileged 'mid' age group, indicating no adverse impact. Furthermore, the Demographic Parity Difference (0.0097), Equal Opportunity Difference (0.0008), and Average Odds Difference (-0.015) are all exceptionally close to zero. This collective evidence points to highly equitable outcomes across all age groups in terms of overall selection rates, true positive rates, and false positive rates.

### Why it is wrong

The consistent performance across fairness metrics for 'age_group' causally demonstrates that the model's decision-making process does not systematically disadvantage any specific age group. The Disparate Impact ratio near 1, combined with negligible differences in demographic parity, equal opportunity, and average odds, indicates that mortgage lending outcomes are distributed equitably across 'young', 'mid', and 'senior' applicants. This applies to both overall access to favorable outcomes and the accuracy of predictions for eligible and ineligible individuals. Thus, the model is not contributing to age-based discrimination in this context.

### How to fix it

Given the current equitable performance, no immediate algorithmic mitigation is required for the 'age_group' attribute. The primary recommendation is to establish a robust **proactive and continuous monitoring plan** focusing on detecting potential fairness degradation over time. This plan should include:
1.  **Automated Drift Detection:** Implement automated alerts for significant 'drift' in data distributions (e.g., shifts in applicant age demographics or feature values impacting age groups) or concept drift in ground truth labels, as these can causally lead to emergent biases.
2.  **Regular Fairness Re-evaluation:** Integrate periodic, ideally quarterly or bi-annual, re-evaluation of all fairness metrics for 'age_group' into the model's MLOps lifecycle.
3.  **Benchmarking & Guidelines Adherence:** Adhere to established guidelines for model development and validation, such as those emphasizing rigorous testing and transparency [5], to inform monitoring strategies and ensure sustained equitable performance.
This proactive posture, supported by comprehensive frameworks for fair machine learning [1], ensures that any future shifts in data or model behavior that might introduce bias are promptly identified and addressed before they can escalate.

### Supporting research

- Bias Detection and Mitigation in AI: A Framework for fair and inclusive Machine Learning Models (2025) | S. Sucharita, Prof. Kamalakanta Shaw | INTERNATIONAL JOURNAL OF SCIENTIFIC RESEARCH IN ENGINEERING AND MANAGEMENT
- Assessing Adherence to TRIPOD+AI Guidelines in Machine Learning Models for Predicting Small for Gestational Age and Fetal Growth Restriction: A Systematic Review. (2025) | G. Zamagni, C. Fregona, Moira Barbieri | American Journal of Obstetrics & Gynecology MFM
- Technical benchmarking of eight machine‑learning algorithms versus penalized logistic regression for predicting in‑hospital death in 117,765 u.S. CLL/SLL admissions (2025) | Xiaoyi Zhang, Abhishek Kumar, L. Xiao | Blood

## Cross-attribute summary

This Cycle 3 fairness audit of the HMDA Mortgage Lending model reveals a nuanced landscape of fairness concerns requiring distinct interventions. The 'race' attribute presents the most critical challenge, marked by severe data sparsity for highly marginalized groups and deeply embedded base-rate disparities in historical lending outcomes. These issues causally lead the model to learn and perpetuate systemic biases present in the ground truth, necessitating comprehensive interventions including targeted data acquisition, explicit fairness-aware algorithms, rigorous causal investigation of historical lending policies, and meaningful community engagement [3]. The 'sex' attribute exhibits a moderate Disparate Impact against females, indicating a quantifiable disparity in favorable mortgage outcomes. While the model maintains predictive accuracy across sexes, this outcome disparity requires targeted post-processing or in-processing techniques and a causal examination of proxy features to ensure equitable access [1, 2]. Conversely, the 'age_group' attribute demonstrates consistently equitable performance across all fairness metrics, requiring no immediate algorithmic intervention but a steadfast commitment to proactive and continuous monitoring to detect any emergent bias [1]. Collectively, these findings underscore that achieving and maintaining AI fairness in sensitive financial applications demands an iterative and holistic approach. This extends beyond just algorithmic adjustments to include critical examination and remediation of underlying data biases, historical inequities, and ongoing oversight throughout the model's lifecycle to ensure ethical and responsible AI deployment.
