# Cross-Dataset Gemini LLM Benchmark Comparison

# Benchmark Comparison: German Credit

Deterministic pipeline vs. Gemini qualitative benchmark over fixed self-refinement cycles.

## Napkin Math

| Metric | Value |
|---|---|
| Model | `gemini-2.5-flash` |
| Wall-clock time | 54.15s |
| API attempts | 1 |
| Input tokens | 5,775 |
| Output tokens | 4,004 |
| Total tokens | 9,779 |
| Input cost | $0.0006 (@ $0.1/1M tokens) |
| Output cost | $0.0016 (@ $0.4/1M tokens) |
| **Total est. cost** | **$0.0022** |

## Preflight Estimate

| Metric | Value |
|---|---|
| Projected input tokens | 2,180 |
| Assumed max output tokens | 5,000 |
| Projected total cost ceiling | $0.0022 |

## Cycle Improvement

| Cycle | Total Score | Summary |
|---|---:|---|
| 1 | 90.0 | Total score: 90.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=0.0, Mitigation specificity=15.0, Research grounding=15.0. |
| 2 | 90.0 | Total score: 90.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=0.0, Mitigation specificity=15.0, Research grounding=15.0. |
| 3 | 90.0 | Total score: 90.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=0.0, Mitigation specificity=15.0, Research grounding=15.0. |

## Severity Comparison

| Attribute | Deterministic Baseline | Gemini | Agreement? |
|---|---|---|---|
| Sex_original | MODERATE (borderline) | MODERATE | Yes |
| AgeGroup_original | MODERATE (borderline) | MODERATE | Yes |
| foreign_worker_original | MODERATE (borderline) | MODERATE | Yes |

## Qualitative Narrative Comparison

### Sex_original

**Deterministic root causes:** Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: Sex (r=1.0).; Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0512). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.; Unequal odds: both true-positive and false-positive rates differ across groups (AOD = -0.1256), indicating the model's errors are systematically distributed along group lines.

**Gemini why it is wrong:** Disparities based on sex in credit decisions are ethically problematic and can reinforce historical socioeconomic inequalities, limiting women's access to essential financial services. The consistently lower selection rate and, specifically, the lower True Positive Rate for females (0.878 vs. 0.9293 for males) causally implies that the model is more likely to incorrectly deny credit to *qualified* female applicants. This directly results in missed financial opportunities for women, hinders their economic participation, and poses significant legal risks related to discrimination. Such systematic bias erodes public trust in AI systems, as emphasized in studies on ethical AI deployment [1, 2]. The observed differential treatment points to decision criteria that are not sex-neutral, potentially driven by proxy variables or historical data biases, necessitating a thorough causal analysis to uncover the true drivers of these disparities, a methodology supported by fairness-aware frameworks [3].

**Deterministic mitigations:** Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.; In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.; Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

**Gemini how to fix it:** To mitigate this sex-based bias and address the observed disparities, a targeted, multi-pronged approach is recommended:
1.  **Pre-processing (Data-level - Targets DPD & DI):** Implement **Reweighing** (AIF360) on the training data. This technique adjusts the weights of individual samples to balance the representation and positive outcome rates across sex groups in the training set, thereby directly aiming to reduce the Demographic Parity Difference and Disparate Impact.
2.  **In-processing (Model Training - Targets EOD & AOD):** Employ fairness-aware algorithms during model training, such as **ExponentiatedGradient** or **GridSearch** (Fairlearn). These methods can be configured to optimize the model while satisfying fairness constraints for Equal Opportunity Difference or Average Odds Difference, ensuring more equitable True Positive Rates and False Positive Rates across sex groups during the learning process.
3.  **Post-processing (Prediction Adjustment - Targets DPD, DI, EOD, AOD):** Apply post-processing techniques like **ThresholdOptimizer** (Fairlearn) or **CalibratedEqOddsPostprocessing** (AIF360). `ThresholdOptimizer` can adjust the classification threshold for each group to equalize selection rates (directly addressing DI and DPD) or True Positive Rates (directly addressing EOD). `CalibratedEqOddsPostprocessing` is designed to equalize both TPR and FPR, directly reducing the Average Odds Difference and ensuring fair error rates after predictions are made.
4.  **Causal Analysis & Feature Engineering (Root Cause Identification):** Conduct a rigorous causal inference analysis to identify and disentangle features that genuinely predict creditworthiness from those acting as spurious proxies for sex [3]. This involves reviewing all input features to detect and mitigate any that inadvertently encode sex bias (e.g., occupation types, marital status) that might drive the observed disparities. Decisions should be based on legitimate, non-discriminatory factors to ensure ethical AI deployment.

### AgeGroup_original

**Deterministic root causes:** Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.; Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: AgeGroup (r=1.0), Age (r=-0.841).; Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0513). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.; Unequal odds: both true-positive and false-positive rates differ across groups (AOD = -0.1649), indicating the model's errors are systematically distributed along group lines.

**Gemini why it is wrong:** Age-based discrimination in financial services can severely hinder younger individuals' financial development, impacting their ability to build credit, secure loans for education or housing, and participate fully in the economy. This bias risks perpetuating cycles of financial disadvantage by unfairly linking creditworthiness to age rather than actual financial behavior or capacity. The lower True Positive Rate for 'under_40' (0.8941 vs. 0.9455) causally means that the model is systematically overlooking *qualified* younger applicants, leading to missed economic opportunities and potential frustration with automated credit systems. This differential treatment raises significant ethical concerns in AI applications, especially in finance, where fair access to credit is paramount [2]. Integrating causal inference methods, as discussed in [3], is crucial to understand whether age itself, or factors spuriously correlated with age, are driving these disparities, ensuring decisions are based on genuine risk factors rather than age-based bias.

**Deterministic mitigations:** Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.; Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.; In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.; Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

**Gemini how to fix it:** To address the age-related bias, mitigation strategies should prioritize ensuring equitable treatment across age groups and rectifying the identified disparities:
1.  **Pre-processing (Data-level - Targets DPD & DI):** Implement **Reweighing** (AIF360) to adjust the weights of training samples. This method helps to balance the representation and selection rates of 'under_40' and '40_plus' groups in the dataset, directly aiming to reduce the Demographic Parity Difference and Disparate Impact.
2.  **In-processing (Model Training - Targets EOD & AOD):** Utilize in-processing fairness algorithms like **ExponentiatedGradient** or **GridSearch** (Fairlearn) during model training. These should be configured to optimize for fairness metrics such as Equal Opportunity Difference or Average Odds Difference, which will encourage the model to achieve more equitable True Positive Rates and False Positive Rates across age groups, directly tackling the differential error rates.
3.  **Post-processing (Prediction Adjustment - Targets EOD & AOD):** Apply post-processing techniques such as **EqOddsPostprocessing** or **CalibratedEqOddsPostprocessing** (AIF360) to adjust the model's prediction thresholds. `EqOddsPostprocessing` focuses on equalizing True Positive Rates, while `CalibratedEqOddsPostprocessing` aims to equalize both True and False Positive Rates, thereby directly addressing the observed Equal Opportunity and Average Odds Differences.
4.  **Feature Review & Domain Expertise (Root Cause Identification):** Conduct a detailed review of features that might implicitly correlate with age (e.g., length of employment, credit history duration, types of credit accounts). Engage domain experts to critically evaluate if such correlations are genuinely indicative of differential risk or if they are acting as unfair proxies for age bias. Adjust features, their encoding, or the model's interpretation strategy accordingly to avoid unfair penalization of younger demographics.

### foreign_worker_original

**Deterministic root causes:** Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.; Proxy discrimination: features that correlate with the protected attribute allow the model to indirectly discriminate even without direct access to the attribute. Proxy features detected: foreign_worker (r=1.0).; Representation bias: severely underrepresented groups make the model less reliable for those populations and can amplify existing disparities.; Unequal opportunity: the model is under-predicting favorable outcomes for the unprivileged group (EOD = -0.0889). Deserving members of the unprivileged group are disproportionately missed or incorrectly classified.; Unequal odds: both true-positive and false-positive rates differ across groups (AOD = +0.2013), indicating the model's errors are systematically distributed along group lines.

**Gemini why it is wrong:** The severe data imbalance fundamentally undermines the statistical reliability and validity of any fairness assessment for this attribute. The insufficient data for group '0' makes it impossible for the model to learn meaningful patterns or make robust predictions for this group. Consequently, relying on the calculated metrics to draw conclusions about bias or to implement mitigations would be counterproductive, potentially leading to flawed or even harmful interventions. This lack of reliable data prevents any meaningful detection or measurement of actual bias, making it impossible to ensure ethical AI deployment for this attribute [1, 2]. Without a representative dataset, the foundational requirement for a trustworthy fairness audit cannot be met, as the metrics themselves are not trustworthy.

**Deterministic mitigations:** Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.; Pre-processing -- Disparate Impact Remover (aif360.algorithms.preprocessing.DisparateImpactRemover): transform feature distributions to reduce correlation with the protected attribute while preserving rank-ordering.; Data collection: gather more samples from underrepresented groups. In the interim, use class-weighted training (class_weight='balanced' in scikit-learn) to upweight minority-group errors.; In-processing -- Prejudice Remover (aif360.algorithms.inprocessing.PrejudiceRemover): add a fairness regularization term during model training that penalizes dependence on the protected attribute.; Post-processing -- Equalized Odds (aif360.algorithms.postprocessing.EqOddsPostprocessing): adjust per-group classification thresholds after training to equalize TPR and FPR across groups.

**Gemini how to fix it:** The absolute **first and most critical step** for 'foreign_worker_original' is to decisively address the severe data imbalance. Without reliable data, any algorithmic fairness mitigation would be premature and potentially misguided:
1.  **Prioritized Data Acquisition:** The most robust solution is to **collect significantly more real-world data** for the 'non-foreign worker' group (group '0'). Aim for a sample size of at least 30-50 records for minimally reliable metric calculation, but larger datasets are always preferable for statistical robustness. This is the paramount action to enable a meaningful fairness assessment.
2.  **Careful Data Augmentation / Synthetic Data Generation:** If immediate data collection is not feasible, explore advanced techniques like **SMOTE** (Synthetic Minority Oversampling Technique) or other methods for generating synthetic data. This must be approached with extreme caution, rigorously validating the synthetic data to ensure it accurately reflects the true underlying distribution without introducing new biases or distorting existing ones. The goal is to create plausible new records for the minority group '0' to enable statistical analysis.
3.  **Strategic Resampling Techniques:** Consider **oversampling** the minority group (non-foreign workers) or selectively **undersampling** the majority group (foreign workers) in the training data. Undersampling should only be considered if the overall dataset is very large, to avoid losing critical information from the majority group. These are interim measures until more real data is acquired.
4.  **Re-evaluate Metrics and Re-Audit:** **After** significantly improving the data balance and representation for group '0', **re-evaluate all fairness metrics for this attribute**. Only then can meaningful algorithmic mitigations be considered. If disparities persist post-data balancing, techniques like **Reweighing**, **DisparateImpactRemover**, or **PrejudiceRemover** (AIF360) could be applied as pre-processing steps, or in-processing/post-processing methods as described for other attributes, to target specific biases.


---

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


---
