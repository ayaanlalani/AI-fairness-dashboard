# Benchmark Comparison: HMDA Mortgage Lending (Georgia)

Deterministic pipeline (AIF360) vs. Gemini LLM (fairlearn only -- no access to our calculations).

## Napkin Math

| Metric | Value |
|---|---|
| Model | `gemini-2.5-flash` |
| Wall-clock time | 121.63s |
| API attempts | 1 |
| Input tokens | 22,910 |
| Output tokens | 1,826 |
| Total tokens | 24,736 |
| Input cost | $0.0023 (@ $0.1/1M tokens) |
| Output cost | $0.0007 (@ $0.4/1M tokens) |
| **Total est. cost** | **$0.0030** |

## Metric Comparison

| Attribute | Metric | Our Pipeline | Gemini LLM | Delta |
|---|---|---:|---:|---:|
| race | Disparate Impact | 0.8956 | 0.3258 | -0.5698 |
| race | Demographic Parity Diff | -0.0776 | -0.5173 | -0.4397 |
| race | Equal Opportunity Diff | 0.0025 | -0.0375 | -0.0400 |
| race | Average Odds Diff | 0.0117 | -0.0128 | -0.0245 |
| race | Theil Index | 0.4838 | 0.3193 | -0.1645 |
| sex | Disparate Impact | 0.9291 | 0.9532 | +0.0241 |
| sex | Demographic Parity Diff | -0.0518 | -0.0357 | +0.0161 |
| sex | Equal Opportunity Diff | -0.0076 | -0.0195 | -0.0119 |
| sex | Average Odds Diff | -0.0084 | -0.0121 | -0.0037 |
| sex | Theil Index | 0.4838 | 0.3193 | -0.1645 |
| age_group | Disparate Impact | 1.0138 | 0.9254 | -0.0884 |
| age_group | Demographic Parity Diff | 0.0097 | -0.0559 | -0.0656 |
| age_group | Equal Opportunity Diff | 0.0008 | -0.0463 | -0.0471 |
| age_group | Average Odds Diff | -0.0150 | -0.0218 | -0.0068 |
| age_group | Theil Index | 0.4838 | 0.3193 | -0.1645 |

## Severity Comparison

| Attribute | Our Pipeline | Gemini LLM | Agreement? |
|---|---|---|---|
| race | MODERATE (borderline | CRITICAL | No |
| sex | MODERATE (borderline | LOW | No |
| age_group | LOW (fair | LOW | Yes |

## Qualitative Narrative Comparison

### race

#### What is wrong

**Our pipeline:** The model's favorable-outcome rate varies across race groups. The highest rate is for **Asian** (78.6%) and the lowest is for **Multiracial** (37.5%), a gap of 41.1%.

- Demographic Parity Difference = -0.0776: the unprivileged group is selected less often.
- Equal Opportunity Difference = +0.0025: among truly deserving candidates, the model catches more unprivileged individuals.
- Average Odds Difference = +0.0117: error rates differ systematically across groups.

**Gemini:** For the 'race' attribute, significant disparities are observed. The unprivileged group ('American Indian') has a selection rate that is 0.33 times that of the privileged group ('White'), indicating a negative disparate impact. The demographic parity difference of -0.5173 further confirms that the unprivileged group is less likely to receive the favorable outcome. Moreover, the equal opportunity difference of -0.0375 implies that the model performs worse for the unprivileged group when they are truly positive (lower True Positive Rate). The average odds difference of -0.0128 suggests overall performance disparities across both false positive and true positive rates. 

#### Why it is wrong

**Our pipeline:** **Data imbalance:**
- Severe size imbalance: White has 1252 records vs Pacific Islander with 5 (250x ratio). Metrics for the smaller group have wide confidence intervals.
- Groups with fewer than 30 records (unreliable metrics): American Indian, Multiracial, Pacific Islander.
- Base-rate disparity in the ground truth: Asian has 79.8% favorable outcomes vs Multiracial at 37.5% (gap = 42.3%). The model may be correctly learning a real disparity in historical outcomes, but this historical pattern itself may reflect systemic bias.

**Root causes identified:**
1. Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.
2. Representation bias: severely underrepresented groups make the model less reliable for those populations and can amplify existing disparities.

**Gemini:** Potential root causes for these disparities include: 1) Data Imbalance: The training data might have fewer samples for unprivileged groups, leading to poorer model learning. 2) Historical Bias: The labels or features in the training data could reflect historical biases against certain groups in mortgage lending decisions. 3) Proxy Features: Features highly correlated with protected attributes (e.g., zip code correlating with race) might act as proxies, inadvertently introducing bias. 4) Algorithmic Bias: The model itself might learn to associate protected attributes with outcomes, even if not explicitly trained to do so.

#### How to fix it

**Our pipeline:** 1. Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.
2. Data collection: gather more samples from underrepresented groups. In the interim, use class-weighted training (class_weight='balanced' in scikit-learn) to upweight minority-group errors.

**Gemini:** To mitigate these biases using Fairlearn, several strategies can be applied: 1) Pre-processing: Techniques like re-sampling (e.g., fairlearn.reductions.exponentiated_gradient on re-weighted data) or data augmentation could balance representation or feature distributions. 2) In-processing: Integrate fairness constraints during model training. `fairlearn.reductions.ExponentiatedGradient` or `fairlearn.reductions.GridSearch` can be used with fairness constraints such as `DemographicParity` or `EqualizedOdds` to train a new model that explicitly aims for fairness. 3) Post-processing: Adjust the model's predictions after training. `fairlearn.postprocessing.ThresholdOptimizer` can be particularly effective for binary classification to optimize thresholds for different groups to satisfy fairness metrics like Equal Opportunity.

### sex

#### What is wrong

**Our pipeline:** The model's favorable-outcome rate varies across sex groups. The highest rate is for **Male** (73.1%) and the lowest is for **Female** (67.9%), a gap of 5.2%.

- Demographic Parity Difference = -0.0518: the unprivileged group is selected less often.
- Equal Opportunity Difference = -0.0076: among truly deserving candidates, the model misses more unprivileged individuals.
- Average Odds Difference = -0.0084: error rates differ systematically across groups.

**Gemini:** For the 'sex' attribute, significant disparities are observed. The unprivileged group ('Female') has a selection rate that is 0.95 times that of the privileged group ('Male'), indicating a negative disparate impact. The demographic parity difference of -0.0357 further confirms that the unprivileged group is less likely to receive the favorable outcome. Moreover, the equal opportunity difference of -0.0195 implies that the model performs worse for the unprivileged group when they are truly positive (lower True Positive Rate). The average odds difference of -0.0121 suggests overall performance disparities across both false positive and true positive rates. 

#### Why it is wrong

**Our pipeline:** 

**Gemini:** Potential root causes for these disparities include: 1) Data Imbalance: The training data might have fewer samples for unprivileged groups, leading to poorer model learning. 2) Historical Bias: The labels or features in the training data could reflect historical biases against certain groups in mortgage lending decisions. 3) Proxy Features: Features highly correlated with protected attributes (e.g., zip code correlating with race) might act as proxies, inadvertently introducing bias. 4) Algorithmic Bias: The model itself might learn to associate protected attributes with outcomes, even if not explicitly trained to do so.

#### How to fix it

**Our pipeline:** 1. Post-processing -- Calibrated Equalized Odds (aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing): a minimal intervention that adjusts thresholds to bring DI above the 0.8 threshold while preserving calibration.

**Gemini:** To mitigate these biases using Fairlearn, several strategies can be applied: 1) Pre-processing: Techniques like re-sampling (e.g., fairlearn.reductions.exponentiated_gradient on re-weighted data) or data augmentation could balance representation or feature distributions. 2) In-processing: Integrate fairness constraints during model training. `fairlearn.reductions.ExponentiatedGradient` or `fairlearn.reductions.GridSearch` can be used with fairness constraints such as `DemographicParity` or `EqualizedOdds` to train a new model that explicitly aims for fairness. 3) Post-processing: Adjust the model's predictions after training. `fairlearn.postprocessing.ThresholdOptimizer` can be particularly effective for binary classification to optimize thresholds for different groups to satisfy fairness metrics like Equal Opportunity.

### age_group

#### What is wrong

**Our pipeline:** No significant disparity detected (DI = 1.0138). The model treats age_group groups approximately equally.

**Gemini:** For the 'age_group' attribute, significant disparities are observed. The unprivileged group ('young') has a selection rate that is 0.93 times that of the privileged group ('mid'), indicating a negative disparate impact. The demographic parity difference of -0.0559 further confirms that the unprivileged group is less likely to receive the favorable outcome. Moreover, the equal opportunity difference of -0.0463 implies that the model performs worse for the unprivileged group when they are truly positive (lower True Positive Rate). The average odds difference of -0.0218 suggests overall performance disparities across both false positive and true positive rates. 

#### Why it is wrong

**Our pipeline:** **Root causes identified:**
1. Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.

**Gemini:** Potential root causes for these disparities include: 1) Data Imbalance: The training data might have fewer samples for unprivileged groups, leading to poorer model learning. 2) Historical Bias: The labels or features in the training data could reflect historical biases against certain groups in mortgage lending decisions. 3) Proxy Features: Features highly correlated with protected attributes (e.g., zip code correlating with race) might act as proxies, inadvertently introducing bias. 4) Algorithmic Bias: The model itself might learn to associate protected attributes with outcomes, even if not explicitly trained to do so.

#### How to fix it

**Our pipeline:** 1. Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.

**Gemini:** To mitigate these biases using Fairlearn, several strategies can be applied: 1) Pre-processing: Techniques like re-sampling (e.g., fairlearn.reductions.exponentiated_gradient on re-weighted data) or data augmentation could balance representation or feature distributions. 2) In-processing: Integrate fairness constraints during model training. `fairlearn.reductions.ExponentiatedGradient` or `fairlearn.reductions.GridSearch` can be used with fairness constraints such as `DemographicParity` or `EqualizedOdds` to train a new model that explicitly aims for fairness. 3) Post-processing: Adjust the model's predictions after training. `fairlearn.postprocessing.ThresholdOptimizer` can be particularly effective for binary classification to optimize thresholds for different groups to satisfy fairness metrics like Equal Opportunity.
