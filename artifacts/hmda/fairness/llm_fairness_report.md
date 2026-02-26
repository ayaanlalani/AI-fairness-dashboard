# LLM Fairness Analysis (Gemini): HMDA Mortgage Lending (Georgia)

**Model:** gemini-2.5-flash
**Method:** Independent analysis using fairlearn (metrics and mitigations)

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

## Computed Metrics

| Attribute | Privileged | DI | DPD | EOD | AOD | Theil | Bias? |
|---|---|---:|---:|---:|---:|---:|---|
| race | White | 0.3258 | -0.5173 | -0.0375 | -0.0128 | 0.3193 | Yes |
| sex | Male | 0.9532 | -0.0357 | -0.0195 | -0.0121 | 0.3193 | No |
| age_group | mid | 0.9254 | -0.0559 | -0.0463 | -0.0218 | 0.3193 | No |

---
## race  (severity: CRITICAL)

### What is wrong

For the 'race' attribute, significant disparities are observed. The unprivileged group ('American Indian') has a selection rate that is 0.33 times that of the privileged group ('White'), indicating a negative disparate impact. The demographic parity difference of -0.5173 further confirms that the unprivileged group is less likely to receive the favorable outcome. Moreover, the equal opportunity difference of -0.0375 implies that the model performs worse for the unprivileged group when they are truly positive (lower True Positive Rate). The average odds difference of -0.0128 suggests overall performance disparities across both false positive and true positive rates. 

### Why it is wrong

Potential root causes for these disparities include: 1) Data Imbalance: The training data might have fewer samples for unprivileged groups, leading to poorer model learning. 2) Historical Bias: The labels or features in the training data could reflect historical biases against certain groups in mortgage lending decisions. 3) Proxy Features: Features highly correlated with protected attributes (e.g., zip code correlating with race) might act as proxies, inadvertently introducing bias. 4) Algorithmic Bias: The model itself might learn to associate protected attributes with outcomes, even if not explicitly trained to do so.

### How to fix it

To mitigate these biases using Fairlearn, several strategies can be applied: 1) Pre-processing: Techniques like re-sampling (e.g., fairlearn.reductions.exponentiated_gradient on re-weighted data) or data augmentation could balance representation or feature distributions. 2) In-processing: Integrate fairness constraints during model training. `fairlearn.reductions.ExponentiatedGradient` or `fairlearn.reductions.GridSearch` can be used with fairness constraints such as `DemographicParity` or `EqualizedOdds` to train a new model that explicitly aims for fairness. 3) Post-processing: Adjust the model's predictions after training. `fairlearn.postprocessing.ThresholdOptimizer` can be particularly effective for binary classification to optimize thresholds for different groups to satisfy fairness metrics like Equal Opportunity.

---
## sex  (severity: LOW)

### What is wrong

For the 'sex' attribute, significant disparities are observed. The unprivileged group ('Female') has a selection rate that is 0.95 times that of the privileged group ('Male'), indicating a negative disparate impact. The demographic parity difference of -0.0357 further confirms that the unprivileged group is less likely to receive the favorable outcome. Moreover, the equal opportunity difference of -0.0195 implies that the model performs worse for the unprivileged group when they are truly positive (lower True Positive Rate). The average odds difference of -0.0121 suggests overall performance disparities across both false positive and true positive rates. 

### Why it is wrong

Potential root causes for these disparities include: 1) Data Imbalance: The training data might have fewer samples for unprivileged groups, leading to poorer model learning. 2) Historical Bias: The labels or features in the training data could reflect historical biases against certain groups in mortgage lending decisions. 3) Proxy Features: Features highly correlated with protected attributes (e.g., zip code correlating with race) might act as proxies, inadvertently introducing bias. 4) Algorithmic Bias: The model itself might learn to associate protected attributes with outcomes, even if not explicitly trained to do so.

### How to fix it

To mitigate these biases using Fairlearn, several strategies can be applied: 1) Pre-processing: Techniques like re-sampling (e.g., fairlearn.reductions.exponentiated_gradient on re-weighted data) or data augmentation could balance representation or feature distributions. 2) In-processing: Integrate fairness constraints during model training. `fairlearn.reductions.ExponentiatedGradient` or `fairlearn.reductions.GridSearch` can be used with fairness constraints such as `DemographicParity` or `EqualizedOdds` to train a new model that explicitly aims for fairness. 3) Post-processing: Adjust the model's predictions after training. `fairlearn.postprocessing.ThresholdOptimizer` can be particularly effective for binary classification to optimize thresholds for different groups to satisfy fairness metrics like Equal Opportunity.

---
## age_group  (severity: LOW)

### What is wrong

For the 'age_group' attribute, significant disparities are observed. The unprivileged group ('young') has a selection rate that is 0.93 times that of the privileged group ('mid'), indicating a negative disparate impact. The demographic parity difference of -0.0559 further confirms that the unprivileged group is less likely to receive the favorable outcome. Moreover, the equal opportunity difference of -0.0463 implies that the model performs worse for the unprivileged group when they are truly positive (lower True Positive Rate). The average odds difference of -0.0218 suggests overall performance disparities across both false positive and true positive rates. 

### Why it is wrong

Potential root causes for these disparities include: 1) Data Imbalance: The training data might have fewer samples for unprivileged groups, leading to poorer model learning. 2) Historical Bias: The labels or features in the training data could reflect historical biases against certain groups in mortgage lending decisions. 3) Proxy Features: Features highly correlated with protected attributes (e.g., zip code correlating with race) might act as proxies, inadvertently introducing bias. 4) Algorithmic Bias: The model itself might learn to associate protected attributes with outcomes, even if not explicitly trained to do so.

### How to fix it

To mitigate these biases using Fairlearn, several strategies can be applied: 1) Pre-processing: Techniques like re-sampling (e.g., fairlearn.reductions.exponentiated_gradient on re-weighted data) or data augmentation could balance representation or feature distributions. 2) In-processing: Integrate fairness constraints during model training. `fairlearn.reductions.ExponentiatedGradient` or `fairlearn.reductions.GridSearch` can be used with fairness constraints such as `DemographicParity` or `EqualizedOdds` to train a new model that explicitly aims for fairness. 3) Post-processing: Adjust the model's predictions after training. `fairlearn.postprocessing.ThresholdOptimizer` can be particularly effective for binary classification to optimize thresholds for different groups to satisfy fairness metrics like Equal Opportunity.
