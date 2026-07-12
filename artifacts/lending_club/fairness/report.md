# AI Fairness Analysis Report

**Dataset**: Lending Club Loan Default Prediction
**Total Records**: 600
**Default Rate**: 12.0%
**Prediction Default Rate**: 0.3%

## Fairness Metrics Summary

### Gender

- **Privileged Group**: male
- **Groups**: female, male
- **Disparate Impact**: N/A
- **Demographic Parity Difference**: -0.007
- **Equal Opportunity Difference**: -0.071
- **Average Odds Difference**: -0.036
- **Bias Status**: ⚠️ BIAS DETECTED

### Income Level

- **Privileged Group**: medium
- **Groups**: low, medium, high
- **Disparate Impact**: N/A
- **Demographic Parity Difference**: 0.007
- **Equal Opportunity Difference**: 0.065
- **Average Odds Difference**: 0.032
- **Bias Status**: ✅ WITHIN THRESHOLD

### Loan Amount Level

- **Privileged Group**: medium
- **Groups**: medium, small, large
- **Disparate Impact**: N/A
- **Demographic Parity Difference**: 0.006
- **Equal Opportunity Difference**: 0.054
- **Average Odds Difference**: 0.027
- **Bias Status**: ✅ WITHIN THRESHOLD

## Interpretation

- **Disparate Impact < 0.8**: Potential bias (80% rule)
- **Values closer to 1.0 or 0.0**: Better fairness
- **Large absolute differences**: Concerning for equity

## Recommendations

⚠️ **Bias detected in one or more protected attributes:**
1. Review model training data for historical bias
2. Consider bias mitigation techniques (reweighting, adversarial debiasing)
3. Implement fairness constraints during model training
4. Monitor model performance across all groups regularly