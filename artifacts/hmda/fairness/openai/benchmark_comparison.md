# Benchmark Comparison: HMDA Mortgage Lending (Georgia)

Deterministic pipeline vs. OpenAI qualitative benchmark over fixed self-refinement cycles.

## Napkin Math

| Metric | Value |
|---|---|
| Model | `o3` |
| Wall-clock time | 14.22s |
| API attempts | 1 |
| Input tokens | 4,032 |
| Output tokens | 1,666 |
| Total tokens | 5,698 |
| Input cost | $0.0403 (@ $10.0/1M tokens) |
| Output cost | $0.0666 (@ $40.0/1M tokens) |
| **Total est. cost** | **$0.1070** |

## Cost Guardrail

| Metric | Value |
|---|---|
| Projected input tokens | 2,327 |
| Max output tokens | 5,000 |
| Projected total cost ceiling | $0.2233 |
| Configured max cost | $37.00 USD (approx. $50.00 CAD) |

## Cycle Improvement

| Cycle | Total Score | Summary |
|---|---:|---|
| 1 | 90.0 | Total score: 90.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=0.0, Mitigation specificity=15.0, Research grounding=15.0. |
| 2 | 85.0 | Total score: 85.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=0.0, Mitigation specificity=10.0, Research grounding=15.0. |
| 3 | 90.0 | Total score: 90.0/100. Completeness=40.0, Severity agreement=20.0, Cause alignment=5.0, Mitigation specificity=10.0, Research grounding=15.0. |

## Severity Comparison

| Attribute | Deterministic Baseline | OpenAI | Agreement? |
|---|---|---|---|
| race | MODERATE (borderline) | moderate | Yes |
| sex | MODERATE (borderline) | moderate | Yes |
| age_group | LOW (fair) | low | Yes |

## Qualitative Narrative Comparison

### race

**Deterministic root causes:** Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.; Representation bias: severely underrepresented groups make the model less reliable for those populations and can amplify existing disparities.

**OpenAI:** Causal chain: (a) Historical label bias – HMDA ground-truth approvals already embed redlining (Sucharita & Shaw §4). (b) Proxy features – census-tract and credit-history-length are collinear with race (Cramér’s V>0.35) and therefore transmit historical bias into the model (Rane et al. Tab 2). (c) Variance amplification – 250× sample-size gap inflates standard errors for minority metrics, making early-stop criteria in training favour majority performance (Osborne & Usani Fig 3).

**Deterministic mitigations:** Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.; Data collection: gather more samples from underrepresented groups. In the interim, use class-weighted training (class_weight='balanced' in scikit-learn) to upweight minority-group errors.

**OpenAI:** Immediate (≤1 quarter):
• Apply calibrated Equalised-Odds post-processing targeted only at Black and Asian sub-scores; keep ΔFPR ≤ +2 pp to stay within lender-loss tolerance.

Short-term (1–2 quarters):
• Augment training data by oversampling minority files with Fair SMOTE-NC; validate that covariate shift on Dodd-Frank variables is < 5 %.
• Retrain with ExponentiatedGradient optimisation under an Equal-Opportunity constraint; tune λ until validation DI ≥ 0.95.

Long-term (>2 quarters):
• Perform causal-feature filtering: drop or orthogonalise features whose Shapley parents causally descend from race; pilot the Osborne & Usani approach that reduced structural bias by 12 % in their study.
• Deploy a race-segmented monitoring dashboard with Wilson CI; auto-freeze model if DI<0.80 or EOD>0.05.

### sex

**Deterministic root causes:** 

**OpenAI:** (a) Training labels: historic datasets show 4.7 pp lower base approval for women (HMDA base-rate difference). (b) Proxy income features: variables like “primary wage-earner” correlate with sex (Cramér’s V = 0.31) and dominate trees (average |Shap| = 0.06) (Rane §3.3). (c) Interaction blind spot: single-axis fairness hides compounded penalties for Black-female applicants, misestimating real harm.

**Deterministic mitigations:** Post-processing -- Calibrated Equalized Odds (aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing): a minimal intervention that adjusts thresholds to bring DI above the 0.8 threshold while preserving calibration.

**OpenAI:** • Run feature-attribution audit; suppress or binarise any feature with |Shap|>0.05 and V ≥ 0.30 with sex.
• Retrain using multi-attribute ExponentiatedGradient jointly constraining race and sex (Sucharita & Shaw framework) to DI ≥ 0.95 on both axes.
• Operational guard-rail: real-time alert if rolling 4-week sex-based DI<0.90.

### age_group

**Deterministic root causes:** Historical / label bias: the ground-truth labels themselves show a significant disparity across groups. The model is learning to replicate outcomes that may embed past discriminatory decisions.

**OpenAI:** Conservative credit policy rules (e.g., higher required cash-reserves) correlate with age-linked retirement status, reducing senior approvals despite similar risk (domain knowledge aligned with Sucharita & Shaw continuous monitoring recommendations).

**Deterministic mitigations:** Pre-processing -- Reweighing (aif360.algorithms.preprocessing.Reweighing): assign sample weights that compensate for historical label imbalance across groups before training.

**OpenAI:** • Keep age in fairness dashboard; trigger review if DI<0.95.
• Simulate recession scenarios to ensure senior FPR does not spike >2 pp.
• Document age-related policy rationale with ADEA citations in the model-risk file.
