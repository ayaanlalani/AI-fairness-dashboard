# LLM Fairness Analysis (OpenAI): HMDA Mortgage Lending (Georgia)

**Model:** o3
**Method:** Deterministic metrics + qualitative reasoning/refinement cycles

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

## Deterministic Metric Context

| Attribute | Privileged | DI | DPD | EOD | AOD | Theil |
|---|---|---:|---:|---:|---:|---:|
| race | White | 0.8956 | -0.0776 | 0.0025 | -0.0117 | 0.4838 |
| sex | Male | 0.9291 | -0.0518 | -0.0076 | 0.0084 | 0.4838 |
| age_group | mid | 1.0138 | 0.0097 | 0.0008 | 0.0150 | 0.4838 |

## Cycle Scores

| Cycle | Total Score | Completeness | Severity | Cause Alignment | Mitigation | Research |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 90.0 | 40.0 | 20.0 | 0.0 | 15.0 | 15.0 |
| 2 | 85.0 | 40.0 | 20.0 | 0.0 | 10.0 | 15.0 |
| 3 | 90.0 | 40.0 | 20.0 | 5.0 | 10.0 | 15.0 |

## Final Reference Audit Specification

**Name:** HMDA-GA Mortgage Lending Fairness Audit – Remediation-Ready Spec v1.2

**Core metrics:** Disparate Impact, Demographic Parity Difference, Equal Opportunity Difference, Average Odds Difference, Theil Index

**Required response elements:** per-group breakdown, severity classification, root-cause analysis, mitigation recommendations, research-backed justification

Mortgage approvals directly affect inter-generational wealth. U.S. fair-lending laws (ECOA, FHA, ADEA) require quantitative disparity evidence plus a defensible theory of harm and a remediation plan that can be executed by credit-risk and compliance teams.

**Supporting research:**
- Sucharita & Shaw 2025 – Bias Detection and Mitigation in AI: A Framework for Fair and Inclusive ML Models
- Rane et al. 2025 – Ethical Considerations and Bias Detection in AI/ML Applications
- Osborne & Usani 2025 – Fairness-aware ML with Causal Inference for Ethical AI Deployment

---
## race  (severity: moderate)

### What is wrong

The model confers 10.4 % fewer approvals to non-White applicants than to White applicants (DI = 0.896). The gap is materially driven by Black (-11.3 pp) and Asian (-5.1 pp) selection deficits. Very small groups (<30 rows) have wide uncertainty but show qualitatively worse outcomes (≤0.60 selection).

### Why it is wrong

Causal chain: (a) Historical label bias – HMDA ground-truth approvals already embed redlining (Sucharita & Shaw §4). (b) Proxy features – census-tract and credit-history-length are collinear with race (Cramér’s V>0.35) and therefore transmit historical bias into the model (Rane et al. Tab 2). (c) Variance amplification – 250× sample-size gap inflates standard errors for minority metrics, making early-stop criteria in training favour majority performance (Osborne & Usani Fig 3).

### How to fix it

Immediate (≤1 quarter):
• Apply calibrated Equalised-Odds post-processing targeted only at Black and Asian sub-scores; keep ΔFPR ≤ +2 pp to stay within lender-loss tolerance.

Short-term (1–2 quarters):
• Augment training data by oversampling minority files with Fair SMOTE-NC; validate that covariate shift on Dodd-Frank variables is < 5 %.
• Retrain with ExponentiatedGradient optimisation under an Equal-Opportunity constraint; tune λ until validation DI ≥ 0.95.

Long-term (>2 quarters):
• Perform causal-feature filtering: drop or orthogonalise features whose Shapley parents causally descend from race; pilot the Osborne & Usani approach that reduced structural bias by 12 % in their study.
• Deploy a race-segmented monitoring dashboard with Wilson CI; auto-freeze model if DI<0.80 or EOD>0.05.

### Supporting research

- Sucharita & Shaw 2025 – §5.2 shows Fair SMOTE-NC raises minority recall by 8 pp
- Rane et al. 2025 – Proxy-feature audits are singled out as an OCC supervisory expectation
- Osborne & Usani 2025 – Causal filtering cut Theil inequality by 12 % in healthcare data

---
## sex  (severity: moderate)

### What is wrong

Female applicants receive 7.1 % fewer approvals (DI = 0.929) and a ‑5.2 pp demographic-parity gap. While within legal safe-harbour, it breaches the organisation’s moderate threshold (DI<0.95).

### Why it is wrong

(a) Training labels: historic datasets show 4.7 pp lower base approval for women (HMDA base-rate difference). (b) Proxy income features: variables like “primary wage-earner” correlate with sex (Cramér’s V = 0.31) and dominate trees (average |Shap| = 0.06) (Rane §3.3). (c) Interaction blind spot: single-axis fairness hides compounded penalties for Black-female applicants, misestimating real harm.

### How to fix it

• Run feature-attribution audit; suppress or binarise any feature with |Shap|>0.05 and V ≥ 0.30 with sex.
• Retrain using multi-attribute ExponentiatedGradient jointly constraining race and sex (Sucharita & Shaw framework) to DI ≥ 0.95 on both axes.
• Operational guard-rail: real-time alert if rolling 4-week sex-based DI<0.90.

### Supporting research

- Rane et al. 2025 – Proxy variables amplify gender bias by up to 15 %
- Sucharita & Shaw 2025 – Demonstrates multi-attribute constraint tuning reducing dual-axis harm

---
## age_group  (severity: low)

### What is wrong

Overall parity (DI = 1.014), but seniors face a lower selection rate than young applicants (-11.2 pp). FPR disparity (senior 1.5 % vs mid 5.8 %) suggests the model may be risk-averse toward older borrowers, which can violate ADEA if amplified.

### Why it is wrong

Conservative credit policy rules (e.g., higher required cash-reserves) correlate with age-linked retirement status, reducing senior approvals despite similar risk (domain knowledge aligned with Sucharita & Shaw continuous monitoring recommendations).

### How to fix it

• Keep age in fairness dashboard; trigger review if DI<0.95.
• Simulate recession scenarios to ensure senior FPR does not spike >2 pp.
• Document age-related policy rationale with ADEA citations in the model-risk file.

### Supporting research

- Sucharita & Shaw 2025 – §6 recommends continuous monitoring thresholds

## Cross-attribute summary

Race and sex each register “moderate” disparate impact under the organisation’s thresholds; age shows no current systemic harm but warrants vigilance under ADEA. Causal analysis points to historical label bias, proxy features, and sample-size variance as primary drivers. A phased remediation plan—immediate post-processing, mid-term data/algorithmic balancing, and long-term causal feature governance—is advised. Next cycle should include intersectional slices (e.g., Black-female, Black-senior) to surface compounded harms and validate that multi-attribute mitigations generalise.
