# Deterministic pipeline vs LLM narrative audit — the contrast finding

Stage 4's deliverable is not a score. It is the *contrast* between what the
deterministic template report produces and what the LLM narrative audit
produces over identical computed context. Both read the same metrics; neither
recomputes them. What differs is what each can say about them.

Spend for the three narrative audits: **$0.3514** (o3, 3 self-refinement cycles
each), plus **$0.0181** for the cross-model rubric judge.

## Where the LLM adds something the template cannot

**1. It reasons about metric fragility, not just metric values.**

The deterministic report states German Credit `foreign_worker_original` has
DI = 0.9402 and flags the subgroup as under the n ≥ 15 reporting floor. The
LLM audit reached the same conclusion and then quantified the consequence:

> "the privileged group ('0') has only six records; one label change would
> swing DI to 0.78 (high severity). Thus, statistical power is inadequate and
> the risk of undetected bias is real."

That sensitivity statement — *one label flip moves this from mild to high
severity* — is not in any template field. It is the difference between
reporting a caveat and explaining why the number should not be trusted.

**2. It refuses the flattering reading of a degenerate model.**

lending_club's classifier predicts non-default for ~99.5% of the test set, so
every attribute lands at near-parity and a naive reading is "this model is
fair". The deterministic findings called this out as *insufficient model
discrimination to measure disparity*. The LLM audit independently reached the
same framing:

> "no protected attribute currently violates the disparate-impact threshold,
> but this stems from a near-trivial classifier that approves almost every
> application. Once realistic credit thresholds are enforced, literature
> suggests latent gender disparities may surface."

It converted a null result into a conditional prediction with a stated
mechanism — and cited Bartlett et al. (2019) on threshold choice as a latent
source of disparate impact to support it.

**3. It sequences remediation by feasibility.**

Template output lists candidate mitigations. The LLM output ordered them into
immediate / sprint / data-acquisition tiers with hyperparameters and numeric
targets: "Calibrated-Equalized-Odds post-processor to cap TPR/FPR gaps ≤ 3 pp",
"Prejudice-Remover regulariser with λ = 0.4", "target DI ↑ to ≥ 0.9", "route
decisions concerning group '0' to manual review until N ≥ 30". Whether those
specific numbers are well-calibrated is untested here; the structural point is
that the narrative form supports prioritisation and the template form does not.

## Where the deterministic pipeline remains strictly better

**1. Every number in it is computed, not asserted.** Track Q measured what
happens when an LLM handles the same metrics: 3 of 9 attribute pairs changed
severity label across four prompt strategies on identical input, and without
embedded research evidence the same model produced 16 fabricated citations
across 24 records. The deterministic report cannot drift and cannot fabricate.

**2. It is free and instant.** $0 and no network, against $0.35 and ~3.5
minutes for three audits.

**3. Its severity labels are reproducible by construction.** The LLM's are not:
gpt-4o agreed with the deterministic baseline on only 33% of German Credit
attribute-cycles, and where it disagreed it over-escalated 7 times against 2
under-escalations.

## The finding

The two are not competing implementations of the same thing, and the framing
"which is more accurate?" is a category error. **The deterministic pipeline
establishes what is true; the narrative audit establishes what it means for a
decision.** Track Q shows the LLM should not be trusted to *set* a severity
label — it is unstable under prompt phrasing and, ungrounded, it fabricates
support. Track H shows it can do something the template cannot: state why a
number is fragile, predict what a null result conceals, and sequence a fix.

The defensible architecture is the one this repo already implements by
accident: Python owns the numbers and the severity thresholds, the LLM owns the
interpretation, and the interpretation is gated on evidence it did not invent.
Guardrail 5 — "LLMs never compute metrics; asserting an unprovided value is a
hallucination flag, not a result" — is doing real work, and Track Q's
16-to-2 fabrication drop is the measurement that justifies it.

## A finding about the guardrails themselves

Two attributes failed gate G4 ("mitigation must name a canonical algorithm"):
German Credit `foreign_worker_original` and lending_club `loan_amount_level`.
Both are data-scarcity findings, and for both the audit proposed acquiring more
records to a stated floor, Bayesian hierarchical shrinkage, bootstrapped
confidence intervals, and manual review until n is adequate — rather than a
post-processing algorithm.

That is the correct answer. No post-processor repairs an n=6 subgroup, and
applying one would manufacture false confidence. `guardrails_baseline.json`
encodes an assumption that every fairness problem has an algorithmic
mitigation, so it penalises the epistemically sound response. The gate was left
as-is and the failure reported, rather than widening the keyword list until it
passed.

The same pattern appeared in Track Q's refusal detector, which flagged 6 of 9
`constrained`-prompt records as refusals when none were: it treats any
narrative field under 40 characters as empty, so `how_to_fix:
"DisparateImpactRemover"` — precise and on-spec in 22 characters — reads as a
refusal. **Both cases are the same methodological hazard: a mechanical proxy
standing in for a judgement, quietly penalising the better answer.** For a
benchmark artifact that is worth reporting as prominently as the model results,
because anyone reusing these harnesses inherits the proxies.
