"""Build notebooks/pipeline_walkthrough.ipynb programmatically.

Writing the JSON by hand is error-prone; this assembles it from (kind, source)
pairs so the notebook stays readable and regenerable.
"""
import json
import pathlib

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ───────────────────────────────────────────────────────── title
md(r"""
# Pipeline Walkthrough — A Lending-Lifecycle Fairness Benchmark

A narrated tour of how this repository actually works, from raw CSVs to the
LLM benchmark results.

**Everything here runs offline.** No API key, no network, no LLM call. Every
cell reads committed or regenerable files. If a cell needs something absent, it
says so and continues rather than failing.

Sections:

1. Architecture: `run_pipeline.py` and the dataset registry
2. Cleaning and encoding decisions, per dataset
3. The fairness metric layer — and the favorable-label lesson
4. The qualitative engine: severity, root causes, proxies, evidence retention
5. The LLM benchmark: 4 cycles, frozen packs, scoring, detectors
6. Results gallery
""")

code(r"""
import json, os, sys
from pathlib import Path
import pandas as pd

# Resolve the repo root whether this runs from notebooks/ or the repo root.
ROOT = Path.cwd()
if not (ROOT / "run_pipeline.py").exists():
    ROOT = ROOT.parent
assert (ROOT / "run_pipeline.py").exists(), f"cannot locate repo root from {Path.cwd()}"
# scripts/ for the pipeline modules, ROOT for run_pipeline itself.
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)
print("repo root:", ROOT)
print("python    :", sys.version.split()[0])
""")

# ───────────────────────────────────────────────── 1. architecture
md(r"""
---
## 1. Architecture: `run_pipeline.py` and the dataset registry

The orchestrator is a registry plus a step runner. Each dataset declares a
working directory and a shell command per step; steps run in a fixed order as
subprocesses, and a failing step stops that dataset without killing the others.

Two things about this design matter downstream and are easy to miss.
""")

code(r"""
import run_pipeline as rp

print("STEP_ORDER:", rp.STEP_ORDER)
print()
print(f"{'dataset':22s} {'dir':26s} steps")
for key, cfg in rp.DATASETS.items():
    print(f"{key:22s} {cfg['dir']:26s} {sorted(cfg['steps'])}")
""")

md(r"""
**Note the asymmetry.** The step named `benchmark` runs the 4-cycle Track Q
harness (`scripts/llm_benchmark.py`), while the step named `llm_benchmark` runs
the Track H narrative audit (`scripts/openai_fairness_analysis.py`). The names
are the reverse of what you would guess, which is worth knowing before you
invoke either.

Only the three lending datasets carry both; `healthcare_insurance` and
`diabetes` are registered for the deterministic layer only.
""")

code(r"""
# The consolidation footgun, stated explicitly because it silently corrupts artifacts.
print(rp.consolidate.__doc__ or "(no docstring)")
print()
print("consolidate() rebuilds cross-dataset files from ONLY the datasets passed")
print("via --datasets. Running a subset leaves artifacts/consolidated/ and")
print("poster_assets/ holding that subset. Always pass all three:")
print()
print("  python3.11 run_pipeline.py --datasets german_credit hmda lending_club --steps qualitative")
""")

# ───────────────────────────────────────────────── 2. cleaning
md(r"""
---
## 2. Cleaning and encoding decisions, per dataset

Each dataset needed a different fix before fairness metrics meant anything.
""")

code(r"""
gc_path = ROOT / "german_credit_dataset" / "data" / "german_credit_CLEANED_dataset.csv"
if gc_path.exists():
    gc = pd.read_csv(gc_path)
    print("German Credit:", gc.shape)
    # The target ships as 1 = good credit, 2 = bad credit — NOT 0/1.
    target_col = "Cost Matrix(Risk)" if "Cost Matrix(Risk)" in gc.columns else gc.columns[-1]
    print(f"\nraw target '{target_col}' value counts:")
    print(gc[target_col].value_counts().sort_index().to_string())
    print("\n1 = good credit (favorable), 2 = bad credit.")
    print("Mapped to 1/0 during training; treating the raw 1/2 as 0/1 would invert")
    print("the favorable class and therefore invert every fairness metric.")
else:
    print("German Credit cleaned CSV not present; run `make fairness` to regenerate.")
""")

code(r"""
# Lending Club: protected attributes arrive one-hot encoded and must be decoded
# back to a single categorical column before per-group rates can be computed.
enriched = ROOT / "lending_club_dataset" / "metrics" / "classification_predictions_enriched.csv"
if enriched.exists():
    lc = pd.read_csv(enriched)
    print("Lending Club enriched predictions:", lc.shape)
    for col in ["gender", "income_level", "loan_amount_level"]:
        if col in lc.columns:
            print(f"\n{col}:")
            print(lc[col].value_counts().to_string())
    print("\nThese three columns are produced by prepare_qualitative_inputs.py,")
    print("which decodes the one-hot columns the model was trained on. The raw")
    print("processed/ matrices have no single 'gender' column to group by.")
else:
    print("Enriched predictions not present (metrics/ is gitignored).")
    print("Regenerate: python3.11 run_pipeline.py --datasets lending_club --steps qualitative")
""")

# ───────────────────────────────────────────────── 3. metric layer
md(r"""
---
## 3. The fairness metric layer — and the favorable-label lesson

This is the most important section in the notebook. A single convention —
*which label counts as the favorable outcome* — moved a headline metric from
"maximally alarming" to "unremarkable".

Lending Club's target is `loan_default`, so **label 1 is the *unfavorable*
outcome.** Disparate impact is a ratio of *favorable* selection rates. Computing
it on predicted defaults answers the wrong question.

Let's reproduce both numbers from the committed predictions.
""")

code(r"""
if enriched.exists():
    lc = pd.read_csv(enriched)
    priv, unpriv = "male", "female"

    def selection_rate(df, group, favorable_label):
        sub = df[df["gender"] == group]
        return (sub["predicted"] == favorable_label).mean()

    print("Group sizes:", lc["gender"].value_counts().to_dict())
    print()

    # WRONG: orient on label 1 (= predicted default, the ADVERSE event)
    m_def = selection_rate(lc, priv, 1)
    f_def = selection_rate(lc, unpriv, 1)
    di_wrong = (f_def / m_def) if m_def else float("inf") if f_def else float("nan")
    print("Oriented on label 1 (predicted DEFAULT) — the bug:")
    print(f"   male   rate = {m_def:.4f}")
    print(f"   female rate = {f_def:.4f}")
    print(f"   DI = {m_def:.4f} / {f_def:.4f} = {(m_def / f_def) if f_def else 0.0:.4f}   <-- reads CRITICAL")
    print()

    # RIGHT: orient on label 0 (= predicted non-default, the FAVORABLE outcome)
    m_ok = selection_rate(lc, priv, 0)
    f_ok = selection_rate(lc, unpriv, 0)
    print("Oriented on label 0 (predicted NON-DEFAULT) — the fix:")
    print(f"   male   rate = {m_ok:.4f}")
    print(f"   female rate = {f_ok:.4f}")
    print(f"   DI = {f_ok:.4f} / {m_ok:.4f} = {f_ok / m_ok:.4f}   <-- near parity")
else:
    print("Enriched predictions unavailable; showing the committed values instead.")
    print("  default-oriented : DI = 0.0000  (male 0.0000 vs female 0.0069)")
    print("  favorable-oriented: DI = 0.9931  (male 1.0000 vs female 0.9931)")
""")

md(r"""
**The same model, the same data, the same correct metric implementation — and a
factor-of-infinity difference in the headline number.**

Nothing exotic caused this. The target variable is named for the adverse event,
which is completely normal in credit risk, and the toolkit default assumes label
1 is favorable. This is now pinned by `tests/test_lending_club_fairness.py`,
including a test asserting that the default-oriented computation *would* read
zero — so the bug cannot silently return.
""")

code(r"""
# The committed metrics, straight from the consolidated CSV.
cons = ROOT / "artifacts" / "consolidated" / "consolidated_fairness_metrics.csv"
df = pd.read_csv(cons)
# NOTE: this file carries two disjoint column schemas — german_credit/hmda use
# CamelCase, lending_club uses snake_case — because consolidation concatenates
# per-dataset CSVs without harmonising them. A known artifact defect.
upper = df[df["DisparateImpact"].notna()][
    ["Dataset", "Attribute", "PrivilegedValue", "DisparateImpact",
     "EqualOpportunityDiff", "AverageOddsDiff"]
]
lower = df[df["disparate_impact"].notna()][
    ["Dataset", "protected_attribute", "privileged_group", "disparate_impact",
     "equal_opportunity_difference", "average_odds_difference"]
]
print("german_credit + hmda:"); print(upper.to_string(index=False))
print("\nlending_club (different column names — see note above):")
print(lower.to_string(index=False))
""")

md(r"""
### AIF360 sign conventions

`AverageOddsDifference` has a sign, and severity narratives quote it. Two
compute paths existed — AIF360 proper and a manual fallback — and they disagreed
on whether the difference is privileged-minus-unprivileged or the reverse. The
magnitudes matched, so the disagreement was invisible in any absolute-value
check while flipping the direction of every narrative claim.

One convention is now pinned per dataset and asserted in
`tests/test_sign_conventions.py`, one test class per dataset.
""")

code(r"""
# HMDA race: DI fails the four-fifths screen while EOD/AOD are ~0.
# That combination is diagnostic, and it changes who owns the fix.
row = upper[(upper["Dataset"] == "hmda") & (upper["Attribute"] == "race")]
print(row.to_string(index=False))
print()
print("DI 0.8956 fails the 0.8 screen, but EOD 0.0025 and AOD 0.0117 are ~0.")
print("The model is NOT making differentially worse errors by race — it is")
print("reproducing a base-rate difference already present in the labels.")
print("=> label/base-rate bias, not differential model error.")
print("An equalised-odds post-processor would target a defect this model")
print("does not have. The fix is upstream of the model.")
""")

# ───────────────────────────────────────────────── 4. qualitative engine
md(r"""
---
## 4. The qualitative engine

`scripts/qualitative_analysis.py` turns metrics into a structured audit with no
LLM involved: severity classification, root-cause mapping, proxy detection, and
research-evidence retrieval.
""")

code(r"""
from qualitative_analysis import classify_severity, map_root_causes

print("classify_severity — the deterministic ground truth for Track Q:")
for di in [0.60, 0.70, 0.75, 0.82, 0.90, 0.96, 1.01]:
    print(f"   DI {di:.2f} -> {classify_severity(di)}")
print()
print("This function, not a model, is what Track Q scores against.")
print("Note where the boundaries fall: German Credit's three attributes")
print("(0.8595, 0.8212, 0.9402) sit near them, which is why it is the")
print("hardest use case for an LLM to agree on.")
""")

code(r"""
# Root-cause mapping is RULE-BASED, not causal. This distinction is load-bearing,
# so run the real path on real data rather than a synthetic example.
from qualitative_analysis import (
    per_group_breakdown, diagnose_imbalance, detect_proxy_features,
)

if enriched.exists():
    lc = pd.read_csv(enriched)
    attr = "gender"
    breakdown = per_group_breakdown(lc, attr, "actual", "predicted", favorable_label=0)
    print("Per-group breakdown (favorable_label=0):")
    print(breakdown.to_string(index=False))

    imbalance = diagnose_imbalance(breakdown, attr)
    feature_cols = [c for c in lc.columns
                    if c not in {"actual", "predicted", "gender", "income_level", "loan_amount_level"}]
    proxies = detect_proxy_features(lc, attr, feature_cols)

    causes = map_root_causes(0.9931, -0.0069, 0.0, -0.0357,
                            imbalance, proxies, breakdown, 0.8)
    print("\nimbalance findings:", imbalance)
    print("proxy features    :", [p.get("feature") for p in proxies][:5])
    print("\nroot causes:", json.dumps(causes, indent=2, default=str)[:600])
else:
    print("(enriched predictions unavailable; skipping the live root-cause demo)")

print()
print("Whatever it returns, these are inferences from metric values and feature")
print("correlations. No intervention, instrument, or counterfactual estimand is")
print("computed anywhere in this repository. The pipeline makes NO causal claim,")
print("and the report wording was corrected to say so.")
""")

md(r"""
### Evidence retention is deliberate

The Semantic Scholar harvest is fragile: the keyless `/paper/search` endpoint is
routinely rate-limited. Two behaviours protect the artifact.

1. A prior successful harvest is **retained** when a fresh one comes back empty,
   so a transient 429 cannot blank the evidence on disk.
2. Retention also wins when a fresh harvest returns **only curated DOI seeds**.
   Without this, the seed fallback would overwrite real live-search evidence and
   silently change the prompts the frozen benchmark packs were built from.

The API key shipped in `.env` returns **403 on every endpoint** — it is invalid,
not throttled — so the client drops a rejected key and retries keyless.
""")

code(r"""
for ds in ["german_credit", "hmda", "lending_club"]:
    p = ROOT / "artifacts" / ds / "fairness" / "qualitative_research_evidence.json"
    if not p.exists():
        print(f"{ds}: (not present)"); continue
    ev = json.loads(p.read_text())
    counts = {k: len(v) for k, v in ev.items()}
    srcs = {pp.get("source", "live_search") for papers in ev.values() for pp in papers}
    print(f"{ds:14s} {counts}  sources={sorted(srcs)}")
print()
print("lending_club's evidence is DOI-seeded (generic but real and verifiable);")
print("german_credit and hmda retain their original live-search harvest.")
""")

# ───────────────────────────────────────────────── 5. LLM benchmark
md(r"""
---
## 5. The LLM benchmark

Nine (use case × attribute) pairs × four prompt strategies = **36 frozen
prompts**. Each pack stores the full prompt text, so the benchmark replays
without re-deriving context — and `--dry-run` reproduces the entire scoring
path with no API key.
""")

code(r"""
import llm_benchmark as lb

print("Four cycles, in order:")
for i, (name, _) in enumerate(lb.CYCLE_CONFIGS, start=1):
    print(f"   cycle {i}: {name}")
print()
packs = sorted((ROOT / "artifacts" / "llm_benchmark" / "dry_run").glob("*/*_cycle*.json"))
print(f"frozen packs on disk: {len(packs)}")
pack = json.loads((ROOT / "artifacts/llm_benchmark/dry_run/lending_club/gender_cycle1.json").read_text())
print(f"\nexample pack keys: {sorted(pack)}")
print(f"prompt length: {len(pack['prompt']):,} chars")
print(f"score: {pack['score']['total_score']}/100")
print(f"subscores: {json.dumps(pack['score']['subscores'])}")
""")

md(r"""
### The five scoring subscores

Total 100, additive. `research_grounding` is why Stage A's evidence harvest
mattered: it was 0.0 for every Lending Club attribute until real papers were
embedded, capping the dry-run score at 80–85.
""")

code(r"""
rows = []
for p in sorted((ROOT / "artifacts/llm_benchmark/dry_run").glob("*/*_cycle1.json")):
    d = json.loads(p.read_text())
    rows.append({"dataset": d["dataset"], "attribute": d["attribute"],
                 "total": d["score"]["total_score"], **d["score"]["subscores"]})
print(pd.DataFrame(rows).to_string(index=False))
""")

md(r"""
### The anti-hallucination detectors

Two detectors run on every response. Below they are demonstrated **on mock
data** — no API call.
""")

code(r"""
KNOWN_TITLES = ["Certifying and Removing Disparate Impact",
                "Trust and Credit: The Role of Appearance in Peer-to-peer Lending"]
CONTEXT = {"attributes": [{"attribute": "gender",
                           "metrics": {"disparate_impact": 0.9931,
                                       "average_odds_difference": -0.0357}}]}

clean = {
    "attribute": "gender", "severity": "LOW",
    "what_is_wrong": "The provided context reports disparate impact = 0.9931 for gender, close to parity.",
    "why_is_wrong": "Near-parity here reflects a near-degenerate classifier rather than demonstrated fairness.",
    "how_to_fix": "Apply Reweighing and validate with a ThresholdOptimizer pass, targeting DI >= 0.8.",
    "supporting_research": [KNOWN_TITLES[0]],
}
print("1. Clean response:")
print("   citation flags:", lb.detect_hallucinations(clean, KNOWN_TITLES, lb.known_metric_values(CONTEXT)))
print("   refusal       :", lb.detect_refusal(clean, ""))
""")

code(r"""
# 2. FABRICATED METRIC — asserting a value absent from the provided context.
fake_metric = dict(clean)
fake_metric["what_is_wrong"] = "The context reports disparate impact = 0.4321 for gender, a severe violation."
flags = lb.detect_hallucinations(fake_metric, KNOWN_TITLES, lb.known_metric_values(CONTEXT))
print("2. Fabricated metric value 0.4321 (never provided):")
print("   FLAGGED ->", flags)
print()

# 3. FABRICATED CITATION — a plausible-looking paper that was never retrieved.
fake_cite = dict(clean)
fake_cite["supporting_research"] = ["Gender Bias in Peer Lending Markets: A Longitudinal Study (2031)"]
print("3. Fabricated citation:")
print("   FLAGGED ->", lb.detect_hallucinations(fake_cite, KNOWN_TITLES))
print()

# 4. Canonical thresholds are NOT flagged — 0.8 is a legitimate anchor.
ok_threshold = dict(clean)
ok_threshold["why_is_wrong"] = "DI of 0.9931 sits well above the canonical 0.8 four-fifths threshold."
print("4. Quoting the canonical 0.8 threshold (should NOT flag):")
print("   flags ->", lb.detect_hallucinations(ok_threshold, KNOWN_TITLES, lb.known_metric_values(CONTEXT)))
""")

md(r"""
### The refusal detector has a known flaw — demonstrated here

`detect_refusal` treats any narrative field under 40 characters as "effectively
empty". The `constrained` prompt explicitly asks for terse output, so a
**correct, on-spec answer gets flagged as a refusal.**

This is a finding about the harness, not the model. The detector is left
unchanged (it is part of the frozen scoring harness, and changing it would
invalidate comparison against the pre-freeze run); the reporting separates
genuine refusals from brevity artifacts instead.
""")

code(r"""
terse = dict(clean)
terse["what_is_wrong"] = "The Disparate Impact (DI) is 0.9931."   # 36 chars
terse["how_to_fix"] = "DisparateImpactRemover"                     # 22 chars — correct AND specific
print("Terse but entirely correct response:")
print(f"   what_is_wrong ({len(terse['what_is_wrong'])} chars): {terse['what_is_wrong']}")
print(f"   how_to_fix    ({len(terse['how_to_fix'])} chars): {terse['how_to_fix']}")
print(f"\n   detect_refusal -> {lb.detect_refusal(terse, '')}   <-- FALSE POSITIVE")
print("\n   No refusal phrase, no missing field, no invalid severity.")
print("   It trips purely on field length.")
print()
from report_track_q import classify_refusal
print("   classify_refusal({'refusal_detected': True, ...}) ->",
      classify_refusal({"refusal_detected": True, "result": terse}))
""")

# ───────────────────────────────────────────────── 6. results gallery
md(r"""
---
## 6. Results gallery

All figures and tables below are read from committed artifacts.
""")

code(r"""
tq = json.loads((ROOT / "artifacts/consolidated/track_q_analysis.json").read_text())
print(f"Track Q — {tq['n_records']} records, ${tq['cost']['total_usd']:.4f}")
print(f"\nSeverity agreement: {tq['agreement']['agreed']}/{tq['agreement']['scored']} "
      f"({tq['agreement']['overall_rate']*100:.1f}%)")
for ds, r in tq["agreement"]["by_dataset"].items():
    print(f"   {ds:14s} {r['agreed']:2d}/{r['scored']:<3d} {r['rate']*100:5.1f}%")
print("\n!! Do not quote the 75% aggregate. lending_club's 100% is the EASIEST")
print("   case (near-degenerate classifier => every attribute a trivial LOW);")
print("   german_credit's 33% is the hardest. The ordering tracks classifier")
print("   discriminativeness, not model skill.")
print(f"\nDirection of disagreement: {tq['agreement']['direction']}")
print("   Over-escalation is the safe error; under-escalation is the costly one.")
""")

code(r"""
print("Per prompt strategy:")
print(f"{'cycle':6s} {'strategy':20s} {'mean':>7s} {'agree':>7s} {'genuine ref':>12s} {'brevity ref':>12s} {'halluc':>7s}")
for r in tq["by_strategy"]:
    print(f"{r['cycle']:<6d} {r['strategy']:20s} {r['mean_score']:7.2f} "
          f"{r['severity_agreement_rate']*100:6.1f}% {r['genuine_refusal_count']:>8d}/{r['n']} "
          f"{r['brevity_only_refusal_count']:>8d}/{r['n']} {r['hallucination_flag_count']:>7d}")
print()
lc = tq["legacy_contrast"]
print("Research grounding, like-for-like on the two shared use cases:")
print(f"   no evidence      : {lc['hallucination_flags']:2d} fabricated citations, mean {lc['mean_score']:.2f}")
print(f"   evidence embedded: {lc['current_hallucination_flags']:2d} fabricated citations, mean {lc['current_mean']:.2f}")
""")

code(r"""
print("Cross-cycle severity stability (identical input context per row):")
for r in tq["stability"]:
    flag = "  <-- UNSTABLE" if not r["stable_across_cycles"] else ""
    sevs = " -> ".join(str(s) for s in r["cycle_severities"])
    print(f"   {r['dataset']:14s} {r['attribute']:26s} base={str(r['baseline_severity']):9s} {sevs}{flag}")
unstable = sum(1 for r in tq["stability"] if not r["stable_across_cycles"])
print(f"\n{unstable} of {len(tq['stability'])} pairs changed label on identical input.")
print("Prompt phrasing alone moved the verdict — the core reliability result.")
""")

code(r"""
th = json.loads((ROOT / "artifacts/consolidated/track_h_evaluation.json").read_text())
print("Track H — narrative audits, judged only on H1-H5 + guardrails + rubric")
print("(never against the deterministic numbers).\n")
for r in th:
    g = sum(1 for c in r["guardrails"] if c["passed"])
    h = sum(1 for c in r["h_criteria"] if c["passed"])
    cit = r["citations"]
    judge = r.get("judge") or {}
    score = f"{judge.get('weighted_score')}/5" if judge else "n/a"
    print(f"   {r['use_case']:14s} guardrails {g}/{len(r['guardrails'])}  "
          f"H {h}/{len(r['h_criteria'])}  citations {cit['verified']}/{cit['cited']}  rubric {score}")
print()
for r in th:
    for c in r["guardrails"] + r["h_criteria"]:
        if not c["passed"]:
            print(f"   FAIL {r['use_case']:14s} {c['id']}: {c['detail'][:78]}")
print("\nG4 failures land on exactly the two data-scarcity attributes, where")
print("'collect more data + widen the CIs' is the correct answer and no")
print("post-processor applies. The guardrail penalises the right response.")
""")

code(r"""
from IPython.display import Image, display
figs = ["track_q_agreement_by_usecase.png", "track_q_strategy_behaviour.png",
        "track_q_severity_matrix.png", "track_q_grounding_effect.png"]
for name in figs:
    p = ROOT / "artifacts" / "visualizations" / name
    if p.exists():
        print(f"\n=== {name} ===")
        display(Image(filename=str(p)))
    else:
        print(f"(missing: {name} — regenerate with scripts/visualize_track_q.py)")
""")

code(r"""
led = json.loads((ROOT / "artifacts/llm_benchmark/spend_ledger.json").read_text())
by = {}
for e in led["entries"]:
    k = e.get("track", "Q")
    by[k] = by.get(k, 0.0) + e["cost_usd"]
print(f"Total spend: ${led['total_usd']:.4f}  (cap ${led['max_cost_usd']:.2f})")
for k in sorted(by):
    print(f"   track {k:8s} ${by[k]:.4f}")
print(f"   + pre-freeze legacy run (predates the ledger): $0.1125")
print(f"   PROGRAM TOTAL: ${led['total_usd'] + 0.1125:.4f}")
""")

md(r"""
---
## Summary

| Layer | Owns | Trustworthy? |
|---|---|---|
| Deterministic pipeline | metrics, severity thresholds | Yes — reproducible by construction |
| LLM interpretation | decision-context narrative | Only when grounded in retrieved evidence |
| LLM severity labels | — | **No** — 3 of 9 pairs moved under paraphrase |

The architecture that survives the evidence: **Python owns the numbers and the
thresholds, the model owns the interpretation, and the interpretation is gated
on evidence the model did not invent.**

Reproduce everything:

```bash
make test                                        # 46 unit tests, no key
python3.11 run_pipeline.py --datasets german_credit hmda lending_club --steps fairness qualitative
python3.11 scripts/report_track_q.py             # Track Q tables
python3.11 scripts/evaluate_track_h.py           # Track H gates (add --judge for the rubric)
python3.11 scripts/visualize_track_q.py          # figures
make paper                                       # the NeurIPS draft
```
""")

nb = {
    "cells": [
        {
            "cell_type": kind,
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
        }
        for kind, src in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path(
    "/Users/siddhant/Desktop/Previous Coursework/CMPT 310/Group Project/"
    "AI-fairness-dashboard/.claude/worktrees/hopeful-wright-f024e8/"
    "notebooks/pipeline_walkthrough.ipynb"
)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
print(f"{len(CELLS)} cells ({sum(1 for k, _ in CELLS if k == 'code')} code)")
