"""
openai_hybrid_self_improve.py

Deepnote-friendly runner for hybrid self-improvement:
- generator (OpenAI fairness output)
- external judge (OpenAI rubric scoring)
- self-reflection prompt updates

It also performs multi-run experiment tracking, derives refined guardrails from
judge failures, and runs a validation pass with the best configuration.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from llm_benchmark_common import build_context_and_baseline, score_llm_output
from openai_fairness_analysis import (
    OUTPUT_SCHEMA,
    build_prompt,
    build_refinement_prompt,
    generate_comparison,
    generate_llm_report,
)
from scholarly_evidence import format_evidence_for_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
CAD_TO_USD_DEFAULT = 0.74
PRICE_INPUT_PER_M = 10.00
PRICE_OUTPUT_PER_M = 40.00

DATASET_PRESETS: dict[str, dict[str, Any]] = {
    "german_credit": {
        "dataset_name": "German Credit",
        "predictions": ARTIFACTS / "german_credit" / "models" / "classification_predictions.csv",
        "fairness_csv": ARTIFACTS / "german_credit" / "fairness" / "fairness_metrics.csv",
        "qualitative_report": ARTIFACTS / "german_credit" / "fairness" / "qualitative_report.md",
        "target": "actual",
        "pred_col": "predicted",
        "favorable_label": 1,
        "protected_attrs": {
            "Sex_original": "male",
            "AgeGroup_original": "40_plus",
            "foreign_worker_original": "0",
        },
    },
    "hmda": {
        "dataset_name": "HMDA Mortgage Lending (Georgia)",
        "predictions": ARTIFACTS / "hmda" / "models" / "classification_predictions.csv",
        "fairness_csv": ARTIFACTS / "hmda" / "fairness" / "fairness_metrics.csv",
        "qualitative_report": ARTIFACTS / "hmda" / "fairness" / "qualitative_report.md",
        "target": "actual",
        "pred_col": "predicted",
        "favorable_label": 1,
        "protected_attrs": {
            "race": "White",
            "sex": "Male",
        },
    },
}

JUDGE_SCHEMA = {
    "name": "fairness_judge_feedback",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "dimension_scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dimension": {"type": "string"},
                        "score_1_to_5": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["dimension", "score_1_to_5", "rationale"],
                    "additionalProperties": False,
                },
            },
            "overall_score_100": {"type": "number"},
            "failures": {"type": "array", "items": {"type": "string"}},
            "prompt_deltas": {"type": "array", "items": {"type": "string"}},
            "guardrail_updates": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "dimension_scores",
            "overall_score_100",
            "failures",
            "prompt_deltas",
            "guardrail_updates",
        ],
        "additionalProperties": False,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _estimate_cost(prompt: str, max_output_tokens: int) -> float:
    input_tokens = math.ceil(len(prompt) / 4)
    input_cost = (input_tokens / 1_000_000) * PRICE_INPUT_PER_M
    output_cost = (max_output_tokens / 1_000_000) * PRICE_OUTPUT_PER_M
    return input_cost + output_cost


def _call_openai_json(
    prompt: str,
    api_key: str,
    model: str,
    schema: dict[str, Any],
    max_output_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a rigorous evaluator. Follow schema exactly."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=max_output_tokens,
        response_format={"type": "json_schema", "json_schema": schema},
    )
    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_cost = (input_tokens / 1_000_000) * PRICE_INPUT_PER_M + (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_M
    raw = (response.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)
    return json.loads(raw), {
        "wall_clock_s": round(time.time() - t0, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "total_cost_usd": round(total_cost, 6),
        "model": model,
    }


def _build_style_hint(prompt_style: str) -> str:
    hints = {
        "baseline": "Keep structure stable and complete.",
        "actionability": "Prioritize measurable remediation steps with owners and thresholds.",
        "evidence_strict": "Only make claims that are tied to supplied evidence and context.",
        "concise_ops": "Be concise and operationally focused; avoid generic prose.",
    }
    return hints.get(prompt_style, hints["baseline"])


def _build_judge_prompt(
    generated_output: dict[str, Any],
    baseline_payload: dict[str, Any],
    rubric: dict[str, Any],
) -> str:
    return (
        "Evaluate the fairness audit output using this rubric and baseline context.\n\n"
        f"Rubric JSON:\n{json.dumps(rubric, indent=2)}\n\n"
        f"Baseline JSON:\n{json.dumps(baseline_payload, indent=2)}\n\n"
        f"Generated output JSON:\n{json.dumps(generated_output, indent=2)}\n\n"
        "Scoring requirements:\n"
        "- Score each rubric dimension from 1 to 5.\n"
        "- Provide overall_score_100 from 0 to 100.\n"
        "- List concrete failures and prompt_deltas for next cycle.\n"
        "- Propose guardrail_updates that reduce hallucination and vague mitigations.\n"
    )


def _build_guardrail_text(guardrails: dict[str, Any]) -> str:
    return (
        "Guardrail profile:\n"
        f"{json.dumps(guardrails, indent=2)}\n"
        "Strictly follow required fields and avoid forbidden patterns."
    )


def _apply_guardrail_checks(result: dict[str, Any], guardrails: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    forbidden = [s.lower() for s in guardrails.get("forbidden_patterns", [])]
    required = guardrails.get("required_per_attribute", [])
    min_cites = int(guardrails.get("min_supporting_research_per_attribute", 0) or 0)
    must_measurable = bool(guardrails.get("must_include_measurable_target", False))
    keywords = [k.lower() for k in guardrails.get("mitigation_must_include_keywords", [])]

    for row in result.get("qualitative", []):
        for field in required:
            if not row.get(field):
                violations.append(f"{row.get('attribute', 'unknown')}: missing field {field}")
        how = (row.get("how_to_fix", "") or "").lower()
        for pat in forbidden:
            if pat in how:
                violations.append(f"{row.get('attribute', 'unknown')}: forbidden pattern '{pat}'")
        cites = row.get("supporting_research", [])
        if len(cites) < min_cites:
            violations.append(f"{row.get('attribute', 'unknown')}: insufficient supporting_research")
        if keywords and not any(k in how for k in keywords):
            violations.append(f"{row.get('attribute', 'unknown')}: mitigation lacks required algorithm keyword")
        if must_measurable and not any(tok in how for tok in ["%", "kpi", "target", "threshold", "reduce", "increase"]):
            violations.append(f"{row.get('attribute', 'unknown')}: mitigation not measurable")
    compliance = max(0.0, 100.0 - (10.0 * len(violations)))
    return {"compliance_score_100": compliance, "violations": violations}


def _derive_refined_guardrails(all_judge_feedback: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    failure_counter: dict[str, int] = {}
    for item in all_judge_feedback:
        for failure in item.get("failures", []):
            normalized = failure.strip().lower()
            failure_counter[normalized] = failure_counter.get(normalized, 0) + 1
    top_failures = sorted(failure_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    high_freq = [f for f, count in top_failures if count >= 2]

    refined = dict(baseline)
    refined["name"] = "fairness_guardrails_refined_v2"
    refined["high_frequency_failures"] = [{"failure": f, "count": c} for f, c in top_failures]
    refined["forbidden_patterns"] = sorted(set(baseline.get("forbidden_patterns", []) + [f for f in high_freq if len(f) < 60]))
    refined["min_supporting_research_per_attribute"] = max(
        int(baseline.get("min_supporting_research_per_attribute", 1)),
        2,
    )
    refined["must_include_measurable_target"] = True
    return refined


def _run_single_experiment(
    exp: dict[str, Any],
    context_payload: dict[str, Any],
    baseline_payload: dict[str, Any],
    research_evidence: list[dict[str, Any]],
    dataset_cfg: dict[str, Any],
    rubric: dict[str, Any],
    guardrails: dict[str, Any],
    out_dir: Path,
    api_key: str,
    generator_model: str,
    judge_model: str,
    cycles: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    run_dir = out_dir / exp["id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    style_hint = _build_style_hint(exp.get("prompt_style", "baseline"))
    research_block = format_evidence_for_prompt(research_evidence)
    prompt = build_prompt(context_payload, dataset_cfg["dataset_name"], research_block) + f"\n\nStyle hint: {style_hint}\n"
    prompt += "\n" + _build_guardrail_text(guardrails)

    projected = _estimate_cost(prompt, max_output_tokens=max_output_tokens) * cycles * 2.0
    cycle_outputs: list[dict[str, Any]] = []
    cycle_scores: list[dict[str, Any]] = []
    judge_feedback_log: list[dict[str, Any]] = []
    prompt_log: list[str] = [prompt]
    total_cost = 0.0
    final_result: dict[str, Any] = {}

    for cycle_idx in range(1, cycles + 1):
        generated, gen_usage = _call_openai_json(
            prompt=prompt,
            api_key=api_key,
            model=generator_model,
            schema=OUTPUT_SCHEMA,
            max_output_tokens=max_output_tokens,
        )
        deterministic_score = score_llm_output(generated, baseline_payload)
        guardrail_score = _apply_guardrail_checks(generated, guardrails)

        judge_prompt = _build_judge_prompt(generated, baseline_payload, rubric)
        judge, judge_usage = _call_openai_json(
            prompt=judge_prompt,
            api_key=api_key,
            model=judge_model,
            schema=JUDGE_SCHEMA,
            max_output_tokens=1500,
        )
        total_cost += gen_usage["total_cost_usd"] + judge_usage["total_cost_usd"]
        hybrid_score = round((0.45 * deterministic_score["total_score"]) + (0.45 * float(judge["overall_score_100"])) + (0.10 * guardrail_score["compliance_score_100"]), 2)
        cycle_scores.append(
            {
                "cycle": cycle_idx,
                "score": {
                    "total_score": hybrid_score,
                    "subscores": deterministic_score["subscores"],
                    "summary": f"hybrid={hybrid_score:.1f}, judge={judge['overall_score_100']:.1f}, guardrails={guardrail_score['compliance_score_100']:.1f}",
                },
            }
        )
        cycle_outputs.append(
            {
                "cycle": cycle_idx,
                "generator_usage": gen_usage,
                "judge_usage": judge_usage,
                "deterministic_score": deterministic_score,
                "judge_feedback": judge,
                "guardrail_check": guardrail_score,
                "result": generated,
            }
        )
        judge_feedback_log.append(judge)
        final_result = generated
        if cycle_idx < cycles:
            deltas = "\n".join(f"- {d}" for d in judge.get("prompt_deltas", []))
            updates = "\n".join(f"- {u}" for u in judge.get("guardrail_updates", []))
            prompt = build_refinement_prompt(
                context_payload=context_payload,
                previous_output=generated,
                dataset_name=dataset_cfg["dataset_name"],
                cycle_idx=cycle_idx + 1,
                research_context=research_block,
            )
            prompt += (
                "\n\nJudge feedback to enforce in next cycle:\n"
                f"{deltas}\n\n"
                "Guardrail update hints:\n"
                f"{updates}\n\n"
                + _build_guardrail_text(guardrails)
            )
            prompt_log.append(prompt)

    report_path = run_dir / "llm_fairness_report.md"
    report_path.write_text(
        generate_llm_report(
            result=final_result,
            context_payload=context_payload,
            cycle_scores=cycle_scores,
            dataset_name=dataset_cfg["dataset_name"],
            model=generator_model,
            usage_stats={
                "model": generator_model,
                "attempts": 1,
                "wall_clock_s": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_cost_usd": 0,
                "output_cost_usd": 0,
                "total_cost_usd": total_cost,
                "price_input_per_m": PRICE_INPUT_PER_M,
                "price_output_per_m": PRICE_OUTPUT_PER_M,
                "projection": {"projected_total_cost_usd": projected, "projected_input_tokens": 0, "projected_output_tokens": max_output_tokens},
                "max_cost_usd": 0,
            },
        ),
        encoding="utf-8",
    )
    comparison_path = run_dir / "benchmark_comparison.md"
    comparison_path.write_text(
        generate_comparison(
            result=final_result,
            baseline_payload=baseline_payload,
            our_qualitative_report=Path(dataset_cfg["qualitative_report"]),
            dataset_name=dataset_cfg["dataset_name"],
            cycle_scores=cycle_scores,
            usage_stats=None,
        ),
        encoding="utf-8",
    )

    (run_dir / "llm_prompt.txt").write_text("\n\n".join(prompt_log), encoding="utf-8")
    (run_dir / "cycles.json").write_text(json.dumps(cycle_outputs, indent=2), encoding="utf-8")
    (run_dir / "llm_raw_response.json").write_text(
        json.dumps(
            {
                "result": final_result,
                "cycles": cycle_outputs,
                "judge_feedback": judge_feedback_log,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "run_id": exp["id"],
        "dataset": dataset_cfg["dataset_name"],
        "final_hybrid_score": cycle_scores[-1]["score"]["total_score"],
        "first_hybrid_score": cycle_scores[0]["score"]["total_score"],
        "cycle_gain": round(cycle_scores[-1]["score"]["total_score"] - cycle_scores[0]["score"]["total_score"], 2),
        "projected_cost_usd": round(projected, 4),
        "actual_cost_usd": round(total_cost, 4),
        "run_dir": str(run_dir),
    }, judge_feedback_log


def run_hybrid_plan(config_path: Path, rubric_path: Path, guardrails_path: Path, out_root: Path) -> Path:
    cfg = _load_json(config_path)
    rubric = _load_json(rubric_path)
    guardrails = _load_json(guardrails_path)
    dataset_key = cfg.get("pilot_dataset", "german_credit")
    if dataset_key not in DATASET_PRESETS:
        raise ValueError(f"Unknown pilot dataset '{dataset_key}'")
    dataset_cfg = DATASET_PRESETS[dataset_key]

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    predictions_df = pd.read_csv(dataset_cfg["predictions"])
    fairness_df = pd.read_csv(dataset_cfg["fairness_csv"])
    context_payload, baseline_payload, research_evidence = build_context_and_baseline(
        predictions_df=predictions_df,
        fairness_df=fairness_df,
        protected_configs=dataset_cfg["protected_attrs"],
        target_col=dataset_cfg["target"],
        pred_col=dataset_cfg["pred_col"],
        favorable_label=dataset_cfg["favorable_label"],
        dataset_name=dataset_cfg["dataset_name"],
    )

    run_stamp = _now_id()
    run_root = out_root / dataset_key / "fairness" / "openai_hybrid" / run_stamp
    run_root.mkdir(parents=True, exist_ok=True)

    cad_to_usd = float(cfg.get("cad_to_usd", CAD_TO_USD_DEFAULT))
    budget_cad = float(cfg.get("budget_cad", 50.0))
    reserve_cad = float(cfg.get("budget_reserve_cad", 0.0))
    effective_budget_cad = max(0.0, budget_cad - reserve_cad)
    budget_usd = effective_budget_cad * cad_to_usd
    registry: list[dict[str, Any]] = []
    all_judge_feedback: list[dict[str, Any]] = []
    spent = 0.0

    for exp in cfg.get("experiments", []):
        if spent >= budget_usd:
            log.warning("Budget exhausted, stopping experiment loop.")
            break
        summary, feedback = _run_single_experiment(
            exp=exp,
            context_payload=context_payload,
            baseline_payload=baseline_payload,
            research_evidence=research_evidence,
            dataset_cfg=dataset_cfg,
            rubric=rubric,
            guardrails=guardrails,
            out_dir=run_root,
            api_key=api_key,
            generator_model=cfg.get("generator_model", "o3"),
            judge_model=cfg.get("judge_model", "o3"),
            cycles=int(cfg.get("cycles", 3)),
            max_output_tokens=int(cfg.get("max_output_tokens", 3000)),
        )
        spent += summary["actual_cost_usd"]
        summary["budget_remaining_usd"] = round(max(0.0, budget_usd - spent), 4)
        registry.append(summary)
        all_judge_feedback.extend(feedback)

    refined_guardrails = _derive_refined_guardrails(all_judge_feedback, guardrails)
    (run_root / "guardrails_refined_v2.json").write_text(json.dumps(refined_guardrails, indent=2), encoding="utf-8")

    if not registry:
        raise RuntimeError("No experiments ran; check budget and config.")

    best = sorted(registry, key=lambda r: r["final_hybrid_score"], reverse=True)[0]
    validation_cfg = {"id": "validation_best_v2", "prompt_style": "actionability"}
    validation_summary, _ = _run_single_experiment(
        exp=validation_cfg,
        context_payload=context_payload,
        baseline_payload=baseline_payload,
        research_evidence=research_evidence,
        dataset_cfg=dataset_cfg,
        rubric=rubric,
        guardrails=refined_guardrails,
        out_dir=run_root,
        api_key=api_key,
        generator_model=cfg.get("generator_model", "o3"),
        judge_model=cfg.get("judge_model", "o3"),
        cycles=int(cfg.get("cycles", 3)),
        max_output_tokens=int(cfg.get("max_output_tokens", 3000)),
    )

    decision = {
        "promote_refined_default": validation_summary["final_hybrid_score"] >= best["final_hybrid_score"],
        "baseline_best_run_id": best["run_id"],
        "baseline_best_score": best["final_hybrid_score"],
        "validation_run_id": validation_summary["run_id"],
        "validation_score": validation_summary["final_hybrid_score"],
        "delta": round(validation_summary["final_hybrid_score"] - best["final_hybrid_score"], 2),
        "budget_usd": round(budget_usd, 2),
        "total_spent_usd": round(spent + validation_summary["actual_cost_usd"], 4),
    }

    (run_root / "run_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    pd.DataFrame(registry).to_csv(run_root / "run_registry.csv", index=False)
    (run_root / "validation_summary.json").write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")
    (run_root / "promotion_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (run_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_path": str(config_path),
                "rubric_path": str(rubric_path),
                "guardrails_path": str(guardrails_path),
                "pilot_dataset": dataset_key,
                "budget_cad": round(budget_cad, 2),
                "budget_reserve_cad": round(reserve_cad, 2),
                "effective_budget_cad": round(effective_budget_cad, 2),
                "budget_usd": round(budget_usd, 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Hybrid self-improvement run completed: %s", run_root)
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI hybrid self-improvement runner (Deepnote-friendly)")
    parser.add_argument("--config", default="configs/openai_hybrid_pilot.json")
    parser.add_argument("--rubric", default="configs/report_quality_rubric.json")
    parser.add_argument("--guardrails", default="configs/guardrails_baseline.json")
    parser.add_argument("--out_root", default="artifacts")
    args = parser.parse_args()

    run_hybrid_plan(
        config_path=(ROOT / args.config).resolve(),
        rubric_path=(ROOT / args.rubric).resolve(),
        guardrails_path=(ROOT / args.guardrails).resolve(),
        out_root=(ROOT / args.out_root).resolve(),
    )


if __name__ == "__main__":
    main()
