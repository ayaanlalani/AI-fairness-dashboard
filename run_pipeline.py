"""
run_pipeline.py

Central orchestrator for the AI Fairness Dashboard.
Runs clean -> train -> fairness -> qualitative -> llm_benchmark for each dataset,
then consolidates results across all datasets.

Usage:
  python3 run_pipeline.py                                    # all datasets, all steps
  python3 run_pipeline.py --datasets german_credit hmda      # specific datasets
  python3 run_pipeline.py --steps fairness qualitative       # specific steps
  python3 run_pipeline.py --datasets hmda --steps qualitative
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
PYTHON = f'"{sys.executable}"'

# ═══════════════════════════════════════════════════════════════════════
#  Dataset registry
#
#  Each entry defines the working directory and the shell commands for
#  each pipeline step.  The "qualitative" step uses the shared module
#  at scripts/qualitative_analysis.py.
# ═══════════════════════════════════════════════════════════════════════

ARTIFACTS = ROOT / "artifacts"
DEFAULT_STEP_TIMEOUT_S = 300
TRAIN_STEP_TIMEOUT_S = 1200
FAIRNESS_STEP_TIMEOUT_S = 1200
LLM_BENCHMARK_TIMEOUT_S = 1200
FOUR_CYCLE_BENCHMARK_TIMEOUT_S = 2400  # 4 cycles × N attrs × retries

DATASETS: dict[str, dict] = {
    "german_credit": {
        "dir": "german_credit_dataset",
        "artifact_dir": "german_credit",
        "steps": {
            "clean": f"{PYTHON} clean-germanCreditData.py",
            "train": f"{PYTHON} scripts/train_models.py",
            "fairness": f"{PYTHON} scripts/compute_fairness.py",
            "qualitative": (
                f"{PYTHON} ../scripts/qualitative_analysis.py"
                " --predictions metrics/classification_predictions.csv"
                " --fairness_csv metrics/fairness/fairness_metrics.csv"
                " --target actual --pred_col predicted --favorable_label 1"
                " --protected_attrs Sex_original,AgeGroup_original,foreign_worker_original"
                " --out_dir metrics/fairness"
                " --dataset_name 'German Credit'"
            ),
            "llm_benchmark": (
                "mkdir -p metrics/fairness/gemini metrics/fairness/openai && "
                "gemini_status=0; openai_status=0; "
                f"{PYTHON} ../scripts/llm_fairness_analysis.py"
                " --predictions metrics/classification_predictions.csv"
                " --fairness_csv metrics/fairness/fairness_metrics.csv"
                " --qualitative_report metrics/fairness/qualitative_report.md"
                " --target actual --pred_col predicted --favorable_label 1"
                " --protected_attrs 'Sex_original:male,AgeGroup_original:40_plus,foreign_worker_original:0'"
                " --out_dir metrics/fairness/gemini"
                " --dataset_name 'German Credit'"
                " || gemini_status=$?; "
                f"{PYTHON} ../scripts/openai_fairness_analysis.py"
                " --predictions metrics/classification_predictions.csv"
                " --fairness_csv metrics/fairness/fairness_metrics.csv"
                " --qualitative_report metrics/fairness/qualitative_report.md"
                " --target actual --pred_col predicted --favorable_label 1"
                " --protected_attrs 'Sex_original:male,AgeGroup_original:40_plus,foreign_worker_original:0'"
                " --out_dir metrics/fairness/openai"
                " --dataset_name 'German Credit'"
                " --max_cost_usd 37.00"
                " || openai_status=$?; "
                "test $gemini_status -eq 0 -a $openai_status -eq 0"
            ),
            "benchmark": (
                f"{PYTHON} ../scripts/llm_benchmark.py"
                " --predictions metrics/classification_predictions.csv"
                " --fairness_csv metrics/fairness/fairness_metrics.csv"
                " --qualitative_report metrics/fairness/qualitative_report.md"
                " --target actual --pred_col predicted --favorable_label 1"
                " --protected_attrs 'Sex_original:male,AgeGroup_original:40_plus,foreign_worker_original:0'"
                " --dataset_name 'German Credit'"
                " --dataset_key german_credit"
                " --out_root ../artifacts/llm_benchmark"
            ),
        },
    },
    "hmda": {
        "dir": "hmda_dataset",
        "artifact_dir": "hmda",
        "steps": {
            "clean": f"{PYTHON} clean_hmda.py",
            "train": f"{PYTHON} scripts/train_models.py",
            "fairness": f"{PYTHON} scripts/compute_fairness.py",
            "qualitative": (
                f"{PYTHON} ../scripts/qualitative_analysis.py"
                " --predictions metrics/classification_predictions.csv"
                " --fairness_csv metrics/fairness/fairness_metrics.csv"
                " --target actual --pred_col predicted --favorable_label 1"
                " --protected_attrs race,sex,age_group"
                " --out_dir metrics/fairness"
                " --dataset_name 'HMDA Mortgage Lending (Georgia)'"
            ),
            "llm_benchmark": (
                "mkdir -p metrics/fairness/gemini metrics/fairness/openai && "
                "gemini_status=0; openai_status=0; "
                f"{PYTHON} ../scripts/llm_fairness_analysis.py"
                " --predictions metrics/classification_predictions.csv"
                " --fairness_csv metrics/fairness/fairness_metrics.csv"
                " --qualitative_report metrics/fairness/qualitative_report.md"
                " --target actual --pred_col predicted --favorable_label 1"
                " --protected_attrs 'race:White,sex:Male,age_group:mid'"
                " --out_dir metrics/fairness/gemini"
                " --dataset_name 'HMDA Mortgage Lending (Georgia)'"
                " || gemini_status=$?; "
                f"{PYTHON} ../scripts/openai_fairness_analysis.py"
                " --predictions metrics/classification_predictions.csv"
                " --fairness_csv metrics/fairness/fairness_metrics.csv"
                " --qualitative_report metrics/fairness/qualitative_report.md"
                " --target actual --pred_col predicted --favorable_label 1"
                " --protected_attrs 'race:White,sex:Male,age_group:mid'"
                " --out_dir metrics/fairness/openai"
                " --dataset_name 'HMDA Mortgage Lending (Georgia)'"
                " --max_cost_usd 37.00"
                " || openai_status=$?; "
                "test $gemini_status -eq 0 -a $openai_status -eq 0"
            ),
            "benchmark": (
                f"{PYTHON} ../scripts/llm_benchmark.py"
                " --predictions metrics/classification_predictions.csv"
                " --fairness_csv metrics/fairness/fairness_metrics.csv"
                " --qualitative_report metrics/fairness/qualitative_report.md"
                " --target actual --pred_col predicted --favorable_label 1"
                " --protected_attrs 'race:White,sex:Male,age_group:mid'"
                " --dataset_name 'HMDA Mortgage Lending (Georgia)'"
                " --dataset_key hmda"
                " --out_root ../artifacts/llm_benchmark"
            ),
        },
    },
    "healthcare_insurance": {
        "dir": "Healthcare-insurance-dataset",
        "steps": {
            "clean": f"{PYTHON} clean_insurance.py --input_path insurance.csv --out_dir processed",
            "train": f"{PYTHON} scripts/train_models.py --data_dir processed",
            "fairness": f"{PYTHON} scripts/compute_fairness.py --data_dir processed --predictions_dir metrics --out_dir metrics/fairness",
            "qualitative": (
                f"{PYTHON} ../scripts/qualitative_analysis.py"
                " --predictions metrics/classification_predictions.csv"
                " --fairness_csv metrics/fairness/fairness_metrics.csv"
                " --target actual --pred_col predicted --favorable_label 1"
                " --protected_attrs sex,smoker,region,age_bucket"
                " --out_dir metrics/fairness"
                " --dataset_name 'Healthcare Insurance'"
            ),
        },
    },
    "diabetes": {
        "dir": "diabetes_dataset",
        "steps": {
            "clean": f"{PYTHON} clean_diabetes_data.py",
            "train": f"{PYTHON} scripts/train_models.py",
            "fairness": f"{PYTHON} compute_fairness.py",
            "qualitative": (
                f"{PYTHON} ../scripts/qualitative_analysis.py"
                " --predictions metrics/classification_predictions.csv"
                " --fairness_csv metrics/fairness/fairness_metrics.csv"
                " --target actual --pred_col predicted --favorable_label 1"
                " --protected_attrs older_age,high_pregnancy_count"
                " --out_dir metrics/fairness"
                " --dataset_name 'Diabetes Prediction'"
            ),
        },
    },
}

STEP_ORDER = ["clean", "train", "fairness", "qualitative", "llm_benchmark", "benchmark", "visualize"]


# ═══════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════

def run_step(dataset_name: str, step_name: str, cmd: str, cwd: Path) -> bool:
    """Run a single pipeline step. Returns True on success."""
    header = f"[{dataset_name} / {step_name}]"
    if step_name == "benchmark":
        timeout_s = FOUR_CYCLE_BENCHMARK_TIMEOUT_S
    elif step_name == "llm_benchmark":
        timeout_s = LLM_BENCHMARK_TIMEOUT_S
    elif step_name == "fairness":
        timeout_s = FAIRNESS_STEP_TIMEOUT_S
    elif step_name == "train":
        timeout_s = TRAIN_STEP_TIMEOUT_S
    else:
        timeout_s = DEFAULT_STEP_TIMEOUT_S
    log.info(f"{header} Starting...")
    log.info(f"{header} cwd={cwd}")
    log.info(f"{header} cmd={cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                log.info(f"{header}   {line}")
        if result.returncode != 0:
            log.error(f"{header} FAILED (exit code {result.returncode})")
            if result.stderr.strip():
                # Filter out aif360 import warnings
                for line in result.stderr.strip().split("\n"):
                    if "WARNING:root:" not in line and "pip install" not in line:
                        log.error(f"{header}   {line}")
            return False
        log.info(f"{header} OK")
        return True
    except subprocess.TimeoutExpired:
        log.error(f"{header} TIMEOUT (>{timeout_s}s)")
        return False
    except Exception as e:
        log.error(f"{header} ERROR: {e}")
        return False


def _copy_artifacts(datasets_run: list[str]) -> None:
    """Copy per-dataset artifacts into the central artifacts/ directory."""
    import shutil

    for ds_name in datasets_run:
        ds_config = DATASETS[ds_name]
        ds_dir = ROOT / ds_config["dir"]
        art_dir = ARTIFACTS / ds_config.get("artifact_dir", ds_name)

        fairness_src = ds_dir / "metrics" / "fairness"
        fairness_dst = art_dir / "fairness"
        fairness_dst.mkdir(parents=True, exist_ok=True)

        models_dst = art_dir / "models"
        models_dst.mkdir(parents=True, exist_ok=True)

        for f in fairness_src.rglob("*"):
            if f.is_file():
                rel = f.relative_to(fairness_src)
                dest = fairness_dst / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)

        metrics_dir = ds_dir / "metrics"
        for pattern in ["classification_predictions.csv", "classification_metrics.json",
                        "logistic_regression_*.json", "random_forest_*.json"]:
            for f in metrics_dir.glob(pattern):
                if f.is_file():
                    shutil.copy2(f, models_dst / f.name)

    log.info(f"  Copied per-dataset artifacts to {ARTIFACTS}")


def consolidate(datasets_run: list[str]) -> None:
    """Copy artifacts and merge cross-dataset reports."""
    import pandas as pd

    log.info("Consolidating cross-dataset results...")

    cons_dir = ARTIFACTS / "consolidated"
    cons_dir.mkdir(parents=True, exist_ok=True)

    # Also keep poster_assets for backward compat
    legacy_dir = ROOT / "poster_assets"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    # ── copy per-dataset artifacts ────────────────────────────────
    _copy_artifacts(datasets_run)

    # ── merge fairness_metrics.csv ─────────────────────────────────
    all_metrics = []
    for ds_name in datasets_run:
        ds_dir = ROOT / DATASETS[ds_name]["dir"]
        csv_path = ds_dir / "metrics" / "fairness" / "fairness_metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.insert(0, "Dataset", ds_name)
            all_metrics.append(df)

    if all_metrics:
        consolidated = pd.concat(all_metrics, ignore_index=True)
        for dest in [cons_dir, legacy_dir]:
            consolidated.to_csv(dest / "consolidated_fairness_metrics.csv", index=False)
        log.info(f"  Saved consolidated metrics ({len(consolidated)} rows)")

    # ── merge qualitative reports ──────────────────────────────────
    combined_lines = ["# Cross-Dataset Qualitative Fairness Analysis", ""]
    for ds_name in datasets_run:
        ds_dir = ROOT / DATASETS[ds_name]["dir"]
        report_path = ds_dir / "metrics" / "fairness" / "qualitative_report.md"
        if report_path.exists():
            content = report_path.read_text(encoding="utf-8")
            combined_lines.append(content)
            combined_lines.append("")
            combined_lines.append("---")
            combined_lines.append("")

    if len(combined_lines) > 2:
        text = "\n".join(combined_lines)
        for dest in [cons_dir, legacy_dir]:
            (dest / "cross_dataset_analysis.md").write_text(text, encoding="utf-8")
        log.info("  Saved cross-dataset qualitative analysis")

    # ── merge LLM benchmark comparisons ───────────────────────────
    for provider in ["gemini", "openai"]:
        benchmark_lines = [f"# Cross-Dataset {provider.title()} LLM Benchmark Comparison", ""]
        for ds_name in datasets_run:
            bench_path = ARTIFACTS / ds_name / "fairness" / provider / "benchmark_comparison.md"
            if not bench_path.exists():
                ds_dir = ROOT / DATASETS[ds_name]["dir"]
                bench_path = ds_dir / "metrics" / "fairness" / provider / "benchmark_comparison.md"
            if bench_path.exists():
                content = bench_path.read_text(encoding="utf-8")
                benchmark_lines.append(content)
                benchmark_lines.append("")
                benchmark_lines.append("---")
                benchmark_lines.append("")

        if len(benchmark_lines) > 2:
            text = "\n".join(benchmark_lines)
            for dest in [cons_dir, legacy_dir]:
                (dest / f"llm_benchmark_comparison_{provider}.md").write_text(text, encoding="utf-8")
            log.info(f"  Saved cross-dataset {provider} LLM benchmark comparison")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Fairness Dashboard — Central Pipeline")
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help=f"Datasets to run (default: all). Choices: {list(DATASETS.keys())}",
    )
    parser.add_argument(
        "--steps", nargs="*", default=None,
        help=f"Steps to run (default: all). Choices: {STEP_ORDER}",
    )
    parser.add_argument(
        "--no-consolidate", action="store_true",
        help="Skip cross-dataset consolidation",
    )
    args = parser.parse_args()

    datasets_to_run = args.datasets or list(DATASETS.keys())
    steps_to_run = args.steps or STEP_ORDER

    # Validate
    for ds in datasets_to_run:
        if ds not in DATASETS:
            log.error(f"Unknown dataset: {ds}. Choices: {list(DATASETS.keys())}")
            sys.exit(1)
    for step in steps_to_run:
        if step not in STEP_ORDER:
            log.error(f"Unknown step: {step}. Choices: {STEP_ORDER}")
            sys.exit(1)

    log.info(f"Datasets: {datasets_to_run}")
    log.info(f"Steps: {steps_to_run}")
    log.info("")

    results: dict[str, dict[str, bool]] = {}
    succeeded: list[str] = []

    for ds_name in datasets_to_run:
        ds_config = DATASETS[ds_name]
        ds_dir = ROOT / ds_config["dir"]

        if not ds_dir.exists():
            log.warning(f"Directory {ds_dir} not found; skipping {ds_name}")
            continue

        results[ds_name] = {}
        all_ok = True

        for step in steps_to_run:
            if step == "visualize":
                continue  # runs globally after consolidation, not per-dataset
            if step not in ds_config["steps"]:
                log.warning(f"[{ds_name}] Step '{step}' not configured; skipping")
                continue

            ok = run_step(ds_name, step, ds_config["steps"][step], ds_dir)
            results[ds_name][step] = ok
            if not ok:
                all_ok = False
                log.warning(f"[{ds_name}] Step '{step}' failed; stopping this dataset")
                break

        if all_ok:
            succeeded.append(ds_name)

    # ── summary ────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 60)
    for ds_name, step_results in results.items():
        status = "OK" if all(step_results.values()) else "FAILED"
        steps_str = "  ".join(
            f"{s}:{'ok' if v else 'FAIL'}" for s, v in step_results.items()
        )
        log.info(f"  {ds_name:25s} [{status}]  {steps_str}")

    if not args.no_consolidate and datasets_to_run:
        consolidate(datasets_to_run)

    # ── visualize (runs once across all datasets) ─────────────────
    if "visualize" in steps_to_run and succeeded:
        viz_cmd = f'{PYTHON} scripts/visualize_benchmark.py'
        run_step("all", "visualize", viz_cmd, ROOT)

    failed = [ds for ds in results if not all(results[ds].values())]
    if failed:
        log.warning(f"Failed datasets: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
