# Fairness Audit Benchmark for Lending Datasets

This repository contains a NeurIPS 2026 Evaluations & Datasets style artifact for testing whether LLM-based fairness auditors produce stable, statistically grounded, remediation-ready audits on lending datasets.

The artifact has two linked parts:

- A deterministic fairness audit pipeline for German Credit and HMDA Georgia.
- An OpenAI-based LLM benchmark that receives Python-computed metric context and is evaluated on whether it produces consistent qualitative diagnoses, severity labels, and mitigation plans.

Gemini outputs are retained only as secondary legacy comparison artifacts where they already exist. The primary benchmark path for this release is OpenAI.

## Reviewer Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

make smoke
make test
make fairness
```

The smoke test and unit tests are local and do not call external APIs. The fairness run regenerates deterministic metrics for the main lending datasets.

> **Note:** if the repository lives in an iCloud-synced location (e.g. `Desktop/`), repo-local virtualenvs can hang on native-library loads during file eviction. The Makefile therefore prefers a system `python3.11` when one is available; installing dependencies against Homebrew `python3.11` is the most reliable local setup.

To run the OpenAI benchmark, set an API key first:

```bash
export OPENAI_API_KEY="..."
make benchmark
```

Optional literature retrieval can use:

```bash
export SEMANTIC_SCHOLAR_API_KEY="..."
```

## Repository Layout

- `run_pipeline.py`: central orchestrator for cleaning, training, deterministic fairness analysis, qualitative analysis, OpenAI benchmarking, and visualization.
- `german_credit_dataset/`: German Credit preprocessing, model training, and fairness scripts (UC1, consumer credit scoring).
- `hmda_dataset/`: HMDA Georgia preprocessing, model training, and fairness scripts (UC2, mortgage underwriting).
- `lending_club_dataset/`: Lending Club preprocessing, model training, and fairness scripts (UC3, P2P default risk; favorable outcome = predicted non-default, `favorable_label 0`).
- `scripts/openai_fairness_analysis.py`: primary OpenAI qualitative benchmark over deterministic metric context.
- `scripts/llm_benchmark.py`: multi-cycle LLM benchmark runner; OpenAI is the artifact default, while Gemini can be selected explicitly for legacy comparison.
- `configs/`: OpenAI pilot/smoke configs, report quality rubric, and guardrails — including `configs/research_guardrails.json`, the machine-readable stage gates (LLM providers stay `"blocked"` until explicitly user-approved).
- `docs/data_cards/`: data cards for German Credit and HMDA Georgia.
- `docs/RESEARCH_STAGING_PROMPT.md`: the staged research program (Stage 0 scaffolding → Stage 5 paper artifacts) spanning the three lending use cases.
- `artifacts/german_credit/fairness/`, `artifacts/hmda/fairness/`, `artifacts/lending_club/fairness/`: current deterministic and LLM audit outputs.
- `artifacts/llm_benchmark/`: frozen dry-run prompt packs (`dry_run/`) and the Track Q run manifest (`RUN_MANIFEST.md`).
- `artifacts/consolidated/`: cross-dataset summaries, the lending-lifecycle baseline, and OpenAI comparison reports.
- `tests/`: pytest suite covering fairness-metric orientation/sign conventions and the LLM benchmark scoring, refusal, and hallucination detectors (`make test`).
- `report/report.tex`: paper/report source.

Older course-project folders for diabetes and healthcare insurance remain in the repository for provenance, but they are not the main NeurIPS artifact scope. Lending Club has been promoted into scope as UC3 of the staged lending-lifecycle research program.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The same dependencies are declared in `pyproject.toml` for `uv` users:

```bash
uv sync
```

## Commands

### Smoke Test

```bash
make smoke
```

Expected behavior: compile the main Python entrypoints and verify that the required datasets, configs, and artifact directories exist. This target does not run the OpenAI API.

### Deterministic Fairness Pipeline

```bash
make fairness
```

Equivalent command:

```bash
python run_pipeline.py \
  --datasets german_credit hmda \
  --steps clean train fairness qualitative visualize
```

Expected outputs:

- `german_credit_dataset/metrics/classification_predictions.csv`
- `german_credit_dataset/metrics/fairness/fairness_metrics.csv`
- `german_credit_dataset/metrics/fairness/qualitative_report.md`
- `hmda_dataset/metrics/classification_predictions.csv`
- `hmda_dataset/metrics/fairness/fairness_metrics.csv`
- `hmda_dataset/metrics/fairness/qualitative_report.md`
- consolidated copies under `artifacts/`

### OpenAI Benchmark

```bash
export OPENAI_API_KEY="..."
make benchmark
```

Equivalent command:

```bash
python run_pipeline.py \
  --datasets german_credit hmda \
  --steps llm_benchmark visualize
```

The OpenAI workflow receives deterministic metric and group-breakdown context prepared by Python. It is not presented as independently recomputing fairness metrics. Its benchmark role is to test whether an LLM can transform fixed quantitative evidence into stable, statistically grounded, remediation-ready audit narratives.

Expected outputs:

- `german_credit_dataset/metrics/fairness/openai/llm_context_payload.json`
- `german_credit_dataset/metrics/fairness/openai/llm_fairness_report.md`
- `german_credit_dataset/metrics/fairness/openai/benchmark_comparison.md`
- `hmda_dataset/metrics/fairness/openai/llm_context_payload.json`
- `hmda_dataset/metrics/fairness/openai/llm_fairness_report.md`
- `hmda_dataset/metrics/fairness/openai/benchmark_comparison.md`
- consolidated OpenAI reports under `artifacts/consolidated/`

### Paper Build

```bash
make paper
```

Expected output:

- `report/report.pdf`

If `latexmk` is unavailable, run:

```bash
cd report
pdflatex report.tex
pdflatex report.tex
```

### Optional Legacy Gemini Comparison

Gemini is not the primary benchmark for this artifact. Existing Gemini results are retained as secondary/legacy comparisons. To run the generic benchmark with Gemini explicitly:

```bash
GEMINI_API_KEY="..." python scripts/llm_benchmark.py ... --model gemini-2.5-flash
```

## Data

Main datasets:

- German Credit: UCI Statlog German Credit plus a local Kaggle-formatted CSV used for human-readable feature names.
- HMDA Georgia: CFPB/FFIEC HMDA records filtered to originated and denied mortgage applications.

See:

- `docs/data_cards/german_credit.md`
- `docs/data_cards/hmda_georgia.md`

The repository license covers this code, documentation, and artifact scaffolding. Dataset files and derived records remain subject to their original source terms and access conditions; see the data cards for source-specific notes.

## Reproducibility Notes

- Random seeds are fixed at `42` in preprocessing/model scripts where train/test splits or model training require randomness.
- Fairness metrics are computed by deterministic Python code, with AIF360 used when available and manual fallback logic otherwise.
- OpenAI model outputs may vary across API/model versions even with fixed prompts. The benchmark therefore records prompts, context payloads, raw responses, model names, token usage, and cost estimates.
- Small subgroup results, especially German Credit `foreign_worker` and rare HMDA race categories, should be treated as statistically fragile.

## Citation

Please cite this artifact using `CITATION.cff`.
