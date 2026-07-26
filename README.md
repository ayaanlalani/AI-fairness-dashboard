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
pip install -r requirements.txt

make smoke
make fairness
```

The smoke test is local and does not call external APIs. The fairness run regenerates deterministic metrics for the two main lending datasets.

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
- `german_credit_dataset/`: German Credit preprocessing, model training, and fairness scripts.
- `hmda_dataset/`: HMDA Georgia preprocessing, model training, and fairness scripts.
- `scripts/openai_fairness_analysis.py`: primary OpenAI qualitative benchmark over deterministic metric context.
- `scripts/llm_benchmark.py`: multi-cycle LLM benchmark runner; OpenAI is the artifact default, while Gemini can be selected explicitly for legacy comparison.
- `configs/`: OpenAI pilot/smoke configs, report quality rubric, and guardrails.
- `docs/data_cards/`: data cards for German Credit and HMDA Georgia.
- `artifacts/german_credit/fairness/` and `artifacts/hmda/fairness/`: current deterministic and LLM audit outputs.
- `artifacts/consolidated/`: cross-dataset summaries and OpenAI comparison reports.
- `report/report.tex`: paper/report source.

Older course-project folders for diabetes, healthcare insurance, and Lending Club remain in the repository for provenance, but they are not the main NeurIPS artifact scope.

## Reviewer Checklist

| Step | Command | Est. time | Needs API key? |
|------|---------|-----------|----------------|
| 1. Install deps | `pip install -r requirements.txt` | ~1 min | No |
| 2. Smoke test | `make smoke` | <5 sec | No |
| 3. Fairness pipeline | `make fairness` | 3–5 min | No (Semantic Scholar optional) |
| 4. OpenAI benchmark | `make benchmark` | ~10 min | Yes (`OPENAI_API_KEY`) |
| 5. Paper build | `make paper` | ~30 sec | No (requires LaTeX) |

Expected outputs after `make fairness`:
- `artifacts/german_credit/fairness/fairness_metrics.csv`
- `artifacts/hmda/fairness/fairness_metrics.csv`
- Qualitative reports and visualizations under each dataset's `metrics/fairness/` directory

Expected outputs after `make benchmark`:
- `artifacts/llm_benchmark/gpt-4o/german_credit/_summary.json`
- `artifacts/llm_benchmark/gpt-4o/hmda/_summary.json`
- Per-attribute per-cycle JSON files with scores, refusal flags, and hallucination flags

## Setup

**Python 3.11 is required.** The `.venv` included in this repository uses Python 3.14 which has an incompatible pandas C extension. Use a fresh environment with Python 3.11:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

The `Makefile` auto-detects `python3.11` when available (takes precedence over `.venv/bin/python`).

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

## Troubleshooting

**`make smoke` fails with import errors**
- Ensure you are using Python 3.11: `python3.11 --version`
- Re-install dependencies: `python3.11 -m pip install -r requirements.txt`

**`make fairness` hangs or is very slow (2–5 min)**
- This is expected: the qualitative step queries Semantic Scholar for research evidence.
- The circuit breaker trips after 2 consecutive timeouts (8 s each), so worst-case is ~16 s of network wait then normal completion.
- Set `SEMANTIC_SCHOLAR_API_KEY` to increase the API rate limit.
- If the hang persists beyond 10 minutes, kill the process and check network connectivity to `api.semanticscholar.org`.

**`aif360` import warns about missing TensorFlow or inFairness**
- These warnings are non-fatal. The benchmark uses only the base AIF360 metrics (no AdversarialDebiasing or SenSR).

**`make benchmark` fails with `OPENAI_API_KEY not set`**
- Export the key before running: `export OPENAI_API_KEY="sk-..."`
- The benchmark does NOT silently fall back to Gemini; it exits with a clear error if the key is absent.

**`make paper` fails with `pdflatex: command not found`**
- Install a LaTeX distribution: `brew install --cask mactex` (macOS) or `apt install texlive-full` (Linux).
- As a workaround, read `report/report.tex` directly or view pre-rendered figures in `artifacts/visualizations/`.

**Python version compatibility**
- The `.venv` in this repo uses Python 3.14 (from course setup) with a broken pandas C extension.
- Always use a separate `python3.11` environment for all pipeline steps.

## Citation

Please cite this artifact using `CITATION.cff`.
