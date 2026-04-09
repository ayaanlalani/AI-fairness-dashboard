# AI Fairness Dashboard

This repo hosts the pipelines we built in CMPT 310 to audit fairness across multiple real-world datasets (diabetes, healthcare insurance, German credit, and Lending Club). Each dataset folder contains scripts to clean data, train predictive models, and export bias diagnostics that feed into a shared set of poster-ready visuals.

## Repository Layout

- `diabetes_dataset/`, `Healthcare-insurance-dataset/`, `german_credit_dataset/`, `lending_club_dataset/`: end-to-end pipelines (cleaning, training, fairness scripts, metrics, and plots) for each dataset.
- `scripts/generate_consolidated_visuals.py`: stacks the per-dataset fairness metrics into consolidated PNGs/CSV for the poster.
- `poster_assets/`: latest consolidated plots (DI overview, gap panels, bias heatmap) plus `consolidated_fairness_metrics.csv`.
- `requirements.txt`: frozen Python environment used to generate all results.

## Environment Setup

1. Install Python 3.10+ (we developed in a venv sitting at `.venv/`).
2. Create and activate a fresh virtual environment, then install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Reproducing Results

All commands assume you are in the repo root with the virtual environment activated.

### Diabetes
```bash
python diabetes_dataset/scripts/train_models.py \
  --data_path diabetes_dataset/data/diabetes_cleaned.csv \
  --out_dir diabetes_dataset/metrics \
  --models_dir diabetes_dataset/models

python diabetes_dataset/compute_fairness.py \
  --predictions_dir diabetes_dataset/metrics \
  --out_dir diabetes_dataset/metrics/fairness
```

### Healthcare Insurance
```bash
python Healthcare-insurance-dataset/scripts/train_models.py \
  --data_dir Healthcare-insurance-dataset/processed \
  --out_dir Healthcare-insurance-dataset/metrics \
  --models_dir Healthcare-insurance-dataset/models

python Healthcare-insurance-dataset/scripts/compute_fairness.py \
  --predictions_dir Healthcare-insurance-dataset/metrics \
  --out_dir Healthcare-insurance-dataset/metrics/fairness
```

### German Credit
```bash
python german_credit_dataset/scripts/compute_fairness.py \
  --data_path german_credit_dataset/data/german_credit_CLEANED_dataset.csv \
  --metrics_dir german_credit_dataset/metrics \
  --fairness_dir german_credit_dataset/metrics/fairness
```

### Lending Club
```bash
python lending_club_dataset/data/clean_lending_club.py \
  --input_path lending_club_dataset/data/loan.csv \
  --out_dir lending_club_dataset/processed

python lending_club_dataset/scripts/train_models.py \
  --data_dir lending_club_dataset/processed \
  --out_dir lending_club_dataset/metrics \
  --models_dir lending_club_dataset/models

python lending_club_dataset/scripts/compute_fairness.py \
  --data_dir lending_club_dataset/processed \
  --predictions_dir lending_club_dataset/metrics \
  --out_dir lending_club_dataset/metrics/fairness
```

### Consolidated Poster Visuals
After each dataset exports `metrics/fairness/fairness_metrics.csv`, regenerate the combined assets:
```bash
python scripts/generate_consolidated_visuals.py --output_dir poster_assets
```

## OpenAI Hybrid Self-Improvement (Deepnote)

This repo includes a Deepnote-friendly runner that performs generator cycles,
judge-model rubric scoring, reflection-based prompt updates, multi-run
selection, refined guardrail proposal, and a validation rerun.

Configuration files:
- `configs/openai_hybrid_pilot.json` (pilot dataset, budget, experiment matrix)
- `configs/report_quality_rubric.json` (judge rubric for report quality)
- `configs/guardrails_baseline.json` (starting guardrails)

Run:
```bash
python scripts/openai_hybrid_self_improve.py \
  --config configs/openai_hybrid_pilot.json \
  --rubric configs/report_quality_rubric.json \
  --guardrails configs/guardrails_baseline.json \
  --out_root artifacts
```

Required env var:
```bash
export OPENAI_API_KEY="..."
```

Outputs are written under:
- `artifacts/<dataset>/fairness/openai_hybrid/<timestamp>/`
- includes cycle logs, `run_registry.csv`, `guardrails_refined_v2.json`, and
  `promotion_decision.json`

## Submission Bundle

To package everything for submission while excluding the virtual environment and other large artifacts:
```bash
cd ..
zip -r AI-fairness-dashboard.zip AI-fairness-dashboard \
  -x "AI-fairness-dashboard/.venv/*" \
     "AI-fairness-dashboard/.git/*" \
     "AI-fairness-dashboard/__pycache__/*"
```

The resulting `AI-fairness-dashboard.zip` contains all source, configs, scripts, metrics, and documentation needed to reproduce the fairness analysis.

