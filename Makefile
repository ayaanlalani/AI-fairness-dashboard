.PHONY: smoke fairness benchmark paper

PYTHON ?= $(shell if test -x .venv/bin/python; then echo .venv/bin/python; else command -v python3; fi)

smoke:
	$(PYTHON) -m py_compile run_pipeline.py scripts/openai_fairness_analysis.py scripts/llm_benchmark.py scripts/qualitative_analysis.py scripts/llm_benchmark_common.py
	test -f german_credit_dataset/data/german_credit_CLEANED_dataset.csv
	test -f hmda_dataset/processed/X_test_processed.csv
	test -f configs/openai_hybrid_smoke.json
	test -d artifacts/german_credit/fairness
	test -d artifacts/hmda/fairness

fairness:
	$(PYTHON) run_pipeline.py --datasets german_credit hmda --steps fairness qualitative visualize

benchmark:
	$(PYTHON) run_pipeline.py --datasets german_credit hmda --steps llm_benchmark visualize

paper:
	cd report && if command -v latexmk >/dev/null 2>&1; then latexmk -pdf -interaction=nonstopmode report.tex; else pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex; fi
