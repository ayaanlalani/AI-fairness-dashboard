.PHONY: smoke test fairness benchmark paper paper-facct overview papers

# Prefer a system python3.11: repo-local venvs can hang under iCloud Drive
# eviction (see artifacts/consolidated/stage1_findings.md, environment note).
PYTHON ?= $(shell command -v python3.11 || { test -x .venv/bin/python && echo .venv/bin/python; } || command -v python3)

smoke:
	$(PYTHON) -m py_compile run_pipeline.py scripts/openai_fairness_analysis.py scripts/llm_benchmark.py scripts/qualitative_analysis.py scripts/llm_benchmark_common.py scripts/scholarly_evidence.py scripts/llm_fairness_analysis.py scripts/openai_hybrid_self_improve.py
	test -f german_credit_dataset/data/german_credit_CLEANED_dataset.csv
	test -f hmda_dataset/processed/X_test_processed.csv
	test -f configs/openai_hybrid_smoke.json
	test -d artifacts/german_credit/fairness
	test -d artifacts/hmda/fairness

test:
	$(PYTHON) -m pytest tests/ -q

fairness:
	$(PYTHON) run_pipeline.py --datasets german_credit hmda lending_club --steps fairness qualitative visualize

benchmark:
	$(PYTHON) run_pipeline.py --datasets german_credit hmda --steps llm_benchmark visualize

# tectonic is the only LaTeX toolchain present on the dev machine; latexmk and
# pdflatex are kept first so CI and other environments are unaffected. tectonic
# runs BibTeX itself; the pdflatex fallback has to be told to, or bibliographies
# render as [?].
#
# $(1) = basename of a .tex file in report/, without the extension.
define build_tex
cd report && \
if command -v latexmk >/dev/null 2>&1; then \
	latexmk -pdf -interaction=nonstopmode $(1).tex; \
elif command -v pdflatex >/dev/null 2>&1; then \
	pdflatex -interaction=nonstopmode $(1).tex && \
	{ command -v bibtex >/dev/null 2>&1 && bibtex $(1) || true; } && \
	pdflatex -interaction=nonstopmode $(1).tex && \
	pdflatex -interaction=nonstopmode $(1).tex; \
elif command -v tectonic >/dev/null 2>&1; then \
	tectonic $(1).tex; \
else \
	echo "No LaTeX toolchain found (need latexmk, pdflatex, or tectonic)." >&2; exit 1; \
fi
endef

# NeurIPS Datasets & Benchmarks variant.
paper:
	$(call build_tex,report)

# FAccT/ACM draft — the primary paper.
paper-facct:
	$(call build_tex,report_facct)

# One-page overview. Must stay one page; see the header of report/overview.tex.
overview:
	$(call build_tex,overview)

papers: paper-facct overview paper
