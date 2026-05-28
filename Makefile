.PHONY: install bench-deps bench-data bench-plain bench-lineage bench-react bench-full test

install:
	pip install -e .

bench-deps:
	pip install -e ".[bench]"

bench-data:
	bash benchmarks/locomo/download.sh

# Plain cosine retrieval (baseline)
bench-plain: bench-data
	python -m benchmarks.locomo.eval --samples 10 --encoder semantic

# Cosine + lineage RRF retrieval
bench-lineage: bench-data
	python -m benchmarks.locomo.eval --samples 10 --encoder semantic --lineage

# Plain cosine + extractive (roberta) ReAct answerer
bench-extractive: bench-data
	python -m benchmarks.locomo.eval --samples 10 --encoder semantic --answerer extractive

# Cosine + lineage + Claude API ReAct answerer (needs ANTHROPIC_API_KEY)
bench-claude: bench-data
	python -m benchmarks.locomo.eval --samples 10 --encoder semantic --lineage --answerer claude

# Cheaper smoke run on Claude — 1 conv / 20 QAs
bench-claude-smoke: bench-data
	python -m benchmarks.locomo.eval --samples 1 --max-qa 20 --encoder semantic --lineage --answerer claude

# Full stack : extractive + lineage
bench-full: bench-data
	python -m benchmarks.locomo.eval --samples 10 --encoder semantic --lineage --answerer extractive

# Quick smoke run on 1 conversation, 20 QAs
bench-smoke: bench-data
	python -m benchmarks.locomo.eval --samples 1 --max-qa 20 --encoder semantic --lineage

test:
	pytest tests/ -v
