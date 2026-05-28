.PHONY: install bench-deps bench-data bench-plain bench-lineage bench-extractive bench-claude bench-claude-smoke bench-full test clean

# uv-based workflow.
# - `make install`     : sync core deps + the project itself (editable)
# - `make bench-deps`  : sync with [bench] extras (sentence-transformers,
#                        transformers, torch, anthropic)
# - `make bench-data`  : download LoCoMo-10 dataset
# - `make bench-*`     : run the benchmark variants
# - `make test`        : run the pytest suite

install:
	uv sync

bench-deps:
	uv sync --extra bench

bench-data:
	bash benchmarks/locomo/download.sh

# Plain cosine retrieval (baseline)
bench-plain: bench-data
	uv run python -m benchmarks.locomo.eval --samples 10 --encoder semantic

# Cosine + lineage RRF retrieval
bench-lineage: bench-data
	uv run python -m benchmarks.locomo.eval --samples 10 --encoder semantic --lineage

# Plain cosine + extractive (roberta) ReAct answerer
bench-extractive: bench-data
	uv run python -m benchmarks.locomo.eval --samples 10 --encoder semantic --answerer extractive

# Cosine + lineage + Claude API ReAct answerer (needs ANTHROPIC_API_KEY)
bench-claude: bench-data
	uv run python -m benchmarks.locomo.eval --samples 10 --encoder semantic --lineage --answerer claude

# Cheap smoke run on Claude — 1 conv / 20 QAs
bench-claude-smoke: bench-data
	uv run python -m benchmarks.locomo.eval --samples 1 --max-qa 20 --encoder semantic --lineage --answerer claude

# Full stack : extractive + lineage
bench-full: bench-data
	uv run python -m benchmarks.locomo.eval --samples 10 --encoder semantic --lineage --answerer extractive

test:
	uv run pytest tests/ -v

clean:
	rm -rf .venv __pycache__ */__pycache__ */*/__pycache__ .pytest_cache *.egg-info
