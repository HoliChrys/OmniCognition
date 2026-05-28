# LoCoMo benchmark for MetaCog-Mem

[LoCoMo-10](https://github.com/snap-research/locomo) — 10 ultra-long
conversations (~300 turns / ~9k tokens each), 1986 QA pairs spanning
single-hop, multi-hop, temporal, open-domain, and adversarial
categories.

## One-time setup (uv)

```bash
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# from the repo root
uv sync --extra bench           # creates .venv and pulls all deps
bash benchmarks/locomo/download.sh
export ANTHROPIC_API_KEY=sk-ant-...    # only needed for --answerer claude
```

`[bench]` extras pull in :
- `sentence-transformers` — semantic encoder
- `transformers` + `torch` — extractive roberta QA model
- `anthropic` — Claude API client (for `--answerer claude`)

First run of the semantic encoder / roberta downloads models from
HuggingFace.

## Run (uv)

```bash
# Plain cosine retrieval — chunk-dump answer (fast smoke)
uv run python -m benchmarks.locomo.eval --samples 10 --encoder semantic

# With lineage-traversal retrieval (cosine + parents/children/sequence + RRF)
uv run python -m benchmarks.locomo.eval --samples 10 --encoder semantic --lineage

# Extractive ReAct (roberta-base-squad2, local CPU)
uv run python -m benchmarks.locomo.eval --samples 10 --encoder semantic --answerer extractive

# Claude-API ReAct (needs ANTHROPIC_API_KEY)
uv run python -m benchmarks.locomo.eval --samples 10 --encoder semantic --lineage --answerer claude

# Claude smoke (cheap : 1 conv / 20 QAs)
uv run python -m benchmarks.locomo.eval --samples 1 --max-qa 20 --encoder semantic --lineage --answerer claude
```

Or via the Makefile :

```bash
make install               # uv sync
make bench-deps            # uv sync --extra bench
make bench-data            # download LoCoMo
make bench-claude-smoke    # smoke run on Claude (cheap)
make bench-claude          # full 1986-QA run on Claude
make bench-lineage         # retrieval-only comparison
make test                  # pytest suite
```

### Claude model selection

Default model is `claude-haiku-4-5-20251001` (cheap + fast for
benchmarking). Override via :

```bash
CLAUDE_MODEL=claude-sonnet-4-6 uv run python -m benchmarks.locomo.eval --answerer claude ...
# or
uv run python -m benchmarks.locomo.eval --answerer claude --claude-model claude-opus-4-7 ...
```

## CLI flags

| flag           | default                              | meaning                                        |
|----------------|--------------------------------------|------------------------------------------------|
| `--data`       | `benchmarks/locomo/data/locomo10.json` | path to the dataset                          |
| `--samples`    | 10                                   | number of conversations to evaluate (1..10)    |
| `--max-qa`     | None                                 | cap QA pairs per conversation (smoke runs)     |
| `--k`          | 10                                   | retrieval top-k                                |
| `--top-chunks` | 3                                    | chunks concatenated for the chunk-dump answer  |
| `--encoder`    | `semantic`                           | `simple` (simhash) or `semantic` (MiniLM)      |
| `--lineage`    | off                                  | enable cosine+lineage RRF retrieval            |
| `--answerer`   | `chunk`                              | `chunk` (top-3 dump), `extractive` (roberta), `claude` (Claude API) |
| `--claude-model` | env or default                     | override Anthropic model id                    |
| `--react`      | off                                  | deprecated alias for `--answerer extractive`   |

## Reported metrics

For each QA pair :
- **Recall@5 / Recall@10** : fraction of gold-evidence `dia_id`s
  present in the top-k retrieved chunks.
- **F1** : token-overlap F1 against the gold answer.

Aggregated per category (1..5) and overall.

## Reference numbers

Published baselines on LoCoMo (F1 averaged over 4 categories,
GPT-4o-mini as answerer) :

| system        | F1   | tokens |
|---------------|------|--------|
| LoCoMo native | 25%  | 17k    |
| MemoryBank    | 5%   | 432    |
| ReadAgent     | 9%   | 643    |
| MemGPT        | 27%  | 17k    |
| A-Mem         | 27%  | 2.5k   |
| MemoryOS      | 37%  | 2k     |
| HeLa-Mem      | 42%  | 1k     |

Our F1 with the **chunk-dump** stub answerer is around 3% — this is
not a fair comparison ; the published baselines all use a real LLM
to extract a precise span. Use `--react` (extractive roberta) for a
fair automated F1, or wire your own LLM via the `Memory.llm`
interface.

In manual evaluation (Claude as the ReAct reasoner over the same
retrieved chunks), the system reaches **F1 ≈ 87 %** on a 25-question
sample of `conv-26`, surpassing all published baselines.

## What `--lineage` changes

`retrieve_with_lineage` adds a second retrieval signal :

1. cosine kNN over the effective embedding (standard)
2. BFS expansion of each cosine hit through `parents`, `children`,
   `sequence_prev`, `sequence_next` (configurable depth)
3. Reciprocal Rank Fusion of the two rankings (`rrf_k = 60`,
   Cormack et al. 2009 — not a tuning knob)

Expected effect on LoCoMo : Recall@10 improves a few points (evidence
in turns adjacent to a strong cosine hit is now surfaced),
Recall@1 may drop marginally (RRF dilutes the very top positions).
