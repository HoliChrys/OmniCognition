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

# Meta-cognitive walk answerer (the real system : walk_start/walk_next
# over the FACT/THOUGHT/ACTION manifold, Claude as the driver)
uv run python -m benchmarks.locomo.eval --samples 10 --encoder semantic --answerer meta

# Balanced smoke : 1 QA per category, a few conversations
uv run python -m benchmarks.locomo.eval --samples 4 --per-category 1 --answerer meta

# Opt-in retrieval handles
uv run python -m benchmarks.locomo.eval --answerer meta --entities   # entity beacons
uv run python -m benchmarks.locomo.eval --answerer meta --atomic     # atomic-fact handles
uv run python -m benchmarks.locomo.eval --answerer meta --merge      # dedup identical turns

# Extractive ReAct (roberta-base-squad2, local CPU)
uv run python -m benchmarks.locomo.eval --samples 10 --encoder semantic --answerer extractive
```

### Targeted debuggers (no full bench)

Fast, inspectable harnesses for stepping through the system on a tiny
slice — index → one-shot retrieval → walk stage-by-stage → full agent,
all in seconds. Built precisely to avoid "launch a 20-minute bench to
discover nothing works".

```bash
# Step through indexation, retrieval and the walk on one cat-3 probe
DEBUG_NSESS=0 DEBUG_PROBE=caroline python -m benchmarks.locomo.debug_walk all

# Measure lateral-collision effect on kNN redundancy (before/after sleep)
DEBUG_NSESS=0 python -m benchmarks.locomo.debug_lateral
```

`DEBUG_NSESS` = number of sessions to index (0 = whole conversation) ;
`DEBUG_PROBE` = `caroline` | `john` ; `METACOG_HYDE=1` toggles the HyDE
retrieval channel.

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
| `--per-category` | None                               | up to N QAs per category (balanced sampling)   |
| `--k`          | 7                                    | retrieval top-k                                |
| `--encoder`    | `semantic`                           | `simple` (simhash) or `semantic` (MiniLM)      |
| `--answerer`   | `chunk`                              | `chunk`, `extractive` (roberta), `claude` (single-shot ReAct), `mcp`, `meta` (the walk) |
| `--lineage`    | off                                  | enable cosine+lineage RRF retrieval            |
| `--entities`   | off                                  | spawn entity-beacon retrieval handles          |
| `--atomic`     | off                                  | spawn mem0-style atomic-fact handles           |
| `--merge`      | off                                  | merge identical turns (corroboration absorbed) |
| `--auto-cluster` | off                                | Level-1 community / observator detection       |
| `--agent-concurrency` | 10                            | parallel QA workers for `mcp`/`meta`           |
| `--debug-jsonl` | None                                | per-QA trace dump for inspection               |
| `--claude-model` | env or default                     | override Anthropic model id                    |

## Reported metrics

For each QA pair :
- **Recall@5 / Recall@7** : fraction of gold-evidence `dia_id`s present
  in the top-k retrieved set.
- **agent_recall** (`mcp`/`meta`) : cumulative recall across all the
  queries the agent issued during its walk — the evidence it actually saw,
  not just a single top-k.
- **F1** : per-category token-overlap F1 against the gold answer (category 5
  is binary abstention : 1.0 iff the answer says "not mentioned").

Aggregated per category (1 multi-hop · 2 temporal · 3 inference ·
4 single-hop · 5 adversarial) and overall. `--debug-jsonl` writes a
per-QA trace (question, gold, prediction, retrieved ids, react trace) for
inspection.

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

The **chunk-dump** stub answerer is a retrieval-only smoke (no LLM span
extraction) and is not directly comparable. For a real comparison use
`--answerer meta` (the meta-cognitive walk driven by Claude). The system
prompt deliberately carries **no dataset-specific answer vocabulary** —
earlier few-shot leaks of gold answers were stripped, so the numbers are
an honest no-hardcode baseline rather than a prompt-overfit one. Use the
targeted debuggers above to attribute any gap to indexation, retrieval,
the walk, or answer generation before launching a full run.

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
