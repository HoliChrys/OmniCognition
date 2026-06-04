# `benchmarks/` — evaluation harnesses

Reproducible, inspectable evaluation of MetaCog-Mem. Everything here is
read-only with respect to the core: a benchmark never alters the memory it
is scored against.

## LoCoMo (`benchmarks/locomo/`)

The primary harness: **LoCoMo-10** (Maharana et al.) — ten ~300-turn
conversations, ~1 986 QA pairs across five categories (1 multi-hop,
2 temporal, 3 inference, 4 single-hop, 5 adversarial). See
[`locomo/README.md`](locomo/README.md) for setup, the CLI, the metrics,
and the full per-version results table (V5 baseline → V11
uncertainty-governed depth).

### Answerers (`--answerer`)
| value | what it is |
|---|---|
| `chunk` | retrieval-only smoke (no LLM span extraction) |
| `extractive` | local roberta-base-squad2 ReAct |
| `claude` | single-shot ReAct over a dumped context |
| `mcp` | tool-using agent over the metacog MCP server |
| `meta` | **the real system** — the meta-cognitive walk (breadth-pivot agent over `walk_start`) |

### Metrics
- **Recall@5 / Recall@7** — gold `dia_id`s in the static top-k.
- **agent_recall** — cumulative gold evidence the walk actually saw across
  all its breadth pivots (the honest retrieval number; ≈ 0.85 at V11 vs a
  static Recall@7 ≈ 0.31).
- **token-F1** — per-category overlap against the gold answer. A loose
  proxy: it counts "progressive" ≠ "liberal" as a miss, so judge answer
  *quality* alongside it.

### Targeted debuggers (no full bench)
```bash
DEBUG_NSESS=0 DEBUG_PROBE=caroline python -m benchmarks.locomo.debug_walk all
DEBUG_NSESS=0 python -m benchmarks.locomo.debug_lateral
```
Step through indexation → one-shot retrieval → the walk stage-by-stage →
the full agent in seconds, to attribute any gap to indexation, retrieval,
the walk, or generation *before* launching a 20-minute run.

## Files
| file | role |
|---|---|
| `eval.py` | the runner; ingestion (incl. deterministic relative→absolute date expansion), metrics, per-category aggregation |
| `mcp_meta_agent.py` | the `meta` answerer: breadth-pivot agent over `walk_start`, deterministic date resolution at answer time |
| `mcp_agent.py` | the `mcp` tool-using ReAct answerer |
| `claude_react.py`, `react_qa.py` | single-shot / extractive baselines |
| `encoders.py` | semantic (MiniLM) and simple (simhash) encoders |
| `official_locomo_eval.py` | the official LoCoMo token-F1 scorer (PorterStemmer) |
| `debug_walk.py`, `debug_lateral.py`, `analyze.py` | inspectable debuggers |
