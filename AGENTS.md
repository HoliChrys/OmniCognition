# AGENTS.md — working in OmniCognition / MetaCog-Mem

Guide for AI coding agents (and humans) operating in this repository.
Read this before making changes.

## What this is

**MetaCog-Mem** is a manifold-based metacognitive memory for LLM agents.
Core commitments (do not violate them):

- **Edge-free.** Relations are geometric proximity produced by `apply_pull`,
  not stored edges. There is no `edges` field. When you need "X relates to Y",
  pull them together in the embedding space — co-location *is* the edge.
- **Hyperparameter-free.** Every threshold is a mathematical constant or emerges
  from the data (e.g. the walk's σ-cap from GUM uncertainty propagation). Do not
  add tunable magic numbers; if you must, justify it and prefer data-derived.
- **Anti-laundering (Cor. 5).** Provenance is typed. `SourceClass.GENERATOR`
  content can become *content* but never *evidence*. Generated nodes
  (THOUGHT/ACTION/event hubs/beacons) are GENERATOR-sourced and must be created
  with `apply_pull` directly, never through an `Observation`. Tests in
  `tests/test_no_laundering.py` enforce this — keep them green.

## Setup & commands

```bash
pip install -e .                 # runtime deps (mcp, nltk)
pip install -e '.[bench]'        # + sentence-transformers, torch, anthropic
python -m pytest tests/ -q       # full suite (~470+ tests, deterministic)
python -m pytest tests/test_event_node.py -q   # a single module
python -m metacog.mcp_server     # run the MCP service (entrypoint metacog-mcp)
```

Benchmarks need `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` in the env
(picked up by `metacog/llm.py:ClaudeLLM`). Tests do NOT need a live LLM — they
use scripted fakes; never make a test depend on the network.

## Architecture map (`metacog/`)

- `epistemic.py` — `Point` (the one schema), `PointKind` (FACT / THOUGHT /
  ACTION / EVENT), `SourceClass`, epistemic state.
- `memory.py` — `Memory`: ingest, retrieve, the event subsystem (`ingest_event`,
  `consolidate_events`, `detect_event_type`, `event_centroid`/`context_centroid`,
  `event_cluster`/`context_members`, named **bags**), bi-temporal validity,
  `scoped_answer` (two-phase scoped→knowledge-base cascade).
- `geometry.py` — `apply_pull` (the edge-free relation), retrieval/spreading.
- `meta_walk.py` — `MetaWalker`: the uncertainty-governed walk. Re-anchors on the
  nearest ACTION each stage and spreads from it. Stops on `step().done`
  (σ/GUM), not a fixed cap. `_relevant_cum` is the committed evidence set
  (uncapped); `_composable_evidence` is a bounded view for answer synthesis.
- `event_schema.py` — event-type schema induction, slot-filling
  (`fill_event_schema`, scoped+KB per core slot, cross-slot dedup),
  `event_search`, `event_action_enrich` (the meta-cognition action bridge).
- `event_extractor.py`, `keywords.py`, `entities.py`, `atomic.py` — LLM-backed
  extractors. All cached, failure-safe, and **never cache an empty result**.
- `enumeration.py` — retrieve/"bag" mode (answer as an exhaustive list).
- `mcp_server.py` — the MCP tool surface.

## Two answer modes (important when evaluating)

A query is either **"generate a focused answer"** (precision/F1 matters; the
uncertainty-stopped walk excels) or **"refer an exhaustive set of elements"**
(recall matters; the event cluster / bag excels), and the two can coexist
(an answer *with* the exhaustive list = the bag). Measure RECALL **and**
precision/F1, and always report set size `n` — high recall at large `n` is
over-selection, not a win. The walk's evidence set is uncertainty-bounded and
**must not be hard-capped**: a subject can have arbitrarily many relevant items.

## Benchmarks (`benchmarks/`)

- `benchmarks/locomo/` — LoCoMo long-conversation QA (the `meta` answerer,
  `mcp_meta_agent.py`, `debug_qa.py` REPL).
- `benchmarks/obliq_bench/` — OBLIQ-Bench (oblique queries, latent relevance,
  **recall-only**). `event_step.py` (per-query step debugger + action-bridge
  ablation), `tree_nav.py` (tree-navigable parallel-query debugger),
  `batch_walk_vs_event.py` (walk-vs-event, both modes).

Empirical note (descriptive/twitter): the meta-cognitive walk dominates the
event channel on small/medium gold (both recall and F1), but on **large
exhaustive gold sets** (e.g. q0395, gold=58) the walk's uncertainty stop caps
its recall while the **event cluster wins recall** — the regime the event
branch earns its cost. The event channel is **additive**: it unions into
agent-recall, never replaces the walk.

## Conventions

- Match surrounding style; keep comment density and naming idiomatic to the file.
- New extractors mirror the existing ones: `_cache` dict, code-fence strip,
  total `try/except -> []`, never cache empty.
- Persistence: `Memory.save()` pickles `points` only. Anything not in `points`
  (registries, bags, threads) must be rebuilt in `load()`.
- Run the relevant tests before committing; keep the full suite green.
