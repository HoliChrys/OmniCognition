# OmniCognition / MetaCog-Mem

A metacognitive memory architecture for LLM agents, built **without
hyperparameters** and with **strict epistemic typing**. Memory is a
manifold of typed points — facts, thoughts, actions, and self-built
tools — that the system retrieves, reasons over, consolidates, and
extends on its own.

## Core ideas

- **Manifold over graph.** No materialized edges. Relations between
  memory items emerge from k-nearest-neighbors over the current
  effective embedding (semantic + position-weighted keyword + BM25 +
  fuzzy, fused by Reciprocal Rank Fusion).

- **Typed points share one DB.** `FACT` / `THOUGHT` / `ACTION` — same
  schema, same A(·) function, same lifecycle. An open, multi-kind **tag**
  vocabulary refines a node's type and role over time (`tool`,
  `tool_workflow`, `entity`, `atomic`, `lateral_absorbed`, …).

- **Anti-laundering, by construction.** Following Romanchuk & Bondar's
  "Semantic Laundering" (arXiv:2601.08333), the system enforces typed
  sources : `Observation(source=GENERATOR)` raises `LaunderingError`. LLM
  outputs become *content* (∈ P) but never *evidence*.

- **No hyperparameters.** Every "threshold" is either a math constant
  (cos π/4, cos π/12) or emerges from the data — `median ± σ` for
  collision, the Poisson floor `λ + √λ` for every spike/recurrence law.

- **Zero deletion.** Latent states (`INVALID`, `DEPRECATED`) are exiled
  geometrically, not removed. Resurrection is automatic if new
  observations re-corroborate them.

- **Epistemic pluralism.** Observators give each source its own parallel
  view on every point (explicit, auto-spawn by polarity, or by semantic
  clustering / Level-1 communities).

## The meta-cognitive walk

Retrieval is a coordinated **walk** over three kinds at once
(`metacog/meta_walk.py`). At each stage it retrieves the nearest FACTs and
ACTIONs to the query, generates any missing kind under a strict token
budget (`source=GENERATOR`, never evidence), reads each retrieved fact
with a **Chain-of-Note** relevance pass, and synthesizes a THOUGHT whose
keywords — drawn only from the *walked* points' preexisting keywords, never
invented — extend a REDUCE-folded reasoning chain that drives the next
stage.

**Depth is governed by uncertainty propagation, not a stage count.**
Following the GUM (BIPM 1995) combination of independent uncertainties,
the walk accumulates `σ_path = √(Σ σ_hop²)` over its evidence chain, where
each `σ_hop` is the smallest keyword-embedding distance from the previous
fact★ to any fact reachable this stage (chain *coherence*, not the
volatility of the fact choice). Three structural rules, no tuning knobs:

- **Floor** — at least `_MIN_STAGES = 3` hops always run.
- **σ-cap** — once `σ_path` exceeds the walk-local emergent threshold
  (median + std of the stage-0 facts' pairwise distances = the manifold's
  local resolution), the chain has wandered out of the coherent
  neighbourhood and the walk stops. A coherent gold trail keeps hops small,
  so σ grows slowly and the walk goes deep on its own; wandering trips it
  fast. Gold-retrieval extension is therefore *intrinsic* to σ — no
  separate (and noisy) relevance gate.
- **Keyword-coverage stop** — a cheap, deterministic metacognitive check:
  once the gathered evidence's keywords cover every query keyword, the walk
  has turns bearing on each facet of the question and stops. Vocabulary-gap
  questions (cat3: "political leaning" vs "LGBTQ advocacy") never reach
  full coverage, so σ governs those instead — they are never cut short.

A single `walk_start` runs this **complete** depth in one call; the agent
above it does only **breadth pivots** (re-issue `walk_start` with different
vocabulary), never stage-by-stage micro-driving. The composable evidence
returned to the answerer is capped (`_MAX_EVIDENCE`, ranked by relevance
label then chain-vocabulary overlap) so a bloated relevant set never drowns
the synthesis — full recall is preserved in `fact_ids_cumulative`.

On LoCoMo (balanced per-category sampling) the walk lifts evidence
**agent_recall to ≈ 0.85** versus a static Recall@7 ≈ 0.31 — it surfaces
~2.7× the gold evidence of a single-pass retriever, and most of the
remaining answer error is metric artefact (token-F1 counts "progressive" ≠
"liberal" as a miss) or evidence labelled on a turn that carries no text
(an attached image). See `benchmarks/locomo/README.md` for the per-version
breakdown.

## One emergent law, four consolidations

The same Poisson floor (`λ + √λ`, no hyperparameter) drives four
self-organizing mechanisms — each a different kind of *deduplication*,
each opt-in and (where it removes nothing) non-destructive:

| mechanism | recurrence of… | → dedup of… | module |
|---|---|---|---|
| **Proximity / identity collision** | points that sit too close / restate each other | redundant **nodes** (fission or merge) | `collision.py` |
| **Lateral collision** | facts co-retrieved under *diverse* queries | wasted **kNN slots** (one keeper, members kept as children, expanded on-target) | `lateral.py` |
| **Spike–Chasles (reasoning)** | reasoning hops repeated along a path | **reasoning-path length** (A→B→C→D ⇒ A→M→D) | `spike.py` |
| **Spike–Chasles (workflows)** | action steps repeated across walks | **workflow step-count** (multi-step ⇒ one optimized `tool_workflow` node) | `spike.py` |
| **Memory skills / tools** | actions re-derived to answer diverse queries | redundant **capability** (one persistent tool node) | `skills.py` |

Two shared design notes: every law weights recurrence by **query
diversity** (an event that merely repeats a seen direction adds ~0, only
distant directions accumulate), and each is **gated** so it stays inert on
small clouds where "redundant" is undefined.

## Memory skills — self-built tools

A capability the system keeps re-deriving crystallizes into a **tool**:
a *normal node* (kind `ACTION`, tagged `tool` + a genre tag
`tool_code`/`tool_command`/`tool_api` + grounding-context tags), scoped by
its keywords. Being a normal node it lives in the same cloud — persisted
on save, found by the walk *recursively*, subject to lateral collision and
Chasles like anything else.

- **Two triggers.** *Recurrence* (`crystallize_skills`): an action
  re-derived across enough diverse queries earns a tool over time.
  *Eager* (`ensure_tool`): the "no tool → generate it → proceed" step —
  on first need, with no recurrence wait. Generation is unconstrained, so
  it never blocks; it only ever grows the self-built set.
- **Explicit tool-intent metacognition** (`assess_tool_intent`): reads a
  thought / generated response and judges whether it implies running code
  or a command underneath — i.e. whether there is a tool to generate.
- **Fast-path** (`match_tool`): an in-scope query returns its covering
  tool so the agent reuses the cached capability and **skips a think
  phase**.

```python
m.skills_enabled = True
m.ensure_tool("search scientific articles about X",
              how="query the article index by topic, rank by relevance")
# 1st call → generates tool_search_articles ; next time → reused, no rethink
```

## Quick start (in-process)

```python
from metacog import Memory

m = Memory(storage_path="memory.pkl")    # or omit for in-memory

m.ingest("dr sarah lives in berkeley", kind="FACT")
m.ingest("dr sarah works as a counselor", kind="FACT")

m.process_turn("who is dr sarah", speaker="user")
result = m.reason("where does dr sarah live?")
print(result["final_answer"])
print(m.audit())                          # zero laundering

# Opt-in self-organization
m.lateral_enabled = True                  # co-retrieval consolidation
m.skills_enabled  = True                  # tool crystallization
m.sleep()                                 # geometric + lateral collision pass
```

## MCP server

The same operations are exposed as an MCP service (21 tools) :

```
ingest              add a FACT / THOUGHT / ACTION
observe             apply an Observation on existing point(s)
process_turn        record a conversation turn (detectors fire)
retrieve            top-k hybrid retrieval
walk_start          run a complete uncertainty-governed walk for a query
walk_next           deprecated (walk_start now runs to completion); no-op
reason              full reasoning trajectory until output convenable
sleep               geometric collision + lateral collapse pass
match_tool          capability cache : is a generated tool covering this query?
ensure_tool         get a tool, generating it if absent ("no tool → generate it")
crystallize_skills  fold recurring actions into persistent tool nodes
list_tools_learned  list the self-built tool set
inspect             dump a point's full state (keywords, tags, spike count)
audit               verify no laundering
stats               system overview
declare_observator  manually declare an observator
detect_polarized    list polarized points
spawn_observators   auto-spawn from a polarized point
route               pick top-k observators for a query
list_communities    Level-1 community / observator detection
save                persist to disk
```

### Run the server

```bash
uv sync                                  # or pip install -e .
uv run metacog-mcp --storage ~/.metacog/state.pkl
```

The server listens on stdio (the standard transport for Claude Code).

### Register in Claude Code

```json
{
  "mcpServers": {
    "metacog": {
      "command": "metacog-mcp",
      "args": ["--storage", "~/.metacog/state.pkl"]
    }
  }
}
```

After restarting Claude Code, the tools appear as `mcp__metacog__*`.

## Tests

```bash
uv run pytest tests/ -v
# or
PYTHONPATH=. pytest tests/ -v
```

**311 tests** across 32 files — laundering invariants, manifold dynamics,
proximity/lateral/identity collision, spike & Chasles compression
(reasoning + workflow), memory-skill / tool crystallization, signal
detection, ACTION execution, reasoning trajectories, observator
multi-perspective, the MCP tool↔memory layer, entity-beacon co-location
and Corollary-5 compliance, and end-to-end scenarios.

## Module layout

| Module | Responsibility |
|---|---|
| `epistemic.py` | `Point`, `Observation`, `A(·)`, state machine, `PointKind`, tags |
| `geometry.py` | manifold ops, `apply_pull`, `apply_exile`, `retrieve_hybrid` |
| `keywords.py` | keyword extraction + position-weighted keyword embeddings |
| `bm25.py` / `fuzzy.py` | BM25 + fuzzy Levenshtein retrieval signals |
| `meta_walk.py` | the coordinated FACT/THOUGHT/ACTION walk + HyDE channel |
| `collision.py` | proximity collision, N-way fission, identity merge, Chasles anchors |
| `lateral.py` | co-retrieval (functional) collision → keepers + on-target expansion |
| `spike.py` | spike-and-path : reasoning & workflow Chasles compression |
| `skills.py` | memory skills : tool crystallization, intent metacognition, fast-path |
| `compression.py` | retrospective trajectory Chasles compression |
| `entities.py` / `atomic.py` | entity beacons & mem0-style atomic-fact handles |
| `communities.py` | Level-1 community / observator detection |
| `detectors.py` | deterministic conversation signals |
| `execution.py` / `executor.py` | ACTION execution + result-FACT lineage |
| `reasoning.py` | `reason()` orchestrator |
| `observator.py` | observators, polarization, delegation |
| `uncertainty.py` | β-uncertainty + keyword-order σ |
| `audit.py` | non-laundering verification |
| `memory.py` | top-level `Memory` wrapper |
| `defaults.py` / `llm.py` | `SimpleEncoder`, `NoOpExecutor`, `ClaudeLLM` |
| `mcp_server.py` | MCP service |

See `benchmarks/locomo/` for the LoCoMo harness, the fast inspectable
debuggers (`debug_walk.py`, `debug_lateral.py`), and the full results
table (V5 baseline → V7 walk-depth adaptive + structured date refs).

## References

- Romanchuk & Bondar, "Semantic Laundering in AI Agent Architectures",
  arXiv:2601.08333.
- Zhu et al., "HeLa-Mem", arXiv:2604.16839 (we keep the spirit, remove the
  hyperparameters and the materialized edges).
- Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without Relevance
  Labels" (HyDE).
- Yu et al. 2024, "Chain-of-Note".
- Cormack et al. 2009, "Reciprocal Rank Fusion".
- Ramsauer et al. 2020, "Hopfield Networks is All You Need".
```

