# `metacog/` — the MetaCog-Mem core

This package is the memory manifold and everything that operates on it. It
has no required hyperparameters: every threshold is a mathematical constant
or emerges from the data. The top-level [`README`](../README.md) gives the
paper-level overview; this document is the **module map and the invariants
each module must preserve**.

## The one schema

Everything is a `Point` (`epistemic.py`). A point of any kind — `FACT`,
`THOUGHT`, `ACTION`, or a self-built tool — carries:

- `embedding_orig` — the raw sentence embedding;
- `keywords` + `keywords_embedding` — the **position-weighted** keyword
  projection (salience-ordered, `1/(i+1)` decay), the handle the walk
  retrieves on;
- a Beta posterior over `n_corrob` / `n_contra` counters → its epistemic
  **σ** (`uncertainty.py`);
- `state` ∈ {`CONJECTURE`, `CORROBORATED`, `INVALID`, `DEPRECATED`, …};
- an open `tags` list (`tool`, `tool_workflow`, `entity`, `atomic`,
  `lateral_absorbed`, `generated`, …) that refines role over time;
- lineage (`parents`, `sequence_prev/next`) and the pulled offsets
  (`delta_active`, `delta_latent`) that *are* the edge-free relations.

## Two invariants, enforced

1. **Anti-laundering (Corollary 5).** `Observation`s carry a `SourceClass`;
   `Observation(source=GENERATOR)` raises `LaunderingError`. LLM output may
   become a `THOUGHT`/`ACTION` (content ∈ P) but never an observation (∈ O)
   that updates counters. `audit.py` verifies this across a whole store.
2. **Zero deletion.** Nothing is removed. `INVALID`/`DEPRECATED` exile a
   point geometrically; re-corroboration resurrects it.

## Module map

### Substrate
| module | responsibility | key invariant |
|---|---|---|
| `epistemic.py` | `Point`, `Observation`, `A(·)`, state machine, `PointKind`, tags | `GENERATOR`→`Observation` is impossible |
| `geometry.py` | manifold ops — `apply_pull`, `apply_exile`, effective embedding, `retrieve_hybrid`, lineage spread | relations are pulls, not edges |
| `uncertainty.py` | β-uncertainty ⊕ keyword-order σ; GUM `propagate`; emergent `prune_threshold` | thresholds emerge (`median ± σ`) |
| `keywords.py` | LLM / frequency keyword extraction; position-weighted embedding | **retry + never cache empty** (silent-failure discipline) |
| `bm25.py`, `fuzzy.py` | content-first BM25 and fuzzy Levenshtein retrieval signals | BM25 indexes raw content, not keyword summaries |

### Retrieval & reasoning
| module | responsibility |
|---|---|
| `meta_walk.py` | the coordinated FACT/THOUGHT/ACTION walk: HyDE channel, Chain-of-Note MAP-REDUCE, thought chain, **uncertainty-governed depth** (floor / σ-cap / coverage-stop), evidence cap |
| `reasoning.py` | `reason()` orchestrator over the walk |
| `compression.py` | retrospective trajectory Chasles compression |
| `anchors.py` | kind-agnostic semantic perception layer for the walk |

### Self-organization (one Poisson floor, `λ + √λ`)
| module | responsibility |
|---|---|
| `collision.py` | proximity collision, N-way fission, identity merge, Chasles anchors |
| `lateral.py` | co-retrieval (functional) collision → keepers + on-target child expansion |
| `spike.py` | spike-and-path: reasoning- and workflow-Chasles compression |
| `skills.py` | memory skills: tool crystallization, tool-intent metacognition, `match_tool` fast-path |
| `communities.py` | Level-1 community / observator detection |

### Perception, perspective, execution
| module | responsibility |
|---|---|
| `detectors.py` | deterministic conversation signals (no LLM) |
| `entities.py`, `atomic.py` | edge-free entity beacons & mem0-style atomic-fact handles |
| `observator.py` | observators: per-source parallel views, polarization, delegation |
| `execution.py`, `executor.py` | ACTION execution + result-FACT lineage |

### Edges of the package
| module | responsibility |
|---|---|
| `memory.py` | the top-level `Memory` wrapper (ingest, observe, walk, sleep, save/load) |
| `llm.py` | `ClaudeLLM` — the only LLM adapter; **retries transient 5xx/overloaded**, outputs always typed `GENERATOR` |
| `defaults.py` | `SimpleEncoder`, `NoOpExecutor`, default extractors |
| `audit.py` | non-laundering verification |
| `mcp_server.py` | the MCP service (below) |

## The walk in one paragraph (`meta_walk.py`)

`MetaWalker.step()` retrieves (hybrid + HyDE), labels relevance
(Chain-of-Note), folds on-target facts into the persistent
`_relevant_cum`, and generates a `THOUGHT` whose keywords come **only**
from walked points. After a floor of `_MIN_STAGES = 3`, the walk stops when
the propagated `σ_path` over the fact★ coherence-hops exceeds the emergent
cutoff **or** when the gathered keywords cover the query. `walk_start`
(in `mcp_server.py`) loops `step()` to completion and returns the bounded
composable evidence (`_MAX_EVIDENCE = 15`) plus at most
`_MAX_RETURNED_FACTS = 20` retrieved turns and the reasoning chain — full
recall stays in `fact_ids_cumulative`.

## Named skills — theoretical tool directories (`memory.py` + `meta_walk.py`)

A **skill** is a named directory tree of tools, built and grown inside the
manifold (no separate store):

- `Memory.build_skill(query, user_id, session_id)` — **task-mode** walk:
  gold is a tool fact as readily as a semantic fact; the depth gate is
  `enough_tools_for_workflow` (LLM, per depth, fail-closed) instead of the
  QA σ-stop. Emits a NAMED skill folder (nested JSON), ingests it, and
  logs the resolution.
- `Memory.ingest_skill(tree, *, name, user_id, session_id, date)` —
  re-indexes the JSON tree as `FACT`s tagged
  `skill·tool·name:<name>·user:<id>·session:<id>·date:<…>`; topology via
  lineage **and** `ref:skill:<name>:path:/:parent:` tokens.
- **Double query** — `nearest_facts_with_fallback(..., section_filter=…)`
  RRF-merges a section-restricted retrieval (the `session:`/`user:`/`name:`
  tags) with the free query; `walk_start(user_id, session_id, …)` wires it.
- **Latent distiller** — `record_resolution()` logs (query, walk-path ids,
  output); `distill_skills()` (run in `sleep()` when `skills_enabled`)
  crystallizes a tool `ACTION` whose `parents` are the explicating
  facts/thoughts/actions, so the next recurrence is retrieved without
  forced metacognition. Idempotent via `_distill_cursor`.

## MCP tools (`mcp_server.py`)

```
ingest · observe · process_turn · retrieve
walk_start          run a COMPLETE uncertainty-governed walk (depth = σ);
                    user_id/session_id add the double-query section boost
walk_keepup         keepup streaming: provisional answer re-written each
                    stage until validated (SSE → a self-rewriting message)
walk_next           deprecated — walk_start runs to completion
reason · sleep
ingest_skill        re-index a named skill-JSON directory tree
build_skill         task-mode walk → synthesise + ingest a named skill
get_session_skill   the skill JSON cached for this (user, session)
match_tool · ensure_tool · crystallize_skills · list_tools_learned
inspect · audit · stats
declare_observator · detect_polarized · spawn_observators · route
list_communities · save
```

## Touching this package

- A new retrieval signal goes through `retrieve_hybrid` (RRF), never a new
  bespoke ranker.
- Any LLM call must **retry on transient errors and never cache an empty
  result** — the single largest regression this codebase suffered was a
  cached `[]` from one 529.
- A new "threshold" must emerge from the population or be a math constant —
  if you find yourself adding a tunable number, derive it instead.
- Generated text is `source=GENERATOR`. If you wrap it in an `Observation`,
  the audit will (correctly) fail.
