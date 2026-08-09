# DOX framework

- DOX is a highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read this root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- This root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are none yet, leave it empty
- Verification must reflect an existing check; if none exists yet, leave it empty and update it when one exists

Default section order: Purpose · Ownership · Local Contracts · Work Guidance · Verification · Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across files unless each scope needs a local version
- Delete stale notes instead of explaining history

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

---

# Project Rail — OmniCognition / MetaCog-Mem

Manifold-based metacognitive memory for LLM agents. These project-wide rules
bind every subtree; child docs add local detail but may not weaken them.

## Invariants (non-negotiable)

- **Edge-free.** Relations are geometric proximity via `apply_pull`, not stored
  edges. No `edges` field — co-location *is* the edge.
- **Hyperparameter-free.** Every threshold is a mathematical constant or emerges
  from the data (e.g. the walk's σ-cap from GUM uncertainty propagation). Do not
  introduce tunable magic numbers.
- **Anti-laundering (Cor. 5).** Provenance is typed. `SourceClass.GENERATOR`
  content can become *content* but never *evidence*. Generated nodes
  (THOUGHT / ACTION / event hubs / beacons) must be created with `apply_pull`
  directly, never through an `Observation`. Enforced by `tests/test_no_laundering.py`.
- **LLM extractors** are cached, fully failure-safe (`try/except -> []`), and
  **never cache an empty result**.
- **Persistence:** `Memory.save()` pickles a whitelist (`points`, observators,
  conversation_log, clocks, `decay_exponent`, `_forget_log`); registries/bags/
  threads are rebuilt in `load()`. The SQL **usage journal is a SEPARATE SQLite
  file** (`<storage>.journal.db`), never pickled — it persists on its own and is
  re-attached on construction.
- **mnema usage journal (opt-in, failure-safe).** `metacog/journal.py` is an
  append-only access-log separate from the store. Structural signals are SQL
  queries (co-retrieval self-join, `path_traversals` for Chasles, hierarchical
  `tags`, `forget_events`). Every consumer degrades to the in-memory path or a
  no-op without a journal — **behaviour must be unchanged when it is absent**.
  ACT-R ranking levers (`recency_weight`/`spreading_weight`) and the explicit
  `forget`/decay-prune are all **OFF/opt-in by default** (mnema's defaults).
- **Tool surface.** `metacog/canonical_tools.py` partitions tools into T1/T2/T3
  (guarded by `tests/test_canonical_tools.py`). `build_app(surface=…)` /
  `METACOG_SURFACE` gates what the MCP EXPOSES; unexposed tools stay callable
  internally. A new `@app.tool()` must be classified or the partition test fails.

## Commands

```bash
pip install -e .                 # runtime deps (mcp, nltk)
pip install -e '.[bench]'        # + sentence-transformers, torch, anthropic
python -m pytest tests/ -q       # full suite (deterministic, no network)
python -m metacog.mcp_server     # MCP service (entrypoint: metacog-mcp)
```

Benchmarks need `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN`. Tests must not.

## Two answer modes (drives evaluation)

A query is either **"generate a focused answer"** (precision/F1; the
uncertainty-stopped walk excels) or **"refer an exhaustive set"** (recall; a
NON-kNN filtered listing — `Memory.filter_list` by event/date/tags/kind —
excels), and they can coexist (answer + bag/list). The exhaustive set is reached
by SCAN over parameters, not similarity top-k : the walk's meta-cognition/ACTION
sees the input relates to an event and launches the targeted filtered query.
Report recall **and** precision/F1, always with set size `n`. The walk's
evidence set is uncertainty-bounded and **must not be hard-capped** — a subject
can have arbitrarily many relevant items.

The exhaustive set is assembled by an AGENTIC loop, not by reading every event :
presearch (scoped, `knowledge_base=False`) surfaces the context events, then
`search_nodes` draws candidates from the filtered pool by sim/regex/fuzzy over
content AND tags ; the agent judges relevance and `collect`s the hits into the
bag. Drawing is WITHOUT REPLACEMENT — a collected node is excluded from later
passes (`exclude_bag`), so the pool shrinks and the loop converges. The whole
loop is orchestrated by `Memory.assemble_set` (auto-route the event, cycle the
modes, judge relevance, collect, converge) — one call returns the assembled bag.

## User Preferences

- Develop only on the designated feature branch; never push to another branch
  without explicit permission.
- The event channel is **additive** — it unions into agent-recall, never
  replaces the walk.

## Child DOX Index

- `metacog/AGENTS.md` — the core library (Point manifold, memory, walk, event
  subsystem, MCP surface)
- `benchmarks/AGENTS.md` — benchmark harnesses and debuggers (indexes locomo,
  obliq_bench)
- `tests/AGENTS.md` — the deterministic test suite and its invariants
- `docs/AGENTS.md` — durable research and literature notes
