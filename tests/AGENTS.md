# tests — the deterministic suite

## Purpose

The full pytest suite (~470+ tests) guarding `metacog`'s invariants and
behavior. Deterministic and offline.

## Ownership

Owns all unit/integration tests and shared fixtures (`conftest.py`).

## Local Contracts

- **No network, no live LLM.** Tests use scripted fake LLMs/extractors and
  `SimpleEncoder`. A test that needs `ANTHROPIC_API_KEY` is wrong.
- **Determinism.** No randomness without a fixed seed; no wall-clock dependence.
- `test_no_laundering.py` enforces Cor. 5 — a manual GENERATOR `Observation` must
  still raise `LaunderingError`. Never weaken it.
- Extractor tests assert "never cache an empty result".
- Persistence tests assert registries/bags are rebuilt on `load()` (the pickle
  whitelist is points + observators + conversation_log + clocks + decay_exponent
  + `_forget_log`; the SQL journal is a separate file, re-attached on construct).
  `test_store_concurrency.py` guards the store write path: stale instances and
  20 concurrent processes/threads lose no fact, a SIGKILL mid-dump leaves a
  loadable store, a corrupt store raises `CorruptStoreError` and is untouched.
  `test_tachikoma_gate.py` pins ONE `Memory` per context at a concurrent first
  access (20 threads → 1 instance; 20 concurrent `remember` → exactly +20),
  and (TAC-353) that a member's write → forget → recall under the context's
  account no longer serves the mirror, on disk too.
  `test_forget_durable.py` (TAC-323) guards that a forget survives a restart:
  MCP `forget` → a new instance does not serve the id; a pickle that lost a
  forget (pending or merged event) is replayed INVALID at `load`; a reverted
  or already-pickled forget is not replayed; a stale writer keeps the forget.
- `test_canonical_tools.py` asserts the tool-tier manifest partitions the live
  `@app.tool()` set EXACTLY — a new tool must be classified or it fails. The
  mnema-layer tests (`test_feedback_loop`, `test_recency_ranking`,
  `test_spreading_activation`, `test_forget`, `test_forget_node`,
  `test_abstention`, `test_tool_lifecycle`) all assert the OFF/opt-in default is
  behaviour-neutral. `test_wiki` covers the OKF layer: feed/render/parse, refs in
  frontmatter+inline+DB, `reconcile_wiki` rewriting refs on merge, wiki->RAG
  ingest, and the EAV field index (query by any field, schema recovered, no
  migrations).
- `test_recall_vectorised.py` keeps the pre-vectorisation recall code verbatim
  as reference and asserts the numpy/memoised path returns the same pools. Its
  real-store case runs only with `METACOG_STORE_COPY=<copy of a store>` (needs
  fastembed) — point it at a COPY, never at a live store.
- `test_context_isolation.py` (C1, TAC-934) proves per-context isolation on
  the FOUR lanes — recall, capture, wiki, index — through the real gated app
  over MCP streamable HTTP, measured in the journals and stores of ALL eight
  contexts of the storage root. Every negative assertion is doubled by its
  positive (the fact IS in c; the scan DID cover eight journals), and a
  missing context header is a 400 that writes nothing. It is the `Done`
  condition of the roadmap's A/B/D tickets: never skip, xfail or weaken it.
  It runs in CI on every push and PR (`.github/workflows/context-isolation.yml`).

## Work Guidance

- A behavior change in `metacog` requires updating or adding a test here in the
  same pass.
- New extractors get a `Fake<Thing>Extractor` (source=GENERATOR) mirroring the
  existing ones.

## Verification

`python -m pytest tests/ -q` must be green before any commit. Targeted runs by
module during development.

## Child DOX Index

No children.
