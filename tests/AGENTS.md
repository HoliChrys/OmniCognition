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
- `test_canonical_tools.py` asserts the tool-tier manifest partitions the live
  `@app.tool()` set EXACTLY — a new tool must be classified or it fails. The
  mnema-layer tests (`test_feedback_loop`, `test_recency_ranking`,
  `test_spreading_activation`, `test_forget`, `test_forget_node`,
  `test_abstention`, `test_tool_lifecycle`) all assert the OFF/opt-in default is
  behaviour-neutral. `test_wiki` covers the OKF layer: feed/render/parse, refs in
  frontmatter+inline+DB, `reconcile_wiki` rewriting refs on merge, wiki->RAG
  ingest, and the EAV field index (query by any field, schema recovered, no
  migrations).

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
