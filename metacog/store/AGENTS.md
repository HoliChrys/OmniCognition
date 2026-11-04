# metacog/store — the storage interface (OO·V2·M1, Linear TAC2-1153)

## Purpose

Where a `Memory`'s state is kept. `Memory` never opens its store itself: it
talks to ONE `StoreBackend`. Two backends: `PickleBackend` (the former
`save()`/`load()` code, moved unchanged — tests and the local plugin, D9) and
`Neo4jBackend` (OO·V2·M3, Linear TAC2-1157: the store as governed `memory_*`
records of the ontology, written and read through tachikoma's batch route).

## Ownership

Owns `base.py` (the `StoreBackend` protocol, `CorruptStoreError`,
`StoreNotFound`, `point_fingerprint` / `diff_points`), `pickle_backend.py`
(`PickleBackend`, the per-store lock `_store_lock`, `_read_store`),
`neo4j_backend.py` (`Neo4jBackend`, `StoreConflictError`, the point codec
`encode_point`/`decode_point`, `scope_of`, the `store_token` contextvar) and
`__init__.py` (`open_backend`, `BACKENDS`, `METACOG_STORE`). `Memory.save()`
/ `load()` / `_merge_from_disk` / `_stage_point_writes` (in `memory.py`) own
the merge semantics and the unit of work; this package owns where the bytes
go. Inherits every root invariant (zero deletion, fail-closed, single writer).

## Local Contracts

- **The protocol** (`StoreBackend`): `kind`, `location`; `lock()` (the
  store's one writer lock, re-entrant in a process); `load()` → the
  whitelisted snapshot dict, `FileNotFoundError`/`StoreNotFound` when never
  written, `CorruptStoreError` when unreadable; `read_if_changed()` (under the
  lock) → the stored state if another writer changed it since this backend's
  last load/commit, else None; `upsert(point)`, `forget(id, entry)`,
  `discard(id)` — the point writes; `commit(state)` — makes them durable
  together with the context state (non-point keys of the snapshot).
- **Unit of work = `Memory.save()`**: lock → merge-on-stale → point writes
  (the diff of `point_fingerprint` against what the backend holds) → one
  commit. A call site never declares what it wrote: the diff is exhaustive
  (a neighbour moved by `apply_pull` is upserted), so adding a site is
  `memory.save()`, nothing else.
- **`point_fingerprint`** keeps tuples (the vectors) BY REFERENCE and copies
  lists / reads dicts one level deep: it relies on vectors being REPLACED,
  never mutated in place (`metacog/AGENTS.md`, geometry).
- **`PickleBackend`**: point writes are counted (`stats`, `last_commit`) and
  grouped; `commit` writes the whole snapshot `<store>.tmp` + fsync +
  `os.replace` + directory fsync; stale detection by the file signature
  (inode, size, mtime_ns); an absent or empty file is "nothing to merge";
  `pickle.dump`/`pickle.load` are looked up at call time (the crash tests
  patch them). The file format is unchanged: a store written before
  OO·V2·M1 loads, and the bytes written are `pickle.dumps(snapshot)`.
- **Selection**: `METACOG_STORE` (default `pickle`) is read by
  `open_backend` when a `Memory` with a `storage_path` first needs its store;
  an unknown name raises `ValueError` (never a silent fallback). An explicit
  `save(path)` / `load(path)` on another path is always a pickle
  export/import.
- **`Neo4jBackend`** (`METACOG_STORE=neo4j`): NO direct Neo4j writer — every
  read and write goes through tachikoma's API (D6, working hypothesis):
  `GET /api/memory/points` (load: every point of the store, an exact scan, no
  `k` — D3), `GET /api/memory/points:revision` (the `read_if_changed` fast
  path), `POST /api/memory/points:batch` (one batch per `commit`: the staged
  points, the discards, the context state and every observator). The store is
  derived from the gate's path layout (`scope_of`: `<root>/<ctx>/memory.pkl`,
  `<root>/<ctx>/accounts/<acc>/memory.pkl`); `METACOG_NEO4J_NAMESPACE` prefixes
  the context — tests and the oracle only. The bearer is the store's OWNER:
  `OMNI_STORE_TOKEN`/`OMNI_STORE_TOKEN_FILE` for a context store, the
  account's own token (`store_token` contextvar or the backend's `_token`) for
  a narrow account's store; none → `CorruptStoreError`.
- **Neo4j invariants**: zero deletion — `discard` archives the record
  (`discarded = True`, no longer served by a load, never deleted), a forget is
  the point's own `INVALID` + `invalidated` plus the `_forget_log` entry on the
  `memory_store` record; fail-closed — API or base unreachable, an unreadable
  answer, a row that does not decode, points without a `memory_store` →
  `CorruptStoreError`, never an empty memory; the codec is EXACT (floats
  round-trip through JSON, tuples/None restored field by field, the point
  order kept by `ordinal`), proven by the oracle (`neo4j.py` in
  `benchmarks/storage_baseline`). Identity (P31): the node id is qualified by
  the store on tachikoma's side (`mem:<ctx>:<id>`); omni's id is `point_id`.
- **The lock between processes is the store's revision**: `commit` sends the
  revision it read; another writer committed since → HTTP 409 →
  `StoreConflictError`, nothing written; `Memory.save()` redoes the unit of
  work (re-read, merge, write) until it commits — each conflict means another
  writer progressed. `lock()` is per process. With per-point writes, another
  writer's change to an EXISTING point that we did not change survives — the
  pickle's whole-file "ours wins" does not carry over (`test_store_concurrency`
  has each case C1–C8 twice: pickle, and `neo4j`). The call sites save only
  `if memory.storage_path:` — a backend injected without a `storage_path` is
  never committed by them.
- **Migrating served stores** (OO·V2·M4, Linear TAC2-1160):
  `scripts/migrate_pickle_to_neo4j.py` is the ONLY tool that moves a served
  `memory.pkl` into `Neo4jBackend` (the oracle's `benchmarks/storage_baseline/
  neo4j.py migrate` is for fresh test namespaces). It reads the pickle with
  `_read_store` (read-only fd, no `Memory`, no `.lock`, sha256 before/after),
  encodes with THIS package's codec (`encode_point` with ordinal = pickle
  position, `encode_observator`, `encode_store`), diffs against the load route
  with the route's `_same` rule, and writes chunks through the batch route
  under the read revision; dry-run by default, `--apply`, `--verify` (reload
  by `Neo4jBackend.load`, fingerprints + digests + journal-forget replay on
  both sides). A store whose Neo4j `t_clock` is ahead of the pickle's is
  refused, and so is one holding points the pickle lacks unless
  `--discard-missing` (archived, never deleted). A codec change must keep its `--verify` IDENTIQUE
  (`tests/test_migrate_pickle_to_neo4j.py`). The usage journal is NOT
  migrated: on `Neo4jBackend` the gate still opens `<storage_path>.journal.db`
  (the storage path keeps the gate's layout).
- **Decisions not taken here**: D3 (exact scan), D6 (writes through the API)
  and D9 (the pickle kept for tests and the local plugin) are working
  hypotheses, reversible.

## Work Guidance

- Behaviour of the pickle path is frozen: any change must keep
  `test_store_concurrency` (C1–C8), `test_snapshot_isolation`,
  `test_forget_durable`, `test_encoder`, `test_context_isolation` green
  WITHOUT modification, and `benchmarks/storage_baseline` IDENTIQUE.

## Verification

`python -m pytest tests/test_store_backend.py tests/test_store_concurrency.py
-q`; the oracle: `benchmarks/storage_baseline` (`build` → `run` on the base →
`run` on the candidate → `compare` rc 0). Neo4j: the same files with `-m neo4j`
and `OMNI_TEST_API_URL` / `OMNI_TEST_TOKEN_FILE` / `OMNI_TEST_NEO4J_HTTP`
(`tests/neo4j_live.py`), plus `test_context_isolation.py`; the oracle by
`benchmarks/storage_baseline/neo4j.py migrate` then `run --memory-factory
benchmarks.storage_baseline.neo4j:factory`. The migration tool:
`python -m pytest tests/test_migrate_pickle_to_neo4j.py -q` (offline), and on a
live API `scripts/migrate_pickle_to_neo4j.py --root … --apply --verify` twice
(the second plans 0).

## Child DOX Index

No children.
