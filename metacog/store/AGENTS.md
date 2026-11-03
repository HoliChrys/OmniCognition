# metacog/store — the storage interface (OO·V2·M1, Linear TAC2-1153)

## Purpose

Where a `Memory`'s state is kept. `Memory` never opens its store itself: it
talks to ONE `StoreBackend`. Today one backend, `PickleBackend` (the former
`save()`/`load()` code, moved unchanged); the Neo4j backend of OO·V2·M3 plugs
in here.

## Ownership

Owns `base.py` (the `StoreBackend` protocol, `CorruptStoreError`,
`StoreNotFound`, `point_fingerprint` / `diff_points`), `pickle_backend.py`
(`PickleBackend`, the per-store lock `_store_lock`, `_read_store`) and
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
- **For the next backend** (OO·V2·M3): `discard` must archive, never delete
  (zero deletion); `commit` that cannot reach the store raises
  `CorruptStoreError`; `read_if_changed` is the optimistic-version check
  (a conflict re-reads and merges, as the pickle does); `load` is per
  context (`location`). With per-point writes, another writer's change to an
  EXISTING point survives — the pickle's "ours wins" on a clash does not
  carry over, and `test_store_concurrency` is rewritten case for case. The
  call sites save only `if memory.storage_path:` — a backend injected
  without a `storage_path` is never committed by them.
- **Decisions not taken here**: D9 (the pickle kept for tests and the local
  plugin) is a working hypothesis, reversible.

## Work Guidance

- Behaviour of the pickle path is frozen: any change must keep
  `test_store_concurrency` (C1–C8), `test_snapshot_isolation`,
  `test_forget_durable`, `test_encoder`, `test_context_isolation` green
  WITHOUT modification, and `benchmarks/storage_baseline` IDENTIQUE.

## Verification

`python -m pytest tests/test_store_backend.py tests/test_store_concurrency.py
-q`; the oracle: `benchmarks/storage_baseline` (`build` → `run` on the base →
`run` on the candidate → `compare` rc 0).

## Child DOX Index

No children.
