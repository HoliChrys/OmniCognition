"""metacog.store — the storage interface of a `Memory` (OO·V2·M1, TAC2-1153).

`StoreBackend` (the protocol), `PickleBackend` (the one-file store, default),
`Neo4jBackend` (the ontology's records, through tachikoma's API — OO·V2·M3),
`open_backend(storage_path)` (the selection by `METACOG_STORE`).
"""
from __future__ import annotations

import os
from typing import Dict, Type

from metacog.store.base import (
    CorruptStoreError,
    StoreBackend,
    StoreNotFound,
    diff_points,
    point_fingerprint,
)
from metacog.store.neo4j_backend import Neo4jBackend, StoreConflictError
from metacog.store.pickle_backend import PickleBackend

ENV = "METACOG_STORE"
DEFAULT = "pickle"

# name -> class taking the store's path. `neo4j` : the ontology, through tachikoma's
# batch route (OO·V2·M3, `neo4j_backend.py`).
BACKENDS: Dict[str, Type] = {"pickle": PickleBackend, "neo4j": Neo4jBackend}


def backend_name() -> str:
    return (os.environ.get(ENV) or DEFAULT).strip().lower() or DEFAULT


def open_backend(storage_path: str) -> StoreBackend:
    """The backend `METACOG_STORE` names, for the store at `storage_path`.
    An unknown name raises (fail-closed) instead of falling back to a store
    the operator did not choose."""
    name = backend_name()
    cls = BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"{ENV}={name!r} : no such store backend "
            f"(available : {', '.join(sorted(BACKENDS))})")
    return cls(storage_path)


__all__ = [
    "BACKENDS", "CorruptStoreError", "DEFAULT", "ENV", "Neo4jBackend", "PickleBackend",
    "StoreBackend", "StoreConflictError", "StoreNotFound", "backend_name", "diff_points",
    "open_backend", "point_fingerprint",
]
