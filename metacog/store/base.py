"""The storage interface of a `Memory` (OO·V2·M1, Linear TAC2-1153).

A `Memory` never touches its store directly : it talks to ONE `StoreBackend`
through six verbs —

* `lock()`            the store's single-writer lock (a context manager,
                      re-entrant within a process) ;
* `load()`            the stored state of the context, as the whitelisted
                      snapshot dict (`points`, `observators`,
                      `conversation_log`, `_t_clock`, `_spike_total_hops`,
                      `decay_exponent`, `_forget_log`, `encoder_id`) ;
                      `FileNotFoundError` (or `StoreNotFound`) when nothing
                      was ever written, `CorruptStoreError` when what is there
                      cannot be read — never an empty state passed off as a
                      loaded one ;
* `read_if_changed()` under the lock : the stored state IF another writer
                      changed it since this backend last loaded/committed it,
                      else None — the input of `Memory`'s merge-on-stale ;
* `upsert(point)`     write ONE point (new or changed since the last sync) ;
* `forget(id, entry)` record ONE forget (the `_forget_log` entry ; the
                      point's INVALID state travels in its own `upsert`) ;
* `discard(id)`       a point the memory no longer holds (collision merge,
                      opt-in decay-prune, tool removal). A backend bound by
                      zero-deletion maps it to a non-destructive archive ;
* `commit(state)`     end of the unit of work : make every write above durable
                      together, with the non-point state of the context. A
                      backend that cannot reach its store raises
                      `CorruptStoreError` (fail-closed).

`Memory.save()` is the unit of work : under `lock()`, it merges what another
writer added (`read_if_changed`), then expresses the change since the last
sync as point writes (`upsert` / `forget` / `discard`, computed by
`diff_points` over `point_fingerprint`), then `commit`s. A batching backend
(`PickleBackend`) groups them into one atomic file ; an incremental one
(OO·V2·M3) writes only those points.

Selection : `METACOG_STORE` (default `pickle`), read by `open_backend` when a
`Memory` with a `storage_path` first needs its store. An unknown name raises.
"""
from __future__ import annotations

import enum
from typing import Any, ContextManager, Dict, Iterable, List, Optional, Protocol, Tuple, runtime_checkable


class CorruptStoreError(RuntimeError):
    """The store file exists but cannot be read back (truncated, not a pickle,
    wrong shape). Raised instead of serving an empty memory ; the file is left
    untouched for recovery."""


class StoreNotFound(FileNotFoundError):
    """Nothing was ever written for this store (first run). A subclass of
    `FileNotFoundError`, so `Memory`'s first-run path catches it unchanged."""


@runtime_checkable
class StoreBackend(Protocol):
    """What a `Memory` needs from its store. See the module docstring."""

    kind: str          # "pickle", …
    location: str      # what an error names (a path, a context)

    def lock(self) -> ContextManager[Any]: ...

    def load(self) -> Dict[str, Any]: ...

    def read_if_changed(self) -> Optional[Dict[str, Any]]: ...

    def upsert(self, point: Any) -> None: ...

    def forget(self, point_id: str, entry: Any) -> None: ...

    def discard(self, point_id: str) -> None: ...

    def commit(self, state: Dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Change detection : which points changed since the last sync.
# ---------------------------------------------------------------------------

_ATOMS = (str, int, float, bool, bytes, type(None))


def _leaf(v: Any) -> Any:
    """Value-snapshot of a nested value, one level deep (lists copied, small
    objects such as `ObservatorView` read field by field)."""
    t = type(v)
    if t in _ATOMS or t is tuple or isinstance(v, enum.Enum):
        return v
    if t is list:
        return tuple(v)
    if t is dict:
        return tuple(v.items())
    d = getattr(v, "__dict__", None)
    if d is not None:
        return (t, tuple(d.items()))
    return v


def _frozen(v: Any) -> Any:
    t = type(v)
    if t in _ATOMS or t is tuple or isinstance(v, enum.Enum):
        # tuples (the vectors) are immutable : kept BY REFERENCE, so an
        # unchanged vector compares by identity, never float by float.
        return v
    if t is list:
        return tuple(v)                 # tags, parents, update_log (refs)
    if t is dict:
        return tuple((k, _leaf(x)) for k, x in v.items())
    try:
        return tuple(v)                 # any other sequence (an array…)
    except TypeError:
        return _leaf(v)


def point_fingerprint(p: Any) -> tuple:
    """A value-snapshot of every field of a point : equal ⇔ unchanged.

    Atoms and tuples are kept as they are (a vector is REPLACED, never mutated
    in place — `metacog/AGENTS.md`), lists are copied, dicts are read one level
    deep. Holding the references keeps them alive, so identity never aliases.
    """
    return tuple((k, _frozen(v)) for k, v in vars(p).items())


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:                   # an un-comparable field : say changed
        return False


_MISSING = object()


def diff_points(points: Iterable[Any], base: Dict[str, tuple]
                ) -> Tuple[List[Any], List[str], Dict[str, tuple]]:
    """(changed points in memory order, ids held at the base and gone since,
    the fingerprints of `points`). A point is changed when it is new or its
    fingerprint differs from the base's."""
    changed: List[Any] = []
    fps: Dict[str, tuple] = {}
    for p in points:
        fp = point_fingerprint(p)
        fps[p.id] = fp
        old = base.get(p.id, _MISSING)
        if old is _MISSING or not _same(old, fp):
            changed.append(p)
    gone = [i for i in base if i not in fps]
    return changed, gone, fps
