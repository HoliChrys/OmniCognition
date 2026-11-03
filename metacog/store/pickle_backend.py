"""`PickleBackend` — the store of a `Memory` as ONE pickle file (TAC-935).

This is the persistence code `Memory.save()` / `load()` always had, moved
behind `StoreBackend` without a change of behaviour (OO·V2·M1, TAC2-1153) :

* ATOMIC : `commit` writes `<store>.tmp`, fsyncs it, `os.replace`s it over
  the store and fsyncs the directory. A process killed mid-write leaves the
  previous complete file or the new complete one — never a truncated pickle.
* ONE WRITER PER STORE : `lock()` is the one lock of the store in this process
  (thread RLock) + an `flock` on `<store>.lock` (other processes), taken at
  the outermost level only. The lock file is never deleted.
* STALE DETECTION : the version of the file this backend last read or wrote
  is its signature (inode, size, mtime_ns) ; `read_if_changed` re-reads the
  file only when the signature moved (an empty or absent file holds nothing).
* FAIL-CLOSED : a file that cannot be unpickled into a dict raises
  `CorruptStoreError` naming it ; the file is left untouched.

The point writes (`upsert` / `forget` / `discard`) are GROUPED : the file is
the whole snapshot, so `commit` writes it once whatever they were. They are
counted (`stats`, `last_commit`) so the interface traffic is measurable.
"""
from __future__ import annotations

import os
import pickle
import threading
from typing import Any, Dict, Optional, Tuple

from metacog.store.base import CorruptStoreError

try:                                    # POSIX ; elsewhere the thread lock alone
    import fcntl as _fcntl
except ImportError:                     # pragma: no cover - non-POSIX
    _fcntl = None


def _read_store(path: str) -> Tuple[Dict[str, Any], os.stat_result]:
    """Unpickle the store at `path` and return (snapshot, stat of the version
    read). FileNotFoundError propagates ; every other failure is a
    `CorruptStoreError` naming the file. The stat comes from the open fd, so it
    describes exactly the bytes that were read (an atomic replace in between
    cannot desynchronise them)."""
    with open(path, "rb") as f:
        st = os.fstat(f.fileno())
        try:
            snapshot = pickle.load(f)
        except Exception as e:
            raise CorruptStoreError(
                f"unreadable memory store {path} ({st.st_size} bytes) : "
                f"{type(e).__name__}: {e}") from e
    if not isinstance(snapshot, dict):
        raise CorruptStoreError(
            f"unreadable memory store {path} : expected a dict snapshot, "
            f"got {type(snapshot).__name__}")
    return snapshot, st


def _fsync_dir(path: str) -> None:
    """fsync the directory holding `path`, so the os.replace itself is durable."""
    fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class _StoreLock:
    """Exclusive, re-entrant lock on ONE store file : a thread RLock (threads
    of this process) + an `flock` on `<store>.lock` (other processes) taken at
    the outermost level only. The lock file is never deleted (deleting it would
    let two processes lock two different inodes)."""

    def __init__(self, store_path: str) -> None:
        self.lock_path = f"{store_path}.lock"
        self.rlock = threading.RLock()
        self.depth = 0
        self.fd: Optional[int] = None

    def __enter__(self) -> "_StoreLock":
        self.rlock.acquire()
        try:
            if self.depth == 0 and _fcntl is not None:
                fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
                try:
                    _fcntl.flock(fd, _fcntl.LOCK_EX)
                except BaseException:
                    os.close(fd)
                    raise
                self.fd = fd
        except BaseException:
            self.rlock.release()
            raise
        self.depth += 1
        return self

    def __exit__(self, *exc: Any) -> None:
        self.depth -= 1
        try:
            if self.depth == 0 and self.fd is not None:
                fd, self.fd = self.fd, None
                try:
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
                finally:
                    os.close(fd)
        finally:
            self.rlock.release()


_STORE_LOCKS: Dict[str, _StoreLock] = {}
_STORE_LOCKS_GUARD = threading.Lock()


def _store_lock(path: str) -> _StoreLock:
    """The ONE lock of a store, shared by every Memory of this process that
    points at it (keyed by real path)."""
    key = os.path.realpath(path)
    with _STORE_LOCKS_GUARD:
        lk = _STORE_LOCKS.get(key)
        if lk is None:
            lk = _STORE_LOCKS[key] = _StoreLock(key)
        return lk


def _signature(st: os.stat_result) -> tuple:
    return (st.st_ino, st.st_size, st.st_mtime_ns)


class PickleBackend:
    """One `memory.pkl` per store. See the module docstring."""

    kind = "pickle"

    def __init__(self, path: str) -> None:
        self.location = path
        self._sig: Optional[tuple] = None      # version last read / written
        self._batch = {"upsert": 0, "forget": 0, "discard": 0}
        self.stats = {"load": 0, "upsert": 0, "forget": 0, "discard": 0,
                      "commit": 0}
        self.last_commit: Dict[str, int] = {}

    def __repr__(self) -> str:
        return f"PickleBackend({self.location!r})"

    # -- lock -----------------------------------------------------------
    def lock(self) -> _StoreLock:
        return _store_lock(self.location)

    # -- read -----------------------------------------------------------
    def load(self) -> Dict[str, Any]:
        snapshot, st = _read_store(self.location)
        self._sig = _signature(st)
        self.stats["load"] += 1
        return snapshot

    def read_if_changed(self) -> Optional[Dict[str, Any]]:
        try:
            st = os.stat(self.location)
        except FileNotFoundError:
            return None
        if st.st_size == 0:
            return None                         # an empty file holds nothing to lose
        if self._sig is not None and self._sig == _signature(st):
            return None                         # nobody wrote since : fast path
        disk, _st = _read_store(self.location)
        return disk

    # -- point writes : grouped into the next commit ----------------------
    def upsert(self, point: Any) -> None:
        self._batch["upsert"] += 1
        self.stats["upsert"] += 1

    def forget(self, point_id: str, entry: Any) -> None:
        self._batch["forget"] += 1
        self.stats["forget"] += 1

    def discard(self, point_id: str) -> None:
        self._batch["discard"] += 1
        self.stats["discard"] += 1

    # -- commit : the whole snapshot, atomically ---------------------------
    def commit(self, state: Dict[str, Any]) -> None:
        batch, self._batch = self._batch, {"upsert": 0, "forget": 0, "discard": 0}
        target = self.location
        tmp = f"{target}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        _fsync_dir(target)
        self._sig = _signature(os.stat(target))
        self.stats["commit"] += 1
        self.last_commit = dict(batch, points=len(state.get("points", [])))
