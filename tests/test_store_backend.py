"""The store interface (OO·V2·M1, Linear TAC2-1153 — OO2-1 / OO2-2).

`Memory.save()` is a unit of work over `metacog.store.StoreBackend` : the
change since the last sync reaches the backend as point writes (`upsert` per
new or changed point, `forget` per new forget, `discard` per point no longer
held), then one `commit`. `PickleBackend` groups them into one file — the
file it writes is the snapshot the former `save()` pickled, byte for byte.

  S1  the default backend is `PickleBackend` and satisfies the protocol ;
      `METACOG_STORE` selects it, an unknown name refuses to construct.
  S2  an ingest = upserts of the new point(s), 1 commit ; a save with no
      change = 0 point write, 1 commit.
  S3  a side effect (`apply_pull` on an older point) is upserted too —
      the diff is exhaustive, not declared by the call site.
  S4  a forget = 1 `forget` + the upsert of the INVALID point.
  S5  a point dropped from memory = 1 `discard`.
  S6  merge-on-stale : what another writer added is merged, and NOT written
      back as ours.
  S7  the file is `pickle.dumps(snapshot)` — same bytes as the former writer.
  S8  an injected backend is used for load and save ; a snapshot has none.
"""
from __future__ import annotations

import pickle

import pytest

from metacog.defaults import SimpleEncoder
from metacog.geometry import apply_pull
from metacog.memory import Memory
from metacog.store import PickleBackend, StoreBackend, open_backend


class CountingBackend(PickleBackend):
    """A PickleBackend that records the ids it is handed."""

    def __init__(self, path):
        super().__init__(path)
        self.log = []

    def upsert(self, point):
        self.log.append(("upsert", point.id))
        super().upsert(point)

    def forget(self, point_id, entry):
        self.log.append(("forget", point_id))
        super().forget(point_id, entry)

    def discard(self, point_id):
        self.log.append(("discard", point_id))
        super().discard(point_id)

    def commit(self, state):
        self.log.append(("commit", len(state["points"])))
        super().commit(state)

    def take(self):
        out, self.log = self.log, []
        return out


def _mem(store, backend=None):
    return Memory(encoder=SimpleEncoder(), storage_path=store,
                  store_backend=backend)


def test_s1_default_backend_and_selection(tmp_path, monkeypatch):
    store = str(tmp_path / "memory.pkl")
    monkeypatch.delenv("METACOG_STORE", raising=False)
    m = _mem(store)
    b = m._own_backend()
    assert isinstance(b, PickleBackend) and isinstance(b, StoreBackend)
    assert b.location == store
    monkeypatch.setenv("METACOG_STORE", "pickle")
    assert isinstance(open_backend(store), PickleBackend)
    monkeypatch.setenv("METACOG_STORE", "nowhere")
    with pytest.raises(ValueError, match="METACOG_STORE"):
        _mem(str(tmp_path / "other.pkl"))
    assert not (tmp_path / "other.pkl").exists()


def test_s2_ingest_is_point_writes_then_one_commit(tmp_path):
    store = str(tmp_path / "memory.pkl")
    b = CountingBackend(store)
    m = _mem(store, b)
    m.ingest("Dr Sarah lives in Berkeley", id="f1")
    m.save()
    log = b.take()
    assert ("upsert", "f1") in log and log[-1] == ("commit", len(m.points))
    assert [e for e in log if e[0] != "upsert"] == [log[-1]]
    m.save()                                   # nothing changed
    assert b.take() == [("commit", len(m.points))]


def test_s3_a_pulled_neighbour_is_upserted(tmp_path):
    store = str(tmp_path / "memory.pkl")
    b = CountingBackend(store)
    m = _mem(store, b)
    m.ingest("alpha beta gamma", id="a")
    m.ingest("delta epsilon zeta", id="z")
    m.save()
    b.take()
    pa = next(p for p in m.points if p.id == "a")
    pz = next(p for p in m.points if p.id == "z")
    apply_pull(pa, pz, +1, m._t_clock, bidirectional=False)   # only `a` moves
    m.save()
    assert b.take() == [("upsert", "a"), ("commit", len(m.points))]
    apply_pull(pa, pz, +1, m._t_clock)                        # both move
    m.save()
    assert b.take() == [("upsert", "a"), ("upsert", "z"),
                        ("commit", len(m.points))]


def test_s4_forget_is_a_forget_write(tmp_path):
    store = str(tmp_path / "memory.pkl")
    b = CountingBackend(store)
    m = _mem(store, b)
    m.ingest("to be forgotten", id="gone")
    m.save()
    b.take()
    assert m.forget_node("gone", "test").get("forgotten")
    m.save()
    log = b.take()
    assert ("forget", "gone") in log and ("upsert", "gone") in log
    assert log[-1][0] == "commit"


def test_s5_a_dropped_point_is_discarded(tmp_path):
    store = str(tmp_path / "memory.pkl")
    b = CountingBackend(store)
    m = _mem(store, b)
    m.ingest("keep me", id="k")
    m.ingest("drop me", id="d")
    m.save()
    b.take()
    m.points = [p for p in m.points if p.id != "d"]
    m.save()
    assert b.take() == [("discard", "d"), ("commit", len(m.points))]


def test_s6_merged_points_are_not_written_back(tmp_path):
    store = str(tmp_path / "memory.pkl")
    a = _mem(store)
    bb = CountingBackend(store)
    b = _mem(store, bb)
    bb.take()
    a.ingest("written by a", id="from_a")
    a.save()
    b.ingest("written by b", id="from_b")
    b.save()
    log = bb.take()
    assert {i for k, i in log if k == "upsert"} == {"from_b"}
    assert {"from_a", "from_b"} <= {p.id for p in b.points}
    assert {"from_a", "from_b"} <= {p.id for p in _mem(store).points}


def test_s7_the_file_is_the_former_snapshot(tmp_path):
    store = str(tmp_path / "memory.pkl")
    m = _mem(store)
    m.ingest("bytes for bytes", id="b")
    m.save()
    assert open(store, "rb").read() == pickle.dumps(m._persist_snapshot())


def test_s8_injected_backend_and_snapshot(tmp_path):
    store = str(tmp_path / "memory.pkl")
    first = _mem(store)
    first.ingest("persisted once", id="p")
    first.save()
    b = CountingBackend(store)
    m = _mem(store, b)
    assert b.stats["load"] == 1 and "p" in {p.id for p in m.points}
    clone = m.snapshot()
    assert clone.store_backend is None and clone.storage_path is None
    with pytest.raises(ValueError):
        clone.save()
