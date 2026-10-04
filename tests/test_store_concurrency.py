"""
Store concurrency (TAC-935) : `Memory.save()` is atomic, single-writer per
store, and never loses another writer's facts ; a corrupt store is NAMED and
refused, never served as an empty memory.

  C1  two stale instances of one store both keep their fact (merge on save).
  C2  a point removed here on purpose is not resurrected by the merge.
  C3  20 PROCESSES capturing at once on one store -> 20 facts, none lost.
  C4  20 THREADS (one Memory each) capturing at once -> 20 facts.
  C5  a process SIGKILLed in the middle of pickle.dump leaves the store
      loadable, with its previous content.
  C6  an exception during the dump leaves the store byte-identical.
  C7  a truncated / empty store raises CorruptStoreError naming the file, the
      file is left untouched — never an empty memory.
  C8  a save never writes over a store that became corrupt on disk.
"""

from __future__ import annotations

import os
import pickle
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from metacog.defaults import SimpleEncoder
from metacog.memory import CorruptStoreError, Memory

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mem(path):
    return Memory(encoder=SimpleEncoder(), storage_path=path)


def _facts_on_disk(path):
    with open(path, "rb") as f:
        return {p.id for p in pickle.load(f)["points"]}


def _seed(path, n=3):
    m = _mem(path)
    for i in range(n):
        m.ingest(f"seed fact {i}", id=f"seed_{i}")
    m.save()
    return m


def test_c1_stale_instances_both_keep_their_fact(tmp_path):
    store = str(tmp_path / "memory.pkl")
    _seed(store)
    a, b = _mem(store), _mem(store)          # both load the same version
    a.ingest("alpha writes first", id="fact_a")
    a.save()
    b.ingest("beta writes second, from a stale snapshot", id="fact_b")
    b.save()                                 # used to overwrite fact_a
    assert {"fact_a", "fact_b"} <= _facts_on_disk(store)
    assert "fact_a" in {p.id for p in b.points}   # merged in RAM too
    assert len(_facts_on_disk(store)) == 5


def test_c2_own_removal_is_not_resurrected(tmp_path):
    store = str(tmp_path / "memory.pkl")
    _seed(store)
    a, b = _mem(store), _mem(store)
    b.ingest("beta adds", id="fact_b")
    b.save()
    a.points = [p for p in a.points if p.id != "seed_0"]   # removed here
    a.save()
    on_disk = _facts_on_disk(store)
    assert "fact_b" in on_disk and "seed_0" not in on_disk


_WORKER = textwrap.dedent("""
    import os, sys, time
    from metacog.defaults import SimpleEncoder
    from metacog.memory import Memory
    store, go, i = sys.argv[1], sys.argv[2], sys.argv[3]
    m = Memory(encoder=SimpleEncoder(), storage_path=store)  # stale on purpose
    open(go + '.ready.' + i, 'w').close()
    while not os.path.exists(go):
        time.sleep(0.005)
    m.ingest('captured fact number ' + i, id='cap_' + i)
    m.save()
""")


def test_c3_twenty_concurrent_processes_lose_nothing(tmp_path):
    store = str(tmp_path / "memory.pkl")
    _seed(store)
    before = _facts_on_disk(store)
    go = str(tmp_path / "go")
    env = dict(os.environ, PYTHONPATH=REPO)
    procs = [subprocess.Popen([sys.executable, "-c", _WORKER, store, go, str(i)],
                              env=env, stderr=subprocess.PIPE)
             for i in range(20)]
    deadline = time.time() + 120
    while sum(os.path.exists(f"{go}.ready.{i}") for i in range(20)) < 20:
        assert time.time() < deadline, "workers did not load in time"
        assert all(p.poll() is None for p in procs), "a worker died early"
        time.sleep(0.01)
    open(go, "w").close()                    # all 20 hold the SAME stale state
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err.decode()
    after = _facts_on_disk(store)
    assert after - before == {f"cap_{i}" for i in range(20)}
    assert len(after) == len(before) + 20


def test_c4_twenty_threads_one_memory_each(tmp_path):
    store = str(tmp_path / "memory.pkl")
    _seed(store)
    mems = [_mem(store) for _ in range(20)]
    barrier = threading.Barrier(20)
    errors = []

    def capture(i):
        try:
            barrier.wait()
            mems[i].ingest(f"thread fact {i}", id=f"thr_{i}")
            mems[i].save()
        except Exception as e:               # surfaced below
            errors.append(e)

    ts = [threading.Thread(target=capture, args=(i,)) for i in range(20)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    assert {f"thr_{i}" for i in range(20)} <= _facts_on_disk(store)
    assert len(_facts_on_disk(store)) == 23


_KILLER = textwrap.dedent("""
    import io, os, pickle, signal, sys
    from metacog.defaults import SimpleEncoder
    from metacog.memory import Memory
    m = Memory(encoder=SimpleEncoder(), storage_path=sys.argv[1])
    m.ingest('this fact dies with the process', id='doomed')
    real_dump = pickle.dump
    def dying_dump(obj, f, *a, **k):
        buf = io.BytesIO()
        real_dump(obj, buf, *a, **k)
        data = buf.getvalue()
        f.write(data[: len(data) // 2])      # half the bytes reach the file
        f.flush()
        os.kill(os.getpid(), signal.SIGKILL)
    pickle.dump = dying_dump
    m.save()
""")


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="POSIX only")
def test_c5_sigkill_mid_write_leaves_a_loadable_store(tmp_path):
    store = str(tmp_path / "memory.pkl")
    _seed(store, n=200)
    before = _facts_on_disk(store)
    r = subprocess.run([sys.executable, "-c", _KILLER, store],
                       env=dict(os.environ, PYTHONPATH=REPO), capture_output=True)
    assert r.returncode == -signal.SIGKILL
    m = _mem(store)                          # loads : no CorruptStoreError
    assert {p.id for p in m.points} == before
    m.ingest("life goes on", id="after")
    m.save()                                 # the orphan .tmp is overwritten
    assert "after" in _facts_on_disk(store)


def test_c6_exception_mid_dump_leaves_store_identical(tmp_path, monkeypatch):
    store = str(tmp_path / "memory.pkl")
    m = _seed(store)
    raw = open(store, "rb").read()
    m.ingest("never written", id="ghost")

    def boom(obj, f, *a, **k):
        f.write(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(pickle, "dump", boom)
    with pytest.raises(OSError):
        m.save()
    assert open(store, "rb").read() == raw


@pytest.mark.parametrize("cut", [0, 0.5])
def test_c7_corrupt_store_is_named_and_refused(tmp_path, cut, capsys):
    store = str(tmp_path / "memory.pkl")
    _seed(store, n=50)
    raw = open(store, "rb").read()
    damaged = raw[: int(len(raw) * cut)]
    with open(store, "wb") as f:
        f.write(damaged)
    with pytest.raises(CorruptStoreError) as ei:
        _mem(store)
    assert store in str(ei.value)
    assert "REFUSING TO SERVE" in capsys.readouterr().err
    assert open(store, "rb").read() == damaged   # left for recovery


def test_c8_save_never_overwrites_a_store_corrupted_on_disk(tmp_path):
    store = str(tmp_path / "memory.pkl")
    m = _seed(store)
    with open(store, "wb") as f:
        f.write(b"\x80\x04garbage")
    m.ingest("would hide the damage", id="late")
    with pytest.raises(CorruptStoreError):
        m.save()
    assert open(store, "rb").read() == b"\x80\x04garbage"
