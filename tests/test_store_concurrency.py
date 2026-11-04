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


# ───────────────────────────────────────────────────────────────────────────
# The SAME cases on `Neo4jBackend` (OO·V2·M3, TAC2-1157 / OO2-9), against a real
# tachikoma API + Neo4j (`tests/neo4j_live.py` ; skipped without OMNI_TEST_API_URL).
# None of the cases above is removed or changed : each one has its twin here, measured
# where the data lands — the store's rows read back through the API's load route, and
# the nodes in Neo4j. What differs is said case by case : there is no file to truncate,
# so « corrupt » becomes « the store cannot be read » (API or base unreachable, an
# unreadable answer), and the lock between processes is the store's revision (409 →
# re-read, merge, write again) instead of an `flock`.
# ───────────────────────────────────────────────────────────────────────────

from tests.neo4j_live import live_ns, requires_live  # noqa: E402,F401


def _n_store(ns, tmp_path, ctx):
    """The store path of `ctx` in the gate's layout — the backend reads the context from
    it — after creating the context in the API's hierarchy."""
    ns.fresh(ctx)                          # each case its own context, empty at start
    d = tmp_path / ctx
    d.mkdir(parents=True, exist_ok=True)
    return str(d / "memory.pkl")


def _n_seed(path, n=3):
    m = _mem(path)
    for i in range(n):
        m.ingest(f"seed fact {i}", id=f"seed_{i}")
    m.save()
    return m


class TestNeo4jStore:
    pytestmark = requires_live

    def test_c1_stale_instances_both_keep_their_fact(self, live_ns, tmp_path):
        store = _n_store(live_ns, tmp_path, "c1")
        _n_seed(store)
        a, b = _mem(store), _mem(store)
        a.ingest("alpha writes first", id="fact_a")
        a.save()
        b.ingest("beta writes second, from a stale snapshot", id="fact_b")
        b.save()
        held = live_ns.point_ids("c1")
        assert {"fact_a", "fact_b"} <= held
        assert "fact_a" in {p.id for p in b.points}
        assert len(held) == 5
        # WHERE IT LANDS : one node per point, in this context and no other.
        assert live_ns.nodes_of("fact_a") == [live_ns.real("c1")]

    def test_c2_own_removal_is_not_resurrected(self, live_ns, tmp_path):
        store = _n_store(live_ns, tmp_path, "c2")
        _n_seed(store)
        a, b = _mem(store), _mem(store)
        b.ingest("beta adds", id="fact_b")
        b.save()
        a.points = [p for p in a.points if p.id != "seed_0"]
        a.save()
        held = live_ns.point_ids("c2")
        assert "fact_b" in held and "seed_0" not in held
        # ZERO DELETION : the removed point is archived, never deleted — its row is
        # still there (discarded) and so is its node.
        archived = {p["point_id"]: p["fields"]["discarded"]
                    for p in live_ns.load("c2", include_discarded=True)["points"]}
        assert archived["seed_0"] is True
        assert live_ns.real("c2") in live_ns.nodes_of("seed_0")
        assert "seed_0" not in {p.id for p in _mem(store).points}

    def test_c3_twenty_concurrent_processes_lose_nothing(self, live_ns, tmp_path):
        store = _n_store(live_ns, tmp_path, "c3")
        _n_seed(store)
        before = live_ns.point_ids("c3")
        go = str(tmp_path / "go")
        env = dict(os.environ, PYTHONPATH=REPO)
        procs = [subprocess.Popen([sys.executable, "-c", _WORKER, store, go, str(i)],
                                  env=env, stderr=subprocess.PIPE)
                 for i in range(20)]
        deadline = time.time() + 240
        while sum(os.path.exists(f"{go}.ready.{i}") for i in range(20)) < 20:
            assert time.time() < deadline, "workers did not load in time"
            assert all(p.poll() is None for p in procs), "a worker died early"
            time.sleep(0.01)
        open(go, "w").close()
        for p in procs:
            _, err = p.communicate(timeout=600)
            assert p.returncode == 0, err.decode()
        after = live_ns.point_ids("c3")
        assert after - before == {f"cap_{i}" for i in range(20)}
        assert len(after) == len(before) + 20

    def test_c4_twenty_threads_one_memory_each(self, live_ns, tmp_path):
        store = _n_store(live_ns, tmp_path, "c4")
        _n_seed(store)
        mems = [_mem(store) for _ in range(20)]
        barrier = threading.Barrier(20)
        errors = []

        def capture(i):
            try:
                barrier.wait()
                mems[i].ingest(f"thread fact {i}", id=f"thr_{i}")
                mems[i].save()
            except Exception as e:
                errors.append(e)

        ts = [threading.Thread(target=capture, args=(i,)) for i in range(20)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert not errors
        held = live_ns.point_ids("c4")
        assert {f"thr_{i}" for i in range(20)} <= held
        assert len(held) == 23

    @pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="POSIX only")
    def test_c5_sigkill_mid_commit_leaves_a_loadable_store(self, live_ns, tmp_path):
        """The process dies with HALF its batch on the wire : the API never gets a
        whole request, so nothing of it is written — the store loads, unchanged."""
        store = _n_store(live_ns, tmp_path, "c5")
        _n_seed(store, n=200)
        before = live_ns.point_ids("c5")
        r = subprocess.run([sys.executable, "-c", _N_KILLER, store],
                           env=dict(os.environ, PYTHONPATH=REPO), capture_output=True)
        assert r.returncode == -signal.SIGKILL, r.stderr.decode()[-2000:]
        assert b"HALF-SENT" in r.stdout
        m = _mem(store)
        assert {p.id for p in m.points} == before
        assert "doomed" not in live_ns.point_ids("c5")
        m.ingest("life goes on", id="after")
        m.save()
        assert "after" in live_ns.point_ids("c5")

    def test_c6_exception_mid_commit_leaves_store_identical(self, live_ns, tmp_path,
                                                            monkeypatch):
        store = _n_store(live_ns, tmp_path, "c6")
        m = _n_seed(store)
        raw = live_ns.load("c6")
        m.ingest("never written", id="ghost")
        import urllib.request as ur

        def boom(*a, **k):
            raise ConnectionResetError("connection reset mid-commit")
        monkeypatch.setattr(ur, "urlopen", boom)
        with pytest.raises(CorruptStoreError):
            m.save()
        monkeypatch.undo()
        assert live_ns.load("c6") == raw

    @pytest.mark.parametrize("cut", [0, 0.5])
    def test_c7_unreadable_store_is_named_and_refused(self, live_ns, tmp_path, cut,
                                                      capsys, monkeypatch):
        """No file to truncate : `cut=0` — nothing answers (the API is down) ;
        `cut=0.5` — something answers, half a body. Both : CorruptStoreError naming the
        store, REFUSING TO SERVE, never an empty memory — and the real store untouched."""
        ctx = f"c7x{int(cut * 10)}"
        store = _n_store(live_ns, tmp_path, ctx)
        _n_seed(store, n=50)
        raw = live_ns.load(ctx)
        server = None
        if cut == 0:
            import socket
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()                       # nothing listens there
        else:
            server, port = _half_body_server()
        monkeypatch.setenv("TACHIKOMA_API_URL", f"http://127.0.0.1:{port}")
        try:
            with pytest.raises(CorruptStoreError) as ei:
                _mem(store)
        finally:
            if server is not None:
                server.shutdown()
        assert f"127.0.0.1:{port}" in str(ei.value) and ctx in str(ei.value)
        assert "REFUSING TO SERVE" in capsys.readouterr().err
        monkeypatch.undo()
        assert live_ns.load(ctx) == raw

    def test_c8_save_never_writes_over_a_store_it_cannot_read(self, live_ns, tmp_path):
        store = _n_store(live_ns, tmp_path, "c8")
        m = _n_seed(store)
        raw = live_ns.load("c8")
        m._own_backend().api = "http://127.0.0.1:9"     # the store becomes unreachable
        m.ingest("would hide the outage", id="late")
        with pytest.raises(CorruptStoreError):
            m.save()
        assert live_ns.load("c8") == raw
        assert "late" not in live_ns.point_ids("c8")


_N_KILLER = textwrap.dedent("""
    import os, signal, socket, sys, urllib.parse
    from metacog.defaults import SimpleEncoder
    from metacog.memory import Memory
    from metacog.store import neo4j_backend as nb
    m = Memory(encoder=SimpleEncoder(), storage_path=sys.argv[1])
    m.ingest('this fact dies with the process', id='doomed')
    real_call = nb.Neo4jBackend._call
    def dying_call(self, method, path, body=None):
        if method != 'POST':
            return real_call(self, method, path, body)
        import json
        data = json.dumps(body).encode()
        u = urllib.parse.urlsplit(self.api)
        s = socket.create_connection((u.hostname, u.port))
        head = (f'POST {path}?{self._query()} HTTP/1.1\\r\\nHost: {u.netloc}\\r\\n'
                f'Authorization: Bearer {self.token()}\\r\\n'
                f'Content-Type: application/json\\r\\n'
                f'Content-Length: {len(data)}\\r\\n\\r\\n').encode()
        s.sendall(head + data[: len(data) // 2])      # half the batch reaches the API
        print('HALF-SENT', flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
    nb.Neo4jBackend._call = dying_call
    m.save()
""")


def _half_body_server():
    """An HTTP server that answers 200 with half a JSON body."""
    import http.server

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"store_version": 3, "points": [{"kind": "memory_fact"'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]
