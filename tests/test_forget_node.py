"""
Explicit on-demand soft-invalidation — mnema's `forget(node_id, reason)`.

Append-only : the node is never deleted, its state is set INVALID so it stops
surfacing in retrieval / the walk, and the required reason is kept in a persisted
forget log. Distinct from the autonomic decay-forgetting pass in sleep.
"""

from __future__ import annotations

import os
import tempfile

from metacog.defaults import SimpleEncoder
from metacog.epistemic import EpistemicState
from metacog.memory import Memory


def _mem():
    m = Memory(encoder=SimpleEncoder())
    for kw in ("apple", "banana", "cherry", "date"):
        m.ingest(f"a fact about {kw}", kind="FACT", id=kw)
    return m


def test_forget_soft_invalidates_append_only():
    m = _mem()
    out = m.forget_node("apple", reason="superseded by banana")
    assert out["forgotten"] == "apple"
    p = next(q for q in m.points if q.id == "apple")
    assert p.state is EpistemicState.INVALID       # marked invalid
    assert "invalidated" in p.tags
    assert any(q.id == "apple" for q in m.points)   # NOT deleted (append-only)


def test_reason_is_required():
    m = _mem()
    assert m.forget_node("apple", reason="")["forgotten"] is None
    assert m.forget_node("apple", reason="   ")["forgotten"] is None
    # unchanged
    assert next(q for q in m.points if q.id == "apple").state is not EpistemicState.INVALID


def test_forgotten_node_drops_out_of_retrieve():
    m = _mem()
    before = [h["id"] for h in m.retrieve("a fact about apple", k=4)]
    assert "apple" in before
    m.forget_node("apple", reason="user corrected")
    after = [h["id"] for h in m.retrieve("a fact about apple", k=4)]
    assert "apple" not in after                     # stops surfacing


def test_forget_unknown_node():
    m = _mem()
    assert m.forget_node("nope", reason="x")["forgotten"] is None


def test_forget_log_persists():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.pkl")
        m = _mem()
        m.storage_path = path
        m.forget_node("apple", reason="superseded by node banana")
        m.save()
        m2 = Memory(encoder=SimpleEncoder(), storage_path=path)
        assert any(e["id"] == "apple" and "superseded" in e["reason"]
                   for e in getattr(m2, "_forget_log", []))
        # the invalidation itself also persisted (state on the point)
        assert next(q for q in m2.points if q.id == "apple").state \
            is EpistemicState.INVALID
