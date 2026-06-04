"""
Tests for the Spike-and-Path Chasles compression (PR n°1.5).

Three emergent formulas (no hyperparameter), one MetaWalker hook
(commit=True only), one Memory API (compress_chasles).
"""

from __future__ import annotations

import math
import pickle
import tempfile
import os

from metacog.defaults import SimpleEncoder
from metacog.epistemic import Point, PointKind, SourceClass
from metacog.memory import Memory
from metacog.meta_walk import MetaWalker
from metacog.spike import (
    auto_compress_chasles,
    chasles_path,
    concentrated,
    is_spiking,
    record_hop,
    spike_threshold,
)


def _pt(id_: str, content: str = "x") -> Point:
    enc = SimpleEncoder()
    return Point(id=id_, content=content,
                 embedding_orig=tuple(enc.encode(content)),
                 kind=PointKind.FACT)


def _mem(n_pts: int = 0) -> Memory:
    m = Memory(encoder=SimpleEncoder())
    for i in range(n_pts):
        m.ingest(f"content number {i}", kind="FACT", id=f"P{i}")
    return m


# ---------------------------------------------------------------------------
# 1) record_hop
# ---------------------------------------------------------------------------


def test_record_hop_increments_n_spike_and_tags():
    m = _mem(0)
    a, b = _pt("A"), _pt("B")
    m.points.extend([a, b])
    record_hop(a, b, m)
    assert b.n_spike == 1
    assert "ref:B" in a.tags
    assert m._spike_total_hops == 1


def test_record_hop_dedups_repeated_ref_tag_keeps_counter():
    m = _mem(0)
    a, b = _pt("A"), _pt("B")
    m.points.extend([a, b])
    for _ in range(3):
        record_hop(a, b, m)
    assert b.n_spike == 3
    assert a.tags.count("ref:B") == 1
    assert m._spike_total_hops == 3


def test_record_hop_noop_on_self_or_none():
    m = _mem(0)
    a = _pt("A")
    m.points.append(a)
    record_hop(a, a, m)                          # self-loop
    record_hop(None, a, m)
    record_hop(a, None, m)
    assert a.n_spike == 0
    assert m._spike_total_hops == 0


# ---------------------------------------------------------------------------
# 2) spike_threshold + is_spiking
# ---------------------------------------------------------------------------


def test_spike_threshold_zero_when_N_small():
    m = _mem(3)
    assert spike_threshold(m) == 0.0


def test_spike_threshold_scales_with_N_and_M():
    m_dense = _mem(100); m_dense._spike_total_hops = 200
    lam = 2.0
    assert math.isclose(spike_threshold(m_dense), lam + lam ** 0.5)
    m_sparse = _mem(1000); m_sparse._spike_total_hops = 200
    lam = 0.2
    assert math.isclose(spike_threshold(m_sparse), lam + lam ** 0.5)
    # Threshold strictly lower for sparser memory (same M, larger N).
    assert spike_threshold(m_sparse) < spike_threshold(m_dense)


def test_is_spiking_below_above_threshold():
    m = _mem(100); m._spike_total_hops = 200
    p = _pt("X"); m.points.append(p)
    p.n_spike = 3                                # below threshold ~3.41
    assert not is_spiking(p, m)
    p.n_spike = 4                                # above
    assert is_spiking(p, m)


# ---------------------------------------------------------------------------
# 3) concentrated
# ---------------------------------------------------------------------------


def test_concentrated_uniform_returns_false():
    # User-specified case : 5 refs all at 20% -> no clear path
    assert concentrated([20, 20, 20, 20, 20]) is False


def test_concentrated_dominant_returns_true():
    assert concentrated([80, 5, 5, 5, 5]) is True


def test_concentrated_edge_cases():
    assert concentrated([]) is False
    assert concentrated([0, 0, 0]) is False
    assert concentrated([10]) is True            # k=1 always concentrated


# ---------------------------------------------------------------------------
# 4) chasles_path
# ---------------------------------------------------------------------------


def _wire_path(memory: Memory, path_ids):
    """Helper : link path_ids in order. Each src on the path gets the
    next node as STRONG modal ref + 2 distractor refs (k=3 ensures the
    concentrated() check can succeed since k=2 is intentionally too
    conservative)."""
    by_id = {p.id: p for p in memory.points}
    fillers = [p for p in memory.points if p.id not in path_ids]
    for src_id, dst_id in zip(path_ids, path_ids[1:]):
        src = by_id[src_id]
        dst = by_id[dst_id]
        dst.n_spike = 50
        for tag in (f"ref:{dst_id}",) + tuple(f"ref:{f.id}" for f in fillers[:2]):
            if tag not in src.tags:
                src.tags.append(tag)            # direct, case-preserving
        for f in fillers[:2]:
            if f.n_spike == 0:
                f.n_spike = 1
    by_id[path_ids[0]].n_spike = 50
    memory._spike_total_hops = max(memory._spike_total_hops, 30)


def test_chasles_path_finds_length_4():
    m = _mem(10)
    _wire_path(m, ["P0", "P1", "P2", "P3"])
    path = chasles_path(m.points[0], m)
    assert [p.id for p in path] == ["P0", "P1", "P2", "P3"]


def test_chasles_path_aborts_when_target_not_spiking():
    m = _mem(10)
    _wire_path(m, ["P0", "P1", "P2", "P3"])
    # Kill P2's spike — chain should stop before reaching P2
    next(p for p in m.points if p.id == "P2").n_spike = 0
    path = chasles_path(m.points[0], m)
    assert [p.id for p in path] == ["P0", "P1"]


def test_chasles_path_aborts_on_uniform_refs():
    m = _mem(10)
    # P0 spikes, but its refs (P1, P2, P3, P4) all have equal spike.
    by_id = {p.id: p for p in m.points}
    by_id["P0"].n_spike = 50
    for nid in ("P1", "P2", "P3", "P4"):
        by_id["P0"].add_tag(f"ref:{nid}")
        by_id[nid].n_spike = 20                      # uniform
    m._spike_total_hops = 30
    path = chasles_path(by_id["P0"], m)
    assert [p.id for p in path] == ["P0"]            # no concentrated next


def test_chasles_path_cycle_protection():
    m = _mem(4)
    by_id = {p.id: p for p in m.points}
    # P0 <-> P1 with strong mutual refs ; both spike.
    by_id["P0"].n_spike = 50; by_id["P1"].n_spike = 50
    by_id["P0"].tags.append("ref:P1")
    by_id["P1"].tags.append("ref:P0")
    m._spike_total_hops = 30
    path = chasles_path(by_id["P0"], m)
    # P0 -> P1, then P1's modal ref is P0 (already seen) -> stop
    assert [p.id for p in path] == ["P0", "P1"]


# ---------------------------------------------------------------------------
# 5) auto_compress_chasles
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Minimal LLM stub satisfying resolve_collision's contract."""

    def extract_common(self, texts):
        return "common"

    def remove_overlap(self, full, common):
        out = full.replace(common, "").strip()
        return out or f"distinct {full[:6]}"          # never empty


def test_auto_compress_skips_path_under_4_no_reset():
    m = _mem(10)
    by_id = {p.id: p for p in m.points}
    # Path of length 3 only : P0 -> P1 -> P2, then P2 has no spiking next.
    by_id["P0"].n_spike = 50; by_id["P1"].n_spike = 50; by_id["P2"].n_spike = 50
    by_id["P0"].add_tag("ref:P1"); by_id["P1"].add_tag("ref:P2")
    # add a dead-end ref on P2 to a non-spiking node so concentrated() may
    # still be True but the next node is_spiking() is False.
    by_id["P2"].add_tag("ref:P3")                     # P3.n_spike = 0
    m._spike_total_hops = 30
    events = auto_compress_chasles(m, _FakeLLM(), m.encoder)
    assert events == []
    # CRITICAL : no reset on a sub-4 path
    assert by_id["P0"].n_spike == 50
    assert by_id["P1"].n_spike == 50
    assert by_id["P2"].n_spike == 50


def test_auto_compress_triggers_compression_and_resets_spike():
    m = _mem(10)
    by_id = {p.id: p for p in m.points}
    # Path P0->P1->P2->P3 with content the FakeLLM can split.
    for i, nid in enumerate(["P0", "P1", "P2", "P3"]):
        by_id[nid].content = f"shared common topic chunk {i}"
        by_id[nid].n_spike = 50
    # 3 refs per intermediate (k=3 so concentrated() can fire).
    by_id["P0"].tags.append("ref:P1")
    for tag in ("ref:P2", "ref:P9", "ref:P8"):
        by_id["P1"].tags.append(tag)
    for tag in ("ref:P3", "ref:P7", "ref:P6"):
        by_id["P2"].tags.append(tag)
    for distractor in ("P9", "P8", "P7", "P6"):
        by_id[distractor].n_spike = 1
    m._spike_total_hops = 30
    events = auto_compress_chasles(m, _FakeLLM(), m.encoder)
    assert len(events) == 1
    # All four path nodes reset to 0 (refractory).
    for nid in ("P0", "P1", "P2", "P3"):
        assert by_id[nid].n_spike == 0
    # Child appended to memory.
    assert any(p.id == events[0].child_id for p in m.points)


def test_auto_compress_rejects_cross_kind_path():
    m = _mem(7)                                     # 4 path + 3 distractors
    by_id = {p.id: p for p in m.points}
    # Make the path cross-kind by retagging — keep refs same-id.
    by_id["P0"].kind = PointKind.FACT
    by_id["P1"].kind = PointKind.FACT
    by_id["P2"].kind = PointKind.THOUGHT             # cross-kind here
    by_id["P3"].kind = PointKind.FACT
    for nid in ("P0", "P1", "P2", "P3"):
        by_id[nid].n_spike = 50
    by_id["P0"].tags.append("ref:P1")
    for tag in ("ref:P2", "ref:P4", "ref:P5"):
        by_id["P1"].tags.append(tag)
    for tag in ("ref:P3", "ref:P4", "ref:P5"):
        by_id["P2"].tags.append(tag)
    by_id["P4"].n_spike = 1; by_id["P5"].n_spike = 1
    m._spike_total_hops = 30
    events = auto_compress_chasles(m, _FakeLLM(), m.encoder)
    assert events == []


# ---------------------------------------------------------------------------
# 6) Persistence
# ---------------------------------------------------------------------------


def test_pickle_roundtrip_n_spike_and_total_hops():
    m = _mem(5)
    by_id = {p.id: p for p in m.points}
    by_id["P0"].n_spike = 7
    by_id["P2"].n_spike = 3
    m._spike_total_hops = 42
    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        m.storage_path = path
        m.save()
        m2 = Memory(encoder=SimpleEncoder(), storage_path=path)
        m2.load()
        ids = {p.id: p for p in m2.points}
        assert ids["P0"].n_spike == 7
        assert ids["P2"].n_spike == 3
        assert m2._spike_total_hops == 42
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 7) MetaWalker hook : commit=False MUST NOT record_hop
# ---------------------------------------------------------------------------


def test_metawalker_commit_false_does_not_record_hop():
    m = _mem(10)
    # commit=False is the benchmark mode : record_hop must be a no-op so
    # benchmark runs stay reproducible and don't mutate Memory.
    w = MetaWalker("any query", m, commit=False)
    try:
        w.step()
        w.step()
    except Exception:
        pass
    assert m._spike_total_hops == 0
    # No "ref:" tag should have appeared on any non-entity / non-atom point.
    for p in m.points:
        for t in p.tags:
            assert not t.startswith("ref:")


# ---------------------------------------------------------------------------
# 8) Cor.5 audit
# ---------------------------------------------------------------------------


def test_spike_source_class_is_computation():
    from metacog.spike import SOURCE
    assert SOURCE is SourceClass.COMPUTATION


# ---------------------------------------------------------------------------
# Workflow (ACTION / tool chain) compression — in-depth Chasles on actions
# ---------------------------------------------------------------------------

def test_action_workflow_chain_compresses_into_shortcut():
    """A recurring multi-step workflow — a chain of ACTION nodes (incl.
    tool nodes, which are kind=ACTION) — is shortened by the in-depth
    Chasles compression into a single optimized intermediate step."""
    from metacog.epistemic import Point, PointKind
    m = _mem(0)
    # A 4-step workflow A0 -> A1 -> A2 -> A3, all kind=ACTION.
    chain = []
    for i in range(4):
        a = Point(id=f"A{i}", content=f"shared workflow step chunk {i}",
                  embedding_orig=tuple(m.encoder.encode(f"step {i}")),
                  kind=PointKind.ACTION, n_spike=50)
        chain.append(a)
    # tag the first two as tool nodes : workflows of tools must compress too
    chain[0].add_tag("tool", "tool_command")
    chain[1].add_tag("tool", "tool_command")
    m.points.extend(chain)
    by_id = {p.id: p for p in m.points}
    # distractor non-spiking nodes so concentrated() can fire (k=3 refs)
    for d in ("D0", "D1", "D2", "D3"):
        m.points.append(Point(id=d, content=d,
                              embedding_orig=tuple(m.encoder.encode(d)),
                              kind=PointKind.ACTION, n_spike=1))
    by_id = {p.id: p for p in m.points}
    by_id["A0"].tags.append("ref:A1")
    for tag in ("ref:A2", "ref:D0", "ref:D1"):
        by_id["A1"].tags.append(tag)
    for tag in ("ref:A3", "ref:D2", "ref:D3"):
        by_id["A2"].tags.append(tag)
    m._spike_total_hops = 30

    events = auto_compress_chasles(m, _FakeLLM(), m.encoder)
    assert len(events) == 1                         # the workflow fired
    ev = events[0]
    # intermediates A1,A2 compressed into one child between anchors A0,A3
    assert set(ev.parent_ids) == {"A1", "A2"}
    assert set(ev.anchor_ids) == {"A0", "A3"}
    child = next(p for p in m.points if p.id == ev.child_id)
    assert child.kind == PointKind.ACTION           # stays an action step
    # the compressed workflow is promoted to a reusable tool node
    from metacog.skills import is_tool
    assert is_tool(child) and "tool_workflow" in child.tags
    # refractory reset on the whole fired path
    for nid in ("A0", "A1", "A2", "A3"):
        assert by_id[nid].n_spike == 0
