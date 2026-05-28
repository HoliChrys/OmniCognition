"""
Tests for the top-level Memory wrapper.

Properties :
  M1   ingest creates a Point and returns it
  M2   observe updates counters
  M3   process_turn records the turn and fires detectors when relevant
  M4   retrieve returns top-k matches with metadata
  M5   reason returns a trajectory summary
  M6   sleep returns a SleepCycleReport-like dict
  M7   audit reports no laundering after a full session
  M8   stats counts by kind / state
  M9   save / load roundtrip preserves state
  M10  declare_observator + route works end-to-end
"""

from __future__ import annotations

import os
import tempfile

from metacog import Memory


def test_ingest_and_inspect():
    m = Memory()
    p = m.ingest("dr sarah lives in berkeley", kind="FACT")
    info = m.inspect(p.id)
    assert info["content"] == "dr sarah lives in berkeley"
    assert info["kind"] == "FACT"


def test_observe_updates_counters():
    m = Memory()
    p = m.ingest("alpha beta gamma")
    m.observe([p.id], polarity=+1.0)
    info = m.inspect(p.id)
    assert info["n_corrob"] == 1
    assert info["n_contra"] == 0


def test_process_turn_records_and_detects():
    m = Memory()
    p = m.ingest("dr sarah counseling career berkeley")
    # First turn: user asks
    m.process_turn("who is dr sarah counseling", speaker="user")
    # Assistant responds, retrieving p
    m.process_turn(
        "dr sarah encouraged counseling career",
        speaker="assistant",
        retrieved_point_ids=[p.id],
    )
    # User corroborates
    obs_list = m.process_turn(
        "dr sarah counseling career and also berkeley conference",
        speaker="user",
    )
    # At least a corroboration signal should fire
    signal_types = {o.signal_type for o in obs_list}
    assert "corroboration" in signal_types


def test_retrieve_returns_top_k():
    m = Memory()
    for content in ["apple pie recipe", "banana bread recipe",
                    "chocolate cake recipe", "stock market today",
                    "weather forecast today"]:
        m.ingest(content)
    results = m.retrieve("recipe", k=3)
    assert len(results) == 3
    for r in results:
        assert "id" in r and "content" in r and "score" in r


def test_reason_returns_summary():
    m = Memory()
    for content in ["dr sarah berkeley conference",
                    "dr sarah counseling career",
                    "dr sarah adoption support"]:
        m.ingest(content)
    summary = m.reason("where is dr sarah")
    assert "final_answer" in summary
    assert "halt_reason" in summary
    assert "iterations" in summary
    assert summary["iterations"] >= 1


def test_sleep_returns_report():
    m = Memory()
    # ingest several points that share content to provoke collisions
    for content in [
        "common token alpha extra1",
        "common token alpha extra2",
        "common token alpha extra3",
        "totally different one",
        "unrelated other",
    ]:
        m.ingest(content)
    report = m.sleep()
    assert "iterations" in report
    assert "resolved_count" in report


def test_audit_after_session():
    m = Memory()
    m.ingest("alpha")
    m.ingest("beta")
    m.observe([m.points[0].id], polarity=+1)
    m.observe([m.points[1].id], polarity=-1)
    report = m.audit()
    assert report["ok"] is True
    assert report["violations"] == []


def test_stats_summarizes_inventory():
    m = Memory()
    m.ingest("f1", kind="FACT")
    m.ingest("t1", kind="THOUGHT")
    m.ingest("a1", kind="ACTION")
    stats = m.stats()
    assert stats["n_points"] == 3
    assert stats["by_kind"]["FACT"] == 1
    assert stats["by_kind"]["THOUGHT"] == 1
    assert stats["by_kind"]["ACTION"] == 1


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "snapshot.pkl")
        m1 = Memory(storage_path=path)
        p = m1.ingest("alpha beta")
        m1.observe([p.id], polarity=+1.0)
        m1.save()
        # New Memory loading from the same path
        m2 = Memory(storage_path=path)
        info = m2.inspect(p.id)
        assert info is not None
        assert info["content"] == "alpha beta"
        assert info["n_corrob"] == 1


def test_observator_lifecycle():
    m = Memory()
    m.ingest("patient consent protocol clinical")
    m.ingest("contract liability statute legal")
    m.declare_observator("medical", name="Dr Alice",
                         keywords=["patient", "consent", "clinical"])
    m.declare_observator("legal", name="Bob the Lawyer",
                         keywords=["contract", "liability", "statute"])
    routes = m.route("patient consent treatment", k=2)
    assert len(routes) == 2
    # Medical should rank higher for this query
    top_ids = [r["id"] for r in routes]
    assert top_ids[0] == "medical"
