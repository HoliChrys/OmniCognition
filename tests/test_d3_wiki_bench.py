"""D3 bench harness (Linear TAC-939): the scoring and Proxy's decision rule,
pinned BEFORE any measurement, plus one offline end-to-end run on a tiny
corpus (SimpleEncoder, scripted LLM, no reranker).

The rule under test (TAC-190 decisions doc, 2026-10-04): per context, keep
the best coverage among strategies with noise <= 5 %, 0 off-topic item and
query p95 <= 2 s; none eligible -> walk (omni); a gap under 10 points -> walk.
"""
from __future__ import annotations

import os

import pytest
import yaml

from benchmarks.d3_wiki import run_d3 as d3

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = os.path.join(HERE, "..", "benchmarks", "d3_wiki", "questions.yaml")


def _cell(coverage=80.0, noise=0.0, off=0, p95=1.0):
    return {"coverage": coverage, "noise": noise, "offtopic_items": off,
            "query_p95_s": p95}


# ── citation ────────────────────────────────────────────────────────────

def test_cited_note_by_id_and_by_tag():
    assert d3.cited_note("notes:mise/undo#abc123") == "notes:mise/undo"
    assert d3.cited_note("f_42", ["deepwiki", "note:notes:trace/README"]) == "notes:trace/README"
    assert d3.cited_note("f_42", ["deepwiki"]) is None


def test_nearest_rank_p95_is_a_measured_value():
    xs = [float(i) for i in range(1, 21)]          # 20 values
    assert d3._pct(xs, 0.95) == 19.0
    assert d3._pct([3.0], 0.95) == 3.0


# ── scoring ─────────────────────────────────────────────────────────────

QS = {"contexts": {"c": {"questions": [
    {"id": "q1", "expected": ["notes:a"], "relevant": ["notes:a", "notes:b"]},
    {"id": "q2", "expected": ["notes:b"], "relevant": ["notes:b"]},
]}}}


def _rec(qid, notes, secs=0.5, off=False, invalid=False):
    return {"ctx": "c", "strategy": "walk", "qid": qid, "offtopic": off,
            "invalid_offtopic": invalid, "seconds": secs,
            "items": [{"id": str(n), "note": n} for n in notes]}


def test_score_coverage_noise_and_offtopic():
    s = d3.score([
        _rec("q1", ["notes:a", None]),            # covered, uncited item is not noise
        _rec("q2", ["notes:a"]),                  # not covered, cites a wrong note: noise
        _rec("o1", ["notes:a"], off=True),        # 1 off-topic item
        _rec("o2", ["x"], off=True, invalid=True),  # store holds the topic: not scored
    ], QS)["c"]["walk"]
    assert s["coverage"] == 50.0
    assert s["noise"] == 0.5
    assert s["items_uncited"] == 1
    assert s["offtopic_items"] == 1 and s["offtopic_asked"] == 1
    assert s["offtopic_invalid"] == ["o2"]
    assert s["query_n"] == 4


# ── Proxy's rule ────────────────────────────────────────────────────────

def test_best_coverage_wins_when_both_eligible_and_gap_is_10_points():
    v = d3.decide({"c": {"walk": _cell(60.0), "sleep": _cell(70.0)}})["c"]
    assert v["strategy"] == "sleep"


def test_gap_under_10_points_keeps_walk():
    v = d3.decide({"c": {"walk": _cell(60.0), "sleep": _cell(69.9)}})["c"]
    assert v["strategy"] == "walk" and "10 points" in v["reason"]


@pytest.mark.parametrize("bad", [dict(noise=0.051), dict(off=1), dict(p95=2.01)])
def test_one_breach_makes_a_strategy_ineligible(bad):
    v = d3.decide({"c": {"walk": _cell(10.0), "sleep": _cell(90.0, **bad)}})["c"]
    assert v["strategy"] == "walk" and "sleep" in v["breaches"]


def test_noise_at_5_percent_and_p95_at_2s_are_inside_the_budget():
    v = d3.decide({"c": {"walk": _cell(10.0, p95=2.5),
                         "sleep": _cell(90.0, noise=0.05, p95=2.0)}})["c"]
    assert v["strategy"] == "sleep"


def test_nobody_eligible_is_walk_with_the_breaches_written():
    v = d3.decide({"c": {"walk": _cell(p95=70.0), "sleep": _cell(p95=72.0)}})["c"]
    assert v["strategy"] == "walk"
    assert v["breaches"]["walk"] and v["breaches"]["sleep"]


# ── the frozen question set ─────────────────────────────────────────────

def test_question_set_is_consistent_with_its_pins():
    q = yaml.safe_load(open(QUESTIONS, encoding="utf-8"))
    assert set(q["contexts"]) == {"global", "tachikoma.paralelle.GenAI"}
    assert q["offtopic"], "at least one off-topic question (TAC-340)"
    for ctx, spec in q["contexts"].items():
        assert spec["questions"], ctx
        for x in spec["questions"]:
            assert set(x["expected"]) <= set(x["relevant"]) <= set(spec["corpus"]), x["id"]


def test_corpus_check_refuses_a_changed_note(tmp_path):
    (tmp_path / "a.md").write_text("alpha")
    pins = d3.note_manifest(str(tmp_path))
    d3.check_corpus(str(tmp_path), pins)
    (tmp_path / "a.md").write_text("alpha, edited")
    with pytest.raises(SystemExit):
        d3.check_corpus(str(tmp_path), pins)


# ── one offline end-to-end run ──────────────────────────────────────────

class _ScriptedLLM:
    def generate(self, prompt, max_tokens=None):
        return ""


def test_end_to_end_on_a_tiny_corpus_never_touches_the_source_store(tmp_path):
    from metacog.defaults import SimpleEncoder
    from metacog.memory import Memory

    notes = tmp_path / "src_notes"
    (notes / "sub").mkdir(parents=True)
    (notes / "undo.md").write_text("# mise undo\nundo:undo annule la derniere action.\n")
    (notes / "sub" / "fuse.md").write_text("# mise fuse\nfuse:off ramene les workspaces.\n")
    store = tmp_path / "live" / "c"
    store.mkdir(parents=True)
    live = Memory(storage_path=str(store / "memory.pkl"), journal_path="auto",
                  encoder=SimpleEncoder(), llm=_ScriptedLLM())
    live.ingest("un fait sans rapport avec les notes", kind="FACT")
    live.save()
    spec = {"notes_source": str(notes), "store_source": str(store),
            "corpus": d3.note_manifest(str(notes)),
            "questions": [{"id": "q1", "q": "comment annuler la derniere action",
                           "expected": ["notes:undo"], "relevant": ["notes:undo"]}]}
    off = [{"id": "o1", "q": "recette du kouign-amann", "absent_marker": "(?i)kouign"}]
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    res = d3.run_context("tachikoma.c", spec, off, str(scratch),
                         (SimpleEncoder(), None), _ScriptedLLM)
    assert res["live_store_untouched"] is True
    assert res["build_empty"]["sleep"]["seeds"] == 2
    assert res["build_empty"]["walk"]["ingest"]["added"] == 2
    got = {(r["strategy"], r["qid"]) for r in res["records"]}
    assert got == {(s, q) for s in ("walk", "sleep", "deployed") for q in ("q1", "o1")}
    scores = d3.score(res["records"], {"contexts": {"tachikoma.c": spec}})
    assert set(scores["tachikoma.c"]) == {"walk", "sleep", "deployed"}
