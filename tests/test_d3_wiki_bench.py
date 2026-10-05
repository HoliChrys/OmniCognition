"""D3 bench harness (Linear TAC-939): the scoring and Proxy's decision rule,
pinned BEFORE any measurement, plus one offline end-to-end run on a tiny
corpus (SimpleEncoder, scripted LLM, no reranker).

The rule under test (TAC-190 decisions doc, 2026-10-04): per context, keep
the best coverage among strategies with noise <= 5 %, 0 off-topic item and
query p95 <= 2 s; none eligible -> walk (omni); a gap under 10 points -> walk.
"""
from __future__ import annotations

import os
import sys

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


def _tiny(tmp_path):
    """A live store + 2 pinned notes + 1 in-topic and 1 off-topic question."""
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
    return spec, off, str(scratch)


def test_end_to_end_on_a_tiny_corpus_never_touches_the_source_store(tmp_path):
    from metacog.defaults import SimpleEncoder

    spec, off, scratch = _tiny(tmp_path)
    res = d3.run_context("tachikoma.c", spec, off, scratch,
                         (SimpleEncoder(), None), _ScriptedLLM)
    assert res["bench_writes_under_live_root"] == []
    assert res["live_store_changed_by_other_process"] is False
    assert res["stopped"] is None and res["ineligible"] == {}
    assert res["llm"]["errors"] == 0 and not res["llm"]["cap_reached"]
    assert res["build_empty"]["sleep"]["seeds"] == 2
    assert res["build_empty"]["walk"]["ingest"]["added"] == 2
    got = {(r["strategy"], r["qid"]) for r in res["records"]}
    assert got == {(s, q) for s in ("walk", "sleep", "deployed") for q in ("q1", "o1")}
    scores = d3.score(res["records"], {"contexts": {"tachikoma.c": spec}})
    assert set(scores["tachikoma.c"]) == {"walk", "sleep", "deployed"}


# ── guards (decision Proxy, TAC-209, 2026-10-05) ────────────────────────

def test_live_write_guard_refuses_and_records_writes_but_not_reads(tmp_path):
    live, elsewhere = tmp_path / "live", tmp_path / "scratch"
    live.mkdir()
    elsewhere.mkdir()
    (live / "memory.pkl").write_bytes(b"live")
    with d3.live_write_guard([str(live)]) as violations:
        assert (live / "memory.pkl").read_bytes() == b"live"     # read: fine
        (elsewhere / "x").write_text("ok")                       # outside: fine
        for attempt in (lambda: open(live / "memory.pkl", "ab"),
                        lambda: open(live / "new", "w"),
                        lambda: os.open(str(live / "memory.pkl"), os.O_RDWR),
                        lambda: os.remove(str(live / "memory.pkl")),
                        lambda: os.replace(str(elsewhere / "x"), str(live / "x"))):
            with pytest.raises(PermissionError):
                attempt()
    assert [v["event"] for v in violations] == [
        "open", "open", "open", "os.remove", "os.rename"]
    assert (live / "memory.pkl").read_bytes() == b"live"
    assert sorted(os.listdir(live)) == ["memory.pkl"]
    # outside the `with`, the (unremovable) hook is idle
    with open(live / "after", "w") as fh:
        fh.write("free again")


@pytest.mark.skipif(sys.version_info < (3, 10),
                    reason="sqlite3.connect is audited from Python 3.10")
def test_live_write_guard_refuses_a_sqlite_connection_unless_read_only(tmp_path):
    import sqlite3
    db = tmp_path / "memory.pkl.journal.db"
    sqlite3.connect(str(db)).close()
    with d3.live_write_guard([str(tmp_path)]) as violations:
        with pytest.raises(PermissionError):
            sqlite3.connect(str(db))
        sqlite3.connect(f"file:{db}?mode=ro", uri=True).close()
    assert [v["event"] for v in violations] == ["sqlite3.connect"]


def test_a_write_swallowed_by_failure_safe_code_still_fails_the_run(tmp_path, monkeypatch):
    from metacog.defaults import SimpleEncoder

    spec, off, scratch = _tiny(tmp_path)
    live_store = spec["store_source"]
    real_build = d3._build

    def leaky(*a, **kw):
        try:                                   # a failure-safe library path
            open(os.path.join(live_store, "memory.pkl"), "ab").close()
        except Exception:
            pass
        return real_build(*a, **kw)
    monkeypatch.setattr(d3, "_build", leaky)
    with pytest.raises(SystemExit, match="write under a live root"):
        d3.run_context("tachikoma.c", spec, off, scratch,
                       (SimpleEncoder(), None), _ScriptedLLM)


class _BrokenClientLLM:
    """Absorbs its failure into "" like ClaudeLLM.generate, but counts it."""

    def __init__(self):
        self.llm_errors = 0

    def generate(self, prompt, max_tokens=None):
        self.llm_errors += 1
        return ""


def test_counting_llm_counts_absorbed_errors_and_enforces_the_cap():
    budget = d3.new_budget(2)
    llm = d3.CountingLLM(_BrokenClientLLM(), budget)
    assert llm.generate("a") == "" and llm.generate("b") == ""
    assert (llm.calls, llm.errors) == (2, 2)
    with pytest.raises(d3.LLMCapReached):
        llm.generate("c")                     # NOT made: inner not called
    assert budget == {"cap": 2, "used": 2, "errors": 2, "refused": 1,
                      "reached": True}
    assert llm._inner.llm_errors == 2


def test_llm_cap_stops_the_context_and_says_so(tmp_path):
    from metacog.defaults import SimpleEncoder

    spec, off, scratch = _tiny(tmp_path)
    res = d3.run_context("tachikoma.c", spec, off, scratch,
                         (SimpleEncoder(), None), _ScriptedLLM, llm_cap=0)
    assert res["llm"]["cap_reached"] and res["llm"]["calls"] == 0
    assert res["stopped"] and "LLM cap" in res["stopped"]
    assert set(res["ineligible"]) == {"walk", "sleep"}
    v = d3.decide({}, {"tachikoma.c": res["ineligible"]})["tachikoma.c"]
    assert v["strategy"] == "walk" and not v["eligible"]


def test_sleep_over_its_build_cap_is_ineligible_and_walk_still_measured(tmp_path, monkeypatch):
    import time as _time

    from metacog.defaults import SimpleEncoder
    from metacog.memory import Memory

    monkeypatch.setattr(Memory, "sleep", lambda self, t=None: _time.sleep(5) or {})
    spec, off, scratch = _tiny(tmp_path)
    res = d3.run_context("tachikoma.c", spec, off, scratch,
                         (SimpleEncoder(), None), _ScriptedLLM, sleep_cap_s=0.3)
    assert res["build_empty"]["sleep"]["cap_exceeded_s"] == 0.3
    assert res["build_empty"]["sleep"]["elapsed_s"] < 5
    assert "sleep" in res["ineligible"] and "sleep" not in res["build_live"]
    assert {r["strategy"] for r in res["records"]} == {"walk", "deployed"}
    scores = d3.score(res["records"], {"contexts": {"tachikoma.c": spec}})
    v = d3.decide({"tachikoma.c": {s: c for s, c in scores["tachikoma.c"].items()
                                   if s in d3.STRATEGIES}},
                  {"tachikoma.c": res["ineligible"]})["tachikoma.c"]
    assert "cap" in v["breaches"]["sleep"][0]


def test_decide_applies_the_rule_among_strategies_that_finished():
    v = d3.decide({"c": {"walk": _cell(40.0)}},
                  {"c": {"sleep": "build(sleep) exceeded the 5400 s cap"}})["c"]
    assert v["strategy"] == "walk" and v["eligible"] == {"walk": 40.0}
    assert v["breaches"]["sleep"] == ["build(sleep) exceeded the 5400 s cap"]


class _ControlLLM:
    def __init__(self, out, errors=0):
        self.out, self.llm_errors, self.model = out, errors, "m"
        self.last_error = "ModuleNotFoundError: No module named 'anthropic'" if errors else None

    def generate(self, prompt, max_tokens=None):
        return self.out


def test_control_call_refuses_to_start_on_an_llm_that_does_not_answer():
    assert d3.control_call(_ControlLLM("OK"))["ok"] is True
    with pytest.raises(SystemExit, match="anthropic"):
        d3.control_call(_ControlLLM("", errors=1))
    with pytest.raises(SystemExit, match="empty answer"):
        d3.control_call(_ControlLLM("  "))
