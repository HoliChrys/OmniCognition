"""
Tests for the LoCoMo eval harness — scoring + debug instrumentation.

Properties :
  E1   f1_score is the token-overlap harmonic mean (precision/recall)
  E2   f1_score is 1.0 on identical strings, 0.0 on disjoint
  E3   f1_score is case- and punctuation-insensitive (tokenizer)
  E4   evaluate_sample(answerer="chunk") runs with no API and reports
       recall_at_5 / recall_at_10 / f1 per category
  E5   --debug-jsonl writer emits one well-formed JSON object per QA
       carrying the documented schema (question, gold, pred, f1, r5,
       r10, retrieved_top10, evidence hits/misses)
  E6   evidence hits/misses are computed against gold dia_ids
"""

from __future__ import annotations

import io
import json

import pytest

from benchmarks.locomo.eval import (
    _clean_session_date,
    evaluate_sample,
    f1_score,
    ingest_conversation,
)
from metacog import Memory
from metacog.defaults import SimpleEncoder


# ---------- E1, E2, E3 ----------------------------------------------------


def test_f1_identical_is_one():
    assert f1_score("7 May 2023", "7 May 2023") == pytest.approx(1.0)


def test_f1_disjoint_is_zero():
    assert f1_score("completely different", "7 May 2023") == 0.0


def test_f1_empty_is_zero():
    assert f1_score("", "anything") == 0.0
    assert f1_score("anything", "") == 0.0


def test_f1_partial_overlap():
    # pred has 1 useful token of 4 ; gold has 1 of 2
    # precision = 1/4, recall = 1/2 → F1 = 2*(.25*.5)/(.25+.5) = 1/3
    f1 = f1_score("in Berkeley last summer", "Berkeley California")
    assert f1 == pytest.approx(1.0 / 3.0, abs=1e-4)


def test_f1_case_and_punctuation_insensitive():
    assert f1_score("Berkeley!", "berkeley") == pytest.approx(1.0)
    assert f1_score("7 May, 2023.", "7 may 2023") == pytest.approx(1.0)


# ---------- fixtures ------------------------------------------------------


def _synthetic_sample() -> dict:
    """A minimal LoCoMo-shaped conversation with 2 QAs whose gold
    evidence points at specific dia_ids."""
    return {
        "sample_id": "test-conv",
        "conversation": {
            "session_1_date_time": "1 May 2023",
            "session_1": [
                {"speaker": "Caroline", "text": "I went to the support group on 7 May 2023.",
                 "dia_id": "D1:1"},
                {"speaker": "Melanie", "text": "I painted a sunrise back in 2022.",
                 "dia_id": "D1:2"},
                {"speaker": "Caroline", "text": "The weather was lovely that spring.",
                 "dia_id": "D1:3"},
            ],
        },
        "qa": [
            {
                "question": "When did Caroline go to the support group?",
                "answer": "7 May 2023",
                "evidence": ["D1:1"],
                "category": 2,
            },
            {
                "question": "When did Melanie paint a sunrise?",
                "answer": "2022",
                "evidence": ["D1:2"],
                "category": 2,
            },
        ],
    }


# ---------- E4 ------------------------------------------------------------


def test_evaluate_sample_chunk_no_api():
    summary = evaluate_sample(
        _synthetic_sample(), "simple",
        k_retrieve=10, top_chunks_for_answer=3,
        answerer="chunk",
    )
    assert summary["sample_id"] == "test-conv"
    assert summary["n_dialog_turns_ingested"] == 3
    assert "overall" in summary
    o = summary["overall"]
    assert o["n"] == 2
    assert 0.0 <= o["recall_at_5"] <= 1.0
    assert 0.0 <= o["recall_at_10"] <= 1.0
    assert 0.0 <= o["f1"] <= 1.0
    # category 2 is present in by_category
    assert 2 in summary["by_category"]


# ---------- E5, E6 --------------------------------------------------------


def test_debug_writer_emits_one_object_per_qa():
    buf = io.StringIO()
    evaluate_sample(
        _synthetic_sample(), "simple",
        k_retrieve=10, top_chunks_for_answer=3,
        answerer="chunk",
        debug_writer=buf,
    )
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 2  # one per QA

    for ln in lines:
        rec = json.loads(ln)  # must be valid JSON
        # Documented schema
        for key in (
            "sample_id", "category", "question", "gold_answer", "pred",
            "f1", "r5", "r10", "retrieved_top10", "gold_evidence",
            "evidence_hits_in_top5", "evidence_missed_top10",
            "react_trace", "tokens_in", "tokens_out", "steps",
        ):
            assert key in rec, f"missing {key}"
        assert isinstance(rec["retrieved_top10"], list)
        # each retrieved chunk dump carries id + kind + content
        for chunk in rec["retrieved_top10"]:
            assert set(chunk.keys()) == {"id", "kind", "content"}
        # chunk answerer makes no API calls → no react steps / tokens
        assert rec["react_trace"] == []
        assert rec["tokens_in"] == 0
        assert rec["tokens_out"] == 0


def test_terse_strips_verbose_agent_wrapping():
    from benchmarks.locomo.mcp_agent import terse
    assert terse("Based on the search results, she went on **7 May 2023**.") \
        == "7 May 2023"
    assert terse("According to the evidence, the answer is Sweden.") == "Sweden"
    assert terse("... which means 2022. Not mentioned.") == "Not mentioned"
    assert terse("Counseling and mental health.") == "Counseling and mental health"
    assert terse("7 May 2023") == "7 May 2023"


def test_clean_session_date_strips_clock_and_comma():
    assert _clean_session_date("1:56 pm on 8 May, 2023") == "8 May 2023"
    assert _clean_session_date("8:56 pm on 20 July, 2023") == "20 July 2023"
    # already-clean date passes through
    assert _clean_session_date("1 May 2023") == "1 May 2023"
    # empty / missing
    assert _clean_session_date("") == ""


def test_ingest_with_dates_prefixes_session_date():
    """with_dates=True attaches the [session date] anchor so relative
    references in the turn can be resolved by the answerer."""
    conv = {
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "text": "I went yesterday.", "dia_id": "D1:3"},
        ],
    }
    m = Memory(encoder=SimpleEncoder())
    n = ingest_conversation(m, conv, with_dates=True)
    assert n == 1
    p = next(q for q in m.points if q.id == "D1:3")
    assert p.content.startswith("[8 May 2023] ")
    assert "yesterday" in p.content.lower()


def test_ingest_without_dates_has_no_prefix():
    conv = {
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "text": "I went yesterday.", "dia_id": "D1:3"},
        ],
    }
    m = Memory(encoder=SimpleEncoder())
    ingest_conversation(m, conv, with_dates=False)
    p = next(q for q in m.points if q.id == "D1:3")
    assert not p.content.startswith("[")
    assert p.content == "Caroline: I went yesterday."


def test_debug_evidence_hits_and_misses_partition():
    buf = io.StringIO()
    evaluate_sample(
        _synthetic_sample(), "simple",
        k_retrieve=10, answerer="chunk", debug_writer=buf,
    )
    for ln in buf.getvalue().splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        gold = set(rec["gold_evidence"])
        hits = set(rec["evidence_hits_in_top5"])
        misses = set(rec["evidence_missed_top10"])
        # hits ⊆ gold, misses ⊆ gold, hits ∩ misses = ∅
        assert hits <= gold
        assert misses <= gold
        assert not (hits & misses)
