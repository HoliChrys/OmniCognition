"""
The cross-encoder reranker wired like mnema : cosine pre-fetch -> joint
(query, doc) scoring -> sigmoid -> top-k, before the ACT-R blends.

  - `make_reranker` resolves METACOG_RERANKER : none / auto (warned fallback
    to None) / fastembed[:model] (never a silent downgrade).
  - `Memory.retrieve` uses `self.reranker` by default, over-fetches
    `rerank_pre`, reorders by the reranker, exposes the raw logit, and stays
    cosine-ordered on `rerank=False` or on a reranker failure.
  - the real jina multilingual model runs only when cached locally or
    METACOG_REAL_EMBED=1 (1.1 GB download).
"""

from __future__ import annotations

import os

import pytest

from metacog import defaults as D
from metacog.defaults import SimpleEncoder, make_reranker
from metacog.memory import Memory


class _Stub:
    """Scores by a fixed preference table ; records what it was asked."""

    def __init__(self, prefer: dict, fail: bool = False):
        self.prefer, self.fail, self.calls = prefer, fail, []

    def rerank(self, query, docs):
        self.calls.append(len(docs))
        if self.fail:
            raise RuntimeError("model down")
        return [self.prefer.get(d, -5.0) for d in docs]


def _corpus(**kw):
    m = Memory(encoder=SimpleEncoder(), **kw)
    m.ingest("alpha beta gamma", kind="FACT", id="A")
    m.ingest("alpha beta delta", kind="FACT", id="B")
    for i in range(8):
        m.ingest(f"unrelated filler number {i} about soup", kind="FACT", id=f"F{i}")
    return m


def test_make_reranker_none_auto_fallback_and_explicit(monkeypatch, capsys):
    assert make_reranker("none") is None and make_reranker("off") is None
    monkeypatch.setenv("METACOG_RERANKER", "none")
    assert make_reranker() is None

    class Boom:
        def __init__(self, *a, **k):
            raise ImportError("no model")
    monkeypatch.setattr(D, "CrossEncoderReranker", Boom)
    assert make_reranker("auto") is None
    assert "retrieval stays cosine-only" in capsys.readouterr().err
    with pytest.raises(ImportError):
        make_reranker("fastembed")
    with pytest.raises(ValueError):
        make_reranker("bogus")


def test_retrieve_reranks_prefetched_candidates_and_exposes_logits():
    stub = _Stub({"alpha beta delta": 4.0, "alpha beta gamma": -1.0})
    m = _corpus(reranker=stub)
    hits = m.retrieve("alpha beta gamma", k=2)          # cosine would put A first
    assert [h["id"] for h in hits] == ["B", "A"]         # the reranker decided
    assert hits[0]["rerank_score"] == 4.0 and 0.98 < hits[0]["score"] < 1.0
    assert stub.calls[0] == 10                           # pre-fetched the whole corpus (<30)
    # opt-out -> cosine order, no logits
    off = m.retrieve("alpha beta gamma", k=2, rerank=False)
    assert off[0]["id"] == "A" and "rerank_score" not in off[0]
    # no reranker wired -> identical to opt-out
    plain = _corpus().retrieve("alpha beta gamma", k=2)
    assert [h["id"] for h in plain] == [h["id"] for h in off]


def test_rerank_pre_bounds_the_prefetch_and_failure_keeps_cosine_order():
    stub = _Stub({}, fail=True)
    m = _corpus(reranker=stub)
    hits = m.retrieve("alpha beta gamma", k=2, rerank_pre=4)
    assert stub.calls[0] == 4                            # bounded pre-fetch
    assert hits[0]["id"] == "A" and len(hits) == 2       # cosine order survived


def test_rerank_runs_before_the_actr_blends():
    from metacog.journal import Journal
    stub = _Stub({"alpha beta delta": 4.0, "alpha beta gamma": -1.0})
    m = _corpus(reranker=stub, journal=Journal())
    m.recency_weight = 0.5                               # need-odds blend ON
    # A was accessed a lot -> need-odds favour A ; the reranker favours B
    for _ in range(5):
        m.record_retrieval(["A"], query_text="q")
    hits = m.retrieve("alpha beta gamma", k=2)
    assert {h["id"] for h in hits} == {"A", "B"}         # both survive the blend
    assert all("rerank_score" in h for h in hits)        # blended ON the rerank


def _model_cached() -> bool:
    if os.environ.get("METACOG_REAL_EMBED") == "1":
        return True
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        TextCrossEncoder(model_name=D.DEFAULT_RERANK_MODEL, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _model_cached(), reason="jina reranker not cached ; "
                    "set METACOG_REAL_EMBED=1 to download it (1.1 GB)")
def test_real_jina_reranker_is_multilingual_and_cached():
    rr = make_reranker("fastembed")
    assert rr.reranker_id == f"fastembed:{D.DEFAULT_RERANK_MODEL}"
    s = rr.rerank("le chat dort sur le canapé",
                  ["the cat is sleeping on the sofa", "quarterly revenue grew"])
    assert s[0] > s[1]                                   # cross-lingual relevance
    assert rr.rerank("le chat dort sur le canapé", ["the cat is sleeping on the sofa"]) == [s[0]]
    m = Memory(encoder=D.make_encoder("fastembed"), reranker=rr)
    m.ingest("Le rapport trimestriel montre une hausse du chiffre d'affaires", kind="FACT", id="fin")
    m.ingest("Le chat de Marie dort tout l'après-midi sur le canapé", kind="FACT", id="cat")
    m.ingest("La randonnée en montagne était magnifique ce week-end", kind="FACT", id="hike")
    m.ingest("Recette de soupe aux carottes et au gingembre", kind="FACT", id="soup")
    top = m.retrieve("where does Marie's cat nap?", k=1)[0]
    assert top["id"] == "cat" and "rerank_score" in top
