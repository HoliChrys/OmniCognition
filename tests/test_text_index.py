"""
Text index (Phase 2) — strict equivalence with the naive per-query scans.

The index memoizes per-doc token artifacts; correctness contract: byte-for-byte
the same tokens/docs/scores as the naive code it replaces, self-healing on
points created outside ingest, re-memoized on keyword mutation, reset on load().
"""

from __future__ import annotations

import re

from metacog.bm25 import bm25_score, stem_tokens, tokenize
from metacog.defaults import SimpleEncoder
from metacog.memory import Memory
from metacog.text_index import TextIndex, bm25_doc_tokens, raw_token_set


def _naive_toks(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _naive_bm25_doc(content, keywords):
    content_tokens = stem_tokens(tokenize(content or "")[:40])
    if keywords:
        kw_tokens = stem_tokens([kw.lower() for kw in keywords])
        cs = set(content_tokens)
        return content_tokens + [t for t in kw_tokens if t not in cs][:8]
    return content_tokens


_TEXTS = ["Iran strike on the refinery!", "  ", "weird —— punctuation 42x",
          "Researching studied cities", ""]


def test_raw_token_set_matches_naive():
    for t in _TEXTS:
        assert raw_token_set(t) == frozenset(_naive_toks(t))


def test_bm25_doc_matches_naive():
    for t in _TEXTS:
        for kws in (None, [], ["Strikes", "refineries", "extra", "more",
                              "words", "again", "and", "again2", "nine"]):
            assert bm25_doc_tokens(t, kws) == _naive_bm25_doc(t, kws)


def test_bm25_score_identical_with_index():
    m = Memory(encoder=SimpleEncoder())
    pts = [m.ingest("iran strike on the refinery", kind="FACT", id="A"),
           m.ingest("markets reacted to oil prices", kind="FACT", id="B"),
           m.ingest("cooking recipes for dinner", kind="FACT", id="C")]
    ti = TextIndex()
    naive = bm25_score("refinery strike oil", pts)
    fast = bm25_score("refinery strike oil", pts, text_index=ti)
    assert [(s, p.id) for s, p in naive] == [(s, p.id) for s, p in fast]


def test_candidates_is_exactly_the_overlap_set():
    m = Memory(encoder=SimpleEncoder())
    a = m.ingest("alpha beta", kind="FACT", id="A")
    b = m.ingest("beta gamma", kind="FACT", id="B")
    c = m.ingest("delta epsilon", kind="FACT", id="C")
    ti = TextIndex()
    cand = {p.id for p in ti.candidates({"beta"}, [a, b, c])}
    assert cand == {"A", "B"}                     # the naive score>0 set


def test_memo_rekeyed_on_keyword_mutation():
    m = Memory(encoder=SimpleEncoder())
    p = m.ingest("iran strike", kind="FACT", id="A")
    ti = TextIndex()
    d1 = ti.bm25_doc(p)
    assert ti.bm25_doc(p) is d1                   # memo hit
    p.keywords = ["refinery"]                     # the doc card mutates keywords
    d2 = ti.bm25_doc(p)
    assert d2 == _naive_bm25_doc(p.content, p.keywords)
    assert d2 != d1


def test_discard_cleans_postings():
    m = Memory(encoder=SimpleEncoder())
    a = m.ingest("alpha beta", kind="FACT", id="A")
    ti = TextIndex()
    ti.raw_tokens(a)
    ti.discard("A")
    assert "A" not in ti.postings.get("alpha", set())


def test_load_resets_index(tmp_path):
    path = str(tmp_path / "mem.pkl")
    m = Memory(encoder=SimpleEncoder(), storage_path=path)
    m.ingest("alpha beta", kind="FACT", id="A")
    m.search_nodes("alpha", mode="sim", on="content")   # warms the index
    m.save()
    m.load()
    assert m._text_index._raw == {}                      # fresh, refills lazily
    got = [p.id for p in m.search_nodes("alpha", mode="sim", on="content")]
    assert got == ["A"]                                  # self-healing
