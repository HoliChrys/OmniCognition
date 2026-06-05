"""
Tests for the compose-set fallback in `MetaWalker._composable_evidence`.

When the Chain-of-Note labels almost nothing relevant (a vocab-gap walk),
the walker would return a near-empty compose set and the agent would
abstain. The fallback re-ranks the unlabelled `_fact_ids_cum` by cosine
similarity to the query embedding and promotes the top-K as
`relevance: "retrieved"` — a weaker confidence than "relevant" so the
prompt can tell them apart.

These tests drive `_composable_evidence` directly on a tiny memory with a
SimpleEncoder so the behaviour is deterministic and no LLM is involved.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory
from metacog.meta_walk import (
    MetaWalker, _FALLBACK_K, _FALLBACK_RELEVANT_GATE,
)


def _build_walker(query: str, points_text):
    m = Memory(encoder=SimpleEncoder())
    for i, t in enumerate(points_text, 1):
        m.ingest(t, kind="FACT", id=f"D:{i}")
    w = MetaWalker(query, m, commit=False)
    # Pre-fill the cumulative pool as if a walk had retrieved everything.
    w._fact_ids_cum = [f"D:{i}" for i in range(1, len(points_text) + 1)]
    return w, m


def test_fallback_triggers_when_relevant_cum_empty_and_pool_large():
    """The cat1-shape case: walker bombed (0 relevant), pool has on-topic
    turns. The fallback must promote the top-K by similarity, labelled
    `retrieved` (not `relevant`)."""
    q = "adopting children family"
    texts = [
        "general greeting hello there",
        "weather is nice today nothing important",
        "researching adoption agencies for our family",  # the on-topic gold
        "another unrelated chat about hobbies",
        "thinking about adopting kids next year",
        "more filler conversation about food",
    ]
    w, _ = _build_walker(q, texts)
    w._relevant_cum = []                       # CoN labelled nothing
    out = w._composable_evidence()
    ids = [e["id"] for e in out]
    labels = {e["id"]: e["relevance"] for e in out}
    # The on-topic turns must surface, all labelled "retrieved"
    assert "D:3" in ids and "D:5" in ids, ids
    assert labels["D:3"] == "retrieved"
    assert labels["D:5"] == "retrieved"
    assert len(out) <= _FALLBACK_K + 0       # cum was empty so at most K


def test_fallback_does_not_trigger_when_cum_above_gate():
    """When the CoN already labelled >= _FALLBACK_RELEVANT_GATE facts, the
    fallback stays quiet — the walker's own judgement is trusted."""
    q = "adopting children family"
    texts = [
        "greeting one", "greeting two", "filler",
        "researching adoption agencies for our family",
        "filler again", "thinking about adopting kids",
    ]
    w, m = _build_walker(q, texts)
    # Seed `_relevant_cum` with two arbitrary points (>= gate); CoN trusted.
    by_id = {p.id: p for p in m.points}
    w._relevant_cum = [by_id["D:1"], by_id["D:2"]]
    out = w._composable_evidence()
    ids = [e["id"] for e in out]
    # Only the CoN-labelled facts come out, no fallback promotion.
    assert ids == ["D:1", "D:2"]
    assert all(e["relevance"] == "relevant" for e in out)


def test_fallback_skips_already_seen_ids():
    """Anything already in `_relevant_cum` must not be re-added."""
    q = "adopting children family"
    texts = ["filler", "researching adoption agencies",
             "thinking about adopting kids", "more filler"]
    w, m = _build_walker(q, texts)
    by_id = {p.id: p for p in m.points}
    w._relevant_cum = [by_id["D:2"]]    # < gate, fallback runs
    out = w._composable_evidence()
    ids = [e["id"] for e in out]
    # D:2 appears only once (no double count)
    assert ids.count("D:2") == 1
    # And the next most-similar (D:3) is promoted as retrieved.
    labels = {e["id"]: e["relevance"] for e in out}
    assert labels["D:2"] == "relevant"
    if "D:3" in labels:
        assert labels["D:3"] == "retrieved"


def test_fallback_silent_when_pool_small():
    """If `_fact_ids_cum` is small (<= _FALLBACK_K), the fallback stays
    quiet — there is nothing useful to promote anyway."""
    q = "adopting children"
    w, _ = _build_walker(q, ["filler one", "researching adoption agencies"])
    w._relevant_cum = []
    out = w._composable_evidence()
    # Pool has 2 ids (< _FALLBACK_K=5), fallback gate skipped.
    assert out == []
