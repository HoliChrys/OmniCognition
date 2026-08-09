"""
ACT-R retrieval-threshold abstention — the last T1 piece.

Retrieval FAILS (returns []) when no chunk is sufficiently activated for the
query : an explicit 'I don't know' instead of the least-bad match. The threshold
is emergent by default (the best match must stand out from the background) and
overridable by a fixed tau (ACT-R's constant retrieval threshold).
"""

from __future__ import annotations

from metacog.memory import Memory


class _Enc:
    """Controlled encoder : each fruit is an orthogonal basis vector ; an unknown
    query maps to the equidistant vector (nothing stands out)."""
    dim = 4
    _BASIS = ("apple", "banana", "cherry", "date")

    def encode(self, text):
        t = (text or "").lower()
        for i, kw in enumerate(self._BASIS):
            if kw in t:
                v = [0.0] * 4
                v[i] = 1.0
                return v
        return [0.25, 0.25, 0.25, 0.25]


def _mem():
    m = Memory(encoder=_Enc())
    for kw in _Enc._BASIS:
        m.ingest(f"a fact about {kw}", kind="FACT", id=kw)
    return m


def test_emergent_abstains_only_when_nothing_stands_out():
    m = _mem()
    assert m.abstains("apple pie recipe") is False       # clear match
    assert m.abstains("grapefruit smoothie") is True     # nothing activated


def test_fixed_tau_threshold():
    m = _mem()
    assert m.abstains("apple", threshold=0.99) is False   # best 1.0 >= tau
    assert m.abstains("apple", threshold=1.01) is True     # nothing that active
    assert m.abstains("grapefruit", threshold=0.6) is True   # best 0.5 < 0.6
    assert m.abstains("grapefruit", threshold=0.4) is False  # best 0.5 >= 0.4


def test_retrieve_abstain_returns_empty_on_failure():
    m = _mem()
    assert m.retrieve("grapefruit smoothie", abstain=True) == []   # failure
    assert len(m.retrieve("apple pie recipe", abstain=True)) > 0    # succeeds


def test_retrieve_default_never_abstains():
    m = _mem()
    # without opt-in, retrieval always returns the least-bad match (backward-compat)
    assert len(m.retrieve("grapefruit smoothie")) > 0


def test_small_corpus_never_abstains():
    m = Memory(encoder=_Enc())
    m.ingest("a fact about apple", kind="FACT", id="apple")
    assert m.abstains("grapefruit") is False              # < 4 -> no background


def test_retrieval_activation_is_max_cosine():
    m = _mem()
    assert round(m.retrieval_activation("apple pie"), 3) == 1.0
    assert round(m.retrieval_activation("grapefruit"), 3) == 0.5
