"""
Bi-temporal validity tests (Graphiti t_valid/t_invalid).

A superseded fact is INVALIDATED (tagged valid:until:<date>), not deleted, so
"as-of" queries still see it. Contradiction-driven invalidation is last-write-
wins, LLM-arbitrated. Deterministic via a scripted fake LLM.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory
from metacog.epistemic import Point, PointKind


def _fact(mem, content, fid):
    return mem.ingest(content, kind="FACT", id=fid)


def test_is_valid_default_and_until():
    enc = SimpleEncoder()
    mem = Memory(encoder=enc)
    p = _fact(mem, "John lives in Paris", "D1")
    assert Memory.is_valid(p)                       # no valid:until -> valid
    mem.invalidate("D1", "2022-05-04")
    assert "valid:until:2022-05-04" in p.tags and "invalidated" in p.tags
    assert not Memory.is_valid(p)                   # no as_of -> superseded
    assert Memory.is_valid(p, as_of="2022-01-01")   # before until -> still held
    assert not Memory.is_valid(p, as_of="2022-09-01")


def test_invalidate_is_idempotent():
    enc = SimpleEncoder()
    mem = Memory(encoder=enc)
    _fact(mem, "x", "D1")
    assert mem.invalidate("D1", "2022-05-04") is True
    assert mem.invalidate("D1", "2022-06-01") is False   # already invalid
    assert mem.invalidate("nope", "2022-01-01") is False


class _YesLLM:
    def generate(self, prompt, max_tokens=4):
        return "yes"


def test_contradiction_invalidates_prior_last_write_wins():
    enc = SimpleEncoder()
    mem = Memory(encoder=enc, llm=_YesLLM())
    old = _fact(mem, "John lives in Paris", "D1")
    # new fact carries an event date tag (the prefix is parsed at ingest)
    new = mem.ingest("[4 May 2022] John moved to London", kind="FACT", id="D2")
    inv = mem.invalidate_contradictions(new, threshold=-1.0)
    assert "D1" in inv
    assert not Memory.is_valid(old)                 # old superseded
    assert "valid:until:2022-05-04" in old.tags     # at the new fact's date
    assert Memory.is_valid(new)                      # new still valid


class _NoLLM:
    def generate(self, prompt, max_tokens=4):
        return "no"


def test_no_contradiction_keeps_prior_valid():
    enc = SimpleEncoder()
    mem = Memory(encoder=enc, llm=_NoLLM())
    old = _fact(mem, "John likes coffee", "D1")
    new = mem.ingest("[4 May 2022] John adopted a puppy", kind="FACT", id="D2")
    assert mem.invalidate_contradictions(new, threshold=-1.0) == []
    assert Memory.is_valid(old)
