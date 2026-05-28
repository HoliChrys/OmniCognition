"""
Tests for PointKind taxonomy (FACT / THOUGHT / ACTION).

Properties :
  K1   Default kind is FACT.
  K2   A(·) treats every kind identically (counter-driven, kind-blind).
  K3   detect_collisions only proposes intra-kind pairs.
  K4   resolve_collision rejects mixed-kind intermediates.
  K5   Child of a collision inherits the parents' shared kind.
  K6   sleep_cycle_collisions never merges cross-kind pairs even if
       they are geometrically close.
  K7   ACTION is a first-class Point with its own counters and state.
"""

from __future__ import annotations

import hashlib
import pytest

from metacog import (
    EpistemicState,
    Point,
    PointKind,
    SourceClass,
    assign_status,
    detect_collisions,
    resolve_collision,
    sleep_cycle_collisions,
)


class _SetOpsLLM:
    def extract_common(self, texts) -> str:
        texts = list(texts)
        if len(texts) < 2:
            return ""
        word_sets = [set(t.lower().split()) for t in texts]
        common = word_sets[0].copy()
        for ws in word_sets[1:]:
            common &= ws
        if not common:
            return ""
        seen = set()
        out = []
        for w in texts[0].lower().split():
            if w in common and w not in seen:
                out.append(w)
                seen.add(w)
        return " ".join(out)

    def remove_overlap(self, full: str, common: str) -> str:
        cset = set(common.lower().split())
        return " ".join(w for w in full.lower().split() if w not in cset)


class _BoWEncoder:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self._cache: dict = {}

    def encode(self, text: str):
        words = text.lower().split()
        if not words:
            return tuple(0.0 for _ in range(self.dim))
        acc = [0.0] * self.dim
        for w in words:
            if w not in self._cache:
                h = int(hashlib.md5(f"bow:{w}".encode()).hexdigest(), 16)
                self._cache[w] = tuple(
                    ((h >> (i * 8)) & 0xFF) / 255.0 for i in range(self.dim)
                )
            wv = self._cache[w]
            for i in range(self.dim):
                acc[i] += wv[i]
        return tuple(v / len(words) for v in acc)


# ---------- K1 -----------------------------------------------------------


def test_default_kind_is_fact():
    p = Point(id="X", content="x", embedding_orig=(0.0,))
    assert p.kind == PointKind.FACT


# ---------- K2 — A(·) is kind-blind --------------------------------------


def test_assign_status_ignores_kind():
    a = Point(id="A", content="x", embedding_orig=(0.0,),
              kind=PointKind.FACT, n_corrob=3, n_contra=1)
    b = Point(id="B", content="y", embedding_orig=(0.0,),
              kind=PointKind.THOUGHT, n_corrob=3, n_contra=1)
    c = Point(id="C", content="z", embedding_orig=(0.0,),
              kind=PointKind.ACTION, n_corrob=3, n_contra=1)
    pop = [a, b, c]
    s_a = assign_status(a, pop)
    s_b = assign_status(b, pop)
    s_c = assign_status(c, pop)
    assert s_a == s_b == s_c


# ---------- K3 — detect_collisions is intra-kind only ----------------------


def test_detect_collisions_intra_kind_only():
    enc = _BoWEncoder()
    pts = [
        Point(id="F1", content="alpha beta common",
              embedding_orig=enc.encode("alpha beta common"),
              kind=PointKind.FACT),
        Point(id="F2", content="gamma delta common",
              embedding_orig=enc.encode("gamma delta common"),
              kind=PointKind.FACT),
        # A THOUGHT geometrically positioned close to F1/F2
        Point(id="T1", content="alpha beta common",
              embedding_orig=enc.encode("alpha beta common"),
              kind=PointKind.THOUGHT),
        Point(id="X1", content="filler one",
              embedding_orig=enc.encode("filler one"),
              kind=PointKind.FACT),
        Point(id="X2", content="filler two",
              embedding_orig=enc.encode("filler two"),
              kind=PointKind.FACT),
    ]
    collisions = detect_collisions(pts, 0.0)
    # Every returned pair must share kind
    for a, b, _ in collisions:
        assert a.kind == b.kind
    # F1↔T1 had identical content but different kinds → must NOT be in
    # the collision set despite their proximity.
    pairs_seen = {(a.id, b.id) for a, b, _ in collisions} | {
        (b.id, a.id) for a, b, _ in collisions
    }
    assert ("F1", "T1") not in pairs_seen


# ---------- K4 — resolve_collision rejects mixed kinds -------------------


def test_resolve_collision_rejects_mixed_kinds():
    enc = _BoWEncoder()
    llm = _SetOpsLLM()
    f = Point(id="F", content="alpha beta", embedding_orig=enc.encode("alpha beta"),
              kind=PointKind.FACT)
    t = Point(id="T", content="alpha beta", embedding_orig=enc.encode("alpha beta"),
              kind=PointKind.THOUGHT)
    with pytest.raises(ValueError):
        resolve_collision([f, t], llm, enc, t_now=1.0)


# ---------- K5 — child inherits parents' kind ----------------------------


def test_child_inherits_kind_from_parents():
    enc = _BoWEncoder()
    llm = _SetOpsLLM()
    t1 = Point(id="T1", content="if a then b extra1",
               embedding_orig=enc.encode("if a then b extra1"),
               kind=PointKind.THOUGHT)
    t2 = Point(id="T2", content="if a then b extra2",
               embedding_orig=enc.encode("if a then b extra2"),
               kind=PointKind.THOUGHT)
    result = resolve_collision([t1, t2], llm, enc, t_now=1.0)
    assert result is not None
    assert result.child.kind == PointKind.THOUGHT


# ---------- K6 — sleep cycle respects kind boundaries ---------------------


def test_sleep_cycle_never_merges_cross_kind():
    enc = _BoWEncoder()
    llm = _SetOpsLLM()
    # A FACT and an ACTION that share much content (geometrically close)
    points = [
        Point(id="F1", content="alpha beta common gamma",
              embedding_orig=enc.encode("alpha beta common gamma"),
              kind=PointKind.FACT),
        Point(id="A1", content="alpha beta common delta",
              embedding_orig=enc.encode("alpha beta common delta"),
              kind=PointKind.ACTION),
        # Padding so threshold can be computed
        Point(id="X1", content="completely different",
              embedding_orig=enc.encode("completely different"),
              kind=PointKind.FACT),
        Point(id="X2", content="another unrelated",
              embedding_orig=enc.encode("another unrelated"),
              kind=PointKind.FACT),
    ]
    report = sleep_cycle_collisions(points, llm, enc, t_now=1.0)
    # F1 and A1 must NOT have been merged
    for event in report.resolved:
        assert not (set(event.parent_ids) == {"F1", "A1"})


# ---------- K7 — ACTION is first-class ------------------------------------


def test_action_point_supports_full_lifecycle():
    """An ACTION-kind Point goes through the same lifecycle as FACT/THOUGHT."""
    enc = _BoWEncoder()
    action = Point(
        id="search_paris_weather",
        content="web_search('paris weather today')",
        embedding_orig=enc.encode("web_search paris weather today"),
        kind=PointKind.ACTION,
    )
    # Default state, no special handling
    assert action.state == EpistemicState.CONJECTURE
    # Counters mutable
    action.n_corrob = 5
    action.n_contra = 1
    pop = [action]
    s = assign_status(action, pop)
    assert s in {EpistemicState.CORROBORATED, EpistemicState.WARRANTED}
