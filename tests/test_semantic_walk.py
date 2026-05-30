"""
Tests for the additive semantic perception layer (PR n°1):
- metacog.anchors.ConceptAnchors  (4 fixed anchors + cosine scorer)
- metacog.meta_walk.walk_semantic (kind-agnostic top-K with scores)

The layer is PERCEPTION only — no routing, no behavior change to
MetaWalker or any existing caller.
"""

from __future__ import annotations

from metacog.anchors import ANCHOR_PHRASES, ConceptAnchors
from metacog.defaults import SimpleEncoder
from metacog.epistemic import Point, PointKind, SourceClass
from metacog.memory import Memory
from metacog.meta_walk import MetaWalker, walk_semantic


# ---------------------------------------------------------------------------
# ConceptAnchors
# ---------------------------------------------------------------------------


def _pt(content: str, kind=PointKind.FACT, id="p1"):
    enc = SimpleEncoder()
    return Point(id=id, content=content,
                 embedding_orig=tuple(enc.encode(content)), kind=kind)


def test_anchors_score_returns_four_dimensions():
    enc = SimpleEncoder()
    anchors = ConceptAnchors(enc)
    p = _pt("some content")
    scores = anchors.score(p, 0.0)
    assert set(scores.keys()) == set(ANCHOR_PHRASES.keys())
    assert set(scores.keys()) == {"factual", "reasoning", "executable", "topical"}
    for v in scores.values():
        assert -1.0 <= v <= 1.0


def test_anchors_source_class_is_computation():
    # Cor.5 audit : anchor scores are COMPUTATION (never enter O).
    assert ConceptAnchors.source is SourceClass.COMPUTATION


def test_anchors_distinguishes_archetypes():
    # Lexical overlap test : each archetype text shares vocabulary with
    # ONE anchor phrase (factual / reasoning / executable). The dominant
    # score should match the expected anchor under the bag-of-words
    # SimpleEncoder.
    enc = SimpleEncoder()
    anchors = ConceptAnchors(enc)
    cases = [
        ("I observed evidence of a witness recall fact", "factual"),
        ("derivation inference reasoning step thought", "reasoning"),
        ("procedure runnable function code action tool", "executable"),
    ]
    for content, expected in cases:
        p = _pt(content)
        assert anchors.dominant(p, 0.0) == expected, (content, anchors.score(p, 0.0))


def test_anchors_cached_no_re_encode_on_repeat_score():
    # Anchors encode once at __init__ ; repeated score() calls must not
    # invoke encoder.encode again.
    class _Counting:
        def __init__(self, inner): self.inner = inner; self.n = 0
        def encode(self, s):
            self.n += 1
            return self.inner.encode(s)
    inner = SimpleEncoder()
    counter = _Counting(inner)
    anchors = ConceptAnchors(counter)
    assert counter.n == len(ANCHOR_PHRASES)        # 4 encodes at init
    p = _pt("anything", id="p2")
    anchors.score(p, 0.0)
    anchors.score(p, 0.0)
    assert counter.n == len(ANCHOR_PHRASES)        # still 4, no re-encode


# ---------------------------------------------------------------------------
# walk_semantic
# ---------------------------------------------------------------------------


def test_walk_semantic_returns_top_k_with_scores():
    m = Memory(encoder=SimpleEncoder())
    for i in range(5):
        m.ingest(f"topic alpha number {i}", kind="FACT", id=f"f{i}")
    out = walk_semantic("alpha", m, k=3)
    assert len(out) == 3
    for r in out:
        assert set(r.keys()) >= {"id", "kind", "tags", "content",
                                  "score", "behavior_scores"}
        assert set(r["behavior_scores"].keys()) == set(ANCHOR_PHRASES.keys())


def test_walk_semantic_is_kind_agnostic():
    m = Memory(encoder=SimpleEncoder())
    m.ingest("witness saw a red car", kind="FACT", id="F1")
    m.ingest("if A then B therefore C reasoning", kind="THOUGHT", id="T1")
    m.ingest("send email to recipient", kind="ACTION", id="A1")
    out = walk_semantic("event happened", m, k=5)
    kinds_seen = {r["kind"] for r in out}
    # At least 2 of the 3 kinds should be reachable without restrict_kind.
    assert len(kinds_seen) >= 2


def test_walk_semantic_empty_memory_returns_empty():
    m = Memory(encoder=SimpleEncoder())
    assert walk_semantic("anything", m, k=5) == []


def test_walk_semantic_no_regression_to_metawalker():
    # MetaWalker still works and can coexist in parallel.
    m = Memory(encoder=SimpleEncoder())
    m.ingest("just one fact", kind="FACT", id="F1")
    out = walk_semantic("fact", m, k=3)
    assert isinstance(out, list)
    w = MetaWalker("fact", m, commit=False)
    stage = w.step()
    assert stage is not None
