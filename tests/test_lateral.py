"""
Tests for LATERAL collision (metacog.lateral) — co-retrieval-driven
consolidation.

Covers the four design invariants :
  1. DIVERSITY discount : a pair surfaced only by near-identical queries
     does NOT accumulate enough weight to fire ; the SAME pair surfaced
     by diverse (orthogonal) queries does.
  2. EMERGENT threshold : firing uses the Poisson lambda+sqrt(lambda)
     law, no hyperparameter.
  3. GATE : detection is a no-op on a small / tag-poor cloud.
  4. RESOLUTION : a fired group folds into one keeper that absorbs the
     others' keywords + counters, and absorbed ids alias to the keeper.
"""

from __future__ import annotations

import math

from metacog.epistemic import Point, PointKind
from metacog.lateral import (
    LateralLedger,
    detect_lateral_groups,
    gate_ready,
    lateral_collapse,
    lateral_threshold,
    record_coretrieval,
)


def _orth(dim: int, axis: int) -> tuple:
    v = [0.0] * dim
    v[axis % dim] = 1.0
    return tuple(v)


def _pt(id_: str, *, kind=PointKind.FACT, n_uses=0, keywords=None, tags=None) -> Point:
    return Point(
        id=id_, content=id_, embedding_orig=(1.0, 0.0, 0.0),
        kind=kind, n_uses=n_uses,
        keywords=list(keywords or []), tags=list(tags or []),
    )


def _tag_rich_cloud(n=80):
    """A cloud that passes the gate : many nodes, rich tags, intersections."""
    pts = []
    for i in range(n):
        pts.append(_pt(f"P{i}", tags=[f"t{i % 10}", f"u{i % 7}"]))
    return pts


# ---------------------------------------------------------------------------
# 1) Diversity discount
# ---------------------------------------------------------------------------


def test_near_duplicate_queries_do_not_inflate_weight():
    """Same pair (A,B) co-retrieved by 20 near-identical queries should
    barely accumulate weight — redundant probing is discounted."""
    led = LateralLedger()
    base = _orth(8, 0)
    for _ in range(20):
        # all queries ~identical to `base`
        record_coretrieval(led, ["A", "B", "C"], base)
    key = ("A", "B")
    # First event counts (~1.0) ; the other 19 add ~0 (novelty < min).
    assert led.weight[key] < 1.5


def test_diverse_queries_accumulate_weight():
    """The SAME pair surfaced by orthogonal queries accumulates full
    weight per novel direction."""
    led = LateralLedger()
    for axis in range(6):
        record_coretrieval(led, ["A", "B", "C"], _orth(8, axis))
    key = ("A", "B")
    # six orthogonal directions => weight well above the near-dup case.
    assert led.weight[key] > 4.0


# ---------------------------------------------------------------------------
# 2) Emergent threshold
# ---------------------------------------------------------------------------


def test_threshold_is_poisson():
    led = LateralLedger()
    # Make one dominant pair (diverse) and a spread of weak pairs.
    for axis in range(6):
        record_coretrieval(led, ["A", "B"], _orth(8, axis))
    for axis in range(6):
        record_coretrieval(led, [f"X{axis}", f"Y{axis}"], _orth(8, axis))
    thr = lateral_threshold(led)
    assert thr is not None
    lam = led.total_weight / led.pairs_seen()
    assert math.isclose(thr, lam + math.sqrt(lam), rel_tol=1e-9)
    # The dominant diverse pair clears it ; a lone weak pair does not.
    assert led.weight[("A", "B")] > thr


def test_threshold_none_when_too_few_pairs():
    led = LateralLedger()
    record_coretrieval(led, ["A", "B"], _orth(8, 0))
    assert lateral_threshold(led) is None


# ---------------------------------------------------------------------------
# 3) Gate
# ---------------------------------------------------------------------------


def test_gate_blocks_small_cloud():
    led = LateralLedger()
    for axis in range(6):
        record_coretrieval(led, ["A", "B"], _orth(8, axis))
    small = [_pt("A", tags=["x", "y"]), _pt("B", tags=["x", "y"])]
    # Even a strongly-firing pair yields no groups on a tiny cloud.
    assert detect_lateral_groups(led, small) == []
    assert gate_ready(small) is False


def test_gate_opens_on_big_tag_rich_cloud():
    assert gate_ready(_tag_rich_cloud(80)) is True


# ---------------------------------------------------------------------------
# 4) Detection + resolution end-to-end
# ---------------------------------------------------------------------------


def test_content_floor_rejects_cross_topic_pair():
    """A pair that co-retrieves strongly under diverse queries but whose
    CONTENT embeddings are dissimilar (different topics) must NOT collide —
    the safety floor protects distinct information."""
    pts = _tag_rich_cloud(80)
    a = next(p for p in pts if p.id == "P0")
    b = next(p for p in pts if p.id == "P1")
    a.embedding_orig = (1.0, 0.0, 0.0)      # orthogonal contents
    b.embedding_orig = (0.0, 1.0, 0.0)      # cosine 0 << 0.72 floor
    led = LateralLedger()
    for axis in range(8):
        record_coretrieval(led, ["P0", "P1"], _orth(16, axis))
    for axis in range(8):
        record_coretrieval(led, [f"P{20+axis}", f"P{40+axis}"], _orth(16, axis))
    # Strong co-retrieval, but content floor blocks the merge.
    assert detect_lateral_groups(led, pts) == []


def test_diverse_coretrieval_collapses_into_keeper():
    pts = _tag_rich_cloud(80)
    # Make P0 and P1 the laterally-redundant pair, retrieved together by
    # many orthogonal queries. Give them distinct keywords + different
    # n_uses so the keeper is deterministic.
    p0 = next(p for p in pts if p.id == "P0")
    p1 = next(p for p in pts if p.id == "P1")
    p0.keywords = ["alpha"]
    p0.n_uses = 10
    p1.keywords = ["beta"]
    p1.n_uses = 3

    led = LateralLedger()
    for axis in range(8):
        record_coretrieval(led, ["P0", "P1"], _orth(16, axis))
    # plus background noise so the threshold is meaningful
    for axis in range(8):
        record_coretrieval(led, [f"P{20+axis}", f"P{40+axis}"], _orth(16, axis))

    groups = detect_lateral_groups(led, pts)
    assert ["P0", "P1"] in groups

    class _Enc:
        def encode(self, text):
            return (float(len(text)), 0.0, 0.0)

    n_before = len(pts)
    report = lateral_collapse(pts, led, _Enc(), t_now=1.0)
    assert len(report.collided) == 1
    # P0 (more uses) is the keeper ; P1 is absorbed and aliased.
    assert report.aliases.get("P1") == "P0"
    keeper = next(p for p in pts if p.id == "P0")
    assert "beta" in keeper.keywords and "alpha" in keeper.keywords  # union
    assert keeper.n_uses == 13                                       # 10 + 3
    # NON-DESTRUCTIVE : P1 is retained (content preserved) but tagged out
    # of the search pool and adopted as a child of the keeper.
    p1 = next((p for p in pts if p.id == "P1"), None)
    assert p1 is not None and "lateral_absorbed" in p1.tags
    assert "P1" in keeper.children
    assert len(pts) == n_before                                     # nothing removed
