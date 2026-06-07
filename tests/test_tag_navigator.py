"""
Tests for the LLM-driven hierarchical tag-scope navigator.

Deterministic : a scripted fake LLM that, at each level, keeps the branch
matching a target path. Verifies the descent leverages the hierarchy (prunes
siblings level by level down to the specific leaf) and the keep-most-specific
cross-cleanup.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.epistemic import Point, PointKind
from metacog.tag_navigator import TagNavigator


def _p(pid, tags):
    return Point(id=pid, content=pid, embedding_orig=SimpleEncoder().encode(pid),
                 kind=PointKind.FACT, keywords=[], tags=tags)


class _Mem:
    """Minimal memory shim : the navigator only needs points + encoder."""
    def __init__(self, points):
        self.points = points
        self.encoder = SimpleEncoder()


def _points():
    return _Mem([
        _p("D5", ["location:geography:stamford", "animal:canine:puppy"]),
        _p("D6", ["location:city:hartford"]),
        _p("D7", ["location:venue:stadium"]),
        _p("D8", ["activity:gaming:online"]),
    ])


class _PathLLM:
    """Drives the navigator straight down to `target`: at each level, DESCEND
    the child on the path to target, or KEEP target once reached. Reads the
    lookahead prompt (options are 2-space-indented; sub-branch lines deeper)."""
    def __init__(self, target):
        self.target = target
        self.calls = 0

    def generate(self, prompt, max_tokens=260):
        self.calls += 1
        opts = [l.strip() for l in prompt.splitlines()
                if l.startswith("  ") and not l.strip().startswith("sub-branches")
                and ":" in l or (l.startswith("  ") and l.strip()
                                 and not l.strip().startswith("sub-branches"))]
        opts = [o for o in opts if o and not o.startswith("sub-branches")]
        out = []
        for o in opts:
            if o == self.target:
                out.append(f"KEEP {o}")
            elif self.target.startswith(o + ":"):
                segs = self.target.split(":")
                child = ":".join(segs[:o.count(":") + 2])  # one level deeper
                out.append(f"DESCEND {child}")
        return "\n".join(out)


def test_descent_reaches_specific_leaf_pruning_siblings():
    nav = TagNavigator(_points(), index=None)
    llm = _PathLLM("location:geography:stamford")
    sel = nav.resolve("where does James live?", llm, max_depth=4)
    # leveraged the hierarchy down to the specific leaf, pruned sibling cities
    assert sel == ["location:geography:stamford"]
    assert "location:city:hartford" not in sel
    assert "location" not in sel              # the catch-all is never the answer


def test_llm_decide_lookahead_keep_and_descend_child():
    """With the one-level lookahead, KEEP stops at a branch (its subtree is the
    scope) and DESCEND names a specific CHILD (n+1) to visit next."""
    nav = TagNavigator(_points(), index=None)

    class _LLM:
        def generate(self, prompt, max_tokens=260):
            # keep location:geography broad; descend activity into its child
            return "KEEP location:geography\nDESCEND activity:gaming"
    keep, nxt = nav._llm_decide(
        "q", ["location:geography", "activity", "location:city"], _LLM())
    assert keep == ["location:geography"]      # stop here -> ns:* subtree
    assert nxt == ["activity:gaming"]          # the n+1 child to visit next
    assert "location:city" not in keep + nxt   # omitted -> dropped


def test_immediate_children_one_level():
    nav = TagNavigator(_points(), index=None)
    kids = nav.immediate_children("location")
    assert set(kids) == {"location:geography", "location:city", "location:venue"}
    assert "location:geography:stamford" not in kids   # that's two levels down


def test_keep_most_specific_collapses_ancestors():
    f = TagNavigator._keep_most_specific
    got = f(["location", "location:geography", "location:geography:stamford",
             "activity:gaming:online"])
    assert set(got) == {"location:geography:stamford", "activity:gaming:online"}


def test_fallback_without_llm_is_semantic_union():
    nav = TagNavigator(_points(), index=None)
    # no .generate -> fallback to the semantic union (never raises)
    sel = nav.resolve("anything", object(), max_depth=3)
    assert isinstance(sel, list)


def test_tools_match_over_glossary():
    nav = TagNavigator(_points(), index=None)
    assert "location:geography:stamford" in nav.exact("stamford")
    assert any("hartford" in ns for ns in nav.regex("hart"))
    assert nav.glossary_under("location:geography") == \
        ["location:geography", "location:geography:stamford"]
