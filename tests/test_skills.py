"""
Tests for MEMORY SKILLS / tool crystallization (metacog.skills).

Invariants :
  1. DIVERSITY : an action re-derived only for near-identical queries does
     not build recurrence weight ; the same action across diverse queries
     does.
  2. EMERGENT threshold : crystallization fires on the Poisson law.
  3. TOOL = NORMAL NODE : a synthesized tool is a kind=ACTION Point tagged
     "tool" + genre + context, scoped by keywords — discoverable by the
     walk and the explicit matcher alike.
  4. FAST-PATH : match_tool returns a covering tool for an in-scope query
     and nothing for an out-of-scope one.
  5. INTENT metacognition : assess_tool_intent flags executable text.
"""

from __future__ import annotations

import math

from metacog.epistemic import Point, PointKind
from metacog.skills import (
    SkillLedger,
    TOOL_TAG,
    assess_tool_intent,
    detect_skill_candidates,
    is_tool,
    match_tool,
    record_action,
    skill_threshold,
    synthesize_tool,
    tools_in,
)


def _orth(dim, axis):
    v = [0.0] * dim
    v[axis % dim] = 1.0
    return tuple(v)


def _action(kws):
    return Point(id="a", content=" ".join(kws), embedding_orig=(1.0, 0.0, 0.0),
                 kind=PointKind.ACTION, keywords=list(kws))


class _Enc:
    """Discriminative hash encoder : distinct words -> distinct directions,
    so in-scope and out-of-scope queries are actually separable."""
    def encode(self, text):
        import hashlib
        v = [0.0] * 8
        for w in (text or "").lower().split():
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % 8] += 1.0
        return tuple(v)


class _LLM:
    def generate(self, prompt, max_tokens=None):
        # Deterministic stub : answers the synthesis + the intent probe.
        if "VERDICT:" in prompt:
            return "VERDICT: yes KIND: command"
        return ("NAME: search_articles\n"
                "WHEN: searching scientific articles\n"
                "HOW: query the article index by topic and rank by relevance")


# ---------------------------------------------------------------------------
# 1) diversity
# ---------------------------------------------------------------------------

def test_near_duplicate_queries_do_not_build_recurrence():
    led = SkillLedger()
    act = _action(["search", "articles", "science"])
    base = _orth(8, 0)
    for _ in range(20):
        record_action(led, act, base, "find papers", ["science"])
    sig = next(iter(led.traces))
    assert led.traces[sig].weight < 1.5      # only the first counted


def test_diverse_queries_build_recurrence():
    led = SkillLedger()
    act = _action(["search", "articles", "science"])
    for axis in range(6):
        record_action(led, act, _orth(8, axis), f"q{axis}", ["science"])
    sig = next(iter(led.traces))
    assert led.traces[sig].weight > 4.0


# ---------------------------------------------------------------------------
# 2) threshold
# ---------------------------------------------------------------------------

def test_threshold_poisson_and_none_when_sparse():
    led = SkillLedger()
    assert skill_threshold(led) is None       # nothing yet
    # one dominant diverse signature + several weak ones
    dom = _action(["search", "articles", "science"])
    for axis in range(6):
        record_action(led, dom, _orth(16, axis), f"q{axis}", ["science"])
    for s in range(5):
        a = _action([f"verb{s}", f"obj{s}", "x"])
        record_action(led, a, _orth(16, s), f"w{s}", ["x"])
    thr = skill_threshold(led)
    assert thr is not None
    lam = led.total_weight / len(led.traces)
    assert math.isclose(thr, lam + math.sqrt(lam), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 3 + 4) crystallize into a tool NODE, then match it
# ---------------------------------------------------------------------------

def test_crystallized_tool_is_a_tagged_action_node_and_matches():
    led = SkillLedger()
    dom = _action(["search", "articles", "science"])
    for axis in range(8):
        record_action(led, dom, _orth(16, axis), f"find papers {axis}",
                      ["science", "research", "papers"])
    # background noise so the threshold is meaningful
    for s in range(6):
        a = _action([f"v{s}", f"o{s}", "z"])
        record_action(led, a, _orth(16, s), f"n{s}", ["z"])

    cands = detect_skill_candidates(led)
    assert cands and cands[0][0] == "articles|science|search"

    enc = _Enc()
    sig, trace = cands[0]
    tool = synthesize_tool(sig, trace, _LLM(), enc, t_now=1.0, genre="command")
    assert tool is not None
    # a tool is a NORMAL node : kind ACTION, tagged tool + genre + context
    assert tool.kind == PointKind.ACTION
    assert is_tool(tool) and TOOL_TAG in tool.tags
    assert any(t.startswith("tool_") for t in tool.tags)   # genre tag
    assert "science" in tool.tags                          # context tag
    assert tool.keywords and tool.keywords_embedding is not None

    # fast-path match : an in-scope query finds it, out-of-scope does not
    from metacog.keywords import position_weighted_keyword_embedding
    in_q = position_weighted_keyword_embedding(["science", "research", "papers"], enc)
    out_q = position_weighted_keyword_embedding(["cooking", "recipe", "dinner"], enc)
    pts = [tool]
    assert match_tool(in_q, pts) is not None
    hit = match_tool(in_q, pts)
    assert hit[0].id == tool.id
    # out-of-scope : either no match or a strictly lower score than in-scope
    out_hit = match_tool(out_q, pts)
    assert out_hit is None or out_hit[1] < hit[1]
    assert tools_in(pts) == [tool]


# ---------------------------------------------------------------------------
# 5) explicit tool-intent metacognition
# ---------------------------------------------------------------------------

def test_assess_tool_intent_flags_executable_text():
    # surface-marker mode (no LLM)
    yes = assess_tool_intent("run a SQL query to fetch the rows", use_llm=False)
    assert yes.is_executable and yes.kind in ("command", "code", "api")
    no = assess_tool_intent("Caroline felt happy about the news", use_llm=False)
    assert not no.is_executable and no.kind == "none"


def test_assess_tool_intent_uses_llm_for_ambiguous():
    v = assess_tool_intent("search the article database for the topic", llm=_LLM())
    assert v.is_executable and v.kind == "command"


# ---------------------------------------------------------------------------
# 6) eager 'no tool -> generate it -> proceed' step
# ---------------------------------------------------------------------------

def test_ensure_tool_generates_then_reuses():
    from metacog.memory import Memory

    class _SimpleEnc:
        def encode(self, text):
            import hashlib
            v = [0.0] * 8
            for w in (text or "").lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 8] += 1.0
            return tuple(v)

    mem = Memory(encoder=_SimpleEnc(), llm=_LLM())
    mem.skills_enabled = True

    # First need : no covering tool -> GENERATE.
    r1 = mem.ensure_tool("search scientific articles about science",
                         how="query the index by topic")
    assert r1["reused"] is False and r1["tool"] is not None
    tid = r1["tool"]["id"]
    assert any(p.id == tid and "tool" in p.tags for p in mem.points)
    n_after_gen = len(mem.points)

    # Same need again : a covering tool now exists -> REUSE, no new node.
    r2 = mem.ensure_tool("search scientific articles about science")
    assert r2["reused"] is True
    assert r2["tool"]["id"] == tid
    assert len(mem.points) == n_after_gen        # nothing added on reuse
