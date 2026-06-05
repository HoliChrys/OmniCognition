"""Tag-filtered retrieval as a two-phase cascade :
  Phase 1 — walk HARD-restricted to the filtered set (a discussion);
  Phase 2 — optional global knowledge-base walk seeded by Phase 1.
Plus the MetaWalker restrict_ids hard pool restriction. Deterministic.
"""
from metacog.memory import Memory
from metacog.defaults import SimpleEncoder
from metacog.keywords import SimpleKeywordExtractor
from metacog.meta_walk import MetaWalker


class _LLM:
    def generate(self, prompt, max_tokens=None):
        if "PROVISIONAL answer" in prompt:
            return "scoped-or-global-answer"
        return ""


def _mem():
    m = Memory(encoder=SimpleEncoder(), extractor=SimpleKeywordExtractor(),
               llm=_LLM())
    # discussion s1 : two messages about "raptor migration"
    m.ingest_message("the raptor migration data looks odd in march",
                     role="user", user_id="u", session_id="s1",
                     timestamp="2026-06-04T10:00:00", block=True)
    m.ingest_message("yes the march raptor counts spiked unexpectedly",
                     role="agent", user_id="u", session_id="s1",
                     timestamp="2026-06-04T10:00:05", block=True)
    # global knowledge (NOT in s1) : a fact about raptors
    m.ingest("raptors migrate along thermal corridors in spring", kind="FACT")
    return m


def test_restrict_ids_hard_limits_the_fact_pool():
    m = _mem()
    s1_ids = {p.id for p in m.points if "session:s1" in p.tags}
    w = MetaWalker("raptor", m, n_stages=4, commit=False, restrict_ids=s1_ids)
    pool = w._all_points()
    facts = [p for p in pool if p.kind.name == "FACT"
             and not p.id.startswith("entity_")]
    # every searchable FACT is from the discussion ; the global raptor fact
    # (not in s1) is excluded.
    assert facts, "no scoped facts"
    assert all("session:s1" in p.tags for p in facts)


def test_scoped_only_stays_in_the_filtered_set():
    m = _mem()
    out = m.scoped_answer("what about the raptors?", tags=["session:s1"],
                          knowledge_base=False, n_stages=4)
    assert "scoped_answer" in out and out["knowledge_base"] is False
    assert "global_answer" not in out
    # scoped evidence ids are all from s1
    s1_ids = {p.id for p in m.points if "session:s1" in p.tags}
    for e in out["scoped_evidence"]:
        assert e["id"] in s1_ids


def test_knowledge_base_runs_second_global_phase():
    m = _mem()
    out = m.scoped_answer("what about the raptors?", tags=["session:s1"],
                          knowledge_base=True, n_stages=4)
    assert out["knowledge_base"] is True
    assert "global_answer" in out and "global_evidence" in out


def test_mcp_scoped_answer_registered():
    import asyncio
    from metacog.mcp_server import build_app
    names = [t.name for t in asyncio.run(build_app(memory=Memory()).list_tools())]
    assert "scoped_answer" in names


def test_scoped_answer_match_exact_hierarchical_ancestor():
    """Filtering on a parent prefix `session` matches every `session:sX`."""
    m = _mem()
    # Add another session so the ancestor filter has something to bind to.
    m.ingest_message("orthogonal note about hawks",
                     role="user", user_id="u", session_id="s9",
                     timestamp="2026-06-04T11:00:00", block=True)
    # `session` is the parent prefix of both `session:s1` and `session:s9` —
    # exact-mode now resolves hierarchically.
    out = m.scoped_answer("anything", tags=["session"],
                          knowledge_base=False, n_stages=2, match="exact")
    ids = {e["id"] for e in out["scoped_evidence"]}
    s_any_ids = {p.id for p in m.points
                 if any(t.startswith("session:") for t in p.tags)}
    # The global raptor FACT carries no `session:*` tag — must stay out.
    global_fact_ids = {p.id for p in m.points
                       if not any(t.startswith("session:") for t in p.tags)
                       and p.kind.name == "FACT"
                       and not p.id.startswith("entity_")}
    assert ids <= s_any_ids
    assert ids.isdisjoint(global_fact_ids)


def test_scoped_answer_match_regex_selects_one_session():
    m = _mem()
    out = m.scoped_answer("anything", tags=[r"^session:s1$"],
                          knowledge_base=False, n_stages=2, match="regex")
    s1_ids = {p.id for p in m.points if "session:s1" in p.tags}
    for e in out["scoped_evidence"]:
        assert e["id"] in s1_ids
