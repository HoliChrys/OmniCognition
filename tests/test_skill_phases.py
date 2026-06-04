"""Skill system phases 2-4 :
  - tools-in-the-walk helpers (task-mode depth gate, skill-JSON synthesis),
  - the double-query section filter,
  - the latent distiller replayed in sleep().

All deterministic (fake LLM), no live API.
"""
import json

from metacog.memory import Memory
from metacog.epistemic import PointKind, SourceClass
from metacog.meta_walk import (
    is_tool_point, enough_tools_for_workflow, synthesize_skill_json,
)


class _ToolLLM:
    """Says YES to workflow-sufficiency and emits a valid skill JSON."""
    def generate(self, prompt, max_tokens=None):
        if "sufficient to build the whole workflow" in prompt:
            return "YES"
        if "SKILL FOLDER" in prompt:
            return json.dumps({
                "name": "solver",
                "tree": {"retrieve.py": "fetch", "rank.py": "rank"},
            })
        return ""


class _NoLLM:
    pass


# ---------------------------------------------------------------------------
# Phase 2 — tools in the walk
# ---------------------------------------------------------------------------

def test_is_tool_point():
    m = Memory()
    pts = m.ingest_skill({"a.py": "x"}, name="n", user_id="u", session_id="s")
    assert all(is_tool_point(p) for p in pts)        # skill-tagged
    plain = m.ingest("just a fact", kind="FACT")
    assert not is_tool_point(plain)


def test_enough_tools_fails_closed_without_tools_or_llm():
    m = Memory()
    tool = m.ingest_skill({"a.py": "x"}, name="n", user_id="u", session_id="s")[0]
    assert enough_tools_for_workflow("task", [], _ToolLLM()) is False   # no tools
    assert enough_tools_for_workflow("task", [tool], _NoLLM()) is False  # no llm
    assert enough_tools_for_workflow("task", [tool], _ToolLLM()) is True


def test_synthesize_skill_json_shape_and_fallback():
    m = Memory()
    tools = m.ingest_skill({"a.py": "x"}, name="n", user_id="u", session_id="s")
    good = synthesize_skill_json("solve X", tools, _ToolLLM())
    assert isinstance(good["name"], str) and isinstance(good["tree"], dict)
    # fallback: no LLM still yields an ingestable {name, tree}
    fb = synthesize_skill_json("solve X please", tools, _NoLLM())
    assert fb["name"] and isinstance(fb["tree"], dict) and fb["tree"]


# ---------------------------------------------------------------------------
# Phase 3 — double-query section filter
# ---------------------------------------------------------------------------

def test_section_filter_boosts_in_section_points():
    from metacog.meta_walk import nearest_facts_with_fallback
    from metacog.defaults import SimpleEncoder
    from metacog.keywords import SimpleKeywordExtractor
    enc, extr = SimpleEncoder(), SimpleKeywordExtractor()
    m = Memory(encoder=enc, extractor=extr)
    # same content in two sessions; the section filter should surface the
    # in-section copy.
    m.ingest_skill({"ranker.py": "rank candidate documents"},
                   name="s1", user_id="u1", session_id="alpha")
    m.ingest_skill({"ranker.py": "rank candidate documents"},
                   name="s2", user_id="u1", session_id="beta")
    q = "rank candidate documents"
    qemb = tuple(enc.encode(q))
    res = nearest_facts_with_fallback(
        qemb, q, m.points, 6, m._now(),
        encoder=enc, extractor=extr,
        section_filter={"session:alpha"},
    )
    # at least one returned point is from the alpha section
    assert any("session:alpha" in p.tags for p in res)


# ---------------------------------------------------------------------------
# Phase 4 — latent distiller
# ---------------------------------------------------------------------------

def test_record_and_distill_creates_linked_tool():
    m = Memory()
    f1 = m.ingest("fetch the docs from the index", kind="FACT")
    f2 = m.ingest("rank them by relevance", kind="ACTION")
    m.record_resolution("how to search docs", [f1.id, f2.id], "doc_search",
                        user_id="u1", session_id="s1")
    out = m.distill_skills()
    assert out["distilled"] == 1
    tool = next(p for p in m.points if p.id == out["tool_ids"][0])
    assert tool.kind == PointKind.ACTION
    assert {"tool", "skill", "distilled"}.issubset(set(tool.tags))
    assert "name:doc_search" in tool.tags
    # linked to the explicating facts/actions (lineage)
    assert set(tool.parents) == {f1.id, f2.id}
    assert tool.keywords_source == SourceClass.GENERATOR


def test_distill_is_idempotent():
    m = Memory()
    f = m.ingest("x", kind="FACT")
    m.record_resolution("q", [f.id], "name1")
    assert m.distill_skills()["distilled"] == 1
    assert m.distill_skills()["distilled"] == 0      # cursor advanced
    m.record_resolution("q2", [f.id], "name2")
    assert m.distill_skills()["distilled"] == 1      # only the new one


def test_distiller_runs_in_sleep_when_enabled():
    m = Memory()
    f = m.ingest("x", kind="FACT")
    m.record_resolution("q", [f.id], "name1")
    m.skills_enabled = False
    rep = m.sleep()
    assert "distilled" not in rep                     # gated off
    m.skills_enabled = True
    rep = m.sleep()
    assert rep.get("distilled") == 1


def test_build_skill_end_to_end_ingests_named_skill():
    m = Memory(llm=_ToolLLM())
    # seed some tool nodes so the task-mode walk has something to gather
    m.ingest_skill({"retrieve.py": "fetch documents by topic"},
                   name="seed", user_id="u1", session_id="s1")
    skill = m.build_skill("assemble a document search workflow",
                          user_id="u1", session_id="s1", n_stages=4)
    assert skill["name"] and isinstance(skill["tree"], dict)
    # the synthesised skill was re-indexed as named points
    assert any(f"name:{skill['name']}" in p.tags for p in m.points)
    # a resolution was recorded for the latent distiller
    assert m._resolution_ledger
