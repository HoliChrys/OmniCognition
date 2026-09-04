"""
OKF wiki layer — the bidirectional, continuously-evolving RAG extension.

A wiki doc keeps its RAG node refs in the OKF frontmatter AND inline (`[[id]]`)
AND the DB link table; tags live in frontmatter + inline. RAG->wiki:
`reconcile_wiki` rewrites refs when a node is merged/forgotten. wiki->RAG:
`ingest_from_wiki` ingests new prose as a node carrying the doc's tags.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.journal import Journal
from metacog.memory import Memory
from metacog import wiki as W


# -- pure helpers -------------------------------------------------------------

def test_render_and_parse_roundtrip():
    md = W.render_okf(type="topic", title="T", tags=["a:b"], refs=["N1", "N2"],
                      body="- fact one [[N1]]\n- fact two [[N2]]", timestamp=1.0)
    d = W.parse_okf(md)
    assert d["type"] == "topic" and d["title"] == "T"
    assert d["refs"] == ["N1", "N2"] and d["tags"] == ["a:b"]
    assert W.body_refs(d["body"]) == ["N1", "N2"]


def test_rewrite_body_refs_and_context_tags():
    assert W.rewrite_body_refs("x [[A]] y [[B]]", {"A": "C"}) == "x [[C]] y [[B]]"
    assert W.context_tags(["fact", "health:x", "refined"]) == ["health:x"]


def test_body_tags_parse_paths_and_ignore_headings_and_periods():
    body = ("# Heading is not a tag\n"
            "Tags: #module:metacog #file:metacog/memory.py #file:.gitignore #a\n"
            "ends with #health:fatigue. and #x-y_z, then #Mixed:Case ; PR #10 is not")
    assert W.body_tags(body) == [
        "module:metacog", "file:metacog/memory.py", "file:.gitignore", "a",
        "health:fatigue", "x-y_z", "mixed:case"]


# -- feed + bidirectional links -----------------------------------------------

def _mem():
    m = Memory(encoder=SimpleEncoder(), journal=Journal())
    a = m.ingest("swollen finger", kind="FACT", id="A")
    a.tags.append("health:macrodactyly")
    b = m.ingest("chronic tiredness", kind="FACT", id="B")
    b.tags.append("health:fatigue")
    return m


def test_feed_wiki_writes_refs_frontmatter_body_and_links():
    m = _mem()
    out = m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic")
    assert set(out["refs"]) == {"A", "B"}
    md = m.wiki_doc("doc:h")
    assert "[[A]]" in md and "[[B]]" in md            # inline refs
    assert "refs:" in md and "#health" in md          # frontmatter + inline tags
    assert set(m.docs_for_node("A")) == {"doc:h"}      # reverse link (RAG->wiki)


def test_reconcile_rewrites_refs_when_node_merged():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic")
    m.forget_node("A", reason="superseded by B", superseded_by="B")
    m.merge_forgotten()
    rec = m.reconcile_wiki()
    assert rec["remapped"] == 1
    md = m.wiki_doc("doc:h")
    assert "[[A]]" not in md and "[[B]]" in md          # ref auto-updated
    assert m.docs_for_node("A") == ["doc:h"]            # A resolves to B's doc


def test_reconcile_flags_stale_when_node_invalidated_without_successor():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic")
    m.forget_node("A", reason="user corrected")         # no successor
    rec = m.reconcile_wiki()
    assert rec["stale"] >= 1
    stale = {r["node_id"]: r["stale"]
             for r in m.journal.wiki_refs_for_doc("doc:h")}
    assert stale.get("A") is True and stale.get("B") is False


def test_ingest_from_wiki_feeds_rag_with_doc_context():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic")
    out = m.ingest_from_wiki("doc:h", "new finding: fingers swell after exertion")
    nid = out["node_id"]
    node = next(p for p in m.points if p.id == nid)
    # the new node carries the doc's tags as context
    assert "health:macrodactyly" in node.tags and "health:fatigue" in node.tags
    # and is linked back into the doc (ref + inline)
    assert nid in [r["node_id"] for r in m.journal.wiki_refs_for_doc("doc:h")]
    assert f"[[{nid}]]" in m.wiki_doc("doc:h")


# -- OKF EAV index : functional queries, schema recovery, no migrations -------

def test_eav_index_makes_frontmatter_queryable():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic")
    # query by ANY frontmatter field — the "functional" part OKF lacks natively
    assert m.wiki_where("type", "topic") == ["doc:h"]
    assert m.wiki_where("tags", "health:fatigue") == ["doc:h"]
    assert m.wiki_where("refs", "A") == ["doc:h"]
    assert m.wiki_where("title", "Health") == ["doc:h"]
    assert m.wiki_where("refs", "ZZZ") == []            # absent value


def test_schema_is_recovered_from_data_no_registry():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic")
    m.feed_wiki("doc:x", "Other", ["A"], type="metric")
    schema = m.okf_schema()
    assert set(schema["topic"]) >= {"type", "title", "tags", "refs", "timestamp"}
    assert "metric" in schema                            # per-type keys, derived


def test_eav_reindexes_on_merge():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic")
    assert set(m.wiki_where("refs")) == {"doc:h"}        # has refs
    m.forget_node("A", reason="superseded by B", superseded_by="B")
    m.merge_forgotten()
    m.reconcile_wiki()
    assert m.wiki_where("refs", "A") == []               # old ref gone from index
    assert m.wiki_where("refs", "B") == ["doc:h"]        # points to successor


def test_import_external_okf_doc():
    m = Memory(encoder=SimpleEncoder(), journal=Journal())
    m.ingest("some fact", kind="FACT", id="N1")
    md = ("---\ntype: runbook\ntitle: Deploy\ntags:\n- ops:deploy\n"
          "refs:\n- N1\n---\n\n- step one [[N1]]\n")
    out = m.import_okf("doc:deploy", md)
    assert out["type"] == "runbook" and "N1" in out["refs"]
    assert m.wiki_where("type", "runbook") == ["doc:deploy"]
    assert m.wiki_where("tags", "ops:deploy") == ["doc:deploy"]
    assert "runbook" in m.okf_schema()


def test_feedback_is_first_order_in_the_wiki():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic")
    # a retrieval returning A is marked USEFUL -> credibility flows to the doc
    rid = m.record_retrieval(["A", "B"], query_text="q")
    m.mark_useful(rid, 2)
    # queryable via the EAV index (first-order field)
    assert m.wiki_where("useful", "2") == ["doc:h"]
    # and visible in the rendered OKF frontmatter
    assert "useful: 2" in m.wiki_doc("doc:h")
    # a useless retrieval registers on the other side
    r2 = m.record_retrieval(["A"], query_text="q2")
    m.mark_useful(r2, 0)
    assert "useless: 1" in m.wiki_doc("doc:h")


def test_emergent_tool_is_auto_registered_in_the_wiki():
    """The moment an emergent tool is created it becomes a wiki concept."""
    class _LLM:  # no generate -> deterministic fallback synthesis
        pass
    m = Memory(encoder=SimpleEncoder(), journal=Journal(), llm=_LLM())
    r = m.ensure_tool("reverse a string end to end", how="iterate backwards")
    tid = r["tool"]["id"]
    doc = f"tool:{tid}"
    # a type=tool OKF doc now exists, referencing the tool node
    assert m.wiki_where("type", "tool") == [doc]
    assert m.wiki_where("refs", tid) == [doc]
    assert f"[[{tid}]]" in m.wiki_doc(doc)


def test_crystallized_tool_is_auto_registered():
    from metacog.epistemic import Point, PointKind
    class _LLM:
        pass
    m = Memory(encoder=SimpleEncoder(), journal=Journal(), llm=_LLM())
    m.skills_enabled = True
    enc = SimpleEncoder()
    pop = {"sort a list ascending": 12, "count the words": 2,
           "parse a date": 2, "trim whitespace": 2}
    for content, reps in pop.items():
        for i in range(reps):
            a = Point(id=f"{content[:5]}{i}", content=content,
                      embedding_orig=tuple(enc.encode(content)),
                      kind=PointKind.ACTION, keywords=content.split())
            m.record_action_generation(a, None, f"{content} — approach {i}", [])
    out = m.crystallize_skills()
    assert out["crystallized"] >= 1
    tid = out["tool_ids"][0]
    assert m.wiki_where("refs", tid) == [f"tool:{tid}"]   # auto-registered


def test_wiki_noop_without_journal():
    m = Memory(encoder=SimpleEncoder())
    assert m.feed_wiki("d", "t", ["A"])["doc_id"] is None
    assert m.wiki_doc("d") is None
    assert m.reconcile_wiki() == {"remapped": 0, "stale": 0, "reasons": {}}
    assert m.ingest_from_wiki("d", "x")["node_id"] is None
    assert m.wiki_where("type", "topic") == [] and m.okf_schema() == {}
    assert m.import_okf("d", "---\ntype: x\n---\n")["doc_id"] is None
