"""
Wiki DRIFT : a doc cites the nodes it was generated from ; when those nodes
change (content / knowledge tags), the doc must follow.

  - every link stores the node's fingerprint at link time ;
  - `sleep` (reconcile_wiki) detects drift per ref with a reason code ;
  - a GENERATED doc (body = rendering of its refs) regenerates itself ;
  - an AUTHORED doc (prose) is NEVER overwritten : its refs are flagged
    `outdated`, the doc is queryable (`wiki_where('outdated')`), check_wiki
    reports it, and `refresh_wiki(doc, body=...)` closes the loop ;
  - `update_tool` refreshes the tool's docs immediately.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.journal import Journal
from metacog.memory import Memory
from metacog import wiki as W


class _LLM:
    def extract_common(self, texts):
        return ""


def _mem():
    m = Memory(encoder=SimpleEncoder(), journal=Journal(), llm=_LLM())
    a = m.ingest("swollen finger", kind="FACT", id="A")
    a.add_tag("health:macrodactyly")
    b = m.ingest("chronic tiredness", kind="FACT", id="B")
    b.add_tag("health:fatigue")
    return m


def _node(m, nid):
    return next(p for p in m.points if p.id == nid)


def test_fingerprint_says_which_part_drifted():
    f0 = W.node_fingerprint("x", ["fact", "health:a"])
    assert W.fingerprint_drift(None, f0) is None and W.fingerprint_drift(f0, f0) is None
    assert W.fingerprint_drift(f0, W.node_fingerprint("y", ["health:a"])) == W.REF_CONTENT_CHANGED
    assert W.fingerprint_drift(f0, W.node_fingerprint("x", ["health:b"])) == W.REF_TAGS_CHANGED
    assert W.fingerprint_drift(f0, W.node_fingerprint("y", ["health:b"])) == W.REF_CHANGED
    # structural tags are not knowledge : they never count as drift
    assert W.node_fingerprint("x", ["health:a"]) == W.node_fingerprint("x", ["fact", "health:a", "refined"])


def test_generated_doc_regenerates_when_a_ref_changes():
    m = _mem()
    out = m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic")
    assert out["body_mode"] == "generated"
    assert m.reconcile_wiki()["refreshed"] == 0                 # nothing drifted
    _node(m, "A").content = "swollen finger after exertion"    # content drift
    _node(m, "B").add_tag("health:sleep")                     # tag drift
    r = m.sleep()
    assert r["wiki_refreshed"] == 1 and r.get("wiki_outdated", 0) == 0
    md = m.wiki_doc("doc:h")
    assert "swollen finger after exertion [[A]]" in md          # body follows
    assert "#health:sleep" in md and "health:sleep" in md.split("---")[1]   # tags follow
    assert m.wiki_where("tags", "health:sleep") == ["doc:h"]     # index follows
    assert m.wiki_where("outdated") == []
    assert m.reconcile_wiki()["refreshed"] == 0                 # re-baselined : idempotent


def test_authored_doc_is_flagged_never_overwritten_then_refreshed_with_new_prose():
    m = _mem()
    prose = "Marie's finger swells [[A]] and she is tired [[B]]. My own analysis."
    out = m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic", body=prose)
    assert out["body_mode"] == "authored"
    _node(m, "A").content = "swollen finger after exertion"
    r = m.sleep()
    assert r["wiki_outdated"] == 1 and r["wiki_refreshed"] == 0
    assert r["wiki_stale_reasons"] == {W.REF_CONTENT_CHANGED: 1}
    assert "My own analysis" in m.wiki_doc("doc:h")             # untouched
    assert "outdated: 1" in m.wiki_doc("doc:h")                 # but it says so
    assert m.wiki_where("outdated") == ["doc:h"]                # and is findable
    v = [x for x in m.check_wiki("doc:h") if x["code"] == "outdated_ref"]
    assert v == [{"doc_id": "doc:h", "code": "outdated_ref",
                  "detail": {"ref": "A", "reason": W.REF_CONTENT_CHANGED}}]
    # refresh without prose : nothing changes, the pending changes are returned
    pend = m.refresh_wiki("doc:h")
    assert pend["refreshed"] is False and pend["reason"] == "authored_body_needs_text"
    assert pend["changes"] == [{"ref": "A", "reason": W.REF_CONTENT_CHANGED,
                                "content": "swollen finger after exertion",
                                "tags": ["health:macrodactyly"]}]
    # refresh with the rewritten prose : stored, re-baselined, flags cleared
    done = m.refresh_wiki("doc:h", body="Her finger swells after exertion [[A]]; tired [[B]].")
    assert done["refreshed"] is True and done["body_mode"] == "authored"
    assert "after exertion [[A]]" in m.wiki_doc("doc:h")
    assert m.wiki_where("outdated") == [] and m.check_wiki("doc:h") == [] or \
        not any(x["code"] == "outdated_ref" for x in m.check_wiki("doc:h"))
    assert m.reconcile_wiki()["outdated"] == 0


def test_drift_resolved_elsewhere_clears_the_flag():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A"], type="topic", body="prose [[A]]")
    _node(m, "A").content = "changed"
    m.reconcile_wiki()
    assert m.wiki_where("outdated") == ["doc:h"]
    _node(m, "A").content = "swollen finger"                    # reverted by hand
    m.reconcile_wiki()
    assert m.wiki_where("outdated") == []


def test_update_tool_refreshes_its_generated_doc_immediately():
    m = _mem()
    tid = m.ensure_tool("reverse a string end to end", how="iterate backwards")["tool"]["id"]
    assert "iterate backwards" in m.wiki_doc(f"tool:{tid}") or tid in m.wiki_doc(f"tool:{tid}")
    m.update_tool(tid, content="reverse a string with slicing s[::-1]")
    assert "reverse a string with slicing s[::-1] [[" in m.wiki_doc(f"tool:{tid}")  # no sleep needed
    assert m.wiki_where("outdated") == []


def test_remapped_ref_gets_a_baseline_not_a_drift_flag():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A"], type="topic", body="see [[A]]")
    m.forget_node("A", reason="superseded by B", superseded_by="B")
    m.merge_forgotten()
    r = m.reconcile_wiki()
    assert r["remapped"] == 1 and r["outdated"] == 0            # the redirect is not drift
    refs = {x["node_id"]: x for x in m.journal.wiki_refs_for_doc("doc:h")}
    assert refs["B"]["fingerprint"] and refs["B"]["outdated"] is None


def test_ingest_from_wiki_keeps_baselines_and_mode():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A"], type="topic")        # generated
    out = m.ingest_from_wiki("doc:h", "fingers swell after exertion")
    refs = {x["node_id"]: x for x in m.journal.wiki_refs_for_doc("doc:h")}
    assert refs["A"]["fingerprint"] and refs[out["node_id"]]["fingerprint"]
    assert m.journal.get_wiki_doc("doc:h")["body_mode"] == "generated"
    _node(m, "A").content = "swollen finger (left hand)"
    m.reconcile_wiki()                                          # regenerates over ALL refs
    md = m.wiki_doc("doc:h")
    assert "(left hand) [[A]]" in md and f"[[{out['node_id']}]]" in md


def test_no_journal_no_ops():
    m = Memory(encoder=SimpleEncoder())
    assert m.refresh_wiki("d")["reason"] == "no_journal"
    assert m.refresh_wiki_for_node("A") == {"docs": [], "refreshed": 0, "outdated": 0}
    assert m.reconcile_wiki()["refreshed"] == 0
