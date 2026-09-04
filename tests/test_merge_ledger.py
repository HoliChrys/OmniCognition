"""
Utopia-inspired safety rails on the mnema / OKF layers :

  1. ACTIVE redirects — a merge is a persistent redirect (ledger), not a
     one-shot rewrite : refs discovered AFTER the merge (a new feed, an external
     bundle, a restart) still land on the keeper ; cascades A->B->C chain.
  2. REVERSIBLE ledger — every forget / merge / collapse is a row `revert_merge`
     can undo (state, tags, redirect, and EXACTLY the wiki refs it rewrote).
  3. EXPLICIT reason codes — stale refs, dropped/unknown refs, tool rejections
     and lifecycle events always say why ; nothing collapses silently.
  4. CONSISTENCY (read-only) vs INFERENCE (materialized, opt-in, separate
     table, asserted wins).
  5. PROPOSAL loop — out-of-vocabulary OKF types are preserved as proposals ;
     emergent tools start PROPOSED and earn ESTABLISHED by use.
"""

from __future__ import annotations

import os
import tempfile

from metacog.defaults import SimpleEncoder
from metacog.epistemic import EpistemicState
from metacog.journal import Journal
from metacog.memory import Memory
from metacog import wiki as W


class _LLM:          # no generate -> deterministic tool synthesis fallback
    def extract_common(self, texts):     # sleep's collision pass : never merges
        return ""


def _mem(journal=None):
    m = Memory(encoder=SimpleEncoder(), journal=journal or Journal(), llm=_LLM())
    a = m.ingest("swollen finger", kind="FACT", id="A")
    a.tags.append("health:macrodactyly")
    b = m.ingest("chronic tiredness", kind="FACT", id="B")
    b.tags.append("health:fatigue")
    m.ingest("third fact", kind="FACT", id="C")
    return m


# -- 1. active redirects ------------------------------------------------------

def test_merge_is_a_persistent_redirect_in_the_ledger():
    m = _mem()
    m.forget_node("A", reason="superseded by B", superseded_by="B")
    m.merge_forgotten()
    hist = m.merge_history("A")
    kinds = {h["kind"] for h in hist}
    assert kinds == {"forget", "merge"}                 # both ops recorded
    assert m.journal.redirect_of("A") == "B"            # the redirect itself
    assert m.absorbed_into("B") == ["A"]                # reverse edge


def test_reference_discovered_after_the_merge_is_redirected():
    """A doc fed AFTER the merge (still naming the old id) lands on the keeper."""
    m = _mem()
    m.forget_node("A", reason="superseded by B", superseded_by="B")
    m.merge_forgotten()
    out = m.feed_wiki("doc:late", "Late", ["A"], type="topic")
    assert out["refs"] == ["B"]
    assert {"ref": "A", "to": "B", "reason": W.REF_REDIRECTED} in out["issues"]
    # an EXTERNAL bundle citing the absorbed id is redirected too (body incl.)
    md = "---\ntype: topic\ntitle: Ext\nrefs:\n- A\n---\n\n- x [[A]]\n"
    imp = m.import_okf("doc:ext", md)
    assert imp["refs"] == ["B"] and "[[B]]" in m.wiki_doc("doc:ext")


def test_redirect_cascade_chains_to_the_final_keeper():
    m = _mem()
    m.forget_node("A", reason="superseded by B", superseded_by="B")
    m.merge_forgotten()
    m.forget_node("B", reason="superseded by C", superseded_by="C")
    m.merge_forgotten()
    assert m.resolve_alias("A") == "C"
    assert set(m.absorbed_into("C")) == {"A", "B"}      # transitive reverse


def test_redirects_survive_a_restart_via_the_journal():
    with tempfile.TemporaryDirectory() as d:
        jp = os.path.join(d, "j.db")
        m = _mem(Journal(jp))
        m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic")
        m.forget_node("A", reason="superseded by B", superseded_by="B")
        m.merge_forgotten()
        m.journal.conn.close()
        # fresh process : empty alias map, same journal file
        m2 = Memory(encoder=SimpleEncoder(), journal=Journal(jp))
        m2.ingest("swollen finger", kind="FACT", id="A")
        m2.ingest("chronic tiredness", kind="FACT", id="B")
        assert m2.resolve_alias("A") == "B"             # re-hydrated
        assert m2.docs_for_node("A") == ["doc:h"]


# -- 2. reversibility ---------------------------------------------------------

def test_revert_merge_restores_node_alias_and_wiki_refs():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic")
    m.feed_wiki("doc:b", "OnlyB", ["B"], type="topic")   # untouched by the merge
    m.forget_node("A", reason="superseded by B", superseded_by="B")
    m.merge_forgotten()
    m.reconcile_wiki()
    assert "[[A]]" not in m.wiki_doc("doc:h")
    r = m.revert_merge("A")
    assert r["reverted"] == 2 and r["restored"] is True and r["docs"] == ["doc:h"]
    a = next(p for p in m.points if p.id == "A")
    assert a.state is not EpistemicState.INVALID and "invalidated" not in a.tags
    assert m.resolve_alias("A") == "A"
    md = m.wiki_doc("doc:h")
    assert "swollen finger [[A]]" in md                  # exactly un-rewritten
    assert "chronic tiredness [[B]]" in md               # the author's B kept
    refs = {r["node_id"] for r in m.journal.wiki_refs_for_doc("doc:h")}
    assert refs == {"A", "B"}
    assert m.wiki_where("refs", "A") == ["doc:h"]        # index follows
    assert "[[B]]" in m.wiki_doc("doc:b")                # the other doc intact
    assert all(h["reverted"] for h in m.merge_history("A"))
    assert m.revert_merge("A")["reason"] == "no_live_ledger_row"   # idempotent


def test_lateral_and_duplicate_merges_are_ledgered_with_reasons():
    m = _mem()
    m._alias("A", "B", "lateral", "functionally redundant with B")
    m._alias("C", "B", "duplicate", "same information as B")
    kinds = {h["absorbed_id"]: (h["kind"], h["reason"]) for h in m.merge_history()}
    assert kinds["A"][0] == "lateral" and "redundant" in kinds["A"][1]
    assert kinds["C"][0] == "duplicate"
    assert set(m.absorbed_into("B")) == {"A", "C"}


# -- 3. explicit reason codes -------------------------------------------------

def test_stale_refs_carry_an_explicit_reason():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A", "B", "C"], type="topic")
    m.forget_node("A", reason="user corrected")          # invalid, no successor
    m.points = [p for p in m.points if p.id != "C"]      # hard-gone
    rec = m.reconcile_wiki()
    assert rec["reasons"] == {W.REF_INVALIDATED: 1, W.REF_MISSING: 1}
    why = {r["node_id"]: r["reason"] for r in m.journal.wiki_refs_for_doc("doc:h")}
    assert why["A"] == W.REF_INVALIDATED and why["C"] == W.REF_MISSING
    assert why["B"] is None


def test_unknown_ref_is_kept_but_flagged_and_reported():
    m = _mem()
    out = m.feed_wiki("doc:x", "X", ["A", "ZZZ"], type="topic")
    assert {"ref": "ZZZ", "reason": W.REF_MISSING} in out["issues"]
    assert "ZZZ" in out["refs"]                          # never dropped silently
    v = [x for x in m.check_wiki("doc:x") if x["code"] == "stale_ref"]
    assert v and v[0]["detail"] == {"ref": "ZZZ", "reason": W.REF_MISSING}


def test_import_without_frontmatter_says_so():
    m = _mem()
    out = m.import_okf("doc:bare", "just prose, no yaml")
    assert {"reason": W.DOC_NO_FRONTMATTER} in out["issues"]


def test_tool_lifecycle_events_have_reasons():
    m = _mem()
    assert m.ensure_tool("")["reason"] == "empty_query"
    r = m.ensure_tool("reverse a string end to end", how="iterate backwards")
    tid = r["tool"]["id"]
    m.report_tool(tid, ok=False)
    m.report_tool(tid, ok=False)                         # auto-retire
    ev = {(e["event"]) for e in m.tool_history(tid)}
    assert {"created", "failed", "retired"} <= ev
    retired = next(e for e in m.tool_history(tid) if e["event"] == "retired")
    assert "consecutive failures" in retired["reason"]
    assert m.retire_tool("nope")["reason"] == "not_a_tool"


# -- 4. consistency (read-only) vs inference (materialized, opt-in) -----------

def test_check_wiki_surfaces_violations_and_writes_nothing():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic")
    doc = m.journal.get_wiki_doc("doc:h")
    # tamper the prose : cite an unlinked id + an inline tag not in frontmatter
    m.journal.upsert_wiki_doc("doc:h", doc["type"], doc["title"], doc["tags"],
                              doc["body"] + "\n- see [[Q]] #ops:deploy", doc["ts"])
    before = (m.journal.wiki_refs_for_doc("doc:h"), m.okf_fields("doc:h"))
    codes = {v["code"] for v in m.check_wiki("doc:h")}
    assert {"body_ref_unlinked", "tag_not_in_frontmatter"} <= codes
    assert (m.journal.wiki_refs_for_doc("doc:h"), m.okf_fields("doc:h")) == before


def test_schema_drift_is_reported_per_type():
    m = _mem()
    for i in range(3):
        m.feed_wiki(f"doc:{i}", f"D{i}", ["A"], type="topic")
    m.feed_wiki("doc:odd", "Odd", [], type="topic")      # no refs -> drift
    codes = [v for v in m.check_wiki("doc:odd") if v["code"] == "schema_drift"]
    assert any(v["detail"]["field"] == "refs" for v in codes)


def test_inference_is_off_by_default_and_separate_from_asserted():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic")
    m.feed_wiki("doc:t", "Tired", ["B"], type="topic")
    m.sleep()                                            # infer_enabled False
    assert m.okf_derived("doc:h") == []
    prev = m.infer_wiki()                                # preview only
    assert prev["applied"] is False and prev["derived"] > 0
    assert m.okf_derived("doc:h") == []
    m.infer_enabled = True
    m.sleep()
    d = {(r["key"], r["value"]): r["rule"] for r in m.okf_derived("doc:h")}
    assert d[("related", "doc:t")] == "shared_ref"
    # derived never leaks into the asserted index ; opt-in on query
    assert m.wiki_where("related", "doc:t") == []
    assert m.wiki_where("related", "doc:t", derived=True) == ["doc:h"]
    assert "related" not in m.okf_fields("doc:h")


def test_asserted_fields_win_over_inference():
    m = _mem()
    m.feed_wiki("doc:h", "Health", ["A", "B"], type="topic")
    # the doc already asserts tags health:* -> node_tag_drift must not re-derive
    a = next(p for p in m.points if p.id == "A")
    a.tags.append("health:new")                          # drift after the feed
    rows = m.infer_wiki(apply=True)
    d = {(r["key"], r["value"]) for r in m.okf_derived("doc:h")}
    assert ("tags", "health:new") in d                   # the drift is derived
    assert ("tags", "health:fatigue") not in d           # asserted -> skipped
    assert rows["applied"] is True


# -- 5. proposal loop ---------------------------------------------------------

def test_unknown_okf_type_is_preserved_as_a_proposal_then_vetted():
    m = _mem()
    out = m.feed_wiki("doc:c", "Commit", ["A"], type="commit")
    assert {"type": "commit", "reason": W.TYPE_PROPOSED} in out["issues"]
    assert m.wiki_where("type", "commit") == ["doc:c"]   # never blocked
    assert [p["value"] for p in m.okf_proposals()] == ["commit"]
    assert any(v["code"] == W.TYPE_PROPOSED for v in m.check_wiki("doc:c"))
    m.vet_okf_type("commit", accept=True)
    assert m.okf_proposals() == []
    assert not any(v["code"].startswith("type_") for v in m.check_wiki("doc:c"))
    assert m.feed_wiki("doc:c2", "C2", ["B"], type="commit")["issues"] == []
    m.vet_okf_type("commit", accept=False)
    assert any(v["code"] == W.TYPE_REJECTED for v in m.check_wiki("doc:c"))
    assert m.wiki_where("type", "commit") == ["doc:c", "doc:c2"]   # untouched


def test_emergent_tool_starts_proposed_and_earns_established():
    m = _mem()
    r = m.ensure_tool("reverse a string end to end", how="iterate backwards")
    tid = r["tool"]["id"]
    assert r["tool"]["status"] == "proposed"
    assert m.wiki_where("status", "proposed") == [f"tool:{tid}"]
    assert m.match_tool("reverse a string end to end") is not None   # usable
    for _ in range(m.tool_promote_after):
        m.report_tool(tid, ok=True)
    m.sleep()                                            # autonomic promotion
    tool = m._find_tool(tid)
    assert m.tool_status(tool) == "established"
    assert m.wiki_where("status", "established") == [f"tool:{tid}"]
    assert "status: established" in m.wiki_doc(f"tool:{tid}")
    assert any(e["event"] == "promoted" for e in m.tool_history(tid))


def test_explicit_promotion_and_no_promotion_with_failure_streak():
    m = _mem()
    tid = m.ensure_tool("count words in text", how="split")["tool"]["id"]
    for _ in range(m.tool_promote_after):
        m.report_tool(tid, ok=True)
    m.report_tool(tid, ok=False)                         # live streak
    assert m.promote_tools()["promoted"] == []
    assert m.promote_tool(tid, reason="reviewed")["status"] == "established"
    assert m.promote_tool("nope")["reason"] == "not_a_tool"


def test_no_journal_no_ops():
    m = Memory(encoder=SimpleEncoder(), llm=_LLM())
    assert m.revert_merge("A")["reason"] == "no_journal"
    assert m.absorbed_into("A") == [] and m.merge_history() == []
    assert m.check_wiki() == [] and m.infer_wiki()["derived"] == 0
    assert m.okf_proposals() == [] and m.vet_okf_type("x", True)["status"] is None
