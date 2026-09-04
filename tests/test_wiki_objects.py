"""
Wiki OBJECTS : seed queries, portions, variables, annotations.

  - a seed query is attached to a portion (or the doc), its result cached ;
    re-run in sleep, a different result re-renders a generated target or
    records a PENDING change (with the diff) on an authored / kept one ;
  - a portion / a variable is an object with an identity : edits are ops
    (set / remove / replace), journaled and reversible, a binding to a
    missing node is refused, the rendered view resolves bindings live ;
  - annotations hang on objects (bibliography in the frontmatter, footnotes
    in the rendered view) and `keep` protects from regeneration / removal.
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
    m.ingest("deploy target is the staging cluster", kind="FACT", id="D1")
    m.ingest("deploy uses blue green rollout", kind="FACT", id="D2")
    m.ingest("kitchen recipe soup carrots", kind="FACT", id="K1")
    m.ingest("guitar chords music lesson", kind="FACT", id="G1")
    return m


def _node(m, nid):
    return next(p for p in m.points if p.id == nid)


# -- parsers ------------------------------------------------------------------

def test_parse_portions_vars_and_edit_in_place():
    body = ('intro\n<portion id="p1" seeds="q1,q2" mode="generated">\n- a [[A]]\n</portion>\n'
            'target: <var name="t" node="D1" field="content"/> end')
    ps = W.parse_portions(body)
    assert ps[0]["id"] == "p1" and ps[0]["seeds"] == ["q1", "q2"] and ps[0]["mode"] == "generated"
    assert ps[0]["body"] == "- a [[A]]"
    assert W.parse_vars(body) == [{"name": "t", "node": "D1", "field": "content"}]
    assert W.body_bindings(body) == ["D1"]
    b2, existed = W.set_portion_body(body, "p1", "- b [[B]]", refs=["B"])
    assert existed and "- b [[B]]" in b2 and 'seeds="q1,q2"' in b2 and "intro" in b2
    assert W.parse_portions(b2)[0]["refs"] == ["B"]
    b3, existed = W.set_portion_body(body, "p9", "new")
    assert not existed and '<portion id="p9">' in b3
    b4, prev = W.set_var_tag(body, "t", "D2", "content")
    assert prev["node"] == "D1" and 'node="D2"' in b4 and 'node="D1"' not in b4
    b5, prev = W.remove_var_tag(body, "t")
    assert prev and "<var" not in b5
    assert W.portion_of(body, "[[A]]") == "p1" and W.portion_of(body, "<var") is None
    rendered = W.resolve_body(body, {"t": "staging"}, [{"target": "t", "kind": "purpose", "note": "where we ship"}])
    assert "target: staging end" in rendered and "<portion" not in rendered
    assert "[^t] (purpose) where we ship" in rendered
    assert "⟨t: unbound⟩" in W.resolve_body(body, {}, [])


# -- seeds --------------------------------------------------------------------

def test_seed_on_generated_portion_renders_caches_and_absorbs_changes():
    m = _mem()
    m.feed_wiki("doc:ops", "Ops", ["D1"], type="topic", body="# Ops\nprose here [[D1]]")
    r = m.set_portion("doc:ops", "deploy", body="", mode="generated")
    assert r["ok"] and r["created"]
    s = m.add_seed("doc:ops", "how do we deploy", target="deploy", k=2)
    assert s["seed_id"] == "q1" and set(s["ids"]) == {"D1", "D2"}
    md = m.wiki_doc("doc:ops")
    assert 'seeds="q1"' in md and "[[D2]]" in md and "blue green rollout" in md
    assert "prose here [[D1]]" in md                                  # authored text kept
    assert set(m.wiki_where("refs", "D2")) == {"doc:ops"}
    assert m.wiki_where("seeds", "how do we deploy") == ["doc:ops"]
    assert m.rerun_seeds()["changed"] == 0                            # stable cache
    # the corpus moves : a new deploy fact outranks D2
    m.ingest("how do we deploy: run the deploy script", kind="FACT", id="D3")
    m.forget_node("D2", reason="obsolete")
    out = m.sleep()
    assert out["seeds_changed"] == 1 and out["seeds_refreshed"] == 1
    md = m.wiki_doc("doc:ops")
    portion = md.split("<portion")[1].split("</portion>")[0]
    assert "[[D3]]" in portion and "[[D2]]" not in portion            # absorbed the diff
    assert all(p["reason"] != "seed_changed" for p in m.wiki_pending("doc:ops"))


def test_seed_on_authored_target_records_pending_with_the_diff():
    m = _mem()
    m.feed_wiki("doc:ops", "Ops", ["D1"], type="topic", body="my prose [[D1]]")   # authored
    s = m.add_seed("doc:ops", "how do we deploy", target="*", k=2)
    assert "my prose [[D1]]" in m.wiki_doc("doc:ops")                 # untouched
    assert "D2" in [r["node_id"] for r in m.journal.wiki_refs_for_doc("doc:ops")]
    m.ingest("how do we deploy: run the deploy script", kind="FACT", id="D3")
    m.forget_node("D2", reason="obsolete")
    out = m.sleep()
    assert out["seeds_pending"] == 1
    p = m.wiki_pending("doc:ops")
    assert p[0]["reason"] == "seed_changed" and p[0]["target"] == "*"
    assert "D3" in p[0]["detail"]["added"] and "D2" in p[0]["detail"]["removed"]
    assert "pending: 1" in m.wiki_doc("doc:ops")
    assert m.wiki_where("pending") == ["doc:ops"]
    assert any(v["code"] == "pending_change" for v in m.check_wiki("doc:ops"))
    ch = m.refresh_wiki("doc:ops")["changes"]                          # the agent reads the diff
    assert any(c.get("reason") == "seed_changed" and "D3" in c["now_content"] for c in ch)
    done = m.refresh_wiki("doc:ops", body="my new prose [[D1]] and [[D3]]")
    assert done["refreshed"] and m.wiki_pending("doc:ops") == []
    assert m.remove_seed("doc:ops", "q1")["removed"]
    assert m.list_seeds("doc:ops") == []


def test_seed_targets_and_no_journal():
    m = _mem()
    m.feed_wiki("doc:ops", "Ops", ["D1"], type="topic")
    assert m.add_seed("doc:ops", "x", target="nope")["reason"] == "unknown_target"
    assert m.add_seed("doc:none", "x")["reason"] == "missing_doc"
    n = Memory(encoder=SimpleEncoder())
    assert n.add_seed("d", "q")["reason"] == "no_journal" and n.rerun_seeds()["reran"] == 0
    assert n.set_var("d", "v", "A")["reason"] == "no_journal"
    assert n.annotate("d", "*", "x")["reason"] == "no_journal"
    assert n.wiki_pending() == [] and n.wiki_ops("d") == []


# -- variables ----------------------------------------------------------------

def test_var_is_a_live_binding_journaled_and_reversible():
    m = _mem()
    m.feed_wiki("doc:ops", "Ops", ["D1"], type="topic", body="Target: (see var)")
    r = m.set_var("doc:ops", "target", "D1")
    assert r["bound"] and r["previous"] is None
    src = m.wiki_doc("doc:ops")
    assert '<var name="target" node="D1" field="content"/>' in src
    rend = m.wiki_doc("doc:ops", view="rendered")
    assert "deploy target is the staging cluster" in rend and "<var" not in rend
    _node(m, "D1").content = "deploy target is the prod cluster"      # the node moves
    assert "prod cluster" in m.wiki_doc("doc:ops", view="rendered")   # rendered live
    assert m.wiki_where("vars", "target") == ["doc:ops"]
    assert m.wiki_where("bindings", "D1") == ["doc:ops"]
    # rebind : op with before/after ; the ref set follows
    r2 = m.set_var("doc:ops", "target", "D2")
    assert r2["previous"]["node"] == "D1"
    assert "D2" in [x["node_id"] for x in m.journal.wiki_refs_for_doc("doc:ops")]
    assert m.set_var("doc:ops", "target", "ZZZ")["reason"] == "missing_node"   # reliability
    ops = m.wiki_ops("doc:ops")
    assert [o["op"] for o in ops] == ["set", "set"] and ops[0]["before"]["node"] == "D1"
    assert m.revert_wiki_op(ops[0]["id"])["reverted"]
    assert 'node="D1"' in m.wiki_doc("doc:ops")
    assert m.revert_wiki_op(ops[0]["id"])["reason"] == "no_live_op"     # once
    # remove, then revert the removal
    assert m.remove_var("doc:ops", "target")["removed"]
    assert "<var" not in m.wiki_doc("doc:ops")
    rm = m.wiki_ops("doc:ops")[0]
    assert rm["op"] == "remove" and m.revert_wiki_op(rm["id"])["reverted"]
    assert 'name="target"' in m.wiki_doc("doc:ops")
    assert m.wiki_vars("doc:ops")[0]["bound"] is True


# -- annotations & keep -------------------------------------------------------

def test_annotations_render_as_bibliography_and_keep_protects():
    m = _mem()
    m.feed_wiki("doc:ops", "Ops", ["D1", "D2"], type="topic")           # generated
    m.set_var("doc:ops", "target", "D1")
    m.set_portion("doc:ops", "rollout", body="- rollout notes [[D2]]", mode="generated")
    assert m.annotate("doc:ops", "target", "the cluster we ship to", kind="purpose")["annotation_id"]
    assert m.annotate("doc:ops", "rollout", "this part must be preserved", kind="keep")["annotation_id"]
    assert m.annotate("doc:ops", "D1", "source of truth for the target", kind="note")["annotation_id"]
    assert m.annotate("doc:ops", "ghost", "x")["reason"] == "unknown_target"
    assert m.annotate("doc:ops", "*", "x", kind="wtf")["reason"] == "unknown_kind"
    src = m.wiki_doc("doc:ops")
    assert "annotations:" in src and "the cluster we ship to" in src   # frontmatter bibliography
    rend = m.wiki_doc("doc:ops", view="rendered")
    assert "[^target] (purpose) the cluster we ship to" in rend
    assert "[^rollout] (keep) this part must be preserved" in rend
    assert m.wiki_where("annotations", "keep") == ["doc:ops"]
    assert m.wiki_where("keep", "rollout") == ["doc:ops"]
    # a generated portion was rendered from its source at creation
    assert "deploy uses blue green rollout [[D2]]" in m.wiki_doc("doc:ops")
    # keep : no removal, no regeneration of that portion
    assert m.remove_portion("doc:ops", "rollout")["reason"] == "kept"
    assert m.set_portion("doc:ops", "rollout", body="overwrite")["reason"] == "kept"
    _node(m, "D2").content = "deploy uses canary rollout"
    m.sleep()
    md = m.wiki_doc("doc:ops")
    assert "blue green rollout [[D2]]" in md and "canary" not in md    # preserved
    assert m.wiki_where("outdated") == ["doc:ops"]                    # but flagged
    # the var can be protected too
    m.annotate("doc:ops", "target", "must stay bound", kind="keep")
    assert m.remove_var("doc:ops", "target")["reason"] == "kept"
    aid = [a for a in m.annotations("doc:ops", "target") if a["kind"] == "keep"][0]["id"]
    assert m.remove_annotation("doc:ops", aid)["removed"]
    assert m.remove_var("doc:ops", "target")["removed"]


def test_doc_level_keep_spares_explicitly_generated_portions():
    m = _mem()
    m.feed_wiki("doc:ops", "Ops", ["D1"], type="topic", body="human intro [[D1]]")
    m.set_portion("doc:ops", "auto", body="- [[D2]]", mode="generated")
    m.set_portion("doc:ops", "notes", body="my notes [[D1]]")            # doc mode: authored
    m.annotate("doc:ops", "*", "keep the human wording", kind="keep")
    assert m._kept("doc:ops", "*") and m._kept("doc:ops", "notes") and m._kept("doc:ops", "D1")
    assert not m._kept("doc:ops", "auto")                              # machine-owned block
    _node(m, "D2").content = "deploy uses canary rollout"
    assert m.reconcile_wiki()["refreshed"] == 1
    assert "canary rollout [[D2]]" in m.wiki_doc("doc:ops") and "human intro [[D1]]" in m.wiki_doc("doc:ops")
    assert m.remove_portion("doc:ops", "notes")["reason"] == "kept"
    assert m.remove_portion("doc:ops", "auto")["removed"]


def test_generated_portion_regenerates_alone_when_its_ref_drifts():
    m = _mem()
    m.feed_wiki("doc:ops", "Ops", ["D1"], type="topic", body="authored intro [[D1]]")
    r = m.set_portion("doc:ops", "facts", body="- [[D2]]", mode="generated")
    assert r["refs"] == ["D2"]                                         # sources in the tag
    assert "deploy uses blue green rollout [[D2]]" in m.wiki_doc("doc:ops")  # rendered now
    _node(m, "D2").content = "deploy uses canary rollout"
    r = m.reconcile_wiki()
    assert r["refreshed"] == 1 and r["outdated"] == 0
    md = m.wiki_doc("doc:ops")
    assert "deploy uses canary rollout [[D2]]" in md and "authored intro [[D1]]" in md
    assert m.remove_portion("doc:ops", "facts")["removed"]
    assert "<portion" not in m.wiki_doc("doc:ops")
    assert "D2" not in [x["node_id"] for x in m.journal.wiki_refs_for_doc("doc:ops")]
