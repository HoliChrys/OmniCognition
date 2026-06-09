"""
Tests for event-type schema induction + schema-driven sub-questions.

Deterministic: a scripted fake LLM returns `slot | core/peripheral` lines.
"""

from __future__ import annotations

from metacog.event_schema import (
    induce_event_schema, slot_subquestions, EventSchema, _SCHEMA_CACHE,
)


class _FakeLLM:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def generate(self, prompt, max_tokens=200):
        self.calls += 1
        return self.text


_WAR = ("belligerents | core\nterritory | core\ntimeline | core\n"
        "casualties | peripheral\ntreaties | peripheral")


def test_induces_slots_and_marks_core():
    _SCHEMA_CACHE.pop("war", None)
    s = induce_event_schema("war", _FakeLLM(_WAR))
    assert s.etype == "war"
    assert s.slots == ["belligerents", "territory", "timeline",
                       "casualties", "treaties"]
    assert s.core == ["belligerents", "territory", "timeline"]
    assert "casualties" not in s.core


def test_schema_is_cached():
    _SCHEMA_CACHE.pop("war", None)
    llm = _FakeLLM(_WAR)
    induce_event_schema("war", llm)
    induce_event_schema("war", llm)            # second call hits the cache
    assert llm.calls == 1


def test_never_caches_empty():
    _SCHEMA_CACHE.pop("nonsense", None)
    induce_event_schema("nonsense", _FakeLLM(""))   # no parseable slots
    assert "nonsense" not in _SCHEMA_CACHE


def test_failsafe_without_llm():
    _SCHEMA_CACHE.pop("uncachedtype", None)
    s = induce_event_schema("uncachedtype", object())   # no .generate, no cache
    assert s.is_empty() and s.etype == "uncachedtype"


def test_slot_subquestions_one_per_slot():
    s = EventSchema(etype="war",
                    slots=["belligerents", "territory"], core=["belligerents"])
    qs = slot_subquestions("the Ukraine war", s)
    assert [slot for slot, _ in qs] == ["belligerents", "territory"]
    assert all("the Ukraine war" in q for _, q in qs)
    assert "belligerents" in qs[0][1]


class _FakeMem:
    """Minimal memory for fill_event_schema: scripted retrieve + llm."""
    def __init__(self, llm, hits_by_kw):
        self.llm = llm
        self.hits_by_kw = hits_by_kw
        class _P:  # minimal point with .id
            def __init__(s, i): s.id = i
        self.points = [_P(i) for kw in hits_by_kw.values() for i in kw]

    def retrieve(self, q, k=3):
        for kw, ids in self.hits_by_kw.items():
            if kw in q:
                return [{"id": i, "content": i, "score": 0.5} for i in ids][:k]
        return []


def test_fill_event_schema_decomposes_and_aggregates():
    from metacog.event_schema import fill_event_schema, _SCHEMA_CACHE
    _SCHEMA_CACHE.pop("war", None)
    llm = _FakeLLM("belligerents | core\nterritory | core\ntimeline | core")
    mem = _FakeMem(llm, {
        "belligerents": ["F1"], "territory": ["F2"], "timeline": ["F3"],
    })
    res = fill_event_schema(mem, "the border war", "war", k_per_slot=2)
    assert res["etype"] == "war"
    assert set(res["filled"]) == {"belligerents", "territory", "timeline"}
    assert res["filled"]["belligerents"][0]["id"] == "F1"
    # union of per-slot hits, deduped, one sub-question per slot
    assert res["fact_ids"] == ["F1", "F2", "F3"]


def test_fill_event_schema_empty_on_no_schema():
    from metacog.event_schema import fill_event_schema, _SCHEMA_CACHE
    _SCHEMA_CACHE.pop("blob", None)
    mem = _FakeMem(_FakeLLM(""), {})
    res = fill_event_schema(mem, "x", "blob")
    assert res["filled"] == {} and res["fact_ids"] == []


class _ClusterMem:
    """Fake memory with an event cluster + gap-fill corpus, for fill_event_schema."""
    def __init__(self, llm, cluster_hits, corpus_hits):
        self.llm = llm
        self.cluster_hits = cluster_hits      # slot-kw -> ids (in cluster)
        self.corpus_hits = corpus_hits        # slot-kw -> ids (full corpus)
        self._event_registry = {"war::the war": "event_x"}
        class _P:
            def __init__(s, i): s.id = i; s.tags = ["event:in:event_x"]
        self._members = [_P(i) for ids in cluster_hits.values() for i in ids]
        self.points = list(self._members)
        self._scoped = True
    def event_cluster(self, eid):
        return self._members
    def retrieve(self, q, k=3):
        # cluster scope vs corpus is signalled by self.points identity
        table = self.cluster_hits if self.points is self._members else self.corpus_hits
        for kw, ids in table.items():
            if kw in q:
                return [{"id": i, "content": i, "score": 0.5} for i in ids][:k]
        return []


def test_cluster_partition_and_gap_fill():
    from metacog.event_schema import fill_event_schema, _SCHEMA_CACHE
    _SCHEMA_CACHE.pop("war", None)
    llm = _FakeLLM("belligerents | core\nterritory | core\ntreaties | peripheral")
    # cluster has belligerents+territory ; 'treaties' only in corpus (peripheral
    # -> no gap-fill) ; make 'territory' empty in cluster to force a core GAP
    mem = _ClusterMem(
        llm,
        cluster_hits={"belligerents": ["F1"]},                 # territory missing
        corpus_hits={"territory": ["F9"], "treaties": ["F7"]},
    )
    res = fill_event_schema(mem, "the war", "war", k_per_slot=2)
    assert res["cluster_size"] == 1
    assert res["filled"]["belligerents"][0]["id"] == "F1"      # from cluster
    # 'territory' is a CORE slot empty in cluster -> GAP -> filled from corpus
    assert "territory" in res["gaps"]
    assert res["filled"]["territory"][0]["id"] == "F9"
    # 'treaties' is peripheral -> not a gap even though empty in cluster
    assert "treaties" not in res["gaps"]


def test_enrich_record_reinforce():
    from metacog.event_schema import (
        enrich_schema, record_fill, reinforce_schema, induce_event_schema,
        _SCHEMA_CACHE, _SCHEMA_STATS, EventSchema)
    _SCHEMA_CACHE.clear(); _SCHEMA_STATS.clear()
    # seed a cached schema for 'war'
    induce_event_schema("war", _FakeLLM(
        "belligerents | core\nterritory | core\ntreaties | peripheral"))

    # A.2 enrich : a leftover cluster fact -> fake LLM proposes a new slot
    class _M:
        llm = _FakeLLM("front lines")
        _event_registry = {"war::the war": "event_x"}
        class _P:
            def __init__(s, i, t): s.id = i; s.content = t; s.tags = []
        def event_cluster(self, eid):
            return [self._P("F9", "troops dug trenches on the front")]
    new = enrich_schema(_M(), "war", "the war",
                        {"slots": ["belligerents"], "fact_ids": ["F1"]})
    assert "front lines" in new
    assert "front lines" in _SCHEMA_CACHE["war"].slots   # added to type schema

    # A.3 reinforce : 'territory' never filled across 3 instances -> dropped ;
    # 'belligerents' always filled -> core.
    sc = _SCHEMA_CACHE["war"]
    for _ in range(3):
        record_fill("war", sc, {"filled": {"belligerents": [1], "treaties": [1]}})
    out = reinforce_schema("war", min_instances=3, core_frac=0.6)
    assert "territory" not in out.slots          # never filled -> dropped
    assert "belligerents" in out.core            # always filled -> core


class _ActLLM:
    """Returns a fixed action string for generate_action (deterministic)."""
    def generate(self, prompt, max_tokens=64):
        return "track the strikes on the energy nodes"


def test_event_action_enrich_bridges_schema_to_cluster():
    """The schema result is folded back via a meta-cognition ACTION : a
    GENERATOR ACTION node, parents = the schema facts, tagged into the cluster
    and pulled — the edge-free beacon that re-injects the findings."""
    from metacog.event_schema import event_action_enrich
    from metacog.epistemic import PointKind
    from metacog.defaults import SimpleEncoder
    from metacog.memory import Memory
    mem = Memory(encoder=SimpleEncoder(), llm=_ActLLM())
    f1 = mem.ingest("missile strike on the oil terminal", kind="FACT", id="D1")
    f2 = mem.ingest("naval blockade of the strait", kind="FACT", id="D2")
    e = mem.ingest_event("strait attack", "attack",
                         source_facts=[f1, f2], salience=-1)
    aid = event_action_enrich(mem, e.id, {"fact_ids": ["D1", "D2"]})
    assert aid is not None
    act = next(p for p in mem.points if p.id == aid)
    assert act.kind is PointKind.ACTION
    assert set(act.parents) >= {"D1", "D2"}                 # lineage preserved
    assert f"event:in:{e.id}" in act.tags and "event:action" in act.tags
    assert any(abs(x) > 0 for x in act.delta_active)        # the action was pulled
    assert aid in {p.id for p in mem.event_cluster(e.id)}   # joined the cluster


def test_event_action_enrich_noop_on_empty():
    """A step that found nothing produces no action -> no influence."""
    from metacog.event_schema import event_action_enrich
    from metacog.defaults import SimpleEncoder
    from metacog.memory import Memory
    mem = Memory(encoder=SimpleEncoder(), llm=_ActLLM())
    e = mem.ingest_event("x", "attack")
    assert event_action_enrich(mem, e.id, {"fact_ids": []}) is None


def test_bag_publish_intersect_and_action_seed_on_intersection():
    """Bags work as named containers : the cluster + each schema slot are
    published as bags, their INTERSECTION is the interpreted nugget, and
    event_action_enrich seeds its ACTION on that intersection when non-empty."""
    from metacog.event_schema import event_action_enrich
    from metacog.epistemic import PointKind
    from metacog.defaults import SimpleEncoder
    from metacog.memory import Memory

    class _ActLLM2:
        def generate(self, prompt, max_tokens=64):
            return "review the attacks"

    mem = Memory(encoder=SimpleEncoder(), llm=_ActLLM2())
    f1 = mem.ingest("missile strike on the oil terminal", kind="FACT", id="D1")
    f2 = mem.ingest("naval blockade of the strait", kind="FACT", id="D2")
    f3 = mem.ingest("a peripheral diplomatic remark", kind="FACT", id="D3")
    e = mem.ingest_event("strait attack", "attack",
                         source_facts=[f1, f2, f3], salience=-1)
    # cluster bag = D1,D2,D3 ; one slot bag = D1,D2 -> intersection = {D1,D2}
    mem.bag_publish_cluster(e.id)
    mem.bag_publish_schema(e.id, {"filled": {
        "target": [{"id": "D1"}, {"id": "D2"}],
    }})
    inter = mem.bag_intersect([f"event:{e.id}", f"event:{e.id}:schema"])
    assert set(inter) == {"D1", "D2"}
    assert set(mem.bag_union([f"event:{e.id}", f"event:{e.id}:schema"])) == {
        "D1", "D2", "D3"}
    # action seed picks the intersection, so parents = {D1,D2}, NOT D3
    aid = event_action_enrich(mem, e.id, {"fact_ids": ["D1", "D2", "D3"]})
    act = next(p for p in mem.points if p.id == aid)
    assert set(act.parents) == {"D1", "D2"}


def test_named_bags_independent_and_clearable():
    from metacog.defaults import SimpleEncoder
    from metacog.memory import Memory
    mem = Memory(encoder=SimpleEncoder())
    mem.bag_add(["A", "B"], bag="x")
    mem.bag_add(["B", "C"], bag="y")
    assert set(mem.bag_intersect(["x", "y"])) == {"B"}
    assert mem.bag_intersect(["x", "z-empty"]) == []
    mem.bag_clear(bag="x")
    assert mem.bag_intersect(["x", "y"]) == []
    # default bag is backwards-compatible
    mem.bag_add(["Z"])
    assert mem._bag == ["Z"]
