"""D10 (Paperclip TAC-2465) — the journal's mutable state as records of the ontology.

`FakeApi` plays tachikoma's two routes with their rules (tachikoma#1758) : a doc's row
version is its lock (409, nothing written), a doc is never removed (`discarded`), a
batch of docs alone does not move the store's revision. The real routes are measured
against a test API (the PR's bench) ; here the contract omni relies on.

The PARITY test is the one that matters : the same calls on a plain SQLite `Journal`
and on a `Neo4jStateJournal`, then a FRESH `Neo4jStateJournal` loaded from the store —
every persisted state table equal, row for row, to the SQLite one.
"""

import copy
import json

import pytest

from metacog.journal import Journal
from metacog.journal_state_neo4j import (PERSISTED_TABLES, Neo4jStateJournal,
                                         SplitJournal, _rows)
from metacog.store.base import CorruptStoreError


class FakeApi:
    def __init__(self) -> None:
        self.docs = {}            # doc_id -> {"version", "fields"}
        self.store = {"version": 0, "fields": {}}
        self.calls = []
        self.down = False

    def _call(self, method, path, body=None):
        self.calls.append((method, path, copy.deepcopy(body)))
        if self.down:
            raise CorruptStoreError("unreachable : ConnectionRefusedError")
        if (method, path) == ("GET", "/api/memory/points:wiki"):
            return 200, {
                "store_version": self.store["version"],
                "okf_vocab": self.store["fields"].get("okf_vocab", "[]"),
                "ref_remaps": self.store["fields"].get("ref_remaps", "[]"),
                "wiki_docs": [{"doc_id": d, "version": r["version"],
                               "fields": copy.deepcopy(r["fields"])}
                              for d, r in sorted(self.docs.items())]}
        if (method, path) == ("POST", "/api/memory/points:batch"):
            conflicts = [w["doc_id"] for w in body.get("wiki_docs", [])
                         if w["version"] != self.docs.get(w["doc_id"], {}).get("version", 0)]
            if conflicts:
                return 409, {"detail": {"conflicts": conflicts}}
            versions = {}
            for w in body.get("wiki_docs", []):
                row = self.docs.setdefault(w["doc_id"], {"version": 0, "fields": {}})
                row["fields"] = {**row["fields"], **w["fields"], "discarded": w["discarded"]}
                row["version"] += 1
                versions[f"wiki:{w['doc_id']}"] = row["version"]
            if body.get("store") is not None:
                self.store["fields"].update(body["store"])
                self.store["version"] += 1
            return 200, {"versions": versions, "store_version": self.store["version"],
                         "store_written": body.get("store") is not None}
        return 404, {"detail": path}


def _scenario(j) -> None:
    j.upsert_wiki_doc("notes:a", "concept", "A", ["wiki", "a"], "# A [[fact_1]]", ts=1.0,
                      body_mode="authored")
    j.set_wiki_refs("notes:a", ["fact_1", "fact_2"], fingerprints={"fact_1": "h1"})
    j.mark_wiki_ref_outdated("notes:a", "fact_2", "content_changed")
    j.index_okf_fields("notes:a", "concept", {"owner": "chrys", "topics": ["x", "y"]})
    j.set_okf_derived("notes:a", [("domain", "infra", "rule-1")], ts=2.0)
    j.upsert_seed("notes:a", "s1", "what is a", "*", 7, ["fact_1"], ts=3.0)
    j.add_annotation("notes:a", "*", "note", "keep it short", author="kusanagi", ts=4.0)
    j.add_pending("notes:a", "*", "seed_moved", {"seed": "s1"}, ts=5.0)
    j.upsert_wiki_doc("notes:b", "howto", "B", [], "# B", ts=6.0)
    j.set_wiki_refs("notes:b", ["fact_3"])
    j.upsert_seed("notes:b", "s2", "how b", "*", 3, [], ts=6.5)
    j.propose_vocab("type", "gadget", doc_id="notes:b", ts=7.0)
    j.vet_vocab("type", "gadget", "accepted", ts=8.0)
    rid = j.log_ref_remap("notes:a", "fact_2", "fact_9", had_new=False, ordinals=[0], ts=9.0)
    j.mark_ref_remap_reverted(rid)
    j.remap_wiki_ref("notes:a", "fact_2", "fact_9")
    j.resolve_pending("notes:a")
    j.delete_wiki_doc("notes:b")              # the doc goes, its seed stays (history)


def _tables(j):
    return {t: _rows(j.conn, t) for t in PERSISTED_TABLES}


def test_parity_with_sqlite_after_a_fresh_reload():
    ref = Journal(":memory:")
    _scenario(ref)
    api = FakeApi()
    j = Neo4jStateJournal(api)
    _scenario(j)
    assert j.pending_docs == set() and j.pending_store is False
    again = Neo4jStateJournal(api)            # a new process : load from the store
    assert _tables(again) == _tables(ref)
    assert _tables(j) == _tables(ref)
    # Reads answer what SQLite answers.
    assert again.get_wiki_doc("notes:a") == ref.get_wiki_doc("notes:a")
    assert again.get_wiki_doc("notes:b") is None
    assert again.seeds_for_doc("notes:b") == ref.seeds_for_doc("notes:b")
    assert again.okf_docs_where("owner", "chrys") == ["notes:a"]
    assert again.vocab_status("type", "gadget") == ref.vocab_status("type", "gadget")


def test_a_deleted_doc_is_archived_with_its_history_never_removed():
    api = FakeApi()
    j = Neo4jStateJournal(api)
    _scenario(j)
    b = api.docs["notes:b"]["fields"]
    assert b["discarded"] is True
    assert json.loads(b["refs"]) == [] and len(json.loads(b["seeds"])) == 1
    assert api.docs["notes:a"]["fields"]["discarded"] is False


def test_one_outer_call_is_one_batch_and_reads_send_nothing():
    api = FakeApi()
    j = Neo4jStateJournal(api)
    api.calls.clear()
    j.upsert_wiki_doc("notes:a", "concept", "A", [], "# A", ts=1.0)
    posts = [c for c in api.calls if c[0] == "POST"]
    assert len(posts) == 1 and [w["doc_id"] for w in posts[0][2]["wiki_docs"]] == ["notes:a"]
    assert "store" not in posts[0][2]            # a doc alone never writes the store
    api.calls.clear()
    j.get_wiki_doc("notes:a"), j.all_wiki_doc_ids(), j.wiki_refs_for_doc("notes:a")
    assert api.calls == []
    j.upsert_wiki_doc("notes:a", "concept", "A", [], "# A", ts=1.0)   # same row
    assert [c for c in api.calls if c[0] == "POST"] == []


def test_the_tag_index_is_never_written_to_the_store():
    api = FakeApi()
    j = Neo4jStateJournal(api)
    api.calls.clear()
    j.log_tags("fact_1", ["a:b", "c"])
    assert api.calls == [] and j.nodes_with_tag("a") == ["fact_1"]


def test_an_unreachable_api_keeps_the_writes_for_the_next_flush():
    api = FakeApi()
    j = Neo4jStateJournal(api)
    api.down = True
    j.upsert_wiki_doc("notes:a", "concept", "A", [], "# A", ts=1.0)   # never raises
    assert j.get_wiki_doc("notes:a")["title"] == "A"                  # served from memory
    assert j.pending_docs == {"notes:a"} and j.stats["failed"] == 1
    api.down = False
    assert j.flush() is True and api.docs["notes:a"]["fields"]["title"] == "A"


def test_a_conflict_rereads_the_versions_and_writes_again():
    api = FakeApi()
    j = Neo4jStateJournal(api)
    j.upsert_wiki_doc("notes:a", "concept", "A", [], "# A", ts=1.0)
    api.docs["notes:a"]["version"] += 5                 # another writer moved it
    j.upsert_wiki_doc("notes:a", "concept", "A2", [], "# A2", ts=2.0)
    assert j.stats["conflict"] == 1 and j.pending_docs == set()
    assert api.docs["notes:a"]["fields"]["title"] == "A2"


def test_a_store_that_cannot_be_read_is_refused_never_served_empty():
    api = FakeApi()
    api.down = True
    with pytest.raises(CorruptStoreError):
        Neo4jStateJournal(api)


def test_split_journal_serves_state_from_neo4j_and_usage_from_sqlite(tmp_path):
    api = FakeApi()
    usage = Journal(str(tmp_path / "memory.pkl.journal.db"))
    j = SplitJournal(usage, Neo4jStateJournal(api))
    j.upsert_wiki_doc("notes:a", "concept", "A", [], "# A", ts=1.0)
    rid = j.log_retrieval("q", ["fact_1"])
    assert rid == 1 and usage.counts()[0] == 1
    assert usage.conn.execute("SELECT COUNT(*) FROM wiki_docs").fetchone()[0] == 0
    assert j.conn.execute("SELECT COUNT(*) FROM wiki_docs").fetchone()[0] == 1
    assert "notes:a" in api.docs
