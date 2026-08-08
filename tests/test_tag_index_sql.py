"""
Hierarchical tag index in the journal (SQL) — adapts our hierarchical-tag
system to the DB substrate.

A node's `a:b:c` tags are indexed in a `tags` table ; a query matches as an
ANCESTOR (querying 'health' returns 'health:condition:x' nodes) via
`tag = ? OR tag LIKE ?||':%'`, and the glossary is depth-ordered in SQL —
the same semantics as metacog.tags._ancestor_of / tag_glossary, now DB-driven.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.journal import Journal
from metacog.memory import Memory


def test_log_tags_and_hierarchical_ancestor_match():
    j = Journal()
    j.log_tags("A", ["health:condition:macrodactyly", "fact"])
    j.log_tags("B", ["health:condition:fatigue"])
    j.log_tags("C", ["building:condition:poor"])
    # 'health' is an ANCESTOR of both A and B, not C.
    assert set(j.nodes_with_tag("health")) == {"A", "B"}
    # deeper ancestor scopes tighter
    assert set(j.nodes_with_tag("health:condition:fatigue")) == {"B"}
    # non-hierarchical is exact-only
    assert j.nodes_with_tag("health", hierarchical=False) == []
    assert set(j.nodes_with_tag("fact")) == {"A"}


def test_tag_glossary_depth_ordered():
    j = Journal()
    j.log_tags("A", ["health:condition:macrodactyly"])
    j.log_tags("B", ["health"])
    j.log_tags("C", ["health:condition"])
    gloss = j.tag_glossary()
    # shallowest first (by ':' count), then alpha
    assert gloss == ["health", "health:condition", "health:condition:macrodactyly"]


def test_log_tags_idempotent():
    j = Journal()
    j.log_tags("A", ["x:y"])
    j.log_tags("A", ["x:y"])                      # duplicate -> ignored
    assert j.nodes_with_tag("x:y") == ["A"]


# -- Memory integration ------------------------------------------------------

def test_memory_reindex_and_tag_scoped():
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    a = m.ingest("swollen finger", kind="FACT", id="A")
    b = m.ingest("chronic tiredness", kind="FACT", id="B")
    a.tags.append("health:condition:macrodactyly")
    b.tags.append("health:condition:fatigue")
    n = m.reindex_tags()
    assert n >= 2
    scoped = {p.id for p in m.tag_scoped("health")}    # ancestor match via SQL
    assert scoped == {"A", "B"}
    assert "health:condition:fatigue" in m.tag_glossary_sql()


def test_memory_tag_index_noop_without_journal():
    m = Memory(encoder=SimpleEncoder())
    m.ingest("x", kind="FACT", id="A")
    assert m.reindex_tags() == 0
    assert m.tag_scoped("health") == []
    assert m.tag_glossary_sql() == []
