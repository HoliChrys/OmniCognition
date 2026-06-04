"""Episodic continuous indexation (async, non-blocking, timestamped) and
push_code (project-doc / tool routing with bidirectional ref: filiation).
Deterministic — fake/marker LLM, no live API.
"""
from metacog.memory import Memory
from metacog.defaults import SimpleEncoder
from metacog.keywords import SimpleKeywordExtractor
from metacog.epistemic import PointKind, SourceClass


def _mem():
    return Memory(encoder=SimpleEncoder(), extractor=SimpleKeywordExtractor())


# ---------------------------------------------------------------------------
# Episodic message index
# ---------------------------------------------------------------------------

def test_message_indexed_with_timestamp_and_role():
    m = _mem()
    m.ingest_message("hello", role="user", user_id="u1", session_id="s1",
                     timestamp="2026-06-04T10:00:00", block=True)
    p = next(p for p in m.points if "message" in p.tags)
    tags = [t.lower() for t in p.tags]
    assert "episode" in tags and "message" in tags
    assert "role:user" in tags
    assert "user:u1" in tags and "session:s1" in tags
    assert any(t.startswith("ts:2026-06-04t10:00:00") for t in tags)
    assert any(t.startswith("date:2026-06-04") for t in tags)
    # timestamp preserved verbatim in the content anchor
    assert p.content.startswith("[2026-06-04T10:00:00] user:")


def test_both_sides_chain_in_order():
    m = _mem()
    m.ingest_message("q", role="user", user_id="u", session_id="s",
                     timestamp="2026-06-04T10:00:00", block=True)
    m.ingest_message("a", role="agent", user_id="u", session_id="s",
                     timestamp="2026-06-04T10:00:05", block=True)
    msgs = [p for p in m.points if "message" in p.tags]
    assert len(msgs) == 2
    roles = {next(t for t in p.tags if t.startswith("role:")) for p in msgs}
    assert roles == {"role:user", "role:agent"}
    agent = next(p for p in msgs if "role:agent" in p.tags)
    user = next(p for p in msgs if "role:user" in p.tags)
    assert agent.sequence_prev == user.id          # chained in order


def test_async_indexation_is_non_blocking_then_flushable():
    m = _mem()
    # async: returns immediately with a queued ack, point not yet present
    ack = m.ingest_message("hi", role="user", user_id="u", session_id="s",
                           timestamp="2026-06-04T10:00:00")
    assert ack == {"queued": True}
    m.flush_index()                                 # wait for the worker
    assert any("message" in p.tags for p in m.points)


# ---------------------------------------------------------------------------
# push_code routing
# ---------------------------------------------------------------------------

CODE = "def run(args: dict):\n    return sorted(args['items'])\n"


def test_project_code_creates_doc_and_tool_with_bidirectional_ref():
    m = _mem()  # no LLM → fallback: project from metadata, tool from markers
    r = m.push_code(CODE, context="sort items", project="omnicog",
                    branch="main", github_user="marcc",
                    user_id="u1", session_id="s1")
    assert r["pushed"] and r["is_project"] and r["is_tool"]
    doc = next(p for p in m.points if p.id == r["doc_id"])
    tool = next(p for p in m.points if p.id == r["tool_id"])
    # doc = semantic FACT documentation, project-tagged
    assert doc.kind == PointKind.FACT
    for t in ("doc", "project:omnicog", "branch:main", "github:marcc"):
        assert t in doc.tags
    # tool = executable ACTION
    assert tool.kind == PointKind.ACTION and tool.exec_spec
    assert "project:omnicog" in tool.tags
    # bidirectional ref: filiation
    assert any(k == f"ref:tool:{tool.id}" for k in doc.keywords)
    assert any(k == f"ref:doc:{doc.id}" for k in tool.keywords)
    assert tool.parents == [doc.id]
    assert doc.keywords_source == SourceClass.GENERATOR


def test_plain_tool_code_no_project_creates_tool_only():
    m = _mem()
    r = m.push_code(CODE, context="sort items")   # no project metadata
    # marker fallback: tool yes, project no → tool only
    assert r["is_tool"] and not r["is_project"]
    assert r["tool_id"] and r["doc_id"] is None


def test_push_empty_code_is_noop():
    m = _mem()
    assert m.push_code("   ") == {"pushed": False}


def test_mcp_tools_registered():
    import asyncio
    from metacog.mcp_server import build_app
    names = [t.name for t in asyncio.run(build_app(memory=_mem()).list_tools())]
    assert {"ingest_message", "push_code"}.issubset(set(names))
