"""
The emergent-tool contract, end to end, over the EXTERNAL MCP surface only :
a task-specific workflow ("ingest a repo commit by commit into the wiki") is
NOT a hardcoded primitive — the agent creates it with `ensure_tool`, it is
auto-registered in the wiki, the agent executes it with the canonical
primitives (ingest+tags, feed_wiki, okf_proposals/vet_okf_type, wiki_where,
check_wiki), reuses it via `match_tool`, and the autonomic sleep (off the
surface) promotes it once it earned its uses.
"""

from __future__ import annotations

import asyncio
import json

from metacog.defaults import SimpleEncoder
from metacog.journal import Journal
from metacog.memory import Memory


class _LLM:                      # no `generate` : deterministic tool synthesis
    def extract_common(self, texts):
        return ""


async def _call(s, name, **kw):
    r = await s.call_tool(name, kw)
    txt = "".join(c.text for c in r.content) or "null"
    return json.loads(txt)


_COMMITS = [
    ("c1", "wiki: reversible merge ledger", ["metacog/journal.py", "tests/test_x.py"]),
    ("c2", "hooks: session capture", ["hooks/capture_session.py"]),
    ("c3", "docs: plugin section", ["README.md", "metacog/README.md"]),
    ("c4", "mcp: gap sentinel in retrieve", ["metacog/mcp_server.py"]),
]


def test_agent_builds_the_commit_wiki_workflow_from_primitives():
    async def go():
        from mcp.shared.memory import create_connected_server_and_client_session
        from metacog.mcp_server import build_app
        mem = Memory(encoder=SimpleEncoder(), journal=Journal(), llm=_LLM())
        app = build_app(memory=mem, surface="external")
        need = ("ingest a git repo commit by commit, tag by touched modules, "
                "feed one wiki doc per commit")
        async with create_connected_server_and_client_session(app) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            assert "feed_wiki" in names and "ensure_tool" in names
            assert "sleep" not in names                    # autonomic, off-surface
            # 1. nothing covers the need -> the agent creates the tool
            assert await _call(s, "match_tool", query=need) is None
            t = await _call(s, "ensure_tool", query=need,
                            how="ingest(tags=[module:..]) then feed_wiki(type=commit)")
            tid = t["tool"]["id"]
            assert t["reused"] is False and t["tool"]["status"] == "proposed"
            assert (await _call(s, "wiki_where", key="type", value="tool"))["docs"] == [f"tool:{tid}"]
            # 2. the recipe, with primitives only
            issues = []
            for sha, subject, files in _COMMITS:
                mods = sorted({f.split("/")[0] for f in files})
                r = await _call(s, "ingest", content=subject, kind="FACT", id=f"commit:{sha}",
                                tags=[f"module:{m}" for m in mods] + [f"file:{f}" for f in files])
                # tags landed on the node (add_tag normalises to lowercase)
                assert "module:" + mods[0].lower() in r["tags"]
                fw = await _call(s, "feed_wiki", doc_id=f"commit:{sha}", title=subject,
                                 node_ids=[f"commit:{sha}"], type="commit")
                issues += fw["issues"]
            assert {i["reason"] for i in issues} == {"type_proposed"}
            assert [p["value"] for p in (await _call(s, "okf_proposals"))["proposals"]] == ["commit"]
            v = await _call(s, "vet_okf_type", type="commit", accept=True)
            assert v["status"] == "accepted" and len(v["docs"]) == 4
            # cross-cutting queries over the wiki
            q = await _call(s, "wiki_where", key="tags", value="module:metacog")
            assert set(q["docs"]) == {"commit:c1", "commit:c3", "commit:c4"}
            assert (await _call(s, "wiki_where", key="tags", value="file:readme.md"))["docs"] == ["commit:c3"]
            assert "commit" in await _call(s, "okf_schema")
            chk = await _call(s, "check_wiki")
            assert not any(v["code"].startswith("type_") for v in chk["violations"])
            doc = (await _call(s, "wiki_doc", doc_id="commit:c2"))["okf"]
            assert "type: commit" in doc and "[[commit:c2]]" in doc and "#module:hooks" in doc
            assert (await _call(s, "docs_for_node", node_id="commit:c2"))["docs"] == ["commit:c2"]
            # 3. reuse, no LLM ; feedback
            m = await _call(s, "match_tool", query=need)
            assert m and m["id"] == tid
            assert (await _call(s, "ensure_tool", query=need))["reused"] is True
            for _ in range(mem.tool_promote_after):
                await _call(s, "report_tool", tool_id=tid, ok=True)
            assert (await _call(s, "wiki_where", key="status", value="proposed"))["docs"] == [f"tool:{tid}"]
        # 4. the autonomic pass (SessionEnd hook / scheduler) promotes it
        out = mem.sleep()
        assert out.get("tools_promoted") == [tid]
        assert mem.wiki_where("status", "established") == [f"tool:{tid}"]
        ev = [e["event"] for e in mem.tool_history(tid)]
        assert ev[:2] == ["created", "reused"] and "promoted" in ev
    asyncio.run(go())


def test_mcp_ingest_tags_are_indexed_in_the_journal():
    async def go():
        from mcp.shared.memory import create_connected_server_and_client_session
        from metacog.mcp_server import build_app
        mem = Memory(encoder=SimpleEncoder(), journal=Journal())
        async with create_connected_server_and_client_session(build_app(memory=mem)) as s:
            await s.initialize()
            r = await _call(s, "ingest", content="x", kind="FACT", id="N",
                            tags=["module:metacog", "file:a.py"])
            assert "module:metacog" in r["tags"]
        assert [p.id for p in mem.tag_scoped("module")] == ["N"]     # SQL tag index, hierarchical
    asyncio.run(go())
