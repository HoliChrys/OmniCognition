"""
The Claude Code plugin layer : gap sentinel in-band + the four hooks.

  - `retrieve` / `walk_start` append the GAP sentinel when no chunk is
    sufficiently activated (the ACT-R threshold), and stay silent otherwise.
  - hooks/recall_gap.py   (PostToolUse) injects the grounding directive ONLY on
                          a sentinel ; silent otherwise ; never crashes.
  - hooks/session_start.py (SessionStart) prints the memory discipline + brain.
  - hooks/capture_session.py (SessionEnd) feeds the user's typed messages into
                          the brain as episodic turns, dedups against what the
                          session indexed live, sleeps, saves ; idempotent.
  - hooks/auto_recall.py  (UserPromptSubmit) is OFF unless METACOG_AUTO_RECALL.
  - manifests : plugin.json / hooks.json / marketplace.json are consistent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory
from metacog.mcp_server import GAP_SENTINEL

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"


def _hook(name: str, payload: dict, env: dict | None = None) -> str:
    """Run a hook script the way Claude Code does : JSON on stdin, stdout back."""
    e = {**os.environ, **(env or {})}
    e.pop("METACOG_AUTO_RECALL", None) if not (env and "METACOG_AUTO_RECALL" in env) else None
    r = subprocess.run([sys.executable, str(HOOKS / name)], input=json.dumps(payload),
                       capture_output=True, text=True, env=e, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _corpus() -> Memory:
    m = Memory(encoder=SimpleEncoder())
    m.ingest("alpha beta gamma delta", kind="FACT", id="A")
    m.ingest("kitchen recipe soup carrots", kind="FACT", id="B")
    m.ingest("quarterly finance report revenue", kind="FACT", id="C")
    m.ingest("hiking trail mountain weather", kind="FACT", id="D")
    m.ingest("guitar chords music lesson", kind="FACT", id="E")
    return m


# -- sentinel in-band ---------------------------------------------------------

def test_retrieve_appends_sentinel_only_on_a_gap():
    import asyncio
    from mcp.shared.memory import create_connected_server_and_client_session
    from metacog.mcp_server import build_app

    async def go():
        m = _corpus()
        assert m.abstains("zzz qqq www") and not m.abstains("alpha beta gamma")
        async with create_connected_server_and_client_session(build_app(memory=m)) as s:
            await s.initialize()
            r = await s.call_tool("retrieve", {"query": "zzz qqq www", "k": 3})
            txt = "".join(c.text for c in r.content)
            assert GAP_SENTINEL in txt
            r = await s.call_tool("retrieve", {"query": "alpha beta gamma", "k": 3})
            txt = "".join(c.text for c in r.content)
            assert GAP_SENTINEL not in txt and '"A"' in txt
    asyncio.run(go())


# -- PostToolUse : recall_gap -------------------------------------------------

def test_recall_gap_hook_injects_directive_only_on_sentinel():
    out = _hook("recall_gap.py", {
        "hook_event_name": "PostToolUse", "tool_name": "mcp__plugin_metacog_metacog__retrieve",
        "tool_response": [{"type": "text", "text": json.dumps(
            [{"id": "A"}, {"gap": True, "sentinel": GAP_SENTINEL}], ensure_ascii=False)}],
    })
    j = json.loads(out)
    assert j["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert "GAP" in j["hookSpecificOutput"]["additionalContext"]
    assert "ingest" in j["hookSpecificOutput"]["additionalContext"]
    # no sentinel -> silent ; garbage on stdin -> silent, exit 0
    assert _hook("recall_gap.py", {"tool_response": {"content": [{"text": "[{\"id\":\"A\"}]"}]}}) == ""
    r = subprocess.run([sys.executable, str(HOOKS / "recall_gap.py")], input="not json",
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout == ""


# -- SessionStart -------------------------------------------------------------

def test_session_start_prints_the_discipline_with_the_brain():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / ".metacog-brain").write_text("# project brain\n./mem/brain.pkl\n")
        out = _hook("session_start.py", {"hook_event_name": "SessionStart", "cwd": d})
    assert "RECALL BEFORE ANSWERING" in out and GAP_SENTINEL in out
    assert "./mem/brain.pkl" in out                      # the marker won
    assert "forget(" in out and "mark_useful" in out and "ensure_tool" in out


def test_storage_resolution_marker_env_default(monkeypatch):
    sys.path.insert(0, str(HOOKS))
    import _common as C
    with tempfile.TemporaryDirectory() as d:
        sub = Path(d) / "a" / "b"
        sub.mkdir(parents=True)
        monkeypatch.delenv("METACOG_STORAGE", raising=False)
        assert C.resolve_storage(str(sub)) == os.path.expanduser(C.DEFAULT_STORAGE)
        monkeypatch.setenv("METACOG_STORAGE", "/tmp/x.pkl")
        assert C.resolve_storage(str(sub)) == "/tmp/x.pkl"
        (Path(d) / ".metacog-brain").write_text("~/proj.pkl\n")   # walked up
        assert C.resolve_storage(str(sub)) == os.path.expanduser("~/proj.pkl")


# -- SessionEnd : capture_session --------------------------------------------

def _transcript(path: Path, session_id: str) -> None:
    rows = [
        {"type": "user", "timestamp": "2026-09-04T10:00:00Z",
         "message": {"role": "user", "content": "I prefer tabs over spaces in this repo"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "noted"}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "x", "content": "ok"}]}},      # tool turn
        {"type": "user", "message": {"role": "user", "content": "<command-name>/clear</command-name>"}},
        {"type": "user", "timestamp": "2026-09-04T10:05:00Z",
         "message": {"role": "user", "content": [{"type": "text", "text": "deploy target is the staging cluster"}]}},
        {"type": "user", "message": {"role": "user",
                                     "content": "<system-reminder>ignore me</system-reminder>"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_user_messages_parser_keeps_only_typed_messages():
    sys.path.insert(0, str(HOOKS))
    import _common as C
    with tempfile.TemporaryDirectory() as d:
        tp = Path(d) / "t.jsonl"
        _transcript(tp, "s1")
        msgs = C.user_messages(str(tp))
    assert [m["content"] for m in msgs] == [
        "I prefer tabs over spaces in this repo", "deploy target is the staging cluster"]
    assert msgs[0]["timestamp"] == "2026-09-04T10:00:00Z"


def test_capture_session_feeds_brain_dedups_sleeps_and_saves():
    with tempfile.TemporaryDirectory() as d:
        tp = Path(d) / "t.jsonl"
        _transcript(tp, "s1")
        brain = str(Path(d) / "mem" / "brain.pkl")
        # the session already indexed the FIRST message live (via the server)
        m = Memory(storage_path=brain, journal_path="auto")
        os.makedirs(os.path.dirname(brain), exist_ok=True)
        m.ingest_message("I prefer tabs over spaces in this repo", role="user",
                         user_id="u", session_id="s1", block=True)
        m.save()
        payload = {"hook_event_name": "SessionEnd", "session_id": "s1",
                   "transcript_path": str(tp), "cwd": d, "reason": "other"}
        r = subprocess.run([sys.executable, str(HOOKS / "capture_session.py")],
                           input=json.dumps(payload), capture_output=True, text=True,
                           env={**os.environ, "METACOG_STORAGE": brain, "METACOG_USER": "u"},
                           timeout=180)
        assert r.returncode == 0, r.stderr
        assert "captured 1 user message" in r.stderr            # only the new one
        m2 = Memory(storage_path=brain, journal_path="auto")
        turns = [p for p in m2.points if "session:s1" in p.tags and "role:user" in p.tags]
        assert len(turns) == 2
        assert any("staging cluster" in p.content for p in turns)
        assert any("[2026-09-04T10:05:00Z]" in p.content for p in turns)  # ts kept
        # idempotent : a second run captures nothing new
        r = subprocess.run([sys.executable, str(HOOKS / "capture_session.py")],
                           input=json.dumps(payload), capture_output=True, text=True,
                           env={**os.environ, "METACOG_STORAGE": brain, "METACOG_USER": "u"},
                           timeout=180)
        assert r.returncode == 0 and "captured" not in r.stderr
        m3 = Memory(storage_path=brain, journal_path="auto")
        assert len([p for p in m3.points if "session:s1" in p.tags]) == 2
        # a missing transcript is a silent no-op
        r = subprocess.run([sys.executable, str(HOOKS / "capture_session.py")],
                           input=json.dumps({**payload, "transcript_path": "/nope"}),
                           capture_output=True, text=True, env={**os.environ, "METACOG_STORAGE": brain})
        assert r.returncode == 0 and r.stdout == ""


def test_capture_session_dump_mode():
    with tempfile.TemporaryDirectory() as d:
        tp = Path(d) / "t.jsonl"
        _transcript(tp, "s1")
        r = subprocess.run([sys.executable, str(HOOKS / "capture_session.py"), "--dump", str(tp)],
                           capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.count("•") == 2


# -- UserPromptSubmit : auto_recall (opt-in) ----------------------------------

def test_auto_recall_is_off_by_default_and_recalls_when_enabled():
    with tempfile.TemporaryDirectory() as d:
        brain = str(Path(d) / "brain.pkl")
        m = _corpus()
        m.storage_path = brain
        m.save()
        payload = {"hook_event_name": "UserPromptSubmit", "cwd": d,
                   "prompt": "tell me about alpha beta gamma"}
        env = {"METACOG_STORAGE": brain}
        assert _hook("auto_recall.py", payload, env) == ""                 # off
        out = _hook("auto_recall.py", payload, {**env, "METACOG_AUTO_RECALL": "1"})
        j = json.loads(out)
        assert j["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert "[A]" in j["hookSpecificOutput"]["additionalContext"]
        # a gap injects nothing (silence beats noise)
        out = _hook("auto_recall.py", {**payload, "prompt": "zzz qqq www yyy"},
                    {**env, "METACOG_AUTO_RECALL": "1"})
        assert out == ""


# -- manifests ----------------------------------------------------------------

def test_plugin_manifests_are_consistent():
    plug = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    hooks = json.loads((ROOT / "hooks" / "hooks.json").read_text())
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    assert plug["name"] == "metacog"
    assert (ROOT / plug["hooks"]).is_file()
    srv = plug["mcpServers"]["metacog"]
    assert srv["args"][0].endswith("bin/metacog-mcp.sh")
    assert os.access(ROOT / "bin" / "metacog-mcp.sh", os.X_OK)
    # every hook script referenced exists
    for ev, entries in hooks["hooks"].items():
        for e in entries:
            for h in e["hooks"]:
                script = h["command"].split('"')[1].replace("${CLAUDE_PLUGIN_ROOT}", str(ROOT))
                assert Path(script).is_file(), (ev, script)
    assert set(hooks["hooks"]) == {"SessionStart", "UserPromptSubmit", "PostToolUse", "SessionEnd"}
    # the PostToolUse matcher covers both the plugin prefix and a project .mcp.json
    import re
    pat = re.compile(hooks["hooks"]["PostToolUse"][0]["matcher"])
    assert pat.fullmatch("mcp__plugin_metacog_metacog__retrieve")
    assert pat.fullmatch("mcp__metacog__walk_start") and not pat.fullmatch("mcp__metacog__ingest")
    assert market["plugins"][0]["name"] == plug["name"]
    assert market["plugins"][0]["version"] == plug["version"]


def test_launcher_resolves_brain_and_surface():
    """The launcher must pick the marker brain and default to the external
    surface — checked by replacing python with a printer."""
    with tempfile.TemporaryDirectory() as d:
        proj = Path(d) / "proj" / "sub"
        proj.mkdir(parents=True)
        (Path(d) / "proj" / ".metacog-brain").write_text("~/team.pkl\n")
        fake = Path(d) / "fakepy"
        fake.write_text("#!/bin/sh\necho \"$@\"; echo SURFACE=$METACOG_SURFACE; echo PP=$PYTHONPATH\n")
        fake.chmod(0o755)
        r = subprocess.run(["bash", str(ROOT / "bin" / "metacog-mcp.sh")], cwd=str(proj),
                           capture_output=True, text=True,
                           env={**os.environ, "METACOG_PYTHON": str(fake),
                                "CLAUDE_PLUGIN_ROOT": str(ROOT)})
        assert r.returncode == 0, r.stderr
        assert f"--storage {os.path.expanduser('~/team.pkl')}" in r.stdout
        assert "SURFACE=external" in r.stdout and f"PP={ROOT}" in r.stdout
