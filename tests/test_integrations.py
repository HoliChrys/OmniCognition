"""
Host integrations : the shared bridge, the Hermes memory provider, the OpenClaw
hook pack, and the install/uninstall commands.

What is actually verified here is everything WE control — the bridge contract,
the provider's behaviour against Hermes' documented interface, the handler's
event→action mapping and its spawn, and the installer's writes (idempotent,
reversible, and never touching a neighbour's entry). Running inside a live
Hermes or OpenClaw is not simulated ; the adapters are written against their
documented APIs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "hooks" / "host_bridge.py"
HANDLER = ROOT / "integrations" / "openclaw" / "hooks" / "metacog-memory" / "handler.js"
NODE = shutil.which("node")


#: An episodic node's content carries its timestamp, so an unpinned `ts` makes
#: the hash encoder's vectors — and therefore the ACT-R abstention decision —
#: differ run to run. Every fixture here pins it.
TS = "2026-01-01T00:00:00Z"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """These tests drive the resolution env ; never leak it into the suite."""
    for k in ("METACOG_STORAGE", "METACOG_ENCODER", "METACOG_USER", "METACOG_ROOT",
              "METACOG_PYTHON", "METACOG_AUTO_RECALL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("METACOG_ENCODER", "simple")


def _env(brain: Path, **extra) -> dict:
    return {**os.environ, "METACOG_STORAGE": str(brain), "METACOG_ENCODER": "simple",
            "METACOG_USER": "u", **extra}


def _bridge(args, brain: Path, stdin: str = "") -> dict:
    r = subprocess.run([sys.executable, str(BRIDGE), "--json", *args],
                       input=stdin, capture_output=True, text=True,
                       env=_env(brain), timeout=180)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().split("\n")[-1])


# -- the shared bridge --------------------------------------------------------

def test_bridge_feed_dedups_and_status_reports():
    with tempfile.TemporaryDirectory() as d:
        brain = Path(d) / "mem.pkl"
        assert _bridge(["status"], brain)["exists"] is False
        out = _bridge(["feed", "--role", "user", "--session", "s1"], brain,
                      stdin="deploy target is the staging cluster")
        assert out["indexed"] is True and out["role"] == "user"
        again = _bridge(["feed", "--role", "user", "--session", "s1"], brain,
                        stdin="deploy target is the staging cluster")
        assert again["indexed"] is False and again["reason"] == "already_indexed"
        assert _bridge(["feed", "--role", "agent", "--session", "s1",
                        "--text", "noted, staging it is"], brain)["indexed"] is True
        assert _bridge(["feed", "--role", "user", "--session", "s1"], brain,
                       stdin="   ")["reason"] == "empty"
        st = _bridge(["status"], brain)
        assert st["exists"] is True and st["points"] == 2


def test_bridge_recall_gap_and_hits():
    with tempfile.TemporaryDirectory() as d:
        brain = Path(d) / "mem.pkl"
        for text in ("alpha beta gamma delta", "kitchen soup carrots",
                     "quarterly finance revenue", "hiking mountain trail",
                     "guitar chords lesson"):
            _bridge(["feed", "--role", "user", "--session", "s", "--ts", TS,
                     "--text", text], brain)
        hit = _bridge(["recall", "--query", "alpha beta gamma", "--k", "2"], brain)
        assert hit["gap"] is False and hit["hits"] and "alpha" in hit["text"]
        gap = _bridge(["recall", "--query", "zzz qqq www yyy"], brain)
        assert gap["gap"] is True and gap["text"] == ""
        forced = _bridge(["recall", "--query", "zzz qqq www yyy", "--gap-notice"], brain)
        assert "NO RELEVANT MEMORY" in forced["text"] and "ground first" in forced["text"]
        short = _bridge(["recall", "--query", "hi"], brain)
        assert short["skipped"] == "query_too_short"


def test_bridge_consolidate_and_never_raises():
    with tempfile.TemporaryDirectory() as d:
        brain = Path(d) / "mem.pkl"
        assert _bridge(["consolidate"], brain)["slept"] is False   # no brain yet
        _bridge(["feed", "--role", "user", "--session", "s", "--text", "a fact"], brain)
        out = _bridge(["consolidate"], brain)
        assert out["slept"] is True and "consolidated" in out["text"]
        # a corrupt brain must be reported, not raised
        Path(brain).write_text("not a pickle")
        r = subprocess.run([sys.executable, str(BRIDGE), "--json", "status"],
                           capture_output=True, text=True, env=_env(brain), timeout=60)
        assert r.returncode == 0 and "error" in r.stdout


# -- the Hermes memory provider ----------------------------------------------

def _provider(brain: Path, **cfg):
    spec = importlib.util.spec_from_file_location(
        "metacog_hermes", ROOT / "integrations" / "hermes" / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod.MetacogMemory(storage=str(brain), user_id="u", **cfg)


def test_hermes_provider_lifecycle_prefetch_sync_and_consolidation():
    with tempfile.TemporaryDirectory() as d:
        brain = Path(d) / "mem.pkl"
        mod, p = _provider(brain)
        assert p.name == "metacog" and p.is_available() is True
        p.initialize("sess-1")
        assert p.prefetch("anything at all") == ""            # empty brain, no noise
        p.sync_turn("the deploy target is the staging cluster",
                    "understood, staging it is", session_id="sess-1")
        for extra in ("kitchen soup carrots", "quarterly finance revenue",
                      "hiking mountain trail"):
            p.sync_turn(extra, "ok", session_id="sess-1")
        got = p.prefetch("what is the deploy target", session_id="sess-1")
        assert "staging cluster" in got and "metacog memory" in got
        assert p.prefetch("hi") == ""                         # too short
        # gap → the grounding directive, never silence. Forced deterministically:
        # whether a hash-encoded corpus abstains on a given string is a property
        # of the encoder, not of this adapter.
        mem = p._memory()
        mem.abstains = lambda q, threshold=None: True
        assert "NO RELEVANT MEMORY" in p.prefetch("zzz qqq www yyy")
        mem.abstains = lambda q, threshold=None: False
        # a turn is never indexed twice
        before = len(p._memory().points)
        p.sync_turn("the deploy target is the staging cluster",
                    "understood, staging it is", session_id="sess-1")
        assert len(p._memory().points) == before
        # compaction stores what is about to be lost
        msg = [{"role": "user", "content": "we ship on fridays"},
               {"role": "assistant", "content": [{"type": "text", "text": "noted"}]},
               {"role": "tool", "content": "ignored"}]
        note = p.on_pre_compress(msg)
        assert "stored 2 turn(s)" in note
        assert any("we ship on fridays" in q.content for q in p._memory().points)
        p.on_session_end(msg)                                  # capture + one sleep
        st = p.status()
        assert st["provider"] == "metacog" and st["points"] >= 6 and st["error"] is None
        p.on_session_switch("sess-2")
        assert p.session_id == "sess-2"
        p.shutdown()
        # everything survived the process
        _, p2 = _provider(brain)
        assert len(p2._memory().points) == st["points"]


def test_hermes_provider_tools():
    with tempfile.TemporaryDirectory() as d:
        brain = Path(d) / "mem.pkl"
        _, p = _provider(brain)
        p.initialize("s")
        names = {t["function"]["name"] for t in p.get_tool_schemas()}
        assert names == {"metacog_recall", "metacog_walk", "metacog_remember",
                         "metacog_forget", "metacog_mark_useful", "metacog_wiki"}
        for t in p.get_tool_schemas():                       # valid OpenAI schemas
            f = t["function"]
            assert t["type"] == "function" and f["description"]
            assert f["parameters"]["type"] == "object" and f["parameters"]["required"]
        out = p.handle_tool_call("metacog_remember",
                                 {"content": "the deploy target is staging",
                                  "tags": ["ops:deploy"]})
        assert out.startswith("stored as ")
        nid = out.split("stored as ")[1].strip()
        node = next(q for q in p._memory().points if q.id == nid)
        assert "ops:deploy" in node.tags
        assert "staging" in p.handle_tool_call("metacog_recall", {"query": "deploy target"})
        w = json.loads(p.handle_tool_call("metacog_wiki",
                                          {"doc_id": "doc:d", "title": "Deploy",
                                           "node_ids": [nid], "type": "runbook"}))
        assert w["refs"] == [nid]
        f = json.loads(p.handle_tool_call("metacog_forget",
                                          {"node_id": nid, "reason": "superseded"}))
        assert f["forgotten"] == nid
        assert p.handle_tool_call("nope", {}).startswith("unknown tool")
        assert "metacog error" in p.handle_tool_call("metacog_mark_useful",
                                                     {"retrieval_id": 999, "score": 9})


def test_hermes_provider_degrades_instead_of_breaking_the_agent():
    """An unopenable brain must mean no memory, never a broken turn."""
    with tempfile.TemporaryDirectory() as d:
        brain = Path(d) / "mem.pkl"
        brain.write_text("not a pickle")
        _, p = _provider(brain)
        assert p.is_available() is False and p._error
        p.initialize("s")                                    # does not raise
        assert p.prefetch("what is the deploy target") == ""
        p.sync_turn("a", "b")                                # swallowed
        p.on_session_end([])
        assert "metacog error" in p.handle_tool_call("metacog_recall", {"query": "x"})
        assert "error" in p.status()


def test_hermes_register_adds_the_status_command(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        mod, _ = _provider(Path(d) / "mem.pkl")
        seen = {}

        class _Ctx:
            def register_command(self, name, handler, description):
                seen[name] = (handler, description)

        monkeypatch.setenv("METACOG_STORAGE", str(Path(d) / "mem.pkl"))
        mod.register(_Ctx())
        assert "metacog" in seen
        assert "brain" in seen["metacog"][0]()
        mod.register(object())                               # older ctx : no crash


# -- the OpenClaw hook pack ---------------------------------------------------

def _node_eval(script: str, env: dict | None = None) -> str:
    r = subprocess.run([NODE, "--input-type=module", "-e", script],
                       capture_output=True, text=True, timeout=120,
                       env={**os.environ, **(env or {})}, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_openclaw_handler_maps_events_to_bridge_calls():
    out = _node_eval(f"""
      import {{ planFor, pickText }} from "file://{HANDLER}";
      const plans = {{
        received: planFor({{type:"message", action:"received", sessionKey:"s1",
                            context:{{text:"hello there"}}}}),
        blocks:   planFor({{type:"message", action:"received", sessionKey:"s1",
                            context:{{message:{{body:"nested body"}}}}}}),
        sent:     planFor({{type:"message", action:"sent", sessionKey:"s1",
                            context:{{text:"a reply", success:true}}}}),
        failed:   planFor({{type:"message", action:"sent", sessionKey:"s1",
                            context:{{text:"a reply", success:false}}}}),
        empty:    planFor({{type:"message", action:"received", sessionKey:"s1",
                            context:{{}}}}),
        newcmd:   planFor({{type:"command", action:"new", sessionKey:"s1", context:{{}}}}),
        compact:  planFor({{type:"session", action:"compact:before", sessionKey:"s1",
                            context:{{}}}}),
        shutdown: planFor({{type:"gateway", action:"shutdown", sessionKey:"s1", context:{{}}}}),
        other:    planFor({{type:"session", action:"patch", sessionKey:"s1", context:{{}}}}),
      }};
      console.log(JSON.stringify(plans));
    """)
    p = json.loads(out)
    assert p["received"]["kind"] == "feed" and p["received"]["stdin"] == "hello there"
    assert "--role" in p["received"]["args"] and "user" in p["received"]["args"]
    assert p["blocks"]["stdin"] == "nested body"           # probes nested shapes
    assert p["sent"]["kind"] == "feed" and "agent" in p["sent"]["args"]
    assert p["failed"] is None                              # undelivered is not a turn
    assert p["empty"] is None and p["other"] is None
    for key in ("newcmd", "compact", "shutdown"):
        assert p[key]["kind"] == "consolidate"
    assert p["newcmd"]["notify"] is True and p["shutdown"]["notify"] is False


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_openclaw_handler_spawns_the_bridge_and_delivers_the_notice():
    with tempfile.TemporaryDirectory() as d:
        rec = Path(d) / "calls.txt"
        fake = Path(d) / "fakepy"
        fake.write_text(
            "#!/bin/sh\n"
            f'{{ echo "ARGV: $@"; echo "STDIN: $(cat)"; }} >> {rec}\n'
            '''echo '{"slept": true, "text": "🧠 metacog: memory consolidated"}'\n''')
        fake.chmod(0o755)
        out = _node_eval(f"""
          import handler from "file://{HANDLER}";
          const ev = {{type:"command", action:"new", sessionKey:"s9",
                       context:{{cwd:"{d}"}}, messages:[]}};
          await handler(ev);
          const feed = {{type:"message", action:"received", sessionKey:"s9",
                         context:{{cwd:"{d}", text:"remember this"}}, messages:[]}};
          await handler(feed);
          console.log(JSON.stringify({{delivered: ev.messages, feedMsgs: feed.messages}}));
        """, env={"METACOG_PYTHON": str(fake)})
        got = json.loads(out)
        assert got["delivered"] == ["🧠 metacog: memory consolidated"]
        assert got["feedMsgs"] == []                        # feeding says nothing
        calls = rec.read_text()
        assert "host_bridge.py --json --cwd" in calls and "consolidate" in calls
        assert "feed --role user --session s9" in calls and "STDIN: remember this" in calls


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_openclaw_handler_never_throws_when_the_bridge_is_broken():
    out = _node_eval(f"""
      import handler from "file://{HANDLER}";
      const ev = {{type:"command", action:"new", sessionKey:"s", context:{{}}, messages:[]}};
      await handler(ev);
      console.log(JSON.stringify(ev.messages));
    """, env={"METACOG_PYTHON": "/nonexistent/python"})
    assert json.loads(out) == []


def test_openclaw_bundle_manifests_are_consistent():
    base = ROOT / "integrations" / "openclaw"
    pkg = json.loads((base / "package.json").read_text())
    assert pkg["type"] == "module" and pkg["openclaw"]["hooks"] == ["./hooks/metacog-memory"]
    for rel in pkg["openclaw"]["hooks"]:
        assert (base / rel / "HOOK.md").is_file()
        assert (base / rel / "handler.js").is_file()
    front = (base / "hooks" / "metacog-memory" / "HOOK.md").read_text().split("---")[1]
    import yaml
    meta = yaml.safe_load(front)["metadata"]["openclaw"]
    assert set(meta["events"]) == {"message:received", "message:sent", "command:new",
                                   "command:reset", "session:compact:before",
                                   "gateway:shutdown"}
    assert meta["requires"]["bins"] == ["python3"]
    mcp = json.loads((base / ".mcp.json").read_text())
    assert "PYTHONPATH" not in json.dumps(mcp)        # OpenClaw rejects such env keys


# -- install / uninstall ------------------------------------------------------

@pytest.fixture()
def fake_home(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("HOME", d)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("OPENCLAW_HOME", raising=False)
        yield Path(d)


def _install(*argv) -> int:
    from metacog import install as I
    importlib.reload(I)
    return I.main(list(argv))


def test_install_claude_is_idempotent_and_reversible(fake_home, capsys):
    from metacog import install as I
    assert _install("install", "claude") == 0
    settings = json.loads((fake_home / ".claude" / "settings.json").read_text())
    events = settings["hooks"]
    assert set(events) == {"SessionStart", "UserPromptSubmit", "PostToolUse", "SessionEnd"}
    cmd = events["SessionEnd"][0]["hooks"][0]["command"]
    assert "capture_session.py" in cmd and str(ROOT) in cmd
    assert events["PostToolUse"][0]["matcher"].startswith("mcp__")
    mcp = json.loads((fake_home / ".claude.json").read_text())
    assert mcp["mcpServers"]["metacog"]["args"] == [str(ROOT / "bin" / "metacog-mcp.sh")]
    # a neighbour's hook and MCP server must survive install AND uninstall
    settings["hooks"]["SessionEnd"].append(
        {"hooks": [{"type": "command", "command": "/other/tool.sh"}]})
    (fake_home / ".claude" / "settings.json").write_text(json.dumps(settings))
    mcp["mcpServers"]["other"] = {"command": "x"}
    (fake_home / ".claude.json").write_text(json.dumps(mcp))
    _install("install", "claude")                              # idempotent
    again = json.loads((fake_home / ".claude" / "settings.json").read_text())
    ours = [e for e in again["hooks"]["SessionEnd"]
            if any(str(ROOT) in h["command"] for h in e["hooks"])]
    assert len(ours) == 1 and len(again["hooks"]["SessionEnd"]) == 2
    assert I.claude_status()["hooks"] == 4 and I.claude_status()["mcp"] is True
    assert _install("uninstall", "claude") == 0
    after = json.loads((fake_home / ".claude" / "settings.json").read_text())
    assert after["hooks"]["SessionEnd"] == [
        {"hooks": [{"type": "command", "command": "/other/tool.sh"}]}]
    left = json.loads((fake_home / ".claude.json").read_text())
    assert "metacog" not in left["mcpServers"] and "other" in left["mcpServers"]
    assert I.claude_status()["hooks"] == 0


def test_install_claude_project_scope(fake_home):
    from metacog import install as I
    with tempfile.TemporaryDirectory() as proj:
        _install("install", "claude", "--scope", "project", "--project", proj)
        assert (Path(proj) / ".claude" / "settings.json").is_file()
        assert "metacog" in json.loads((Path(proj) / ".mcp.json").read_text())["mcpServers"]
        assert not (fake_home / ".claude" / "settings.json").exists()
        _install("uninstall", "claude", "--scope", "project", "--project", proj)
        assert I.claude_status(scope="project", project=proj)["hooks"] == 0


def test_install_hermes_links_the_provider_and_selects_it(fake_home):
    from metacog import install as I
    import yaml
    assert _install("install", "hermes") == 0
    dst = fake_home / ".hermes" / "plugins" / "metacog"
    assert dst.is_symlink() and Path(os.readlink(dst)) == ROOT / "integrations" / "hermes"
    cfg = yaml.safe_load((fake_home / ".hermes" / "config.yaml").read_text())
    assert cfg["plugins"]["enabled"] == ["metacog"] and cfg["memory"]["provider"] == "metacog"
    st = I.hermes_status()
    assert st["linked"] and st["enabled"] and st["provider"] == "metacog"
    _install("install", "hermes")                              # idempotent
    cfg = yaml.safe_load((fake_home / ".hermes" / "config.yaml").read_text())
    assert cfg["plugins"]["enabled"] == ["metacog"]
    assert _install("uninstall", "hermes") == 0
    assert not dst.exists()
    cfg = yaml.safe_load((fake_home / ".hermes" / "config.yaml").read_text())
    assert cfg["plugins"]["enabled"] == [] and "provider" not in cfg.get("memory", {})


def test_install_openclaw_links_the_hook_and_enables_it(fake_home):
    from metacog import install as I
    assert _install("install", "openclaw") == 0
    dst = fake_home / ".openclaw" / "hooks" / "metacog-memory"
    assert dst.is_symlink() and (dst / "HOOK.md").is_file()
    cfg = json.loads((fake_home / ".openclaw" / "openclaw.json").read_text())
    assert cfg["hooks"]["internal"]["enabled"] is True
    assert cfg["hooks"]["internal"]["entries"]["metacog-memory"]["enabled"] is True
    assert cfg["mcpServers"]["metacog"]["command"] == "bash"
    assert "PYTHONPATH" not in json.dumps(cfg["mcpServers"])
    st = I.openclaw_status()
    assert st["linked"] and st["enabled"] and st["mcp"]
    assert _install("uninstall", "openclaw") == 0
    assert not dst.exists()
    cfg = json.loads((fake_home / ".openclaw" / "openclaw.json").read_text())
    assert "metacog-memory" not in cfg["hooks"]["internal"]["entries"]
    assert "mcpServers" not in cfg


def test_install_all_dry_run_writes_nothing(fake_home, capsys):
    assert _install("install", "all", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "[dry-run] would install: claude, hermes, openclaw" in out
    assert "hook pack" in out and "provider" in out
    assert not (fake_home / ".claude").exists() and not (fake_home / ".hermes").exists()
    assert not (fake_home / ".openclaw").exists()


def test_install_leaves_a_foreign_directory_alone(fake_home):
    from metacog import install as I
    squatter = fake_home / ".openclaw" / "hooks" / "metacog-memory"
    squatter.mkdir(parents=True)
    (squatter / "HOOK.md").write_text("someone else's hook")
    plan = I.run("uninstall", ["openclaw"])
    assert any("left alone (not ours)" in s for s in plan.steps)
    assert (squatter / "HOOK.md").read_text() == "someone else's hook"


def test_install_copy_mode_and_status_command(fake_home, capsys):
    from metacog import install as I
    assert _install("install", "hermes", "--copy") == 0
    dst = fake_home / ".hermes" / "plugins" / "metacog"
    assert dst.is_dir() and not dst.is_symlink()
    assert (dst / "__init__.py").is_file() and (dst / ".metacog-installed").is_file()
    assert _install("uninstall", "hermes") == 0 and not dst.exists()
    assert _install("status") == 0
    out = capsys.readouterr().out
    assert "brain:" in out and "claude" in out and "hermes" in out and "openclaw" in out
