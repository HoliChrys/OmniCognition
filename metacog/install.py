"""
Install / uninstall the metacog memory into an agent host.

One command per host, idempotent both ways :

    python -m metacog.install status
    python -m metacog.install install  claude|hermes|openclaw|all [--dry-run]
    python -m metacog.install uninstall claude|hermes|openclaw|all [--dry-run]

What each host gets :

  claude    the four hooks wired into `settings.json` (SessionStart injects the
            discipline · PostToolUse forces grounding on a recall gap ·
            SessionEnd captures the session + sleeps · UserPromptSubmit
            auto-recall, opt-in) and the MCP server in `.mcp.json`.
            `--scope user` (default, ~/.claude) or `--scope project`.
  hermes    the memory provider linked into ~/.hermes/plugins/metacog, enabled
            in config.yaml (`plugins.enabled` + `memory.provider`).
  openclaw  the internal hook pack linked into ~/.openclaw/hooks/, enabled in
            openclaw.json (`hooks.internal.entries`), plus the MCP server.

Everything written is OURS and tagged by an absolute path into this repo, so an
uninstall removes exactly what an install added and never touches a neighbour's
entry. `--dry-run` prints the plan and writes nothing. A host that is not
installed on this machine is reported, not failed : you can wire a config
before the host exists.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"
LAUNCHER = ROOT / "bin" / "metacog-mcp.sh"
OPENCLAW_HOOK_SRC = ROOT / "integrations" / "openclaw" / "hooks" / "metacog-memory"
HERMES_SRC = ROOT / "integrations" / "hermes"

HOSTS = ("claude", "hermes", "openclaw")
#: Claude Code hook wiring : event -> (script, matcher, timeout).
CLAUDE_HOOKS: Tuple[Tuple[str, str, Optional[str], int], ...] = (
    ("SessionStart", "session_start.py", None, 10),
    ("UserPromptSubmit", "auto_recall.py", None, 20),
    ("PostToolUse", "recall_gap.py",
     r"mcp__(plugin_metacog_)?metacog__(retrieve|walk_start)", 10),
    ("SessionEnd", "capture_session.py", None, 120),
)


def home() -> Path:
    return Path(os.path.expanduser("~"))


# -- small file helpers (never clobber a neighbour's config) ------------------

def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def read_yaml(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, ImportError):
        return {}
    except Exception:
        return {}


def write_yaml(path: Path, data: dict) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")


def dig(d: dict, *keys: str) -> dict:
    """Walk/create nested dicts."""
    cur = d
    for k in keys:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    return cur


def link(src: Path, dst: Path, copy: bool = False) -> str:
    """Symlink (or copy) `src` to `dst`, replacing OUR previous link only."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return "already linked"
        if dst.is_symlink():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy:
        shutil.copytree(src, dst)
        return "copied"
    dst.symlink_to(src, target_is_directory=True)
    return "linked"


def unlink(dst: Path, src: Path) -> str:
    """Remove `dst` only when it is our link / our copy."""
    if dst.is_symlink():
        if Path(os.readlink(dst)) != src:
            return "left alone (not ours)"
        dst.unlink()
        return "unlinked"
    if dst.is_dir() and (dst / ".metacog-installed").exists():
        shutil.rmtree(dst)
        return "removed"
    if dst.exists():
        return "left alone (not ours)"
    return "absent"


# -- claude code --------------------------------------------------------------

def _claude_dir(scope: str, project: Optional[str]) -> Path:
    if scope == "project":
        return Path(project or os.getcwd()).resolve() / ".claude"
    return home() / ".claude"


def _claude_mcp_path(scope: str, project: Optional[str]) -> Path:
    """Project scope uses the repo's `.mcp.json` (shared, committable) ; user
    scope uses `~/.claude.json`, where Claude Code keeps user-level config."""
    if scope == "project":
        return Path(project or os.getcwd()).resolve() / ".mcp.json"
    return home() / ".claude.json"


def _is_ours(command: str) -> bool:
    return str(HOOKS) in str(command)


def claude_install(plan: "Plan", scope: str = "user",
                   project: Optional[str] = None, **kw) -> None:
    base = _claude_dir(scope, project)
    settings = base / "settings.json"
    data = read_json(settings)
    hooks = dig(data, "hooks")
    for event, script, matcher, timeout in CLAUDE_HOOKS:
        entry: Dict[str, Any] = {"hooks": [{
            "type": "command",
            "command": f'python3 "{HOOKS / script}"',
            "timeout": timeout,
        }]}
        if matcher:
            entry["matcher"] = matcher
        kept = [e for e in hooks.get(event, [])
                if not any(_is_ours(h.get("command", ""))
                           for h in (e.get("hooks") or []))]
        hooks[event] = kept + [entry]
    plan.write_json(settings, data, f"claude: 4 hooks in {settings}")

    mcp_path = _claude_mcp_path(scope, project)
    mcp = read_json(mcp_path)
    dig(mcp, "mcpServers")["metacog"] = {
        "command": "bash", "args": [str(LAUNCHER)],
        "env": {"METACOG_SURFACE": os.environ.get("METACOG_SURFACE", "external")},
    }
    plan.write_json(mcp_path, mcp, f"claude: MCP server in {mcp_path}")
    plan.note("claude: alternative — `/plugin marketplace add HoliChrys/OmniCognition` "
              "then `/plugin install metacog@omnicognition`")


def claude_uninstall(plan: "Plan", scope: str = "user",
                     project: Optional[str] = None, **kw) -> None:
    base = _claude_dir(scope, project)
    settings = base / "settings.json"
    data = read_json(settings)
    hooks = data.get("hooks") or {}
    removed = 0
    for event in list(hooks):
        kept = []
        for e in hooks.get(event, []):
            if any(_is_ours(h.get("command", "")) for h in (e.get("hooks") or [])):
                removed += 1
                continue
            kept.append(e)
        if kept:
            hooks[event] = kept
        else:
            hooks.pop(event, None)
    if not hooks:
        data.pop("hooks", None)
    plan.write_json(settings, data, f"claude: removed {removed} hook(s) from {settings}")

    mcp_path = _claude_mcp_path(scope, project)
    mcp = read_json(mcp_path)
    if (mcp.get("mcpServers") or {}).pop("metacog", None) is not None:
        if not mcp["mcpServers"]:
            mcp.pop("mcpServers")
        plan.write_json(mcp_path, mcp, f"claude: removed MCP server from {mcp_path}")


def claude_status(scope: str = "user", project: Optional[str] = None, **kw) -> dict:
    settings = _claude_dir(scope, project) / "settings.json"
    hooks = (read_json(settings).get("hooks") or {})
    n = sum(1 for ev in hooks.values() for e in ev
            for h in (e.get("hooks") or []) if _is_ours(h.get("command", "")))
    mcp_path = _claude_mcp_path(scope, project)
    return {"host": "claude", "present": (_claude_dir(scope, project)).exists(),
            "hooks": n, "mcp": "metacog" in (read_json(mcp_path).get("mcpServers") or {}),
            "config": str(settings)}


# -- hermes -------------------------------------------------------------------

def _hermes_dir() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (home() / ".hermes"))


def hermes_install(plan: "Plan", copy: bool = False, **kw) -> None:
    dst = _hermes_dir() / "plugins" / "metacog"
    plan.link(HERMES_SRC, dst, copy, f"hermes: provider → {dst}")
    cfg_path = _hermes_dir() / "config.yaml"
    cfg = read_yaml(cfg_path)
    enabled = dig(cfg, "plugins").setdefault("enabled", [])
    if not isinstance(enabled, list):
        enabled = []
    if "metacog" not in enabled:
        enabled.append("metacog")
    cfg["plugins"]["enabled"] = enabled
    dig(cfg, "memory")["provider"] = "metacog"
    plan.write_yaml(cfg_path, cfg,
                    f"hermes: plugins.enabled += metacog, memory.provider = metacog "
                    f"({cfg_path})")
    plan.note("hermes: only ONE external memory provider can be active at a time — "
              "this sets metacog as it; the built-in MEMORY.md stays active too")


def hermes_uninstall(plan: "Plan", **kw) -> None:
    dst = _hermes_dir() / "plugins" / "metacog"
    plan.unlink(dst, HERMES_SRC, f"hermes: provider ← {dst}")
    cfg_path = _hermes_dir() / "config.yaml"
    cfg = read_yaml(cfg_path)
    if not cfg:
        return
    enabled = (cfg.get("plugins") or {}).get("enabled")
    if isinstance(enabled, list) and "metacog" in enabled:
        cfg["plugins"]["enabled"] = [e for e in enabled if e != "metacog"]
    if (cfg.get("memory") or {}).get("provider") == "metacog":
        cfg["memory"].pop("provider")
        if not cfg["memory"]:
            cfg.pop("memory")
    plan.write_yaml(cfg_path, cfg, f"hermes: disabled in {cfg_path}")


def hermes_status(**kw) -> dict:
    d = _hermes_dir()
    cfg = read_yaml(d / "config.yaml")
    dst = d / "plugins" / "metacog"
    return {"host": "hermes", "present": d.exists(), "linked": dst.exists(),
            "enabled": "metacog" in ((cfg.get("plugins") or {}).get("enabled") or []),
            "provider": (cfg.get("memory") or {}).get("provider"),
            "config": str(d / "config.yaml")}


# -- openclaw -----------------------------------------------------------------

def _openclaw_dir() -> Path:
    return Path(os.environ.get("OPENCLAW_HOME") or (home() / ".openclaw"))


def openclaw_install(plan: "Plan", copy: bool = False, **kw) -> None:
    dst = _openclaw_dir() / "hooks" / "metacog-memory"
    plan.link(OPENCLAW_HOOK_SRC, dst, copy, f"openclaw: hook pack → {dst}")
    cfg_path = _openclaw_dir() / "openclaw.json"
    cfg = read_json(cfg_path)
    internal = dig(cfg, "hooks", "internal")
    internal["enabled"] = True
    dig(internal, "entries")["metacog-memory"] = {"enabled": True}
    # NB: no PYTHONPATH here — OpenClaw rejects interpreter-startup env keys
    # before spawning a stdio MCP server; the launcher sets it internally.
    dig(cfg, "mcpServers")["metacog"] = {
        "command": "bash", "args": [str(LAUNCHER)],
        "env": {"METACOG_SURFACE": os.environ.get("METACOG_SURFACE", "external")},
    }
    plan.write_json(cfg_path, cfg,
                    f"openclaw: hook enabled + MCP server in {cfg_path}")
    plan.note("openclaw: recall is the MCP server's job (internal hooks are "
              "observers and cannot inject context)")


def openclaw_uninstall(plan: "Plan", **kw) -> None:
    dst = _openclaw_dir() / "hooks" / "metacog-memory"
    plan.unlink(dst, OPENCLAW_HOOK_SRC, f"openclaw: hook pack ← {dst}")
    cfg_path = _openclaw_dir() / "openclaw.json"
    cfg = read_json(cfg_path)
    if not cfg:
        return
    entries = ((cfg.get("hooks") or {}).get("internal") or {}).get("entries") or {}
    entries.pop("metacog-memory", None)
    if (cfg.get("mcpServers") or {}).pop("metacog", None) is not None:
        if not cfg["mcpServers"]:
            cfg.pop("mcpServers")
    plan.write_json(cfg_path, cfg, f"openclaw: disabled in {cfg_path}")


def openclaw_status(**kw) -> dict:
    d = _openclaw_dir()
    cfg = read_json(d / "openclaw.json")
    entries = ((cfg.get("hooks") or {}).get("internal") or {}).get("entries") or {}
    dst = d / "hooks" / "metacog-memory"
    return {"host": "openclaw", "present": d.exists(), "linked": dst.exists(),
            "enabled": bool((entries.get("metacog-memory") or {}).get("enabled")),
            "mcp": "metacog" in (cfg.get("mcpServers") or {}),
            "config": str(d / "openclaw.json")}


ACTIONS: Dict[str, Dict[str, Callable]] = {
    "claude": {"install": claude_install, "uninstall": claude_uninstall,
               "status": claude_status},
    "hermes": {"install": hermes_install, "uninstall": hermes_uninstall,
               "status": hermes_status},
    "openclaw": {"install": openclaw_install, "uninstall": openclaw_uninstall,
                 "status": openclaw_status},
}


class Plan:
    """Collects what would change ; applies it unless `dry_run`."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.steps: List[str] = []
        self.notes: List[str] = []

    def note(self, text: str) -> None:
        self.notes.append(text)

    def write_json(self, path: Path, data: dict, label: str) -> None:
        self.steps.append(label)
        if not self.dry_run:
            write_json(path, data)

    def write_yaml(self, path: Path, data: dict, label: str) -> None:
        self.steps.append(label)
        if not self.dry_run:
            try:
                write_yaml(path, data)
            except ImportError:
                self.steps[-1] = label + " — SKIPPED (PyYAML missing)"

    def link(self, src: Path, dst: Path, copy: bool, label: str) -> None:
        self.steps.append(label)
        if not self.dry_run:
            how = link(src, dst, copy)
            if copy:
                (dst / ".metacog-installed").write_text("copied by metacog.install\n")
            self.steps[-1] = f"{label} ({how})"

    def unlink(self, dst: Path, src: Path, label: str) -> None:
        self.steps.append(label)
        if not self.dry_run:
            self.steps[-1] = f"{label} ({unlink(dst, src)})"


def run(action: str, hosts: List[str], **kw) -> Plan:
    plan = Plan(dry_run=bool(kw.get("dry_run")))
    for host in hosts:
        ACTIONS[host][action](plan, **kw)
    return plan


def status_all(**kw) -> List[dict]:
    return [ACTIONS[h]["status"](**kw) for h in HOSTS]


def _brain_line() -> str:
    sys.path.insert(0, str(HOOKS))
    try:
        from _common import resolve_storage
        p = resolve_storage(os.getcwd())
        return f"brain: {p} ({'exists' if os.path.exists(os.path.expanduser(p)) else 'not created yet'})"
    except Exception as exc:
        return f"brain: unresolved ({exc})"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="metacog-install",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("install", "uninstall"):
        s = sub.add_parser(name, help=f"{name} the metacog memory into a host")
        s.add_argument("host", choices=[*HOSTS, "all"])
        s.add_argument("--dry-run", action="store_true",
                       help="print the plan, write nothing")
        s.add_argument("--scope", choices=["user", "project"], default="user",
                       help="claude only: ~/.claude (default) or <project>/.claude")
        s.add_argument("--project", default=None, help="claude --scope project dir")
        s.add_argument("--copy", action="store_true",
                       help="copy instead of symlinking (hermes / openclaw)")
    sub.add_parser("status", help="what is installed where")

    a = p.parse_args(argv)
    if a.action == "status":
        print(_brain_line())
        for st in status_all():
            host = st.pop("host")
            bits = " · ".join(f"{k}={v}" for k, v in st.items())
            print(f"{host:9s} {bits}")
        return 0

    hosts = list(HOSTS) if a.host == "all" else [a.host]
    plan = run(a.action, hosts, dry_run=a.dry_run, scope=a.scope,
               project=a.project, copy=a.copy)
    head = f"[dry-run] would {a.action}" if a.dry_run else a.action + "ed"
    print(f"{head}: {', '.join(hosts)}")
    for s in plan.steps:
        print(f"  · {s}")
    for n in plan.notes:
        print(f"  ! {n}")
    if not a.dry_run:
        print(_brain_line())
    return 0


if __name__ == "__main__":
    sys.exit(main())
