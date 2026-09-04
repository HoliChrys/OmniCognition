"""
Shared plumbing for the metacog Claude Code hooks.

Every hook is a small Python script fed the hook JSON on stdin ; it must NEVER
break the session : silent + exit 0 on anything unexpected. This module holds
what they share — payload reading, transcript parsing (the user's own typed
messages only), the storage-path resolution (same rule as the MCP launcher :
`METACOG_STORAGE` env, overridable per project by a `.metacog-brain` marker
file walked up from the session's cwd), and the `metacog` import path.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

#: The plugin root = the repo root (the `metacog/` package sits next to `hooks/`).
PLUGIN_ROOT = Path(os.environ.get("CLAUDE_PLUGIN_ROOT")
                   or Path(__file__).resolve().parents[1])

#: Default brain when nothing else is configured (shared across projects).
DEFAULT_STORAGE = "~/.metacog/memory.pkl"

#: Per-project brain marker : first non-empty, non-comment line = storage path.
BRAIN_MARKER = ".metacog-brain"

#: Must match metacog.mcp_server.GAP_SENTINEL (kept literal so the hook works
#: even when the package is not importable).
GAP_SENTINEL = "⚠ NO RELEVANT MEMORY (gap)"

# Markers that flag a non-message (tool result / command / harness content).
_SKIP_PREFIX = ("<command-", "[Request interrupted", "<local-command",
                "Caveat:", "<user-memory-input>", "Base directory for this skill:",
                "[Image:", "<task-notification>", "This session is being continued",
                "[SYSTEM NOTIFICATION", "<bash-", "<system-reminder>")
_IDE_TAG_RE = re.compile(r"<ide_(opened_file|selection)>.*?</ide_\1>", re.DOTALL)
_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def read_payload() -> dict:
    """The hook JSON from stdin ({} on anything unparseable)."""
    try:
        return json.loads(sys.stdin.read() or "{}") or {}
    except (json.JSONDecodeError, ValueError):
        return {}


def emit_context(event: str, text: str) -> None:
    """Inject `text` into the model's context for this hook event."""
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event,
                                             "additionalContext": text}}))


def response_text(payload: dict) -> str:
    """Flatten a PostToolUse `tool_response` (str / content blocks / dict)."""
    resp = payload.get("tool_response")
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in resp)
    if isinstance(resp, dict):
        if isinstance(resp.get("content"), list):
            return " ".join(b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in resp["content"])
        return resp.get("text", "") or json.dumps(resp, ensure_ascii=False)
    return ""


def resolve_storage(cwd: Optional[str] = None) -> str:
    """The brain this session writes to. A `.metacog-brain` marker found by
    walking up from `cwd` wins (a dev repo keeps its OWN memory instead of
    polluting the shared one — this must live here, not in settings, because
    Claude Code merges hooks across scopes) ; else `METACOG_STORAGE` ; else the
    shared default."""
    if cwd:
        p = Path(cwd)
        for d in (p, *p.parents):
            marker = d / BRAIN_MARKER
            try:
                if marker.is_file():
                    for line in marker.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line and not line.startswith("#"):
                            return os.path.expanduser(line)
            except OSError:
                pass
    return os.path.expanduser(os.environ.get("METACOG_STORAGE") or DEFAULT_STORAGE)


def _clean(text: str) -> str:
    text = _IDE_TAG_RE.sub("", text)
    text = _REMINDER_RE.sub("", text)
    return text.strip()


def user_messages(transcript_path: str) -> List[dict]:
    """The user's own typed messages from a Claude Code transcript (JSONL), in
    order, as {content, timestamp} — tool results, slash commands, reminders and
    attachments filtered out. [] on any read error."""
    out: List[dict] = []
    try:
        lines = Path(transcript_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") != "user":
            continue
        msg = r.get("message")
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            if any(isinstance(b, dict) and b.get("type") == "tool_result"
                   for b in content):
                continue                        # a tool turn, not a typed message
            content = "\n".join(b.get("text", "") for b in content
                                if isinstance(b, dict) and b.get("type") == "text")
        if not isinstance(content, str):
            continue
        if content.lstrip().startswith(_SKIP_PREFIX):
            continue
        cleaned = _clean(content)
        if cleaned and not cleaned.startswith(_SKIP_PREFIX):
            out.append({"content": cleaned, "timestamp": r.get("timestamp")})
    return out


def load_memory(storage: str):
    """Open the brain at `storage` (created if absent) with the same defaults
    the MCP server uses. Imports `metacog` from the plugin root when it is not
    installed."""
    if str(PLUGIN_ROOT) not in sys.path:
        sys.path.insert(0, str(PLUGIN_ROOT))
    from metacog.defaults import make_encoder
    from metacog.memory import Memory
    os.makedirs(os.path.dirname(os.path.abspath(storage)), exist_ok=True)
    # the SAME encoder resolution as the MCP server (METACOG_ENCODER / auto ->
    # fastembed), so hook and server agree on the brain's embedding space
    return Memory(storage_path=storage, journal_path="auto",
                  encoder=make_encoder())
