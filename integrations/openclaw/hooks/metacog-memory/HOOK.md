---
name: metacog-memory
description: "Feed every OpenClaw turn into the metacog memory and consolidate it offline"
homepage: https://github.com/HoliChrys/OmniCognition
metadata:
  {
    "openclaw": {
      "emoji": "🧠",
      "events": [
        "message:received",
        "message:sent",
        "command:new",
        "command:reset",
        "session:compact:before",
        "gateway:shutdown"
      ],
      "requires": { "bins": ["python3"] }
    }
  }
---

# metacog-memory

Gives an OpenClaw gateway the same standing memory discipline the Claude Code
plugin installs: **every turn is fed**, and the memory **consolidates itself
offline**.

## What it does, per event

| event | what happens |
|---|---|
| `message:received` | the inbound message is indexed as an episodic turn (`role=user`) |
| `message:sent` | the delivered reply is indexed (`role=agent`); skipped when `context.success` is false |
| `command:new` / `command:reset` | the session ends → one `sleep()` cycle (collision, decay-fit, forget-merge, wiki reconcile + seed re-run, tool promotion) then `save()`; a one-line notice is returned to the channel |
| `session:compact:before` | same consolidation before compaction, so what leaves the context window is already in the memory; the notice reaches the compaction caller |
| `gateway:shutdown` | a final `save()` |

Recall is deliberately **not** here: OpenClaw internal hooks are observers —
they cannot inject context into a turn. The agent reads its memory through the
**MCP server** instead (`.mcp.json` in this bundle, tool `retrieve`, which
carries the in-band gap sentinel when nothing is activated).

## Side effects

Writes to the brain resolved, in order, from a `.metacog-brain` marker walked
up from the workspace, `METACOG_STORAGE`, then `~/.metacog/memory.pkl` — plus
its SQLite journal next to it. Nothing else is touched, nothing leaves the
machine (embedding and reranking run locally). Each event spawns one short-lived
`python3` process and is bounded by a timeout; every failure is swallowed and
logged, never raised into the gateway.

## Verify

```bash
python3 <repo>/hooks/host_bridge.py status          # where the brain is, what is in it
openclaw hooks list                                  # metacog-memory present
openclaw hooks enable metacog-memory
# then say something in a channel and re-run:
python3 <repo>/hooks/host_bridge.py status          # the point count grew
```

Set `METACOG_HOOK_DEBUG=1` to have the handler log each event it acts on.
