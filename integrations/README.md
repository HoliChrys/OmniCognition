# Host integrations — the same memory, three agents

The memory itself is host-agnostic: a manifold of typed points, a SQL usage
journal, an OKF wiki, exposed over MCP. What differs between agent hosts is
*where the memory hooks into the turn loop*. Rather than reimplementing the
discipline three times, each host gets a thin adapter over one shared core.

| host | recall (read) | feed + consolidate (write) | shape |
|---|---|---|---|
| **Claude Code** | MCP `retrieve` / `walk_start`, plus an opt-in `UserPromptSubmit` hook | `SessionStart` rule · `PostToolUse` gap forcing · `SessionEnd` capture + `sleep` | a plugin (`.claude-plugin/`) |
| **Hermes Agent** | `prefetch` → fenced into `<memory-context>` | `sync_turn` · `on_pre_compress` · `on_session_end` | a **memory provider** (Python) |
| **OpenClaw** | MCP `retrieve` (hooks cannot inject context) | `message:received` / `message:sent` · `command:new` / `reset` · `session:compact:before` · `gateway:shutdown` | an **internal hook pack** (+ MCP) |

Everything writes into the **same brain**, resolved the same way everywhere: a
`.metacog-brain` marker walked up from the working directory, else
`METACOG_STORAGE`, else `~/.metacog/memory.pkl`. Point one host at a project
brain and another at the shared one, or let them share — it is one file plus its
journal.

## Install / uninstall

```bash
python -m metacog.install status                    # what is installed where
python -m metacog.install install hermes            # or claude | openclaw | all
python -m metacog.install install all --dry-run     # print the plan, write nothing
python -m metacog.install uninstall openclaw
```

(or `metacog-install …` once the package is installed). Both directions are
**idempotent**, and an uninstall only removes what an install wrote: entries are
identified by an absolute path into this repo, so a neighbour's hook, MCP server
or plugin is never touched. `--copy` installs a copy instead of a symlink;
`--scope project --project DIR` wires Claude Code into one repo rather than the
user profile.

## Hermes Agent — a memory provider

Hermes already has the shape this memory wants, so metacog implements its
`agent.memory_provider.MemoryProvider` interface directly (both are Python — no
subprocess, no bridge):

| Hermes calls | metacog does |
|---|---|
| `initialize(session_id)` | open the brain (production encoder, journal attached) |
| `prefetch(query)` | the cheap `retrieve` (no walk, no LLM) → the lines Hermes fences into `<memory-context>`; on a **gap** it returns the grounding directive instead of noise |
| `sync_turn(user, assistant)` | index both sides as episodic nodes, deduplicated |
| `on_pre_compress(messages)` | store what is about to leave the context window, and say so |
| `on_session_end(messages)` | capture what the turn loop missed, then one `sleep()` cycle |
| `on_session_switch` / `shutdown` | save |
| `get_tool_schemas` / `handle_tool_call` | `metacog_recall` · `metacog_walk` · `metacog_remember` · `metacog_forget` · `metacog_mark_useful` · `metacog_wiki` |

The installer links `integrations/hermes` into `~/.hermes/plugins/metacog`, adds
`metacog` to `plugins.enabled` and sets `memory.provider: metacog` in
`~/.hermes/config.yaml`. Only **one** external provider can be active at a time
in Hermes; the built-in `MEMORY.md` keeps working alongside it.

## OpenClaw — an internal hook pack plus the MCP server

OpenClaw internal hooks are **observers**: what a handler returns does not
block, cancel or modify the operation, and `event.messages` is only delivered
for `/new`, `/reset` and the compaction events. So the split is explicit:

- **reading** the memory is the **MCP server**'s job — the agent calls
  `retrieve` (which carries the in-band gap sentinel when nothing is
  activated), `walk_start`, and the wiki tools;
- **writing** it is the hook's job — `message:received` and `message:sent`
  index each turn, `command:new` / `command:reset` /
  `session:compact:before` run one `sleep()` and reply with a one-line notice,
  `gateway:shutdown` saves.

`handler.js` shells out to `hooks/host_bridge.py`, so brain resolution, encoder
and dedup are exactly the ones Claude Code uses. It is bounded by a timeout and
swallows every failure — a hook must never break a gateway. Set
`METACOG_HOOK_DEBUG=1` to trace what it acts on.

One OpenClaw specific: it **rejects interpreter-startup environment keys**
(`PYTHONPATH` and friends) before spawning a stdio MCP server, so the config
never passes one — `bin/metacog-mcp.sh` sets it internally instead.

## The shared bridge

`hooks/host_bridge.py` is the host-agnostic CLI every non-Python host shells
out to. It is also the quickest way to check a brain by hand:

```bash
python3 hooks/host_bridge.py status
python3 hooks/host_bridge.py recall --query "how do we deploy" --gap-notice
python3 hooks/host_bridge.py feed --role user --session s1 --text "we ship on fridays"
python3 hooks/host_bridge.py consolidate
```

Add `--json` for machine-readable output. Every subcommand exits 0 and stays
silent on failure, by design.

## What is verified, and what is not

`tests/test_integrations.py` covers the bridge contract, the Hermes provider
(lifecycle, prefetch/gap, dedup, compaction, tools, and graceful degradation on
an unopenable brain), the OpenClaw handler (event→action mapping, the spawn,
the notice delivery, and that a broken bridge never throws) and the installer
(idempotent, reversible, leaves a foreign entry alone).

The adapters are written against the **documented** Hermes and OpenClaw APIs;
they have not been run inside a live install of either. If a host changes its
contract, the adapter is the only thing to touch — the memory underneath is the
same one Claude Code drives.
