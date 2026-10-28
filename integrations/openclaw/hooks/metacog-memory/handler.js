/**
 * OpenClaw internal hook → metacog memory.
 *
 * Internal hooks are OBSERVERS: what they return does not block, cancel or
 * modify the operation, and only `/new`, `/reset` and the compaction events
 * deliver `event.messages` back to a channel. So this handler feeds the memory
 * and consolidates it; RECALL is the MCP server's job (tool `retrieve`).
 *
 * All work goes through the host-agnostic bridge (`hooks/host_bridge.py`), so
 * the brain resolution, the encoder and the dedup are exactly the ones the
 * Claude Code plugin uses. Bounded (timeout), silent on failure: a hook must
 * never break the gateway.
 */

import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
/** Repo root: <repo>/integrations/openclaw/hooks/metacog-memory → up 4. */
const REPO_ROOT = process.env.METACOG_ROOT || resolve(HERE, "..", "..", "..", "..");
const BRIDGE = resolve(REPO_ROOT, "hooks", "host_bridge.py");
const PYTHON = process.env.METACOG_PYTHON || "python3";
const TIMEOUT_MS = Number(process.env.METACOG_HOOK_TIMEOUT_MS || 20000);
const DEBUG = /^(1|true|yes)$/i.test(process.env.METACOG_HOOK_DEBUG || "");

const log = (...a) => { if (DEBUG) console.error("[metacog]", ...a); };

/** Run the bridge; resolve with its parsed JSON, or null on any failure. */
function bridge(args, { stdin = "", timeout = TIMEOUT_MS } = {}) {
  return new Promise((done) => {
    let child;
    try {
      child = spawn(PYTHON, [BRIDGE, "--json", ...args], {
        stdio: ["pipe", "pipe", "pipe"],
      });
    } catch (err) {
      log("spawn failed", err?.message);
      return done(null);
    }
    let out = "";
    let settled = false;
    const finish = (value) => { if (!settled) { settled = true; done(value); } };
    const timer = setTimeout(() => { try { child.kill("SIGKILL"); } catch {} finish(null); },
                             timeout);
    child.stdout.on("data", (d) => { out += d; });
    child.stderr.on("data", (d) => log("stderr:", String(d).trim()));
    child.on("error", (err) => { clearTimeout(timer); log("error", err?.message); finish(null); });
    child.on("close", () => {
      clearTimeout(timer);
      try { finish(JSON.parse(out.trim().split("\n").pop() || "null")); }
      catch { finish(null); }
    });
    try { child.stdin.end(stdin); } catch { /* closed already */ }
  });
}

/**
 * The message text, wherever this producer put it. `context` is a read-only
 * observation whose shape varies by channel, so probe the plausible fields
 * instead of assuming one.
 */
export function pickText(context = {}) {
  const candidates = [
    context.text, context.body, context.content, context.transcript,
    context.message?.text, context.message?.body, context.message?.content,
    context.payload?.text, context.payload?.body,
  ];
  const hit = candidates.find((v) => typeof v === "string" && v.trim());
  return hit ? hit.trim() : "";
}

/** The workspace/project dir, so a `.metacog-brain` marker is honoured. */
export function pickCwd(context = {}) {
  return context.workspace || context.cwd || context.projectDir || process.cwd();
}

/** Which bridge call an event maps to (exported for testing). */
export function planFor(event = {}) {
  const key = `${event.type}:${event.action}`;
  const ctx = event.context || {};
  const cwd = pickCwd(ctx);
  const session = event.sessionKey || ctx.sessionKey || "openclaw";

  if (key === "message:received" || key === "message:transcribed") {
    const text = pickText(ctx);
    if (!text) return null;
    return { kind: "feed", args: ["--cwd", cwd, "feed", "--role", "user",
                                  "--session", session], stdin: text };
  }
  if (key === "message:sent") {
    if (ctx.success === false) return null;          // undelivered: not a turn
    const text = pickText(ctx);
    if (!text) return null;
    return { kind: "feed", args: ["--cwd", cwd, "feed", "--role", "agent",
                                  "--session", session], stdin: text };
  }
  if (key === "command:new" || key === "command:reset"
      || key === "session:compact:before" || key === "gateway:shutdown") {
    return { kind: "consolidate", args: ["--cwd", cwd, "consolidate",
                                         "--session", session],
             notify: key !== "gateway:shutdown" };
  }
  return null;
}

export default async function handler(event) {
  const plan = planFor(event);
  if (!plan) return;
  log(event.type + ":" + event.action, "→", plan.kind);
  const result = await bridge(plan.args, { stdin: plan.stdin || "" });
  if (!result) return;
  // `event.messages` is only delivered for /new, /reset and compaction — append
  // synchronously, before the handler resolves, or it misses delivery.
  if (plan.notify && result.slept && Array.isArray(event.messages) && result.text) {
    event.messages.push(result.text);
  }
}
