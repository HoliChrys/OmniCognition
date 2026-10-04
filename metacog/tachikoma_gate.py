"""The TACHIKOMA GATE — one omni memory PER CONTEXT, served behind a header.

OmniCognition was born single-tenant: one app, one `Memory`, one store. Tachikoma
is multi-context — each context (`tachikoma.paralelle.GenAI`, `global`, …) has
its OWN long-term memory, and the app proxy (`/api/memory`) says which one it
serves through the `x-tachikoma-context` header. Same contract as mnema, so the
tachikoma memory plugin can bind either engine without knowing the motor.

THREE PIECES, NONE OF THEM TOUCHES THE TOOLS:

1. `ContextualMemory` — a PROXY whose every attribute access resolves the
   `Memory` instance of the current context (a contextvar). The ~40 tools of
   `mcp_server` close over `memory` with no context parameter: they call
   `memory.ingest(...)`, and the proxy picks — ON EVERY CALL — the memory of
   the context that is speaking. Zero changes in `mcp_server.py`: the
   `memory=` parameter of `build_app` already exists to inject a populated
   memory; we inject ours.

2. The Starlette middleware — reads `x-tachikoma-context` from EVERY HTTP
   request and sets the contextvar. No header: 400, like mnema ("en HTTP le
   contexte est obligatoire") — fail-closed, never a default memory that would
   mix contexts. Then THE ACL, on every request (`authorize`, mnema's
   `acl.authorize_read` ported): WHO (`/api/auth/me`), the context EXISTS
   (`/api/hierarchy/<ctx>` — else `_resolve` would `makedirs` a memory under
   any name), and MAY READ it (`/api/acl/check`). Measured before (TAC-214):
   a token with rights on GenAI only wrote into `demo.sandbox.alice`, HTTP 200.

3. The PER-CONTEXT DEEPWIKI — the `notes/` folder of the tachikoma context IS
   its wiki: every `.md` is ingested (`import_okf`) the first time the memory
   of that context is touched, its refs linked into the RAG. The wiki lives
   where the notes already live — not in a parallel store that would drift.

Storage: `<storage_root>/<ctx>/memory.pkl` — one memory per context, isolated,
persistent. `global` and `tachikoma.paralelle.GenAI` share NOTHING by accident.
"""
from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any, Optional

#: The contextvar of the served context. Set by the middleware, read by the proxy.
_current_ctx: ContextVar[str] = ContextVar("tachikoma_context", default="")

#: The header tachikoma sends — same value as mnema's `CTX_HEADER`, taken
#: verbatim so a proxy never speaks a dialect the server does not listen to.
CTX_HEADER = "x-tachikoma-context"

# ── THE ACL — who may read which memory (mnema's `acl.py`, ported) ──────────
#
# omni replaced mnema and kept its header but not its ACL: the gate checked the
# header's PRESENCE, nothing else (TAC-214). The router (`/api/memory`) does no
# authorization by design — "l'autorisation est celle du MOTEUR" — and forwards
# the bearer token for exactly this. Same three questions, same sources, same
# fail-closed answers as mnema: the policy is tachikoma's, not re-invented here.

_API = os.environ.get("TACHIKOMA_API_URL", "http://127.0.0.1:8000").rstrip("/")

#: The COMMON NOTEBOOK — the ONLY name that escapes authorization (not
#: authentication). `general` is no tachikoma context: no hierarchy, no ACL
#: resource — there is nothing to ask. Anything written there is readable by
#: ANY valid token, by construction (mnema's documented exception, kept as is).
GENERAL = os.environ.get("MNEMA_GENERAL_CTX", "general")

#: 30 s, mnema's measured ceiling: warm the check costs ~13 ms, but the first
#: rights resolution of a fresh API was measured at 17.8 s. A timeout yields a
#: REFUSAL (fail-closed) — too short protects no one, it lies.
_ACL_TIMEOUT = float(os.environ.get("OMNI_ACL_TIMEOUT", "30"))


class Denied(PermissionError):
    """An authorization refusal. ALWAYS carries the reason and the HTTP code."""

    def __init__(self, reason: str, status: int = 403):
        super().__init__(reason)
        self.status = status


def _api(path: str, token: str, payload: Optional[dict] = None) -> tuple[int, Any]:
    """One call to the tachikoma API with the caller's token. (code, json)."""
    import json
    import urllib.error
    import urllib.request

    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{_API}{path}", data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_ACL_TIMEOUT) as r:
            return r.status, json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        return e.code, None
    except OSError as e:
        raise Denied(f"ACL injoignable ({_API}) : {e}", 503) from e


def authorize(token: str, ctx: str) -> str:
    """The `user_id` if the token's bearer may READ the memory of `ctx`.

    Raises `Denied` otherwise — no fallback, no default context. Serving the
    wrong memory in silence is worse than refusing. Writes go through the same
    check, exactly like mnema (it gated every tool on `read`).
    """
    if not token:
        raise Denied("aucun jeton fourni", 401)
    # 1. WHO — never decode the token ourselves: the authority checks the signature.
    code, body = _api("/api/auth/me", token)
    if code != 200 or not isinstance(body, dict) or not body.get("user_id"):
        raise Denied(f"jeton rejeté par l'ACL (HTTP {code})", 401)
    user = str(body["user_id"])

    # Authentication required, authorization does not apply — `general` only.
    if ctx == GENERAL:
        return user

    # 2. WHERE — existence FIRST, it is a guard: `_resolve` does `os.makedirs`,
    # so an authorized unknown name would GIVE BIRTH to a memory (measured:
    # `contexts/contexte.inconnu.personne/`, 2026-09-18). `quote(safe="")`: the
    # name comes from the caller — `../x` must not leave the route.
    import urllib.parse
    code, _ = _api(f"/api/hierarchy/{urllib.parse.quote(ctx, safe='')}", token)
    if code == 404:
        raise Denied(f"le contexte {ctx!r} n'existe pas")
    if code != 200:
        # We could NOT ASK: refuse (fail-closed), but never call an outage a fact.
        raise Denied(f"impossible de vérifier le contexte {ctx!r} : l'API a rendu "
                     f"HTTP {code} — panne en amont, refus par prudence", 503)

    # 3. WHAT — ask the ACL; the resource is the BARE context name (mnema measured
    # that `ctx:<name>` only "works" for admins through the bypass).
    code, body = _api("/api/acl/check", token,
                      {"user": user, "action": "read", "resource": ctx})
    if code != 200 or not isinstance(body, dict):
        raise Denied(f"contrôle d'accès indisponible pour {user!r} (HTTP {code})", 503)
    if body.get("allowed") is True:
        return user
    raise Denied(f"{user!r} n'a pas 'read' sur le contexte {ctx!r}")


def _bearer(headers: Any) -> str:
    auth = headers.get("authorization") or ""
    return auth[7:].strip() if auth[:7].lower() == "bearer " else ""


class ContextualMemory:
    """A delegate: every call goes to the `Memory` of the current context.

    `build_app(memory=…)` attaches THIS object to the tools; they do
    `memory.ingest(…)` and the attribute is resolved HERE, through
    `__getattr__`, toward the instance of the context. Instances are created
    lazily (first access) and kept: startup only pays for the contexts served.

    ATTRIBUTES SET (`memory.x = …`) go through to the active instance —
    `build_app` itself sets some (`recency_weight`) right after injection: a
    proxy that swallowed them would break the ACT-R levers without a single
    engine test noticing.
    """

    def __init__(self, storage_root: str, notes_root: Optional[str] = None):
        self._root = os.path.expanduser(storage_root)
        #: The root of tachikoma contexts — where each `notes/` folder lives.
        self._notes_root = os.path.expanduser(notes_root) if notes_root else None
        self._memories: dict[str, Any] = {}
        self._ingested: set[str] = set()

    # ── resolution ────────────────────────────────────────────────────
    def _ctx_name(self) -> str:
        name = _current_ctx.get().strip()
        if not name:
            raise RuntimeError(
                "no context served — the x-tachikoma-context header is "
                "mandatory (fail-closed, like mnema)")
        return name

    def _resolve(self) -> Any:
        from metacog.defaults import make_encoder, make_reranker
        from metacog.memory import Memory

        name = self._ctx_name()
        m = self._memories.get(name)
        if m is None:
            path = os.path.join(self._root, name, "memory.pkl")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            m = Memory(storage_path=path, journal_path="auto",
                       encoder=make_encoder(), reranker=make_reranker())
            self._memories[name] = m
        # THE DEEPWIKI IS INGESTED ON EVERY FIRST ACCESS OF THE PROCESS, not
        # only at creation: a memory loaded from the pickle (server restart)
        # must find ITS notes too — otherwise the wiki only lives in the
        # process that created the memory, and every restart loses it.
        # `_ingested` bounds it to once per (process, context).
        self._ingest_notes(name, m)
        return m

    def _ingest_notes(self, ctx: str, m: Any) -> None:
        """The context's deepwiki: its `notes/` folder, ingested once.

        Every `.md` becomes an OKF doc (`import_okf` — frontmatter + refs
        linked into the RAG), recursively. ONCE per (ctx, process).

        ANCESTOR SEEDING — the inheritance mnema serves on query, the wiki
        serves at ingestion: alongside the context's OWN notes, every
        ANCESTOR's notes are ingested too (a child context's wiki sees its
        parents' notes — `tachikoma.paralelle.GenAI` also gets `tachikoma`'s
        and the root's). The ancestor docs are marked by their doc id
        (`notes:<ancestor>/…` namespace below), so a caller can tell an
        inherited note from a local one. Measured on mnema: its recall marks
        inherited hits `← hérité`; the wiki answers the same question.

        THE MAPPING IS THE CONTEXT PATH, NOT ITS DOTTED NAME: the context
        `tachikoma.paralelle.GenAI` lives at
        `<notes_root>/tachikoma/paralelle/GenAI` — the dots of the name are
        the slashes of the folder (measured: the root is `tachikoma`, its
        notes at `<root>/notes`, GenAI's at
        `<root>/tachikoma/paralelle/GenAI/notes`). The FIRST segment often
        repeats the root name, so the without-first-segment form is tried
        first; the ROOT context's folder IS the notes_root itself.
        """
        if ctx in self._ingested or not self._notes_root:
            return
        self._ingested.add(ctx)
        n_docs = n_points = 0
        # OWN notes first, then every ancestor's (nearest first).
        for source_ctx, folder in self._note_folders(ctx):
            prefix = "" if source_ctx == ctx else f"{source_ctx}/"
            for root, _dirs, files in sorted(os.walk(folder)):
                for filename in sorted(files):
                    if not filename.endswith(".md"):
                        continue
                    path = os.path.join(root, filename)
                    # The doc id keeps the RELATIVE path (collision-proof) and,
                    # for ancestors, the source context (inheritance visible).
                    rel = os.path.relpath(path, folder)[:-3]
                    try:
                        with open(path, encoding="utf-8", errors="replace") as fh:
                            body = fh.read()
                        doc_id = f"notes:{prefix}{rel}"
                        # TWO INGESTIONS, TWO ROLES — both asked for:
                        # 1. import_okf: the DOC (frontmatter, refs, wiki_where/
                        #    wiki_doc queryable) — the deepwiki structure.
                        m.import_okf(doc_id, body)
                        n_docs += 1
                        # 2. ingest: the CONTENT as a RAG point — measured gap:
                        #    retrieve/walk never saw the notes (docs live in the
                        #    journal, the RAG indexes points). Without this, a
                        #    question whose answer is IN a note returns empty.
                        #    The point is tagged with its doc id + source ctx so
                        #    the hit can be traced back to the note it came from.
                        try:
                            # The engine's ingest() takes NO tags (measured:
                            # TypeError) — the MCP tool adds them to the point
                            # AFTER creation (add_tag + journal tag index).
                            # We mirror the tool exactly.
                            p = m.ingest(body, kind="FACT")
                            p.add_tag(f"note:{doc_id}", f"ctx:{source_ctx}",
                                      "deepwiki")
                            try:
                                if m.journal is not None:
                                    m.journal.log_tags(p.id, p.tags)
                            except Exception:  # noqa: BLE001 — doc index only
                                pass
                            n_points += 1
                        except Exception:  # noqa: BLE001 — doc stays even if the point fails
                            print(f"[gate] content point failed: {doc_id}",
                                  flush=True)
                    except Exception:  # noqa: BLE001 — one unreadable note never stops the wiki
                        print(f"[gate] unreadable note skipped: "
                              f"{source_ctx}/notes/{rel}.md", flush=True)
        if n_docs or n_points:
            print(f"[gate] deepwiki of {ctx!r}: {n_docs} doc(s) + "
                  f"{n_points} content point(s) ingested (own + ancestors)",
                  flush=True)

    def _note_folders(self, ctx: str) -> list[tuple[str, str]]:
        """(source_context, notes_folder) for ctx and its ancestors, own first.

        The ancestor chain of `a.b.c` is `[a.b.c, a.b, a]` — each prefix of
        the dotted name, nearest parent first, the root last. The same
        candidate mapping as a single context applies per ancestor.
        """
        segments = ctx.split(".")
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        # own context first, then progressively shorter prefixes (ancestors)
        for i in range(len(segments), 0, -1):
            src = ".".join(segments[:i])
            if not src or src in seen:
                continue
            seen.add(src)
            sub = segments[:i]
            candidates = []
            if len(sub) > 1:
                candidates.append(os.path.join(self._notes_root, *sub[1:], "notes"))
                candidates.append(os.path.join(self._notes_root, *sub, "notes"))
            else:
                # SINGLE-SEGMENT = the galaxy/root: its own folder if it
                # exists, else the notes_root itself stands in.
                candidates.append(os.path.join(self._notes_root, *sub, "notes"))
                candidates.append(os.path.join(self._notes_root, "notes"))
            folder = next((d for d in candidates if os.path.isdir(d)), "")
            if folder and all(folder != f for _s, f in out):
                out.append((src, folder))
        return out

    # ── delegation ────────────────────────────────────────────────────
    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        setattr(self._resolve(), name, value)


def context_gate(authorize_fn: Any = None) -> Any:
    """The middleware class: context header, then the ACL, on EVERY request.

    `authorize_fn(token, ctx) -> user` defaults to `authorize` (resolved at call
    time); tests inject a fake. The check is blocking urllib — run in a thread
    so a cold 17 s ACL never freezes the other contexts' requests.
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class _ContextGate(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            ctx = (request.headers.get(CTX_HEADER) or "").strip()
            if not ctx:
                return JSONResponse(
                    {"detail": "en HTTP le contexte est obligatoire : "
                               f"l'en-tête {CTX_HEADER} est absent"}, status_code=400)
            import anyio
            try:
                await anyio.to_thread.run_sync(
                    authorize_fn or authorize, _bearer(request.headers), ctx)
            except Denied as e:
                return JSONResponse({"detail": str(e)}, status_code=e.status)
            _current_ctx.set(ctx)
            return await call_next(request)

    return _ContextGate


def build_gated_app(storage_root: str, notes_root: str,
                    surface: Optional[str] = None,
                    authorize_fn: Any = None) -> Any:
    """The omni MCP app, GATED per context — ready to serve over HTTP.

    This is the tachikoma deployment entry point: the FastMCP app of
    `build_app` (no tool changed), wrapped in a middleware that sets the
    context after the ACL said yes. `storage_root` isolates memories
    (`<root>/<ctx>/memory.pkl`); `notes_root` is the root of tachikoma context
    folders (`<root>/<ctx>/notes` is the context's wiki).
    """
    from metacog.mcp_server import build_app

    proxy = ContextualMemory(storage_root, notes_root)
    mcp_app = build_app(memory=proxy, surface=surface)

    # The streamable HTTP app of FastMCP, under OUR middleware: every request
    # carries its context before any tool runs.
    inner = mcp_app.streamable_http_app()

    # THE LIFESPAN CARRIES OVER, it is vital: the streamable session manager
    # of FastMCP initializes its task group INSIDE its app's lifespan. Our
    # wrapper must carry THE SAME lifespan — borrow it from inner, never
    # rewrite it. Without it: "Task group is not initialized", every request
    # 500, server alive but deaf.
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.routing import Mount

    return Starlette(
        # THE SAME LIFESPAN as the MCP app (the session manager lives there).
        lifespan=inner.router.lifespan_context,
        middleware=[Middleware(context_gate(authorize_fn))],
        routes=[Mount("/", app=inner)]), mcp_app, inner


def main() -> None:
    """The gated server launcher — the same variables processes.toml sets.

    OMNI_HTTP_PORT (default 8788), OMNI_STORAGE_ROOT, OMNI_NOTES_ROOT,
    METACOG_SURFACE (optional).
    """
    import uvicorn

    port = int(os.environ.get("OMNI_HTTP_PORT", "8788"))
    storage_root = os.environ.get(
        "OMNI_STORAGE_ROOT", os.path.expanduser("~/.omni/contexts"))
    notes_root = os.environ.get("OMNI_NOTES_ROOT", "")
    surface = os.environ.get("METACOG_SURFACE") or None

    outer, mcp_app, inner = build_gated_app(storage_root, notes_root, surface)
    print(f"[gate] omni gated per context on :{port} — store {storage_root}, "
          f"notes {notes_root or '(disabled)'}", flush=True)
    # WE SERVE `outer` — the wrapper WITH the context middleware. Serving
    # `inner` (the bare MCP app) would serve without the gate: measured,
    # initialize without header returned 200. The lifespan is inner's,
    # carried by outer (see build_gated_app).
    uvicorn.run(outer, host=os.environ.get("OMNI_HTTP_HOST", "127.0.0.1"),
                port=port)


if __name__ == "__main__":
    main()
