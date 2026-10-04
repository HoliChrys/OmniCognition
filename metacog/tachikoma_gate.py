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
   mix contexts.

3. The PER-CONTEXT DEEPWIKI — the `notes/` folder of the tachikoma context IS
   its wiki: every `.md` is ingested (`import_okf`) the first time the memory
   of that context is touched, its refs linked into the RAG. The wiki lives
   where the notes already live — not in a parallel store that would drift.

4. THE ACCOUNT (TAC-213) — the right to read is the ACCOUNT's, not the
   context's. Every caller operates under an account, carried by the
   `x-tachikoma-account` header. By default (header absent, or equal to the
   context) the account IS the context's — its manager's — and the caller
   is served the context memory, as before. A NARROWER account (an agent
   operating in a lobby under its own account) is served ITS OWN memory,
   `<storage_root>/<ctx>/accounts/<account>/memory.pkl`: it reads only what
   its account wrote. What it ingests is ALSO filed in the context memory,
   tagged `account:<account>` — the context tag, so the manager sees what
   the agents of its context learned. Mirrored today: `ingest` (and
   `remember`, which delegates to it). Other writes of a narrow account stay
   in its own memory — the safe direction (the manager sees less, nobody
   sees more).

   The gate does not VERIFY that the caller owns the account it claims: it
   listens on loopback behind the tachikoma router, which authenticates the
   token and decides the account (same trust boundary as the context
   header).

Storage: `<storage_root>/<ctx>/memory.pkl` — one memory per context, isolated,
persistent. `global` and `tachikoma.paralelle.GenAI` share NOTHING by accident.
Context and account names are VALIDATED before any path is built (fail-closed):
a name carrying `/` or `..` would otherwise create folders anywhere.
"""
from __future__ import annotations

import os
import re
from contextvars import ContextVar
from typing import Any, Optional

#: The contextvar of the served context. Set by the middleware, read by the proxy.
_current_ctx: ContextVar[str] = ContextVar("tachikoma_context", default="")

#: The contextvar of the caller's account ("" = the context's own account).
_current_account: ContextVar[str] = ContextVar("tachikoma_account", default="")

#: The header tachikoma sends — same value as mnema's `CTX_HEADER`, taken
#: verbatim so a proxy never speaks a dialect the server does not listen to.
CTX_HEADER = "x-tachikoma-context"

#: The account the caller operates under (TAC-213). Absent = the context's.
ACCOUNT_HEADER = "x-tachikoma-account"

#: A tachikoma context name: dotted segments, no `/`, no empty segment — so no
#: `..` and no absolute path can ever reach `os.path.join`.
_CTX_NAME = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*")

#: An account id (user or agent id, e-mail-like ids included): ONE path
#: component — no `/`, never starting with a dot, no `..`.
_ACCOUNT_NAME = re.compile(r"[A-Za-z0-9_@-][A-Za-z0-9_.@-]*")


def valid_context_name(name: str) -> bool:
    """True when `name` is a context name safe to use as a storage path."""
    return bool(name) and _CTX_NAME.fullmatch(name) is not None


def valid_account_name(name: str) -> bool:
    """True when `name` is an account id safe to use as ONE path component."""
    return (bool(name) and _ACCOUNT_NAME.fullmatch(name) is not None
            and ".." not in name)


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
        if not valid_context_name(name):
            raise RuntimeError(f"invalid context name {name!r} — refused "
                               "(fail-closed: no path is built from it)")
        return name

    def _account_name(self, ctx: str) -> str:
        """The NARROW account of the caller, or "" when it is the context's."""
        account = _current_account.get().strip()
        if not account or account == ctx:
            return ""
        if not valid_account_name(account):
            raise RuntimeError(f"invalid account name {account!r} — refused")
        return account

    def _memory_at(self, key: str) -> Any:
        """The `Memory` stored under `<root>/<key>/`, created on first use."""
        from metacog.defaults import make_encoder, make_reranker
        from metacog.memory import Memory

        m = self._memories.get(key)
        if m is None:
            path = os.path.join(self._root, key, "memory.pkl")
            # Belt and braces after the name validation: the store never
            # leaves the root, whatever reached here.
            root = os.path.realpath(self._root)
            if not os.path.realpath(path).startswith(root + os.sep):
                raise RuntimeError(f"store {key!r} escapes the storage root")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            m = Memory(storage_path=path, journal_path="auto",
                       encoder=make_encoder(), reranker=make_reranker())
            self._memories[key] = m
        return m

    def _context_memory(self) -> Any:
        """The memory of the context ITSELF — its account's, the manager's."""
        name = self._ctx_name()
        m = self._memory_at(name)
        # THE DEEPWIKI IS INGESTED ON EVERY FIRST ACCESS OF THE PROCESS, not
        # only at creation: a memory loaded from the pickle (server restart)
        # must find ITS notes too — otherwise the wiki only lives in the
        # process that created the memory, and every restart loses it.
        # `_ingested` bounds it to once per (process, context).
        self._ingest_notes(name, m)
        return m

    def _resolve(self) -> Any:
        """The memory the caller READS: its account's, under its context.

        The context's own account gets the context memory (deepwiki
        included). A narrow account gets ONLY its own memory — no notes, no
        other account's facts: it reads what its account wrote.
        """
        name = self._ctx_name()
        account = self._account_name(name)
        if not account:
            return self._context_memory()
        return self._memory_at(os.path.join(name, "accounts", account))

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

    # ── the context tag of a narrow account's writes ──────────────────
    def _mirrored_ingest(self, own: Any, account: str):
        """`ingest` for a narrow account: its memory, AND the context's.

        The point lands in the account's memory (what it will read back)
        and a copy lands in the context memory tagged `account:<account>`
        (what the manager reads). The tool adds its tags to the returned
        point afterwards: `_MirroredPoint` forwards them to both copies.
        """
        def ingest(content: str, *args: Any, **kwargs: Any) -> Any:
            p = own.ingest(content, *args, **kwargs)
            p.add_tag(f"account:{account}")
            _log_tags(own, p)
            ctx_mem = self._context_memory()
            # The explicit id stays the account's: in the context memory it
            # could collide with another account's — the engine names it.
            q = ctx_mem.ingest(content, kind=kwargs.get("kind", "FACT"))
            q.add_tag(f"account:{account}")
            _log_tags(ctx_mem, q)
            return _MirroredPoint(p, q, ctx_mem)
        return ingest

    def _mirrored_save(self, own: Any):
        """`save` for a narrow account: both stores the write touched."""
        def save(*args: Any, **kwargs: Any) -> Any:
            out = own.save(*args, **kwargs)
            ctx_mem = self._context_memory()
            if ctx_mem.storage_path:
                ctx_mem.save()
            return out
        return save

    # ── delegation ────────────────────────────────────────────────────
    def __getattr__(self, name: str) -> Any:
        m = self._resolve()
        account = self._account_name(self._ctx_name())
        if account and name == "ingest":
            return self._mirrored_ingest(m, account)
        if account and name == "save":
            return self._mirrored_save(m)
        return getattr(m, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        setattr(self._resolve(), name, value)


def _log_tags(m: Any, p: Any) -> None:
    """Index a point's tags in its memory's journal (no-op without one)."""
    try:
        if m.journal is not None:
            m.journal.log_tags(p.id, p.tags)
    except Exception:  # noqa: BLE001 — the tag index is an index, not the store
        pass


class _MirroredPoint:
    """The account's point, whose tags also reach its context-memory copy."""

    def __init__(self, own: Any, mirror: Any, mirror_memory: Any):
        self._own = own
        self._mirror = mirror
        self._mirror_memory = mirror_memory

    def add_tag(self, *tags: str) -> "_MirroredPoint":
        self._own.add_tag(*tags)
        self._mirror.add_tag(*tags)
        _log_tags(self._mirror_memory, self._mirror)
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._own, name)


def build_gated_app(storage_root: str, notes_root: str,
                    surface: Optional[str] = None) -> Any:
    """The omni MCP app, GATED per context — ready to serve over HTTP.

    This is the tachikoma deployment entry point: the FastMCP app of
    `build_app` (no tool changed), wrapped in a middleware that sets the
    context. `storage_root` isolates memories (`<root>/<ctx>/memory.pkl`);
    `notes_root` is the root of tachikoma context folders (`<root>/<ctx>/notes`
    is the context's wiki).
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
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.routing import Mount

    class _ContextGate(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            from starlette.responses import JSONResponse
            ctx = (request.headers.get(CTX_HEADER) or "").strip()
            if not ctx:
                return JSONResponse(
                    {"detail": "en HTTP le contexte est obligatoire : "
                               f"l'en-tête {CTX_HEADER} est absent"}, status_code=400)
            if not valid_context_name(ctx):
                return JSONResponse(
                    {"detail": f"nom de contexte invalide dans {CTX_HEADER} : "
                               f"{ctx!r} (segments [A-Za-z0-9_-] séparés par "
                               "des points)"}, status_code=400)
            account = (request.headers.get(ACCOUNT_HEADER) or "").strip()
            if account and not valid_account_name(account):
                return JSONResponse(
                    {"detail": f"nom de compte invalide dans {ACCOUNT_HEADER} : "
                               f"{account!r}"}, status_code=400)
            _current_ctx.set(ctx)
            _current_account.set(account)
            return await call_next(request)

    return Starlette(
        # THE SAME LIFESPAN as the MCP app (the session manager lives there).
        lifespan=inner.router.lifespan_context,
        middleware=[Middleware(_ContextGate)],
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
