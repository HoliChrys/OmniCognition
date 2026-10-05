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
   ONLY THE CONTEXT'S OWN NOTES (TAC-936, rule C3): inheritance is served AT
   QUERY TIME by the caller, which asks each ancestor's memory in turn
   (`memory_engines.recall_inherited` in tachikoma) — never copied here.

4. THE ACCOUNT (TAC-213) — the right to read is the ACCOUNT's, not only the
   context's. Every caller operates under an account. By default (no
   `x-tachikoma-account` header, or the context's own name) it is the
   context's account — its manager's — and the caller is served the context
   memory, as before. A NARROWER account (an agent operating in a lobby under
   its own account) sends `x-tachikoma-account: <its user_id>` and is served
   ITS OWN memory, `<storage_root>/<ctx>/accounts/<account>/memory.pkl`: it
   reads only what its account wrote. The claim is VERIFIED: the account must
   be the user the ACL authenticated — one can narrow to oneself, never borrow
   another account (403). What a narrow account ingests is ALSO filed in the
   context memory, tagged `account:<account>`: the context tag, so the manager
   sees what the agents of its context learned. Mirrored: `ingest` (and
   `remember`, which delegates to it). Other writes of a narrow account stay
   in its own memory — the safe direction (the manager sees less, nobody sees
   more).

5. THE MODELS ARE THE PROCESS'S, NOT THE CONTEXT'S (TAC-237) — every
   `Memory` of the gate shares ONE encoder and ONE reranker (`models()`). They
   hold no per-context state. Measured before: each context and each narrow
   account built its own pair, ~2.0 GB of ONNX sessions per key, never freed
   — omni idled at 7.16 GB. The pair is loaded ONCE, in a worker thread by the
   middleware, so the event loop never freezes for the 4-10 s of the load.

6. ONE SESSION, ONE CONTEXT (TAC-934) — an MCP session runs every call under
   the context of the `initialize` that opened it (its server task copied
   that request's contextvars). The gate binds each session id to its
   (context, account) and refuses (409) a call whose headers name another
   pair: the header and the memory served can never disagree. The four
   isolation lanes (recall, capture, wiki, index) are pinned end to end in
   `tests/test_context_isolation.py`.

Storage: `<storage_root>/<ctx>/memory.pkl` — one memory per context, isolated,
persistent. `global` and `tachikoma.paralelle.GenAI` share NOTHING by accident.
Context and account names are VALIDATED before any path is built or any ACL
call leaves (fail-closed): a name carrying `/` or `..` would otherwise create
folders anywhere (measured: `contexts/tachikoma/paralelle/GenAI/`, nested).
"""
from __future__ import annotations

import os
import re
import threading
from contextvars import ContextVar
from typing import Any, Optional

#: The contextvar of the served context. Set by the middleware, read by the proxy.
_current_ctx: ContextVar[str] = ContextVar("tachikoma_context", default="")

#: The contextvar of the caller's NARROW account ("" = the context's own).
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

# ── THE ACL — who may read which memory (mnema's `acl.py`, ported) ──────────
#
# omni replaced mnema and kept its header but not its ACL: the gate checked the
# header's PRESENCE, nothing else (TAC-214). The router (`/api/memory`) does no
# authorization by design — "l'autorisation est celle du MOTEUR" — and forwards
# the bearer token for exactly this. Same three questions, same sources, same
# fail-closed answers as mnema: the policy is tachikoma's, not re-invented here.

_API = os.environ.get("TACHIKOMA_API_URL", "http://127.0.0.1:8000").rstrip("/")

# NO COMMON NOTEBOOK (TAC-936, rule C3). mnema's `general` — readable by any
# valid token, outside the hierarchy — was the one name that escaped
# authorization (`MNEMA_GENERAL_CTX`). It is gone: a recall climbs ctx → its
# ancestors → `global`, nothing beside the chain. What everyone must read is
# written in `global`. `general` is now an ordinary name: no hierarchy entry,
# so the existence check refuses it like any unknown context.

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
        #: THE encoder and THE reranker of the gate (TAC-237), loaded once.
        self._models: Optional[tuple[Any, Any]] = None
        self._models_lock = threading.Lock()

    # ── the models: ONE pair per gate, shared by every memory ───────────
    def models(self) -> tuple[Any, Any]:
        """`(encoder, reranker)` — built on first call, then shared.

        Every `Memory` of the gate gets THESE instances: the encoder maps a
        text to its vector and the reranker a (query, doc) pair to its score,
        whatever the context — their memo caches are context-free too.
        Measured before (TAC-237): one pair per context AND per narrow
        account, encoder +640 MB, reranker +1.33 GB, never released.

        Locked: the middleware loads them in a worker thread while a tool on
        the event loop may ask for them — one load, never two.
        """
        with self._models_lock:
            if self._models is None:
                from metacog.defaults import make_encoder, make_reranker
                self._models = (make_encoder(), make_reranker())
            return self._models

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
            encoder, reranker = self.models()
            m = Memory(storage_path=path, journal_path="auto",
                       encoder=encoder, reranker=reranker)
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
        """The context's deepwiki: its OWN `notes/` folder, ingested once.

        Every `.md` becomes an OKF doc (`import_okf` — frontmatter + refs
        linked into the RAG), recursively. ONCE per (ctx, process).

        NO ANCESTOR SEEDING (TAC-936, rule C3). This used to COPY every
        ancestor's notes into the child's memory (`notes:<ancestor>/…`): a
        note corrected in the ancestor stayed stale in every child that had
        ingested it, since `_ingested` bounds ingestion to once per process.
        Inheritance is now served AT QUERY TIME, by the caller asking each
        ancestor's memory in turn; the inherited mark is the `origin_ctx` of
        the contract, not a doc-id prefix. The copies an older gate left
        behind are soft-forgotten here (`_forget_inherited_copies`).

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
        self._forget_inherited_copies(ctx, m)
        n_docs = n_points = 0
        folder = self._note_folder(ctx)
        if folder:
            for root, _dirs, files in sorted(os.walk(folder)):
                for filename in sorted(files):
                    if not filename.endswith(".md"):
                        continue
                    path = os.path.join(root, filename)
                    # The doc id keeps the RELATIVE path (collision-proof).
                    rel = os.path.relpath(path, folder)[:-3]
                    try:
                        with open(path, encoding="utf-8", errors="replace") as fh:
                            body = fh.read()
                        doc_id = f"notes:{rel}"
                        # TWO INGESTIONS, TWO ROLES — both asked for:
                        # 1. import_okf: the DOC (frontmatter, refs, wiki_where/
                        #    wiki_doc queryable) — the deepwiki structure.
                        m.import_okf(doc_id, body)
                        n_docs += 1
                        # 2. ingest: the CONTENT as a RAG point — measured gap:
                        #    retrieve/walk never saw the notes (docs live in the
                        #    journal, the RAG indexes points). Without this, a
                        #    question whose answer is IN a note returns empty.
                        #    The point is tagged with its doc id + its ctx so
                        #    the hit can be traced back to the note it came from.
                        try:
                            # The engine's ingest() takes NO tags (measured:
                            # TypeError) — the MCP tool adds them to the point
                            # AFTER creation (add_tag + journal tag index).
                            # We mirror the tool exactly.
                            p = m.ingest(body, kind="FACT")
                            p.add_tag(f"note:{doc_id}", f"ctx:{ctx}",
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
                              f"{ctx}/notes/{rel}.md", flush=True)
        if n_docs or n_points:
            print(f"[gate] deepwiki of {ctx!r}: {n_docs} doc(s) + "
                  f"{n_points} content point(s) ingested (own notes only)",
                  flush=True)

    @staticmethod
    def _forget_inherited_copies(ctx: str, m: Any) -> None:
        """Soft-forget the ancestor notes an older gate COPIED into `ctx`.

        Those content points carry `deepwiki` and the `ctx:<ancestor>` tag of
        their source. Left in place, a recall from the child would serve them
        as LOCAL memories — stale, and mislabeled. `forget_node` is reversible
        (state INVALID + ledger row), never a deletion; idempotent, since an
        invalidated point already carries `invalidated`.
        """
        # `add_tag` lowercases: `ctx:tachikoma.paralelle.GenAI` is stored as
        # `ctx:tachikoma.paralelle.genai` — compare in that form, or the
        # context's OWN notes would read as foreign and be forgotten.
        own = f"ctx:{ctx}".lower()
        stale = [p.id for p in getattr(m, "points", [])
                 if "deepwiki" in p.tags and "invalidated" not in p.tags
                 and any(t.startswith("ctx:") and t != own for t in p.tags)]
        for node_id in stale:
            m.forget_node(node_id, "TAC-936: ancestor note copied at ingestion; "
                                   "inheritance is served at query time")
        if stale:
            print(f"[gate] {ctx!r}: {len(stale)} inherited note copie(s) "
                  f"soft-forgotten (rule C3)", flush=True)

    def _note_folder(self, ctx: str) -> str:
        """The `notes/` folder of `ctx` itself — never an ancestor's — or "".

        The dotted name maps to the folder path; the FIRST segment often
        repeats the root name, so the without-first-segment form is tried
        first. A single-segment name is the root: its own folder if it
        exists, else the notes_root itself stands in.
        """
        sub = ctx.split(".")
        if len(sub) > 1:
            candidates = [os.path.join(self._notes_root, *sub[1:], "notes"),
                          os.path.join(self._notes_root, *sub, "notes")]
        else:
            candidates = [os.path.join(self._notes_root, *sub, "notes"),
                          os.path.join(self._notes_root, "notes")]
        return next((d for d in candidates if os.path.isdir(d)), "")

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


#: The MCP session header (streamable HTTP transport).
SESSION_HEADER = "mcp-session-id"


class _SessionBindings:
    """session id → the (context, account) it was opened under.

    Bounded like the SDK's own session table (`max_sessions`, 10 000 by
    default): past the bound the OLDEST binding is dropped, and a call on
    that session gets a 404 from the gate — the client re-initializes, as
    MCP prescribes for an unknown session. Fail-closed: a session the gate
    does not know is never served.
    """

    BOUND = 10_000

    def __init__(self) -> None:
        from collections import OrderedDict
        self._map: "OrderedDict[str, tuple[str, str]]" = OrderedDict()
        self._lock = threading.Lock()

    def bind(self, session: str, pair: tuple[str, str]) -> None:
        with self._lock:
            self._map[session] = pair
            while len(self._map) > self.BOUND:
                self._map.popitem(last=False)

    def get(self, session: str) -> Optional[tuple[str, str]]:
        with self._lock:
            return self._map.get(session)


def context_gate(authorize_fn: Any = None, warm: Any = None) -> Any:
    """The middleware class: context header, then the ACL, on EVERY request.

    `authorize_fn(token, ctx) -> user` defaults to `authorize` (resolved at call
    time); tests inject a fake. The check is blocking urllib — run in a thread
    so a cold 17 s ACL never freezes the other contexts' requests.

    `warm()` (the gate's `models`) runs in a thread too, AFTER the ACL said
    yes: the first authorized request loads the ONNX models there — 4-10 s
    measured on the event loop before (TAC-237), omni deaf to everyone. Once
    loaded it returns at once. Only an authorized caller can trigger the load.
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    sessions = _SessionBindings()

    class _ContextGate(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            ctx = (request.headers.get(CTX_HEADER) or "").strip()
            if not ctx:
                return JSONResponse(
                    {"detail": "en HTTP le contexte est obligatoire : "
                               f"l'en-tête {CTX_HEADER} est absent"}, status_code=400)
            # NAMES BEFORE ANYTHING (TAC-213): no path, no ACL call is built
            # from a name that is not a context / account name.
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
            import anyio
            try:
                user = await anyio.to_thread.run_sync(
                    authorize_fn or authorize, _bearer(request.headers), ctx)
            except Denied as e:
                return JSONResponse({"detail": str(e)}, status_code=e.status)
            # THE ACCOUNT IS VERIFIED, never trusted: a caller narrows to ITS
            # OWN account (the user the ACL just authenticated), or stays on
            # the context's. Naming another account would read its memory.
            if account and account != ctx and account != user:
                return JSONResponse(
                    {"detail": f"{user!r} ne peut pas opérer sous le compte "
                               f"{account!r} : seul son propre compte (ou celui "
                               "du contexte) est permis"}, status_code=403)
            # A SESSION SERVES THE CONTEXT IT WAS OPENED UNDER (TAC-934). The
            # MCP session's server task is started by the `initialize`
            # request and copies ITS contextvars: every later call of that
            # session runs under the context of the initialize, whatever
            # header it carries. Measured: a session opened under `iso-alpha`,
            # called with `x-tachikoma-context: iso-beta` (authorized for
            # beta), was served alpha's fact. So a session is bound to its
            # (context, account) and a request that names another pair is
            # refused — never served a memory its header did not ask for.
            # The context's own account is "" whether the header is absent
            # or names the context — the proxy reads both as the same memory.
            pair = (ctx, "" if account == ctx else account)
            session = (request.headers.get(SESSION_HEADER) or "").strip()
            if session:
                bound = sessions.get(session)
                if bound is None:
                    return JSONResponse(
                        {"detail": f"session MCP inconnue du gate : {session!r} "
                                   "— rouvrir une session (initialize)"},
                        status_code=404)
                if bound != pair:
                    return JSONResponse(
                        {"detail": f"la session {session!r} a été ouverte sous le "
                                   f"contexte {bound[0]!r} (compte {bound[1]!r}) : "
                                   f"elle ne sert pas {ctx!r} (compte {pair[1]!r}) "
                                   "— une session par contexte"}, status_code=409)
            if warm is not None:
                await anyio.to_thread.run_sync(warm)
            _current_ctx.set(ctx)
            _current_account.set(account)
            response = await call_next(request)
            opened = (response.headers.get(SESSION_HEADER) or "").strip()
            if opened and not session:
                sessions.bind(opened, pair)
            return response

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
        middleware=[Middleware(context_gate(authorize_fn, warm=proxy.models))],
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
