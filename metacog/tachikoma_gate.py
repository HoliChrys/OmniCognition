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
   its wiki: every `.md` is ingested (`import_okf` for the doc, `ingest` for
   its content point), its refs linked into the RAG. The wiki lives where the
   notes already live — not in a parallel store that would drift. It is
   REFRESHED, not photographed (TAC-938): once per request the folder is
   compared with what was ingested (mtime + size per note), so a note added,
   corrected or deleted is reflected WITHOUT a restart. `ingest_notes()` is
   the explicit form, and its report says what the folder holds — including
   "this context has no notes", never an empty list read as an outage.
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

Storage: `<storage_root>/<ctx>/memory.pkl` — one memory per context, isolated,
persistent. `global` and `tachikoma.paralelle.GenAI` share NOTHING by accident.
Context and account names are VALIDATED before any path is built or any ACL
call leaves (fail-closed): a name carrying `/` or `..` would otherwise create
folders anywhere (measured: `contexts/tachikoma/paralelle/GenAI/`, nested).
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
import threading
from contextvars import ContextVar
from typing import Any, Optional

#: The contextvar of the served context. Set by the middleware, read by the proxy.
_current_ctx: ContextVar[str] = ContextVar("tachikoma_context", default="")

#: The contextvar of the caller's NARROW account ("" = the context's own).
_current_account: ContextVar[str] = ContextVar("tachikoma_account", default="")

#: One fresh object PER HTTP REQUEST (set by the middleware): the notes folder
#: is re-read once per request, not once per attribute access of the proxy.
#: `None` (no request — library use, tests) re-reads on every access.
_request_stamp: ContextVar[Optional[object]] = ContextVar(
    "tachikoma_request", default=None)

#: The root of the tachikoma context hierarchy — every ancestor chain ends
#: there. Its notes folder IS the notes_root's own `notes/` (see `notes_folder`).
ROOT_CONTEXT = "global"

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


def notes_folder(notes_root: str, ctx: str) -> Optional[str]:
    """THE MAPPING dotted context name → its `notes/` folder (TAC-938).

    ONE rule, no existence cascade — the folder need not exist (an absent
    folder is a context WITHOUT notes, said as such, never another folder
    tried in its place):

    - `notes_root` is the folder of the TREE ROOT context, named by its last
      path component. Deployed: `/opt/tachikoma-fs/global/tachikoma` → the
      tree root is `tachikoma`.
    - the tree root (`tachikoma`) and the hierarchy root (`global`, above
      it) → `<notes_root>/notes`. Both read the same folder: `global` has no
      folder of its own under this root (measured layout, 2026-10-04).
    - a descendant `tachikoma.paralelle.GenAI` →
      `<notes_root>/paralelle/GenAI/notes` — the dots are the slashes, and
      the first segment IS the root folder, never repeated in the path.
    - any other name (`demo.sandbox.alice`) → None: its notes do not live
      under this root. Not "no notes" — "not mine to read".
    """
    if not notes_root or not valid_context_name(ctx):
        return None
    root = os.path.normpath(os.path.expanduser(notes_root))
    tree = os.path.basename(root)
    if ctx in (tree, ROOT_CONTEXT):
        return os.path.join(root, "notes")
    head, _, rest = ctx.partition(".")
    if head != tree or not rest:
        return None
    return os.path.join(root, *rest.split("."), "notes")

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
        #: ctx → {doc_id: (mtime_ns, size)} of the notes as last ingested.
        #: In memory only: after a restart the first read compares with the
        #: STORE (content-addressed point ids), so nothing is re-ingested twice.
        self._notes_seen: dict[str, dict[str, tuple[int, int]]] = {}
        #: ctx → the REAL path of its notes folder, as resolved by the last
        #: pass that read it (TAC-255). Deployed, `notes/` is a FUSE link to
        #: the local disk, and the FUSE answers ENOENT for it while its
        #: backend is down: only the real path can CONFIRM an absence.
        #: In memory only — unknown after a restart (see `_scan_notes`).
        self._notes_real: dict[str, str] = {}
        #: ctx → the request stamp of its last notes check (once per request).
        self._notes_checked: dict[str, Optional[object]] = {}
        #: Contexts whose ancestor copies were swept (C3), once per process.
        self._c3_swept: set[str] = set()
        #: THE encoder and THE reranker of the gate (TAC-237), loaded once.
        self._models: Optional[tuple[Any, Any]] = None
        self._models_lock = threading.Lock()
        #: key → the lock that makes its `Memory` be born ONCE (TAC-228).
        #: Per key: the first access of one context never waits on another's.
        self._key_locks: dict[str, threading.Lock] = {}
        self._key_locks_lock = threading.Lock()

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
        """The `Memory` stored under `<root>/<key>/`, created on first use.

        ONE instance per key, whatever the concurrency (TAC-228). Measured
        before: 20 concurrent `remember` at the first access of a context
        built several `Memory` for it, one kept in the cache — the orphans
        still wrote the same `memory.pkl`, and since the merge-on-save (C2)
        each fact landed TWICE (+40 for 20). Double-checked under a lock PER
        KEY: the creation of one context never serialises another's.
        """
        m = self._memories.get(key)
        if m is not None:
            return m
        with self._key_locks_lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            m = self._memories.get(key)
            if m is None:
                m = self._new_memory(key)
                self._memories[key] = m
        return m

    def _new_memory(self, key: str) -> Any:
        from metacog.memory import Memory

        path = os.path.join(self._root, key, "memory.pkl")
        # Belt and braces after the name validation: the store never
        # leaves the root, whatever reached here.
        root = os.path.realpath(self._root)
        if not os.path.realpath(path).startswith(root + os.sep):
            raise RuntimeError(f"store {key!r} escapes the storage root")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        encoder, reranker = self.models()
        return Memory(storage_path=path, journal_path="auto",
                      encoder=encoder, reranker=reranker)

    def _context_memory(self) -> Any:
        """The memory of the context ITSELF — its account's, the manager's."""
        name = self._ctx_name()
        m = self._memory_at(name)
        if name not in self._c3_swept:
            self._c3_swept.add(name)
            self._forget_inherited_copies(name, m)
        # THE DEEPWIKI IS CHECKED ONCE PER REQUEST (TAC-938), not once per
        # process: a note added or corrected after the first access used to
        # stay out — or stale — until the `[omni]` process restarted. The
        # check is a stat of the folder; only a changed note costs an ingest.
        stamp = _request_stamp.get()
        if stamp is None or self._notes_checked.get(name) is not stamp:
            self._notes_checked[name] = stamp
            self._refresh_notes(name, m)
        return m

    def ingest_notes(self) -> dict:
        """The CONTRACT operation `ingest_notes(ctx)` — the served context's
        notes re-read NOW, and the report of what the folder holds.

        A REAL method (not delegated): the MCP tool `ingest_notes` finds it on
        the proxy's class; a bare `Memory` has none and says "unsupported".
        """
        name = self._ctx_name()
        m = self._memory_at(name)
        self._notes_checked[name] = _request_stamp.get()
        return self._refresh_notes(name, m)

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

    # ── the deepwiki: the context's OWN notes, kept in step with the folder ──
    #
    # TWO INGESTIONS PER NOTE, TWO ROLES — both kept, deliberately (TAC-938
    # names them so no engine silently does only one):
    #   1. the DOC — `import_okf(doc_id, body)`: frontmatter, refs, the
    #      `wiki_list` / `wiki_doc` / `wiki_where` surface. Lives in the journal.
    #   2. the CONTENT — `ingest(body)` as a RAG point. Measured gap: retrieve /
    #      walk never saw the notes (docs live in the journal, the RAG indexes
    #      points) — without it a question whose answer is IN a note returns
    #      empty. Its id CITES the note: `<doc_id>#<sha256[:12] of the body>`,
    #      so a recall hit names the `notes:*` doc it came from, and the id
    #      itself tells whether the store holds the CURRENT version.
    #
    # ONLY THE CONTEXT'S OWN NOTES: inheritance is served at query time by the
    # caller (TAC-936 / C3), never copied here.

    def _refresh_notes(self, ctx: str, m: Any) -> dict:
        """Bring the context's deepwiki in step with its `notes/` folder.

        - a note ADDED is ingested (doc + content point);
        - a note CORRECTED is re-ingested and every older content point of it
          is soft-forgotten (`forget_node`, superseded by the new one) — the
          old version is no longer retrievable;
        - a note DELETED leaves the wiki (doc removed) and its points are
          soft-forgotten.

        NO ANCESTOR SEEDING (TAC-936, rule C3): only `ctx`'s own folder is
        read. The ancestor copies an older gate left behind are swept by
        `_forget_inherited_copies`, once per (ctx, process), in
        `_context_memory`.

        Cheap when nothing moved: one stat per note, compared with the
        fingerprints (mtime_ns, size) of the last pass. After a restart there
        are no fingerprints: the store is the reference (content-addressed
        ids), so an unchanged note is recognised, not re-ingested.

        Returns the REPORT — `state` says what the folder is:
        `ok` (notes indexed), `no_notes` (the folder is absent or holds no
        `.md`: this context HAS no notes), `outside` (the context's notes do
        not live under this notes_root), `disabled` (no notes_root at all),
        `error` (the folder could not be READ — nothing was removed, nothing
        saved, the next pass retries). A note that cannot be read is listed
        in `errors`, never skipped silently, and is retried on the next pass.
        """
        folder = notes_folder(self._notes_root, ctx) if self._notes_root else None
        report: dict = {"ctx": ctx, "folder": folder, "state": "ok", "notes": 0,
                        "added": [], "updated": [], "removed": [],
                        "unchanged": 0, "errors": []}
        if not self._notes_root:
            report["state"] = "disabled"
            return report
        if folder is None:
            report["state"] = "outside"
            return report

        try:
            on_disk = _scan_notes(folder, self._notes_real.get(ctx))
        except OSError as exc:
            # An unconfirmed absence of a context whose notes are UNKNOWN
            # removes nothing either way: it is a context without notes.
            if not (isinstance(exc, _UnconfirmedAbsence)
                    and not self._notes_seen.get(ctx)
                    and not _own_note_points(ctx, m)):
                # A READ ERROR IS NEVER AN ABSENCE (TAC-243, TAC-255).
                # Measured: the FUSE answered EAGAIN under load (TAC-243),
                # then ENOENT during a `tachikoma-api` restart (TAC-255) —
                # each time the pass removed all the docs of GenAI. Nothing
                # is touched; the fingerprints stay those of the last good
                # pass, so the next one retries.
                report["state"] = "error"
                report["errors"].append({"doc_id": None,
                                         "error": f"{type(exc).__name__}: {exc}"})
                print(f"[gate] deepwiki of {ctx!r}: folder unreadable, nothing "
                      f"changed: {report['errors'][0]['error']}", flush=True)
                return report
            on_disk = {}
        else:
            try:   # strict: a component that does not answer keeps the old one
                self._notes_real[ctx] = os.path.realpath(folder, strict=True)
            except OSError:
                pass
        report["notes"] = len(on_disk)
        fingerprints = {doc: fp for doc, (_path, fp) in on_disk.items()}
        seen = self._notes_seen.get(ctx)
        if seen is not None and seen == fingerprints:
            report["unchanged"] = len(on_disk)
            report["state"] = "ok" if on_disk else "no_notes"
            return report

        own = _own_note_points(ctx, m)
        known_ids = {p.id for p in getattr(m, "points", [])}
        new_seen: dict[str, tuple[int, int]] = {}
        for doc_id, (path, fp) in on_disk.items():
            if seen is not None and seen.get(doc_id) == fp:
                new_seen[doc_id] = fp
                report["unchanged"] += 1
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    body = fh.read()
            except OSError as exc:
                report["errors"].append({"doc_id": doc_id,
                                         "error": f"{type(exc).__name__}: {exc}"})
                continue
            olds = own.get(doc_id.lower(), [])
            pid = f"{doc_id}#{hashlib.sha256(body.encode('utf-8')).hexdigest()[:12]}"
            current = [p for p in olds if p.id == pid]
            had_doc = _wiki_doc_exists(m, doc_id)
            if current and (had_doc or m.journal is None):
                keep = current[0]
                report["unchanged"] += 1
            else:
                try:
                    keep = _ingest_note(m, ctx, doc_id, body, pid, known_ids)
                except Exception as exc:  # noqa: BLE001 — said in the report, retried next pass
                    report["errors"].append({"doc_id": doc_id,
                                             "error": f"{type(exc).__name__}: {exc}"})
                    continue
                report["updated" if (olds or had_doc) else "added"].append(doc_id)
            # Every OTHER point of this note is an older version or a duplicate
            # left by a once-per-process gate: out of retrieval.
            for p in olds:
                if p.id != keep.id:
                    m.forget_node(p.id, f"note {doc_id} superseded by {keep.id}",
                                  superseded_by=keep.id)
            new_seen[doc_id] = fp

        # Deleted notes: what this gate ingested for ctx and the folder no
        # longer holds. Only OWN deepwiki traces are touched — never an
        # ancestor copy (C3's), never a doc another tool wrote.
        present = {d.lower() for d in on_disk}
        gone = {d for d in (seen or {}) if d.lower() not in present}
        for key, points in own.items():
            if key in present:
                continue
            for p in points:
                m.forget_node(p.id, f"note {key} removed from notes/")
            gone |= {d for d in _note_doc_ids(m) if d.lower() == key}
        for doc_id in sorted(gone):
            if m.journal is not None and _wiki_doc_exists(m, doc_id):
                m.journal.delete_wiki_doc(doc_id)
            report["removed"].append(doc_id)

        self._notes_seen[ctx] = new_seen
        changed = report["added"] or report["updated"] or report["removed"]
        if changed and getattr(m, "storage_path", None):
            m.save()
        if changed:
            print(f"[gate] deepwiki of {ctx!r}: +{len(report['added'])} "
                  f"~{len(report['updated'])} -{len(report['removed'])} "
                  f"({len(on_disk)} note(s) in {folder})", flush=True)
        if report["errors"]:
            print(f"[gate] deepwiki of {ctx!r}: {len(report['errors'])} "
                  f"note(s) not ingested: {report['errors']}", flush=True)
        report["state"] = "ok" if on_disk else "no_notes"
        return report

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
        if stale and getattr(m, "storage_path", None):
            m.save()
        if stale:
            print(f"[gate] {ctx!r}: {len(stale)} inherited note copie(s) "
                  f"soft-forgotten (rule C3)", flush=True)

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


class _UnconfirmedAbsence(OSError):
    """The notes folder answered ENOENT / ENOTDIR and nothing confirms it."""


def _scan_notes(folder: str, real: Optional[str] = None,
                ) -> dict[str, tuple[str, tuple[int, int]]]:
    """{doc_id: (path, (mtime_ns, size))} for every `.md` under `folder`,
    recursively (measured: GenAI's `notes/trace/` held 14 notes a flat read
    missed). The doc id keeps the RELATIVE path — collision-proof. An absent
    folder is an empty dict: a context without notes.

    ONLY an absent folder is absent (TAC-243): `ENOENT` / `ENOTDIR`, or a
    path that is not a directory. Any other `OSError` — on the folder or on
    any sub-folder (`EAGAIN` of the FUSE, `EIO`, `ENOTCONN`) — RAISES: an
    unreadable folder read as empty would remove every note it holds.
    `os.path.isdir` and a bare `os.walk` both swallow those errors.

    And ENOENT itself is only an absence once CONFIRMED (TAC-255): the FUSE
    ResourceFS answers ENOENT for a folder that exists whenever its backend
    cannot resolve it (measured: `tachikoma-api` restarting, 17 docs of
    GenAI removed then re-added). `real` is the folder's real path at the
    last pass that read it — deployed, the local disk the FUSE link points
    at. The disk confirms: absent there too → absent; present there → the
    FUSE lied, RAISES. A folder that is its own real path (no link) is
    answered by the filesystem that holds it → absent. No `real` (nothing
    read since the start) → `_UnconfirmedAbsence`: the caller removes
    nothing if the context has known notes. A notes folder deleted while
    the gate was down is therefore confirmed by no pass — its docs stay
    until the folder is read again, e.g. recreated EMPTY."""
    out: dict[str, tuple[str, tuple[int, int]]] = {}
    try:
        st = os.stat(folder)
    except (FileNotFoundError, NotADirectoryError) as exc:
        if real is None:
            raise _UnconfirmedAbsence(
                exc.errno, f"{exc.strerror}, unconfirmed: its real path is "
                "unknown (not read since the start)", folder) from exc
        if real != folder:
            try:
                st = os.stat(real)
            except (FileNotFoundError, NotADirectoryError):
                return out          # the disk itself says absent
            if stat.S_ISDIR(st.st_mode):
                raise OSError(exc.errno, f"{exc.strerror} through the FUSE, "
                              f"yet present on disk at {real}", folder) from exc
        return out
    if not stat.S_ISDIR(st.st_mode):
        return out

    def _fail(exc: OSError) -> None:
        raise exc

    for root, _dirs, files in sorted(os.walk(folder, onerror=_fail)):
        for filename in sorted(files):
            if not filename.endswith(".md"):
                continue
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, folder)[:-3].replace(os.sep, "/")
            try:
                st = os.stat(path)
                fp = (st.st_mtime_ns, st.st_size)
            except OSError:
                # Listed but not stat-able: kept, so the read fails LOUDLY
                # in the report instead of the note vanishing from the wiki.
                fp = (-1, -1)
            out[f"notes:{rel}"] = (path, fp)
    return out


def _own_note_points(ctx: str, m: Any) -> dict[str, list[Any]]:
    """The context's OWN live deepwiki content points, keyed by lowercased
    doc id. Own = tagged `ctx:<ctx>` (an ancestor copy carries its source's).
    `add_tag` lowercases, hence the key; the content-addressed id
    `<doc_id>#<hash>` keeps the case."""
    from metacog.epistemic import EpistemicState

    own_tag = f"ctx:{ctx}".lower()
    out: dict[str, list[Any]] = {}
    for p in getattr(m, "points", []):
        tags = p.tags or []
        if "deepwiki" not in tags or own_tag not in tags:
            continue
        if p.state in (EpistemicState.INVALID, EpistemicState.DEPRECATED):
            continue
        if "#" in p.id and p.id.startswith("notes:"):
            key = p.id.rsplit("#", 1)[0].lower()
        else:   # a point written before TAC-938: auto id, the doc in its tag
            key = next((t[len("note:"):] for t in tags
                        if t.startswith("note:notes:")), "")
        if key:
            out.setdefault(key, []).append(p)
    return out


def _wiki_doc_exists(m: Any, doc_id: str) -> bool:
    return m.journal is not None and m.journal.get_wiki_doc(doc_id) is not None


def _note_doc_ids(m: Any) -> list[str]:
    """The `notes:*` doc ids of the memory's journal ([] without one)."""
    if m.journal is None:
        return []
    return [d for d in m.journal.all_wiki_doc_ids() if d.startswith("notes:")]


def _ingest_note(m: Any, ctx: str, doc_id: str, body: str, pid: str,
                 known_ids: set) -> Any:
    """The two ingestions of one note — the DOC, then the CONTENT point."""
    m.import_okf(doc_id, body)
    # A body that comes BACK (A → B → A) finds its old id taken by the
    # forgotten point: a suffix, never a reuse of a forgotten node.
    free, n = pid, 1
    while free in known_ids:
        n += 1
        free = f"{pid}.{n}"
    # The engine's ingest() takes NO tags (measured: TypeError) — the MCP tool
    # adds them to the point AFTER creation (add_tag + journal tag index).
    p = m.ingest(body, kind="FACT", id=free)
    known_ids.add(p.id)
    p.add_tag(f"note:{doc_id}", f"ctx:{ctx}", "deepwiki")
    _log_tags(m, p)
    return p


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
            if warm is not None:
                await anyio.to_thread.run_sync(warm)
            _current_ctx.set(ctx)
            _current_account.set(account)
            # A fresh stamp: this request re-reads the notes once (TAC-938).
            _request_stamp.set(object())
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
