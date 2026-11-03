"""Shared plumbing of the storage-baseline oracle (OO0-5 / OO0-6, TAC2-1146).

ONE way to talk to a memory, for the builder, the oracle and the latency
probe: the PRODUCTION tool functions of `metacog.mcp_server.build_app`,
bound to a gate-style memory (`metacog.tachikoma_gate.ContextualMemory` by
default) that resolves the context and the narrow account from the same
contextvars the gate's middleware sets. No HTTP, no ACL round-trip: what is
measured is the memory, not the transport.

THE INJECTION POINT for another storage backend is `memory_factory`: a
callable `(storage_root, notes_root) -> memory` returning any object that
serves the gate's contract (attribute access resolves the memory of the
current context / account). Pass it as `module:callable` on the command
line (`--memory-factory`). The default builds today's pickle-backed
`ContextualMemory`. A Neo4j-backed factory (OO·V2·M3) plugs in here and the
oracle runs unchanged.

Guards, all fail-loud:
- no LLM: every `ClaudeLLM.generate` call is counted and answered "" — the
  count is written in every output and a non-zero count fails the run;
- offline: `HF_HUB_OFFLINE=1` by default (the fastembed models must be in
  the cache: encoder MiniLM-L12 384 + jina reranker, both production);
- deterministic ids: `uuid.uuid4` is seeded for the duration of a build.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import random
import subprocess
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
    os.environ.pop(_k, None)            # the oracle never reaches an LLM

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

#: The tree root of the oracle contexts: `notes_folder()` maps
#: `oracle.<x>` to `<notes_root>/<x>/notes` when `notes_root` ends in `oracle`.
TREE = "oracle"
CONTEXTS = ("oracle.small", "oracle.medium", "oracle.narrow")
#: The narrow account of `oracle.narrow` (a lobby member's own store).
NARROW_ACCOUNT = "oracle-member"
#: Views a query is asked under: (context, account or "" for the context's).
VIEWS: Dict[str, List[Tuple[str, str]]] = {
    "oracle.small": [("oracle.small", "")],
    "oracle.medium": [("oracle.medium", "")],
    "oracle.narrow": [("oracle.narrow", ""), ("oracle.narrow", NARROW_ACCOUNT)],
}


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain",
                                "--", "metacog"], capture_output=True,
                               text=True).stdout.strip()
        return out.stdout.strip() + ("+dirty-metacog" if dirty else "")
    except Exception:
        return "unknown"


# ── guards ────────────────────────────────────────────────────────────────
class LLMGuard:
    """Count (and neutralise) every ClaudeLLM.generate call."""

    calls = 0

    @classmethod
    def install(cls) -> None:
        from metacog import llm as _llm
        if getattr(_llm.ClaudeLLM.generate, "_oracle_guard", False):
            return

        def generate(self, *args: Any, **kwargs: Any) -> str:
            cls.calls += 1
            return ""
        generate._oracle_guard = True     # type: ignore[attr-defined]
        _llm.ClaudeLLM.generate = generate  # type: ignore[assignment]


@contextlib.contextmanager
def seeded_uuid(seed: int) -> Iterator[None]:
    """Deterministic `uuid.uuid4` (point ids are `<kind>_<uuid4[:8]>`)."""
    rng = random.Random(seed)
    original = uuid.uuid4
    uuid.uuid4 = lambda: uuid.UUID(int=rng.getrandbits(128), version=4)  # type: ignore[assignment]
    try:
        yield
    finally:
        uuid.uuid4 = original  # type: ignore[assignment]


# ── the memory factory (the injection point) ──────────────────────────────
_MODELS: Optional[Tuple[Any, Any]] = None


def shared_models() -> Tuple[Any, Any]:
    """(encoder, reranker): the PRODUCTION pair, loaded once per process.
    Refuses the SimpleEncoder fallback — a hash encoder is not a baseline."""
    global _MODELS
    if _MODELS is None:
        from metacog.defaults import make_encoder, make_reranker
        enc = make_encoder("fastembed")       # raises instead of falling back
        rr = make_reranker("fastembed")      # idem: no silent "no reranker"
        _MODELS = (enc, rr)
    return _MODELS


def default_memory_factory(storage_root: str, notes_root: Optional[str]) -> Any:
    """Today's backend: the gate's `ContextualMemory` over pickle stores,
    sharing ONE production encoder + reranker (as the gate does)."""
    from metacog.tachikoma_gate import ContextualMemory
    proxy = ContextualMemory(storage_root, notes_root)
    models = shared_models()
    proxy._models = models                    # the gate's own cache slot
    return proxy


def resolve_factory(spec: Optional[str]) -> Callable[[str, Optional[str]], Any]:
    if not spec:
        return default_memory_factory
    mod, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"--memory-factory {spec!r}: expected module:callable")
    return getattr(importlib.import_module(mod), attr)


# ── the driver: production tools under a (context, account) ──────────────
class Driver:
    """The MCP tool functions of `build_app`, called under a context."""

    def __init__(self, storage_root: str, notes_root: Optional[str],
                 factory: Optional[Callable[[str, Optional[str]], Any]] = None):
        LLMGuard.install()
        from metacog.mcp_server import build_app
        self.memory = (factory or default_memory_factory)(storage_root, notes_root)
        self.app = build_app(memory=self.memory, surface="all")
        self._tools = {t.name: t.fn for t in self.app._tool_manager.list_tools()}

    @contextlib.contextmanager
    def under(self, ctx: str, account: str = "") -> Iterator[None]:
        """Serve `ctx` (and the narrow `account`) like the gate's middleware:
        the two contextvars, and one more request for the notes refresh."""
        from metacog import tachikoma_gate as gate
        t1 = gate._current_ctx.set(ctx)
        t2 = gate._current_account.set(account)
        gate._requests_seen += 1
        try:
            yield
        finally:
            gate._current_account.reset(t2)
            gate._current_ctx.reset(t1)

    def tool(self, _tool_name: str, /, **kwargs: Any) -> Any:
        return self._tools[_tool_name](**kwargs)

    def mem(self) -> Any:
        """The concrete memory served under the current context/account."""
        resolve = getattr(self.memory, "_resolve", None)
        return resolve() if resolve is not None else self.memory


# ── fingerprints: what was built, so a diff is never blamed on the backend ──
def _vec_digest(h: "hashlib._Hash", v: Any) -> None:
    if v is None:
        h.update(b"-")
        return
    h.update(json.dumps([float(x) for x in v]).encode())


def store_fingerprint(memory: Any) -> Dict[str, Any]:
    """Content and vector digests of ONE memory's points (order-free).

    `content` covers id, kind, epistemic state, content, sorted tags,
    lineage and sequence links; `vectors` covers embedding_orig and both
    deltas at full float precision. A backend that keeps content but rounds
    vectors shows up as content-equal / vectors-different."""
    hc = hashlib.sha256()
    hv = hashlib.sha256()
    pts = sorted(getattr(memory, "points", []), key=lambda p: p.id)
    for p in pts:
        hc.update(json.dumps([p.id, p.kind.value, p.state.value, p.content,
                              sorted(p.tags), list(p.parents),
                              list(p.children), p.sequence_prev,
                              p.sequence_next]).encode())
        hv.update(p.id.encode())
        for fld in ("embedding_orig", "delta_active", "delta_latent"):
            _vec_digest(hv, getattr(p, fld, None))
    kinds: Dict[str, int] = {}
    for p in pts:
        kinds[p.kind.value] = kinds.get(p.kind.value, 0) + 1
    return {"points": len(pts), "by_kind": dict(sorted(kinds.items())),
            "content_sha256": hc.hexdigest(), "vectors_sha256": hv.hexdigest(),
            "t_clock": float(getattr(memory, "_t_clock", 0.0))}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_digest(root: str) -> Dict[str, str]:
    """sha256 of every file under `root` (relative path -> digest)."""
    out: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            out[os.path.relpath(full, root)] = file_sha256(full)
    return out
