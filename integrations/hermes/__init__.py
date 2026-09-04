"""
metacog as a Hermes Agent **memory provider**.

Hermes already has the shape this memory wants : it prefetches context before a
turn, syncs the turn afterwards, and extracts at session end. So instead of
bolting hooks on the side, metacog implements Hermes' own
`agent.memory_provider.MemoryProvider` interface — and because both are Python,
it imports the package directly (no subprocess, no bridge).

  initialize        open the brain (the SAME resolution as every other host :
                    a `.metacog-brain` marker > METACOG_STORAGE > ~/.metacog)
  prefetch          the cheap recall (retrieve, no walk, no LLM) → the string
                    Hermes fences into <memory-context> ; on a GAP it returns
                    the grounding directive instead of noise
  sync_turn         index both sides of the turn as episodic nodes
  on_pre_compress   what is about to leave the context window is stored first
  on_session_end    capture what is left, then ONE sleep() cycle + save
  on_session_switch save and re-key
  shutdown          save

Tools (`get_tool_schemas` / `handle_tool_call`) expose the primitives the agent
should drive itself : recall, remember, walk (oblique/multi-hop), forget,
mark_useful, and the OKF wiki entry point.

Install: see `integrations/README.md` or `python -m metacog.install install hermes`.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

# The repo root, so `metacog` imports whether this lives in ~/.hermes/plugins
# as a symlink or as a copy next to the package. `os.path.realpath` resolves the
# symlink Hermes installs, so the repo is found from either shape.
_HERE = os.path.dirname(os.path.realpath(__file__))
_ROOT = os.environ.get("METACOG_ROOT") or os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, os.path.join(_ROOT, "hooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:                                  # the real base class inside a Hermes install
    from agent.memory_provider import MemoryProvider as _Base
except Exception:                     # standalone (tests, docs, lint) — same surface
    class _Base:                      # type: ignore[no-redef]
        """Stub stand-in when Hermes is not importable."""

PROVIDER_NAME = "metacog"
_MAX_CHARS = 240


def _brain(cwd: Optional[str] = None) -> str:
    from _common import resolve_storage            # the shared host resolution
    return resolve_storage(cwd)


class MetacogMemory(_Base):
    """metacog (manifold + journal + OKF wiki) behind Hermes' memory contract."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        cfg = dict(config or {})
        cfg.update(kwargs)
        self.config = cfg
        self.storage = cfg.get("storage") or _brain(cfg.get("cwd"))
        self.k = int(cfg.get("k", 5))
        self.user_id = str(cfg.get("user_id") or os.environ.get("METACOG_USER")
                           or os.environ.get("USER") or "user")
        self.session_id = ""
        self._mem = None
        self._error: Optional[str] = None

    # -- identity ---------------------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        """Configured and ready : the package imports and the brain opens."""
        try:
            self._memory()
            return True
        except Exception as exc:
            self._error = repr(exc)[:200]
            return False

    # -- lifecycle --------------------------------------------------------

    def _memory(self):
        if self._mem is None:
            from _common import load_memory
            self._mem = load_memory(self.storage)
        return self._mem

    def initialize(self, session_id: str = "", **kwargs) -> None:
        """Open the brain once at agent startup. Never raises : a memory that
        cannot open must degrade to no memory, not to a broken agent."""
        self.session_id = session_id or self.session_id
        try:
            self._memory()
        except Exception as exc:
            self._error = repr(exc)[:200]

    def shutdown(self) -> None:
        self._save()

    def _save(self) -> None:
        try:
            if self._mem is not None:
                self._mem.save()
        except Exception:
            pass

    def on_session_switch(self, new_session_id: str = "", *,
                          parent_session_id: str = "", reset: bool = False,
                          rewound: bool = False, **kwargs) -> None:
        self._save()
        self.session_id = new_session_id or self.session_id

    # -- the turn loop ----------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "", **kwargs) -> str:
        """The recall Hermes fences into `<memory-context>` before the turn.

        Cheap path only (hybrid retrieve — no walk, no LLM). Returns "" when
        there is nothing to say, and the GAP directive when the memory is
        actively empty for this query : silence would let the model answer from
        its prior as if it remembered."""
        q = (query or "").strip()
        if len(q) < 8:
            return ""
        try:
            mem = self._memory()
            if mem.abstains(q):
                from host_bridge import GAP_DIRECTIVE
                return GAP_DIRECTIVE
            hits = [h for h in mem.retrieve(q, k=self.k) if h.get("id")]
        except Exception as exc:
            self._error = repr(exc)[:200]
            return ""
        if not hits:
            return ""
        lines = [f"- [{h['id']}] {str(h.get('content', ''))[:_MAX_CHARS]}" for h in hits]
        return ("metacog memory — relevant to this turn (rate what you use with "
                "metacog_mark_useful):\n" + "\n".join(lines))

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "",
                  messages: Optional[List[Dict[str, Any]]] = None,
                  **kwargs) -> None:
        """Persist a completed turn — both sides, as episodic nodes."""
        sid = session_id or self.session_id or "hermes"
        for content, role in ((user_content, "user"), (assistant_content, "agent")):
            self._ingest_message(content, role, sid)
        self._save()

    def _ingest_message(self, content: str, role: str, session_id: str) -> bool:
        text = (content or "").strip()
        if not text:
            return False
        try:
            mem = self._memory()
            for p in mem.points:                       # never double-index a turn
                tags = p.tags or []
                if f"session:{session_id}" in tags and f"role:{role}" in tags:
                    body = p.content.split(": ", 1)[-1] if ": " in p.content else p.content
                    if body.strip() == text:
                        return False
            mem.ingest_message(text, role=role, user_id=self.user_id,
                               session_id=session_id, block=True)
            return True
        except Exception as exc:
            self._error = repr(exc)[:200]
            return False

    # -- consolidation ----------------------------------------------------

    def on_pre_compress(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """What is about to leave the context window is stored FIRST — the
        point of a memory is that compaction stops being a loss."""
        n = self._capture(messages)
        self._save()
        return (f"metacog stored {n} turn(s) from the compacted span; recall them "
                "with metacog_recall." if n else "")

    def on_session_end(self, messages: Optional[List[Dict[str, Any]]] = None,
                       **kwargs) -> None:
        """Capture whatever the turn loop missed, then ONE offline cycle."""
        self._capture(messages or [])
        try:
            self._memory().sleep()
        except Exception as exc:
            self._error = repr(exc)[:200]
        self._save()

    def _capture(self, messages: List[Dict[str, Any]]) -> int:
        sid = self.session_id or "hermes"
        n = 0
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "")).lower()
            if role not in ("user", "assistant", "agent"):
                continue                                # tool/system turns are noise
            content = m.get("content")
            if isinstance(content, list):               # content blocks
                content = " ".join(b.get("text", "") for b in content
                                   if isinstance(b, dict))
            if self._ingest_message(content or "", "user" if role == "user" else "agent", sid):
                n += 1
        return n

    # -- tools ------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """OpenAI function-calling schemas for the primitives the agent drives."""
        def fn(name, desc, props, required):
            return {"type": "function", "function": {
                "name": name, "description": desc,
                "parameters": {"type": "object", "properties": props,
                               "required": required}}}
        return [
            fn("metacog_recall",
               "Search the persistent memory (cheap, no LLM). Call this BEFORE "
               "answering anything that could depend on past context. Returns "
               "hits with their node ids and a retrieval_id.",
               {"query": {"type": "string", "description": "what to look for"},
                "k": {"type": "integer", "description": "how many hits (default 5)"}},
               ["query"]),
            fn("metacog_walk",
               "Deep, uncertainty-governed multi-hop search for an OBLIQUE or "
               "multi-step question the cheap recall cannot answer. Costs LLM "
               "calls — use metacog_recall first.",
               {"query": {"type": "string"}}, ["query"]),
            fn("metacog_remember",
               "Store a DURABLE fact, decision, preference or constraint. Never "
               "store ephemeral chatter — the turn loop already indexes the "
               "conversation.",
               {"content": {"type": "string"},
                "kind": {"type": "string", "enum": ["FACT", "THOUGHT", "ACTION"]},
                "tags": {"type": "array", "items": {"type": "string"},
                         "description": "hierarchical index tags, e.g. ops:deploy"}},
               ["content"]),
            fn("metacog_forget",
               "Correct the memory: soft-invalidate ONE node (append-only, "
               "reversible). Use it when the user corrects a remembered fact, "
               "naming the successor node in superseded_by.",
               {"node_id": {"type": "string"}, "reason": {"type": "string"},
                "superseded_by": {"type": "string"}},
               ["node_id", "reason"]),
            fn("metacog_mark_useful",
               "Rate a past retrieval 0 (useless) / 1 / 2 (useful). This is the "
               "supervised signal that calibrates the memory's decay.",
               {"retrieval_id": {"type": "integer"},
                "score": {"type": "integer", "enum": [0, 1, 2]}},
               ["retrieval_id", "score"]),
            fn("metacog_wiki",
               "Write the OKF wiki: build or update a concept doc from memory "
               "node ids (the doc keeps live refs to them and follows them when "
               "they change).",
               {"doc_id": {"type": "string"}, "title": {"type": "string"},
                "node_ids": {"type": "array", "items": {"type": "string"}},
                "type": {"type": "string", "description": "OKF concept type"}},
               ["doc_id", "title", "node_ids"]),
        ]

    def handle_tool_call(self, tool_name: str,
                         arguments: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Dispatch a provider tool. Always returns a string (Hermes shows it to
        the model) ; an error is reported, never raised."""
        a = dict(arguments or {})
        try:
            mem = self._memory()
            if tool_name == "metacog_recall":
                hits = mem.retrieve(str(a.get("query", "")), k=int(a.get("k", self.k)))
                if not hits:
                    from host_bridge import GAP_DIRECTIVE
                    return GAP_DIRECTIVE
                rid = hits[0].get("retrieval_id")
                body = "\n".join(f"- [{h['id']}] {str(h.get('content',''))[:_MAX_CHARS]}"
                                 for h in hits if h.get("id"))
                return body + (f"\n(retrieval_id={rid} — rate it with "
                               "metacog_mark_useful)" if rid is not None else "")
            if tool_name == "metacog_walk":
                out = mem.walk(str(a.get("query", ""))) if hasattr(mem, "walk") else None
                if out is None:                          # the library entry point
                    from metacog.meta_walk import MetaWalker
                    w = MetaWalker(str(a.get("query", "")), mem)
                    last = w.step()
                    while not last.done:
                        last = w.step()
                    out = last.to_dict()
                ev = out.get("relevant_collected") or out.get("facts") or []
                return json.dumps({"evidence": ev[:10],
                                   "reasoning": out.get("reasoning_chain", [])[-3:]},
                                  ensure_ascii=False)[:4000]
            if tool_name == "metacog_remember":
                p = mem.ingest(str(a.get("content", "")), kind=str(a.get("kind", "FACT")))
                tags = [str(t) for t in (a.get("tags") or [])]
                if tags:
                    p.add_tag(*tags)
                    if mem.journal is not None:
                        mem.journal.log_tags(p.id, p.tags)
                self._save()
                return f"stored as {p.id}"
            if tool_name == "metacog_forget":
                r = mem.forget_node(str(a.get("node_id", "")), str(a.get("reason", "")),
                                    superseded_by=a.get("superseded_by"))
                self._save()
                return json.dumps(r, ensure_ascii=False)
            if tool_name == "metacog_mark_useful":
                mem.mark_useful(int(a.get("retrieval_id", 0)), int(a.get("score", 1)))
                return "rated"
            if tool_name == "metacog_wiki":
                r = mem.feed_wiki(str(a.get("doc_id", "")), str(a.get("title", "")),
                                  [str(n) for n in (a.get("node_ids") or [])],
                                  type=str(a.get("type", "note")))
                self._save()
                return json.dumps(r, ensure_ascii=False)
            return f"unknown tool {tool_name}"
        except Exception as exc:
            return f"metacog error in {tool_name}: {exc}"

    # -- introspection ----------------------------------------------------

    def status(self) -> Dict[str, Any]:
        try:
            mem = self._memory()
            docs = len(mem.journal.all_wiki_doc_ids()) if mem.journal is not None else 0
            return {"provider": PROVIDER_NAME, "brain": self.storage,
                    "points": len(mem.points), "wiki_docs": docs, "error": self._error}
        except Exception as exc:
            return {"provider": PROVIDER_NAME, "brain": self.storage,
                    "error": repr(exc)[:200]}


#: Hermes' loader looks for a MemoryProvider subclass in this module.
Provider = MetacogMemory


def register(ctx) -> None:
    """Directory-plugin entry point. The memory provider path already gives
    Hermes the tools ; this only adds the `/metacog` command so a user can see
    where the brain is and what is in it, and is failure-safe on an older ctx."""
    try:
        provider = MetacogMemory()

        def _status(*_a, **_k):
            s = provider.status()
            return (f"metacog — brain {s.get('brain')} · {s.get('points', '?')} points "
                    f"· {s.get('wiki_docs', '?')} wiki docs"
                    + (f" · error: {s['error']}" if s.get("error") else ""))

        if hasattr(ctx, "register_command"):
            ctx.register_command("metacog", _status,
                                 "where the metacog brain is and what is in it")
    except Exception:
        pass
