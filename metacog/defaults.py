"""
MetaCog-Mem — default implementations of injectable protocols.

The encoder and the ACTION executor have simple deterministic defaults
defined here. The LLM is NOT — production uses ClaudeLLM (Haiku) from
`metacog.llm`. There is no stub LLM in the system : every generation
goes through Claude, no fallback.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from typing import List, Optional, Tuple

from metacog.execution import ExecutionOutcome, ExecutionResult

#: mnema's production embedder : multilingual (FR/EN) paraphrase model, 384
#: dims, ONNX on CPU via fastembed — no torch, no GPU, no API. (mnema measured a
#: 17x larger correct-vs-distractor margin than bge-small-en on French
#: paraphrase recall, same dims.)
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SimpleEncoder:
    """Deterministic simhash-style encoder. dim=32 by default.

    Each unique word maps to a ±1 vector ; the encoding of a text is
    the L2-normalized sum. Unrelated texts give cosines near 0,
    identical texts give cosine = 1. This is the TEST / fallback encoder :
    it has no semantics (a paraphrase is a different vector). Production
    uses `FastEmbedEncoder` (see `make_encoder`).
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self._cache: dict = {}

    @property
    def encoder_id(self) -> str:
        return f"simple:{self.dim}"

    def _word_vec(self, word: str) -> Tuple[float, ...]:
        if word not in self._cache:
            h = int(hashlib.md5(f"metacog:{word}".encode()).hexdigest(), 16)
            self._cache[word] = tuple(
                1.0 if (h >> i) & 1 else -1.0 for i in range(self.dim)
            )
        return self._cache[word]

    def encode(self, text: str) -> Tuple[float, ...]:
        words = text.lower().split()
        if not words:
            return tuple(0.0 for _ in range(self.dim))
        acc = [0.0] * self.dim
        for w in words:
            wv = self._word_vec(w)
            for i in range(self.dim):
                acc[i] += wv[i]
        norm = math.sqrt(sum(x * x for x in acc))
        if norm < 1e-12:
            return tuple(0.0 for _ in range(self.dim))
        return tuple(x / norm for x in acc)


class FastEmbedEncoder:
    """The production bi-encoder — the same stack as mnema : a real semantic
    model (multilingual paraphrase MiniLM by default) run LOCALLY as ONNX by
    fastembed. L2-normalised output, so cosine = dot. The model is imported
    and loaded lazily at construction (first run downloads ~0.2 GB to the
    fastembed cache) ; per-text results are memoised because the memory
    re-encodes the same query / content many times in one walk."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL,
                 cache_dir: Optional[str] = None, max_cache: int = 50_000) -> None:
        from fastembed import TextEmbedding
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self._cache: dict = {}
        self._max_cache = max_cache
        self.dim: int = len(self.encode("dimension probe"))

    @property
    def encoder_id(self) -> str:
        return f"fastembed:{self.model_name}"

    @staticmethod
    def _l2(v) -> Tuple[float, ...]:
        vals = [float(x) for x in v]
        n = math.sqrt(sum(x * x for x in vals))
        return tuple(x / n for x in vals) if n > 1e-12 else tuple(vals)

    def encode(self, text: str) -> Tuple[float, ...]:
        key = text or ""
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        arr = next(iter(self._model.embed([key or " "])))
        v = self._l2(arr)
        if len(self._cache) >= self._max_cache:
            self._cache.clear()
        self._cache[key] = v
        return v

    def encode_batch(self, texts: List[str]) -> List[Tuple[float, ...]]:
        """Batched path (bulk ingest / re-encode) — one ONNX call for many."""
        todo = [t for t in dict.fromkeys(texts) if (t or "") not in self._cache]
        if todo:
            for t, arr in zip(todo, self._model.embed([t or " " for t in todo])):
                self._cache[t or ""] = self._l2(arr)
        return [self._cache[t or ""] for t in texts]


def encoder_id(encoder) -> str:
    """The identity stamped into a persisted memory so a later process can
    tell whether the stored embeddings still match its encoder."""
    eid = getattr(encoder, "encoder_id", None)
    if eid:
        return str(eid)
    return f"{type(encoder).__name__}:{getattr(encoder, 'dim', '?')}"


def make_encoder(spec: Optional[str] = None, *, warn: bool = True):
    """Resolve the production encoder from `spec` or `METACOG_ENCODER` :

      auto (default)      FastEmbedEncoder if fastembed + its model are
                          available, else SimpleEncoder with a stderr warning
      fastembed           FastEmbedEncoder with the default multilingual model
                          (raises if unavailable — no silent downgrade)
      fastembed:<model>   or a bare `org/model` name : that fastembed model
      simple[:dim]        the deterministic hash encoder (tests / offline)

    The MCP server, the plugin launcher and every hook call this with no
    argument, so they all agree on ONE encoder for a given brain."""
    spec = (spec if spec is not None else os.environ.get("METACOG_ENCODER", "auto")).strip()
    if spec.startswith("simple"):
        dim = int(spec.split(":", 1)[1]) if ":" in spec else 32
        return SimpleEncoder(dim)
    if spec in ("", "auto", "fastembed") or spec.startswith("fastembed:") or "/" in spec:
        model = DEFAULT_EMBED_MODEL
        if spec.startswith("fastembed:") and spec.split(":", 1)[1]:
            model = spec.split(":", 1)[1]
        elif "/" in spec:
            model = spec
        try:
            return FastEmbedEncoder(model)
        except Exception as exc:
            if spec not in ("", "auto"):
                raise
            if warn:
                print(f"[metacog] fastembed encoder unavailable "
                      f"({type(exc).__name__}: {str(exc)[:80]}) — falling back to "
                      "SimpleEncoder (hash embeddings, no semantics). Install "
                      "`fastembed` or set METACOG_ENCODER=simple to silence.",
                      file=sys.stderr)
            return SimpleEncoder()
    raise ValueError(f"unknown encoder spec {spec!r} "
                     "(auto | fastembed[:model] | org/model | simple[:dim])")


class CrossEncoderReranker:
    """Optional LOCAL cross-encoder for the oblique judge's pre-filter.

    Unlike the bi-encoder (SimpleEncoder / any embedder), which encodes the
    query and each doc SEPARATELY into fixed vectors and compares by cosine, a
    cross-encoder feeds the (query, doc) pair JOINTLY through the model, so its
    attention makes every query token interact with every doc token. It answers
    "does THIS doc take THIS stance?", not just "same topic?" — and it does so
    for ZERO LLM tokens (small ONNX model on CPU). That is mnema's token lever :
    the relevance judgment leaves the LLM entirely for the confident cases.

    Backed by fastembed ; imported LAZILY so nothing pays the cost unless a
    reranker is actually wired into Memory(reranker=...). `rerank` returns one
    score per doc (higher = more relevant), aligned to `docs`."""

    def __init__(self, model_name: str =
                 "jinaai/jina-reranker-v2-base-multilingual") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, docs: List[str]) -> List[float]:
        if not docs:
            return []
        return [float(s) for s in self._model.rerank(query, docs)]


class NoOpExecutor:
    """Default ACTION executor : returns SUCCESS with an echo. Override
    for real-world side effects (web search, shell, API calls, etc.)."""

    def execute(self, action_content: str) -> ExecutionResult:
        return ExecutionResult(
            outcome=ExecutionOutcome.SUCCESS,
            result_content=f"[no-op] {action_content}",
        )
