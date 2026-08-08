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
from typing import List, Optional, Tuple

from metacog.execution import ExecutionOutcome, ExecutionResult


class SimpleEncoder:
    """Deterministic simhash-style encoder. dim=32 by default.

    Each unique word maps to a ±1 vector ; the encoding of a text is
    the L2-normalized sum. Unrelated texts give cosines near 0,
    identical texts give cosine = 1.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self._cache: dict = {}

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
