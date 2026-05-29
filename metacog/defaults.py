"""
MetaCog-Mem — default implementations of injectable protocols.

These are deliberately simple so the system runs out-of-the-box with
ZERO external dependencies. Callers should override them with real
implementations (sentence-transformers for the encoder, Claude/GPT
for the LLM, real executors for ACTIONs) for production use.

All defaults are DETERMINISTIC (no random state, no LLM call) so
behavior is reproducible.
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


class SimpleLLM:
    """Deterministic LLM stub. Implements all three injectable
    protocols (ReasoningLLM + LLMExtractor) so the system runs without
    a real model.

    Override for production : plug Claude / GPT / a local model that
    implements the same methods.
    """

    def __init__(self) -> None:
        self._answer_history: List[str] = []

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Deterministic free-text generator used by meta_walk.

        Stub : extracts the tail of the prompt (the last data-bearing
        line) and returns its first `max_tokens // 2` words. Produces
        non-empty deterministic text so tests can exercise the walk
        without an LLM. Override with a Claude-backed generator in
        production.
        """
        # Take the last non-empty line of the prompt as context.
        lines = [ln.strip() for ln in (prompt or "").splitlines() if ln.strip()]
        if not lines:
            return ""
        words = lines[-1].split()
        # ≈ 2 words per token for a deterministic stub.
        return " ".join(words[: max(1, max_tokens // 2)])

    def synthesize_step(
        self,
        query: str,
        previous_answer: Optional[str],
        new_point_content: str,
    ) -> str:
        """Concatenate previous answer with the new content's distinctive
        words. Stabilizes when new content adds nothing."""
        prev_words = set((previous_answer or "").lower().split())
        new_words = [
            w for w in new_point_content.split()
            if w.lower() not in prev_words
        ]
        if previous_answer is None:
            return new_point_content
        if not new_words:
            return previous_answer
        return f"{previous_answer} {' '.join(new_words[:5])}".strip()

    def propose_action(
        self, query: str, current_answer: str
    ) -> Optional[str]:
        """Default : never propose an action. Override to enable
        automatic action generation."""
        return None

    def extract_common(self, texts) -> str:
        """Word-set intersection, preserving the order of the first text."""
        texts = list(texts)
        if len(texts) < 2:
            return ""
        word_sets = [set(t.lower().split()) for t in texts]
        common = word_sets[0].copy()
        for ws in word_sets[1:]:
            common &= ws
        if not common:
            return ""
        seen: set = set()
        out: List[str] = []
        for w in texts[0].lower().split():
            if w in common and w not in seen:
                out.append(w)
                seen.add(w)
        return " ".join(out)

    def remove_overlap(self, full: str, common: str) -> str:
        cset = set(common.lower().split())
        return " ".join(w for w in full.lower().split() if w not in cset)


class NoOpExecutor:
    """Default ACTION executor : returns SUCCESS with an echo. Override
    for real-world side effects (web search, shell, API calls, etc.)."""

    def execute(self, action_content: str) -> ExecutionResult:
        return ExecutionResult(
            outcome=ExecutionOutcome.SUCCESS,
            result_content=f"[no-op] {action_content}",
        )
