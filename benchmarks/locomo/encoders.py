"""Real encoder adapters for the LoCoMo benchmark.

The default `SimpleEncoder` (simhash, dim=32) in metacog.defaults is
deterministic but semantically weak. For benchmarking we plug in a
real sentence-transformers model.
"""

from __future__ import annotations

from typing import Tuple

_MODEL_CACHE = {}


class SemanticEncoder:
    """SentenceTransformer-backed encoder. L2-normalized output."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        if model_name not in _MODEL_CACHE:
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        self.model = _MODEL_CACHE[model_name]
        self.dim = self.model.get_sentence_embedding_dimension()
        # Memoize per-string encodings. Entity beacons re-encode the SAME
        # short values thousands of times ("wealth", "john", "january",
        # "2023", date parts) — without this, --entities ingestion does
        # ~10k redundant CPU transformer passes per conversation.
        self._enc_cache: dict = {}

    def encode(self, text: str) -> Tuple[float, ...]:
        key = text or " "
        hit = self._enc_cache.get(key)
        if hit is not None:
            return hit
        vec = self.model.encode(
            key,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        out = tuple(float(x) for x in vec)
        self._enc_cache[key] = out
        return out
