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

    def encode(self, text: str) -> Tuple[float, ...]:
        vec = self.model.encode(
            text or " ",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return tuple(float(x) for x in vec)
