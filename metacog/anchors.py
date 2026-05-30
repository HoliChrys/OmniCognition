"""
Concept anchors for kind-agnostic semantic perception.

Four fixed natural-language phrases are encoded once and cached ; each
Point is scored by cosine similarity to all four, yielding a behavior
profile (factual / reasoning / executable / topical). The downstream
orchestrator uses this profile to dispatch one of three FLAT paths
(executable tool / direct factual reply / staged metawalk reasoning) —
but THIS module only computes the perception, never routes.

Pure COMPUTATION : no LLM call. Source class is COMPUTATION for audit
(Cor. 5) — the anchor scores never enter the observation set O.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from metacog.epistemic import SourceClass
from metacog.geometry import cosine, effective_embedding


# Open-vocabulary anchor phrases. Adding a new behavior dimension later
# is a one-line addition here ; no schema change required.
ANCHOR_PHRASES: Dict[str, str] = {
    "factual":    "observation evidence witness fact recall",
    "reasoning":  "derivation inference reasoning thought step",
    "executable": "procedure runnable function code action tool",
    "topical":    "topic subject theme content",
}


class ConceptAnchors:
    """Cached anchor embeddings + cosine scorer.

    Build once per encoder/Memory ; reused across walks. Encoding cost is
    paid at __init__ (4 `encoder.encode` calls), scoring per point is
    four cosine calls — O(1) per node.
    """

    source: SourceClass = SourceClass.COMPUTATION

    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self._anchor_embeddings: Dict[str, Tuple[float, ...]] = {
            name: tuple(encoder.encode(phrase))
            for name, phrase in ANCHOR_PHRASES.items()
        }

    @property
    def names(self):
        return tuple(self._anchor_embeddings.keys())

    def score(self, point: Any, t_now: float) -> Dict[str, float]:
        """Behavior scores for `point` against each anchor.

        Each score is the cosine between the point's effective embedding
        (the live runtime view that includes apply_pull deltas) and the
        anchor's frozen embedding. Values are in [-1, 1] ; higher = the
        point reads more like that behavioral archetype.
        """
        eff = effective_embedding(point, t_now)
        return {
            name: cosine(eff, emb)
            for name, emb in self._anchor_embeddings.items()
        }

    def dominant(self, point: Any, t_now: float) -> str:
        """Convenience : the anchor name with the highest score."""
        scores = self.score(point, t_now)
        return max(scores, key=scores.get)
