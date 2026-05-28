"""
MetaCog-Mem — signal detectors (OBSERVER → COMPUTATION → Observation).

Transforms raw user turns (OBSERVER) into typed Observations via
deterministic text and embedding operations (COMPUTATION). No LLM in
the detection loop : Corollary 5 protected.

Six families of signals are foreseen ; this module implements five.
The sixth (TIME OBSOLESCENCE) is triggered periodically by the sleep
cycle, not by individual user turns, and is left for a follow-up.

  1. REFORMULATION       polarity -1   user repeats a similar query
  2. CONTRADICTION       polarity -1   user explicitly negates
  3. CORROBORATION       polarity +1   user confirms and extends
  4. REPETITION_OF_NEED  polarity -1   memory had the info but didn't surface
  5. CONTINUATION        polarity +1   conversation flows on a new topic

Magnitudes are normalized to ±1 because the rest of the system only
reads the SIGN of polarity (counter increments are unit). The
`signal_type` carries the semantic differentiation for audit.

Cosine thresholds used here are angular conventions, not tuning knobs :
    cos(π/4) ≈ 0.707  (45° — highly similar)
    cos(π/3) = 0.5    (60° — moderately similar)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

from metacog.epistemic import (
    EpistemicState,
    Observation,
    Point,
    SourceClass,
)
from metacog.geometry import cosine, effective_embedding


# ---------------------------------------------------------------------------
# Angular conventions (cos of standard angles — pure math)
# ---------------------------------------------------------------------------

COS_HIGHLY_SIMILAR = math.cos(math.pi / 4)        # 45°  ≈ 0.7071
COS_MODERATELY_SIMILAR = math.cos(math.pi / 3)    # 60°  = 0.5


# ---------------------------------------------------------------------------
# Default lexicons (injectable — these are language config, not
# hyperparameters of the algorithm)
# ---------------------------------------------------------------------------

DEFAULT_NEGATION_MARKERS = frozenset({
    "no", "not", "never", "wrong", "false", "incorrect", "nope",
    "non", "pas", "jamais", "faux", "erreur", "erroné",
})

DEFAULT_FRUSTRATION_MARKERS = frozenset({
    "actually", "i meant", "i mean", "wait", "rather",
    "en fait", "je voulais dire", "plutôt",
})

DEFAULT_CONTINUATION_MARKERS = frozenset({
    "ok", "okay", "thanks", "thank you", "and", "then", "so",
    "merci", "et", "ensuite", "donc", "alors",
})

DEFAULT_TRANSITION_MARKERS = frozenset({
    "by the way", "another question", "different topic", "switching",
    "au fait", "autre question", "changement de sujet",
})

DEFAULT_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "for", "with", "on", "at", "by", "from",
    "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "do", "does", "did", "have", "has", "had",
    "le", "la", "les", "un", "une", "des", "du", "de", "d",
    "est", "sont", "était", "été", "à", "en", "pour", "avec",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "ce", "ça", "cette", "ces", "qui", "que",
})


# ---------------------------------------------------------------------------
# Encoder protocol (injected from outside)
# ---------------------------------------------------------------------------


class Encoder(Protocol):
    def encode(self, text: str) -> Tuple[float, ...]: ...


# ---------------------------------------------------------------------------
# Conversation log
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    """One turn of the conversation.

    `retrieved_point_ids` is populated for assistant turns — it is
    the R_final returned by `retrieve()` when the assistant produced
    this turn. Used by detectors to attribute polarity to the right
    points.
    """

    timestamp: float
    speaker: str            # "user" | "assistant"
    text: str
    embedding: Optional[Tuple[float, ...]] = None
    retrieved_point_ids: List[str] = field(default_factory=list)


@dataclass
class ConversationLog:
    turns: List[TurnRecord] = field(default_factory=list)

    def record(self, turn: TurnRecord) -> None:
        self.turns.append(turn)

    def previous_user_query(self, before_ts: float) -> Optional[TurnRecord]:
        for t in reversed(self.turns):
            if t.timestamp < before_ts and t.speaker == "user":
                return t
        return None

    def previous_assistant_turn(self, before_ts: float) -> Optional[TurnRecord]:
        for t in reversed(self.turns):
            if t.timestamp < before_ts and t.speaker == "assistant":
                return t
        return None


# ---------------------------------------------------------------------------
# Tokenization helpers (pure COMPUTATION)
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ']+")


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _content_words(text: str, stopwords) -> set:
    return {w for w in _tokens(text) if w not in stopwords}


def _contains_phrase(text: str, markers) -> bool:
    lower = text.lower()
    return any(m in lower for m in markers)


def _has_token(text: str, markers) -> bool:
    return bool(set(_tokens(text)) & set(markers))


def _starts_with_any(text: str, markers) -> bool:
    lower = text.strip().lower()
    return any(lower.startswith(m) for m in markers)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def detect_reformulation(
    new_user_turn: TurnRecord,
    log: ConversationLog,
    transition_markers=DEFAULT_TRANSITION_MARKERS,
) -> Optional[Observation]:
    """User re-asks a query highly similar to the immediately previous
    one, without a topic-change marker → previous answer didn't satisfy.
    """
    prev = log.previous_user_query(new_user_turn.timestamp)
    if prev is None or prev.embedding is None or new_user_turn.embedding is None:
        return None
    if _contains_phrase(new_user_turn.text, transition_markers):
        return None

    sim = cosine(new_user_turn.embedding, prev.embedding)
    if sim < COS_HIGHLY_SIMILAR:
        return None

    prev_assistant = log.previous_assistant_turn(new_user_turn.timestamp)
    if prev_assistant is None or not prev_assistant.retrieved_point_ids:
        return None

    return Observation(
        source=SourceClass.COMPUTATION,
        signal_type="reformulation",
        polarity=-1.0,
        target_node_ids=list(prev_assistant.retrieved_point_ids),
        timestamp=new_user_turn.timestamp,
        raw_content=f"cos(q_t, q_t-1)={sim:.3f} >= {COS_HIGHLY_SIMILAR:.3f}",
    )


def detect_contradiction(
    new_user_turn: TurnRecord,
    log: ConversationLog,
    negation_markers=DEFAULT_NEGATION_MARKERS,
    frustration_markers=DEFAULT_FRUSTRATION_MARKERS,
    stopwords=DEFAULT_STOPWORDS,
) -> Optional[Observation]:
    """User says something negative AND shares content words with the
    previous assistant turn → the assistant's claim is being rejected.
    """
    prev_assistant = log.previous_assistant_turn(new_user_turn.timestamp)
    if prev_assistant is None or not prev_assistant.retrieved_point_ids:
        return None

    has_neg = _has_token(new_user_turn.text, negation_markers)
    has_frust = _contains_phrase(new_user_turn.text, frustration_markers)
    if not (has_neg or has_frust):
        return None

    new_words = _content_words(new_user_turn.text, stopwords)
    asst_words = _content_words(prev_assistant.text, stopwords)
    overlap = new_words & asst_words
    if not overlap:
        return None

    return Observation(
        source=SourceClass.COMPUTATION,
        signal_type="contradiction",
        polarity=-1.0,
        target_node_ids=list(prev_assistant.retrieved_point_ids),
        timestamp=new_user_turn.timestamp,
        raw_content=f"neg/frust + overlap={sorted(overlap)}",
    )


def detect_corroboration(
    new_user_turn: TurnRecord,
    log: ConversationLog,
    stopwords=DEFAULT_STOPWORDS,
    negation_markers=DEFAULT_NEGATION_MARKERS,
    frustration_markers=DEFAULT_FRUSTRATION_MARKERS,
) -> Optional[Observation]:
    """User confirms (shares words with the assistant's answer) AND
    extends (adds words not in the answer), without any negation.
    """
    prev_assistant = log.previous_assistant_turn(new_user_turn.timestamp)
    if prev_assistant is None or not prev_assistant.retrieved_point_ids:
        return None

    if _has_token(new_user_turn.text, negation_markers):
        return None
    if _contains_phrase(new_user_turn.text, frustration_markers):
        return None

    new_words = _content_words(new_user_turn.text, stopwords)
    asst_words = _content_words(prev_assistant.text, stopwords)
    if not new_words or not asst_words:
        return None
    overlap = new_words & asst_words
    extension = new_words - asst_words
    if not overlap or not extension:
        return None

    return Observation(
        source=SourceClass.COMPUTATION,
        signal_type="corroboration",
        polarity=+1.0,
        target_node_ids=list(prev_assistant.retrieved_point_ids),
        timestamp=new_user_turn.timestamp,
        raw_content=f"overlap={sorted(overlap)}, ext={sorted(extension)}",
    )


def detect_continuation(
    new_user_turn: TurnRecord,
    log: ConversationLog,
    continuation_markers=DEFAULT_CONTINUATION_MARKERS,
    negation_markers=DEFAULT_NEGATION_MARKERS,
    frustration_markers=DEFAULT_FRUSTRATION_MARKERS,
) -> Optional[Observation]:
    """Turn starts with a continuation marker, no negation/frustration,
    and topic clearly shifted from the previous user query
    → the assistant got the previous turn right, conversation moves on.
    """
    prev_assistant = log.previous_assistant_turn(new_user_turn.timestamp)
    if prev_assistant is None or not prev_assistant.retrieved_point_ids:
        return None

    if not _starts_with_any(new_user_turn.text, continuation_markers):
        return None
    if _has_token(new_user_turn.text, negation_markers):
        return None
    if _contains_phrase(new_user_turn.text, frustration_markers):
        return None

    prev_user = log.previous_user_query(new_user_turn.timestamp)
    if (prev_user is not None
            and prev_user.embedding is not None
            and new_user_turn.embedding is not None):
        sim = cosine(new_user_turn.embedding, prev_user.embedding)
        if sim >= COS_MODERATELY_SIMILAR:
            # Same topic — not a continuation, the user is still on it.
            return None

    return Observation(
        source=SourceClass.COMPUTATION,
        signal_type="continuation",
        polarity=+1.0,
        target_node_ids=list(prev_assistant.retrieved_point_ids),
        timestamp=new_user_turn.timestamp,
        raw_content="continuation marker + topic shift",
    )


def detect_repetition_of_need(
    new_user_turn: TurnRecord,
    log: ConversationLog,
    points: Sequence[Point],
    t_now: float,
) -> Optional[Observation]:
    """The user provides info that an existing memory point ALREADY
    captured — but that point was NOT retrieved in the last assistant
    turn. The memory had it; we missed it. Penalize the missed point.
    """
    if new_user_turn.embedding is None:
        return None

    prev_assistant = log.previous_assistant_turn(new_user_turn.timestamp)
    used_ids = set(prev_assistant.retrieved_point_ids) if prev_assistant else set()

    matches: List[str] = []
    for p in points:
        if p.id in used_ids:
            continue
        if p.state in {EpistemicState.INVALID, EpistemicState.DEPRECATED}:
            continue
        eff = effective_embedding(p, t_now)
        sim = cosine(new_user_turn.embedding, eff)
        if sim > COS_HIGHLY_SIMILAR:
            matches.append(p.id)

    if not matches:
        return None

    return Observation(
        source=SourceClass.COMPUTATION,
        signal_type="repetition_of_need",
        polarity=-1.0,
        target_node_ids=matches,
        timestamp=new_user_turn.timestamp,
        raw_content=f"unused matches: {matches}",
    )


# ---------------------------------------------------------------------------
# Orchestration : run every detector on a new user turn
# ---------------------------------------------------------------------------


def analyze_user_turn(
    new_user_turn: TurnRecord,
    log: ConversationLog,
    points: Sequence[Point],
    encoder: Optional[Encoder] = None,
) -> List[Observation]:
    """Run every detector. Returns the non-None observations.

    If `new_user_turn.embedding` is None and an encoder is provided,
    embeds the text in place (idempotent).
    """
    if new_user_turn.embedding is None and encoder is not None:
        new_user_turn.embedding = tuple(encoder.encode(new_user_turn.text))

    candidates = [
        detect_reformulation(new_user_turn, log),
        detect_contradiction(new_user_turn, log),
        detect_corroboration(new_user_turn, log),
        detect_continuation(new_user_turn, log),
        detect_repetition_of_need(new_user_turn, log, points, new_user_turn.timestamp),
    ]
    return [obs for obs in candidates if obs is not None]
