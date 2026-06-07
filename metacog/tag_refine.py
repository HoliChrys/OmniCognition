"""
Latent-sleep tag refinement — flat phrase keywords → hierarchical tags.

A FACT indexed with the phrase keyword "fingers too big" carries the right
words, but a query has to *guess* those words. The discriminative handle for
an INFERENCE question ("what are John's health problems?") is not the surface
phrase — it is the LATENT category the phrase belongs to. "fingers too big"
is, facet by facet, an OBJECT (`body:finger`) and a CONDITION
(`health:condition:swelling`). Tagged that way, the turn co-locates with every
other health-condition turn under the existing hierarchical-ancestry tag
matching (`metacog.tags`), so a walk SCOPED to `health:condition` surfaces it
regardless of the words the question used.

This runs in LATENT SLEEP (offline consolidation), not on the hot ingest
path : it is a slow, LLM-backed taxonomy pass that replays the cloud's phrase
keywords and crystallizes their namespace tags — the tag analog of the skill
distiller. Same robustness discipline as the rest of the LLM layer : strip
code fences, retry nothing here (the caller's sleep cycle is itself
re-runnable), and NEVER cache an empty result (one 529 must not permanently
suppress a phrase's tags).

Tags are INDEXING metadata (Cor. 5 : provenance GENERATOR) — never an
observation, never a stored relation, only a label. Edge-free.
"""

from __future__ import annotations

from typing import Any, List, Sequence

_REFINE_CACHE: dict = {}

# A phrase worth refining carries semantic content : multi-word phrases
# ("fingers too big") and rare single nouns. Pure structural / provenance
# tags ("fact", "atomic", "generated", "entity", "date", a bare speaker
# name) are NOT refined — they are already the namespace.
_SKIP_TAGS = {
    "fact", "action", "thought", "atomic", "generated", "entity", "date",
    "tool", "executed", "lateral_absorbed", "refined",
}


def _is_refinable(phrase: str) -> bool:
    p = (phrase or "").strip()
    if not p or p.lower() in _SKIP_TAGS:
        return False
    if ":" in p:                 # already hierarchical
        return False
    # multi-word phrase, or a single token of real length
    return len(p.split()) >= 2 or len(p) >= 5


_PROMPT = (
    "Decompose each phrase into HIERARCHICAL namespace tags of the form "
    "domain:subdomain:value (taxonomy paths), lowercase, ':' separator. "
    "Produce SEPARATE facets:\n"
    "- the OBJECT/entity facet (what it is), e.g. body:finger\n"
    "- the CONDITION/state facet (what state it is in), e.g. "
    "health:condition:overweight\n"
    "Emit a health:condition:<inferred-condition> tag ONLY when the phrase "
    "describes a GENUINE physical ailment, symptom, injury, bodily limitation, "
    "or clear health RISK (infer the latent condition, not the surface words) "
    "— NEVER for mood, wellbeing, hobbies, emotions, or positive states. Each "
    "tag must have at least one ':' (a namespace). At most 4 tags per phrase. "
    "Output ONLY the comma-separated tags for the single phrase, no prose, no "
    "code fences.\n\n"
    "Phrase: {phrase}\nTags:"
)


def refine_phrase(phrase: str, llm: Any) -> List[str]:
    """Return hierarchical namespace tags for one flat phrase.

    Returns [] when there is no usable LLM, the phrase is not refinable, or
    on failure — the caller treats refinement as strictly additive.
    """
    p = (phrase or "").strip()
    if not _is_refinable(p):
        return []
    key = p.lower()
    if key in _REFINE_CACHE:
        return list(_REFINE_CACHE[key])
    if not hasattr(llm, "generate"):
        return []
    try:
        raw = (llm.generate(_PROMPT.format(phrase=p), max_tokens=80) or "").strip()
    except Exception:
        return []
    # strip a stray code fence
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
    tags: List[str] = []
    for part in raw.replace("\n", ",").split(","):
        t = part.strip().strip("`").lower()
        # keep only well-formed hierarchical tags
        if t and ":" in t and t not in _SKIP_TAGS and t not in tags:
            tags.append(t)
    tags = tags[:4]
    if tags:                              # never cache an empty result
        _REFINE_CACHE[key] = list(tags)
    return tags


def refine_tags(phrases: Sequence[str], llm: Any) -> List[str]:
    """Refine a bag of phrases into the deduped union of their hierarchical
    tags. Order-stable, failure-safe."""
    out: List[str] = []
    for ph in phrases or []:
        for t in refine_phrase(ph, llm):
            if t not in out:
                out.append(t)
    return out
