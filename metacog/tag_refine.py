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
    "- the OBJECT/entity facet (what it is), e.g. health:body:finger\n"
    "- the CONDITION/state facet (what state it is in), e.g. "
    "health:condition:overweight\n"
    "The FIRST segment (the ROOT) MUST be one of this CLOSED list — never "
    "invent another root, pick the closest:\n"
    "  person place time health activity emotion mind relationship object "
    "food media event education work finance nature attribute communication\n"
    "Map consistently: a romantic/dating/crush/attraction signal -> "
    "relationship:romantic ; studying/school/learning -> education ; a city or "
    "country -> place:geography ; a physical ailment/symptom/limitation -> "
    "health:condition (ONLY for a genuine health issue, never mood/hobby). "
    "Sub-segments after the root are free. Each tag has >=2 segments. At most 4 "
    "tags per phrase. Output ONLY the comma-separated tags, no prose, no code "
    "fences.\n\n"
    "Phrase: {phrase}\nTags:"
)

# CLOSED root vocabulary : the only allowed first segment. Keeps a concept in
# ONE home (romance -> relationship, studies -> education) instead of scattering
# it across 141 invented roots, so a tag namespace is a deterministic handle.
_DOMAINS = {
    "person", "place", "time", "health", "activity", "emotion", "mind",
    "relationship", "object", "food", "media", "event", "education", "work",
    "finance", "nature", "attribute", "communication",
}

# Stray roots the LLM still emits -> their canonical home. Applied post-parse so
# even an off-vocabulary root is coerced into the closed set.
_ROOT_ALIASES = {
    # place
    "location": "place", "geography": "place", "environment": "place",
    "architecture": "place", "building": "place", "venue": "place",
    "infrastructure": "place", "city": "place", "terrain": "place",
    # health / body
    "medical": "health", "condition": "health", "anatomy": "health",
    "biology": "health", "body": "health", "symptom": "health",
    "substance": "health", "wellness": "health", "fitness": "health",
    # mind
    "psychology": "mind", "mental": "mind", "cognition": "mind",
    "personality": "mind", "perception": "mind", "mindset": "mind",
    "psychological": "mind",
    # relationship (incl. romance + social)
    "social": "relationship", "interaction": "relationship",
    "romance": "relationship", "romantic": "relationship",
    "dating": "relationship", "companionship": "relationship",
    "bond": "relationship", "affection": "relationship",
    "attraction": "relationship", "family": "relationship",
    # emotion (generic feelings only)
    "mood": "emotion", "sentiment": "emotion", "feeling": "emotion",
    # education
    "learning": "education", "skill": "education", "knowledge": "education",
    "academic": "education", "science": "education", "study": "education",
    # work / finance
    "occupation": "work", "career": "work", "employment": "work",
    "profession": "work", "business": "work", "job": "work",
    "commerce": "finance", "economics": "finance", "money": "finance",
    "transaction": "finance",
    # object
    "artifact": "object", "device": "object", "technology": "object",
    "tool": "object", "equipment": "object", "furniture": "object",
    "vehicle": "object", "instrument": "object", "product": "object",
    "material": "object", "software": "object", "gadget": "object",
    # media
    "content": "media", "music": "media", "audio": "media",
    "literature": "media", "narrative": "media", "information": "media",
    "document": "media", "entertainment": "media",
    # event
    "experience": "event", "competition": "event", "achievement": "event",
    "outcome": "event",
    # nature
    "animal": "nature", "plant": "nature", "creature": "nature",
    "weather": "nature",
    # attribute
    "quality": "attribute", "appearance": "attribute", "aesthetic": "attribute",
    "style": "attribute", "measurement": "attribute", "quantity": "attribute",
    "property": "attribute", "texture": "attribute", "size": "attribute",
    # person
    "demographic": "person", "demographics": "person", "role": "person",
    "people": "person", "name": "person",
    # activity
    "behavior": "activity", "action": "activity", "sport": "activity",
    "exercise": "activity", "hobby": "activity", "recreation": "activity",
    "task": "activity", "leisure": "activity",
    # communication
    "language": "communication", "speech": "communication",
}


def _canonicalize(tag: str) -> str:
    """Coerce a tag's ROOT into the closed domain vocabulary (sub-segments
    untouched). `relationship:romantic` stays ; `romance:crush:x` ->
    `relationship:crush:x` ; `medical:procedure` -> `health:procedure`."""
    segs = tag.split(":")
    root = segs[0]
    if root in _DOMAINS:
        return tag
    canon = _ROOT_ALIASES.get(root)
    if canon:
        segs[0] = canon
        return ":".join(segs)
    return tag                                  # unknown -> leave as-is


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
        if not (t and ":" in t and t not in _SKIP_TAGS):
            continue
        t = _canonicalize(t)                    # coerce root into closed vocab
        if t not in tags:
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
