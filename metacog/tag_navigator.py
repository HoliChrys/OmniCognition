"""
LLM-driven tag-scope resolution — a ReAct loop over the 4 tag tools.

Mechanically unioning every namespace a query's segments touch is dumb : the
generic segment "location" matches all 13 location:* namespaces and the scope
explodes to the whole cloud (the conv-47 "Connecticut" failure). The right
unit of intelligence is the LLM : it semantic-searches the tag index, looks up
where a hit SITS in the structured glossary, then — judging against the actual
input — KEEPS the probable namespaces and PRUNES the nonsense, iterating.

This module gives the LLM the four tag tools as a small action language and
runs the loop:

    SEMANTIC <query>   — nearest tag segments (embedding) -> their namespaces
    GLOSSARY <prefix>  — the children/structure under a namespace prefix
    FUZZY <pattern>    — Levenshtein-matched namespaces (typo/morphology)
    REGEX <pattern>    — regex-matched namespaces
    SELECT <ns, ns…>   — commit the final scope namespaces (ends the loop)

Each step the LLM sees the question + accumulated observations and issues ONE
action ; the loop executes it, appends the result, and asks again until SELECT
or a step budget. Fully failure-safe : no LLM, a parse error, or an empty
result falls back to the plain semantic union, so callers never break.
"""

from __future__ import annotations

import re
from typing import Any, List, Sequence

from metacog.tags import tag_glossary, parent_prefixes
from metacog.tag_index import TagIndex
from metacog.fuzzy import fuzzy_match


class TagNavigator:
    """The four tag tools over a memory's structured glossary, plus the
    LLM loop that drives them to a pruned scope."""

    def __init__(self, memory: Any, *, index: TagIndex = None) -> None:
        self.memory = memory
        # The navigable tree must include the LEAVES (the full tags), not only
        # the parent prefixes tag_glossary returns — otherwise the descent can
        # never reach location:geography:stamford (the leaf is where the answer
        # lives). Build every hierarchical tag PLUS all its ancestor prefixes.
        allns: set = set()
        for p in memory.points:
            for t in getattr(p, "tags", None) or []:
                if isinstance(t, str) and ":" in t:
                    allns.add(t)
                    for pref in parent_prefixes(t):
                        allns.add(pref)
        self.glossary: List[str] = sorted(allns, key=lambda s: (s.count(":"), s))
        self._gset = set(self.glossary)
        self.index = index or TagIndex(memory.encoder).build(memory.points)

    # ---- the 4 tools : each returns matching glossary NAMESPACES ----------
    def semantic(self, query: str, k: int = 5) -> List[str]:
        out: List[str] = []
        for r in self.index.search(query, self.memory.points, k=k):
            for ns in r["namespaces"]:
                if ns not in out:
                    out.append(ns)
        return out

    def glossary_under(self, prefix: str) -> List[str]:
        p = (prefix or "").strip().lower()
        if not p:
            return []
        return [ns for ns in self.glossary
                if ns == p or ns.startswith(p + ":")]

    def fuzzy(self, pattern: str) -> List[str]:
        pat = (pattern or "").strip().lower()
        if not pat:
            return []
        out: List[str] = []
        for ns in self.glossary:
            if any(fuzzy_match(pat, seg) for seg in ns.split(":")):
                out.append(ns)
        return out

    def regex(self, pattern: str) -> List[str]:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []
        return [ns for ns in self.glossary if rx.search(ns)]

    def exact(self, pattern: str) -> List[str]:
        pat = (pattern or "").strip().lower()
        return [ns for ns in self.glossary
                if pat == ns or pat in ns.split(":")]

    # ---- the LLM loop ----------------------------------------------------
    _ACTIONS = {
        "SEMANTIC": "semantic", "GLOSSARY": "glossary_under",
        "FUZZY": "fuzzy", "REGEX": "regex", "EXACT": "exact",
    }

    @staticmethod
    def _keep_most_specific(namespaces: Sequence[str]) -> List[str]:
        """Cross-of-search-and-structure cleanup : when both a catch-all
        ancestor ('location') and a specific descendant
        ('location:geography:stamford') are selected, keep ONLY the
        descendant — the generic ancestor is the noise that exploded the
        scope. A namespace with no selected descendant is kept as-is."""
        ns = [n for n in dict.fromkeys(namespaces) if n]
        out: List[str] = []
        for n in ns:
            # drop n if some OTHER selected ns is a strict descendant of n
            if any(o != n and (o == n or o.startswith(n + ":")) for o in ns):
                continue
            out.append(n)
        return out

    def _prompt(self, question: str, log: List[str]) -> str:
        return (
            "You scope a memory search to the right TAG NAMESPACES for a "
            "question. Tags are hierarchical (domain:subdomain:value). You have "
            "tools; issue exactly ONE per turn, then read the result and decide "
            "the next. At EVERY step judge each candidate ONLY by how well it "
            "fits THIS question/input — KEEP the few namespaces probable given "
            "the input, PRUNE everything that does not serve it (a generic "
            "catch-all like 'location' that matches everything is noise; prefer "
            "the specific namespace the input implies, e.g. the city the input "
            "is really about). When confident, SELECT the final scope.\n\n"
            "TOOLS (one per line, then stop):\n"
            "  SEMANTIC <phrase>   nearest tag segments by meaning -> namespaces\n"
            "  GLOSSARY <prefix>   the structure (children) under a namespace\n"
            "  FUZZY <word>        namespaces with a fuzzy-matching segment\n"
            "  REGEX <pattern>     namespaces matching a regex\n"
            "  EXACT <word>        namespaces containing exactly this segment\n"
            "  SELECT ns1, ns2     commit final scope namespaces (ends)\n\n"
            f"QUESTION: {question}\n\n"
            + "\n".join(log)
            + "\nNext action:"
        )

    def immediate_children(self, ns: str) -> List[str]:
        """Glossary namespaces exactly ONE level below `ns`."""
        d = ns.count(":")
        return [g for g in self.glossary
                if g.startswith(ns + ":") and g.count(":") == d + 1]

    def _llm_decide(self, question: str, options: Sequence[str], llm: Any
                    ) -> "tuple[list, list]":
        """One focused LLM call deciding, for each sibling namespace, whether
        to KEEP it as the scope (stop here — the whole subtree, i.e. the
        `ns:*` wildcard, is in) or DESCEND it (refine into its children).
        Branches the input doesn't imply are simply dropped.

        Returns (keep, descend). It is the LLM — not a fixed depth — that
        chooses when to stop : a broad question keeps `health:condition` (the
        oblique leaf comes along via ancestry) ; a precise one descends
        `location` toward the specific city. Returns ([],[]) on failure."""
        opts = list(dict.fromkeys(options))
        if not opts:
            return [], []
        prompt = (
            "Walking a tag tree to scope a search to what the QUESTION needs. "
            "For each tag namespace below choose ONE:\n"
            "  KEEP <ns>     — stop here; this whole branch (ns and everything "
            "under it) is the right scope.\n"
            "  DESCEND <ns>  — too broad; refine into its sub-branches next.\n"
            "Omit a namespace entirely to DROP it (off-topic, or a catch-all "
            "that matches everything). Use world knowledge (a city implies its "
            "state/country; a symptom implies its condition). You decide when a "
            "branch is specific enough — descending to a leaf is NOT required. "
            "Output one KEEP/DESCEND line per chosen namespace, nothing else.\n\n"
            f"QUESTION: {question}\n\n"
            "NAMESPACES:\n" + "\n".join(f"  {o}" for o in opts) + "\n"
        )
        try:
            raw = (llm.generate(prompt, max_tokens=220) or "").strip()
        except Exception:
            return [], []
        keep: List[str] = []
        descend: List[str] = []
        oset = set(opts)
        for ln in raw.splitlines():
            s = ln.strip()
            verb, _, arg = s.partition(" ")
            ns = arg.strip().lower()
            if ns not in oset:
                continue
            if verb.strip().upper() == "KEEP" and ns not in keep:
                keep.append(ns)
            elif verb.strip().upper() == "DESCEND" and ns not in descend:
                descend.append(ns)
        return keep, descend

    def resolve(self, question: str, llm: Any, *,
                max_depth: int = 4, k: int = 6) -> List[str]:
        """Descend the tag HIERARCHY, letting the LLM choose at each level
        whether to KEEP a branch as the scope (stop — the whole `ns:*` subtree
        is in, via ancestry matching) or DESCEND it (refine). It is the LLM,
        not a fixed depth, that decides when a branch is specific enough :

          • a BROAD question keeps `health:condition` — the oblique leaf
            (health:condition:macrodactyly) comes along for free ;
          • a PRECISE one descends `location -> location:geography ->
            location:geography:stamford`.

        Stopping is never forced (descending to a leaf is optional). Falls back
        to the semantic union on failure."""
        fallback = self._keep_most_specific(self.semantic(question, k=k))
        if not hasattr(llm, "generate") or not self.glossary:
            return fallback
        try:
            # top-level domains seen by the semantic search.
            roots: List[str] = []
            for ns in self.semantic(question, k=k):
                r = ns.split(":")[0]
                if r in self._gset and r not in roots:
                    roots.append(r)
            if not roots:
                return fallback
            resolved: List[str] = []
            frontier = roots
            for _ in range(max_depth):
                if not frontier:
                    break
                keep, descend = self._llm_decide(question, frontier, llm)
                resolved.extend(keep)
                children: List[str] = []
                for ns in descend:
                    kids = self.immediate_children(ns)
                    if kids:
                        children.extend(kids)
                    else:
                        resolved.append(ns)   # nothing to refine into : keep
                frontier = children
            resolved.extend(frontier)          # leftover at max_depth
            sel = self._keep_most_specific(resolved)
            # DISCRIMINATIVENESS gate (IDF) : a namespace matching most of the
            # cloud (person:name:james — every turn the speaker is in) is no
            # scope filter ; it would flood the OR-union. Drop the ubiquitous
            # ones, keeping the discriminative branches that actually narrow.
            n = len(self.memory.points) or 1
            disc = [ns for ns in sel if self._point_count(ns) <= 0.3 * n]
            sel = disc or sel
            return sel or fallback
        except Exception:
            return fallback

    def _point_count(self, ns: str) -> int:
        from metacog.tags import filter_points
        return len(filter_points(self.memory.points, [ns], mode="exact"))


def resolve_tag_scope(question: str, memory: Any, llm: Any, *,
                      max_steps: int = 5, k: int = 5,
                      index: TagIndex = None) -> List[str]:
    """Convenience : LLM-driven scope namespaces for `question`. Failure-safe
    (returns the plain semantic union if the loop can't run)."""
    try:
        nav = TagNavigator(memory, index=index)
        return nav.resolve(question, llm, max_steps=max_steps, k=k)
    except Exception:
        return []
