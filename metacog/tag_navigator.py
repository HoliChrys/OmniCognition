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
        """Decide, with a ONE-LEVEL LOOKAHEAD, where to stop. For each branch
        we show its sub-branches (n+1) so the LLM judges whether descending
        actually buys NEEDED precision for the input:

          KEEP <ns>      — keep this branch; descending into its shown
                           sub-branches would NOT add relevant precision (this
                           level n was the right stop). The whole ns:* subtree
                           is the scope.
          DESCEND <child>— a specific sub-branch (n+1) is a better, more
                           precise match; go there.

        So stopping at depth n is only confirmed after looking at n+1 — if
        n+1 offers nothing more precise, n stands. Branches the input doesn't
        imply are dropped. Returns (keep_nodes, next_children). ([],[]) on
        failure / no LLM."""
        opts = list(dict.fromkeys(options))
        if not opts:
            return [], []
        # build the lookahead view : each option with its immediate children.
        lines: List[str] = []
        child_set: set = set()
        for o in opts:
            kids = self.immediate_children(o)
            for c in kids:
                child_set.add(c)
            shown = ", ".join(kids[:12]) if kids else "(leaf — no sub-branches)"
            lines.append(f"  {o}\n      sub-branches: {shown}")
        prompt = (
            "Walking a tag tree to scope a search to what the QUESTION needs. "
            "Each branch is shown WITH its sub-branches (one level deeper) so "
            "you can look ahead before deciding. For each branch choose ONE:\n"
            "  KEEP <ns>       — its sub-branches add NO needed precision for "
            "the question; stop here, the whole ns:* subtree is the scope.\n"
            "  DESCEND <child> — a specific sub-branch is a better, more precise "
            "match; name that child (must be one shown above).\n"
            "Drop (omit) anything off-topic or a catch-all matching everything. "
            "Use world knowledge (a city implies its state/country; a symptom "
            "implies its condition). Descending to a leaf is NOT required — "
            "stop as soon as the extra depth stops helping. Output one "
            "KEEP/DESCEND line per decision, nothing else.\n\n"
            f"QUESTION: {question}\n\n"
            "BRANCHES:\n" + "\n".join(lines) + "\n"
        )
        try:
            raw = (llm.generate(prompt, max_tokens=260) or "").strip()
        except Exception:
            return [], []
        keep: List[str] = []
        nxt: List[str] = []
        oset = set(opts)
        for ln in raw.splitlines():
            verb, _, arg = ln.strip().partition(" ")
            ns = arg.strip().lower()
            v = verb.strip().upper()
            if v == "KEEP" and ns in oset and ns not in keep:
                keep.append(ns)
            elif v == "DESCEND" and ns in child_set and ns not in nxt:
                nxt.append(ns)          # already the n+1 child to visit
        return keep, nxt

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
                # lookahead decision : KEEP nodes (n stands) + the n+1 children
                # the LLM judged worth descending into.
                keep, nxt = self._llm_decide(question, frontier, llm)
                resolved.extend(keep)
                # a descended child with no further sub-branches is a leaf : it
                # will simply be KEEPt at the next round (its lookahead shows no
                # sub-branches), so just carry it forward.
                frontier = nxt
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
