"""
OKF wiki layer — a bidirectional, continuously-evolving bridge between the wiki
(Google OKF markdown : concept-per-file, YAML frontmatter) and the RAG nodes.

Each wiki document keeps the ids of the RAG nodes it was built from, in BOTH the
OKF frontmatter (`refs:` — a doc can cite many) AND inline in the body as
Obsidian-style wikilinks `[[node_id]]`; tags live in the frontmatter and inline
(`#tag`). The link table lives in the journal (`wiki_refs`), so the two evolve
together : when a node is forgotten→merged (alias redirect) or superseded, the
wiki refs are rewritten ; when the wiki gains new prose, it is ingested back into
the RAG with the doc's tags as context. This module holds only the pure
render/parse/rewrite helpers — the DB + sync live in journal.py / memory.py.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import yaml

#: Obsidian-style wikilink to a RAG node : [[node_id]].
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)\]\]")

#: Inline `#tag` in a body (not a markdown heading : no space after the #).
#: Tags are hierarchical and may carry paths (`#file:metacog/memory.py`,
#: `#file:.gitignore`) ; a trailing sentence period is NOT part of the tag.
INLINE_TAG_RE = re.compile(
    r"(?<![\w#])#([a-z0-9_.][a-z0-9_:\-./]*[a-z0-9_]|[a-z0-9_])", re.IGNORECASE)

#: Structural tags that are plumbing, not knowledge — kept off the wiki surface.
_STRUCTURAL_TAGS = {
    "fact", "thought", "action", "event", "refined", "invalidated",
    "tool", "eager", "deprecated", "generated", "atomic", "entity",
    "proposed", "established",
}

#: OKF concept types accepted out of the box ; any OTHER type is preserved as
#: a vocabulary PROPOSAL (never rejected, never silently canonical) until vetted.
KNOWN_OKF_TYPES = frozenset({"note", "topic", "tool"})

# -- explicit reason codes : nothing degrades silently -------------------------
#: A ref whose target node does not exist (never did, or was hard-removed).
REF_MISSING = "missing_node"
#: A ref whose target was soft-invalidated (forget / contradiction) with no successor.
REF_INVALIDATED = "invalidated"
#: A ref whose target is a DEPRECATED node.
REF_DEPRECATED = "deprecated"
#: A ref that was redirected old->new by a merge (informational, not stale).
REF_REDIRECTED = "redirected"
#: A doc imported without any YAML frontmatter (parsed as a bare note).
DOC_NO_FRONTMATTER = "no_frontmatter"
#: A doc whose `type` is outside the vetted vocabulary (kept as a proposal).
TYPE_PROPOSED = "type_proposed"
#: A doc whose `type` was explicitly REJECTED when vetted.
TYPE_REJECTED = "type_rejected"
REASON_CODES = frozenset({
    REF_MISSING, REF_INVALIDATED, REF_DEPRECATED, REF_REDIRECTED,
    DOC_NO_FRONTMATTER, TYPE_PROPOSED, TYPE_REJECTED,
})


def body_refs(body: str) -> List[str]:
    """All node ids wikilinked in a doc body, in order (dedup preserved)."""
    return list(dict.fromkeys(WIKILINK_RE.findall(body or "")))


def body_tags(body: str) -> List[str]:
    """All inline `#tag`s of a doc body, lowercased, in order (dedup). A
    purely numeric `#10` (an issue / PR number in prose) is not a tag."""
    return list(dict.fromkeys(
        t.lower() for t in INLINE_TAG_RE.findall(body or "")
        if any(ch.isalpha() for ch in t)))


def rewrite_body_refs(body: str, mapping: Dict[str, str]) -> str:
    """Rewrite `[[old]]` -> `[[new]]` per `mapping` (unmapped ids untouched)."""
    return WIKILINK_RE.sub(
        lambda m: f"[[{mapping.get(m.group(1), m.group(1))}]]", body or "")


def rewrite_body_refs_traced(body: str, mapping: Dict[str, str]
                             ) -> tuple:
    """Like `rewrite_body_refs`, but also returns the TRACE needed to undo it
    precisely : {old: [ordinals]} — for each rewritten id, which occurrences
    (0-based, counted among ALL `[[new]]` links of the result) came from
    `[[old]]`. Occurrences of `[[new]]` the author wrote stay unmarked."""
    ordinal: Dict[str, int] = {}
    trace: Dict[str, List[int]] = {}

    def _sub(m):
        old = m.group(1)
        new = mapping.get(old, old)
        k = ordinal.get(new, 0)
        ordinal[new] = k + 1
        if new != old:
            trace.setdefault(old, []).append(k)
        return f"[[{new}]]"

    return WIKILINK_RE.sub(_sub, body or ""), trace


def revert_body_refs(body: str, new: str, old: str,
                     ordinals: Sequence[int]) -> str:
    """Undo one traced rewrite : turn the given occurrences (ordinals among all
    `[[new]]` links) back into `[[old]]`, leaving the others untouched."""
    wanted = set(int(i) for i in ordinals)
    seen = {"k": -1}

    def _sub(m):
        if m.group(1) != new:
            return m.group(0)
        seen["k"] += 1
        return f"[[{old}]]" if seen["k"] in wanted else m.group(0)

    return WIKILINK_RE.sub(_sub, body or "")


def context_tags(tags: Sequence[str]) -> List[str]:
    """The knowledge tags of a node — structural plumbing dropped, order kept."""
    return [t for t in (tags or []) if t not in _STRUCTURAL_TAGS]


def render_okf(*, type: str, title: str, tags: Sequence[str],
               refs: Sequence[str], body: str,
               timestamp: Optional[float] = None,
               extra: Optional[Dict] = None) -> str:
    """Serialize one OKF concept : YAML frontmatter (type required ; title, tags,
    refs, then any `extra` first-order fields e.g. feedback credibility, then
    timestamp) + markdown body. `refs` is the frontmatter mirror of the body's
    `[[…]]` links — a doc may carry several."""
    fm: Dict = {"type": type, "title": title}
    if tags:
        fm["tags"] = list(tags)
    if refs:
        fm["refs"] = list(refs)                 # multiple node refs per doc
    for k, v in (extra or {}).items():
        fm[k] = v
    if timestamp is not None:
        fm["timestamp"] = timestamp
    front = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{front}\n---\n\n{(body or '').strip()}\n"


def parse_okf(text: str) -> Dict:
    """Parse an OKF doc back into {type, title, tags, refs, timestamp, body}."""
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text or "", re.DOTALL)
    if not m:
        return {"type": "note", "title": "", "tags": [], "refs": [],
                "timestamp": None, "body": (text or "").strip(),
                "has_frontmatter": False}
    fm = yaml.safe_load(m.group(1)) or {}
    return {
        "type": fm.get("type", "note"),
        "title": fm.get("title", ""),
        "tags": list(fm.get("tags", []) or []),
        "refs": list(fm.get("refs", []) or []),
        "timestamp": fm.get("timestamp"),
        "body": (m.group(2) or "").strip(),
        "has_frontmatter": True,
    }


def default_body(items: Sequence[tuple], tags: Sequence[str]) -> str:
    """Deterministic (LLM-free) body from (node_id, content) pairs : one bullet
    per node with its inline `[[ref]]`, plus an inline `#tag` line — so tags and
    refs live in the prose too, not only the frontmatter."""
    lines = [f"- {content.strip()} [[{nid}]]" for nid, content in items]
    tag_line = " ".join(f"#{t}" for t in context_tags(tags))
    if tag_line:
        lines.append("")
        lines.append(f"Tags: {tag_line}")
    return "\n".join(lines)
