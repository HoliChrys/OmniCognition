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

#: Structural tags that are plumbing, not knowledge — kept off the wiki surface.
_STRUCTURAL_TAGS = {
    "fact", "thought", "action", "event", "refined", "invalidated",
    "tool", "eager", "deprecated", "generated", "atomic", "entity",
}


def body_refs(body: str) -> List[str]:
    """All node ids wikilinked in a doc body, in order (dedup preserved)."""
    return list(dict.fromkeys(WIKILINK_RE.findall(body or "")))


def rewrite_body_refs(body: str, mapping: Dict[str, str]) -> str:
    """Rewrite `[[old]]` -> `[[new]]` per `mapping` (unmapped ids untouched)."""
    return WIKILINK_RE.sub(
        lambda m: f"[[{mapping.get(m.group(1), m.group(1))}]]", body or "")


def context_tags(tags: Sequence[str]) -> List[str]:
    """The knowledge tags of a node — structural plumbing dropped, order kept."""
    return [t for t in (tags or []) if t not in _STRUCTURAL_TAGS]


def render_okf(*, type: str, title: str, tags: Sequence[str],
               refs: Sequence[str], body: str,
               timestamp: Optional[float] = None) -> str:
    """Serialize one OKF concept : YAML frontmatter (type required ; title, tags,
    refs, timestamp) + markdown body. `refs` is the frontmatter mirror of the
    body's `[[…]]` links — a doc may carry several."""
    fm: Dict = {"type": type, "title": title}
    if tags:
        fm["tags"] = list(tags)
    if refs:
        fm["refs"] = list(refs)                 # multiple node refs per doc
    if timestamp is not None:
        fm["timestamp"] = timestamp
    front = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{front}\n---\n\n{(body or '').strip()}\n"


def parse_okf(text: str) -> Dict:
    """Parse an OKF doc back into {type, title, tags, refs, timestamp, body}."""
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text or "", re.DOTALL)
    if not m:
        return {"type": "note", "title": "", "tags": [], "refs": [],
                "timestamp": None, "body": (text or "").strip()}
    fm = yaml.safe_load(m.group(1)) or {}
    return {
        "type": fm.get("type", "note"),
        "title": fm.get("title", ""),
        "tags": list(fm.get("tags", []) or []),
        "refs": list(fm.get("refs", []) or []),
        "timestamp": fm.get("timestamp"),
        "body": (m.group(2) or "").strip(),
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
