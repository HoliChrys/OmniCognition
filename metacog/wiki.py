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
from typing import Dict, List, Optional, Sequence, Tuple

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
#: A cited node's CONTENT changed since the doc last saw it (drift).
REF_CONTENT_CHANGED = "content_changed"
#: A cited node's knowledge TAGS changed since the doc last saw it (drift).
REF_TAGS_CHANGED = "tags_changed"
#: Both.
REF_CHANGED = "content_and_tags_changed"
REASON_CODES = frozenset({
    REF_MISSING, REF_INVALIDATED, REF_DEPRECATED, REF_REDIRECTED,
    DOC_NO_FRONTMATTER, TYPE_PROPOSED, TYPE_REJECTED,
    REF_CONTENT_CHANGED, REF_TAGS_CHANGED, REF_CHANGED,
})

#: Body provenance : a GENERATED body is the deterministic rendering of the
#: doc's refs (`default_body`) and may be regenerated when they drift ; an
#: AUTHORED body is prose someone wrote and is only ever FLAGGED.
BODY_GENERATED = "generated"
BODY_AUTHORED = "authored"


def node_fingerprint(content: str, tags: Sequence[str]) -> str:
    """What a doc "saw" of a node when it linked it : two short hashes,
    `<content>:<knowledge tags>`, so drift can say WHICH part changed."""
    import hashlib
    c = hashlib.sha1((content or "").strip().encode("utf-8")).hexdigest()[:12]
    t = hashlib.sha1(",".join(sorted(context_tags(tags))).encode("utf-8")).hexdigest()[:12]
    return f"{c}:{t}"


def fingerprint_drift(old: Optional[str], new: str) -> Optional[str]:
    """The drift reason code between two fingerprints (None = unchanged or
    unknown baseline)."""
    if not old or old == new:
        return None
    oc, _, ot = old.partition(":")
    nc, _, nt = new.partition(":")
    if oc != nc and ot != nt:
        return REF_CHANGED
    return REF_CONTENT_CHANGED if oc != nc else REF_TAGS_CHANGED


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


def default_body(items: Sequence[tuple], tags: Sequence[str],
                 with_tags: bool = True) -> str:
    """Deterministic (LLM-free) body from (node_id, content) pairs : one bullet
    per node with its inline `[[ref]]`, plus an inline `#tag` line — so tags and
    refs live in the prose too, not only the frontmatter. `with_tags=False`
    (a portion) omits the tag line."""
    lines = [f"- {content.strip()} [[{nid}]]" for nid, content in items]
    tag_line = " ".join(f"#{t}" for t in context_tags(tags)) if with_tags else ""
    if tag_line:
        lines.append("")
        lines.append(f"Tags: {tag_line}")
    return "\n".join(lines)


# =============================================================================
# OBJECTS in the body — portions, variables, annotations
#
# Parts of a doc are not runs of characters but OBJECTS with an identity :
#   <portion id="p1" seeds="q1,q2" mode="generated"> ... </portion>
#       a block, optionally fed by seed queries, regenerable on its own
#   <var name="deploy_target" node="N42" field="content"/>
#       an inline BINDING to a node : rendered from the node, never copied
# Edits are operations on the object (set params / remove / replace), each
# journaled and reversible, so a ref keeps its identity and its history.
# Annotations (notes typed `note` / `purpose` / `keep` / `todo`) hang on a
# portion id, a var name, a node ref or the whole doc (`*`) ; they render in
# the frontmatter like a bibliography and as footnotes in the resolved view.
# =============================================================================

PORTION_RE = re.compile(r"<portion\b([^>]*)>(.*?)</portion>", re.DOTALL | re.IGNORECASE)
VAR_RE = re.compile(r"<var\b([^>]*?)/>", re.IGNORECASE)
_ATTR_RE = re.compile(r'([A-Za-z_][\w-]*)\s*=\s*"([^"]*)"')

ANNOTATION_KINDS = ("note", "purpose", "keep", "todo")
#: The whole-doc target for seeds / annotations.
DOC_TARGET = "*"


def _attrs(s: str) -> Dict[str, str]:
    return {k.lower(): v for k, v in _ATTR_RE.findall(s or "")}


def parse_portions(body: str) -> List[Dict]:
    """All `<portion>` blocks : {id, seeds, refs, mode, body(inner), start,
    end}. `refs` are the portion's EXPLICIT node refs (attribute) — for a
    generated portion the inline `[[…]]` of its body are a rendering, not a
    source, so the sources live in the tag."""
    out = []
    for m in PORTION_RE.finditer(body or ""):
        a = _attrs(m.group(1))
        seeds = [s.strip() for s in a.get("seeds", "").split(",") if s.strip()]
        refs = [s.strip() for s in a.get("refs", "").split(",") if s.strip()]
        out.append({"id": a.get("id", ""), "seeds": seeds, "refs": refs,
                    "mode": (a.get("mode") or None), "body": m.group(2).strip(),
                    "start": m.start(), "end": m.end()})
    return out


def parse_vars(body: str) -> List[Dict]:
    """All `<var/>` bindings : {name, node, field}."""
    out = []
    for m in VAR_RE.finditer(body or ""):
        a = _attrs(m.group(1))
        if a.get("name"):
            out.append({"name": a["name"], "node": a.get("node"),
                        "field": a.get("field") or "content"})
    return out


def body_bindings(body: str) -> List[str]:
    """Node ids bound by `<var/>` tags (they are refs too), in order, dedup."""
    return list(dict.fromkeys(v["node"] for v in parse_vars(body) if v.get("node")))


def portion_tag(pid: str, inner: str, seeds: Sequence[str] = (),
                mode: Optional[str] = None, refs: Sequence[str] = ()) -> str:
    attrs = f'id="{pid}"'
    if seeds:
        attrs += f' seeds="{",".join(seeds)}"'
    if refs:
        attrs += f' refs="{",".join(refs)}"'
    if mode:
        attrs += f' mode="{mode}"'
    return f"<portion {attrs}>\n{(inner or '').strip()}\n</portion>"


def var_tag(name: str, node: str, field: str = "content") -> str:
    return f'<var name="{name}" node="{node}" field="{field}"/>'


def set_portion_body(body: str, pid: str, inner: str,
                     seeds: Optional[Sequence[str]] = None,
                     mode: Optional[str] = None,
                     refs: Optional[Sequence[str]] = None) -> Tuple[str, bool]:
    """Replace a portion's inner text (and optionally its attrs ; None keeps)
    in place ; append a new portion when `pid` is absent. Returns (body,
    existed)."""
    for p in parse_portions(body):
        if p["id"] == pid:
            tag = portion_tag(pid, inner,
                              p["seeds"] if seeds is None else seeds,
                              p["mode"] if mode is None else mode,
                              p["refs"] if refs is None else refs)
            return body[:p["start"]] + tag + body[p["end"]:], True
    sep = "\n\n" if (body or "").strip() else ""
    return (body or "").rstrip() + sep + portion_tag(pid, inner, seeds or (), mode,
                                                     refs or ()), False


def remove_portion_tag(body: str, pid: str) -> Tuple[str, Optional[Dict]]:
    """Drop a portion block entirely. Returns (body, the removed portion)."""
    for p in parse_portions(body):
        if p["id"] == pid:
            return (body[:p["start"]].rstrip() + "\n" + body[p["end"]:].lstrip()).strip(), p
    return body, None


def set_var_tag(body: str, name: str, node: str,
                field: str = "content") -> Tuple[str, Optional[Dict]]:
    """Rebind a variable in place (all its occurrences) ; append the tag when
    the name is new. Returns (body, previous binding or None)."""
    prev = next((v for v in parse_vars(body) if v["name"] == name), None)
    new = var_tag(name, node, field)
    if prev is None:
        sep = "\n" if (body or "").strip() else ""
        return (body or "").rstrip() + sep + new, None
    def _sub(m):
        return new if _attrs(m.group(1)).get("name") == name else m.group(0)
    return VAR_RE.sub(_sub, body or ""), prev


def remove_var_tag(body: str, name: str) -> Tuple[str, Optional[Dict]]:
    prev = next((v for v in parse_vars(body) if v["name"] == name), None)
    if prev is None:
        return body, None
    out = VAR_RE.sub(lambda m: "" if _attrs(m.group(1)).get("name") == name
                     else m.group(0), body or "")
    return re.sub(r"[ \t]+\n", "\n", out).strip(), prev


def portion_of(body: str, needle: str) -> Optional[str]:
    """The id of the portion whose inner text contains `needle` (a `[[ref]]`
    or a `<var/>` tag), or None when it sits outside every portion."""
    for p in parse_portions(body):
        if needle in body[p["start"]:p["end"]]:
            return p["id"]
    return None


def resolve_body(body: str, values: Dict[str, str],
                 annotations: Sequence[Dict] = ()) -> str:
    """The READ view : `<var/>` tags replaced by their live value (an unbound
    one says so), portion tags stripped to their inner text, and the
    annotations appended as footnotes (`[^target]` style bibliography)."""
    def _var(m):
        a = _attrs(m.group(1))
        name = a.get("name", "?")
        return values.get(name, f"⟨{name}: unbound⟩")
    out = VAR_RE.sub(_var, body or "")
    out = PORTION_RE.sub(lambda m: m.group(2).strip(), out)
    notes = [a for a in annotations if a.get("note")]
    if notes:
        out = out.rstrip() + "\n\nAnnotations:\n" + "\n".join(
            f"[^{a['target']}] ({a['kind']}) {a['note']}" for a in notes)
    return out.strip()
