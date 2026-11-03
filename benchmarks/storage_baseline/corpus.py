"""The FROZEN text corpus the oracle contexts are built from (OO0-5, TAC2-1178).

No LoCoMo data ships with the repository (`benchmarks/locomo/data/` holds a
`.gitkeep`, the dataset is downloaded), and the oracle must build offline.
So the contexts are fed from the repository's OWN prose — README, AGENTS.md
and docs — cut into paragraph chunks, ONCE, and frozen in
`fixtures/corpus.json` with the sha256 and commit of every source file.
Later edits of those docs never move the oracle: `build.py` reads the frozen
file only. `python -m benchmarks.storage_baseline.corpus --freeze` rewrites
it (a new corpus = a new oracle version, never an in-place edit).

Chunking: blank-line paragraphs, one chunk per list item, fenced code and
table rows dropped, Markdown emphasis kept as text, chunks under 40 chars
dropped, chunks over 700 chars split on sentence ends. Each chunk carries
the heading it sits under (a tag `section:<slug>` at build time).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CORPUS_PATH = os.path.join(HERE, "fixtures", "corpus.json")

#: Roles whose files are kept WHOLE (a note is one document, not chunks).
WHOLE = {"narrow_notes"}

#: Which repository files feed which role. Disjoint on purpose: a context
#: never holds another context's text, and the held-out file is never
#: ingested anywhere (it feeds the off-memory queries and the `remember`
#: latency probe).
SOURCES: Dict[str, List[str]] = {
    "small": ["docs/john_walkthrough.md"],
    "medium": ["README.md", "benchmarks/README.md",
               "benchmarks/locomo/README.md"],
    "narrow_context": ["metacog/AGENTS.md", "metacog/README.md"],
    "narrow_notes": ["docs/ingest_index_plan.md",
                     "docs/LITERATURE_REVIEW_2026.md",
                     "benchmarks/d3_wiki/PROTOCOL.md"],
    "narrow_account": ["tests/AGENTS.md"],
    "held_out": ["docs/research_event_schema_rag.md"],
}

MIN_CHARS = 40
MAX_CHARS = 700
_SENT_END = re.compile(r"(?<=[.!?;:])\s+")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "top"


def _split_long(text: str) -> List[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    out: List[str] = []
    cur = ""
    for sent in _SENT_END.split(text):
        if cur and len(cur) + 1 + len(sent) > MAX_CHARS:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    # A single sentence longer than MAX_CHARS stays whole: never cut a word.
    return out


def chunk_markdown(text: str) -> List[Dict[str, str]]:
    """Paragraph / list-item chunks of a Markdown text, with their heading."""
    chunks: List[Dict[str, str]] = []
    section = "top"
    para: List[str] = []
    in_fence = False

    def flush() -> None:
        if not para:
            return
        body = " ".join(re.sub(r"^\s*>\s?", "", line).strip()
                        for line in para).strip()
        para.clear()
        for piece in _split_long(body):
            if len(piece) >= MIN_CHARS:
                chunks.append({"section": section, "text": piece})

    for line in text.splitlines():
        if line.strip().startswith("```"):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING.match(line)
        if m:
            flush()
            section = _slug(m.group(2))
            continue
        if not line.strip() or line.lstrip().startswith("|"):
            flush()
            continue
        if _LIST_ITEM.match(line):
            flush()
            para.append(_LIST_ITEM.sub("", line, count=1))
            continue
        para.append(line)
    flush()
    return chunks


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def freeze() -> Dict[str, object]:
    """Read the source files NOW and write `fixtures/corpus.json`."""
    roles: Dict[str, List[Dict[str, object]]] = {}
    for role, paths in SOURCES.items():
        files = []
        for rel in paths:
            with open(os.path.join(REPO, rel), "rb") as f:
                raw = f.read()
            entry = {"path": rel, "sha256": hashlib.sha256(raw).hexdigest()}
            if role in WHOLE:
                entry["body"] = raw.decode("utf-8")
            else:
                entry["chunks"] = chunk_markdown(raw.decode("utf-8"))
            files.append(entry)
        roles[role] = files
    corpus = {"version": 1, "commit": _git_commit(), "roles": roles}
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    return corpus


def load() -> Dict[str, object]:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="freeze or summarise the corpus")
    ap.add_argument("--freeze", action="store_true",
                    help="re-read the source files and rewrite fixtures/corpus.json")
    a = ap.parse_args()
    corpus = freeze() if a.freeze else load()
    for role, files in corpus["roles"].items():
        n = sum(len(f.get("chunks", [None])) for f in files)
        print(f"{role:15s} {n:4d} {'docs  ' if role in WHOLE else 'chunks'}  "
              + ", ".join(f"{f['path']}" for f in files))
    print(f"commit {corpus['commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
