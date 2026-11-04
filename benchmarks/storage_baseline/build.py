"""Build the three oracle contexts, reproducibly (OO0-5, TAC2-1178).

    python -m benchmarks.storage_baseline.build --work <dir>

writes `<dir>/stores/` (an OMNI_STORAGE_ROOT), `<dir>/notes/oracle/` (an
OMNI_NOTES_ROOT whose tree root is `oracle`) and `<dir>/build_manifest.json`.
`<dir>` must not exist (or be empty): a build never writes over anything.

Every write goes through the PRODUCTION tools of `build_app` under the
gate's `ContextualMemory` (see `harness.Driver`), with the production
encoder (fastembed MiniLM-L12, 384) and reranker, no LLM, and seeded point
ids — so two builds give the same points, contents and vectors (checked by
the manifest's fingerprints, and by `compare.py` on two oracle runs).

| context | what it holds |
| -- | -- |
| `oracle.small` | `docs/john_walkthrough.md` as `remember` facts, 4 THOUGHTs, a declared observator, observations (+/-), 1 forget, 4 conversation turns |
| `oracle.medium` | `README.md` + the benchmark READMEs (~250 facts; the LoCoMo README chained by `sequence_prev`), 20 THOUGHT/ACTION, 3 EVENT hubs, two declared observators, observations, 5 forgets (one superseded) |
| `oracle.narrow` | `metacog/AGENTS.md` + `metacog/README.md` as the manager's facts, 3 notes in its `notes/` folder (the deepwiki), and the narrow account `oracle-member` writing `tests/AGENTS.md` (mirrored into the context memory, one mirrored forget) |

Nothing here reads or writes a served store.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

from benchmarks.storage_baseline import corpus as _corpus
from benchmarks.storage_baseline import harness as H

SEED = 1146
FORMAT = 1


def _chunks(c: Dict[str, Any], role: str) -> List[Dict[str, str]]:
    out = []
    for f in c["roles"][role]:
        for ch in f["chunks"]:
            out.append({"file": f["path"], **ch})
    return out


def _tags(ch: Dict[str, str]) -> List[str]:
    return [f"file:{ch['file']}", f"section:{ch['section']}"]


def _first_sentence(text: str, n: int = 160) -> str:
    s = text.split(". ")[0]
    return s[:n].rstrip()


def _build_small(d: H.Driver, c: Dict[str, Any], rng: random.Random) -> None:
    ctx = "oracle.small"
    chunks = _chunks(c, "small")
    ids: List[str] = []
    with d.under(ctx):
        for ch in chunks:
            r = d.tool("remember", content=ch["text"], tags=_tags(ch),
                       source="extracted_fact")
            ids.append(r["id"])
        sections: Dict[str, List[int]] = {}
        for i, ch in enumerate(chunks):
            sections.setdefault(ch["section"], []).append(i)
        for sec in sorted(sections)[:4]:
            i = sections[sec][0]
            d.tool("ingest", kind="THOUGHT",
                   content=f"Reading of section {sec.replace('-', ' ')}: "
                           f"{_first_sentence(chunks[i]['text'])}",
                   tags=["generated", f"section:{sec}"])
        d.tool("declare_observator", id="reviewer", name="Doc reviewer",
               keywords=["presearch", "walk"])
        picks = rng.sample(ids, 6)
        d.tool("observe", target_ids=picks[:3], polarity=0.8)
        d.tool("observe", target_ids=picks[3:5], polarity=0.6,
               observator_id="reviewer")
        d.tool("observe", target_ids=[picks[5]], polarity=-0.7)
        d.tool("forget", node_id=ids[-1], reason="oracle: superseded wording")
        d.tool("process_turn", text="How does the presearch round work?",
               speaker="user")
        d.tool("process_turn", text="It reformulates the question first.",
               speaker="assistant", retrieved_point_ids=ids[:2])
        d.tool("process_turn", text="No, that is wrong, it searches literally.",
               speaker="user")
        d.tool("process_turn", text="Noted.", speaker="assistant")


def _build_medium(d: H.Driver, c: Dict[str, Any], rng: random.Random) -> None:
    ctx = "oracle.medium"
    chunks = _chunks(c, "medium")
    ids: List[str] = []
    with d.under(ctx):
        mem = d.mem()
        prev = None
        for ch in chunks:
            if ch["file"] == "benchmarks/locomo/README.md":
                # The episodic path: ordered chunks chained by sequence_prev
                # (Memory.ingest — the tool surface has no sequence argument).
                p = mem.ingest(ch["text"], kind="FACT", sequence_prev=prev)
                p.add_tag(*_tags(ch))
                if mem.journal is not None:
                    mem.journal.log_tags(p.id, p.tags)
                prev = p.id
                ids.append(p.id)
            else:
                r = d.tool("remember", content=ch["text"], tags=_tags(ch),
                           source="extracted_fact")
                ids.append(r["id"])
        mem.save()
        sections: Dict[str, List[int]] = {}
        for i, ch in enumerate(chunks):
            sections.setdefault(ch["section"], []).append(i)
        secs = sorted(sections)
        for j, sec in enumerate(secs[:20]):
            i = sections[sec][0]
            kind = "ACTION" if j % 4 == 3 else "THOUGHT"
            lead = "To apply" if kind == "ACTION" else "Takeaway of"
            d.tool("ingest", kind=kind,
                   content=f"{lead} {sec.replace('-', ' ')}: "
                           f"{_first_sentence(chunks[i]['text'])}",
                   tags=["generated", f"section:{sec}"])
        pts = {p.id: p for p in mem.points}
        for name, etype, lo, hi in (("LoCoMo evaluation", "benchmark", 0, 40),
                                    ("OBLIQ bench", "benchmark", 40, 80),
                                    ("release of the walk", "milestone", 80, 140)):
            facts = [pts[i] for i in ids[lo:hi] if i in pts]
            mem.ingest_event(name, etype, source_facts=facts)
        mem.save()
        d.tool("declare_observator", id="reviewer", name="Bench reviewer",
               keywords=["locomo", "benchmark", "recall"])
        d.tool("declare_observator", id="maintainer", name="Maintainer",
               keywords=["mcp", "server", "install"])
        picks = rng.sample(ids, 20)
        for a in range(0, 10, 2):
            d.tool("observe", target_ids=picks[a:a + 2], polarity=0.9)
        d.tool("observe", target_ids=picks[10:13], polarity=0.5,
               observator_id="reviewer")
        d.tool("observe", target_ids=picks[13:15], polarity=-0.8,
               observator_id="maintainer")
        for k in range(4):
            d.tool("forget", node_id=picks[15 + k], reason="oracle: stale")
        d.tool("forget", node_id=picks[19], reason="oracle: duplicate",
               superseded_by=picks[0])


def _build_narrow(d: H.Driver, c: Dict[str, Any], rng: random.Random,
                  notes_root: str) -> None:
    ctx = "oracle.narrow"
    from metacog.tachikoma_gate import notes_folder
    folder = notes_folder(notes_root, ctx)
    os.makedirs(folder, exist_ok=True)
    for f in c["roles"]["narrow_notes"]:
        name = os.path.splitext(os.path.basename(f["path"]))[0].lower()
        with open(os.path.join(folder, f"{name}.md"), "w", encoding="utf-8") as out:
            out.write(f["body"])
    ids: List[str] = []
    with d.under(ctx):                       # the manager: context memory
        for ch in _chunks(c, "narrow_context"):
            r = d.tool("remember", content=ch["text"], tags=_tags(ch),
                       source="extracted_fact")
            ids.append(r["id"])
        d.tool("observe", target_ids=rng.sample(ids, 3), polarity=0.7)
        d.tool("forget", node_id=ids[5], reason="oracle: manager correction")
    own: List[str] = []
    with d.under(ctx, H.NARROW_ACCOUNT):     # the narrow account
        for ch in _chunks(c, "narrow_account"):
            r = d.tool("remember", content=ch["text"], tags=_tags(ch),
                       source="extracted_fact")
            own.append(r["id"])
        d.tool("forget", node_id=own[-1], reason="oracle: member correction")


def build(work: str, factory_spec: str | None = None) -> Dict[str, Any]:
    work = os.path.abspath(work)
    if os.path.exists(work) and os.listdir(work):
        raise SystemExit(f"[build] {work} is not empty: refused (a build never "
                         "writes over anything)")
    stores = os.path.join(work, "stores")
    notes_root = os.path.join(work, "notes", H.TREE)
    os.makedirs(stores, exist_ok=True)
    os.makedirs(notes_root, exist_ok=True)
    c = _corpus.load()
    factory = H.resolve_factory(factory_spec)
    with H.seeded_uuid(SEED):
        d = H.Driver(stores, notes_root, factory)
        rng = random.Random(SEED)
        _build_small(d, c, rng)
        _build_medium(d, c, rng)
        _build_narrow(d, c, rng, notes_root)
    fps: Dict[str, Any] = {}
    for ctx, views in H.VIEWS.items():
        for vctx, account in views:
            with d.under(vctx, account):
                key = vctx + (f"/accounts/{account}" if account else "")
                fps[key] = H.store_fingerprint(d.mem())
    enc, _ = H.shared_models()
    from metacog.defaults import encoder_id
    manifest = {
        "format": FORMAT, "seed": SEED, "corpus_version": c["version"],
        "corpus_commit": c["commit"], "code_commit": H.git_commit(),
        "encoder_id": encoder_id(enc), "llm_calls": H.LLMGuard.calls,
        "stores": fps,
    }
    with open(os.path.join(work, "build_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    if H.LLMGuard.calls:
        raise SystemExit(f"[build] {H.LLMGuard.calls} LLM call(s): the oracle "
                         "must be LLM-free")
    return manifest


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build the three oracle contexts")
    ap.add_argument("--work", required=True, help="empty output directory")
    ap.add_argument("--memory-factory", default=None,
                    help="module:callable (storage_root, notes_root) -> memory")
    a = ap.parse_args(argv)
    m = build(a.work, a.memory_factory)
    for key, fp in m["stores"].items():
        print(f"{key:40s} {fp['points']:4d} points {fp['by_kind']} "
              f"content {fp['content_sha256'][:12]} vectors {fp['vectors_sha256'][:12]}")
    print(f"llm_calls={m['llm_calls']} encoder={m['encoder_id']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
