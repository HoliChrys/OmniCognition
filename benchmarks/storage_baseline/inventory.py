"""READ-ONLY inventory of the omni stores under a storage root (OO0-4, TAC2-1177).

For every `memory.pkl` found under `--root` (the context stores
`<root>/<ctx>/memory.pkl` AND the narrow-account stores
`<root>/<ctx>/accounts/<user>/memory.pkl`) and its `memory.pkl.journal.db`:

  * points per kind and per epistemic state, pickle size, encoder id and
    vector dimension;
  * distinct tags (and tag assignments), observators, conversation turns,
    forget-log entries;
  * `update_log`: number of audit entries and their pickled size (the field
    that grows without bound, DESIGN_OMNI_ONTOLOGIE §11);
  * rows per journal table, wiki docs;
  * the raw Neo4j sizing inputs: float32 bytes of the 384-d vectors a point
    carries today, and the node count a migration would add.

NOTHING IS WRITTEN under the root. The pickle is read with `pickle.load` on
a read-only file descriptor — never through `Memory(...)`, whose
construction opens the journal read-write (`CREATE TABLE IF NOT EXISTS`) and
whose `save()` rewrites the store. The journal is opened with SQLite's
`mode=ro` URI. Every file read is fingerprinted (size, mtime_ns, sha256)
before and after; a change is reported per file (a live writer, never this
script). `--out` must lie outside the root.

Standalone on purpose: it imports nothing from `benchmarks`, only `metacog`
(the classes the pickle needs), so it can be copied alone next to a served
checkout and run with that checkout's venv, without touching the checkout:

    ~/repos/omni/.venv/bin/python /tmp/omni_inventory.py \
        --root ~/data_tachikoma/omni/contexts --out /tmp/omni_inventory.json

Prints a Markdown table (one row per store + the total) and writes the full
JSON to `--out`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pickle
import sqlite3
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

STORE_NAME = "memory.pkl"
#: Vector fields a Point carries today (`metacog/epistemic.py`): the three of
#: the effective-vector geometry plus the keyword embedding.
VECTOR_FIELDS = ("embedding_orig", "delta_active", "delta_latent",
                 "keywords_embedding")
#: Inventory format version: bump when a field changes meaning.
FORMAT = 1


def _fingerprint(path: str) -> Optional[Dict[str, Any]]:
    """(size, mtime_ns, sha256) of a file, or None when it does not exist."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "sha256": h.hexdigest()}


def _dir_listing(path: str) -> List[str]:
    try:
        return sorted(os.listdir(path))
    except FileNotFoundError:
        return []


def find_stores(root: str) -> List[str]:
    """Every `memory.pkl` under `root`, sorted. Symlinked dirs are not
    followed (a served root links `notes/` into FUSE)."""
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        if STORE_NAME in filenames:
            found.append(os.path.join(dirpath, STORE_NAME))
    return found


def _enum_name(v: Any) -> str:
    return getattr(v, "name", None) or str(v)


def inspect_pickle(path: str) -> Dict[str, Any]:
    """Unpickle the store READ-ONLY and count what it holds."""
    with open(path, "rb") as f:            # O_RDONLY: nothing can be written
        snap = pickle.load(f)
    if not isinstance(snap, dict):
        return {"error": f"not a dict snapshot ({type(snap).__name__})"}
    points = snap.get("points", []) or []
    kinds: Counter = Counter()
    states: Counter = Counter()
    tags: Counter = Counter()
    vec_dims: Counter = Counter()
    vectors_present: Counter = Counter()
    upd_entries = 0
    upd_bytes = 0
    upd_max = 0
    exec_logs = 0
    content_chars = 0
    for p in points:
        kinds[_enum_name(getattr(p, "kind", "?"))] += 1
        states[_enum_name(getattr(p, "state", "?"))] += 1
        for t in getattr(p, "tags", []) or []:
            tags[str(t)] += 1
        for fld in VECTOR_FIELDS:
            v = getattr(p, fld, None)
            if v:
                vectors_present[fld] += 1
                if fld == "embedding_orig":
                    vec_dims[len(v)] += 1
        ul = getattr(p, "update_log", []) or []
        upd_entries += len(ul)
        upd_max = max(upd_max, len(ul))
        if ul:
            upd_bytes += len(pickle.dumps(ul, protocol=pickle.HIGHEST_PROTOCOL))
        exec_logs += len(getattr(p, "execution_log", []) or [])
        content_chars += len(getattr(p, "content", "") or "")
    observators = snap.get("observators", {}) or {}
    conv = snap.get("conversation_log")
    turns = getattr(conv, "turns", []) if conv is not None else []
    dim = vec_dims.most_common(1)[0][0] if vec_dims else 0
    n_vectors = sum(vectors_present.values())
    return {
        "points": len(points),
        "points_by_kind": dict(sorted(kinds.items())),
        "points_by_state": dict(sorted(states.items())),
        "encoder_id": snap.get("encoder_id"),
        "embedding_dim": dim,
        "embedding_dims_seen": {str(k): v for k, v in sorted(vec_dims.items())},
        "distinct_tags": len(tags),
        "tag_assignments": sum(tags.values()),
        "account_tags": sorted(t for t in tags if t.startswith("account:")),
        "observators": len(observators),
        "observators_auto_comm": sum(1 for k in observators
                                     if str(k).startswith("auto-comm-")),
        "conversation_turns": len(turns),
        "forget_log": len(snap.get("_forget_log", []) or []),
        "t_clock": snap.get("_t_clock"),
        "update_log_entries": upd_entries,
        "update_log_max_per_point": upd_max,
        "update_log_pickled_bytes": upd_bytes,
        "execution_log_entries": exec_logs,
        "content_chars": content_chars,
        "vectors_present": dict(sorted(vectors_present.items())),
        # Neo4j sizing inputs (float32, as a Neo4j vector property stores
        # them): every vector the point holds today, plus the node's own
        # `embedding` (DESIGN §4.3: `embedding_orig` AND `embedding`).
        "vector_bytes_float32": n_vectors * dim * 4,
        "vector_bytes_float32_with_node_embedding": (n_vectors + len(points)) * dim * 4,
    }


def inspect_journal(path: str) -> Dict[str, Any]:
    """Rows per table of the journal, opened read-only (`mode=ro`)."""
    uri = "file:" + os.path.abspath(path) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        rows = {n: conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]
                for n in names}
        distinct_tags = None
        if "tags" in rows:
            distinct_tags = conn.execute(
                "SELECT COUNT(DISTINCT tag) FROM tags").fetchone()[0]
    finally:
        conn.close()
    return {"tables": len(rows), "rows": rows,
            "rows_total": sum(rows.values()),
            "wiki_docs": rows.get("wiki_docs", 0),
            "distinct_tags": distinct_tags}


def inventory(root: str) -> Dict[str, Any]:
    root = os.path.realpath(os.path.expanduser(root))
    stores = find_stores(root)
    out: List[Dict[str, Any]] = []
    for pkl in stores:
        d = os.path.dirname(pkl)
        jdb = pkl + ".journal.db"
        listing_before = _dir_listing(d)
        fp_pkl = _fingerprint(pkl)
        fp_jdb = _fingerprint(jdb)
        entry: Dict[str, Any] = {
            "store": os.path.relpath(d, root),
            "path": pkl,
            "pkl_bytes": fp_pkl["size"] if fp_pkl else None,
            "journal_path": jdb if fp_jdb else None,
            "journal_bytes": fp_jdb["size"] if fp_jdb else None,
        }
        try:
            entry.update(inspect_pickle(pkl))
        except Exception as e:              # report, never stop the inventory
            entry["error"] = f"{type(e).__name__}: {e}"
        if fp_jdb:
            try:
                entry["journal"] = inspect_journal(jdb)
            except Exception as e:
                entry["journal"] = {"error": f"{type(e).__name__}: {e}"}
        after_pkl = _fingerprint(pkl)
        after_jdb = _fingerprint(jdb)
        entry["unchanged"] = {
            "pkl": after_pkl == fp_pkl,
            "journal": after_jdb == fp_jdb,
            "dir_listing": _dir_listing(d) == listing_before,
        }
        entry["sha256"] = {"pkl": fp_pkl and fp_pkl["sha256"],
                           "journal": fp_jdb and fp_jdb["sha256"]}
        out.append(entry)
    # Journals without a pickle (a context that was recalled but never
    # written: the journal exists, the store does not).
    orphan_journals = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if (STORE_NAME + ".journal.db") in filenames and STORE_NAME not in filenames:
            jdb = os.path.join(dirpath, STORE_NAME + ".journal.db")
            fp = _fingerprint(jdb)
            item: Dict[str, Any] = {"store": os.path.relpath(dirpath, root),
                                    "journal_path": jdb,
                                    "journal_bytes": fp["size"] if fp else None}
            try:
                item["journal"] = inspect_journal(jdb)
            except Exception as e:
                item["journal"] = {"error": f"{type(e).__name__}: {e}"}
            item["unchanged"] = {"journal": _fingerprint(jdb) == fp}
            orphan_journals.append(item)
    total = _total(out)
    return {"format": FORMAT,
            "taken_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "host": os.uname().nodename, "root": root,
            "stores": out, "journals_without_store": orphan_journals,
            "total": total}


def _total(stores: List[Dict[str, Any]]) -> Dict[str, Any]:
    kinds: Counter = Counter()
    rows: Counter = Counter()
    t: Dict[str, Any] = {"stores": len(stores)}
    for key in ("points", "pkl_bytes", "journal_bytes", "observators",
                "update_log_entries", "update_log_pickled_bytes",
                "vector_bytes_float32", "vector_bytes_float32_with_node_embedding",
                "conversation_turns", "forget_log", "tag_assignments"):
        t[key] = sum(int(s.get(key) or 0) for s in stores)
    for s in stores:
        kinds.update(s.get("points_by_kind", {}))
        rows.update((s.get("journal") or {}).get("rows", {}))
    t["points_by_kind"] = dict(sorted(kinds.items()))
    t["journal_rows"] = dict(sorted(rows.items()))
    t["wiki_docs"] = rows.get("wiki_docs", 0)
    t["all_unchanged"] = all(all(s.get("unchanged", {}).values()) for s in stores)
    return t


def _mb(n: Optional[int]) -> str:
    return "—" if n is None else f"{n / 1e6:.2f}"


def markdown(inv: Dict[str, Any]) -> str:
    """The table pasted in Linear: one row per store, then the total."""
    lines = [f"Inventaire omni — {inv['taken_at']} — hôte `{inv['host']}` — "
             f"racine `{inv['root']}`", "",
             "| store | points | FACT | THOUGHT | ACTION | EVENT | INVALID | "
             "pkl (Mo) | journal (Mo) | lignes journal | tags distincts | "
             "observateurs | docs wiki | update_log (entrées / Mo) | "
             "vecteurs f32 (Mo) | inchangé |",
             "| -- | " + " | ".join(["--"] * 15) + " |"]

    def row(s: Dict[str, Any], name: str) -> str:
        k = s.get("points_by_kind", {})
        st = s.get("points_by_state", {})
        j = s.get("journal") or {}
        same = "oui" if all((s.get("unchanged") or {}).values()) else "**NON**"
        return (f"| `{name}` | {s.get('points', '—')} | {k.get('FACT', 0)} | "
                f"{k.get('THOUGHT', 0)} | {k.get('ACTION', 0)} | {k.get('EVENT', 0)} | "
                f"{st.get('INVALID', 0)} | {_mb(s.get('pkl_bytes'))} | "
                f"{_mb(s.get('journal_bytes'))} | {j.get('rows_total', '—')} | "
                f"{s.get('distinct_tags', '—')} | {s.get('observators', '—')} | "
                f"{j.get('wiki_docs', '—')} | {s.get('update_log_entries', '—')} / "
                f"{_mb(s.get('update_log_pickled_bytes'))} | "
                f"{_mb(s.get('vector_bytes_float32'))} | {same} |")

    for s in inv["stores"]:
        lines.append(row(s, s["store"]) if "error" not in s
                     else f"| `{s['store']}` | ERREUR : {s['error']} |")
    t = inv["total"]
    lines.append(
        f"| **total ({t['stores']} stores)** | {t['points']} | "
        f"{t['points_by_kind'].get('FACT', 0)} | {t['points_by_kind'].get('THOUGHT', 0)} | "
        f"{t['points_by_kind'].get('ACTION', 0)} | {t['points_by_kind'].get('EVENT', 0)} | — | "
        f"{_mb(t['pkl_bytes'])} | {_mb(t['journal_bytes'])} | "
        f"{sum(t['journal_rows'].values())} | — | {t['observators']} | {t['wiki_docs']} | "
        f"{t['update_log_entries']} / {_mb(t['update_log_pickled_bytes'])} | "
        f"{_mb(t['vector_bytes_float32'])} | "
        f"{'oui' if t['all_unchanged'] else '**NON**'} |")
    if inv["journals_without_store"]:
        lines += ["", "Journaux sans `memory.pkl` : " + ", ".join(
            f"`{j['store']}` ({(j.get('journal') or {}).get('rows_total', '?')} lignes)"
            for j in inv["journals_without_store"])]
    if t["journal_rows"]:
        lines += ["", "Lignes par table (total) : " + ", ".join(
            f"`{k}` {v}" for k, v in t["journal_rows"].items())]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.environ.get("OMNI_STORAGE_ROOT"),
                    help="storage root (default $OMNI_STORAGE_ROOT)")
    ap.add_argument("--out", help="JSON output path (must be OUTSIDE the root)")
    a = ap.parse_args(argv)
    if not a.root:
        ap.error("--root or $OMNI_STORAGE_ROOT is required")
    root = os.path.realpath(os.path.expanduser(a.root))
    if a.out:
        out = os.path.realpath(os.path.expanduser(a.out))
        if out == root or out.startswith(root + os.sep):
            ap.error(f"--out {out} is under the storage root: refused (read-only)")
    inv = inventory(root)
    print(markdown(inv))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(inv, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"\n[inventory] JSON -> {a.out}", file=sys.stderr)
    return 0 if inv["total"]["all_unchanged"] else 3


if __name__ == "__main__":
    sys.exit(main())
