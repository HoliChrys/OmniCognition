"""
Append-only usage JOURNAL (SQLite) — the mnema access-log model.

Two tables mirror mnema's separation of the memory STORE from its usage
TELEMETRY :

  retrievals    one row per search : the query, the ids it returned, an
                optional usefulness score (`mark_useful`), the wall-clock ts.
  access_events one row per (returned node, retrieval) : which node was served
                by which search, at which rank, when.

Co-retrieval — "nodes historically surfaced TOGETHER" — is then a SQL
self-join on `access_events.retrieval_id`, computed ON DEMAND, never a stored
graph. This is the edgeless spreading-activation analogue : the topology lives
in the log of retrievals, not in maintained edges (contrast our manifold's
`apply_pull`). `mark_useful` writes the supervised 0/1/2 label that a later
decay-fit reads ; `access_timestamps` exposes the raw access history a
need-odds decay would consume.

Node ids are TEXT here — OmniCognition point ids are strings ("A", "gloss_A",
dia-ids) — unlike mnema's INTEGER ids. The journal is SEPARATE from the pickled
point manifold : usage telemetry is append-only and naturally relational, so it
lives in its own SQLite file and survives independently of Memory.save()/load().
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import List, Optional, Sequence, Tuple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS retrievals (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text         TEXT,
    returned_node_ids  TEXT,                       -- JSON array of string ids
    useful             INTEGER,                    -- NULL | 0 | 1 | 2
    ts                 REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS access_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id       TEXT    NOT NULL,
    retrieval_id  INTEGER NOT NULL REFERENCES retrievals(id),
    rank          INTEGER NOT NULL,
    ts            REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_access_retrieval ON access_events(retrieval_id);
CREATE INDEX IF NOT EXISTS idx_access_node      ON access_events(node_id);

CREATE TABLE IF NOT EXISTS hops (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id   TEXT NOT NULL,                          -- from node
    dst_id   TEXT NOT NULL,                          -- to node (accumulates spike)
    ts       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hops_src ON hops(src_id);
CREATE INDEX IF NOT EXISTS idx_hops_dst ON hops(dst_id);

CREATE TABLE IF NOT EXISTS collision_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    kind              TEXT NOT NULL,                  -- 'fission' | 'chasles' | 'lateral'
    child_id          TEXT,                           -- the keeper / shortcut / common child
    parent_ids        TEXT NOT NULL,                  -- JSON array (the collided / intermediates)
    anchor_ids        TEXT NOT NULL,                  -- JSON array (boundaries ; [] for proximity)
    trigger_distance  REAL,
    threshold         REAL,
    ts                REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_collision_kind  ON collision_events(kind);
CREATE INDEX IF NOT EXISTS idx_collision_child ON collision_events(child_id);

CREATE TABLE IF NOT EXISTS path_traversals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    signature     TEXT NOT NULL,      -- 'A>B>C>D' : direction-preserving, the group key
    start_id      TEXT NOT NULL,      -- Chasles anchor (from)
    end_id        TEXT NOT NULL,      -- Chasles anchor (to)
    intermediates TEXT NOT NULL,      -- JSON array [B, C] : what a shortcut absorbs
    length        INTEGER NOT NULL,   -- node count (start + intermediates + end)
    ts            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_paths_sig ON path_traversals(signature);
CREATE INDEX IF NOT EXISTS idx_paths_len ON path_traversals(length);

CREATE TABLE IF NOT EXISTS tags (
    node_id  TEXT NOT NULL,
    tag      TEXT NOT NULL,                          -- hierarchical: a:b:c
    depth    INTEGER NOT NULL,                        -- number of ':' (sort key)
    PRIMARY KEY (node_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_tags_tag   ON tags(tag);
CREATE INDEX IF NOT EXISTS idx_tags_depth ON tags(depth);

CREATE TABLE IF NOT EXISTS forget_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id       TEXT NOT NULL,                     -- the soft-invalidated node
    reason        TEXT NOT NULL,                     -- 'superseded by X' / 'user corrected'
    superseded_by TEXT,                              -- successor to merge into (nullable)
    ts            REAL NOT NULL,
    merged        INTEGER NOT NULL DEFAULT 0         -- 0 = pending latent merge, 1 = done
);
CREATE INDEX IF NOT EXISTS idx_forget_node   ON forget_events(node_id);
CREATE INDEX IF NOT EXISTS idx_forget_merged ON forget_events(merged);

CREATE TABLE IF NOT EXISTS wiki_docs (
    doc_id    TEXT PRIMARY KEY,
    type      TEXT NOT NULL,
    title     TEXT NOT NULL,
    tags      TEXT NOT NULL,                      -- JSON array
    body      TEXT NOT NULL,                      -- markdown (inline [[refs]]/#tags)
    ts        REAL NOT NULL,
    body_mode TEXT                                -- 'generated' (from refs) | 'authored'
);
CREATE TABLE IF NOT EXISTS wiki_refs (
    doc_id      TEXT NOT NULL,
    node_id     TEXT NOT NULL,                     -- a RAG node this doc cites
    stale       INTEGER NOT NULL DEFAULT 0,        -- 1 = target gone/invalid
    reason      TEXT,                              -- WHY stale (explicit code)
    fingerprint TEXT,                              -- node content/tags hash at link time
    outdated    TEXT,                              -- drift code when the node changed since
    PRIMARY KEY (doc_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_wikiref_node ON wiki_refs(node_id);
CREATE INDEX IF NOT EXISTS idx_wikiref_doc  ON wiki_refs(doc_id);

CREATE TABLE IF NOT EXISTS okf_fields (
    doc_id TEXT NOT NULL,
    type   TEXT NOT NULL,                          -- the OKF concept type
    key    TEXT NOT NULL,                          -- a frontmatter field name
    value  TEXT NOT NULL,                          -- one value (list items = rows)
    PRIMARY KEY (doc_id, key, value)
);
CREATE INDEX IF NOT EXISTS idx_okf_kv   ON okf_fields(key, value);
CREATE INDEX IF NOT EXISTS idx_okf_type ON okf_fields(type);

-- MERGE LEDGER : every destructive identity op (forget / merge / collapse) is a
-- row, so it is (a) a persistent REDIRECT absorbed->keeper that outlives the
-- in-memory alias map and (b) REVERSIBLE (snapshot = what revert restores).
CREATE TABLE IF NOT EXISTS merge_ledger (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    absorbed_id TEXT NOT NULL,                     -- id that stops being canonical
    keeper_id   TEXT,                              -- survivor (NULL = forget, no successor)
    kind        TEXT NOT NULL,                     -- 'forget'|'merge'|'lateral'|'duplicate'
    reason      TEXT NOT NULL,                     -- explicit reason, never blank
    snapshot    TEXT NOT NULL,                     -- JSON {state, tags} before the op
    ts          REAL NOT NULL,
    reverted    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ledger_absorbed ON merge_ledger(absorbed_id);
CREATE INDEX IF NOT EXISTS idx_ledger_keeper   ON merge_ledger(keeper_id);

-- Which wiki refs were rewritten by which redirect (so a revert can un-rewrite
-- EXACTLY those, and nothing else).
CREATE TABLE IF NOT EXISTS wiki_ref_remaps (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id   TEXT NOT NULL,
    old_id   TEXT NOT NULL,
    new_id   TEXT NOT NULL,
    had_new  INTEGER NOT NULL DEFAULT 0,           -- doc already linked new_id before
    ordinals TEXT NOT NULL DEFAULT '[]',           -- which [[new]] occurrences came from old
    ts       REAL NOT NULL,
    reverted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_remap_old ON wiki_ref_remaps(old_id);

-- Tool lifecycle events with an explicit reason code (created / reused /
-- promoted / failed / retired / rejected ...) : nothing about a tool is silent.
CREATE TABLE IF NOT EXISTS tool_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_id TEXT NOT NULL,
    event   TEXT NOT NULL,
    reason  TEXT,
    ts      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_toolev_tool ON tool_events(tool_id);

-- DERIVED (inferred) OKF fields live in their OWN table, never in okf_fields :
-- asserted facts always win, and inference is off unless enabled.
CREATE TABLE IF NOT EXISTS okf_derived (
    doc_id TEXT NOT NULL,
    key    TEXT NOT NULL,
    value  TEXT NOT NULL,
    rule   TEXT NOT NULL,                          -- which inference rule produced it
    ts     REAL NOT NULL,
    PRIMARY KEY (doc_id, key, value)
);
CREATE INDEX IF NOT EXISTS idx_okfd_kv ON okf_derived(key, value);

-- SEED QUERIES : semantic queries attached to a portion (or the whole doc,
-- target '*') ; the ranked result is CACHED at creation and re-run offline —
-- a different result is a change the attached portion must absorb.
CREATE TABLE IF NOT EXISTS wiki_seeds (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id  TEXT NOT NULL,
    seed_id TEXT NOT NULL,
    query   TEXT NOT NULL,
    target  TEXT NOT NULL DEFAULT '*',              -- portion id | '*'
    k       INTEGER NOT NULL DEFAULT 7,
    cached  TEXT NOT NULL,                         -- JSON ranked node ids
    ts      REAL NOT NULL,
    UNIQUE (doc_id, seed_id)
);
-- ANNOTATIONS : typed notes on a portion / var / ref / the doc ('*').
CREATE TABLE IF NOT EXISTS wiki_annotations (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id  TEXT NOT NULL,
    target  TEXT NOT NULL,
    kind    TEXT NOT NULL,                         -- note | purpose | keep | todo
    note    TEXT NOT NULL,
    author  TEXT,
    ts      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wikiann_doc ON wiki_annotations(doc_id);
-- OPERATIONS on wiki objects (set / remove / replace a var or portion) :
-- a ref never silently becomes a different string — every edit is a row
-- with its before / after, reversible.
CREATE TABLE IF NOT EXISTS wiki_ops (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id   TEXT NOT NULL,
    target   TEXT NOT NULL,                        -- 'var:<name>' | 'portion:<id>'
    op       TEXT NOT NULL,                        -- set | remove
    before   TEXT,                                 -- JSON state before (NULL = created)
    after    TEXT,                                 -- JSON state after  (NULL = removed)
    ts       REAL NOT NULL,
    reverted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_wikiops_doc ON wiki_ops(doc_id);
-- PENDING changes a doc must absorb (a seed whose result moved on an authored
-- or kept target) : what changed, for whom, until resolved.
CREATE TABLE IF NOT EXISTS wiki_pending (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id   TEXT NOT NULL,
    target   TEXT NOT NULL,
    reason   TEXT NOT NULL,
    detail   TEXT NOT NULL,                        -- JSON
    ts       REAL NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_wikipend_doc ON wiki_pending(doc_id);

-- Out-of-vocabulary OKF terms (a new concept `type`) are PRESERVED as proposals
-- rather than rejected or silently accepted : vetted later (accepted/rejected).
CREATE TABLE IF NOT EXISTS okf_vocab (
    kind      TEXT NOT NULL,                       -- 'type'
    value     TEXT NOT NULL,
    status    TEXT NOT NULL,                       -- 'proposed'|'accepted'|'rejected'
    first_doc TEXT,
    ts        REAL NOT NULL,
    PRIMARY KEY (kind, value)
);
"""

#: Columns added after a table first shipped ; applied idempotently on open so an
#: existing journal file keeps working (the ONLY migration-ish step we do).
_LATE_COLUMNS = [
    ("wiki_refs", "reason", "TEXT"),
    ("wiki_refs", "fingerprint", "TEXT"),
    ("wiki_refs", "outdated", "TEXT"),
    ("wiki_docs", "body_mode", "TEXT"),
]


class Journal:
    """Append-only SQLite usage log. `path=":memory:"` (default) is ephemeral ;
    pass a file path for a journal that outlives the process."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        for table, col, decl in _LATE_COLUMNS:
            have = {r["name"] for r in
                    self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if col not in have:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        self.conn.commit()

    # -- write ---------------------------------------------------------------

    def log_retrieval(self, query_text: str,
                      returned_node_ids: Sequence[str],
                      ts: Optional[float] = None) -> int:
        """Record one search : a `retrievals` row + one `access_events` row per
        returned id (rank 0 = best). Returns the new retrieval_id — the single
        handle threaded to the caller for a later `mark_useful`."""
        ts = time.time() if ts is None else ts
        ids = [str(i) for i in returned_node_ids]
        cur = self.conn.execute(
            "INSERT INTO retrievals(query_text, returned_node_ids, useful, ts) "
            "VALUES (?, ?, NULL, ?)",
            (query_text, json.dumps(ids), ts),
        )
        rid = int(cur.lastrowid)
        self.conn.executemany(
            "INSERT INTO access_events(node_id, retrieval_id, rank, ts) "
            "VALUES (?, ?, ?, ?)",
            [(nid, rid, rank, ts) for rank, nid in enumerate(ids)],
        )
        self.conn.commit()
        return rid

    def mark_useful(self, retrieval_id: int, score: int) -> None:
        """Annotate a past retrieval with a 0/1/2 usefulness score (the
        supervised label a decay-fit consumes). Targets `retrievals.id`."""
        if score not in (0, 1, 2):
            raise ValueError(f"score must be 0, 1 or 2 ; got {score!r}")
        self.conn.execute(
            "UPDATE retrievals SET useful = ? WHERE id = ?", (score, retrieval_id)
        )
        self.conn.commit()

    def log_hop(self, src_id: str, dst_id: str,
                ts: Optional[float] = None) -> None:
        """Record one multi-hop transition src → dst (dst accumulates spike
        energy). The Chasles trigger is then SQL-derivable : `hop_target_counts`
        is the n_spike analogue, `modal_next` follows the modal path."""
        ts = time.time() if ts is None else ts
        self.conn.execute(
            "INSERT INTO hops(src_id, dst_id, ts) VALUES (?, ?, ?)",
            (str(src_id), str(dst_id), ts),
        )
        self.conn.commit()

    def log_path(self, node_ids: Sequence[str],
                 ts: Optional[float] = None) -> Optional[int]:
        """Record one TRAVERSED path (>= 2 nodes) as a first-class, countable
        unit — the Chasles relation is about a *path*, not a single hop. The
        `signature` ('A>B>C>D') is the group key : how often the same path is
        travelled becomes a plain `GROUP BY signature HAVING COUNT(*) >= k`
        query. Append-only ; returns the row id (None if < 2 nodes)."""
        ids = [str(i) for i in node_ids]
        if len(ids) < 2:
            return None
        ts = time.time() if ts is None else ts
        cur = self.conn.execute(
            "INSERT INTO path_traversals(signature, start_id, end_id, "
            "intermediates, length, ts) VALUES (?, ?, ?, ?, ?, ?)",
            (">".join(ids), ids[0], ids[-1], json.dumps(ids[1:-1]),
             len(ids), ts),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def log_collision_event(self, kind: str, *, child_id: Optional[str],
                            parent_ids: Sequence[str],
                            anchor_ids: Sequence[str] = (),
                            trigger_distance: Optional[float] = None,
                            threshold: Optional[float] = None,
                            ts: Optional[float] = None) -> int:
        """Append one collision/compression event to the audit log (mnema's
        CollisionEvent → SQL). `kind` ∈ {'fission', 'chasles', 'lateral'}.
        Returns the new event id."""
        ts = time.time() if ts is None else ts
        cur = self.conn.execute(
            "INSERT INTO collision_events(kind, child_id, parent_ids, anchor_ids,"
            " trigger_distance, threshold, ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (kind, child_id, json.dumps([str(i) for i in parent_ids]),
             json.dumps([str(i) for i in anchor_ids]),
             trigger_distance, threshold, ts),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def log_tags(self, node_id: str, tags: Sequence[str]) -> None:
        """Index a node's hierarchical tags (idempotent). Empty/None tags are
        skipped ; `depth` (the `:` count) is stored as the glossary sort key."""
        rows = [(str(node_id), t, t.count(":"))
                for t in (tags or []) if isinstance(t, str) and t]
        if rows:
            self.conn.executemany(
                "INSERT OR IGNORE INTO tags(node_id, tag, depth) VALUES (?, ?, ?)",
                rows,
            )
            self.conn.commit()

    def log_forget(self, node_id: str, reason: str,
                   superseded_by: Optional[str] = None,
                   ts: Optional[float] = None) -> int:
        """Record a soft-invalidation as a DB event so the LATENT merge can
        consume it later (merged=0 until processed). Returns the event id."""
        ts = time.time() if ts is None else ts
        cur = self.conn.execute(
            "INSERT INTO forget_events(node_id, reason, superseded_by, ts, merged)"
            " VALUES (?, ?, ?, ?, 0)",
            (str(node_id), str(reason),
             None if superseded_by is None else str(superseded_by), ts),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def pending_forgets(self) -> List[dict]:
        """Forget events not yet processed by the latent merge (merged=0),
        oldest first. The offline merge reads these, acts, then marks them."""
        rows = self.conn.execute(
            "SELECT id, node_id, reason, superseded_by, ts FROM forget_events "
            "WHERE merged = 0 ORDER BY id ASC"
        ).fetchall()
        return [{"id": r["id"], "node_id": r["node_id"], "reason": r["reason"],
                 "superseded_by": r["superseded_by"], "ts": r["ts"]}
                for r in rows]

    def mark_forget_merged(self, event_id: int) -> None:
        """Mark a forget event as processed by the latent merge (idempotent)."""
        self.conn.execute(
            "UPDATE forget_events SET merged = 1 WHERE id = ?", (int(event_id),))
        self.conn.commit()

    # -- wiki layer (OKF docs <-> RAG nodes) ---------------------------------

    def upsert_wiki_doc(self, doc_id: str, type: str, title: str,
                        tags: Sequence[str], body: str,
                        ts: Optional[float] = None,
                        body_mode: Optional[str] = None) -> None:
        """Create or replace a wiki doc's content (frontmatter + body).
        `body_mode` : 'generated' (the body is the deterministic rendering of
        its refs — safe to regenerate) or 'authored' (prose someone wrote —
        never overwritten automatically). None keeps the existing mode
        (defaulting to 'generated' for a new doc)."""
        ts = time.time() if ts is None else ts
        if body_mode is None:
            row = self.conn.execute(
                "SELECT body_mode FROM wiki_docs WHERE doc_id = ?",
                (str(doc_id),)).fetchone()
            body_mode = (row["body_mode"] if row is not None else None) or "generated"
        self.conn.execute(
            "INSERT INTO wiki_docs(doc_id, type, title, tags, body, ts, body_mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(doc_id) DO UPDATE SET "
            "type=excluded.type, title=excluded.title, tags=excluded.tags, "
            "body=excluded.body, ts=excluded.ts, body_mode=excluded.body_mode",
            (str(doc_id), type, title, json.dumps(list(tags)), body, ts, body_mode),
        )
        self.conn.commit()

    def get_wiki_doc(self, doc_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT doc_id, type, title, tags, body, ts, body_mode FROM wiki_docs "
            "WHERE doc_id = ?", (str(doc_id),)).fetchone()
        if row is None:
            return None
        return {"doc_id": row["doc_id"], "type": row["type"],
                "title": row["title"], "tags": json.loads(row["tags"]),
                "body": row["body"], "ts": row["ts"],
                "body_mode": row["body_mode"] or "generated"}

    def set_wiki_refs(self, doc_id: str, node_ids: Sequence[str],
                      fingerprints: Optional[dict] = None) -> None:
        """Replace a doc's node refs (the canonical wiki<->node links), with
        each node's fingerprint at link time when known ({node_id: fp})."""
        fps = fingerprints or {}
        self.conn.execute("DELETE FROM wiki_refs WHERE doc_id = ?", (str(doc_id),))
        self.conn.executemany(
            "INSERT OR IGNORE INTO wiki_refs(doc_id, node_id, stale, fingerprint) "
            "VALUES (?, ?, 0, ?)",
            [(str(doc_id), str(n), fps.get(str(n))) for n in node_ids])
        self.conn.commit()

    def set_ref_fingerprint(self, doc_id: str, node_id: str,
                            fingerprint: Optional[str],
                            clear_outdated: bool = True) -> None:
        """Record what the doc last saw of this node (and clear its drift)."""
        if clear_outdated:
            self.conn.execute(
                "UPDATE wiki_refs SET fingerprint = ?, outdated = NULL "
                "WHERE doc_id = ? AND node_id = ?",
                (fingerprint, str(doc_id), str(node_id)))
        else:
            self.conn.execute(
                "UPDATE wiki_refs SET fingerprint = ? WHERE doc_id = ? AND node_id = ?",
                (fingerprint, str(doc_id), str(node_id)))
        self.conn.commit()

    def mark_wiki_ref_outdated(self, doc_id: str, node_id: str,
                               reason: Optional[str]) -> None:
        """Flag (reason code) / clear (None) a ref whose node drifted since the
        doc last saw it."""
        self.conn.execute(
            "UPDATE wiki_refs SET outdated = ? WHERE doc_id = ? AND node_id = ?",
            (reason, str(doc_id), str(node_id)))
        self.conn.commit()

    def wiki_refs_for_doc(self, doc_id: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT node_id, stale, reason, fingerprint, outdated FROM wiki_refs "
            "WHERE doc_id = ? ORDER BY node_id", (str(doc_id),)).fetchall()
        return [{"node_id": r["node_id"], "stale": bool(r["stale"]),
                 "reason": r["reason"], "fingerprint": r["fingerprint"],
                 "outdated": r["outdated"]} for r in rows]

    def docs_referencing(self, node_id: str) -> List[str]:
        """Reverse link : which wiki docs cite this node (the RAG->wiki edge)."""
        rows = self.conn.execute(
            "SELECT DISTINCT doc_id FROM wiki_refs WHERE node_id = ? "
            "ORDER BY doc_id", (str(node_id),)).fetchall()
        return [r["doc_id"] for r in rows]

    def remap_wiki_ref(self, doc_id: str, old: str, new: str) -> None:
        """Redirect a ref old->new (a merged node) ; clears any stale flag."""
        self.conn.execute("DELETE FROM wiki_refs WHERE doc_id = ? AND node_id = ?",
                          (str(doc_id), str(old)))
        self.conn.execute(
            "INSERT OR IGNORE INTO wiki_refs(doc_id, node_id, stale) "
            "VALUES (?, ?, 0)", (str(doc_id), str(new)))
        self.conn.commit()

    def mark_wiki_ref_stale(self, doc_id: str, node_id: str,
                            stale: bool = True,
                            reason: Optional[str] = None) -> None:
        """Flag / clear a stale ref WITH its explicit reason code (a cleared
        flag drops the reason)."""
        self.conn.execute(
            "UPDATE wiki_refs SET stale = ?, reason = ? "
            "WHERE doc_id = ? AND node_id = ?",
            (1 if stale else 0, reason if stale else None,
             str(doc_id), str(node_id)))
        self.conn.commit()

    # -- merge ledger : persistent redirects + reversibility ------------------

    def log_merge(self, absorbed_id: str, keeper_id: Optional[str], kind: str,
                  reason: str, snapshot: Optional[dict] = None,
                  ts: Optional[float] = None) -> int:
        """Append one destructive identity op. `keeper_id` None = a forget with
        no successor (a redirect to nowhere) ; otherwise absorbed->keeper is a
        persistent redirect. `snapshot` is what a revert restores."""
        ts = time.time() if ts is None else ts
        cur = self.conn.execute(
            "INSERT INTO merge_ledger(absorbed_id, keeper_id, kind, reason, "
            "snapshot, ts, reverted) VALUES (?, ?, ?, ?, ?, ?, 0)",
            (str(absorbed_id), None if keeper_id is None else str(keeper_id),
             str(kind), str(reason or "unspecified"),
             json.dumps(snapshot or {}), ts))
        self.conn.commit()
        return int(cur.lastrowid)

    def redirect_of(self, node_id: str) -> Optional[str]:
        """ONE redirect step : the keeper the latest non-reverted ledger row
        sends `node_id` to (None if it is canonical / only forgotten)."""
        row = self.conn.execute(
            "SELECT keeper_id FROM merge_ledger WHERE absorbed_id = ? AND "
            "reverted = 0 AND keeper_id IS NOT NULL ORDER BY id DESC LIMIT 1",
            (str(node_id),)).fetchone()
        return None if row is None else row["keeper_id"]

    def active_redirects(self) -> dict:
        """All live redirects {absorbed: keeper} (latest non-reverted row per
        absorbed id) — used to re-hydrate the in-memory alias map on restart."""
        rows = self.conn.execute(
            "SELECT absorbed_id, keeper_id FROM merge_ledger WHERE reverted = 0 "
            "AND keeper_id IS NOT NULL ORDER BY id ASC").fetchall()
        return {r["absorbed_id"]: r["keeper_id"] for r in rows}

    def absorbed_by(self, keeper_id: str) -> List[str]:
        """Direct reverse redirect : ids currently redirected INTO `keeper_id`."""
        rows = self.conn.execute(
            "SELECT DISTINCT absorbed_id FROM merge_ledger WHERE keeper_id = ? "
            "AND reverted = 0 ORDER BY absorbed_id", (str(keeper_id),)).fetchall()
        return [r["absorbed_id"] for r in rows]

    def merge_history(self, node_id: Optional[str] = None,
                      include_reverted: bool = True) -> List[dict]:
        """Ledger rows (newest first), all or for one absorbed id."""
        q = ("SELECT id, absorbed_id, keeper_id, kind, reason, snapshot, ts, "
             "reverted FROM merge_ledger")
        args: tuple = ()
        conds = []
        if node_id is not None:
            conds.append("absorbed_id = ?")
            args += (str(node_id),)
        if not include_reverted:
            conds.append("reverted = 0")
        if conds:
            q += " WHERE " + " AND ".join(conds)
        rows = self.conn.execute(q + " ORDER BY id DESC", args).fetchall()
        return [{"id": r["id"], "absorbed_id": r["absorbed_id"],
                 "keeper_id": r["keeper_id"], "kind": r["kind"],
                 "reason": r["reason"], "snapshot": json.loads(r["snapshot"]),
                 "ts": r["ts"], "reverted": bool(r["reverted"])} for r in rows]

    def mark_merge_reverted(self, ledger_id: int) -> None:
        self.conn.execute("UPDATE merge_ledger SET reverted = 1 WHERE id = ?",
                          (int(ledger_id),))
        self.conn.commit()

    def log_ref_remap(self, doc_id: str, old_id: str, new_id: str, *,
                      had_new: bool = False, ordinals: Sequence[int] = (),
                      ts: Optional[float] = None) -> int:
        """Record one ref rewrite old->new in a doc precisely enough to undo
        it : whether the doc ALREADY linked new (so revert must not unlink it)
        and which `[[new]]` occurrences in the prose came from `[[old]]`."""
        ts = time.time() if ts is None else ts
        cur = self.conn.execute(
            "INSERT INTO wiki_ref_remaps(doc_id, old_id, new_id, had_new, "
            "ordinals, ts, reverted) VALUES (?, ?, ?, ?, ?, ?, 0)",
            (str(doc_id), str(old_id), str(new_id), 1 if had_new else 0,
             json.dumps([int(i) for i in ordinals]), ts))
        self.conn.commit()
        return int(cur.lastrowid)

    def ref_remaps_from(self, old_id: str) -> List[dict]:
        """Live (non-reverted) wiki ref rewrites that sent `old_id` somewhere."""
        rows = self.conn.execute(
            "SELECT id, doc_id, old_id, new_id, had_new, ordinals "
            "FROM wiki_ref_remaps WHERE old_id = ? AND reverted = 0 "
            "ORDER BY id ASC", (str(old_id),)).fetchall()
        return [{"id": r["id"], "doc_id": r["doc_id"], "old_id": r["old_id"],
                 "new_id": r["new_id"], "had_new": bool(r["had_new"]),
                 "ordinals": json.loads(r["ordinals"])} for r in rows]

    def mark_ref_remap_reverted(self, remap_id: int) -> None:
        self.conn.execute("UPDATE wiki_ref_remaps SET reverted = 1 WHERE id = ?",
                          (int(remap_id),))
        self.conn.commit()

    # -- tool lifecycle events (explicit reason codes) ------------------------

    def log_tool_event(self, tool_id: str, event: str,
                       reason: Optional[str] = None,
                       ts: Optional[float] = None) -> int:
        ts = time.time() if ts is None else ts
        cur = self.conn.execute(
            "INSERT INTO tool_events(tool_id, event, reason, ts) VALUES (?, ?, ?, ?)",
            (str(tool_id), str(event), reason, ts))
        self.conn.commit()
        return int(cur.lastrowid)

    def tool_events(self, tool_id: Optional[str] = None) -> List[dict]:
        if tool_id is None:
            rows = self.conn.execute(
                "SELECT tool_id, event, reason, ts FROM tool_events "
                "ORDER BY id ASC").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT tool_id, event, reason, ts FROM tool_events "
                "WHERE tool_id = ? ORDER BY id ASC", (str(tool_id),)).fetchall()
        return [{"tool_id": r["tool_id"], "event": r["event"],
                 "reason": r["reason"], "ts": r["ts"]} for r in rows]

    # -- derived OKF fields (materialized inference, separate table) ----------

    def set_okf_derived(self, doc_id: str, rows: Sequence[tuple],
                        ts: Optional[float] = None) -> int:
        """Replace a doc's DERIVED fields with `rows` = [(key, value, rule)].
        Never touches okf_fields (the asserted facts)."""
        ts = time.time() if ts is None else ts
        self.conn.execute("DELETE FROM okf_derived WHERE doc_id = ?", (str(doc_id),))
        data = [(str(doc_id), str(k), str(v), str(rule), ts) for k, v, rule in rows]
        if data:
            self.conn.executemany(
                "INSERT OR IGNORE INTO okf_derived(doc_id, key, value, rule, ts) "
                "VALUES (?, ?, ?, ?, ?)", data)
        self.conn.commit()
        return len(data)

    def okf_derived_of(self, doc_id: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT key, value, rule FROM okf_derived WHERE doc_id = ? "
            "ORDER BY key, value", (str(doc_id),)).fetchall()
        return [{"key": r["key"], "value": r["value"], "rule": r["rule"]}
                for r in rows]

    def okf_derived_docs_where(self, key: str,
                               value: Optional[str] = None) -> List[str]:
        if value is None:
            rows = self.conn.execute(
                "SELECT DISTINCT doc_id FROM okf_derived WHERE key = ? "
                "ORDER BY doc_id", (str(key),)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT DISTINCT doc_id FROM okf_derived WHERE key = ? AND value = ?"
                " ORDER BY doc_id", (str(key), str(value))).fetchall()
        return [r["doc_id"] for r in rows]

    def clear_okf_derived(self) -> None:
        self.conn.execute("DELETE FROM okf_derived")
        self.conn.commit()

    # -- wiki objects : seeds / annotations / ops / pending -------------------

    def upsert_seed(self, doc_id: str, seed_id: str, query: str, target: str,
                    k: int, cached: Sequence[str],
                    ts: Optional[float] = None) -> None:
        ts = time.time() if ts is None else ts
        self.conn.execute(
            "INSERT INTO wiki_seeds(doc_id, seed_id, query, target, k, cached, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(doc_id, seed_id) DO UPDATE SET "
            "query=excluded.query, target=excluded.target, k=excluded.k, "
            "cached=excluded.cached, ts=excluded.ts",
            (str(doc_id), str(seed_id), query, target or "*", int(k),
             json.dumps([str(i) for i in cached]), ts))
        self.conn.commit()

    def set_seed_cache(self, doc_id: str, seed_id: str, cached: Sequence[str],
                       ts: Optional[float] = None) -> None:
        ts = time.time() if ts is None else ts
        self.conn.execute(
            "UPDATE wiki_seeds SET cached = ?, ts = ? WHERE doc_id = ? AND seed_id = ?",
            (json.dumps([str(i) for i in cached]), ts, str(doc_id), str(seed_id)))
        self.conn.commit()

    def seeds_for_doc(self, doc_id: Optional[str] = None) -> List[dict]:
        q = "SELECT doc_id, seed_id, query, target, k, cached, ts FROM wiki_seeds"
        args: tuple = ()
        if doc_id is not None:
            q += " WHERE doc_id = ?"
            args = (str(doc_id),)
        rows = self.conn.execute(q + " ORDER BY id ASC", args).fetchall()
        return [{"doc_id": r["doc_id"], "seed_id": r["seed_id"], "query": r["query"],
                 "target": r["target"], "k": r["k"], "cached": json.loads(r["cached"]),
                 "ts": r["ts"]} for r in rows]

    def remove_seed(self, doc_id: str, seed_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM wiki_seeds WHERE doc_id = ? AND seed_id = ?",
            (str(doc_id), str(seed_id)))
        self.conn.commit()
        return cur.rowcount > 0

    def add_annotation(self, doc_id: str, target: str, kind: str, note: str,
                       author: Optional[str] = None,
                       ts: Optional[float] = None) -> int:
        ts = time.time() if ts is None else ts
        cur = self.conn.execute(
            "INSERT INTO wiki_annotations(doc_id, target, kind, note, author, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(doc_id), str(target), kind, note, author, ts))
        self.conn.commit()
        return int(cur.lastrowid)

    def annotations_for_doc(self, doc_id: str,
                            target: Optional[str] = None) -> List[dict]:
        q = ("SELECT id, doc_id, target, kind, note, author, ts FROM wiki_annotations "
             "WHERE doc_id = ?")
        args: list = [str(doc_id)]
        if target is not None:
            q += " AND target = ?"
            args.append(str(target))
        rows = self.conn.execute(q + " ORDER BY id ASC", args).fetchall()
        return [{"id": r["id"], "doc_id": r["doc_id"], "target": r["target"],
                 "kind": r["kind"], "note": r["note"], "author": r["author"],
                 "ts": r["ts"]} for r in rows]

    def remove_annotation(self, annotation_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM wiki_annotations WHERE id = ?",
                                (int(annotation_id),))
        self.conn.commit()
        return cur.rowcount > 0

    def log_wiki_op(self, doc_id: str, target: str, op: str,
                    before: Optional[dict], after: Optional[dict],
                    ts: Optional[float] = None) -> int:
        ts = time.time() if ts is None else ts
        cur = self.conn.execute(
            "INSERT INTO wiki_ops(doc_id, target, op, before, after, ts, reverted) "
            "VALUES (?, ?, ?, ?, ?, ?, 0)",
            (str(doc_id), str(target), op,
             None if before is None else json.dumps(before),
             None if after is None else json.dumps(after), ts))
        self.conn.commit()
        return int(cur.lastrowid)

    def wiki_ops_for_doc(self, doc_id: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT id, doc_id, target, op, before, after, ts, reverted FROM wiki_ops "
            "WHERE doc_id = ? ORDER BY id DESC", (str(doc_id),)).fetchall()
        return [{"id": r["id"], "doc_id": r["doc_id"], "target": r["target"],
                 "op": r["op"],
                 "before": None if r["before"] is None else json.loads(r["before"]),
                 "after": None if r["after"] is None else json.loads(r["after"]),
                 "ts": r["ts"], "reverted": bool(r["reverted"])} for r in rows]

    def get_wiki_op(self, op_id: int) -> Optional[dict]:
        r = self.conn.execute(
            "SELECT id, doc_id, target, op, before, after, ts, reverted FROM wiki_ops "
            "WHERE id = ?", (int(op_id),)).fetchone()
        if r is None:
            return None
        return {"id": r["id"], "doc_id": r["doc_id"], "target": r["target"],
                "op": r["op"],
                "before": None if r["before"] is None else json.loads(r["before"]),
                "after": None if r["after"] is None else json.loads(r["after"]),
                "ts": r["ts"], "reverted": bool(r["reverted"])}

    def mark_wiki_op_reverted(self, op_id: int) -> None:
        self.conn.execute("UPDATE wiki_ops SET reverted = 1 WHERE id = ?", (int(op_id),))
        self.conn.commit()

    def add_pending(self, doc_id: str, target: str, reason: str, detail: dict,
                    ts: Optional[float] = None) -> int:
        ts = time.time() if ts is None else ts
        cur = self.conn.execute(
            "INSERT INTO wiki_pending(doc_id, target, reason, detail, ts, resolved) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (str(doc_id), str(target), reason, json.dumps(detail), ts))
        self.conn.commit()
        return int(cur.lastrowid)

    def pending_for_doc(self, doc_id: Optional[str] = None) -> List[dict]:
        q = ("SELECT id, doc_id, target, reason, detail, ts FROM wiki_pending "
             "WHERE resolved = 0")
        args: tuple = ()
        if doc_id is not None:
            q += " AND doc_id = ?"
            args = (str(doc_id),)
        rows = self.conn.execute(q + " ORDER BY id ASC", args).fetchall()
        return [{"id": r["id"], "doc_id": r["doc_id"], "target": r["target"],
                 "reason": r["reason"], "detail": json.loads(r["detail"]),
                 "ts": r["ts"]} for r in rows]

    def resolve_pending(self, doc_id: str, target: Optional[str] = None) -> int:
        if target is None:
            cur = self.conn.execute(
                "UPDATE wiki_pending SET resolved = 1 WHERE doc_id = ? AND resolved = 0",
                (str(doc_id),))
        else:
            cur = self.conn.execute(
                "UPDATE wiki_pending SET resolved = 1 WHERE doc_id = ? AND target = ? "
                "AND resolved = 0", (str(doc_id), str(target)))
        self.conn.commit()
        return cur.rowcount

    # -- OKF vocabulary proposals --------------------------------------------

    def propose_vocab(self, kind: str, value: str, doc_id: Optional[str] = None,
                      ts: Optional[float] = None) -> bool:
        """Preserve an out-of-vocabulary term as a PROPOSAL (idempotent ; an
        already-vetted term keeps its status). True if newly proposed."""
        ts = time.time() if ts is None else ts
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO okf_vocab(kind, value, status, first_doc, ts) "
            "VALUES (?, ?, 'proposed', ?, ?)",
            (str(kind), str(value), doc_id, ts))
        self.conn.commit()
        return cur.rowcount > 0

    def vet_vocab(self, kind: str, value: str, status: str,
                  ts: Optional[float] = None) -> None:
        """Set a term's status ('accepted' | 'rejected' | 'proposed')."""
        if status not in ("accepted", "rejected", "proposed"):
            raise ValueError(f"bad vocab status {status!r}")
        ts = time.time() if ts is None else ts
        self.conn.execute(
            "INSERT INTO okf_vocab(kind, value, status, first_doc, ts) "
            "VALUES (?, ?, ?, NULL, ?) ON CONFLICT(kind, value) DO UPDATE SET "
            "status = excluded.status", (str(kind), str(value), status, ts))
        self.conn.commit()

    def vocab_status(self, kind: str, value: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT status FROM okf_vocab WHERE kind = ? AND value = ?",
            (str(kind), str(value))).fetchone()
        return None if row is None else row["status"]

    def vocab(self, kind: Optional[str] = None,
              status: Optional[str] = None) -> List[dict]:
        q = "SELECT kind, value, status, first_doc, ts FROM okf_vocab"
        conds, args = [], []
        if kind is not None:
            conds.append("kind = ?")
            args.append(str(kind))
        if status is not None:
            conds.append("status = ?")
            args.append(str(status))
        if conds:
            q += " WHERE " + " AND ".join(conds)
        rows = self.conn.execute(q + " ORDER BY ts ASC, value", args).fetchall()
        return [{"kind": r["kind"], "value": r["value"], "status": r["status"],
                 "first_doc": r["first_doc"], "ts": r["ts"]} for r in rows]

    def all_wiki_doc_ids(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT doc_id FROM wiki_docs ORDER BY doc_id").fetchall()
        return [r["doc_id"] for r in rows]

    # -- OKF EAV field index (schema-less ; no migrations) --------------------

    def index_okf_fields(self, doc_id: str, type: str, fields: dict) -> None:
        """(Re)index a doc's OKF frontmatter as flat (key, value) rows — every
        field queryable, list items exploded to one row each, no schema/migration.
        Replaces any prior rows for the doc."""
        self.conn.execute("DELETE FROM okf_fields WHERE doc_id = ?", (str(doc_id),))
        rows = []
        for key, val in (fields or {}).items():
            if val is None:
                continue
            vals = val if isinstance(val, (list, tuple)) else [val]
            for v in vals:
                rows.append((str(doc_id), str(type), str(key), str(v)))
        if rows:
            self.conn.executemany(
                "INSERT OR IGNORE INTO okf_fields(doc_id, type, key, value) "
                "VALUES (?, ?, ?, ?)", rows)
        self.conn.commit()

    def okf_docs_where(self, key: str, value: Optional[str] = None) -> List[str]:
        """Doc ids having field `key` (= `value` if given) — querying the EAV
        index by any frontmatter field, no fixed schema needed."""
        if value is None:
            rows = self.conn.execute(
                "SELECT DISTINCT doc_id FROM okf_fields WHERE key = ? "
                "ORDER BY doc_id", (str(key),)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT DISTINCT doc_id FROM okf_fields WHERE key = ? AND value = ? "
                "ORDER BY doc_id", (str(key), str(value))).fetchall()
        return [r["doc_id"] for r in rows]

    def okf_fields_of(self, doc_id: str) -> dict:
        """All indexed frontmatter fields of a doc, {key: [values]}."""
        rows = self.conn.execute(
            "SELECT key, value FROM okf_fields WHERE doc_id = ? ORDER BY key",
            (str(doc_id),)).fetchall()
        out: dict = {}
        for r in rows:
            out.setdefault(r["key"], []).append(r["value"])
        return out

    def okf_schema(self) -> dict:
        """The RECOVERED schema, derived from the data (no registry) : the set of
        frontmatter keys seen per type, {type: [keys]}. This is how you get the
        schema back out of a schema-less OKF bundle."""
        rows = self.conn.execute(
            "SELECT DISTINCT type, key FROM okf_fields ORDER BY type, key"
        ).fetchall()
        out: dict = {}
        for r in rows:
            out.setdefault(r["type"], []).append(r["key"])
        return out

    def okf_types(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT type FROM okf_fields ORDER BY type").fetchall()
        return [r["type"] for r in rows]

    # -- read ----------------------------------------------------------------

    def nodes_with_tag(self, tag: str, hierarchical: bool = True) -> List[str]:
        """Node ids carrying `tag`. With `hierarchical` (default) the tag matches
        as an ANCESTOR too — querying 'health' returns nodes tagged
        'health:condition:x' (SQL : tag = ? OR tag LIKE ?||':%'). Sorted."""
        tag = str(tag)
        if hierarchical:
            rows = self.conn.execute(
                "SELECT DISTINCT node_id FROM tags WHERE tag = ? OR tag LIKE ? "
                "ORDER BY node_id ASC", (tag, tag + ":%"),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT DISTINCT node_id FROM tags WHERE tag = ? ORDER BY node_id",
                (tag,),
            ).fetchall()
        return [r["node_id"] for r in rows]

    def tag_glossary(self) -> List[str]:
        """Distinct tags ordered by hierarchy DEPTH (shallowest first) then
        alphabetically — the SQL hierarchical tag index / glossary."""
        rows = self.conn.execute(
            "SELECT DISTINCT tag, depth FROM tags ORDER BY depth ASC, tag ASC"
        ).fetchall()
        return [r["tag"] for r in rows]

    def hop_target_counts(self) -> List[Tuple[str, int]]:
        """(dst_id, hop_count) most-hopped-to first — the SQL n_spike analogue
        (a node's accumulated activation energy)."""
        rows = self.conn.execute(
            "SELECT dst_id, COUNT(*) AS c FROM hops GROUP BY dst_id "
            "ORDER BY c DESC, dst_id ASC"
        ).fetchall()
        return [(r["dst_id"], int(r["c"])) for r in rows]

    def effective_spike(self, node_id: str) -> int:
        """Refractory-aware spike count : hops landing on `node_id` AFTER its
        last Chasles fire (a compression event that included it as child /
        intermediate / anchor). Append-only refractory — the 'reset' is derived
        from the last-fire timestamp, nothing is mutated. This is the SQL n_spike
        the Chasles trigger reads."""
        nid = str(node_id)
        quoted = f'"{nid}"'
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM hops WHERE dst_id = ? AND ts > COALESCE(("
            "  SELECT MAX(ts) FROM collision_events WHERE kind = 'chasles' AND ("
            "    child_id = ? OR instr(parent_ids, ?) > 0 "
            "    OR instr(anchor_ids, ?) > 0)), 0)",
            (nid, nid, quoted, quoted),
        ).fetchone()
        return int(row["c"])

    def frequent_paths(self, min_len: int = 4, min_freq: int = 2,
                       since_ts: float = 0.0) -> List[dict]:
        """Paths travelled >= `min_freq` times, `min_len`+ nodes long, after
        `since_ts` — the raw 'this path is often taken' query (no refractory).
        `min_len=4` = start + >=2 intermediates + end, matching compress_chasles.
        Most-travelled first. Each row is a Chasles candidate."""
        rows = self.conn.execute(
            "SELECT signature, start_id, end_id, intermediates, length, "
            "COUNT(*) AS freq, MAX(ts) AS last_ts FROM path_traversals "
            "WHERE length >= ? AND ts > ? GROUP BY signature "
            "HAVING freq >= ? ORDER BY freq DESC, signature ASC",
            (min_len, since_ts, min_freq),
        ).fetchall()
        return [self._path_row(r) for r in rows]

    def chasles_candidates(self, min_len: int = 4,
                           min_freq: int = 2) -> List[dict]:
        """The SQL Chasles TRIGGER : frequently-travelled paths whose traversals
        post-date the last Chasles fire for their OWN start/end anchors
        (per-signature append-only refractory — nothing is mutated, the reset is
        derived from collision_events like `effective_spike`). Just query this to
        know which paths to collapse into a shortcut."""
        rows = self.conn.execute(
            "SELECT signature, start_id, end_id, intermediates, length, "
            "COUNT(*) AS freq, MAX(ts) AS last_ts FROM path_traversals p "
            "WHERE length >= ? AND ts > COALESCE((SELECT MAX(ts) FROM "
            "collision_events c WHERE c.kind = 'chasles' "
            "AND instr(c.anchor_ids, '\"'||p.start_id||'\"') > 0 "
            "AND instr(c.anchor_ids, '\"'||p.end_id||'\"') > 0), 0) "
            "GROUP BY signature HAVING freq >= ? "
            "ORDER BY freq DESC, signature ASC",
            (min_len, min_freq),
        ).fetchall()
        return [self._path_row(r) for r in rows]

    @staticmethod
    def _path_row(r) -> dict:
        return {
            "signature": r["signature"],
            "start_id": r["start_id"],
            "end_id": r["end_id"],
            "intermediates": json.loads(r["intermediates"]),
            "length": int(r["length"]),
            "freq": int(r["freq"]),
            "last_ts": r["last_ts"],
        }

    def co_retrieved_pairs(self, window: Optional[int] = None
                           ) -> List[Tuple[Tuple[str, str], int]]:
        """All co-retrieved pairs (a < b) with their co-occurrence count — the
        SQL self-join that feeds lateral collision. `window` (rank distance) None
        = any two nodes in the same retrieval (mnema-style) ; an int restricts to
        pairs within that many ranks (parity with the in-memory adjacency ledger).
        Returns [((a, b), cooc)]."""
        sql = (
            "SELECT a.node_id AS x, b.node_id AS y, COUNT(*) AS cooc "
            "FROM access_events a "
            "JOIN access_events b ON a.retrieval_id = b.retrieval_id "
            "AND a.node_id < b.node_id "
        )
        params: tuple = ()
        if window is not None:
            sql += "AND ABS(a.rank - b.rank) <= ? "
            params = (window,)
        sql += "GROUP BY a.node_id, b.node_id"
        rows = self.conn.execute(sql, params).fetchall()
        return [((r["x"], r["y"]), int(r["cooc"])) for r in rows]

    def modal_next(self, src_id: str) -> Optional[str]:
        """The most frequent successor of `src_id` in the hop log — the modal
        step a SQL-driven `chasles_path` follows. None if src never hopped."""
        row = self.conn.execute(
            "SELECT dst_id FROM hops WHERE src_id = ? "
            "GROUP BY dst_id ORDER BY COUNT(*) DESC, dst_id ASC LIMIT 1",
            (str(src_id),),
        ).fetchone()
        return row["dst_id"] if row is not None else None

    def collision_events(self, kind: Optional[str] = None) -> List[dict]:
        """The collision/compression audit trail, oldest first. Filter by `kind`
        ('fission' | 'chasles' | 'lateral') or None for all."""
        if kind is None:
            rows = self.conn.execute(
                "SELECT * FROM collision_events ORDER BY id ASC").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM collision_events WHERE kind = ? ORDER BY id ASC",
                (kind,)).fetchall()
        return [{
            "id": int(r["id"]), "kind": r["kind"], "child_id": r["child_id"],
            "parent_ids": json.loads(r["parent_ids"] or "[]"),
            "anchor_ids": json.loads(r["anchor_ids"] or "[]"),
            "trigger_distance": r["trigger_distance"],
            "threshold": r["threshold"], "ts": float(r["ts"]),
        } for r in rows]

    def find_co_retrieved(self, seed_node_ids: Sequence[str],
                          k: int = 7) -> List[Tuple[str, int]]:
        """Nodes co-retrieved with the seeds, most-frequent first. A SQL
        self-join on `access_events.retrieval_id` — edgeless, never touches
        embeddings. Returns [(node_id, cooc_count)]."""
        seeds = [str(i) for i in seed_node_ids]
        if not seeds:
            return []
        ph = ",".join("?" for _ in seeds)
        sql = (
            "SELECT b.node_id AS node_id, COUNT(*) AS cooc "
            "FROM access_events a "
            "JOIN access_events b ON a.retrieval_id = b.retrieval_id "
            f"WHERE a.node_id IN ({ph}) AND b.node_id NOT IN ({ph}) "
            "GROUP BY b.node_id "
            "ORDER BY cooc DESC, b.node_id ASC "
            "LIMIT ?"
        )
        rows = self.conn.execute(sql, (*seeds, *seeds, k)).fetchall()
        return [(r["node_id"], int(r["cooc"])) for r in rows]

    def node_usefulness(self, node_id: str) -> dict:
        """First-order feedback for a node : how many USEFUL (>=2) vs USELESS (0)
        retrievals returned it (from the mark_useful labels). The credibility
        signal the wiki carries (OKF-style usage_count)."""
        pos = self.conn.execute(
            "SELECT COUNT(DISTINCT r.id) AS c FROM retrievals r "
            "JOIN access_events a ON a.retrieval_id = r.id "
            "WHERE a.node_id = ? AND r.useful >= 2", (str(node_id),)).fetchone()["c"]
        neg = self.conn.execute(
            "SELECT COUNT(DISTINCT r.id) AS c FROM retrievals r "
            "JOIN access_events a ON a.retrieval_id = r.id "
            "WHERE a.node_id = ? AND r.useful = 0", (str(node_id),)).fetchone()["c"]
        return {"useful": int(pos), "useless": int(neg)}

    def retrieval_returned_ids(self, retrieval_id: int) -> List[str]:
        """The node ids a past retrieval returned (rank order) — used to score it
        useful once we learn which of them were actually used."""
        rows = self.conn.execute(
            "SELECT node_id FROM access_events WHERE retrieval_id = ? "
            "ORDER BY rank ASC", (int(retrieval_id),),
        ).fetchall()
        return [r["node_id"] for r in rows]

    def access_timestamps(self, node_id: str) -> List[float]:
        """Every access ts for a node, ascending — the raw history a need-odds
        power-law decay would sum over."""
        rows = self.conn.execute(
            "SELECT ts FROM access_events WHERE node_id = ? ORDER BY ts ASC",
            (str(node_id),),
        ).fetchall()
        return [float(r["ts"]) for r in rows]

    def useful_retrievals(self, min_score: int = 2,
                          max_score: Optional[int] = None
                          ) -> List[Tuple[int, str, List[str]]]:
        """Scored retrievals in [min_score, max_score] — the decay-fit training
        set. min_score=2 (default) = the useful positives ; min_score=0,
        max_score=0 = the useless negatives. Returns [(retrieval_id, query_text,
        returned_node_ids)]."""
        clause = "useful IS NOT NULL AND useful >= ?"
        params: List[int] = [min_score]
        if max_score is not None:
            clause += " AND useful <= ?"
            params.append(max_score)
        rows = self.conn.execute(
            f"SELECT id, query_text, returned_node_ids FROM retrievals "
            f"WHERE {clause} ORDER BY id ASC", params,
        ).fetchall()
        return [(int(r["id"]), r["query_text"] or "",
                 json.loads(r["returned_node_ids"] or "[]")) for r in rows]

    def counts(self) -> Tuple[int, int]:
        """(n_retrievals, n_access_events) — cheap health/telemetry probe."""
        r = self.conn.execute("SELECT COUNT(*) AS c FROM retrievals").fetchone()
        a = self.conn.execute("SELECT COUNT(*) AS c FROM access_events").fetchone()
        return int(r["c"]), int(a["c"])

    def close(self) -> None:
        self.conn.close()
