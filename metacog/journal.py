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

CREATE TABLE IF NOT EXISTS tags (
    node_id  TEXT NOT NULL,
    tag      TEXT NOT NULL,                          -- hierarchical: a:b:c
    depth    INTEGER NOT NULL,                        -- number of ':' (sort key)
    PRIMARY KEY (node_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_tags_tag   ON tags(tag);
CREATE INDEX IF NOT EXISTS idx_tags_depth ON tags(depth);
"""


class Journal:
    """Append-only SQLite usage log. `path=":memory:"` (default) is ephemeral ;
    pass a file path for a journal that outlives the process."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
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
