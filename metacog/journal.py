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

    # -- read ----------------------------------------------------------------

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

    def useful_retrievals(self, min_score: int = 2
                          ) -> List[Tuple[int, str, List[str]]]:
        """Retrievals whose usefulness score >= min_score — the training set for
        a decay-fit. Returns [(retrieval_id, query_text, returned_node_ids)]."""
        rows = self.conn.execute(
            "SELECT id, query_text, returned_node_ids FROM retrievals "
            "WHERE useful IS NOT NULL AND useful >= ? ORDER BY id ASC",
            (min_score,),
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
