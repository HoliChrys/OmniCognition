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
    doc_id  TEXT PRIMARY KEY,
    type    TEXT NOT NULL,
    title   TEXT NOT NULL,
    tags    TEXT NOT NULL,                        -- JSON array
    body    TEXT NOT NULL,                        -- markdown (inline [[refs]]/#tags)
    ts      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS wiki_refs (
    doc_id  TEXT NOT NULL,
    node_id TEXT NOT NULL,                         -- a RAG node this doc cites
    stale   INTEGER NOT NULL DEFAULT 0,            -- 1 = target gone/invalid
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
                        ts: Optional[float] = None) -> None:
        """Create or replace a wiki doc's content (frontmatter + body)."""
        ts = time.time() if ts is None else ts
        self.conn.execute(
            "INSERT INTO wiki_docs(doc_id, type, title, tags, body, ts) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(doc_id) DO UPDATE SET "
            "type=excluded.type, title=excluded.title, tags=excluded.tags, "
            "body=excluded.body, ts=excluded.ts",
            (str(doc_id), type, title, json.dumps(list(tags)), body, ts),
        )
        self.conn.commit()

    def get_wiki_doc(self, doc_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT doc_id, type, title, tags, body, ts FROM wiki_docs "
            "WHERE doc_id = ?", (str(doc_id),)).fetchone()
        if row is None:
            return None
        return {"doc_id": row["doc_id"], "type": row["type"],
                "title": row["title"], "tags": json.loads(row["tags"]),
                "body": row["body"], "ts": row["ts"]}

    def set_wiki_refs(self, doc_id: str, node_ids: Sequence[str]) -> None:
        """Replace a doc's node refs (the canonical wiki<->node links)."""
        self.conn.execute("DELETE FROM wiki_refs WHERE doc_id = ?", (str(doc_id),))
        self.conn.executemany(
            "INSERT OR IGNORE INTO wiki_refs(doc_id, node_id, stale) "
            "VALUES (?, ?, 0)", [(str(doc_id), str(n)) for n in node_ids])
        self.conn.commit()

    def wiki_refs_for_doc(self, doc_id: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT node_id, stale FROM wiki_refs WHERE doc_id = ? "
            "ORDER BY node_id", (str(doc_id),)).fetchall()
        return [{"node_id": r["node_id"], "stale": bool(r["stale"])} for r in rows]

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
                            stale: bool = True) -> None:
        self.conn.execute(
            "UPDATE wiki_refs SET stale = ? WHERE doc_id = ? AND node_id = ?",
            (1 if stale else 0, str(doc_id), str(node_id)))
        self.conn.commit()

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
