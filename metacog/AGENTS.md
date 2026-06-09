# metacog — core library

## Purpose

The MetaCog-Mem library: an epistemically-typed Point manifold, edge-free
geometric relations, an uncertainty-governed retrieval walk, the event
subsystem, and the MCP service. Importable as `metacog`.

## Ownership

Owns all runtime memory/retrieval/reasoning logic. Benchmarks and tests depend
on it; it depends on neither. Inherits all root invariants (edge-free,
hyperparameter-free, anti-laundering, never-cache-empty, save/load rebuild).

## Local Contracts

- `epistemic.py` — `Point` (one schema for all kinds), `PointKind`
  (FACT / THOUGHT / ACTION / EVENT), `SourceClass`, epistemic state. A new kind
  is a deliberate, schema-level change.
- `geometry.py` — `apply_pull(a, b, sign, t_now)` moves `a` toward `b`; first
  pull = 1/(1+0) = 1.0. The ONLY way to relate points. Plus retrieval / spreading.
- `memory.py` — `Memory`: ingest/retrieve; event subsystem (`ingest_event`,
  `consolidate_events` multi-signal merge, `detect_event_type` centroid routing,
  `event_centroid`/`context_centroid`, `event_cluster`/`context_members`,
  `event_absorb`); **named bags** (`bag_add/items/clear` with a `bag` kwarg,
  `bag_intersect`/`bag_union`/`bag_names`, `bag_publish_cluster/_schema`; each
  bag carries description+schema via `bag_meta`/`bag_overview`, and the
  description tracks the schema — a schema change re-derives the description;
  `curated_bags` = the map of non-`event:` bags for the answer);
  bi-temporal validity (`is_valid`/`invalidate`); `scoped_answer` (two-phase
  scoped→knowledge_base cascade) and `filter_list` — the NON-kNN filtered
  listing (scan by `event_id` / `date_from`/`date_to` / `tags` / `kind`,
  returns every match ordered by date, no embedding/walk/top-k; also reachable
  via `scoped_answer(list_only=True)`). The default bag mirrors into `_bag` for
  backwards compatibility.
- `meta_walk.py` — `MetaWalker`: re-anchors on the nearest ACTION each stage and
  spreads from it; stops on `step().done` (σ/GUM), not a fixed cap. `_relevant_cum`
  is the committed evidence set (uncapped); `_composable_evidence` is the bounded
  view for synthesis. `generate_action` / `meta_thought` produce GENERATOR nodes.
  `_maybe_event_scan`: meta-cognition trigger — when a focused fact carries
  `event:in:<id>` and the query is a set request, launches `filter_list` over
  that event (date-bounded by a year in the query) and folds the exhaustive scan
  into the evidence (recall-gated, once per event). Replaces the parallel event
  query with a targeted filtered one inside the single walk.
- `event_schema.py` — schema induction per event TYPE (= a CONTEXT), slot-filling
  (`fill_event_schema` with scoped+KB per core slot and cross-slot dedup),
  `event_search`, `event_action_enrich` (meta-cognition bridge: the schema result
  seeds a `generate_action` beacon that re-anchors the walk; seeds on the bag
  INTERSECTION when non-empty).
- Extractors (`event_extractor.py`, `keywords.py`, `entities.py`, `atomic.py`):
  cached, failure-safe, never cache empty.
- `enumeration.py` — retrieve/"bag" mode detection + `format_bag_answer`.
  `bag_render.py` — the agent's bag rendering strategies: `raw`, `extract`
  (head/tail ellipsis), `interpret` (one interpretation per node), `mapreduce`
  (rolling batch summary) + placement `inject`/`bare`. `render_bag` (one list)
  and `render_bags` (a MAP of lists -> dynamic multi-value inject: prose with one
  `{{name}}` placeholder per list, unplaced lists appended). `strategy="auto"`/
  `placement="auto"` let the agent decide.
- `mcp_server.py` — the MCP tool surface (`build_app`). `event:action` beacons are
  excluded from `retrieve`'s search pool. Bag-domain tools: `collect(ids, bag,
  description)`, `bag(name)`, `bags()` (overview with description/schema for
  decisions), `bag_render(name, strategy, placement)`. Retrieval tools include
  `scoped_answer` and `scoped_list` (the non-kNN filtered listing).

## Work Guidance

- Match each file's surrounding style and comment density.
- Relate points with `apply_pull`, never a new edge structure.
- Keep generated content GENERATOR-sourced; never route generated nodes through
  an `Observation`.

## Verification

`python -m pytest tests/ -q` (the suite covers this package). Targeted:
`tests/test_no_laundering.py`, `test_event_node.py`, `test_event_schema.py`,
`test_kind.py`, `test_spreading.py`, `test_persistence.py`.

## Child DOX Index

No child docs — this is a flat module package.
