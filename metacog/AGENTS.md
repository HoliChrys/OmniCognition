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
  via `scoped_answer(list_only=True)`); `search_nodes` — relevance search
  (semantic | sim | regex | fuzzy) over a filtered pool, on content AND/OR tags,
  WITHOUT REPLACEMENT (`exclude_bag` removes already-collected nodes), the
  agent's collect-loop primitive — semantic mode scores each doc by
  max(cos(doc), cos(its gloss)); `assemble_set` — the ORCHESTRATED loop
  (auto-route the event → loop semantic→sim→fuzzy→regex search_nodes →
  `_judge_relevance` (LLM, recall-first fallback) → collect, without
  replacement, until the pool is exhausted); **tone reading**
  (`tone_reading=True`, opt-in) — irony/hyperbole/echoed-register and the
  author's INTENDED meaning are document properties, read ONCE at ingest
  (`_read_tone` cached never-empty, `_spawn_tone` → `tone:*` tag + `gloss_<id>`
  THOUGHT via apply_pull), so `oblique_labels(per_item="auto")` BATCHES over
  glosses (1 call) instead of per-item judging, and never on derived ids; a
  SECOND-OPINION pass then re-judges only the batch-rejected candidates with the
  live per-item stance THOUGHT (cost ∝ rejects, accepts never overturned). The
  FUNNEL (Phase 3) pre-stage: cos(proposition, doc|stance card|gloss) with an
  EMERGENT threshold (mean+std of the candidate set's own sims) PRE-ACCEPTS the
  clear matches before the batch — never overturned, skipped on sets < 4;
  `stance_of(fact_id)` reads the card's stance THOUGHT.
  The default bag mirrors into `_bag` for backwards compatibility.
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
- `text_index.py` — Phase-2 per-doc token index: raw token sets, stemmed BM25
  docs, postings — memoized at first touch (compute-on-miss, self-healing;
  keyword-keyed memo survives the card's keyword overwrite). Consumed by
  `search_nodes` (content tokens; tags stay live-scanned because they mutate)
  and `bm25_score(text_index=)` via `retrieve_hybrid`. Not pickled; reset in
  `load()`.
- `doc_card.py` — the DOCUMENT CARD (Phase 1 of docs/ingest_index_plan.md):
  ONE structured LLM reading per ingested FACT (keywords, entities, event,
  tone+intended gloss, stance, answerable questions). `Memory(
  doc_card_extractor=...)`: `_spawn_card` dispatches the single reading to the
  existing spawners via their `pre=` hooks and creates `stance_<id>` /
  `dq_<id>_<n>` THOUGHTs (later-phase consumers, derived prefixes). A failed/
  unparseable card is never cached and the individual extractors run as
  fallback (`card_ok` guards).
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
  `scoped_answer`, `scoped_list` (non-kNN filtered listing), `search_nodes`
  (tri-modal relevance step over content+tags, without replacement vs the bag),
  and `assemble_set` (the whole orchestrated loop in one call).

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
