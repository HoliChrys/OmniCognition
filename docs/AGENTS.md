# docs — durable research notes

## Purpose

Durable design and literature material backing the architecture. Reference, not
runtime; nothing here is imported.

## Ownership

Owns long-form research write-ups and diagrams.

## Local Contracts

- `LITERATURE_REVIEW_2026.md` — the literature survey.
- `research_event_schema_rag.md` — the event-schema / events-in-RAG deep report
  (temporal extent, gravitation, schema typology, sub-questions, Graphiti/Zep,
  DyG-RAG) mapped onto MetaCog-Mem primitives.
- `john_walkthrough.md` + `john_diagram.mmd` — a worked end-to-end example.
- `ingest_index_plan.md` — the ACTIVE phased plan for ingest-time indexing
  (Document Card, inverted index, stance cards, doc2query, hybrid kNN cache).
  Update phase status here as phases land; delete when fully shipped.

## Work Guidance

- These are reference documents; keep them coherent with the implemented system.
  When the design changes materially, update the relevant report or delete the
  stale claim — do not let docs contradict code.

## Verification

None (prose). Cross-check claims against `metacog` when editing.

## Child DOX Index

No children.
