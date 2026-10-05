# D3 — deep-wiki strategies, compared per context (Linear TAC-939)

Two ways to grow a context's deep wiki coexist and were never measured against
each other. This protocol is frozen **before** any measurement: the question
set and this file are versioned, and a run names the commit it ran on.

## The two strategies — where they live

The deep-wiki cycle the ticket attributes to mnema (`feed_wiki`, `wiki_doc`,
sleep → seeds → rerun → absorb) is the "mnema layer" of this repository
(`metacog/memory.py`). The mnema *package* exposes no wiki tool over MCP:
TAC-218 found no `wiki_*` tool, and tachikoma's `memory_engines.py` marks
`MnemaEngine.list_docs/read_doc` unsupported. If the deployed package turns out
to carry a sleep cycle of its own, it is measured as a third column, under
TAC-321's bench-window rules.

| strategy | build (offline) | answer (query time) |
|---|---|---|
| `sleep` (mnema layer) | notes → `import_okf` + content point; `add_seed`; `Memory.sleep()` → `reconcile_wiki` → `rerun_seeds` (re-run, diff, absorb into generated docs / pending on authored) | `retrieve(q, k=5)`, cited through the wiki docs of the hits |
| `walk` (omni) | notes → `import_okf` + content point (D2 `_ingest_note`) | the uncertainty-stopped walk (`meta_walk`), cited through its committed evidence set (never hard-capped) |

**Shared corpus by construction.** Both strategies run in one process, on
the same scratch copy of one context's notes and store. Each note is read from
`notes_folder(OMNI_NOTES_ROOT, ctx)` and pinned by its sha256 in the run's
manifest. A run never opens the live `memory.pkl` or `*.journal.db` of
`global` or `tachikoma.paralelle.GenAI` for writing. It works on a copy under
a scratch root.

## Contexts

- `global`: the heavy context. Its notes folder is `<notes_root>/notes`.
- `tachikoma.paralelle.GenAI`: the light context. Its notes folder is
  `<notes_root>/paralelle/GenAI/notes`.

## Question set — `questions.yaml`

- **In-topic:** each question has a known answer located in one note. It
  carries `expected` (the note doc ids that answer it) and `relevant` (expected
  plus any note that legitimately bears on it). Questions are written as
  paraphrases, never as the note's title or a copied sentence.
- **Off-topic:** at least one per context. The right answer is "I have
  nothing". It is answered against every context.
- Once a measurement has run, the file is not edited. A new question set is a
  new file and a new run.

## Measures (per strategy × context)

A returned item **cites** note `d` when its id is `notes:<d>#…` or it carries
the tag `note:<d>`. An item that resolves to no note cites nothing.

- **Coverage** = in-topic questions with at least one returned item citing an
  `expected` note ÷ in-topic questions.
- **Noise** = in-topic answers holding at least one item that cites nothing, or
  cites a note outside `relevant` ÷ in-topic answers. Reported with the
  item-level precision and `n`.
- **Off-topic recall** = items returned (gap notices excluded) summed over the
  off-topic questions. It must be 0.
- **Build cost** = wall time and LLM calls (counted on the `Memory.llm` client)
  from an empty scratch store to a built wiki.
- **Query cost** = wall time of the answer call, p50 / p95 / max over all
  questions, after one uncounted warm-up query. Measured on the same host,
  in the same window, for both strategies.

## Decision rule (Proxy, 2026-10-04 — TAC-190 decisions doc, applied as is)

Per context, keep the strategy with the best coverage among those that meet
both conditions:

1. noise ≤ 5 % of answers **and** 0 items on the off-topic questions;
2. query cost p95 ≤ 2 s (the E1 budget).

- If no strategy meets both conditions on a context → `walk` (omni), and the
  breach is written down with its numbers.
- A tie, or a coverage gap under 10 points → `walk` (omni), the active engine.
- If `sleep` wins on a context, it stays a per-context option for that
  context. If it wins nowhere, it is retired, and Linear TAC-34 / TAC-38 are
  closed with the reason.

The kept strategy is written as a **per-context configuration entry** that
the API can read. It is never a global variable.
