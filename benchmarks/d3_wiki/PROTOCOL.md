# D3 — deep-wiki strategies, compared per context (Linear TAC-939)

Two ways to grow a context's deep wiki coexist and were never measured against
each other. This protocol, the question set (`questions.yaml`) and the harness
(`run_d3.py`) are frozen **before** any measurement. A run names the commit it
ran on and the sha256 of the question set. Editing either after a run voids
that run.

## The two strategies, and where they live

The deep-wiki cycle the ticket attributes to mnema (`feed_wiki`, `wiki_doc`,
sleep → seeds → rerun → absorb) exists **only** in this repository, as the
"mnema layer" of `metacog/memory.py`. TAC-327 checked the deployed mnema
package (editable install, source `3593a93`) and found no `sleep`, wiki or
seed function; its only consolidation is `reflect.py` (`run_reflection`). Both
strategies therefore run in one omni process.

| column | build | answer (MCP tool, surface `external` as deployed) |
|---|---|---|
| `walk` | notes ingested by the gate's own pass (`ContextualMemory._refresh_notes`: doc via `import_okf` + content point `notes:<d>#<sha12>`) | `walk_start(query)`; items = `relevant_collected` (never hard-capped) |
| `sleep` | same ingest, then one seed per note (`add_seed(doc, <first heading>, target="*")`), then `Memory.sleep()` (`reconcile_wiki` → `rerun_seeds` → absorb / pending) | `recall(query, k=5)` (retrieve + relevance floor + gap sentinel) |
| `deployed` (reference, **not** a candidate) | same ingest as `walk` | `recall(query, k=5)`: what tachikoma's `OmniEngine.recall` calls today |

## Corpus: common by construction

- Each strategy runs on **its own scratch copy** of the context's live store
  (`memory.pkl` + `memory.pkl.journal.db`), because `sleep()` mutates the
  store. Into that copy go the **same** notes, copied into the scratch notes
  root at the folder `notes_folder()` maps the context to.
- The notes are pinned by sha256 in `questions.yaml`. The harness refuses a
  corpus that differs (fail-closed).
- The live store is only read: any write the bench attempts under a live root
  is refused and fails the run (see the amendment below; the sha256 before and
  after are informational).
- `global`: the folder omni reads for it (`<root>/notes` → `contexts/tachikoma/notes`)
  is **empty** (TAC-327). The bench corpus is `contexts/global/notes` (80 `.md`).
  omni does not read that folder in production. This is a bench choice, not a
  production change. The mapping gap is reported separately.
- `tachikoma.paralelle.GenAI`: `contexts/tachikoma.paralelle.GenAI/notes` (16 `.md`),
  the folder omni reads.

## Measures (per column × context)

A returned item **cites** note `d` when its id is `notes:<d>#…` or it carries
the tag `note:notes:<d>`. An item that resolves to no note is **uncited**: a
fact of the live store or a generated node.

- **Coverage** = in-topic questions with at least one item citing an
  `expected` note ÷ in-topic questions.
- **Noise** = in-topic answers holding at least one item that cites a note
  **outside** `relevant` ÷ in-topic answers. Uncited items are not judged:
  the store's facts have no gold. They are reported (`items_uncited`), along
  with the precision over cited items and `n`.
- **Off-topic** = items returned (gap notices excluded) summed over the
  off-topic questions, asked on every context. It must be 0. If a point of the
  copied store already matches the question's `absent_marker`, the question is
  reported `invalid_offtopic` and not scored: the store does hold the topic.
- **Build cost** = wall time and LLM calls, counted on `Memory.llm`, failures
  included:
  - `build_empty`: from an empty store with the notes only;
  - `build_live`: on the copy of the live store. This includes the load time.
- **Query cost** = wall time of each answer call, p50 / p95 / max (nearest
  rank) over all questions, in-topic and off-topic. One uncounted warm-up runs
  first. The LLM calls and errors are counted per answer. An LLM error is a
  degraded answer and is reported, never hidden.

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

`decide()` in `run_d3.py` is this rule. It is unit-tested at its boundaries in
`tests/test_d3_wiki_bench.py`. The kept strategy is written as a
**per-context configuration entry** that the API can read. It is never a
global variable.

## Amendment — Proxy, 2026-10-05 (TAC-209), before the rerun

The first run (`c852ba2`) is void: GenAI was measured without an LLM (the
venv had no `anthropic`; `generate` turned every failure into `""` without
counting it), and `global` failed twice on the live-store hash, which the
served gate legitimately changes while the bench runs. The rules, the
measures and the question set above are unchanged. The guard rails become:

- **Live store.** The run fails if the **bench** writes under a live root:
  the stores' root (parent of `store_source`) or `notes_source`. An audit
  hook refuses and records every such attempt (`open` in a write mode or with
  `O_WRONLY`/`O_RDWR`/`O_CREAT`/`O_TRUNC`/`O_APPEND`, `sqlite3.connect`
  without `mode=ro`, path-mutating `os`/`shutil` events). The sha256 before
  and after are kept for information (`live_store_changed_by_other_process`)
  and no longer fail the run. The gate keeps serving: no write window is cut.
- **LLM.** One control call before anything else; if it fails or answers
  nothing, the bench does not start. Every client failure is counted
  (`ClaudeLLM.llm_errors`, per answer and per context). The bench runs in its
  own venv (or a `--target` overlay on `PYTHONPATH`), never the served one.
  Credentials: the host's LiteLLM gateway if it serves the model omni asks
  (`CLAUDE_MODEL`), else the operator's `ANTHROPIC_*`.
- **LLM cap.** At most 1 000 LLM calls per context (`--llm-cap`). Beyond, the
  call is refused, the context stops, `stopped` says where, and a strategy
  whose answers did not all come back is not measured.
- **Build cap.** `build(sleep)` is capped at 90 min (`--sleep-build-cap-s`),
  one attempt per context. Beyond, sleep is **ineligible on that context for
  build cost** — a result, not a failure — and the rule applies among the
  strategies that finished (`decide(scores, ineligible)`).
