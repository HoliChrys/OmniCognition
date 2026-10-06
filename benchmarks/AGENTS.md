# benchmarks — evaluation harnesses

## Purpose

Reproducible evaluation of `metacog` on external benchmarks. Read-only consumer
of the library; never imported by it.

## Ownership

Owns benchmark drivers, debuggers, and result interpretation. The two benchmark
families are owned by their own child docs (below). This parent owns only
cross-benchmark conventions.

## Local Contracts

- Benchmarks require an LLM (`ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN`); they
  are slow and cost tokens. Run them in the background and monitor; never block.
- Report the **two answer modes** (recall for exhaustive retrieval, precision/F1
  for focused answers) and always the set size `n`. Never hard-cap the walk's
  evidence set.
- The event channel is **additive** — measure walk-alone vs walk∪event, never
  event-replaces-walk.
- **Step debuggers (no full benchmark, no LLM):** `debug_chasles_workflow.py`
  steps the SQL-journal workflow (ingest → walk feed → co-retrieval → feedback →
  need-odds → spreading → tags → chasles → refractory → decay-fit → forget →
  forget_explicit → abstain → persist) on SimpleEncoder + a fake LLM, printing
  `[ok]/[FAIL]` and exiting non-zero — a smoke gate. `debug_skill_lifecycle.py`
  steps the tool tier (need → create → organic retrieve → reuse → discriminate →
  crystallize → lifecycle → persist). Both take `--steps a,b` / `--list`.
- **Calibration probe (real encoder, NOT a benchmark):** `calibrate_levers.py`
  sweeps `recency_weight`/`spreading_weight` on curated oblique probes with the
  real MiniLM encoder (`--favorable` = the spreading-bridge regime). Measures
  mechanism efficacy, not a LoCoMo/OBLIQ score.
- **D3 deep-wiki bench (`d3_wiki/`):** protocol, frozen question set and
  harness for `sleep` vs `walk` per context. It runs against COPIES of live
  stores and must never write under a live root (audit hook, run fails); it
  starts only after a successful LLM control call, counts every LLM error,
  caps SUCCESSFUL LLM calls per context (a failed call backs off and retries;
  an LLM that stays down stops the context as `llm_unavailable`) and
  `build(sleep)` time; every stop still writes the summary. Rules in
  `d3_wiki/PROTOCOL.md`; tests in `tests/test_d3_wiki_bench.py`.

## Work Guidance

- Per-question memories (gold + `--bg` distractors) are the unit; do not index
  the full corpus for a single-query test.
- A metric that needs the gold count (e.g. `k=|gold|`) is a reference ceiling,
  not a system capability — label it as such.

## Verification

No automated check; benchmarks are manual experiments. Validate a harness change
by running one small query end-to-end and confirming the table renders.

## Child DOX Index

- `locomo/AGENTS.md` — LoCoMo long-conversation QA harness and answerers
- `obliq_bench/AGENTS.md` — OBLIQ-Bench oblique-query harness and debuggers
- `d3_wiki/PROTOCOL.md` — D3 (Linear TAC-939): `sleep` vs `walk` deep-wiki
  strategies per context. `PROTOCOL.md`, `questions.yaml` (notes pinned by
  sha256) and `run_d3.py` are frozen before any run; never edit them after a
  measurement. The harness only reads live stores (it works on scratch
  copies), refuses a SimpleEncoder fallback, and checks the live sha256 before
  and after. Its scoring and `decide()` are tested in
  `tests/test_d3_wiki_bench.py` (offline).
