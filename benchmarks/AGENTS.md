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
