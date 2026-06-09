# benchmarks/obliq_bench — OBLIQ-Bench oblique queries

## Purpose

Evaluate `metacog` on OBLIQ-Bench: oblique queries whose relevance is LATENT
(no surface term matches), a **recall** benchmark. The regime that exercises the
event subsystem and the answer-space machinery.

## Ownership

Owns the OBLIQ per-question builders, step/tree debuggers, and the walk-vs-event
batch.

## Local Contracts

- `debug_qa.py` — per-question memory builder (`build_obliq_memory`) + the reused
  `QADebugger` REPL. `qa.py` is an alias.
- `event_step.py` — single-query step debugger: baseline retrieve, baseline WALK,
  events spawned, consolidate, centroid route, event_search, and the action-bridge
  ablation ([1b] clean walk → [7] walk after event_search → [8] walk after the
  beacon; Δ(8-7) isolates the bridge). `build` and `_walk` are reused by the batch.
- `tree_nav.py` — tree-navigable parallel-query debugger: ROOT → query →
  {retrieval | context} → [event:type] → event → slot, with `cd/up/ls/tree/
  recall/schema`. Expensive branches expand lazily.
- `batch_walk_vs_event.py` — walk-alone vs walk+event across queries; reports
  recall AND F1 AND set size `n`; the walk set is `_relevant_cum`
  (uncertainty-stopped, uncapped).

## Work Guidance

- OBLIQ scores recall, but recall is only honest at a controlled/emergent `n` —
  always report `n` so over-selection is visible.
- Never hard-cap the walk's evidence; `n_stages` is a generous ceiling only,
  `step().done` (uncertainty) is the real terminator.

## Verification

Manual. Established empirical pattern (descriptive/twitter): the walk dominates
the event channel on small/medium gold (recall AND F1); on **large exhaustive
gold sets** (e.g. q0395 gold=58) the walk's uncertainty stop caps its recall
(~0.56) while the **event cluster wins recall** (~0.74) — the regime that earns
the event branch. The walk wins F1 (focused answer) throughout.

## Child DOX Index

No children.
