# benchmarks/locomo — LoCoMo long-conversation QA

## Purpose

Evaluate `metacog` on LoCoMo (multi-session conversational QA). The headline
result: the walk raises evidence agent-recall from a static ~0.31 (Recall@7) to
≈0.85.

## Ownership

Owns the LoCoMo drivers, answerers, encoders, and the QA debug REPL. `data/`
holds the LoCoMo conversations.

## Local Contracts

- `eval.py` — main evaluation entrypoint (`--answerer meta`, `--samples`,
  `--max-qa`, `--debug-jsonl`).
- `mcp_meta_agent.py` — the agentic meta-walk answerer. `_extract_fact_ids`
  unions walk + event-channel ids (`fact_ids_cumulative`, `cluster_ids`,
  `context_ids`, `fact_ids`) — ADDITIVE. The parallel bag renders a list answer
  at the end only when non-empty; the rest of the process is unchanged.
- `debug_qa.py` — the `QADebugger` REPL (presearch / clues / walk / auto /
  recall / step). Reused verbatim by the OBLIQ debugger.
- `encoders.py` — `SemanticEncoder` (sentence-transformers).
- `claude_react.py` / `react_qa.py` / `mcp_agent.py` — baseline answerers.

## Work Guidance

- Entity/event ids (`entity_*`, `event_*`, `act_*`) never equal a LoCoMo
  `dia_id`, so they are never counted as gold evidence — keep that property when
  adding beacon kinds.

## Verification

Manual. `python -m benchmarks.locomo.eval --answerer meta --samples 3 --max-qa 10`.

## Child DOX Index

No children.
