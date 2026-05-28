# OmniCognition / MetaCog-Mem

A metacognitive memory architecture for LLM agents, built without
hyperparameters and with strict epistemic typing.

## Core ideas

- **Manifold over graph.** No materialized edges. Relations between
  memory items emerge from k-nearest-neighbors over the current
  effective embedding. The same manifold supports many graph views
  on demand.

- **Three kinds of points share one DB.** `FACT` / `THOUGHT` /
  `ACTION`. All three have the same schema, the same A(·) function,
  the same lifecycle. ACTIONs replace the old "skill" concept :
  they are contextualized interventions stored in the manifold,
  not external functions in a registry.

- **Anti-laundering, by construction.** Following Romanchuk &
  Bondar's "Semantic Laundering" (arXiv:2601.08333), the system
  enforces typed sources : an `Observation` constructed with
  `source=GENERATOR` raises `LaunderingError`. All counter updates
  trace back to `OBSERVER` (world events) or `COMPUTATION`
  (deterministic transforms). LLMs produce *content* (which lives
  in P, the proposition space) but never *evidence*.

- **No hyperparameters.** Every "threshold" is either a math
  constant (cos π/4, cos π/12) or emerges from the data
  distribution (median ± σ). Step sizes derive from observation
  counts : `1 / (1 + n_obs)`.

- **Zero deletion.** Points never leave the cloud. Latent states
  (`INVALID`, `DEPRECATED`) are exiled geometrically via
  `apply_exile` — pushed away from the active centroid so they
  don't surface in typical kNN queries. They remain available for
  audit and for resurrection if new observations re-corroborate
  them.

- **Retrospective compression.** When a reasoning trajectory turns
  out to be near-straight (Chasles-like), the intermediate points
  collapse into a single child positioned between the trajectory
  boundaries. The meta-graph self-optimizes through use.

## Module layout

| Module | Responsibility |
|---|---|
| `metacog/epistemic.py` | `Point`, `Observation`, `A(·)`, state machine, `PointKind` |
| `metacog/geometry.py` | manifold ops, `apply_pull`, `apply_exile`, `retrieve` |
| `metacog/collision.py` | proximity collision, N-way fission, Chasles anchors |
| `metacog/compression.py` | retrospective Chasles compression |
| `metacog/detectors.py` | 5 deterministic conversation signals |
| `metacog/execution.py` | ACTION execution + result FACT lineage |
| `metacog/reasoning.py` | `reason()` orchestrator |
| `metacog/audit.py` | non-laundering verification |

## Pipeline

```
        query
          │
          ▼
   ┌────────────────────────┐
   │  reason()              │  greedy traversal until output convenable
   └─────────┬──────────────┘
             │
   greedy kNN over effective embeddings
             │
             ▼
   ┌─────────────────────────┐
   │  visite FACT/THOUGHT    │
   │  n_uses += 1            │
   └─────────┬───────────────┘
             │
   synthesize_step (LLM = GENERATOR content)
             │
             ▼
   ┌─────────────────────────┐
   │  nouveau THOUGHT        │
   │  (parent = visited)     │
   └─────────┬───────────────┘
             │
   propose_action ? oui
             │
             ▼
   ┌─────────────────────────┐
   │  nouvelle ACTION        │
   │  (parent = THOUGHT)     │
   └─────────┬───────────────┘
             │
   execute_action via ExecutorProtocol (OBSERVER)
             │
             ▼
   ┌─────────────────────────┐
   │  result FACT            │
   │  (parent = ACTION)      │
   └─────────┬───────────────┘
             │
   apply_observation → counters + state + exile
             │
             ↺ next iteration with answer_emb as anchor
             │
   stability reached (cos > cos π/12)
             │
             ▼
   compress_trajectory (Chasles eager)
             │
             ▼
   final answer + optimized meta-graph
```

## Tests

```
PYTHONPATH=. pytest tests/ -v
```

The suite covers laundering invariants, manifold dynamics, collision
resolution, Chasles compression, signal detection, ACTION execution,
reasoning trajectories, and one end-to-end integration scenario.

## References

- Romanchuk & Bondar, "Semantic Laundering in AI Agent
  Architectures", arXiv:2601.08333.
- Zhu et al., "HeLa-Mem: Hebbian Learning and Associative Memory
  for LLM Agents", arXiv:2604.16839 (we keep the spirit, remove the
  hyperparameters and the materialized edges).
- Ramsauer et al. 2020, "Hopfield Networks is All You Need" — the
  Hopfield-attention equivalence that justifies dropping
  materialized edges.
