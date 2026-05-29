# OmniCognition / MetaCog-Mem

A metacognitive memory architecture for LLM agents, built without
hyperparameters and with strict epistemic typing.

## Core ideas

- **Manifold over graph.** No materialized edges. Relations between
  memory items emerge from k-nearest-neighbors over the current
  effective embedding.

- **Three kinds of points share one DB.** `FACT` / `THOUGHT` /
  `ACTION`. All three have the same schema, the same A(·) function,
  the same lifecycle.

- **Anti-laundering, by construction.** Following Romanchuk &
  Bondar's "Semantic Laundering" (arXiv:2601.08333), the system
  enforces typed sources : `Observation(source=GENERATOR)` raises
  `LaunderingError`. LLM outputs become *content* (∈ P) but never
  *evidence*.

- **No hyperparameters.** Every "threshold" is either a math
  constant (cos π/4, cos π/12) or emerges from data (median ± σ).

- **Zero deletion.** Latent states (`INVALID`, `DEPRECATED`) are
  exiled geometrically, not removed. Resurrection is automatic if
  new observations re-corroborate them.

- **Retrospective compression.** Near-straight reasoning trajectories
  get Chasles-compressed : the intermediates collapse into one
  child positioned between the boundaries.

- **Pluralisme épistémique.** Observators give each source its own
  parallel view on every point. Three creation paths : explicit,
  auto-spawn by polarity, auto-spawn by semantic clustering.

## Quick start (in-process)

```python
from metacog import Memory

m = Memory(storage_path="memory.pkl")    # or omit for in-memory

m.ingest("dr sarah lives in berkeley", kind="FACT")
m.ingest("dr sarah works as a counselor", kind="FACT")

m.process_turn("who is dr sarah", speaker="user")
m.process_turn(
    "dr sarah is a counselor in berkeley",
    speaker="assistant",
    retrieved_point_ids=[p.id for p in m.points[:2]],
)

result = m.reason("where does dr sarah live?")
print(result["final_answer"])
print(m.audit())   # zero laundering
```

## MCP server

The same operations are exposed as an MCP service. Tools advertised :

```
ingest              add a FACT / THOUGHT / ACTION
observe             apply an Observation on existing point(s)
process_turn        record a conversation turn (detectors fire)
retrieve            top-k semantic retrieval
reason              full reasoning trajectory until output convenable
sleep               run a collision sleep cycle
inspect             dump a point's full state
audit               verify no laundering
stats               system overview
declare_observator  manually declare an observator
detect_polarized    list polarized points
spawn_observators   auto-spawn from a polarized point
route               pick top-k observators for a query
save                persist to disk
```

### Run the server

```bash
uv sync                                  # or pip install -e .
uv run metacog-mcp --storage ~/.metacog/state.pkl
```

The server listens on stdio (the standard transport for Claude Code).

### Register in Claude Code

Add to your Claude Code MCP configuration :

```json
{
  "mcpServers": {
    "metacog": {
      "command": "metacog-mcp",
      "args": ["--storage", "~/.metacog/state.pkl"]
    }
  }
}
```

After restarting Claude Code, the tools appear as `mcp__metacog__*`
in any session.

## Tests

```bash
uv run pytest tests/ -v
# or
PYTHONPATH=. pytest tests/ -v
```

126+ tests covering laundering invariants, manifold dynamics,
collision and compression, signal detection, ACTION execution,
reasoning trajectories, observator multi-perspective, the high-level
Memory wrapper, and one end-to-end integration scenario.

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
| `metacog/observator.py` | Observators, polarization, delegation |
| `metacog/audit.py` | non-laundering verification |
| `metacog/memory.py` | top-level `Memory` wrapper |
| `metacog/defaults.py` | `SimpleEncoder`, `NoOpExecutor` |
| `metacog/llm.py`      | `ClaudeLLM` (Haiku, lazy) — the only LLM, no stub fallback |
| `metacog/mcp_server.py` | MCP service |

## References

- Romanchuk & Bondar, "Semantic Laundering in AI Agent Architectures",
  arXiv:2601.08333.
- Zhu et al., "HeLa-Mem", arXiv:2604.16839 (we keep the spirit,
  remove the hyperparameters and the materialized edges).
- Ramsauer et al. 2020, "Hopfield Networks is All You Need".
