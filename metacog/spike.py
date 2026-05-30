"""
Spike-and-Path Chasles compression — emergent auto-compression.

The walker accumulates "spikes" on nodes traversed during multi-hop
reasoning. When a chain of nodes shows BOTH per-node spike significance
(above the Poisson noise floor) AND concentrated ref distributions
(modal ref dominates), the chain becomes a candidate Chasles path :
start -> intermediates (>= 2) -> end. The existing resolve_collision
(metacog/collision.py) compresses the intermediates into a single child
geometrically anchored between start and end, creating a shortcut in
the manifold.

All thresholds are emergent / parameter-free :

  spike_threshold(memory) = lambda + sqrt(lambda)     where lambda = M/N
    -- 1 sigma above the Poisson baseline ; scales with corpus size
       and total recorded hops so 2 spikes over 10 nodes does NOT have
       the same significance as 2 spikes over 1000 nodes.

  concentrated(spikes)    = max(spikes) > 2 * sum/k
    -- modal ref must be at least 2x its uniform share to make the
       chain followable. Uniform distributions abort the path.

After a successful Chasles compression, all nodes of the fired path are
reset to n_spike = 0 (refractory period) so the same structure cannot
cascade into another firing without re-accumulating evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from metacog.epistemic import Point, SourceClass

if TYPE_CHECKING:
    from metacog.collision import CollisionEvent
    from metacog.memory import Memory


# All computations here are pure COMPUTATION (no LLM call inside this
# module ; the LLM is only invoked via resolve_collision called by
# auto_compress_chasles, which is itself Cor.5 compliant).
SOURCE: SourceClass = SourceClass.COMPUTATION


# ---------------------------------------------------------------------------
# 1) Activation — record a single multi-hop transition
# ---------------------------------------------------------------------------


def record_hop(prev: Point, curr: Point, memory: "Memory") -> None:
    """Register one multi-hop : `prev -> curr`.

    Increments `curr.n_spike` (the target accumulates activation energy),
    adds a `ref:<curr.id>` tag on `prev` (deduplicated by Point.add_tag),
    and bumps the global counter on `memory` used to calibrate the
    Poisson baseline of `spike_threshold`. No-op when either argument is
    None or when prev is curr (self-loop)."""
    if prev is None or curr is None or prev is curr:
        return
    curr.n_spike += 1
    # Bypass Point.add_tag — it lowercases tag values, which would
    # mangle case-sensitive ids ("ref:D1:7" -> "ref:d1:7" -> lookup miss).
    # Refs are a structural protocol, not a semantic tag, so we append
    # the raw "ref:<id>" with dedup directly.
    tag = f"ref:{curr.id}"
    if tag not in prev.tags:
        prev.tags.append(tag)
    memory._spike_total_hops += 1


# ---------------------------------------------------------------------------
# 2) Spike threshold — Poisson 1-sigma above noise floor
# ---------------------------------------------------------------------------


def spike_threshold(memory: "Memory") -> float:
    """Emergent threshold ; a node spikes when n_spike > this.

    Builds on the Poisson model : if hops were uniformly random over
    the N points of the memory and we recorded M of them, each point
    would expect lambda = M/N hits. A node "spikes" iff its count is
    one standard deviation (sqrt(lambda)) above that baseline. No
    hyperparameter — fully calibrated by the population state.

    Returns 0.0 when N < 4 (corpus too small for the Poisson model to
    be meaningful — every nonzero count is treated as a spike)."""
    n = len(memory.points)
    if n < 4:
        return 0.0
    m = memory._spike_total_hops
    lam = m / n
    return lam + lam ** 0.5


def is_spiking(point: Point, memory: "Memory") -> bool:
    """True when the point's spike count is statistically significant."""
    return point.n_spike > spike_threshold(memory)


# ---------------------------------------------------------------------------
# 3) Concentration — modal stands out from a uniform distribution
# ---------------------------------------------------------------------------


def concentrated(spikes: List[int]) -> bool:
    """True when the modal value clearly dominates the others.

    Rule : max(spikes) > 2 * (sum/k). Under uniform distribution every
    entry equals sum/k, so the modal can never exceed twice that share
    — it would require concentrating energy on one entry. Conservative
    for small k (the user-confirmed semantics : with only a few refs a
    weak modal isn't worth following)."""
    if not spikes:
        return False
    total = sum(spikes)
    if total == 0:
        return False
    k = len(spikes)
    if k == 1:
        return True
    return max(spikes) > 2 * (total / k)


# ---------------------------------------------------------------------------
# 4) Chasles path — follow the modal as long as the conditions hold
# ---------------------------------------------------------------------------


def chasles_path(start: Point, memory: "Memory",
                 max_len: int = 8) -> List[Point]:
    """Walk from `start` following the modal ref at each step.

    Stops when : the current node is no longer spiking, has no refs,
    the refs distribution is uniform, the modal target is not spiking,
    a cycle is reached, or max_len is hit. Returns the path traversed
    (length 1 = no successor, length >= 4 = valid Chasles candidate)."""
    if start is None:
        return []
    path: List[Point] = [start]
    seen = {start.id}
    by_id = {p.id: p for p in memory.points}
    while len(path) < max_len:
        curr = path[-1]
        if not is_spiking(curr, memory):
            break
        ref_ids = [t[4:] for t in curr.tags if t.startswith("ref:")]
        nodes = [by_id[r] for r in ref_ids if r in by_id]
        if not nodes:
            break
        spikes = [n.n_spike for n in nodes]
        if not concentrated(spikes):
            break
        next_node = nodes[spikes.index(max(spikes))]
        if next_node.id in seen:                         # cycle protection
            break
        if not is_spiking(next_node, memory):
            break
        path.append(next_node)
        seen.add(next_node.id)
    return path


# ---------------------------------------------------------------------------
# 5) Auto-compression — gate >= 4 nodes, reset spikes only on firing
# ---------------------------------------------------------------------------


def auto_compress_chasles(memory: "Memory", llm: Any,
                          encoder: Any) -> List["CollisionEvent"]:
    """Detect spike-driven Chasles paths and compress them.

    For each currently spiking node, walk a Chasles path ; if the path
    has at least 4 SAME-KIND nodes (1 start + >= 2 intermediates + 1 end,
    matching resolve_collision's contract), call resolve_collision on
    the intermediates with start / end as anchors. On a successful
    compression, reset n_spike to 0 on every node of the fired path
    (refractory period — prevents the same structure from firing again
    until new activations accumulate). Paths shorter than 4 are dropped
    silently WITHOUT resetting any counter — the nodes stay charged
    and may form a valid path on the next compress_chasles call.

    Returns the list of CollisionEvent records (one per firing)."""
    from metacog.collision import resolve_collision

    t_now = memory._now() if hasattr(memory, "_now") else 0.0
    events: List["CollisionEvent"] = []
    fired: set = set()
    candidates = [p for p in memory.points if is_spiking(p, memory)]
    for start in candidates:
        if start.id in fired:
            continue
        path = chasles_path(start, memory)
        if len(path) < 4:
            continue                                     # cancel, no reset
        if len({p.kind for p in path}) != 1:
            continue                                     # cross-kind paths reject
        intermediates = path[1:-1]                       # guaranteed >= 2
        anchors = [path[0], path[-1]]
        result = resolve_collision(
            intermediates=intermediates, anchors=anchors,
            llm=llm, encoder=encoder, t_now=t_now,
        )
        if result is None:
            continue
        memory.points.append(result.child)
        for node in path:                                # refractory reset
            node.n_spike = 0
            fired.add(node.id)
        events.append(result.event)
    return events
