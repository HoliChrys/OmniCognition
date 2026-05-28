"""
Test harness for non-laundering.

Each test corresponds to a property derived from Romanchuk & Bondar 2026:

  T1  OBSERVER-sourced signals update confidence.
  T2  COMPUTATION-sourced signals update confidence.
  T3  GENERATOR-sourced signals are REJECTED at construction (Cor. 5).
  T4  Every update produces an audit entry.
  T5  Global invariant: no GENERATOR appears in any audit log.
  T6  A(·) ⊥ P : same counters ⇒ same state, regardless of content.
  T7  Contradictions trigger SUSPECT then INVALID.
  T8  inputs_of_A contains no fields from P.
"""

from __future__ import annotations

import random

import pytest

from metacog import (
    AuditEntry,
    EpistemicState,
    LaunderingError,
    Node,
    NoLaunderingInvariant,
    Observation,
    SourceClass,
    apply_observation,
    assert_no_laundering,
    assign_status,
    inputs_of_A,
)


# ---------- T1, T2 ---------------------------------------------------------

def test_observer_signal_accepted():
    node = Node(id="n1", content="Dr. Sarah at Berkeley")
    obs = Observation(
        source=SourceClass.OBSERVER,
        signal_type="corroboration",
        polarity=+0.8,
        target_node_ids=["n1"],
        timestamp=1.0,
        raw_content="user: yes, Berkeley exactly",
    )
    entry = apply_observation(node, obs, population=[node])
    assert node.n_corrob == 1
    assert entry.confidence_after > entry.confidence_before


def test_computation_signal_accepted():
    node = Node(id="n1", content="meeting tomorrow")
    obs = Observation(
        source=SourceClass.COMPUTATION,
        signal_type="reformulation",
        polarity=-0.3,
        target_node_ids=["n1"],
        timestamp=1.0,
        raw_content="cosine(q_t, q_t+1)=0.82",
    )
    apply_observation(node, obs, population=[node])
    assert node.n_contra == 1


# ---------- T3 -------------------------------------------------------------

def test_generator_signal_rejected_at_construction():
    with pytest.raises(LaunderingError) as exc_info:
        Observation(
            source=SourceClass.GENERATOR,
            signal_type="llm_judgment",
            polarity=+0.9,
            target_node_ids=["n1"],
            timestamp=1.0,
            raw_content="LLM-as-judge says this answer is good",
        )
    msg = str(exc_info.value)
    assert "Corollary 5" in msg or "GENERATOR" in msg


# ---------- T4 -------------------------------------------------------------

def test_audit_trail_is_complete():
    node = Node(id="n1", content="...")
    for i in range(5):
        obs = Observation(
            source=SourceClass.OBSERVER,
            signal_type="corroboration",
            polarity=+0.5,
            target_node_ids=["n1"],
            timestamp=float(i),
        )
        apply_observation(node, obs, population=[node])
    assert len(node.update_log) == 5
    for entry in node.update_log:
        assert isinstance(entry, AuditEntry)
        assert entry.observation.source in {SourceClass.OBSERVER, SourceClass.COMPUTATION}


# ---------- T5 -------------------------------------------------------------

def test_global_no_laundering_invariant():
    nodes = [Node(id=f"n{i}", content=f"...{i}") for i in range(10)]
    rng = random.Random(42)
    for _ in range(50):
        n = rng.choice(nodes)
        obs = Observation(
            source=rng.choice([SourceClass.OBSERVER, SourceClass.COMPUTATION]),
            signal_type=rng.choice(["reformulation", "corroboration", "contradiction"]),
            polarity=rng.uniform(-1, 1),
            target_node_ids=[n.id],
            timestamp=rng.random(),
        )
        apply_observation(n, obs, population=nodes)

    report = assert_no_laundering(nodes)
    assert report.ok
    assert report.by_source.get(SourceClass.GENERATOR, 0) == 0
    assert report.total_updates == 50


def test_invariant_detects_injection():
    """If someone bypasses Observation() and forges a GENERATOR audit entry
    directly, the audit must still catch it."""
    node = Node(id="n1", content="...")
    obs = Observation(
        source=SourceClass.OBSERVER,
        signal_type="corroboration",
        polarity=+0.5,
        target_node_ids=["n1"],
        timestamp=0.0,
    )
    apply_observation(node, obs, population=[node])

    # Forge a GENERATOR-sourced entry by mutating the audit log directly.
    object.__setattr__(node.update_log[0].observation, "source", SourceClass.GENERATOR)

    with pytest.raises(NoLaunderingInvariant) as exc_info:
        assert_no_laundering([node])
    assert "LAUNDERING" in str(exc_info.value)


# ---------- T6 — A(·) ⊥ P  -------------------------------------------------

def test_A_independent_of_P_two_nodes():
    """Same counters, completely different `content` (∈ P) → same state."""
    a = Node(id="a", content="Dr. Sarah was at Berkeley", n_corrob=3, n_contra=1)
    b = Node(id="b", content="The capital of Mongolia is Ulaanbaatar", n_corrob=3, n_contra=1)
    pop = [a, b]
    assert assign_status(a, pop) == assign_status(b, pop)


def test_A_independent_of_P_random():
    """Across many random counter configurations, swapping `content` never
    changes A(·)."""
    rng = random.Random(7)
    for _ in range(100):
        nc, nx = rng.randint(0, 12), rng.randint(0, 12)
        a = Node(id="a", content="X" * rng.randint(1, 50), n_corrob=nc, n_contra=nx)
        b = Node(id="b", content="Y" * rng.randint(1, 50), n_corrob=nc, n_contra=nx)
        pop = [a, b] + [
            Node(id=f"p{i}", content="filler",
                 n_corrob=rng.randint(0, 5),
                 n_contra=rng.randint(0, 5))
            for i in range(8)
        ]
        assert assign_status(a, pop) == assign_status(b, pop)


# ---------- T7 -------------------------------------------------------------

def test_contradiction_path_to_invalid():
    node = Node(id="n1", content="...")
    pop = [node]
    for i in range(3):
        obs = Observation(
            source=SourceClass.OBSERVER,
            signal_type="contradiction",
            polarity=-1.0,
            target_node_ids=["n1"],
            timestamp=float(i),
        )
        apply_observation(node, obs, population=pop)
    assert node.state == EpistemicState.INVALID


def test_corroboration_path_to_warranted_or_corroborated():
    nodes = [Node(id=f"n{i}", content=f"...{i}") for i in range(6)]
    target = nodes[0]
    for i in range(8):
        obs = Observation(
            source=SourceClass.OBSERVER,
            signal_type="corroboration",
            polarity=+1.0,
            target_node_ids=[target.id],
            timestamp=float(i),
        )
        apply_observation(target, obs, population=nodes)
    assert target.state in {EpistemicState.WARRANTED, EpistemicState.CORROBORATED}
    assert target.n_corrob == 8
    assert target.n_contra == 0


# ---------- T8 — inputs_of_A inspection -----------------------------------

def test_inputs_of_A_contains_no_P_fields():
    node = Node(id="n1", content="this string is in P", n_corrob=4, n_contra=1)
    inputs = inputs_of_A(node)
    assert "content" not in inputs
    assert "embedding" not in inputs
    assert "raw_content" not in inputs
    for v in inputs.values():
        assert isinstance(v, int), f"non-integer input found: {v!r}"
