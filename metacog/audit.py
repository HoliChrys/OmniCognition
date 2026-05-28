"""
MetaCog-Mem — audit / non-laundering verification.

Provides invariants checkable at any point: every confidence update
across the system must trace back to a non-GENERATOR observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from metacog.epistemic import Node, SourceClass


class NoLaunderingInvariant(AssertionError):
    """Raised when the audit log contains a forbidden source class."""


@dataclass(frozen=True)
class AuditReport:
    total_updates: int
    by_source: dict[SourceClass, int]
    violations: list[str]

    @property
    def ok(self) -> bool:
        return not self.violations


def audit(nodes: Iterable[Node]) -> AuditReport:
    """Walk every audit entry across every node, count by source class,
    flag any GENERATOR sighting."""
    counts: dict[SourceClass, int] = {s: 0 for s in SourceClass}
    violations: list[str] = []
    total = 0
    for node in nodes:
        for entry in node.update_log:
            total += 1
            src = entry.observation.source
            counts[src] = counts.get(src, 0) + 1
            if src == SourceClass.GENERATOR:
                violations.append(
                    f"LAUNDERING: node={node.id} signal={entry.observation.signal_type}"
                )
    return AuditReport(total_updates=total, by_source=counts, violations=violations)


def assert_no_laundering(nodes: Iterable[Node]) -> AuditReport:
    """Audit + raise if any violation."""
    report = audit(nodes)
    if not report.ok:
        raise NoLaunderingInvariant(
            f"{len(report.violations)} laundering violation(s):\n"
            + "\n".join(report.violations)
        )
    return report


def inputs_of_A(node: Node) -> dict[str, object]:
    """Return the literal inputs that A(·) reads for a given node.

    Useful for static inspection — verifies that no field of P-type
    (e.g. `content`) appears in the input set.
    """
    return {
        "n_corrob": node.n_corrob,
        "n_contra": node.n_contra,
        "n_uses": node.n_uses,
        # confidence and uncertainty are pure functions of the counters above.
    }
