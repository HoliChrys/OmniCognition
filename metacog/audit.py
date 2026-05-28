"""
MetaCog-Mem — audit / non-laundering verification.

Provides invariants checkable at any point: every confidence update
across the system must trace back to a non-GENERATOR observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from metacog.epistemic import Point, SourceClass


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


def audit(points: Iterable[Point]) -> AuditReport:
    """Walk every audit entry across every point, count by source class,
    flag any GENERATOR sighting."""
    counts: dict[SourceClass, int] = {s: 0 for s in SourceClass}
    violations: list[str] = []
    total = 0
    for point in points:
        for entry in point.update_log:
            total += 1
            src = entry.observation.source
            counts[src] = counts.get(src, 0) + 1
            if src == SourceClass.GENERATOR:
                violations.append(
                    f"LAUNDERING: point={point.id} signal={entry.observation.signal_type}"
                )
    return AuditReport(total_updates=total, by_source=counts, violations=violations)


def assert_no_laundering(points: Iterable[Point]) -> AuditReport:
    """Audit + raise if any violation."""
    report = audit(points)
    if not report.ok:
        raise NoLaunderingInvariant(
            f"{len(report.violations)} laundering violation(s):\n"
            + "\n".join(report.violations)
        )
    return report


def inputs_of_A(point: Point) -> dict[str, object]:
    """Return the literal inputs that A(·) reads for a given point.

    Useful for static inspection — verifies that no field of P-type
    (e.g. `content`, `embedding_orig`, deltas) appears in the input set.
    """
    return {
        "n_corrob": point.n_corrob,
        "n_contra": point.n_contra,
        "n_uses": point.n_uses,
        "n_revision": point.n_revision,
        # confidence and uncertainty are pure functions of the counters above.
    }
