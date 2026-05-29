"""
MetaCog-Mem — MCP server.

Exposes the Memory wrapper as an MCP service so any MCP-aware client
(Claude Code, Claude Desktop, etc.) can interact with the metacognitive
memory.

Run :
  python -m metacog.mcp_server [--storage PATH]

Registers as the "metacog" server. Tools advertised :
  ingest          add a new FACT / THOUGHT / ACTION
  observe         apply an Observation on existing point(s)
  process_turn    record a conversation turn (detectors fire if user)
  retrieve        top-k semantic retrieval
  reason          full reasoning trajectory until output convenable
  sleep           run a collision sleep cycle
  inspect         dump a point's full state
  audit           verify no laundering
  stats           system overview
  declare_observator  manually declare an observator
  spawn_observators   auto-spawn from a polarized point
  route           pick the top-k observators for a query
  save / load     persistence
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from metacog.memory import Memory


def build_app(
    storage_path: Optional[str] = None,
    memory: Optional[Memory] = None,
) -> FastMCP:
    """Build the metacog MCP app.

    `memory` lets a caller inject an already-populated Memory (e.g. a
    benchmark conversation pre-ingested in-process) instead of creating
    an empty one. When omitted, a fresh Memory(storage_path=…) is used.
    """
    app = FastMCP("metacog")
    if memory is None:
        memory = Memory(storage_path=storage_path)

    @app.tool()
    def ingest(content: str, kind: str = "FACT", id: Optional[str] = None) -> dict:
        """Add a new point to the memory.

        Args:
          content: the text content (∈ P).
          kind:    one of FACT | THOUGHT | ACTION (default FACT).
          id:      optional explicit id (auto-generated otherwise).
        """
        p = memory.ingest(content, kind=kind, id=id)
        if memory.storage_path:
            memory.save()
        return memory.inspect(p.id)

    @app.tool()
    def observe(
        target_ids: List[str],
        polarity: float,
        source: str = "OBSERVER",
        signal_type: str = "external",
        observator_id: str = "default",
        raw_content: Optional[str] = None,
    ) -> dict:
        """Apply an Observation on one or more points.

        polarity > 0 → corroboration, < 0 → contradiction.
        source must be OBSERVER or COMPUTATION (GENERATOR is rejected).
        """
        obs = memory.observe(
            target_ids=target_ids,
            polarity=polarity,
            source=source,
            signal_type=signal_type,
            observator_id=observator_id,
            raw_content=raw_content,
        )
        if memory.storage_path:
            memory.save()
        return {
            "signal_type": obs.signal_type,
            "polarity": obs.polarity,
            "target_node_ids": list(obs.target_node_ids),
            "observator_id": obs.observator_id,
            "applied_to": [memory.inspect(tid) for tid in obs.target_node_ids],
        }

    @app.tool()
    def process_turn(
        text: str,
        speaker: str = "user",
        retrieved_point_ids: Optional[List[str]] = None,
    ) -> dict:
        """Record a conversation turn. If speaker=user, runs detectors
        and applies the detected Observations."""
        observations = memory.process_turn(
            text=text, speaker=speaker, retrieved_point_ids=retrieved_point_ids,
        )
        if memory.storage_path:
            memory.save()
        return {
            "speaker": speaker,
            "text": text,
            "detected_signals": [
                {
                    "signal_type": o.signal_type,
                    "polarity": o.polarity,
                    "targets": list(o.target_node_ids),
                }
                for o in observations
            ],
        }

    @app.tool()
    def retrieve(
        query: str,
        k: int = 5,
        observator_id: Optional[str] = None,
        use_hybrid: bool = True,
        use_lineage: bool = True,
        use_spreading: bool = True,
        prefer_kind: Optional[str] = None,
    ) -> List[dict]:
        """Retrieve top-k points for a query.

        Args:
          query:        natural-language search text.
          k:            number of points to return.
          observator_id: route through a named observator's view (optional).
          use_hybrid:   keyword-embedding cosine + BM25 fallback + RRF
                        (default True ; the primary retrieval path).
          use_lineage:  expand the fused top-k along parent/child/sequence
                        links with uncertainty-pruned RRF (default True).
          use_spreading: geometric spreading activation — expand the base
                        top-k to their manifold neighbours (edge-free
                        analog of associative spreading; default True).
          prefer_kind:  boost a PointKind in ranking — FACT | THOUGHT |
                        ACTION. Use ACTION for "how do I X" queries.
        """
        return memory.retrieve(
            query, k=k, observator_id=observator_id,
            use_hybrid=use_hybrid, use_lineage=use_lineage,
            use_spreading=use_spreading, prefer_kind=prefer_kind,
        )

    @app.tool()
    def reason(query: str, with_executor: bool = True, apply_compression: bool = True) -> dict:
        """Run a full reasoning trajectory until output convenable."""
        result = memory.reason(query, with_executor=with_executor, apply_compression=apply_compression)
        if memory.storage_path:
            memory.save()
        return result

    @app.tool()
    def sleep() -> dict:
        """Run a sleep cycle of collision resolution."""
        result = memory.sleep()
        if memory.storage_path:
            memory.save()
        return result

    @app.tool()
    def inspect(point_id: str) -> Optional[dict]:
        """Return the full state of a single point."""
        return memory.inspect(point_id)

    @app.tool()
    def audit() -> dict:
        """Audit the memory for laundering violations."""
        return memory.audit()

    @app.tool()
    def stats() -> dict:
        """System overview : counts by kind and state, observators, turns."""
        return memory.stats()

    @app.tool()
    def declare_observator(
        id: str, name: str = "", keywords: Optional[List[str]] = None,
    ) -> dict:
        """Explicitly declare an Observator with optional expertise keywords."""
        obs = memory.declare_observator(id, name=name, keywords=keywords)
        if memory.storage_path:
            memory.save()
        return {
            "id": obs.id,
            "name": obs.name,
            "keywords": list(obs.keywords),
        }

    @app.tool()
    def detect_polarized() -> List[str]:
        """List ids of points with current polarization."""
        return memory.detect_polarized_points()

    @app.tool()
    def spawn_observators(point_id: str, strategy: str = "clustering") -> List[str]:
        """Auto-spawn observators on a polarized point. Strategy ∈
        {clustering, polarity}."""
        ids = memory.spawn_observators(point_id, strategy=strategy)
        if memory.storage_path:
            memory.save()
        return ids

    @app.tool()
    def route(query: str, k: int = 1) -> List[dict]:
        """Pick the top-k observators most aligned with the query."""
        return memory.route(query, k=k)

    @app.tool()
    def save() -> str:
        """Persist the current state to disk."""
        if not memory.storage_path:
            return "no storage_path configured"
        memory.save()
        return f"saved to {memory.storage_path}"

    return app


def main():
    parser = argparse.ArgumentParser(prog="metacog.mcp_server")
    parser.add_argument(
        "--storage",
        default=os.environ.get("METACOG_STORAGE", None),
        help="Path to the persistence file (pickle). If omitted, in-memory only.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="MCP transport. Default stdio (standard for Claude Code).",
    )
    args = parser.parse_args()

    app = build_app(storage_path=args.storage)
    app.run(transport=args.transport)


if __name__ == "__main__":
    main()
