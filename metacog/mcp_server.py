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

# Hard cap on facts returned by walk_start : a deep walk can retrieve 100+
# turns, and shipping all their content to the agent is the dominant
# input-token cost. The composable (relevance-ranked) evidence comes first,
# topped up to this bound — recall is unaffected (fact_ids_cumulative still
# lists every id).
_MAX_RETURNED_FACTS = 20

# Weight of the original-question alignment anchor when re-ranking the
# clue_search merge (Slice B) and the walk's per-stage facts (Slice C).
# Additive: final = (1-α)·relative + α·alignment. Kept moderately high —
# the PRF / Rocchio literature warns that under-weighting the original
# query lets feedback (here: noisy clues / co-present topics) drift the
# ranking. The relative signal is never discarded (α < 1).
_ANCHOR_ALPHA = 0.5


def build_app(
    storage_path: Optional[str] = None,
    memory: Optional[Memory] = None,
) -> FastMCP:
    """Build the metacog MCP app.

    `memory` lets a caller inject an already-populated Memory (e.g. a
    benchmark conversation pre-ingested in-process) instead of creating
    an empty one. When omitted, a fresh Memory(storage_path=…) is used.
    """
    from metacog.meta_walk import MetaWalker, WalkerRegistry, _NEIGHBOUR_GATE

    app = FastMCP(
        "metacog",
        instructions=(
            "Continuously feed this memory. TWO standing obligations for the "
            "client:\n"
            "1. INDEX EVERY MESSAGE. For each turn of the conversation — BOTH "
            "the user's and your own — call `ingest_message(content, role, "
            "user_id, session_id, timestamp)` with the message's timestamp. "
            "Indexing is async and never blocks your searches.\n"
            "2. PUSH EVERY CODE BLOCK. Whenever you produce code, call "
            "`push_code(code, context, project, branch, github_user, user_id, "
            "session_id)`. The server decides whether it is project "
            "documentation, a reusable tool, or both, and files it "
            "accordingly. If the code belongs to a project, pass the project / "
            "branch / github_user so the doc node is tagged and the derived "
            "tool keeps a bidirectional link back to it."
        ),
    )
    if memory is None:
        memory = Memory(storage_path=storage_path)

    walkers = WalkerRegistry()

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

        k is capped at 7 (the system's retrieval budget).
        """
        k = min(max(1, k), 7)
        return memory.retrieve(
            query, k=k, observator_id=observator_id,
            use_hybrid=use_hybrid, use_lineage=use_lineage,
            use_spreading=use_spreading, prefer_kind=prefer_kind,
        )

    # ----------------------------------------------------------------
    # Meta-cognitive walk — streaming, step-by-step
    # ----------------------------------------------------------------

    @app.tool()
    def walk_start(
        query: str,
        n_stages: int = 16,
        facts_per_stage: int = 7,
        actions_per_stage: int = 3,
        commit: bool = False,
        observator_id: str = "",
        user_id: str = "",
        session_id: str = "",
    ) -> dict:
        """Run a COMPLETE meta-cognitive walk and return its result.

        The walk traverses three coordinated kinds (FACT / ACTION /
        THOUGHT) over the memory. Its DEPTH is governed by UNCERTAINTY
        PROPAGATION, not by the caller : a floor of at least 3 hops always
        runs, then the walk continues as long as it keeps surfacing new
        gold-relevant evidence, and stops once the propagated σ over the
        evidence chain exceeds the manifold's local resolution (the
        gold-retrieval-gated σ-cap). There is no fixed maximum — `n_stages`
        is only a generous safety bound.

        This single call loops the walk to completion internally, so the
        agent does NOT micro-drive depth stage-by-stage. The agent's job
        is BREADTH : read the gathered evidence, answer if it is enough, or
        call `walk_start` again with different / more targeted vocabulary
        (a breadth pivot).

        Result schema :
          walk_id              : str  (the walk is already complete/closed)
          stages_run           : int  — how many hops the σ-walk took
          facts                : union of retrieved FACTs across ALL stages,
                                 each with full INDEXED CONTENT, keywords,
                                 confidence, uncertainty, relevance label
          relevant_collected   : the MAP-REDUCE set of every on-target fact
                                 gathered across the whole walk
                                 ({id, content, relevance}) — COMPOSE THE
                                 ANSWER over this
          reasoning_chain      : the THOUGHT chain (one reflection per hop,
                                 each extending the prior) — the walk's
                                 reasoning thread
          fact_ids_cumulative  : every FACT id seen (effective retrieval set)
          sigma_path           : final propagated uncertainty
          drifted / n_relevant : last-hop content-relevance signals — if the
                                 whole walk drifted, pivot with new vocabulary
          done                 : always True (the walk ran to completion)
        """
        # Double-query section pre-filter : when a user/session is given,
        # the section's own points (skills/tools/turns of THIS user/session)
        # get a retrieval rank boost, alongside the free semantic query.
        section = {f"{k}:{v}" for k, v in
                   (("user", user_id), ("session", session_id)) if v} or None
        walker = MetaWalker(
            query, memory,
            n_stages=max(1, n_stages),
            facts_per_stage=max(1, facts_per_stage),
            actions_per_stage=max(1, actions_per_stage),
            commit=commit,
            observator_id=observator_id or None,
            section_filter=section,
        )
        walk_id = walkers.open(walker)
        # Loop the walk to completion : depth is the walk's own
        # uncertainty-propagation decision, not a per-tool-call step.
        _SAFETY = 40
        agg_facts: dict = {}
        stages_run = 0
        last = None
        while stages_run < _SAFETY:
            last = walker.step()
            stages_run += 1
            for f in (last.facts or []):
                fid = f.get("id") if isinstance(f, dict) else None
                if fid and fid not in agg_facts:
                    agg_facts[fid] = f
            if last.done:
                break
        out = last.to_dict()
        out["walk_id"] = walk_id
        out["stages_run"] = stages_run
        # TOKEN DISCIPLINE — the agent never needs the full per-stage union
        # (a deep walk can retrieve 100+ facts ; sending all their content
        # is the dominant input-token sink). Expose the COMPOSABLE evidence
        # (the bounded, relevance-ranked on-target set) as the primary
        # `facts`, and only top up with a few extra retrieved turns. The
        # walk's σ-cap/coverage early-returns leave the final StageOutput's
        # relevant_collected empty, so pull it straight from the walker.
        evidence = walker._composable_evidence()
        ev_ids = {e["id"] for e in evidence}
        extra = [f for f in agg_facts.values()
                 if isinstance(f, dict) and f.get("id") not in ev_ids]
        out["relevant_collected"] = evidence
        out["facts"] = (
            [agg_facts[e["id"]] for e in evidence if e["id"] in agg_facts]
            + extra
        )[:_MAX_RETURNED_FACTS]
        # MULTIPLE POSSIBILITIES — the answer turn on a vocabulary-gap case
        # often carries none of the question's words (so the seed never ranks
        # it) but sits one hop away on the conversation chain from a turn the
        # walk DID surface. So ALWAYS offer the sequence-adjacent turns of
        # everything retrieved, tagged relevance="neighbor" : they go into
        # `neighbor_possibilities` and the `facts` union the agent reads.
        # (Gating on a raw "thin grounding" count was wrong — a walk can
        # collect several relevant-LOOKING facts yet still miss the gold, so
        # it never looked thin and the neighbours were never offered.)
        # They are promoted into the COMPOSE set (`relevant_collected`) only
        # when the real on-target evidence is itself thin, so strong cases
        # are not diluted while 0-clue cat3 paths still get them to compose.
        already = {f.get("id") for f in out["facts"] if isinstance(f, dict)}
        neighbors = [n for n in walker.neighbor_possibilities()
                     if n["id"] not in already]
        if neighbors:
            out["neighbor_possibilities"] = neighbors
            out["facts"] = (out["facts"] + neighbors)[:_MAX_RETURNED_FACTS]
            # Neighbours are part of the effective retrieval set : add them to
            # `fact_ids_cumulative` so recall / the answerer's id-extraction
            # credit them (otherwise the lineage bridge is invisible to both
            # the metric and the deterministic date resolver).
            nb_ids = [n["id"] for n in neighbors]
            cum = list(out.get("fact_ids_cumulative") or [])
            out["fact_ids_cumulative"] = cum + [
                i for i in nb_ids if i not in set(cum)
            ]
            if len(evidence) < _NEIGHBOUR_GATE:
                out["relevant_collected"] = evidence + neighbors
        out["reasoning_chain"] = [
            t.content for t in getattr(walker, "_thought_chain", [])
        ]
        out["done"] = True
        walkers.close(walk_id)
        return out

    @app.tool()
    def walk_next(walk_id: str) -> dict:
        """DEPRECATED. `walk_start` now runs the whole uncertainty-governed
        walk to completion in one call (depth is decided by σ-propagation,
        not by the caller). This remains only for backward compatibility :
        the walk is already complete, so it reports done immediately. To go
        further, issue a new `walk_start` with different vocabulary (a
        breadth pivot)."""
        walker = walkers.get(walk_id)
        if walker is None:
            return {"done": True,
                    "note": "walk already complete — call walk_start to pivot"}
        # If somehow still open, drive it to completion too.
        last = walker.step()
        while not last.done:
            last = walker.step()
        walkers.close(walk_id)
        out = last.to_dict()
        out["done"] = True
        return out

    @app.tool()
    def walk_keepup(
        query: str, n_stages: int = 16,
        user_id: str = "", session_id: str = "",
    ) -> dict:
        """KEEPUP mode — organic streaming answer. Runs the uncertainty-
        governed walk and returns the full SNAPSHOT TRAJECTORY: one entry
        per stage, each a PROVISIONAL answer (re-written as evidence
        accumulates) plus the live thought and depth-gate state. The LAST
        snapshot is the validated final answer — generation is continuous,
        so there is no separate final pass to wait for.

        Over the SSE transport a client renders these as a message that
        rewrites itself until `done=True`. Returns
        {"snapshots": [...], "final": <last answer>, "stages": n}.
        """
        snaps = list(memory.answer_keepup(
            query, n_stages=max(1, n_stages),
            user_id=user_id, session_id=session_id,
        ))
        final = next((s["answer"] for s in reversed(snaps) if s.get("answer")), "")
        return {"snapshots": snaps, "final": final, "stages": len(snaps)}

    @app.tool()
    def scoped_answer(
        query: str, tags: list, knowledge_base: bool = False,
        n_stages: int = 16, match: str = "exact",
    ) -> dict:
        """Tag-filtered retrieval as a TWO-PHASE cascade. Phase 1 runs the
        walk HARD-restricted to points carrying ALL `tags` (e.g.
        ["session:s1"] — a discussion) to find what is being talked about
        there. If knowledge_base=True, Phase 2 then searches the WHOLE
        memory seeded by Phase-1's finding (discussion context → global KB);
        otherwise the answer stays strictly within the filtered set.

        `match` controls how each tag in `tags` is resolved:
          - "exact"  : equality OR hierarchical ancestry (filtering on
                       `ref:date` matches a point carrying `ref:date:2022`).
          - "fuzzy"  : segment-wise Levenshtein (typos on tag names).
          - "regex"  : `re.search` on each tag, e.g. `r"^session:s\\d+$"`.

        Returns the scoped answer/evidence (+ global ones when enabled)."""
        return memory.scoped_answer(
            query, tags=list(tags or []),
            knowledge_base=bool(knowledge_base), n_stages=max(1, n_stages),
            match=str(match or "exact"),
        )

    @app.tool()
    def list_tags() -> dict:
        """Return the GLOSSARY of tag NAMESPACES across the whole memory.

        Hierarchical tags use `:` as the separator (`ref:date:2022`,
        `session:s1`). The glossary keeps every PARENT PREFIX (the
        namespace keys) and drops the leaf (the concrete value): `ref`,
        `ref:date`, `session`, … Flat tags (no `:`) are included verbatim.

        The catalogue is ordered by hierarchy depth first (shallowest
        namespaces first), then alphabetically — top-level namespaces
        appear before their children. Use this to discover which tag
        filters are available for `scoped_answer` / `presearch` /
        `walk_start`."""
        from metacog.tags import tag_glossary
        return {"namespaces": tag_glossary(memory.points)}

    @app.tool()
    def presearch(
        queries: list, k_per_query: int = 3, tags: list = None,
        match: str = "exact", observator_id: str = "",
    ) -> dict:
        """BATCH RECONNAISSANCE — return the top-`k_per_query` nearest hits
        for EACH query, WITHOUT triggering a walk. The agent's gate before
        spending a full uncertainty-governed walk on a query: a query whose
        pre-search is empty/weak is reformulated and re-batched until the
        results look probative; ONLY THEN the agent calls `walk_start` on
        the validated query.

        Without this gate a weak query still cascades through the walk and
        the knowledge graph for nothing — the gate keeps that cost behind a
        cheap kNN check.

        Parameters
        ----------
        queries        : list[str]  — the batch of candidate queries.
        k_per_query    : int        — top-N kept per query (default 3, the
                                      smallest informative aperture).
        tags           : list[str]  — optional pre-filter; only points
                                      satisfying ALL these tags under `match`
                                      are searched. Same tri-modal semantics
                                      as `scoped_answer`.
        match          : str        — "exact" | "fuzzy" | "regex".
        observator_id  : str        — optional Level-1 community lens.

        Returns `{"results": [{"query", "hits": [{"id","content","score",
        "kind","tags"}], "n_hits": int}], "n_queries": int}`.
        """
        from metacog.tags import filter_points
        qs = [str(q) for q in (queries or []) if q]
        tagset = list(tags or [])
        if tagset:
            allowed = filter_points(memory.points, tagset, mode=match)
            scope = [p for p in memory.points if p.id in allowed]
        else:
            scope = None  # full memory, retrieve handles it
        # Stash + restore points to honour the tag pre-filter without
        # adding a new code path through retrieve().
        out_results = []
        original = memory.points
        try:
            if scope is not None:
                memory.points = scope
            k = max(1, int(k_per_query))
            for q in qs:
                hits = memory.retrieve(
                    q, k=k, observator_id=observator_id or None,
                )
                trimmed = [
                    {
                        "id": h["id"],
                        "content": h["content"],
                        "score": h["score"],
                        "kind": h["kind"],
                        "tags": [
                            t for t in (next(
                                (p.tags for p in original if p.id == h["id"]),
                                [],
                            ))
                        ],
                    }
                    for h in hits
                ]
                out_results.append(
                    {"query": q, "hits": trimmed, "n_hits": len(trimmed)}
                )
        finally:
            memory.points = original
        return {"results": out_results, "n_queries": len(qs)}

    @app.tool()
    def clue_search(
        question: str, k_per_clue: int = 3, n_clues: int = 6,
        observator_id: str = "",
    ) -> dict:
        """INFERENCE-question expander. For a question whose answer is never
        stated directly ("What might John's financial status be?"), the
        evidence is a casual aside in a vocabulary that shares NOTHING with
        the question ("my kids have so much" ⇒ well-off). Querying in the
        question's words retrieves the wrong turns; HyDE on a hypothetical
        ANSWER ("John is wealthy") also misses — it stays in the money
        register, not the evidence register.

        `clue_search` instead has the server BRAINSTORM concrete first-person
        chat lines that would each be EVIDENCE for a DIFFERENT plausible
        answer (spanning the answer space — well-off AND struggling), then
        retrieves over those clue lines. Whichever clue matches a real turn
        surfaces the actual aside, and you infer the answer from what stuck.

        Use this when a plain `presearch`/`walk_start` on the question
        drifts or returns only generic on-topic turns. Returns
        `{"clues":[...], "results":[{clue,hits}], "merged":[top unique hits]}`.
        """
        from metacog.clues import clue_utterances
        clues = clue_utterances(question, memory.llm, n=max(1, int(n_clues)))
        if not clues:
            return {"clues": [], "results": [], "merged": [],
                    "note": "no clues generated (no usable LLM?)"}
        k = max(1, int(k_per_clue))
        tag_by_id = {p.id: list(p.tags) for p in memory.points}
        per_clue = []
        merged: dict = {}              # id -> best hit (highest score)
        for c in clues:
            hits = memory.retrieve(c, k=k, observator_id=observator_id or None)
            trimmed = [
                {"id": h["id"], "content": h["content"], "score": h["score"],
                 "kind": h["kind"], "tags": tag_by_id.get(h["id"], [])}
                for h in hits
            ]
            per_clue.append({"clue": c, "hits": trimmed, "n_hits": len(trimmed)})
            for h in trimmed:
                cur = merged.get(h["id"])
                if cur is None or h["score"] > cur["score"]:
                    merged[h["id"]] = h
        merged_list = sorted(merged.values(), key=lambda h: -h["score"])
        # LINEAGE BRIDGE — the clues often land on the SPACED neighbours of
        # the real aside (kids/family clues hit D5:3 and D5:6, the gold D5:5
        # sits between them). Gap-fill the conversation chain between the
        # merged hits so the in-between answer turn surfaces too.
        from metacog.meta_walk import lineage_neighbors
        # Generous bridge for clue recon : many clue hits act as seeds, so a
        # wider window + larger cap is needed for a dist-2 in-between turn
        # (the gold one step past a clue hit) to survive the cap race.
        nbrs = lineage_neighbors([h["id"] for h in merged_list], memory.points,
                                 window=3, cap=40)
        if nbrs:
            for n in nbrs:
                n["tags"] = tag_by_id.get(n["id"], [])
            merged_list = merged_list + nbrs
        # SLICE B — Rocchio / ReformIR anchor to the ORIGINAL question.
        # The per-clue / bridge scores are RELATIVE (to a brainstormed clue),
        # so a noisy clue ("ordered books on gardening") or a co-present topic
        # drifts the ranking. Re-rank ADDITIVELY against the original
        # question's typed alignment (ColBERT MaxSim) so the on-question turn
        # ("Researching adoption agencies" for "what did X research?") rises
        # and off-question noise sinks — without discarding the clue signal.
        try:
            from metacog.query_anchor import (
                build_query_anchor, alignment_score)
            anchor = build_query_anchor(
                question,
                entity_extractor=getattr(memory, "entity_extractor", None),
                keyword_extractor=getattr(memory, "extractor", None),
                encoder=getattr(memory, "encoder", None),
            )
            if not anchor.is_empty() and merged_list:
                pt = {p.id: p for p in memory.points}
                scs = [h["score"] for h in merged_list
                       if isinstance(h.get("score"), (int, float))]
                lo, hi = (min(scs), max(scs)) if scs else (0.0, 1.0)
                rng = (hi - lo) or 1.0
                for h in merged_list:
                    s = h.get("score")
                    rel = ((s - lo) / rng
                           if isinstance(s, (int, float)) else 0.0)
                    p = pt.get(h["id"])
                    al = alignment_score(
                        anchor, h.get("content", ""),
                        getattr(p, "embedding_orig", None) if p else None)
                    h["align"] = round(al, 4)
                    h["rank_score"] = round(
                        (1.0 - _ANCHOR_ALPHA) * rel + _ANCHOR_ALPHA * al, 4)
                merged_list.sort(key=lambda h: -h.get("rank_score", 0.0))
        except Exception:
            pass
        # Expose fact_ids_cumulative so recall / id-extraction credit the
        # clue hits + lineage bridge (same field walk_start uses), otherwise
        # a clue_search that DID surface the gold scores recall 0.
        cum_ids: list = []
        seen_c: set = set()
        for h in merged_list:
            if h["id"] not in seen_c:
                seen_c.add(h["id"])
                cum_ids.append(h["id"])
        return {"clues": clues, "results": per_clue, "merged": merged_list,
                "neighbor_possibilities": nbrs, "fact_ids_cumulative": cum_ids}

    @app.tool()
    def reason(query: str, with_executor: bool = True, apply_compression: bool = True) -> dict:
        """Run a full reasoning trajectory until output convenable."""
        result = memory.reason(query, with_executor=with_executor, apply_compression=apply_compression)
        if memory.storage_path:
            memory.save()
        return result

    # Per-(user, session) skill-JSON cache : the last skill folder built or
    # ingested in a session, keyed by the caller's user token + session id.
    _session_skills: dict = {}

    @app.tool()
    def ingest_skill(
        tree: dict, name: str, user_id: str, session_id: str,
        date: str = "",
    ) -> dict:
        """Re-index a NAMED skill — a theoretical directory tree of tools
        (nested JSON: dir→sub-object, file→description) — as tagged points
        the walk can retrieve like any fact. Tags every node
        skill·tool·name:<name>·user:<id>·session:<id>·date:<…> and encodes
        the directory topology via lineage + ref:skill:* tokens. Also caches
        the skill JSON in this (user, session)."""
        pts = memory.ingest_skill(
            tree, name=name, user_id=user_id, session_id=session_id,
            date=date or None,
        )
        _session_skills[(user_id, session_id)] = {"name": name, "tree": tree}
        if memory.storage_path:
            memory.save()
        return {"skill": name, "points_created": len(pts),
                "ids": [p.id for p in pts]}

    @app.tool()
    def build_skill(
        query: str, user_id: str, session_id: str, n_stages: int = 8,
    ) -> dict:
        """Build a skill end-to-end (TASK mode) : a walk gathers tools
        (gold = tool fact OR semantic fact) until enough to elaborate the
        complete workflow, synthesises a NAMED skill folder, re-indexes it,
        and caches it in this (user, session). Returns the skill JSON
        {"name","tree"}."""
        skill = memory.build_skill(
            query, user_id=user_id, session_id=session_id, n_stages=n_stages,
        )
        _session_skills[(user_id, session_id)] = skill
        if memory.storage_path:
            memory.save()
        return skill

    @app.tool()
    def get_session_skill(user_id: str, session_id: str) -> dict:
        """Return the skill JSON cached for this (user, session), or {}."""
        return _session_skills.get((user_id, session_id), {})

    @app.tool()
    def capture_code_tool(
        code: str, context: str = "", lang: str = "python",
        user_id: str = "", session_id: str = "",
    ) -> dict:
        """Chat-side hook : after generating code, ask whether it is a
        REUSABLE TOOL and, if so, FEED IT INTO THE RAG. Self-assesses via
        tool-intent ; on yes, re-indexes the code as an executable tool node
        (tagged tool·skill·code·name:·user:·session:·date:) the walk can
        retrieve like any fact and reuse next time. Returns
        {"captured": bool, "id", "name", "kind"}."""
        p = memory.capture_code_tool(
            code, context=context, lang=lang,
            user_id=user_id, session_id=session_id,
        )
        if memory.storage_path and p is not None:
            memory.save()
        if p is None:
            return {"captured": False}
        name = next((t.split(":", 1)[1] for t in p.tags
                     if t.startswith("name:")), "")
        kind = next((t.split(":", 1)[1] for t in p.tags
                     if t.startswith("kind:")), "")
        return {"captured": True, "id": p.id, "name": name, "kind": kind}

    @app.tool()
    def ingest_message(
        content: str, role: str, user_id: str, session_id: str,
        timestamp: str = "",
    ) -> dict:
        """EPISODIC FEED — index ONE conversation message (role = "user" or
        "agent") of a session, preserving its timestamp. Call this for EVERY
        turn on BOTH sides. Indexing is ASYNC and NON-BLOCKING : the message
        is queued and indexed in the background, so it never delays your
        searches. Messages chain in order (sequence_prev) and are tagged
        episode·message·role:·user:·session:·ts:·date:."""
        memory.ingest_message(
            content, role=role, user_id=user_id, session_id=session_id,
            timestamp=timestamp or None, block=False,
        )
        return {"queued": True}

    @app.tool()
    def push_code(
        code: str, context: str = "", project: str = "", branch: str = "",
        github_user: str = "", lang: str = "python",
        user_id: str = "", session_id: str = "",
    ) -> dict:
        """PUSH generated code into the RAG. The server EVALUATES it and
        routes it : PROJECT code → a semantic FACT documentation node tagged
        project·branch·github ; TOOL-able code → a metacognitive rewrite
        re-indexed as an executable tool ; BOTH → both, with a bidirectional
        ref: filiation so the tool remembers its project and the project doc
        points at the derived tool. Pass project/branch/github_user when the
        code belongs to a project. Returns the routing summary."""
        r = memory.push_code(
            code, context=context, project=project, branch=branch,
            github_user=github_user, lang=lang,
            user_id=user_id, session_id=session_id,
        )
        if memory.storage_path and r.get("pushed"):
            memory.save()
        return r

    @app.tool()
    def sleep() -> dict:
        """Run a sleep cycle of collision resolution."""
        result = memory.sleep()
        if memory.storage_path:
            memory.save()
        return result

    @app.tool()
    def match_tool(query: str) -> Optional[dict]:
        """Capability cache : does a previously-GENERATED tool already cover
        this query? Returns the tool (id, content, keywords, tags, score) or
        null. Call this BEFORE thinking from scratch — a hit means the
        approach is already known and can be reused, skipping a think
        phase."""
        return memory.match_tool(query)

    @app.tool()
    def ensure_tool(query: str, how: str = "", genre: str = "command") -> dict:
        """Get a tool for this need, generating it if absent. If a
        previously-generated tool covers `query`, it is REUSED (skip
        thinking). Otherwise a new tool is GENERATED from `query` + the
        intended approach `how` and added to the self-built set. This is the
        'no tool -> generate it -> proceed' step : it never blocks, it only
        ever adds a capability. Returns {tool, reused}."""
        result = memory.ensure_tool(query, how=how or None, genre=genre)
        if memory.storage_path:
            memory.save()
        return result

    @app.tool()
    def crystallize_skills() -> dict:
        """Crystallize recurring actions (re-derived across diverse queries)
        into persistent TOOL nodes — normal kind=ACTION points tagged
        "tool", scoped by keywords, added to the cloud so the walk finds
        them recursively and they persist on save."""
        result = memory.crystallize_skills()
        if memory.storage_path:
            memory.save()
        return result

    @app.tool()
    def list_tools_learned() -> List[dict]:
        """List the TOOL nodes the system has generated so far (the closed,
        self-built capability set), each with its id, content, genre/context
        tags and scope keywords."""
        from metacog.skills import tools_in
        return [
            {"id": p.id, "content": p.content,
             "tags": list(p.tags or []), "keywords": list(p.keywords or [])}
            for p in tools_in(memory.points)
        ]

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
    def list_communities() -> List[dict]:
        """List the Level-1 topical communities (auto-detected observators).

        Each is {id, name, keywords, n_points}. Pass an `id` to
        `walk_start(observator_id=…)` to FOCUS the walk's FACT retrieval on
        that community's turns (entity beacons + scaffolding stay visible).
        Returns [] when no communities were detected — then just walk over
        the whole memory as usual.
        """
        out: List[dict] = []
        for oid, obs in memory.observators.items():
            if not oid.startswith("auto-comm-"):
                continue
            n = sum(1 for p in memory.points if oid in p.observator_views)
            out.append({
                "id": oid,
                "name": obs.name,
                "keywords": obs.keywords[:8],
                "n_points": n,
            })
        return out

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
        choices=["stdio", "sse", "streamable-http"],
        default=os.environ.get("METACOG_TRANSPORT", "stdio"),
        help="MCP transport. stdio (default, for Claude Code) ; sse or "
             "streamable-http for live-streaming HTTP clients (keepup).",
    )
    parser.add_argument(
        "--host", default=os.environ.get("METACOG_HOST", "127.0.0.1"),
        help="Bind host for the sse / streamable-http transports.",
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("METACOG_PORT", "8765")),
        help="Bind port for the sse / streamable-http transports.",
    )
    args = parser.parse_args()

    app = build_app(storage_path=args.storage)
    if args.transport in ("sse", "streamable-http"):
        # Configure the HTTP bind for the streaming transports.
        try:
            app.settings.host = args.host
            app.settings.port = args.port
        except Exception:
            pass
    app.run(transport=args.transport)


if __name__ == "__main__":
    main()
