# OmniCognition / MetaCog-Mem

### A Hyperparameter-Free Metacognitive Memory Manifold for LLM Agents

*Epistemically-typed points, edge-free geometric relations, and a
retrieval walk whose depth is governed by the propagation of uncertainty.*

---

## Abstract

We present **MetaCog-Mem**, a memory architecture for large-language-model
agents built on three commitments: (i) memory is a **manifold of
epistemically-typed points** — facts, thoughts, actions, and self-built
tools sharing one schema and one lifecycle — rather than a graph of
materialized edges; (ii) every threshold is either a mathematical constant
or **emerges from the data** (no tunable hyperparameters); and (iii) the
provenance of every datum is **typed and audited**, so generated content
can become *content* but never *evidence* (anti-laundering by
construction). Retrieval is a coordinated **walk** over the three point
kinds at once. Its central novelty is that **walk depth is governed by the
propagation of uncertainty** (GUM 1995) over the retrieved evidence chain:
a hard floor of three hops, an emergent σ-cap derived from the manifold's
own local resolution, and a deterministic keyword-coverage stop — no
fixed maximum and no learned controller. A single invocation runs the
walk to completion; an agent above it performs only *breadth pivots*. On
the LoCoMo long-conversation benchmark the walk raises evidence
**agent-recall from a static 0.31 (Recall@7) to ≈ 0.85**, while a
keyword-oriented token-discipline pass cuts the worst-case per-query input
cost by **84 %** (118k → 19k tokens). Relative temporal expressions are
resolved to absolute dates deterministically at both ingestion and answer
time. The system is implemented as a Python library and an MCP service;
all claims are reproducible from the included LoCoMo harness.

---

## 1. Introduction

LLM agents that converse over long horizons need a memory that (a)
*retrieves* the evidence bearing on a question even when the question and
the evidence share no surface vocabulary, (b) *reasons* across several
hops without losing the thread, (c) *consolidates* redundancy on its own,
and (d) never silently promotes its own guesses to facts. Graph-of-edges
memories satisfy (a)–(c) at the price of materialized relations and a
thicket of tuning knobs; they rarely address (d).

MetaCog-Mem takes the opposite stance on each axis. Relations are not
stored — they are the **geometric pull** between points in a
keyword-embedding manifold (§2). The retrieval walk is not bounded by a
hand-set depth — it is bounded by the **accumulated uncertainty of its own
evidence chain** (§4). Consolidation is not scheduled — it fires from a
single **Poisson floor** that is parameter-free (§5). And provenance is
not advisory — a generated `Observation` raises a `LaunderingError` at the
type level (§2.3).

The remainder of this document specifies each component and reports the
LoCoMo evaluation. Module-level documentation lives in
[`metacog/README.md`](metacog/README.md); the benchmark harness and the
full per-version results table live in
[`benchmarks/locomo/README.md`](benchmarks/locomo/README.md).

---

## 2. The epistemic substrate

### 2.1 Typed points share one database

A memory is a set of `Point`s (`metacog/epistemic.py`). Every point —
whether a `FACT`, a `THOUGHT`, an `ACTION`, or a self-built tool — carries
the same schema: an original embedding, a position-weighted **keyword
embedding** (the projection of the point onto an entity manifold), a Beta
posterior over corroboration/contradiction counters, an open **tag**
vocabulary (`tool`, `tool_workflow`, `entity`, `atomic`,
`lateral_absorbed`, …) that refines the point's role over time, and a
lineage. One schema, one assimilation function `A(·)`, one state machine.

### 2.2 Manifold over graph

There are no materialized edges. The relation between two points *is* their
proximity under the current **effective embedding** — a Reciprocal-Rank
fusion of semantic cosine, position-weighted keyword cosine, BM25 on raw
content, and fuzzy Levenshtein (`metacog/geometry.py`,
`metacog/bm25.py`, `metacog/fuzzy.py`). Two points are "linked" exactly
when `apply_pull` has drawn them together; the pull's step size
`1/(1+n_obs)` is itself parameter-free. Edge-free entity *beacons*
(`metacog/entities.py`) reproduce the effect of an extraction edge purely
by co-location.

### 2.3 Anti-laundering by construction

Following Romanchuk & Bondar's *Semantic Laundering*
(arXiv:2601.08333), provenance is a **type**. Observations carry a
`SourceClass`; constructing an `Observation(source=GENERATOR)` raises
`LaunderingError`. LLM output may enter the manifold as *content* (∈ P)
— a generated `THOUGHT` or `ACTION` — but it can never enter the
observation set O that updates a point's epistemic counters. The
`metacog/audit.py` pass verifies the invariant holds over a whole store
(Corollary 5).

### 2.4 No hyperparameters, no deletion

Every "threshold" is a mathematical constant (`cos π/4`, `cos π/12`) or
emerges from the population (`median ± σ` for collision; the Poisson floor
`λ + √λ` for every recurrence law). Invalidated points are not removed:
the states `INVALID`/`DEPRECATED` **exile** a point geometrically, and new
corroborating observations resurrect it automatically.

---

## 3. The meta-cognitive walk

Retrieval is a coordinated traversal over the three kinds at once
(`metacog/meta_walk.py`). Each stage:

1. **retrieves** the nearest FACTs and ACTIONs to the current seed (hybrid
   RRF), with an additive **HyDE** channel (Gao et al. 2022) that
   generates three hypothetical evidence lines and RRF-merges their
   retrieval — closing the vocabulary gap on abstract "inference"
   questions;
2. **reads** each retrieved fact with a **Chain-of-Note** relevance pass
   (Yu et al. 2024), folding the on-target subset into a persistent
   MAP-REDUCE accumulator that no later stage can discard;
3. **synthesizes** a `THOUGHT` whose keywords are drawn *only* from the
   walked points' preexisting keywords (never invented), extending a
   reduce-folded reasoning chain that seeds the next stage.

A single `walk_start` runs the walk to **completion** (§4); the agent
above it performs only **breadth pivots** — re-issuing `walk_start` with
different vocabulary when a thread is exhausted — never stage-by-stage
micro-driving. This alignment of the two control loops is what lets the
uncertainty-governed depth actually run at inference time.

### 3.1 Keepup — organic streaming

Because the depth-gate (§4) already evaluates sufficiency at every stage,
the system can answer **organically** instead of waiting for a final pass.
In **keepup** mode (`Memory.answer_keepup`, MCP `walk_keepup`) a
*provisional* answer is synthesised from the very first walk and
**re-written every stage** as evidence accumulates; the live `THOUGHT` is
streamed alongside it. The walk runs once, continuously — so the moment
the σ / coverage gate validates (`done`), the provisional answer is
*already* the final one: there is no separate final-generation latency. If
the answer was correct at stage 4 and stage 5 only confirms it, the user
saw the right answer a stage early. Generation is therefore continuous and
asynchronous; over the SSE transport (§7.2) the client renders a single
message that rewrites itself until validated.

---

## 4. Uncertainty-governed depth

The walk's defining property is that **its depth is the propagation of
uncertainty over the evidence chain**, in the sense of the *Guide to the
Expression of Uncertainty in Measurement* (BIPM, GUM 1995;
`metacog/uncertainty.py`). Let `fact★_k` be the fact the walk commits to
at stage *k*. We accumulate

```
        σ_path  =  √( Σ_k  σ_hop,k² )
```

where, crucially, `σ_hop,k` is **not** the volatility of the fact choice
but a measure of **chain coherence**:

```
        σ_hop,k  =  min_{f ∈ retrieved_k}  ( 1 − cos(fact★_{k−1}, f) )
```

— the smallest keyword-embedding distance from where the chain last stood
to anything reachable this stage. Three structural rules, no tuning knobs,
govern termination:

- **Floor.** The first `_MIN_STAGES = 3` hops always run.
- **σ-cap.** Once `σ_path` exceeds the walk-local emergent threshold —
  the median + std of the stage-0 facts' pairwise distances, i.e. the
  manifold's *own* local resolution — the evidence chain has wandered out
  of the coherent neighbourhood and the walk stops. A coherent gold trail
  keeps consecutive hops small, so σ grows slowly and the walk goes deep
  on its own; wandering adds a large hop and trips the cap fast.
  **Gold-retrieval extension is therefore intrinsic to σ**, requiring no
  separate relevance gate. (An earlier Chain-of-Note gold-gate was
  removed: its signal was stochastic and bimodal — one run labelled every
  turn relevant and the walk ran 16 stages collecting 80 facts, the next
  labelled none and the walk was strangled at the floor.)
- **Keyword-coverage stop.** A deterministic, LLM-free metacognitive
  check: once the gathered evidence's keywords cover every query keyword,
  the walk holds turns bearing on each facet of the question and stops.
  Vocabulary-gap questions never reach full coverage, so σ governs those —
  they are never cut short.

Put as a single termination predicate, the walk halts at stage *k* iff

```
   k ≥ _MIN_STAGES   AND   ( σ_path(k) > cutoff   OR   covers(query, evidence) )
```

with `cutoff = median(d₀) + std(d₀)` over the stage-0 pairwise distances.
**Uncertainty is the primary stop**: the walk continues precisely as long
as its evidence chain stays inside the manifold's own resolution, and
terminates the moment the accumulated σ says the chain can no longer be
trusted — no fixed maximum, no learned controller, no caller override
(§7.2). The floor guarantees a minimum exploration; coverage is an early
exit for questions whose vocabulary the evidence already spans.

**Why this matters empirically.** A naive depth measure — the
stage-to-stage drift of the re-anchored seed — is ≈ 0 (the seed is
re-anchored on the query each hop) and never trips; a fact★-to-fact★ drift
is roughly constant (~0.25/hop, the volatility of *choosing* the
least-uncertain fact) and trips uniformly at 4–5 stages with no adaptive
range. The coherence signal above is the one that produces *adaptive*
depth: easy queries cap early via coverage, vocabulary-gap inference
queries dig as long as the trail stays coherent.

### 4.1 Token discipline

Depth without bound is expensive. Two keyword-oriented, deterministic
caps hold the cost down without touching recall:

- `walk_start` returns at most `_MAX_RETURNED_FACTS = 20` retrieved turns
  (a deep walk can surface 100 +; shipping all their content to the agent
  was the dominant input-token sink).
- The composable evidence handed to synthesis is capped to
  `_MAX_EVIDENCE = 15`, ranked by relevance label then **chain-vocabulary**
  overlap (so vocabulary-distant gold the walk bridged to is kept, not the
  raw query terms).

Full recall is preserved in `fact_ids_cumulative`; only what is *composed
over* is bounded. On the worst observed case these caps cut per-query
input from **118k to 19k tokens (−84 %)** with an unchanged answer.

### 4.2 Deterministic temporal resolution

Temporal questions need the **absolute** dates the gold answers use, but
conversation turns speak in **relative** terms ("last year", "yesterday",
"3 years ago"). MetaCog-Mem resolves this in **two deterministic stages**,
no LLM in the resolution path:

**Stage 1 — at ingestion** (`benchmarks/locomo/eval.py`,
`_expand_relative_dates`). Each turn carries its session date as a content
prefix (`[8 May 2023] Melanie: …`). Every relative expression in the turn
is computed against that anchor and the turn is augmented with **two**
forms — a typed token for retrieval and a plain clause for generation:

| expression (turn) | session date | appended to content |
|---|---|---|
| "… last year" | 8 May 2023 | `[ref:date:year:2022] (absolute date: 2022)` |
| "yesterday …" | 17 March 2022 | `[ref:date:day:16 ref:date:month:march ref:date:year:2022] (absolute date: 16 March 2022)` |
| "for 3 years" | 27 March 2023 | `[ref:date:year:~2020] (absolute date: around 2020)` |
| "last month" | 10 June 2023 | `[ref:date:month:may ref:date:year:2023] (absolute date: May 2023)` |

The `ref:date:*` compounds are *typed* tokens (so the keyword extractor and
BM25 match `month:march`, not a bare `march` diluted by every dated
prefix); the `(absolute date: …)` clause is what the generator reads. The
original wording is never altered — the expansion is additive.

**Stage 2 — at answer time** (`benchmarks/locomo/mcp_meta_agent.py`,
`_resolve_relative_date_answer`). If the generated answer still *is* or
*contains* a relative phrase, a post-processor finds the evidence turn that
carries the **same** phrase plus an `(absolute date: X)` clause and
rewrites the answer to `X` (falling back to the unique absolute date seen
if the phrase match is ambiguous). This closes the temporal **0-clue**
case — *"When did Melanie paint a sunrise?"* where the gold turn is an
image with no retrievable text: the walk answers from the dated neighbour,
the generator echoes "last year", and Stage 2 rewrites it to **"2022"**
(token-F1 = 1.0). Both stages are pure string/arithmetic operations and
deterministic across runs.

---

## 5. Self-organizing consolidation — one law, four mechanisms

A single Poisson floor (`λ + √λ`, no hyperparameter) drives four
self-organizing forms of *deduplication*, each opt-in and (where it
removes nothing) non-destructive:

| mechanism | recurrence of… | → dedup of… | module |
|---|---|---|---|
| **Proximity / identity collision** | points that sit too close / restate each other | redundant **nodes** (fission or merge) | `collision.py` |
| **Lateral collision** | facts co-retrieved under *diverse* queries | wasted **kNN slots** (one keeper, members expanded on-target) | `lateral.py` |
| **Spike–Chasles (reasoning)** | reasoning hops repeated along a path | **path length** (A→B→C→D ⇒ A→M→D) | `spike.py` |
| **Spike–Chasles (workflows)** | action steps repeated across walks | **workflow step-count** (⇒ one `tool_workflow` node) | `spike.py` |
| **Memory skills / tools** | actions re-derived for diverse queries | redundant **capability** (one persistent tool node) | `skills.py` |

Two shared notes: every law weights recurrence by **query diversity** (an
event that repeats a seen direction adds ≈ 0; only distant directions
accumulate), and each is **gated** so it stays inert on small clouds where
"redundant" is undefined.

### 5.1 Memory skills — self-built (automated) tools

A capability the system keeps re-deriving crystallizes into a **tool**: a
*normal node* (kind `ACTION`, tagged `tool` + a genre tag
`tool_code`/`tool_command`/`tool_api` + grounding-context tags), scoped by
its keywords. Being a normal node it lives in the same cloud — persisted on
save, found by the walk **recursively**, and subject to lateral collision
and Chasles compression like anything else. The tool set is therefore
self-built, self-pruning, and self-retrieving; nothing about it is a
special case.

Three mechanisms govern automation (`skills.py`):

- **Two triggers.**
  *Recurrence* (`crystallize_skills`) — an action re-derived across enough
  **diverse** queries (the same Poisson floor `λ + √λ`, diversity-weighted)
  earns a persistent tool over time.
  *Eager* (`ensure_tool`) — the "no tool → generate it → proceed" step:
  on first need, with no recurrence wait. Generation is unconstrained, so
  it never blocks; it only ever *grows* the self-built set.
- **Tool-intent metacognition** (`assess_tool_intent`) — reads a `THOUGHT`
  or a generated response and judges whether it implies running code or a
  command underneath, i.e. whether there is a tool to generate at all.
- **Fast-path reuse** (`match_tool`) — an in-scope query returns its
  covering tool via keyword-cosine (threshold `cos`-based, not tuned), so
  the agent reuses the cached capability and **skips a think phase**.

```python
m.skills_enabled = True
m.ensure_tool("search scientific articles about X",
              how="query the article index by topic, rank by relevance")
# 1st call → generates tool_search_articles (ACTION, tag=tool,tool_api…)
# next time a query falls in its keyword scope → match_tool returns it,
#   the think phase is skipped, and the capability is reused verbatim.
```

Because a tool is just an `ACTION` node, the walk (§3) can *chain through*
it like any other point: a self-built capability becomes available to
multi-hop reasoning the moment it crystallizes, and a `tool_workflow`
(a Chasles-compressed chain of actions, §5) is reused as a single optimised
step.

### 5.2 Named skills — theoretical directories of tools

A **skill** is a *named theoretical directory* of tools that solves a
complex task. It is built, retrieved, and grown entirely inside the
manifold — no special store.

**Construction (task mode).** `Memory.build_skill(query, user_id,
session_id)` runs a walk in **task mode**: gold is a *tool fact* as
readily as a *semantic fact* (tools ride the same retrieval), and the
depth gate is no longer the QA σ-stop but *"have we gathered enough tools
to elaborate the complete workflow?"* (`enough_tools_for_workflow`, an LLM
check per depth, fail-closed). The output is a **skill folder** as nested
JSON — keys are theoretical file names — with a generated `name`.

**Ingestion (`Memory.ingest_skill`).** The skill JSON is re-indexed as
ordinary points: each dir/file becomes a `FACT` tagged
`skill · tool · name:<name> · user:<id> · session:<id> · date:<…>`
(`skill_dir`/`skill_file`). The directory **topology** is encoded twice in
agreement — edge-free lineage (`parents` + `apply_pull` clustering) and
typed `ref:skill:<name>:path:` / `:parent:` tokens (the same device as
`ref:date:*`) — so the structure is itself retrievable.

**Scoped retrieval (double query).** Recall pre-filters the **section**
(`session:`/`user:`/`name:` tags) and RRF-merges it with the free semantic
query (`section_filter`): a user's own this-session skills get a rank boost
without excluding the global cloud. The MCP layer caches the live skill
JSON per `(user, session)`.

**Latent distiller (in `sleep()`).** Every solved task is logged with its
**resolution path** (`record_resolution`: query, the walked point-ids, the
output). The next `sleep()` replays the ledger (`distill_skills`) and
crystallizes, per resolution, a tool `ACTION` node whose `parents` are the
semantic facts/thoughts/actions that **explicate** it, co-located by
`apply_pull`. The next time the task recurs the walk retrieves the tool
**directly** and chains through it — no forced metacognition. Deterministic
and idempotent (a consumed-cursor over the ledger).

**Capture-on-generation (chat side).** Whenever the assistant *generates
code*, `Memory.capture_code_tool(code, context, …)` (MCP
`capture_code_tool`) asks whether it is a **reusable tool**
(`assess_tool_intent`: surface markers + LLM adjudication) and, on a yes,
**feeds it into the RAG** — an executable `ACTION` node (`exec_spec`)
tagged `tool·skill·code·name:<def-name>·user:·session:·date:`, retrieved by
the walk like any fact and reused via `match_tool`. A snippet judged purely
illustrative is dropped (returns `None`, chat never blocked).

**`push_code` — evaluate & route.** The richer entry point
(`Memory.push_code`, MCP `push_code`) the client calls for *every* code
block. The server **evaluates** it and routes:
- **project code** → a semantic FACT **documentation** node tagged
  `doc·project:<…>·branch:<…>·github:<user>·user:·session:·date:` that
  teaches about the project;
- **tool-able code** → a **metacognitive rewrite** (`_rewrite_as_tool`,
  exposing `def run(args)`) re-indexed as an executable tool;
- **both** → both, kept in **bidirectional `ref:` filiation**
  (`ref:tool:<id>` on the doc, `ref:doc:<id>` on the tool, plus parents and
  co-location) so the tool remembers the project it came from and the
  project doc points at the derived tool.

### 5.3 Episodic conversation index

Every message of a session — **both** the user's and the agent's — is fed
in **continuously**, with its **timestamp preserved**
(`Memory.ingest_message(content, role, user_id, session_id, timestamp)`,
MCP `ingest_message`). Each becomes a FACT anchored `[<ts>] <role>: …`,
tagged `episode·message·role:<user|agent>·user:·session:·ts:·date:` and
chained in order (`sequence_prev`) per `(user, session)`. The timestamp
anchor also feeds the deterministic temporal resolution (§4.2).

Indexation is **async and non-blocking**: a daemon worker drains a queue in
the background, so feeding a message never delays a concurrent search
(`flush_index()` / `block=True` force synchrony for tests and shutdown).
The server's MCP `instructions` make this a standing client obligation —
index every message, push every code block.

---

## 6. Evaluation

We evaluate on **LoCoMo-10** (Maharana et al.) — ten ~300-turn
conversations, five question categories (multi-hop, temporal, inference,
single-hop, adversarial), balanced one-per-category sampling. The answerer
is Claude Haiku with **no dataset-specific answer vocabulary** in the
prompt.

| metric | prior (V9) | current (V11) |
|---|---|---|
| evidence **agent-recall** | 0.643 | **0.849** |
| static Recall@7 | 0.306 | 0.306 |
| answer token-F1 | 0.676 | 0.63 – 0.68 |
| worst-case input tokens / QA | — | **19k** (was 118k) |

The walk surfaces ≈ 2.7× the gold evidence of a single-pass retriever.
Much of the residual answer error is **metric artefact**: token-F1 counts
"progressive" ≠ "liberal" as a miss, and some gold turns carry only an
attached image (no text to retrieve), yet the walk answers correctly from
the neighbouring dated turn. The per-version history (V5 baseline → V11)
and the methodology are in
[`benchmarks/locomo/README.md`](benchmarks/locomo/README.md).

A note on robustness: the single largest empirical gain this line of work
produced was not an algorithm but the removal of a **silent failure** — a
keyword extractor that cached an empty result after one transient API
error, dropping every subsequent query to a mismatched embedding space.
Retry-and-never-cache-empty discipline now covers every LLM call.

---

## 7. Usage

### 7.1 In-process library

```python
from metacog import Memory

m = Memory(storage_path="memory.pkl")    # or omit for in-memory
m.ingest("dr sarah lives in berkeley", kind="FACT")
m.ingest("dr sarah works as a counselor", kind="FACT")

m.process_turn("who is dr sarah", speaker="user")
print(m.reason("where does dr sarah live?")["final_answer"])
print(m.audit())                          # zero laundering

m.lateral_enabled = True                  # co-retrieval consolidation
m.skills_enabled  = True                  # tool crystallization
m.sleep()                                 # geometric + lateral collision pass
```

### 7.2 MCP service

Every operation is exposed as an MCP service (`metacog/mcp_server.py`), so
an external agent (e.g. Claude Code) drives the same memory the library
uses. Register it:

```json
{ "mcpServers": { "metacog": {
    "command": "metacog-mcp",
    "args": ["--storage", "~/.metacog/state.pkl"] } } }
```
```bash
uv sync && uv run metacog-mcp --storage ~/.metacog/state.pkl
```

**Transports.** stdio is the default (for Claude Code). For live-streaming
HTTP clients — the keepup message that rewrites itself (§3.1) — run the
server over **SSE** or streamable-http:

```bash
uv run metacog-mcp --transport sse --host 127.0.0.1 --port 8765
#   (or --transport streamable-http ; or METACOG_TRANSPORT=sse)
```

Tools appear as `mcp__metacog__*`:

```
# memory I/O
ingest              add a FACT / THOUGHT / ACTION
observe             apply an Observation on existing point(s)
process_turn        record a conversation turn (detectors fire)
retrieve            top-k hybrid retrieval (RRF)

# the walk  ── depth is decided by the walk, not the caller ──
walk_start          run a COMPLETE uncertainty-governed walk for a query
                    (loops to its own σ-termination; returns the bounded
                     composable evidence + reasoning chain). user_id/
                     session_id add the double-query section boost. The
                     agent does BREADTH pivots — re-issue walk_start with
                     new vocabulary; it never micro-drives depth.
walk_keepup         KEEPUP (§3.1): the snapshot trajectory — a provisional
                    answer re-written each stage until validated. Over SSE,
                    a message that rewrites itself until done=True.
walk_next           DEPRECATED — walk_start now runs to completion.

# reasoning & consolidation
reason              full reasoning trajectory until a usable answer
sleep               geometric + lateral collapse, and the latent skill distiller

# skills & self-built tools (§5)
ingest_skill        re-index a named skill-JSON directory tree
build_skill         task-mode walk → synthesise + ingest a named skill
get_session_skill   the skill JSON cached for this (user, session)
ingest_message      EPISODIC: index a message (user/agent), async, timestamped
push_code           evaluate & route generated code → project doc and/or tool
capture_code_tool   feed generated code into the RAG if it is a reusable tool
match_tool          fast-path: does a generated tool already cover this query?
ensure_tool         get a tool, generating it if absent ("no tool → make it")
crystallize_skills  fold recurring actions into persistent tool nodes
list_tools_learned  list the self-built tool set

# introspection & perspective
inspect             dump a point's full state (keywords, tags, σ, spike count)
audit               verify no laundering (Corollary 5)
stats               system overview
declare_observator · detect_polarized · spawn_observators · route
list_communities    Level-1 community / observator detection
save                persist to disk
```

The crucial semantic difference from a conventional retrieval tool:
**`walk_start` owns the depth decision.** A caller cannot ask for "one more
hop"; the walk runs until its own uncertainty propagation says stop (§4).
The agent's only lever is *breadth* — start another walk aimed at a
different facet. This is what keeps the σ-termination, the floor, and the
coverage-stop authoritative at inference time instead of being overridden
by an eager caller.

### 7.3 Tests & benchmark

```bash
uv run pytest tests/ -q                                  # 330 tests
uv run python -m benchmarks.locomo.eval \
    --answerer meta --samples 5 --per-category 1 --encoder semantic
```

---

## References

1. Romanchuk & Bondar. *Semantic Laundering in AI Agent Architectures.*
   arXiv:2601.08333.
2. Zhu et al. *HeLa-Mem.* arXiv:2604.16839. (We keep the reflective spirit;
   we remove the hyperparameters and the materialized edges.)
3. Gao et al. *Precise Zero-Shot Dense Retrieval without Relevance Labels*
   (HyDE), 2022.
4. Yu et al. *Chain-of-Note*, EMNLP 2024.
5. Maharana et al. *LoCoMo: Evaluating Very Long-Term Conversational
   Memory.*
6. BIPM. *Guide to the Expression of Uncertainty in Measurement* (GUM),
   1995.
7. Cormack et al. *Reciprocal Rank Fusion*, 2009.
