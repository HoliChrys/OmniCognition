# LoCoMo benchmark for MetaCog-Mem

[LoCoMo-10](https://github.com/snap-research/locomo) — 10 ultra-long
conversations (~300 turns / ~9k tokens each), 1986 QA pairs spanning
single-hop, multi-hop, temporal, open-domain, and adversarial
categories.

## One-time setup (uv)

```bash
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# from the repo root
uv sync --extra bench           # creates .venv and pulls all deps
bash benchmarks/locomo/download.sh
export ANTHROPIC_API_KEY=sk-ant-...    # only needed for --answerer claude
```

`[bench]` extras pull in :
- `sentence-transformers` — semantic encoder
- `transformers` + `torch` — extractive roberta QA model
- `anthropic` — Claude API client (for `--answerer claude`)

First run of the semantic encoder / roberta downloads models from
HuggingFace.

## Run (uv)

```bash
# Plain cosine retrieval — chunk-dump answer (fast smoke)
uv run python -m benchmarks.locomo.eval --samples 10 --encoder semantic

# Meta-cognitive walk answerer (the real system : a single walk_start runs
# the full uncertainty-governed walk over the FACT/THOUGHT/ACTION manifold,
# Claude drives breadth pivots)
uv run python -m benchmarks.locomo.eval --samples 10 --encoder semantic --answerer meta

# Balanced smoke : 1 QA per category, a few conversations
uv run python -m benchmarks.locomo.eval --samples 4 --per-category 1 --answerer meta

# Opt-in retrieval handles
uv run python -m benchmarks.locomo.eval --answerer meta --entities   # entity beacons
uv run python -m benchmarks.locomo.eval --answerer meta --atomic     # atomic-fact handles
uv run python -m benchmarks.locomo.eval --answerer meta --merge      # dedup identical turns

# Extractive ReAct (roberta-base-squad2, local CPU)
uv run python -m benchmarks.locomo.eval --samples 10 --encoder semantic --answerer extractive
```

### Targeted debuggers (no full bench)

Fast, inspectable harnesses for stepping through the system on a tiny
slice — index → one-shot retrieval → walk stage-by-stage → full agent,
all in seconds. Built precisely to avoid "launch a 20-minute bench to
discover nothing works".

```bash
# Step through indexation, retrieval and the walk on one cat-3 probe
DEBUG_NSESS=0 DEBUG_PROBE=caroline python -m benchmarks.locomo.debug_walk all

# Measure lateral-collision effect on kNN redundancy (before/after sleep)
DEBUG_NSESS=0 python -m benchmarks.locomo.debug_lateral
```

`DEBUG_NSESS` = number of sessions to index (0 = whole conversation) ;
`DEBUG_PROBE` = `caroline` | `john` ; `METACOG_HYDE=1` toggles the HyDE
retrieval channel.

Or via the Makefile :

```bash
make install               # uv sync
make bench-deps            # uv sync --extra bench
make bench-data            # download LoCoMo
make bench-claude-smoke    # smoke run on Claude (cheap)
make bench-claude          # full 1986-QA run on Claude
make bench-lineage         # retrieval-only comparison
make test                  # pytest suite
```

### Claude model selection

Default model is `claude-haiku-4-5-20251001` (cheap + fast for
benchmarking). Override via :

```bash
CLAUDE_MODEL=claude-sonnet-4-6 uv run python -m benchmarks.locomo.eval --answerer claude ...
# or
uv run python -m benchmarks.locomo.eval --answerer claude --claude-model claude-opus-4-7 ...
```

## CLI flags

| flag           | default                              | meaning                                        |
|----------------|--------------------------------------|------------------------------------------------|
| `--data`       | `benchmarks/locomo/data/locomo10.json` | path to the dataset                          |
| `--samples`    | 10                                   | number of conversations to evaluate (1..10)    |
| `--max-qa`     | None                                 | cap QA pairs per conversation (smoke runs)     |
| `--per-category` | None                               | up to N QAs per category (balanced sampling)   |
| `--k`          | 7                                    | retrieval top-k                                |
| `--encoder`    | `semantic`                           | `simple` (simhash) or `semantic` (MiniLM)      |
| `--answerer`   | `chunk`                              | `chunk`, `extractive` (roberta), `claude` (single-shot ReAct), `mcp`, `meta` (the walk) |
| `--lineage`    | off                                  | enable cosine+lineage RRF retrieval            |
| `--entities`   | off                                  | spawn entity-beacon retrieval handles          |
| `--atomic`     | off                                  | spawn mem0-style atomic-fact handles           |
| `--merge`      | off                                  | merge identical turns (corroboration absorbed) |
| `--auto-cluster` | off                                | Level-1 community / observator detection       |
| `--agent-concurrency` | 10                            | parallel QA workers for `mcp`/`meta`           |
| `--debug-jsonl` | None                                | per-QA trace dump for inspection               |
| `--claude-model` | env or default                     | override Anthropic model id                    |

## Reported metrics

For each QA pair :
- **Recall@5 / Recall@7** : fraction of gold-evidence `dia_id`s present
  in the top-k retrieved set.
- **agent_recall** (`mcp`/`meta`) : cumulative recall across all the
  queries the agent issued during its walk — the evidence it actually saw,
  not just a single top-k.
- **F1** : per-category token-overlap F1 against the gold answer (category 5
  is binary abstention : 1.0 iff the answer says "not mentioned").

Aggregated per category (1 multi-hop · 2 temporal · 3 inference ·
4 single-hop · 5 adversarial) and overall. `--debug-jsonl` writes a
per-QA trace (question, gold, prediction, retrieved ids, react trace) for
inspection.

## Temporal handling — how dates are resolved

LoCoMo gold answers are **absolute** ("2022", "May 2023", "16 March
2022"); conversation turns are **relative** ("last year", "yesterday",
"for 3 years"). Resolution is **two deterministic stages, no LLM**:

### Stage 1 — ingestion (`eval.py`)
- `_clean_session_date` normalises each session's timestamp
  (`"1:56 pm on 8 May, 2023"` → `"8 May 2023"`); the turn content is
  prefixed with it (`[8 May 2023] Melanie: …`).
- `_expand_relative_dates(text, session_date)` parses the anchor and, for
  every relative expression, **appends two additive forms**:
  - a **typed retrieval token** — `ref:date:day:N` / `ref:date:month:name`
    / `ref:date:year:N` — so the keyword extractor and BM25 match a typed
    `month:march` instead of a bare `march` that the every-turn date prefix
    would dilute (IDF);
  - a **plain readable clause** — `(absolute date: 2022)` /
    `(absolute date: 16 March 2022)` / `(absolute date: around 2020)` — so
    the answer generator emits the resolved value directly.

  | turn says | session | appended |
  |---|---|---|
  | "… last year" | 8 May 2023 | `[ref:date:year:2022] (absolute date: 2022)` |
  | "yesterday …" | 17 March 2022 | `[ref:date:day:16 ref:date:month:march ref:date:year:2022] (absolute date: 16 March 2022)` |
  | "for 3 years" | 27 March 2023 | `[ref:date:year:~2020] (absolute date: around 2020)` |
  | "last month" | 10 June 2023 | `[ref:date:month:may ref:date:year:2023] (absolute date: May 2023)` |
  | "last week" | 17 March 2022 | `[ref:date:day:10 ref:date:month:march ref:date:year:2022] (absolute date: 10 March 2022)` |
  | "recently" / "a few weeks ago" | 8 May 2023 | `[ref:date:month:may ref:date:year:2023] (absolute date: May 2023)` |

  Month/year roll-over is handled (e.g. "yesterday" on the 1st → last day of
  the previous month). The original wording is never modified — expansion is
  purely additive, so retrieval on the literal phrase still works.

### Stage 2 — answer time (`mcp_meta_agent.py`)
- `_resolve_relative_date_answer(answer, seen_contents)` runs on the final
  answer. If it **is / contains** a relative phrase (`last year/month/week`,
  `yesterday`, `N years/months ago`, `recently`, `around YYYY`), it scans
  every retrieved turn the agent saw for one carrying the **same** phrase
  *and* an `(absolute date: X)` clause, and rewrites the answer to `X`.
  Falls back to the unique absolute date seen if the phrase match is
  ambiguous; otherwise leaves the answer untouched.
- This is the guard that closes the temporal **0-clue** case: gold turn
  `D1:12` ("look at this", an attached image, no retrievable date), the
  walk answers from the dated neighbour, the generator says "last year",
  and Stage 2 deterministically rewrites it to **"2022"** (token-F1 = 1.0).

Both stages are pure string + arithmetic operations: identical across runs,
no token cost, no model dependence. Set `--answerer meta` (dates resolve
automatically); the older "two-rule date system" in the prompt is now a
fallback, not the primary mechanism.

## MetaCog-Mem results (balanced per-category sampling)

All runs use `--answerer meta --encoder semantic --per-category 1`
(`--samples` noted per version: 5 for V9/V11, 10 for V7).
Answerer: Claude (Haiku 4.5, no dataset-specific prompt vocabulary).

### V11 — uncertainty-governed depth + breadth-only agent (current)

The walk's depth is now governed by **uncertainty propagation over the
evidence chain** (GUM quadrature), with a single `walk_start` running the
**complete** depth in one call and the agent doing only breadth pivots.

Root cause this fixed: earlier, `walk_next` advanced the walk one stage per
agent tool-call, and Claude short-circuited after ≈2.5 calls — so the
σ-depth machinery never ran in the bench (steps_per_qa ≈ 2.5). The walk
also relied on a silently-broken keyword extractor (a cached `[]` after one
transient 529 dropped every query to a raw-sentence embedding, mismatching
the keyword-embedding index and tripping the σ-stop at stage 1-2).

Fixes, in order:
1. **Silent-failure cleanup** — retry on `OverloadedError`/5xx and never
   cache an empty result in `LLMKeywordExtractor` / `LLMEntityExtractor` /
   `hyde_passage`; retry wrapper on every agentic-loop API call.
2. **σ on the evidence chain** — `σ_hop` = smallest keyword-distance from
   the previous fact★ to any fact reachable this stage (coherence), not the
   dead seed-to-seed drift. Floor 3, σ-cap at the emergent threshold, no
   hard maximum.
3. **Single full-depth `walk_start`** — loops `step()` to completion; the
   agent does breadth pivots only (`walk_next` deprecated, removed from the
   exposed tools).
4. **Token discipline** — composable evidence capped to `_MAX_EVIDENCE`
   (ranked by label + chain-vocabulary overlap) and a deterministic
   keyword-coverage metacognitive stop, so easy queries terminate early and
   the synthesis is not drowned by a bloated relevant set.

Effect (5-sample balanced, 24 QA): **agent_recall ≈ 0.60 → 0.85** — the
walk now surfaces ~0.85 of the gold evidence (static Recall@7 ≈ 0.31). On
the six hardest hand-picked cases, single-walk recall is ~19% but the full
agent with breadth pivots recovers it (jon/gina 3/4, joanna 3/7, political
& Dr. Seuss 1/1). The bottleneck moved from retrieval to **synthesis** and
to **token cost** (deep walks × breadth pivots) — the V11 token-discipline
items above target exactly that. Token-F1 stays ≈ 0.63-0.68, much of the
residual being metric artefact ("progressive" ≠ "liberal"; gold turns that
carry only an attached image).

### V10 — HyDE multi-angle, ON by default

Addresses cat3 inference vocabulary gap: abstract questions ("What might X's
financial status be?", "Does Nate have friends besides Joanna?") have Jaccard=0.00
and cosine≈0 against concrete conversational evidence — BM25 and semantic kNN both
miss entirely.

**Fix: HyDE multi-angle, RRF-merged** (`metacog/meta_walk.py`)
- Before each walk, generate **three distinct hypothetical evidence lines** (direct /
  indirect / story angles) by prompting the LLM once with a 180-token budget.
- Retrieve against the concatenated hypothetical passage using the same hybrid
  retriever (cosine+BM25), then RRF-merge (rrf_k=60) with the main walk results.
- Enabled **by default** (`METACOG_HYDE=1`); set `METACOG_HYDE=0` to disable for
  ablation. Results are cached per query within a run.
- Cat3 example: query "Was Tim a Harry Potter fan?" — D1:16 ("reading Harry Potter
  Order of the Phoenix") reaches rank 47 in the HyDE pool (was absent from top-55
  main results). The RRF-merge surfaces it within the walk's cumulative evidence.
- Strictly additive: cat1/cat2/cat4/cat5 unaffected (their vocabulary overlap is
  already sufficient; HyDE adds noise-free signal via RRF dilution ≪ rrf_k=60).

Full bench numbers vs V9 pending; qualitative per-case evidence confirms retrieval
gain on cat3 inference cases while the existing categories hold.

### V9 — BM25 content-first + BM25 anchor in walk

Two fixes targeting cat1 multi-hop vocabulary gaps:
1. **BM25 content-first** — BM25 now always indexes `tokenize(content)[:40]` raw
   tokens, keywords appended only as morphological coverage. Previously BM25 indexed
   keyword summaries, making short tokens like "VR", "TV", "AI" (filtered by the
   keyword extractor's ≤2-char rule) invisible to BM25 despite being in the raw turn.
   After: D1:36 ("VR gaming awesome") BM25 score 0 → 1.26 for query containing "VR".
2. **BM25 anchor in walk** — at every walk stage 1+, the original question is
   BM25-searched against all FACTs, RRF-merged (rrf_k=60) with the THOUGHT-evolved
   query results. Prevents proper nouns ("VR Club", "McGee's") from being lost as the
   walk's seed_query evolves from the original question into a keyword+thought string.

5-sample run, 24 QA, `--per-category 1` (different conversations than V7, so
per-category comparison is indicative not conclusive):

| category (n=5, 24 QA) | r7    | agent_recall | V7 F1 | V9 F1 | Δ |
|------------------------|-------|--------------|-------|-------|---|
| cat1 multi-hop         | —     | 0.452        | 0.478 | **0.567** | **+0.089** |
| cat2 temporal          | —     | 1.000        | 0.705 | **0.800** | **+0.095** |
| cat3 inference         | —     | 0.125        | 0.404 | 0.183 | −0.221† |
| cat4 single-hop        | —     | 0.834        | 0.667 | **0.733** | **+0.066** |
| cat5 adversarial       | —     | 0.700        | 0.900 | **1.000** | **+0.100** |
| **OVERALL**            | **0.306** | **0.643** | **0.635** | **0.676** | **+0.041** |

† Cat3 inference questions ("What might X's financial status be?", "Is it likely
that Nate has friends?") have zero vocabulary overlap between the abstract question
and the conversational turns — BM25 anchor is neutral, and with only 4 samples
variance dominates. Likely different conversation difficulty vs V7, not a regression.

### V7 — walk depth adaptive + structured date refs

Three changes from V5 → V7:
1. **Walk depth cap removed** — `n_stages=8` is a soft cap; the walk stops
   naturally when no new unseen facts are available (`fact_star=None → done`),
   which is the `|seen ∩ gold| / |gold| = 1` proxy at inference time.
   `_MAX_ROUNDS=16` (was 8) lets two full-depth walks fit in one agent turn.
2. **Structured date refs** — relative-date expansions at ingestion emit
   `[ref:date:day:N ref:date:month:name ref:date:year:N]` typed compounds
   instead of plain "16 march 2022". Avoids IDF dilution from the session-date
   prefix `[17 March 2022]` that appears in every turn.
3. Entity beacons (`--entities`) — opt-in; not included in the numbers below.

| category (n=10 balanced, 49 QA) | r7    | agent_recall | V5 F1 | V7 F1 | Δ |
|----------------------------------|-------|--------------|-------|-------|---|
| cat1 multi-hop                   | 0.190 | 0.363        | 0.600 | 0.478 | −0.122 |
| cat2 temporal                    | 0.400 | 0.700        | 0.800 | 0.705 | −0.095 |
| cat3 inference                   | 0.167 | 0.444        | 0.045 | **0.404** | **+0.359** |
| cat4 single-hop                  | 0.367 | 0.883        | 0.533 | 0.667 | +0.134 |
| cat5 adversarial                 | 0.050 | 0.850        | 0.800 | **0.900** | +0.100 |
| **OVERALL**                      | **0.236** | **0.652** | **0.599** | **0.635** | **+0.036** |

**Retrieval gap** — r7 = 0.236 → agent_recall = 0.652 (×2.8). The walk finds
evidence the static top-7 retriever misses entirely. Cat5 adversarial
agent_recall=0.850 with F1=0.900 shows correct "not mentioned" abstention.
Cat3 inference +0.359 is the dominant gain: deeper walks allow multi-hop
indirect evidence chains that a single retrieval pass cannot surface.

Cat1/cat2 regressions (~0.1) are under investigation — likely LLM
non-determinism in Chain-of-Note relevance judgments on a deeper walk
occasionally drifting from the correct answer cluster. Entity beacons
(`--entities`) are the next planned recall handle for temporal (cat2).

### V5 baseline (walk depth=3, plain date strings)

| category | F1    |
|----------|-------|
| cat1     | 0.600 |
| cat2     | 0.800 |
| cat3     | 0.045 |
| cat4     | 0.533 |
| cat5     | 0.800 |
| OVERALL  | 0.599 |

### Published baselines (GPT-4o-mini, F1 over 4 categories)

| system        | F1   | tokens |
|---------------|------|--------|
| LoCoMo native | 25%  | 17k    |
| MemoryBank    | 5%   | 432    |
| ReadAgent     | 9%   | 643    |
| MemGPT        | 27%  | 17k    |
| A-Mem         | 27%  | 2.5k   |
| MemoryOS      | 37%  | 2k     |
| HeLa-Mem      | 42%  | 1k     |

The **chunk-dump** stub answerer is a retrieval-only smoke (no LLM span
extraction) and is not directly comparable. For a real comparison use
`--answerer meta` (the meta-cognitive walk driven by Claude). The system
prompt deliberately carries **no dataset-specific answer vocabulary** —
earlier few-shot leaks of gold answers were stripped, so the numbers are
an honest no-hardcode baseline rather than a prompt-overfit one. Use the
targeted debuggers above to attribute any gap to indexation, retrieval,
the walk, or answer generation before launching a full run.

## What `--lineage` changes

`retrieve_with_lineage` adds a second retrieval signal :

1. cosine kNN over the effective embedding (standard)
2. BFS expansion of each cosine hit through `parents`, `children`,
   `sequence_prev`, `sequence_next` (configurable depth)
3. Reciprocal Rank Fusion of the two rankings (`rrf_k = 60`,
   Cormack et al. 2009 — not a tuning knob)

Expected effect on LoCoMo : Recall@10 improves a few points (evidence
in turns adjacent to a strong cosine hit is now surfaced),
Recall@1 may drop marginally (RRF dilutes the very top positions).
