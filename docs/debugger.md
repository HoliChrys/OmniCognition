# `debug_qa` — the step-by-step QA debugger

`benchmarks/locomo/debug_qa.py` is an interactive REPL that drives the
LoCoMo retrieval / answer tools **one at a time** against a single question,
showing — live — which tool fires, what it returns, and whether the gold
evidence has surfaced yet. It exists so you can see exactly *where* a hard
question (typically a cat3 inference) goes wrong and test a fix in seconds,
instead of launching a ~1-hour full benchmark to discover nothing moved.

Every bug fixed in the drift-resistance / clustering line of work was found
here: the *"January 2023"* answer-clobber, the cat3 over-enumeration, the
Slice-C semantic-bias regression, the yes/no abstraction leak, the verbose
single-hop F1 loss. The debugger is the microscope.

---

## Why it is fast

The conversation memory is built **once and cached to `/tmp`** (embedding
600+ turns is the expensive step). The first run on a probe ingests and
pickles; every later run loads instantly. The cheap tools (`presearch`,
`retrieve`, `show`, `gold`, `clues`) then cost **zero or one** LLM call, so
you can iterate on retrieval/phrasing in seconds. Only `walk` / `auto` /
`step` pay the full multi-stage walk.

> The cache lives at `/tmp/locomo_qa_<sample>_<nsess>.pkl`. It is wiped when
> the container restarts; the first run after that rebuilds it. Pass
> `--rebuild` to force a fresh build.

---

## Running it

```bash
# a built-in probe (john = conv-41 cat3, caroline = conv-26 cat3)
python -m benchmarks.locomo.debug_qa --probe john
python -m benchmarks.locomo.debug_qa --probe caroline

# any question over any conversation
python -m benchmarks.locomo.debug_qa --sample conv-26 \
    --question "What did Caroline research?" \
    --gold D2:8 --answer "Adoption agencies" --category 1

# non-interactive: a scripted sequence (commands separated by ';')
python -m benchmarks.locomo.debug_qa --probe john \
    --script "gold; presearch John financial status | John kids have so much; clues; recall"

# or a script file (one command per line) — robust to shell quoting
python -m benchmarks.locomo.debug_qa --probe john --script-file scenario.txt

# force a fresh memory build
python -m benchmarks.locomo.debug_qa --probe john --rebuild
```

### CLI options

| flag | meaning |
| --- | --- |
| `--probe {john,caroline}` | a built-in (sample + question + gold + category) |
| `--sample conv-NN` | conversation id (default `conv-41`) |
| `--question "…"` | the question to debug |
| `--gold D1:9,D1:11` | comma-separated gold dia-ids (for recall tracking) |
| `--answer "…"` | the gold answer text (for `answer` token-F1 scoring) |
| `--category N` | LoCoMo category 1–5 (default 3) |
| `--nsess N` | ingest only the first N sessions (0 = whole conversation) |
| `--script "a; b; c"` | run commands then exit |
| `--script-file FILE` | run commands from a file (one per line / `;`) then exit |
| `--rebuild` | re-embed the memory cache from scratch |

---

## What you see — the gold tracking

Every retrieval command flags gold evidence ids with **`✓GOLD`**, and the
session keeps a **cumulative recall** across all the ids any tool has
surfaced so far:

```
cumulative recall = 1.000  hits=['D5:5']  gold=['D5:5']  (seen 67 ids)
```

This separates the two failure modes cleanly: **recall < 1** ⇒ a *retrieval*
problem (the gold was never surfaced); **recall = 1 but a wrong answer** ⇒ a
*composition / format* problem (the gold was found but the answer is wrong
or too verbose).

---

## Command reference

### Cheap — no walk (0–1 LLM call, instant-ish)

| command | what it does |
| --- | --- |
| `q` | show the question / gold answer / category |
| `gold` | show the gold evidence ids + their turn text |
| `show <id>` | a turn's text + its ±2 `sequence_prev`/next neighbours |
| `tools` | list the MCP tools the agent can call |
| `presearch <q1> \| <q2> \| …` | batch recon — top-k nearest per query, gold flagged, **no walk** |
| `retrieve <query>` | raw top-k hybrid retrieval |
| `clues` | run `clue_search` — generated evidence-register clue lines, per-clue hits, the lineage-bridge neighbours, and whether the gold surfaced (1 LLM call) |
| `recall` | cumulative gold recall across the session |
| `answer <text>` | score token-F1 of `<text>` vs the gold answer |
| `reset` | clear the accumulated retrieved-id set |

### Expensive — full walk / agent (many sequential LLM calls)

| command | what it does |
| --- | --- |
| `walk <query>` | one full `walk_start`: stages, drift, `relevant_collected`, `neighbor_possibilities`, `fact_ids_cumulative` (gold flagged) |
| `scoped <q> :: t1,t2 [kb]` | tag-scoped walk (`kb` also queries the knowledge base) |
| `auto` | run the **full autonomous agent** end-to-end, tracing each round, and score F1 |

### Driving the algorithm vs. steering by hand

| command | what it does |
| --- | --- |
| `step` | let the **algo** take the next step — one autonomous agent round: the model picks the next tool from the live conversation, it runs, and execution pauses for your inspection (same client / system prompt / tool surface as the real agent) |
| `say <text>` | inject a user nudge into the live conversation |
| `checkpoint [name]` / `cp` | snapshot the conversation state, to branch from |
| `restore [name]` | rewind to a checkpoint and try a different path |
| `msgs` | list the live conversation messages |

Manual tool calls (`presearch` / `walk` / `scoped` / `retrieve` / `clues`)
are **recorded into the live conversation**, so you can steer by hand and
then `step` to let the algo continue from where you left it.
`checkpoint`/`restore` let you replay a step and compare scenarios — the
core loop for "what if the agent had walked *this* query instead?".

---

## Worked examples

**Is the gold reachable at all, and with which phrasing?** (cheap, instant)

```
qa> gold
qa> presearch John financial status money | John kids have so much
```
On `john` this immediately shows the literal phrasing misses `D5:5` while
*"John kids have so much"* ranks it #4 — i.e. the answer lives in the
*answer's* vocabulary, not the question's.

**Does the clue bridge surface the gold?** (1 LLM call)

```
qa> clues
```
Shows the generated clue lines, which turns each retrieves, the lineage
bridge neighbours, and `gold∈merged`.

**Branch a scenario without re-running everything**

```
qa> step              # let the algo pick + run the next tool
qa> cp before_walk    # snapshot
qa> walk John kids have so much abundance   # steer by hand
qa> recall
qa> restore before_walk                      # rewind, try another path
```

**Full end-to-end check + score**

```
qa> auto
```

---

## How it is used in this repo

The debugger is the recommended first move for any LoCoMo regression —
attribute the gap to *indexation*, *retrieval*, *the walk*, or *answer
composition* before paying for a full run. The targeted-debugger philosophy
is summarised in [`benchmarks/locomo/README.md`](../benchmarks/locomo/README.md);
the deepest worked trace (the cat3 `john` cascade through presearch →
clue_search → walk → inference synthesis) is in
[`john_walkthrough.md`](john_walkthrough.md).
