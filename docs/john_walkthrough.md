# John's financial status — a step-by-step walkthrough

> A real cat3 (open-ended inference) case from LoCoMo `conv-41`, driven
> live in `benchmarks/locomo/debug_qa.py` against the full 663-turn
> conversation memory. This document is the trace that backs the
> use-case diagram in the project README.

## The question and the gold

| | |
| --- | --- |
| **Question** | *"What might John's financial status be?"* |
| **Gold answer** | *Middle-class or wealthy* |
| **Gold evidence id** | `D5:5` |
| **Gold turn** | *[28 January 2023] John: "It's definitely isn't, Maria. **My kids have so much and others don't.** We really need to do something about it."* |
| **Category** | 3 — open-ended inference (the answer is the label the evidence implies, not a quote) |

### The pathology in one line

The question's vocabulary (`financial`, `money`, `wealth`) shares **no
content words** with the evidence (`kids`, `have`, `so much`). A literal
retrieval can therefore never reach `D5:5` — the answer is in a different
linguistic register. The conversation is also dominated by John's charity
/ community activity, so naive queries pull his caritative persona
(`D6:6`, `D29:4`, `D16:3`) and confidently mislead.

### The lineage neighbourhood of the gold

```
prev D5:3  John: Yup, education is essential for a successful society…
prev D5:4  Maria: Yeah, John. Our kids are our future; they should have…
 ★   D5:5  John: My kids have so much and others don't. We really need…   ← gold
next D5:6  Maria: Sure, it's not right that some kids get all they need…
next D5:7  John: Wow, Maria! That's really making a big impact…
```

The bridge later targets exactly this two-turn radius.

---

## Round 0 — `presearch` (literal phrasings, fails on purpose)

> Three reformulations in the **question's** register, top-5 per query,
> deterministic embedding retrieval, no LLM cost on the agent side.

```
qa> presearch
       John financial status money income
     | John wealthy rich poor struggling
     | John spending habits possessions lifestyle
```

Top hits across the three queries (no `✓` = `D5:5` absent):

| query | rank 1 | rank 2 | rank 3 |
| --- | --- | --- | --- |
| `financial status money income` | D6:6 (0.35) | D6:14 (0.34) | D29:4 (0.32) |
| `wealthy rich poor struggling` | D6:9 (0.44) | D6:6 (0.41) | D2:6 (0.39) |
| `spending habits possessions lifestyle` | D3:5 (0.36) | D22:11 (0.36) | D22:9 (0.35) |

**Recall = 0.000**. Every literal-vocabulary path lands on John's
charity / community turns. The agent reads this and **switches register**.

---

## Round 1 — `clue_search` (evidence-register, generated server-side)

> Single LLM call (~5 s), brainstorms **N concrete first-person chat lines
> that would each be EVIDENCE for a different plausible answer**,
> spanning the answer space (well-off ↔ struggling, direct ↔ indirect).
> Then top-3 retrieval per clue and a **lineage bridge** (gap-fill ±3
> turns on each conversation chain).

### Generated clues (6 of them)

```
· Just put a down payment on a lake house, finally got approved.
· My car's been in the shop for two weeks and I can't afford to pick it up.
· Took the kids to Disney last month, they're still talking about it nonstop.
· Been eating ramen most nights because rent went up again.
· My parents keep asking when I'm going to help them with their mortgage,
  but honestly I'm barely keeping my head above water.
· Got promoted to senior manager and the stock options are looking sweet.
```

Each clue is a deliberately **concrete chat utterance**, not an abstract
restatement of the question.

### Per-clue retrieval (top-3 each)

| clue | hits (3 each) |
| --- | --- |
| *Lake house down payment* | D14:17, D21:8, D1:10 |
| *Car in the shop* | **D11:1**, D11:3, D2:1 |
| *Disney with the kids* | **D5:6**, D8:7, D24:9 |
| *Ramen / rent up* | D10:9, D2:23, D13:18 |
| *Parents asking for help* | D6:9, **D5:6**, D1:10 |
| *Promotion / stock options* | D19:12, D19:8, D19:7 |

Notice **D5:6** (the next turn after the gold) and **D11:1** (a "had a
rough patch" turn near the gold's chain) — neighbours of the gold get
hit even though `D5:5` itself doesn't.

### Lineage bridge — fill the gaps between hits

> For every conversation chain the clue hits touched, the bridge
> **fills the contiguous turns between the lowest and highest position**
> (bounded so a sparse pair across a whole session doesn't drag the
> session in), plus ±3 turns of edge expansion. Deterministic, no LLM.

```
40 bridge neighbours offered, including:
   D5:5 ★GOLD      ← in-between turn between D5:6 (Disney clue) and surrounding hits
   D5:7            ← edge of D5:6
   D11:2           ← in-between D11:1 and D11:3
   D19:9 / D19:10 / D19:11  ← in-between the three promotion-clue hits
   D8:6 / D8:8     ← edges of D8:7
   D13:17 / D13:19 ← edges of D13:18
   D1:9 / D1:11    ← edges of D1:10
```

The bridge **surfaces `D5:5` even though no clue retrieved it directly**.
Its `merged` field now contains the gold; its `fact_ids_cumulative`
includes it; the bench's recall metric credits it.

**Recall after round 1 = 1.000** (was 0.000 after round 0).

---

## Round 2 — `walk_start` (the agent picks up the clue vocabulary)

> The agent's next round reads round 1's results and seeds a walk with
> a query in the **answer's register**, not the question's:

```
walk_start(query = "John kids have so much possessions resources family wealth")
```

> A walk is a multi-stage Chain-of-Note over the manifold: at each stage
> it ranks the top-k facts under the current σ, generates a `THOUGHT`
> from them, may pick an `ACTION` to traverse, and labels each retrieved
> fact `relevant | contradicts | partial`. Depth is governed by σ
> propagation (no fixed cap).

| stage | retrieved facts (top-7) | thought (compressed) |
| --- | --- | --- |
| 0 | D5:4, **D5:5**, D17:9, D10:13, D10:15, D6:9, … | "John speaks of his kids having a lot — comparison with others" |
| 1 | D5:6, D29:14, D24:16, D22:7, D3:5, … | "the same theme: abundance vs. need, framing of giving" |
| 2 | D20:4, D15:16, D31:9, D32:17, … | "consistent across sessions: comfort + civic engagement" |
| … | (9 stages total) | drifted = **False**, σ_path = 0.63 |

**Outcome:**

- `relevant_collected` (15) includes **`D5:5` labelled `relevant`** — it
  is now in the *composition* set, not just retrieved.
- `fact_ids_cumulative` = 68 ids (incl. the bridge neighbours).
- `drifted = False` (the walk stayed on topic).
- Walk cost: ~9 stages × ~2 LLM calls (CoN + thought) ≈ **18 sequential calls**.

---

## Round 3 — `final_answer` (inference, not extraction)

> The agent calls `final_answer`. The forced-final instruction is
> **conditional on question type**: `_is_inference_q` matches
> `might / would / could / likely / status / leaning`, and the agent is
> told to *infer a canonical label*, **not** copy the evidence verbatim.

| run | answer | F1 |
| --- | --- | --- |
| literal-extract (old default) | *"Likely yes, kids have so much"* | 0.0 |
| inference hint, take 1 | *"Likely modest or struggling"* | 0.0 |
| `answer wealthy` (manual sanity probe) | *wealthy* | **0.500** |

The composition is still the brittle step on `john` — the conversation
is genuinely ambiguous in surface (John talks at length about unemployment
in his community), so even after the gold is in the compose set, the
canonical label can flip. This is the documented residual: `john` is the
**no_bridge ∩ ambiguous** tail of cat3. On 25 % of cat3 (`neighbor_bridge`
class — e.g. *caroline*) the same pipeline produces a clean recall 1.0
**and** a clean F1.

---

## What the diagram in the README shows

The three columns mirror the three logical tracks:

1. **Indexed substrate (left)** — the typed memory points the walk
   composes over: `FACT`, `ACTION`, `THOUGHT`, with the sequence-linked
   conversation chains.
2. **Tools / decisions (centre)** — the round-by-round tool calls
   (`presearch`, `clue_search`, `walk_start`, `final_answer`) with one
   line of intent each.
3. **Generation track (right)** — what each tool produces that *did
   not exist before this call*: clue utterances, bridge neighbours,
   the chain-of-note `THOUGHT`s, the final inferred label.

The horizontal separators are the agent rounds. Three small dots between
edges mean *plus N more of the same kind* — the diagram is a paper-style
résumé, not a 1:1 audit log.
