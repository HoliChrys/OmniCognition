# benchmarks/storage_baseline — the storage oracle (OO·V0·M2, Linear TAC2-1146)

## Purpose

Prove that a change of STORAGE (pickle → another backend, OO·V2·M1–M4)
changes nothing the memory serves: same ids, same scores, on the same
contexts and the same frozen queries. Also: a read-only inventory of served
stores (OO0-4) and the latency baseline of `recall` / `remember` / `save()`
(OO0-6). Read-only consumer of `metacog`; never imported by it.

## Ownership

Owns `inventory.py`, `corpus.py` + `fixtures/corpus.json`, `queries.py` +
`queries/`, `build.py`, `run.py`, `compare.py`, `latency.py`, `harness.py`,
`neo4j.py` (OO·V2·M3: `migrate` copies a build's pickle stores into a TEST
namespace of a tachikoma API through `Neo4jBackend`; `factory` is the
`--memory-factory` that serves them) and the committed `baselines/`.

## Local Contracts

- **No LLM, no network.** `harness.LLMGuard` counts every `ClaudeLLM.generate`
  (answered `""`); a non-zero count fails build, run and latency.
  `HF_HUB_OFFLINE=1`. The encoder is the PRODUCTION one (fastembed
  `paraphrase-multilingual-MiniLM-L12-v2`, 384) and so is the reranker
  (jina v2) — `make_encoder("fastembed")` / `make_reranker("fastembed")`
  raise instead of falling back. Both models must be in the fastembed cache.
- **Never write a served store.** `inventory.py` unpickles on a read-only fd
  (never `Memory(...)`: its construction opens the journal read-write) and
  opens the journal `mode=ro`; it fingerprints every file before/after and
  refuses an `--out` under the root. `run.py` / `latency.py` work on a fresh
  temporary COPY and fail if the source tree's digests moved.
- **Frozen inputs.** `fixtures/corpus.json` (chunks of repo docs, with sha256 and
  commit) and `queries/<ctx>.v<N>.json` (50 per context: 25 window, 10
  section, 5 held-out, 10 hand-written incl. 2 off-topic) are never edited
  after a baseline: a new set is a new version / file. No LoCoMo data ships
  in the repo, hence the repo-prose corpus.
- **Reproducible build.** `build.py` writes only into an empty `--work`;
  `uuid.uuid4` is seeded (`SEED = 1146`), so two builds give byte-identical
  pickles. Contexts: `oracle.small` (37 points), `oracle.medium` (275:
  FACT/THOUGHT/ACTION/EVENT, sequences, observators, forgets),
  `oracle.narrow` (112 + notes + the narrow account `oracle-member`, 18).
- **Injection.** Every entry point takes `--memory-factory module:callable`,
  `(storage_root, notes_root) -> memory` serving the gate's contract
  (`ContextualMemory` by default). A new backend is compared by building or
  migrating the SAME data, running with its factory, and `compare.py`.
- **Neo4j** (`neo4j.py`): the stores are migrated into a FRESH namespace for
  each run (a run advances the clocks; a namespace is never reused); a narrow
  account's store is written and read under that account's own token (dev
  login of the test API, granted on its context). Never against a served API.
- **What is compared.** Per (view, query): `retrieve_plain`
  (`retrieve(k=7, rerank=False)`), `retrieve` (hybrid + lineage + spreading,
  no reranker) and `recall` (the MCP tool, reranker + floor + gap). Ids,
  kinds, states, content digests, `score` / `rerank_score` bit-for-bit
  (`--atol` only for a backend known to round, and then said). Wall-clock
  fields and `retrieval_id` are not part of the oracle. Store fingerprints
  (content / vector digests) are reported apart: a content digest that
  moved is a different build, not a backend difference.

- **`baselines/v1/`**: `build_manifest.json` (fingerprints of the four
  stores) and `oracle_run.json` (the reference run: 200 (view, query) pairs
  × 3 probes = 600 comparisons) and `latency.json` (`latency.py`: p50/p95 of
  `recall`, `retrieve` without reranker, `remember` incl. its `save()`, and
  `save()` alone), all from the orchestration container of
  2026-11-03 (4 CPU, onnxruntime 1.31.1, fastembed 0.9.1). A new baseline
  is a new `v<N>/`, never a rewrite.

## Work Guidance

- Order matters: a recall advances the logical clock (`_t_clock`), and the
  effective vector depends on it. Queries run in the frozen order on a fresh
  copy; never reuse a copy between two runs.
- Bit-for-bit holds on ONE machine and ONE set of wheels (onnxruntime picks
  CPU kernels at run time). `baselines/` is the reference measured in the
  container that wrote it; before comparing a backend elsewhere (`tk-omni`),
  re-run the pickle baseline THERE first and compare against that.
- For copied SERVED contexts (OO0-3, on `tk-omni`): `queries.py
  --from-journal <copy>/memory.pkl.journal.db --ctx <ctx> --hand <10.json>`,
  then `run.py --no-manifest` against a root holding those copies.

## Verification

- Determinism witness: two `run.py` on the same `--work` → `compare.py`
  exits 0 (IDENTIQUE).
- Positive witness: `run.py --drop-from-baseline <first run>` → `compare.py`
  exits 1 (DIFFÉRENT) and names the views and queries.
- Read-only witness: `inventory.py` rc 0 and `inchangé = oui` on every store;
  an `--out` under the root is refused (rc 2).

## Child DOX Index

No children.
