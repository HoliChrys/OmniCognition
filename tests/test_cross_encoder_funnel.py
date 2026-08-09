"""Cross-encoder pre-filter for the oblique judge (mnema's token lever).

When a local cross-encoder is wired into Memory(reranker=...), the funnel scores
(proposition, doc) pairs JOINTLY for zero LLM tokens and gates THREE emergent
bands: HIGH pre-accepted (skip the batch), LOW auto-rejected (NO LLM call at
all), MIDDLE -> the batch judge. Without a reranker the bi-encoder cosine only
ever pre-accepts (never auto-rejects) — the batch still sees every non-accepted
item. Deterministic via a fake reranker + a prompt-capturing fake LLM.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory


class _FakeRerank:
    """Returns fixed per-doc scores, aligned to `docs` (== candidate order)."""
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def rerank(self, query, docs):
        self.calls += 1
        assert len(docs) == len(self.scores)
        return list(self.scores)


class _CapLLM:
    """Captures batch prompts; labels every listed item relevant."""
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_tokens=90):
        self.prompts.append(prompt)
        out = []
        for ln in prompt.splitlines():
            s = ln.strip()
            if s.startswith("[") and "]" in s and s[1:s.index("]")].isdigit():
                out.append(f"{s[1:s.index(']')]}: relevant")
        return "\n".join(out)


def _cands(m):
    ids = ["hi0", "hi1", "mid0", "mid1", "lo0", "lo1"]
    return [m.ingest(f"content about {i}", kind="FACT", id=i) for i in ids]


def test_cross_encoder_gates_three_bands():
    m = Memory(encoder=SimpleEncoder(), llm=_CapLLM(),
               reranker=_FakeRerank([0.9, 0.9, 0.5, 0.5, 0.1, 0.1]))
    cands = _cands(m)
    labels = m.oblique_labels("the stance", cands,
                              proposition="the stance", per_item=False)
    # HIGH band pre-accepted, MIDDLE relevant via batch, LOW auto-rejected.
    assert labels == ["relevant", "relevant", "relevant", "relevant",
                      "irrelevant", "irrelevant"]
    # Exactly ONE batch call, and it saw ONLY the two middle-band items.
    assert len(m.llm.prompts) == 1
    p = m.llm.prompts[0]
    assert "content about mid0" in p and "content about mid1" in p
    assert "content about hi0" not in p        # pre-accepted, never sent
    assert "content about lo0" not in p        # auto-rejected, never sent


def test_low_band_costs_no_llm():
    """Auto-rejected (clear-low) candidates are never handed to the LLM — the
    token saving vs. the bi-encoder path, which would batch them."""
    llm = _CapLLM()
    m = Memory(encoder=SimpleEncoder(), llm=llm,
               reranker=_FakeRerank([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    cands = _cands(m)
    m.oblique_labels("s", cands, proposition="s", per_item=False)
    # 3 high pre-accepted, 3 clear-low auto-rejected -> nothing to batch.
    assert llm.prompts == []


def test_without_reranker_never_auto_rejects():
    """Bi-encoder fallback : blunt, so it only pre-accepts — every non-accepted
    candidate still reaches the batch (no zero-token rejection band)."""
    llm = _CapLLM()
    m = Memory(encoder=SimpleEncoder(), llm=llm)      # no reranker
    cands = _cands(m)
    labels = m.oblique_labels("content about mid0", cands,
                              proposition="content about mid0", per_item=False)
    # Fallback never fabricates an irrelevant without the LLM saying so; the
    # batch labelled all-relevant, so nothing is dropped pre-LLM.
    assert all(x == "relevant" for x in labels)
    assert len(llm.prompts) >= 1                       # the batch still ran
