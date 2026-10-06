"""No prompt over the context window is ever sent (TAC-405, D3 bench TAC-394:
`sleep` → `resolve_collision` → `extract_common` sent 15 × 358 094 tokens to a
200 000-token window). `extract_common` chunks and merges under the budget;
`generate` refuses what still does not fit. No network: the client is faked
and records every prompt it receives."""
from __future__ import annotations

from types import SimpleNamespace

from metacog.collision import resolve_collision
from metacog.epistemic import Point
from metacog.llm import ClaudeLLM


class _Recorder:
    """Fake Anthropic client: records each user prompt, answers `reply`."""

    def __init__(self, reply="they all mention the shared fact"):
        self.prompts = []
        self.reply = reply
        self.messages = self

    def create(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.reply)],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0))


def _llm(window=200_000):
    llm = ClaudeLLM(context_window=window)
    llm._client = _Recorder()
    return llm


def _over_limit_texts():
    # ~1.5 MB of passages: several times the 200k-token window, as in D3.
    return [f"passage {i} shares the fact. " + "lorem ipsum " * 12_000
            for i in range(12)]


def test_extract_common_never_exceeds_the_budget():
    llm = _llm()
    texts = _over_limit_texts()
    assert sum(len(t.encode()) for t in texts) > 3 * llm.context_window
    assert llm.extract_common(texts) == "they all mention the shared fact"
    budget = llm.prompt_budget(64)
    prompts = llm._client.prompts
    assert len(prompts) > 1                       # chunked, then merged
    assert all(len(p.encode()) <= budget for p in prompts)
    # every passage reached a chunk call (nothing dropped)
    for i in range(len(texts)):
        assert any(f"passage {i} shares" in p for p in prompts)
    assert llm.llm_errors == 0


def test_a_single_passage_over_the_budget_is_cut_not_sent_whole():
    llm = _llm(window=5_000)
    huge = "x" * 50_000
    assert llm.extract_common([huge, "short"]) != ""
    assert all(len(p.encode()) <= llm.prompt_budget(64)
               for p in llm._client.prompts)
    assert llm.llm_errors == 0


def test_an_empty_partial_common_means_nothing_shared():
    llm = _llm()
    llm._client.reply = ""
    assert llm.extract_common(_over_limit_texts()) == ""
    assert len(llm._client.prompts) == 1          # stops at the first chunk


def test_generate_refuses_an_over_budget_prompt_without_sending_it():
    llm = _llm(window=1_000)
    assert llm.generate("y" * 2_000) == ""
    assert llm._client.prompts == []
    assert llm.llm_errors == 1 and llm.last_error.startswith("PromptTooLong")
    # remove_overlap keeps its fallback (the full text, untrimmed)
    assert llm.remove_overlap("z" * 2_000, "common") == "z" * 2_000
    assert llm._client.prompts == []


def test_resolve_collision_over_limit_stays_under_the_budget():
    llm = _llm()
    enc = SimpleNamespace(encode=lambda text: (1.0, 0.0))
    pts = [Point(id=f"p{i}", content=t, embedding_orig=(1.0, 0.0))
           for i, t in enumerate(_over_limit_texts())]
    res = resolve_collision(pts, llm, enc, t_now=1.0)
    assert res is not None and res.child.content
    assert llm.llm_errors == 0
    assert all(len(p.encode()) <= llm.prompt_budget(64)
               for p in llm._client.prompts)
