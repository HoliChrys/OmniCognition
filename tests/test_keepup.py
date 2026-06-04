"""Keepup mode — organic, continuously-rewriting answer generation.

A provisional answer is emitted from the first walk and re-written each
stage until the depth-gate validates ; the last snapshot is the final
answer (no separate final-generation pass). Deterministic (fake LLM).
"""
from metacog.memory import Memory
from metacog.defaults import SimpleEncoder
from metacog.keywords import SimpleKeywordExtractor


class _LLM:
    """Returns a stable bare-value answer for the provisional synthesis,
    and empty for the walk's other generations (kept minimal)."""
    def generate(self, prompt, max_tokens=None):
        if "PROVISIONAL answer" in prompt:
            return "Berkeley"
        return ""


def _mem():
    m = Memory(encoder=SimpleEncoder(), extractor=SimpleKeywordExtractor(),
               llm=_LLM())
    for i in range(8):
        m.ingest(f"sarah lives in berkeley note {i}", kind="FACT")
    return m


def test_keepup_yields_snapshots_and_a_final():
    m = _mem()
    snaps = list(m.answer_keepup("where does sarah live?", n_stages=8))
    assert snaps, "keepup yielded nothing"
    # every snapshot has the streaming contract
    for s in snaps:
        assert set(s) >= {"stage", "answer", "done", "evidence_count",
                          "sigma_path"}
    # exactly one terminal snapshot, and it is the last
    assert snaps[-1]["done"] is True
    assert sum(1 for s in snaps if s["done"]) == 1


def test_keepup_converges_to_a_provisional_answer():
    m = _mem()
    snaps = list(m.answer_keepup("where does sarah live?", n_stages=8))
    # once past the floor an answer is present and stays the validated final
    answered = [s for s in snaps if s["answer"]]
    assert answered, "no provisional answer ever emitted"
    assert snaps[-1]["answer"] == "Berkeley"


def test_keepup_stages_are_monotonic():
    m = _mem()
    snaps = list(m.answer_keepup("where does sarah live?", n_stages=8))
    stages = [s["stage"] for s in snaps]
    assert stages == sorted(stages)


def test_mcp_walk_keepup_tool_registered():
    import asyncio
    from metacog.mcp_server import build_app
    app = build_app(memory=Memory())
    names = [t.name for t in asyncio.run(app.list_tools())]
    assert "walk_keepup" in names
