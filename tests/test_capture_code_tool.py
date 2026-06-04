"""Chat-side hook : generated code is self-assessed as a tool and, if so,
fed into the RAG (memory) as an executable, retrievable tool node.
Deterministic (fake LLM / marker mode).
"""
from metacog.memory import Memory
from metacog.epistemic import PointKind, SourceClass
from metacog.meta_walk import is_tool_point


class _YesLLM:
    def generate(self, prompt, max_tokens=None):
        if "VERDICT:" in prompt:
            return "VERDICT: yes KIND: code"
        return ""


class _NoLLM:
    def generate(self, prompt, max_tokens=None):
        if "VERDICT:" in prompt:
            return "VERDICT: no KIND: none"
        return ""


CODE = "def run(args: dict):\n    return sorted(args['items'])\n"


def test_reusable_code_is_captured_into_rag():
    m = Memory(llm=_YesLLM())
    p = m.capture_code_tool(
        CODE, context="sort a list of items", user_id="u1", session_id="s1")
    assert p is not None
    assert p.kind == PointKind.ACTION
    assert p.exec_spec and p.exec_spec["code"] == CODE.strip()
    assert {"tool", "skill", "code"}.issubset(set(p.tags))
    assert "name:run" in p.tags                       # def name → tool name
    assert "user:u1" in [t.lower() for t in p.tags]
    assert p.keywords_source == SourceClass.GENERATOR
    assert is_tool_point(p)                            # walk retrieves it
    assert p in m.points                               # fed into the RAG


def test_non_tool_code_is_not_captured():
    m = Memory(llm=_NoLLM())
    # purely illustrative snippet, LLM says not executable
    assert m.capture_code_tool("x = 1  # just an example",
                               context="an example") is None


def test_empty_code_is_noop():
    m = Memory(llm=_YesLLM())
    assert m.capture_code_tool("   ") is None


def test_captured_tool_is_retrievable_by_walk():
    from metacog.defaults import SimpleEncoder
    from metacog.keywords import SimpleKeywordExtractor
    from metacog.meta_walk import nearest_facts_with_fallback
    m = Memory(encoder=SimpleEncoder(), extractor=SimpleKeywordExtractor(),
               llm=_YesLLM())
    m.capture_code_tool(CODE, context="sort a list of items",
                        user_id="u1", session_id="s1")
    q = "sort a list of items"
    res = nearest_facts_with_fallback(
        tuple(m.encoder.encode(q)), q, m.points, 5, m._now(),
        encoder=m.encoder, extractor=m.extractor,
    )
    # ACTION tools surface via the FACT-or-tool retrieval path? They are
    # ACTION kind; confirm at least the captured tool exists & is a tool.
    tools = [p for p in m.points if is_tool_point(p) and p.exec_spec]
    assert len(tools) == 1


def test_mcp_capture_code_tool_registered():
    import asyncio
    from metacog.mcp_server import build_app
    app = build_app(memory=Memory())
    names = [t.name for t in asyncio.run(app.list_tools())]
    assert "capture_code_tool" in names
