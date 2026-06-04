"""
Tests for the sandboxed Python executor + Memory tool API.

A "tool" is an ACTION Point that carries BOTH the "tool" tag AND a
populated exec_spec. The PyExecutor rolls the embedded `def run(args)`
in an isolated Python subprocess (timeout, env-stripped, json IO) and
returns either {ok, result} or {ok=False, error}.
"""

from __future__ import annotations

import os
import pickle
import tempfile

import pytest

from metacog.defaults import SimpleEncoder
from metacog.epistemic import Point, PointKind, SourceClass
from metacog.executor import EXECUTOR_SOURCE, PyExecutor
from metacog.memory import Memory


# ---------------------------------------------------------------------------
# PyExecutor (subprocess sandbox)
# ---------------------------------------------------------------------------


def test_pyexecutor_runs_simple_function():
    out = PyExecutor().execute(
        {"lang": "python", "code": "def run(args): return args['x'] + 1"},
        {"x": 41},
    )
    assert out == {"ok": True, "result": 42}


def test_pyexecutor_timeout():
    out = PyExecutor(timeout=2).execute(
        {"lang": "python",
         "code": "import time\ndef run(args):\n    time.sleep(10)\n    return 'done'"},
        {},
    )
    assert out["ok"] is False
    assert "timeout" in out["error"]


def test_pyexecutor_handles_exception():
    out = PyExecutor().execute(
        {"lang": "python",
         "code": "def run(args): raise ValueError('boom')"},
        {},
    )
    assert out["ok"] is False
    assert "boom" in out["error"].lower() or "valueerror" in out["error"].lower()


def test_pyexecutor_non_json_output_when_no_run():
    out = PyExecutor().execute(
        {"lang": "python", "code": "print('hello')"},
        {},
    )
    assert out["ok"] is False


def test_pyexecutor_rejects_unknown_lang():
    out = PyExecutor().execute({"lang": "ruby", "code": "puts 1"}, {})
    assert out["ok"] is False
    assert "lang" in out["error"]


def test_pyexecutor_rejects_no_run_function():
    out = PyExecutor().execute(
        {"lang": "python", "code": "x = 1"},
        {},
    )
    assert out["ok"] is False
    assert "def run" in out["error"]


def test_pyexecutor_rejects_non_dict_args():
    out = PyExecutor().execute(
        {"lang": "python", "code": "def run(args): return 1"},
        "not a dict",  # type: ignore[arg-type]
    )
    assert out["ok"] is False


def test_pyexecutor_isolated_env_pythonpath_ignored():
    # -I mode ignores PYTHONPATH. If we'd inherited it, we could import a
    # module from a custom path. We can't easily test PYTHONPATH directly,
    # but we can verify env is minimal : os.environ.get('PATH') in the
    # child should be empty / unset.
    out = PyExecutor().execute(
        {"lang": "python",
         "code": "import os\ndef run(args): return os.environ.get('PATH', '')"},
        {},
    )
    assert out["ok"] is True
    assert out["result"] == ""             # PATH was stripped from env


def test_pyexecutor_tolerates_stdout_pollution_before_result():
    # User code prints debug info before returning. Last __RESULT__ line
    # is the canonical result.
    out = PyExecutor().execute(
        {"lang": "python",
         "code": "def run(args):\n    print('debug log')\n    print('more debug')\n    return {'a': 1}"},
        {},
    )
    assert out == {"ok": True, "result": {"a": 1}}


def test_executor_source_is_computation():
    assert EXECUTOR_SOURCE is SourceClass.COMPUTATION


# ---------------------------------------------------------------------------
# Memory tool API
# ---------------------------------------------------------------------------


def _mem() -> Memory:
    return Memory(encoder=SimpleEncoder())


def test_ingest_tool_creates_action_with_tool_tag():
    m = _mem()
    t = m.ingest_tool("Add 1 to x",
                      "def run(args): return args['x'] + 1")
    assert t.kind is PointKind.ACTION
    assert t.has_tag("tool")
    assert t.has_tag("action")              # auto kind tag
    assert t.exec_spec is not None
    assert t.exec_spec["lang"] == "python"
    assert "x" in t.exec_spec["code"]


def test_execute_tool_creates_executed_fact():
    m = _mem()
    t = m.ingest_tool("Add 1", "def run(args): return args['x'] + 1")
    before = len(m.points)
    out = m.execute_tool(t.id, {"x": 41})
    assert out["ok"] is True and out["result"] == 42 and out["fact_id"]
    after = len(m.points)
    assert after == before + 1                  # one FACT created
    fact = next(p for p in m.points if p.id == out["fact_id"])
    assert fact.kind is PointKind.FACT
    assert fact.has_tag("executed")
    assert t.id in fact.parents
    assert fact.content == "42"


def test_execute_tool_rejects_non_tool_point():
    m = _mem()
    f = m.ingest("just a fact", kind="FACT", id="F1")
    with pytest.raises(ValueError, match="not tagged 'tool'"):
        m.execute_tool("F1", {})


def test_execute_tool_rejects_tool_without_exec_spec():
    m = _mem()
    p = m.ingest("an action", kind="ACTION", id="A1")
    p.add_tag("tool")                            # tagged but no exec_spec
    with pytest.raises(ValueError, match="no exec_spec"):
        m.execute_tool("A1", {})


def test_execute_tool_rejects_unknown_id():
    m = _mem()
    with pytest.raises(ValueError, match="unknown tool id"):
        m.execute_tool("does_not_exist", {})


def test_execute_tool_failure_no_fact_created():
    m = _mem()
    t = m.ingest_tool("Boom", "def run(args): raise ValueError('nope')")
    before = len(m.points)
    out = m.execute_tool(t.id, {})
    assert out["ok"] is False and out["fact_id"] is None
    assert len(m.points) == before               # no FACT on failure


def test_find_tools_returns_only_tools():
    m = _mem()
    m.ingest("plain action one", kind="ACTION", id="A1")
    m.ingest("plain action two", kind="ACTION", id="A2")
    t = m.ingest_tool("the tool action", "def run(args): return 1")
    out = m.find_tools("action", k=5)
    assert all(p.has_tag("tool") for p in out)
    assert t in out


def test_find_tools_semantic_ordering():
    m = _mem()
    add = m.ingest_tool("add numbers together",
                        "def run(args): return args['a']+args['b']")
    date = m.ingest_tool("parse a date string into year",
                         "def run(args): return 2023")
    top = m.find_tools("addition sum", k=2)
    # The add tool should rank first under simple keyword/cosine match.
    assert top[0] is add
    assert date in top


def test_pickle_roundtrip_exec_spec():
    m = _mem()
    t = m.ingest_tool("Add 1", "def run(args): return args['x']+1")
    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        m.storage_path = path
        m.save()
        m2 = Memory(encoder=SimpleEncoder(), storage_path=path)
        m2.load()
        t2 = next(p for p in m2.points if p.id == t.id)
        assert t2.exec_spec == t.exec_spec
        assert t2.has_tag("tool")
    finally:
        os.unlink(path)


def test_tool_action_carries_both_action_and_tool_tags():
    m = _mem()
    t = m.ingest_tool("anything", "def run(args): return 1")
    # Universal tagging auto-adds "action" ; ingest_tool adds "tool".
    assert set(["action", "tool"]).issubset(set(t.tags))


def test_execute_tool_with_injected_mock_executor():
    # The executor is injectable for testing : we can stub PyExecutor
    # entirely and still exercise the Memory wiring.
    m = _mem()
    t = m.ingest_tool("Anything", "def run(args): return None")

    class _Stub:
        def execute(self, spec, args):
            return {"ok": True, "result": "stubbed"}

    out = m.execute_tool(t.id, {}, executor=_Stub())
    assert out["ok"] is True and out["result"] == "stubbed"
    fact = next(p for p in m.points if p.id == out["fact_id"])
    assert fact.content == "stubbed" and fact.has_tag("executed")
