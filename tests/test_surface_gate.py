"""
The MCP surface gate : build_app registers only the tools of the chosen surface.
The real FastMCP is absent here, so we drive the extracted gating helper with a
fake app that records which tool names get registered.
"""

from __future__ import annotations

from metacog.mcp_server import _install_surface_gate
from metacog import canonical_tools as C


class _FakeApp:
    """Mimics FastMCP.tool() : a decorator factory that records registrations."""

    def __init__(self):
        self.registered = []

    def tool(self, *a, **kw):
        def deco(fn):
            self.registered.append(fn.__name__)
            return fn
        return deco


def _register(app, names):
    """Simulate the @app.tool() decorators for a list of tool names."""
    for name in names:
        def make(n):
            def f():
                return None
            f.__name__ = n
            return f
        app.tool()(make(name))


def test_gate_none_exposes_all():
    app = _FakeApp()
    _install_surface_gate(app, None)     # no gate
    _register(app, ["retrieve", "clue_search", "route"])
    assert set(app.registered) == {"retrieve", "clue_search", "route"}


def test_gate_external_registers_only_exposed():
    app = _FakeApp()
    _install_surface_gate(app, C.surface_tools("external"))
    # mix exposed + internal tools
    _register(app, ["retrieve", "walk_start", "ensure_tool",   # exposed
                    "clue_search", "route", "bag", "walk_next"])  # internal
    assert set(app.registered) == {"retrieve", "walk_start", "ensure_tool"}
    assert "clue_search" not in app.registered
    assert "route" not in app.registered


def test_gate_external_light_is_minimal():
    app = _FakeApp()
    _install_surface_gate(app, C.surface_tools("external_light"))
    _register(app, ["ingest_message", "push_code", "walk_start",
                    "retrieve", "ensure_tool", "sleep"])
    assert set(app.registered) == {"ingest_message", "push_code", "walk_start"}


def test_unexposed_tool_stays_callable():
    """A gated-out tool is returned undecorated — still a usable function."""
    app = _FakeApp()
    _install_surface_gate(app, {"retrieve"})

    @app.tool()
    def clue_search():
        return "still works"

    assert clue_search() == "still works"     # callable internally
    assert "clue_search" not in app.registered  # but not on the MCP surface
