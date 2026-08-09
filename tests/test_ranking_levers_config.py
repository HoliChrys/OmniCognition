"""
Production wiring for the ACT-R ranking levers : build_app resolves
recency_weight / spreading_weight from explicit arg > env var > leave-as-is, so
a deploy can turn them on without code changes. The resolver is mcp-free and
unit-testable (the full build_app needs FastMCP, absent here).
"""

from __future__ import annotations

import pytest

from metacog.mcp_server import _resolve_lever


def test_explicit_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv("METACOG_RECENCY_WEIGHT", "0.9")
    assert _resolve_lever(0.3, "METACOG_RECENCY_WEIGHT") == 0.3


def test_env_used_when_no_arg(monkeypatch):
    monkeypatch.setenv("METACOG_SPREADING_WEIGHT", "0.25")
    assert _resolve_lever(None, "METACOG_SPREADING_WEIGHT") == 0.25


def test_none_when_neither(monkeypatch):
    monkeypatch.delenv("METACOG_RECENCY_WEIGHT", raising=False)
    assert _resolve_lever(None, "METACOG_RECENCY_WEIGHT") is None


def test_empty_env_is_none(monkeypatch):
    monkeypatch.setenv("METACOG_RECENCY_WEIGHT", "")
    assert _resolve_lever(None, "METACOG_RECENCY_WEIGHT") is None


def test_zero_arg_is_respected():
    # explicit 0.0 must turn the lever OFF, not fall through to env/None
    assert _resolve_lever(0.0, "METACOG_RECENCY_WEIGHT") == 0.0
