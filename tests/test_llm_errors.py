"""ClaudeLLM.generate never fails silently (decision Proxy, TAC-209): a client
failure still answers "" (callers' fallbacks unchanged) but increments
`llm_errors`, keeps the reason in `last_error`, and warns on the FIRST one
only. No network: the client is faked or its import is made to fail."""
from __future__ import annotations

import logging
import sys

import pytest

from metacog.llm import ClaudeLLM, MissingCredential


class _Boom:
    class messages:
        @staticmethod
        def create(**kw):
            raise ConnectionError("gateway refused")


def test_client_failure_is_counted_and_warned_once(caplog):
    llm = ClaudeLLM()
    llm._client = _Boom()
    with caplog.at_level(logging.WARNING, logger="metacog.llm"):
        assert llm.generate("hello") == ""
        assert llm.generate("again") == ""
    assert llm.llm_errors == 2 and llm.n_calls == 0
    assert llm.last_error == "ConnectionError: gateway refused"
    warnings = [r for r in caplog.records if r.name == "metacog.llm"]
    assert len(warnings) == 1 and "gateway refused" in warnings[0].getMessage()


def test_missing_anthropic_package_is_an_error_not_a_silent_empty(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)   # import -> ModuleNotFoundError
    llm = ClaudeLLM()
    assert llm.generate("hello") == ""
    assert llm.llm_errors == 1
    assert llm.last_error.startswith("ModuleNotFoundError")
    # every derived call goes through generate: counted too, fallbacks unchanged
    assert llm.synthesize_step("q", "prev", "new") == "prev"
    assert llm.remove_overlap("full", "common") == "full"
    assert llm.llm_errors == 3


def test_missing_credential_still_raises(monkeypatch):
    pytest.importorskip("anthropic")
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                "CLAUDE_SESSION_INGRESS_TOKEN_FILE"):
        monkeypatch.delenv(var, raising=False)
    llm = ClaudeLLM()
    with pytest.raises(MissingCredential):
        llm.generate("hello")
    assert llm.llm_errors == 0


def test_empty_prompt_is_not_a_call():
    llm = ClaudeLLM()
    assert llm.generate("") == "" and llm.llm_errors == 0
