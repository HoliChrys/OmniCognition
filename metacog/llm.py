"""
ClaudeLLM — the only LLM the system uses.

A thin Anthropic-SDK adapter that implements every protocol the rest
of metacog needs :

  - `generate(prompt, max_tokens)`            → meta_walk + synthesis
  - `synthesize_step(query, prev, new)`       → reasoning trajectory
  - `propose_action(query, current_answer)`   → reasoning trajectory
  - `extract_common(texts)` / `remove_overlap` → collision resolution

The client is created LAZILY : the credential is read only when the
first LLM call happens, so importing metacog (and running tests that
never touch the LLM) works without an API key.

Cor. 5 holds at the architectural level : ClaudeLLM outputs are always
treated as GENERATOR. Callers that wrap them into Point objects mark
`source=SourceClass.GENERATOR` ; they NEVER enter the observation set.

Failures : `generate` answers "" on a client failure (callers keep their
fallback), but never silently — each one increments `llm_errors` (reason in
`last_error`) and the first one logs a warning. `MissingCredential` is the
only exception that propagates.

Prompt budget (TAC-405) : no prompt larger than the model's context window is
ever sent. The bound is the prompt's UTF-8 byte count — a byte-level BPE never
emits more tokens than bytes, so bytes ≤ budget ⇒ tokens ≤ budget without a
tokenizer or a network call. `generate` refuses an over-budget prompt (counted
in `llm_errors` as `PromptTooLong`, answers ""). `extract_common` never reaches
that refusal : it chunks its passages under the budget and merges the partial
commons (map-reduce — the shared content of all = the shared content of the
partial commons), and cuts a single passage that alone exceeds the budget.

Credential resolution :
  api_key arg  →  ANTHROPIC_API_KEY  →  ANTHROPIC_AUTH_TOKEN
sk-ant-oat* tokens go through auth_token= (OAuth bearer), everything
else through api_key= (x-api-key).
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Sequence, Tuple

_log = logging.getLogger(__name__)

try:
    import anthropic as _ant_module
    _RETRYABLE_ERRORS = tuple(
        e for e in (
            getattr(_ant_module, "OverloadedError", None),
            getattr(_ant_module, "InternalServerError", None),
        )
        if e is not None
    )
except ImportError:
    _RETRYABLE_ERRORS = ()


DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
DEFAULT_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "64"))
# The model's context window (input + output), in tokens — a property of the
# model, not a knob : 200k for the Claude 4.x family (the API's own
# "prompt is too long: N tokens > 200000 maximum").
CONTEXT_WINDOW = int(os.environ.get("CLAUDE_CONTEXT_WINDOW", "200000"))

_COMMON_HEADER = (
    "Extract the SHARED content these passages have in common as "
    "ONE short sentence. No prose, just the sentence.\n\n"
)


def _nbytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _cut(text: str, nbytes: int) -> str:
    """Head of `text` in at most `nbytes` UTF-8 bytes (no split character)."""
    return text.encode("utf-8")[:max(0, nbytes)].decode("utf-8", "ignore")


class PromptTooLong(ValueError):
    """A prompt over the context-window budget — refused, never sent."""


class MissingCredential(RuntimeError):
    """Raised when an LLM call is attempted with no resolvable token.

    We never silently fall back to a stub : if the system is asked to
    generate, it generates with Claude, or it errors out with a clear
    message. Set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN for OAuth
    sessions) before running anything that triggers generation.
    """


def _read_ingress_token() -> Optional[str]:
    """OAuth bearer token file used by managed remote-exec environments
    (Claude Code on the web / mobile). Same fallback the LoCoMo agent uses."""
    path = os.environ.get("CLAUDE_SESSION_INGRESS_TOKEN_FILE")
    if not path:
        return None
    try:
        with open(path) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _resolve_credential(api_key: Optional[str]) -> Tuple[str, bool]:
    """Return (token, is_auth_token). `is_auth_token` is True for OAuth
    bearer tokens (ANTHROPIC_AUTH_TOKEN / sk-ant-oat / ingress token file)
    which must go through anthropic's auth_token= rather than api_key=."""
    if api_key:
        return api_key, api_key.startswith("sk-ant-oat")
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key, env_key.startswith("sk-ant-oat")
    auth_env = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if auth_env:
        return auth_env, True
    ingress = _read_ingress_token()
    if ingress:
        return ingress, True
    raise MissingCredential(
        "Set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN / "
        "CLAUDE_SESSION_INGRESS_TOKEN_FILE for OAuth sessions) — "
        "metacog generates with Claude, no stub fallback."
    )


class ClaudeLLM:
    """Claude-backed LLM. Lazy client init ; usage counters exposed."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: Optional[str] = None,
        system: str = "You answer concisely and never explain.",
        context_window: int = CONTEXT_WINDOW,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.system = system
        self._api_key_override = api_key
        self._client = None  # lazy
        self.tokens_in = 0
        self.tokens_out = 0
        self.n_calls = 0
        # Client failures `generate` absorbed into "" (missing `anthropic`
        # package, auth refused, retries exhausted…). Without it an LLM that
        # never answers is indistinguishable from one that answers "" (D3).
        self.llm_errors = 0
        self.last_error: Optional[str] = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
            token, is_auth = _resolve_credential(self._api_key_override)
            # OAuth bearer tokens go through auth_token= ; standard API
            # keys through api_key=.
            if is_auth:
                self._client = anthropic.Anthropic(
                    auth_token=token, base_url=base_url
                )
            else:
                self._client = anthropic.Anthropic(
                    api_key=token, base_url=base_url
                )
        return self._client

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _record_error(self, exc: BaseException) -> None:
        """Count a client failure; warn on the FIRST one only (the return
        value stays "" so callers' fallbacks are unchanged)."""
        self.llm_errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        if self.llm_errors == 1:
            _log.warning("ClaudeLLM.generate failed, answering \"\" "
                         "(model=%s): %s — further failures are only "
                         "counted in llm_errors", self.model, self.last_error)

    def prompt_budget(self, max_tokens: Optional[int] = None) -> int:
        """Bytes a user prompt may take : the window minus the answer's
        `max_tokens` and the system prompt (both counted in the window)."""
        out = max(8, max_tokens or self.max_tokens)
        return self.context_window - out - _nbytes(self.system)

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if not prompt:
            return ""
        budget = max(8, max_tokens or self.max_tokens)
        limit = self.prompt_budget(budget)
        if _nbytes(prompt) > limit:
            # Never sent : the API would answer 400 "prompt is too long".
            self._record_error(PromptTooLong(
                f"{_nbytes(prompt)} bytes > budget {limit}"))
            return ""
        for attempt in range(4):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=budget,
                    temperature=0,
                    system=self.system,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except MissingCredential:
                raise
            except Exception as exc:
                # Retry on transient server errors (529 overloaded, 5xx).
                if (_RETRYABLE_ERRORS and isinstance(exc, _RETRYABLE_ERRORS)
                        and attempt < 3):
                    time.sleep(2 ** attempt)
                    continue
                self._record_error(exc)
                return ""
        self.n_calls += 1
        if hasattr(resp, "usage"):
            self.tokens_in += getattr(resp.usage, "input_tokens", 0) or 0
            self.tokens_out += getattr(resp.usage, "output_tokens", 0) or 0
        if not resp.content:
            return ""
        return "".join(
            getattr(b, "text", "") for b in resp.content
            if getattr(b, "type", "") == "text"
        ).strip()

    # ------------------------------------------------------------------
    # Reasoning trajectory protocol
    # ------------------------------------------------------------------

    def synthesize_step(
        self,
        query: str,
        previous_answer: Optional[str],
        new_point_content: str,
    ) -> str:
        prompt = (
            f"QUERY: {query}\n"
            f"PREVIOUS: {previous_answer or '(none)'}\n"
            f"NEW EVIDENCE: {new_point_content}\n\n"
            "Update the answer in one short phrase. No prose."
        )
        return self.generate(prompt, max_tokens=64) or (previous_answer or "")

    def propose_action(
        self, query: str, current_answer: str
    ) -> Optional[str]:
        prompt = (
            f"QUERY: {query}\n"
            f"CURRENT ANSWER: {current_answer}\n\n"
            "If a concrete action would settle this, output its imperative "
            "form (≤ 12 words). Otherwise output 'none'."
        )
        out = self.generate(prompt, max_tokens=32)
        if not out or out.strip().lower().startswith("none"):
            return None
        return out

    # ------------------------------------------------------------------
    # Collision content surgery
    # ------------------------------------------------------------------

    def extract_common(self, texts: Sequence[str]) -> str:
        """The content `texts` share, every call under the prompt budget.

        Chunk and merge : passages are packed greedily into prompts that fit
        the budget, each chunk yields its partial common, and the partial
        commons are merged by the same procedure until one call covers them.
        Sharing is an intersection, so common(all) = common(partial commons) —
        and one empty partial common means nothing is shared by all (stop).
        A passage that alone exceeds the budget is cut to it : the common of
        its head is still shared content, only possibly less of it."""
        texts = [t for t in texts if t]
        if not texts:
            return ""
        room = self.prompt_budget(64) - _nbytes(_COMMON_HEADER)
        merging = False
        while True:
            items = [f"- {_cut(t, room - _nbytes('- '))}" for t in texts]
            chunks: List[List[str]] = [[]]
            used = 0
            for item in items:
                # +1 : the newline joining it to the previous item
                size = _nbytes(item) + (1 if chunks[-1] else 0)
                if chunks[-1] and used + size > room:
                    chunks.append([])
                    used, size = 0, _nbytes(item)
                chunks[-1].append(item)
                used += size
            if merging and len(chunks) == len(texts):
                return ""   # the budget cannot hold two commons : no merge
            partials: List[str] = []
            for chunk in chunks:
                common = self.generate(_COMMON_HEADER + "\n".join(chunk),
                                       max_tokens=64)
                if not common:
                    return ""
                partials.append(common)
            if len(partials) == 1:
                return partials[0]
            texts, merging = partials, True

    def remove_overlap(self, full: str, common: str) -> str:
        if not full or not common:
            return full
        prompt = (
            f"FULL: {full}\n"
            f"COMMON: {common}\n\n"
            "Output FULL with the COMMON information removed, in one short "
            "sentence. No prose."
        )
        out = self.generate(prompt, max_tokens=64)
        return out or full
