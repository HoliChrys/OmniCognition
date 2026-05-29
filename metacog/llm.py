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

Credential resolution :
  api_key arg  →  ANTHROPIC_API_KEY  →  ANTHROPIC_AUTH_TOKEN
sk-ant-oat* tokens go through auth_token= (OAuth bearer), everything
else through api_key= (x-api-key).
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence


DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
DEFAULT_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "64"))


class MissingCredential(RuntimeError):
    """Raised when an LLM call is attempted with no resolvable token.

    We never silently fall back to a stub : if the system is asked to
    generate, it generates with Claude, or it errors out with a clear
    message. Set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN for OAuth
    sessions) before running anything that triggers generation.
    """


def _resolve_credential(api_key: Optional[str]) -> str:
    token = (
        api_key
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
    if not token:
        raise MissingCredential(
            "Set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN for OAuth "
            "sessions) — metacog generates with Claude, no stub fallback."
        )
    return token


class ClaudeLLM:
    """Claude-backed LLM. Lazy client init ; usage counters exposed."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: Optional[str] = None,
        system: str = "You answer concisely and never explain.",
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.system = system
        self._api_key_override = api_key
        self._client = None  # lazy
        self.tokens_in = 0
        self.tokens_out = 0
        self.n_calls = 0

    @property
    def client(self):
        if self._client is None:
            import anthropic

            token = _resolve_credential(self._api_key_override)
            if token.startswith("sk-ant-oat"):
                self._client = anthropic.Anthropic(auth_token=token)
            else:
                self._client = anthropic.Anthropic(api_key=token)
        return self._client

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if not prompt:
            return ""
        budget = max(8, max_tokens or self.max_tokens)
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=budget,
                system=self.system,
                messages=[{"role": "user", "content": prompt}],
            )
        except MissingCredential:
            raise
        except Exception:
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
        if not texts:
            return ""
        block = "\n".join(f"- {t}" for t in texts)
        prompt = (
            "Extract the SHARED content these passages have in common as "
            "ONE short sentence. No prose, just the sentence.\n\n" + block
        )
        return self.generate(prompt, max_tokens=64)

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
