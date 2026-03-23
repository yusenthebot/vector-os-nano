"""OpenAIProvider — LLM provider for any OpenAI-compatible API.

Works with:
- OpenAI (api.openai.com)
- Ollama (localhost:11434 with /v1 suffix)
- LM Studio (localhost:1234/v1)
- vLLM, TGI, and other OpenAI-compatible servers

No ROS2 imports. No hardcoded secrets.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from vector_os_nano.core.types import TaskPlan
from vector_os_nano.llm.claude import parse_plan_response
from vector_os_nano.llm.prompts import build_planning_prompt

log = logging.getLogger(__name__)

_DEFAULT_API_BASE = "https://api.openai.com/v1"
_DEFAULT_REQUEST_TIMEOUT = 30.0


class OpenAIProvider:
    """LLM provider using any OpenAI-compatible chat completions API.

    Supports the LLMProvider protocol:
    - plan(): build system prompt, send to API, parse TaskPlan
    - query(): simple chat completion for free-form questions

    Args:
        api_key: API key (use "ollama" or empty string for local servers).
        model: model identifier string.
        api_base: base URL of the OpenAI-compatible API.
        max_history: max conversation turns to keep (default 20).
        temperature: sampling temperature (default 0.0).
        max_tokens: max tokens in completion (default 2048).
        timeout: HTTP timeout in seconds (default 30.0).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str = _DEFAULT_API_BASE,
        max_history: int = 20,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = _DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self.api_key: str = api_key
        self.model: str = model
        self.api_base: str = api_base
        self.max_history: int = max_history
        self._temperature: float = temperature
        self._max_tokens: int = max_tokens
        self._endpoint: str = f"{api_base.rstrip('/')}/chat/completions"
        self._http: httpx.Client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def plan(
        self,
        goal: str,
        world_state: dict[str, Any],
        skill_schemas: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
        model_override: str | None = None,
    ) -> TaskPlan:
        """Decompose a natural-language goal into a TaskPlan.

        Returns an empty TaskPlan on any network or parse error.
        """
        system_prompt = build_planning_prompt(skill_schemas, world_state)

        messages: list[dict[str, Any]] = []
        if history:
            messages.extend(history[-self.max_history:])
        messages.append({"role": "user", "content": goal})

        raw = self._chat_completion(system_prompt, messages, model_override)
        return parse_plan_response(goal, raw)

    def query(
        self,
        prompt: str,
        image: Any = None,
        model_override: str | None = None,
    ) -> str:
        """Send a free-form prompt and return the LLM's text response.

        Returns an error description string on failure — never raises.
        """
        system_prompt = (
            "You are a helpful assistant for a robot system. "
            "Answer concisely and accurately."
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        return self._chat_completion(system_prompt, messages, model_override)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat_completion(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        model_override: str | None = None,
    ) -> str:
        """POST to chat/completions endpoint and return assistant text."""
        model = model_override or self.model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = self._http.post(self._endpoint, json=payload, headers=headers)
            resp.raise_for_status()
        except httpx.TimeoutException:
            log.warning("LLM request timed out")
            return "LLM error: request timed out"
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:300]
            log.warning("LLM HTTP error %s: %s", exc.response.status_code, body)
            return f"LLM error: HTTP {exc.response.status_code}"
        except httpx.RequestError as exc:
            log.warning("LLM network error: %s", exc)
            return f"LLM error: network error — {exc}"

        return self._extract_text(resp.json())

    def _extract_text(self, data: dict[str, Any]) -> str:
        """Extract assistant content from an OpenAI-format response dict."""
        try:
            content = data["choices"][0]["message"].get("content") or ""
            return content
        except (KeyError, IndexError, TypeError) as exc:
            log.warning("Unexpected LLM response shape: %s", exc)
            return "LLM error: unexpected response format"
