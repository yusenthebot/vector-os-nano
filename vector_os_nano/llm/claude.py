"""ClaudeProvider — LLM provider for Anthropic Claude via OpenRouter or direct API.

Uses httpx (synchronous) for HTTP calls. Supports:
- OpenRouter API (default): https://openrouter.ai/api/v1
- Direct Anthropic API: https://api.anthropic.com/v1

No ROS2 imports. No hardcoded secrets.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from vector_os_nano.core.types import TaskPlan, TaskStep
from vector_os_nano.llm.prompts import build_planning_prompt

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "anthropic/claude-haiku-4-5"
_DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
_REQUEST_TIMEOUT = 30.0  # seconds


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` code fences from LLM output.

    LLMs frequently wrap JSON in markdown code blocks. This strips them so
    json.loads() can parse the content.
    """
    # Match ```json ... ``` or ``` ... ``` (greedy — take the last ```)
    pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_plan_response(goal: str, raw_text: str) -> TaskPlan:
    """Parse an LLM text response into a TaskPlan.

    Handles:
    - Valid JSON with "steps" list
    - Clarification responses with "requires_clarification": true
    - Markdown-wrapped JSON (```json ... ```)
    - Malformed or empty responses (returns empty TaskPlan)

    Args:
        goal: the original planning goal (preserved in TaskPlan.goal).
        raw_text: raw text from the LLM response.

    Returns:
        TaskPlan — never raises.
    """
    if not raw_text or not raw_text.strip():
        return TaskPlan(goal=goal)

    cleaned = _strip_markdown_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning("Failed to parse LLM response as JSON: %.200s", raw_text)
        return TaskPlan(goal=goal)

    if not isinstance(data, dict):
        log.warning("LLM response is not a JSON object: %s", type(data))
        return TaskPlan(goal=goal)

    # Clarification response
    if data.get("requires_clarification"):
        return TaskPlan(
            goal=goal,
            steps=[],
            requires_clarification=True,
            clarification_question=data.get("clarification_question"),
        )

    # Normal plan response
    raw_steps = data.get("steps", [])
    if not isinstance(raw_steps, list):
        log.warning("LLM 'steps' field is not a list: %s", type(raw_steps))
        return TaskPlan(goal=goal)

    steps: list[TaskStep] = []
    for raw in raw_steps:
        if not isinstance(raw, dict):
            log.warning("Skipping non-dict step: %s", raw)
            continue
        try:
            step = TaskStep.from_dict(raw)
            steps.append(step)
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("Skipping malformed step %s: %s", raw, exc)

    return TaskPlan(goal=goal, steps=steps)


# ---------------------------------------------------------------------------
# ClaudeProvider
# ---------------------------------------------------------------------------

class ClaudeProvider:
    """LLM provider using Anthropic Claude via OpenRouter or direct API.

    Supports the LLMProvider protocol:
    - plan(): build system prompt, send to LLM, parse TaskPlan
    - query(): simple chat completion for free-form questions

    Args:
        api_key: API key for OpenRouter or Anthropic.
        model: model identifier (OpenRouter format by default).
        api_base: API base URL. Defaults to OpenRouter.
        max_history: maximum conversation turns to keep (default 20).
        temperature: sampling temperature (default 0.0 for determinism).
        max_tokens: max tokens in the completion (default 2048).
        timeout: HTTP request timeout in seconds (default 30.0).
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        api_base: str | None = None,
        max_history: int = 20,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = _REQUEST_TIMEOUT,
    ) -> None:
        self._api_key: str = api_key
        self.model: str = model
        self.api_base: str = api_base or _DEFAULT_API_BASE
        self.max_history: int = max_history
        self._temperature: float = temperature
        self._max_tokens: int = max_tokens
        self._endpoint: str = f"{self.api_base.rstrip('/')}/chat/completions"
        self._http: httpx.Client = httpx.Client(timeout=timeout)

    def __repr__(self) -> str:
        return f"ClaudeProvider(model={self.model!r}, api_base={self.api_base!r})"

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def plan(
        self,
        goal: str,
        world_state: dict[str, Any],
        skill_schemas: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
    ) -> TaskPlan:
        """Decompose a natural-language goal into a TaskPlan.

        Builds the planning system prompt from skill schemas and world state,
        sends the goal to the LLM, and parses the JSON response.

        Returns an empty TaskPlan (no steps) on any network or parse error.
        """
        system_prompt = build_planning_prompt(skill_schemas, world_state)

        messages: list[dict[str, Any]] = []

        # Inject history (trimmed to max_history)
        if history:
            trimmed = history[-self.max_history:]
            messages.extend(trimmed)

        messages.append({"role": "user", "content": goal})

        raw_response = self._chat_completion(system_prompt, messages)
        return parse_plan_response(goal, raw_response)

    def query(
        self,
        prompt: str,
        image: Any = None,
    ) -> str:
        """Send a free-form prompt and return the LLM's text response.

        Image support is reserved for future multimodal use.

        Returns an error description string on network failure — never raises.
        """
        system_prompt = (
            "You are a helpful assistant for a robot system. "
            "Answer concisely and accurately."
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        return self._chat_completion(system_prompt, messages)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat_completion(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> str:
        """POST to the chat completions endpoint and return the assistant text.

        Returns an error description string on any failure.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
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
        """Extract assistant content text from an OpenAI-format response."""
        try:
            content = data["choices"][0]["message"].get("content") or ""
            return content
        except (KeyError, IndexError, TypeError) as exc:
            log.warning("Unexpected LLM response shape: %s", exc)
            return "LLM error: unexpected response format"
