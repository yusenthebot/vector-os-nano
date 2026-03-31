"""LocalProvider — Ollama wrapper using the OpenAI-compatible API.

Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1,
so this is a wrapper with Ollama-specific defaults and all required methods.

Supports any Ollama-hosted model: llama3, mistral, gemma2, phi3, etc.
Also works with LM Studio (same API, different port).

No ROS2 imports. No API key required for local inference.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import openai

from vector_os_nano.core.types import TaskPlan

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama3"
_DEFAULT_HOST = "http://localhost:11434"

# Classification prompt for intent detection
_CLASSIFY_PROMPT = """You are an intent classifier for a robot arm control system.
Classify the user message into ONE of these categories:
- "task": User wants the robot to perform an action (pick, place, move, etc.)
- "chat": User is having a conversation or asking a question
- "query": User is asking about the current state (objects, positions, etc.)
- "direct": User is giving a direct command that matches a built-in skill

Respond with ONLY the category name, nothing else.

Examples:
"pick up the red cup" -> task
"hello how are you" -> chat
"what objects are on the table" -> query
"home" -> direct
"帮我抓杯子" -> task
"你好" -> chat
"""


class LocalProvider:
    """LLM provider wrapping a locally-running Ollama instance.

    Uses Ollama's OpenAI-compatible /v1 endpoint. No API key needed
    for local inference.

    Args:
        model: Ollama model name (default "llama3:8b").
        host: base URL of the Ollama server (default "http://localhost:11434").
        **kwargs: additional options (temperature, max_tokens, etc.).
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
        **kwargs,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._api_base = f"{self._host}/v1"
        self._temperature = kwargs.get("temperature", 0.0)
        self._max_tokens = kwargs.get("max_tokens", 2048)
        self._client = openai.OpenAI(
            api_key="ollama",  # placeholder, not needed for local
            base_url=self._api_base,
        )
        logger.info(f"LocalProvider initialized: {self._model} @ {self._api_base}")

    def __repr__(self) -> str:
        return f"LocalProvider(model={self._model!r})"

    def _chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Low-level chat completion via OpenAI client."""
        try:
            response = self._client.chat.completions.create(
                model=model or self._model,
                messages=messages,
                temperature=temperature if temperature is not None else self._temperature,
                max_tokens=max_tokens or self._max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error(f"LocalProvider chat error: {exc}")
            return f"Error: {exc}"

    def plan(
        self,
        goal: str,
        world_state: dict[str, Any],
        skill_schemas: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
        model_override: str | None = None,
    ) -> TaskPlan:
        """Decompose a goal into a TaskPlan using LLM.

        Args:
            goal: natural-language instruction (e.g. "pick up the red cup").
            world_state: serialized world model snapshot.
            skill_schemas: list of skill schemas from SkillRegistry.to_schemas().
            history: optional prior conversation turns.
            model_override: optional model identifier.

        Returns:
            TaskPlan with steps, or a clarification request.
        """
        model = model_override or self._model

        # Build system prompt with available skills
        skills_text = "\n".join(
            f"- {s['name']}: {s.get('description', '')}"
            for s in skill_schemas
        )
        system_prompt = f"""You are a robot arm planner. Given a goal and available skills, output a JSON plan.

Available skills:
{skills_text}

World state:
{json.dumps(world_state, indent=2, ensure_ascii=False)}

Respond with JSON only:
{{"goal": "...", "steps": [{{"skill_name": "...", "parameters": {{...}}}}], "requires_clarification": false}}

If goal is ambiguous, set requires_clarification=true and add clarification_question.
If the task is impossible, explain why in a message field."""

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        if history:
            for h in history[-10:]:  # limit history
                role = h.get("role", "user")
                content = h.get("content", "")
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": goal})

        response_text = self._chat(messages, model=model, temperature=0.0)

        try:
            data = json.loads(response_text)
            steps = []
            for i, s in enumerate(data.get("steps", [])):
                from vector_os_nano.core.types import TaskStep
                steps.append(TaskStep(
                    step_id=f"s{i+1}",
                    skill_name=s.get("skill_name", "unknown"),
                    parameters=s.get("parameters", {}),
                    depends_on=[f"s{i}"] if i > 0 else [],
                    preconditions=[],
                    postconditions=[],
                ))
            return TaskPlan(
                goal=data.get("goal", goal),
                steps=steps,
                message=data.get("message"),
                requires_clarification=data.get("requires_clarification", False),
                clarification_question=data.get("clarification_question"),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"Failed to parse plan JSON: {exc}\nResponse: {response_text[:500]}")
            return TaskPlan(goal=goal, steps=[], message=response_text)

    def query(
        self,
        prompt: str,
        image: Any = None,
        model_override: str | None = None,
    ) -> str:
        """Answer a free-form question (optionally with an image).

        Args:
            prompt: natural-language question.
            image: optional image data.
            model_override: optional model identifier.

        Returns:
            LLM response as plain string.
        """
        model = model_override or self._model
        messages = [
            {"role": "user", "content": prompt}
        ]
        return self._chat(messages, model=model)

    def classify(
        self,
        user_message: str,
        model_override: str | None = None,
    ) -> str:
        """Classify user intent: task, chat, query, or direct.

        Args:
            user_message: The user's input string.
            model_override: optional model identifier.

        Returns:
            One of: "task", "chat", "query", "direct"
        """
        model = model_override or self._model
        messages = [
            {"role": "system", "content": _CLASSIFY_PROMPT},
            {"role": "user", "content": user_message},
        ]
        result = self._chat(messages, model=model, temperature=0.0)
        result = result.strip().lower()
        if result in ("task", "chat", "query", "direct"):
            return result
        # Fallback: simple heuristic
        if any(kw in result for kw in ["抓", "pick", "放", "place", "移动", "move", "拿"]):
            return "task"
        if any(kw in result for kw in ["什么", "what", "哪里", "where", "几个", "how many"]):
            return "query"
        if any(kw in result for kw in ["你好", "hello", "hi", "帮忙", "help"]):
            return "chat"
        return "direct"

    def chat(
        self,
        message: str,
        system_prompt: str = "",
        history: list[dict[str, Any]] | None = None,
        model_override: str | None = None,
        image: Any = None,
    ) -> str:
        """Conversational chat with optional history and system prompt.

        Args:
            message: user's message.
            system_prompt: system-level instructions.
            history: prior conversation turns.
            model_override: optional model identifier.
            image: optional image (not supported for local models).

        Returns:
            Assistant's response string.
        """
        model = model_override or self._model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for h in history[-20:]:  # limit history
                role = h.get("role", "user")
                content = h.get("content", "")
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        return self._chat(messages, model=model)

    def summarize(
        self,
        original_request: str,
        trace: str,
        model_override: str | None = None,
    ) -> str:
        """Summarize execution results in natural language.

        Args:
            original_request: the user's original command.
            trace: execution trace string.
            model_override: optional model identifier.

        Returns:
            Human-readable summary.
        """
        model = model_override or self._model
        prompt = f"""Summarize this robot arm execution for the user in one sentence:

Request: {original_request}
Execution: {trace}

Be brief and natural."""
        messages = [{"role": "user", "content": prompt}]
        return self._chat(messages, model=model, temperature=0.1)

    def decide_next_action(
        self,
        goal: str,
        observation: str,
        available_actions: list[str],
        history: list[dict[str, Any]] | None = None,
        model_override: str | None = None,
    ) -> str:
        """Decide the next action given current observation.

        Args:
            goal: the overall goal.
            observation: what the robot just observed/did.
            available_actions: list of available action names.
            history: prior conversation turns.
            model_override: optional model identifier.

        Returns:
            Next action name to execute.
        """
        model = model_override or self._model
        actions_text = ", ".join(available_actions)
        prompt = f"""Given the goal and observation, choose the next action.

Goal: {goal}
Observation: {observation}
Available actions: {actions_text}

Respond with ONLY the action name, nothing else."""
        messages = [{"role": "user", "content": prompt}]
        result = self._chat(messages, model=model, temperature=0.0)
        # Try to match against available actions
        result = result.strip().lower()
        for action in available_actions:
            if action.lower() in result:
                return action
        return result if result in available_actions else available_actions[0] if available_actions else ""
