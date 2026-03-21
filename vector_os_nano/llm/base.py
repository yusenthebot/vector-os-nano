"""LLMProvider Protocol — the interface all LLM providers must satisfy.

This module defines the structural contract. No concrete implementation here.
All providers (ClaudeProvider, OpenAIProvider, LocalProvider) implement
this protocol.

Design notes:
- runtime_checkable enables isinstance() checks for protocol compliance
- plan() takes structured inputs and returns a TaskPlan
- query() is for free-form questions that return a string
- No ROS2 imports anywhere in this subpackage
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vector_os_nano.core.types import TaskPlan


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM-backed task planners and query responders.

    All implementations must provide:
    - plan(): convert a natural-language goal + world state into a TaskPlan
    - query(): answer a free-form question (optionally with an image)

    Example::

        provider = ClaudeProvider(api_key="sk-...")
        assert isinstance(provider, LLMProvider)  # protocol check

        plan = provider.plan(
            goal="pick up the red cup",
            world_state=world_model.to_dict(),
            skill_schemas=registry.to_schemas(),
        )
    """

    def plan(
        self,
        goal: str,
        world_state: dict[str, Any],
        skill_schemas: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
    ) -> TaskPlan:
        """Decompose a goal into a TaskPlan.

        Args:
            goal: natural-language instruction (e.g. "pick up the red cup").
            world_state: serialized world model snapshot.
            skill_schemas: list of skill schemas from SkillRegistry.to_schemas().
            history: optional prior conversation turns for multi-turn planning.

        Returns:
            TaskPlan with steps, or a clarification request if goal is ambiguous.
        """
        ...

    def query(
        self,
        prompt: str,
        image: Any = None,
    ) -> str:
        """Send a free-form prompt and return the LLM's text response.

        Args:
            prompt: natural-language question or instruction.
            image: optional image data (format depends on provider).

        Returns:
            LLM response as a plain string. Never raises — returns an
            error description string on failure.
        """
        ...
