"""Session memory for Vector OS Nano SDK.

Provides a bounded, structured conversation memory that persists across
all agent mode transitions (chat, task, query). Replaces the raw
_conversation_history list in agent.py.

Thread-safety: not thread-safe. Designed for single-threaded agent use.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from vector_os_nano.core.types import ExecutionResult


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryEntry:
    """Single entry in session memory.

    role:       "user" | "assistant" | "system" | "task_result"
    content:    Text content of the entry.
    timestamp:  Unix timestamp (time.time()) at creation.
    entry_type: "chat" | "task" | "query" | "task_result"
    metadata:   Structured extras, e.g. skill name, success flag, world_diff.
    """

    role: str
    content: str
    timestamp: float
    entry_type: str = "chat"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SessionMemory
# ---------------------------------------------------------------------------


class SessionMemory:
    """Bounded, structured conversation memory across all agent modes.

    Maintains a chronological list of MemoryEntry objects.  The list is
    capped at max_entries; when the cap is exceeded the oldest entry is
    discarded (FIFO).

    Provides:
      - add_user_message / add_assistant_message  — raw conversational turns
      - add_task_result                           — condensed task execution record
      - get_llm_history                           — formatted for LLM API calls
      - get_last_task_context                     — metadata for reference resolution
    """

    def __init__(self, max_entries: int = 50) -> None:
        self._entries: list[MemoryEntry] = []
        self._max_entries: int = max_entries

    # ------------------------------------------------------------------
    # Public add methods
    # ------------------------------------------------------------------

    def add_user_message(self, content: str, entry_type: str = "chat") -> None:
        """Record a user message.

        Args:
            content:    Message text.
            entry_type: "chat" | "task" | "query"
        """
        entry = MemoryEntry(
            role="user",
            content=content,
            timestamp=time.time(),
            entry_type=entry_type,
            metadata={"intent": entry_type, "instruction": content},
        )
        self._entries.append(entry)
        self._trim()

    def add_assistant_message(self, content: str, entry_type: str = "chat") -> None:
        """Record an assistant response.

        Args:
            content:    Message text.
            entry_type: "chat" | "task" | "query"
        """
        entry = MemoryEntry(
            role="assistant",
            content=content,
            timestamp=time.time(),
            entry_type=entry_type,
        )
        self._entries.append(entry)
        self._trim()

    def add_task_result(
        self,
        instruction: str,
        result: ExecutionResult,
        world_diff: dict[str, Any] | None = None,
    ) -> None:
        """Record a completed task execution as a condensed summary.

        Builds a human-readable summary and stores it as a task_result entry
        with role="assistant" so it flows naturally into LLM history.

        Args:
            instruction: The original user instruction text.
            result:      ExecutionResult from the task executor.
            world_diff:  Optional dict of world state changes to include.
        """
        summary = _build_task_summary(instruction, result, world_diff)
        metadata: dict[str, Any] = {
            "instruction": instruction,
            "success": result.success,
            "status": result.status,
            "steps_completed": result.steps_completed,
            "steps_total": result.steps_total,
        }
        if result.failure_reason:
            metadata["failure_reason"] = result.failure_reason
        if world_diff:
            metadata["world_diff"] = world_diff

        entry = MemoryEntry(
            role="task_result",
            content=summary,
            timestamp=time.time(),
            entry_type="task_result",
            metadata=metadata,
        )
        self._entries.append(entry)
        self._trim()

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_llm_history(self, max_turns: int = 20) -> list[dict[str, str]]:
        """Return conversation history formatted for LLM API calls.

        Rules:
          - Only "user" and "assistant" roles are included.
          - "task_result" entries are mapped to role="assistant".
          - "system" entries are excluded.
          - The most recent max_turns entries (after role filtering) are returned.

        Returns:
            List of {"role": "user"|"assistant", "content": str}.
        """
        formatted: list[dict[str, str]] = []
        for entry in self._entries:
            if entry.role == "system":
                continue
            role = "assistant" if entry.role == "task_result" else entry.role
            formatted.append({"role": role, "content": entry.content})

        return formatted[-max_turns:]

    def get_last_task_context(self) -> dict[str, Any] | None:
        """Return metadata of the most recent task_result entry.

        Used by the planner to resolve pronouns ("it", "that one") by
        inspecting what the last executed task did.

        Returns:
            Metadata dict of the last task_result, or None if no task has
            been executed in this session.
        """
        for entry in reversed(self._entries):
            if entry.role == "task_result":
                return dict(entry.metadata)
        return None

    def clear(self) -> None:
        """Remove all memory entries."""
        self._entries.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[MemoryEntry]:
        """Return a shallow copy of all entries (preserves immutability)."""
        return list(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim(self) -> None:
        """Drop the oldest entries when the list exceeds max_entries."""
        excess = len(self._entries) - self._max_entries
        if excess > 0:
            del self._entries[:excess]


# ---------------------------------------------------------------------------
# Task summary builder (module-level, keeps methods short)
# ---------------------------------------------------------------------------


def _build_task_summary(
    instruction: str,
    result: ExecutionResult,
    world_diff: dict[str, Any] | None,
) -> str:
    """Build a condensed one-line summary of a task execution result.

    Format (success):
        "Task: <instruction> — completed. Steps: s1(ok) -> s2(ok). World: k=v."

    Format (failure):
        "Task: <instruction> — failed at <skill>. Steps: s1(ok) -> s2(failed: reason). World: no changes."
    """
    status_label = result.status if result.status else ("completed" if result.success else "failed")

    # Build step trace string
    step_parts: list[str] = []
    for step in result.trace:
        step_status = _normalise_step_status(step.status)
        if step_status == "failed" and step.error:
            step_parts.append(f"{step.skill_name}(failed: {step.error})")
        else:
            step_parts.append(f"{step.skill_name}({step_status})")
    steps_str = " -> ".join(step_parts) if step_parts else "no steps"

    # Determine failure context
    if not result.success and result.failed_step:
        status_label = f"failed at {result.failed_step.skill_name}"

    # Build world diff string
    world_str = _build_world_str(world_diff)

    return f"Task: {instruction} — {status_label}. Steps: {steps_str}. World: {world_str}."


def _normalise_step_status(status: str) -> str:
    """Normalise verbose step status strings to short tokens.

    Maps "success" / "precondition_failed" / "execution_failed" /
    "postcondition_failed" / "skipped" to "ok" / "failed" / "skipped".
    """
    if status in ("success", "ok"):
        return "ok"
    if status == "skipped":
        return "skipped"
    return "failed"


def _build_world_str(world_diff: dict[str, Any] | None) -> str:
    """Format world_diff dict as a compact key=value string."""
    if not world_diff:
        return "no changes"
    parts = [f"{k}={v}" for k, v in world_diff.items()]
    return ", ".join(parts)
