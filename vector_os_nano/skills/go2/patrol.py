# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""PatrolSkill -- multi-room patrol for the Go2 quadruped.

Navigates to each target room in sequence, calls LookSkill (or DescribeSkill
as fallback) to observe the scene with VLM, and returns a structured report.

Skill registry lookup order:
1. context.services["skill_registry"].get("look")       -- LookSkill (future)
2. context.services["skill_registry"].get("describe")   -- DescribeSkill (fallback)
3. Direct import of NavigateSkill / DescribeSkill       -- last resort

Total patrol timeout: 5 minutes by default, configurable via params or
context.config["patrol_timeout"].

Falls through to the next room on navigation failure; aborts entirely if the
robot falls (base z < 0.12 m).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.navigate import _detect_current_room

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT: float = 300.0   # 5 minutes
_DEFAULT_MAX_ROOMS: int = 8
_FALL_Z_THRESHOLD: float = 0.12   # metres — robot has fallen if z < this


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_skill(registry: Any | None, name: str) -> Any | None:
    """Return a skill from registry, or None if registry absent / skill missing."""
    if registry is None:
        return None
    return registry.get(name)


def _resolve_look_skill(context: SkillContext) -> Any | None:
    """Return the best available observation skill instance.

    Priority:
    1. skill_registry["look"]
    2. skill_registry["describe"]
    3. Direct import of DescribeSkill (no registry)
    """
    registry: Any = context.services.get("skill_registry")

    for candidate in ("look", "describe"):
        skill_inst = _get_skill(registry, candidate)
        if skill_inst is not None:
            return skill_inst

    # Last resort: direct import
    try:
        from vector_os_nano.skills.describe import DescribeSkill
        return DescribeSkill()
    except ImportError:
        return None


def _resolve_navigate_skill(context: SkillContext) -> Any | None:
    """Return NavigateSkill from registry or direct import."""
    registry: Any = context.services.get("skill_registry")
    nav = _get_skill(registry, "navigate")
    if nav is not None:
        return nav

    try:
        from vector_os_nano.skills.navigate import NavigateSkill
        return NavigateSkill()
    except ImportError:
        return None


def _robot_has_fallen(context: SkillContext) -> bool:
    """Return True if base z-position is below the fall threshold."""
    base = context.base
    if base is None:
        return False
    try:
        pos = base.get_position()
        return float(pos[2]) < _FALL_Z_THRESHOLD
    except Exception:
        return False


def _extract_description(look_result: SkillResult) -> str:
    """Pull a human-readable description out of a look/describe SkillResult.

    Key priority (matches actual skill output fields):
    - "summary"     -- LookSkill / DescribeSceneSkill full-description mode
    - "description" -- DescribeSkill caption mode (legacy)
    - "answer"      -- DescribeSkill query mode (legacy)
    """
    if not look_result.success:
        return f"(observation failed: {look_result.error_message})"
    data = look_result.result_data
    for key in ("summary", "description", "answer"):
        value = data.get(key)
        if value:
            return str(value)
    return "(no description returned)"


# ---------------------------------------------------------------------------
# PatrolSkill
# ---------------------------------------------------------------------------

@skill(
    aliases=[
        "patrol",
        "巡逻",
        "巡视",
        "patrol the house",
        "check all rooms",
        "检查所有房间",
    ],
    direct=False,
)
class PatrolSkill:
    """Patrol all rooms in the house, navigating to each and describing what is seen."""

    name: str = "patrol"
    description: str = (
        "Patrol all rooms in the house, navigating to each and describing what is seen."
    )
    parameters: dict = {
        "rooms": {
            "type": "array",
            "items": {"type": "string"},
            "required": False,
            "description": "Specific rooms to patrol (default: all unvisited rooms)",
        },
        "max_rooms": {
            "type": "integer",
            "required": False,
            "default": _DEFAULT_MAX_ROOMS,
            "description": "Maximum rooms to visit in one patrol",
        },
        "timeout": {
            "type": "number",
            "required": False,
            "default": _DEFAULT_TIMEOUT,
            "description": "Total patrol timeout in seconds (default 300)",
        },
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"patrolled": True}
    failure_modes: list[str] = ["no_base", "navigation_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.base is None:
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        timeout: float = float(
            params.get(
                "timeout",
                context.config.get("patrol_timeout", _DEFAULT_TIMEOUT),
            )
        )
        max_rooms: int = int(params.get("max_rooms", _DEFAULT_MAX_ROOMS))

        # Resolve sub-skills
        navigate_skill = _resolve_navigate_skill(context)
        look_skill = _resolve_look_skill(context)

        if navigate_skill is None:
            return SkillResult(
                success=False,
                error_message="NavigateSkill not available",
                diagnosis_code="no_navigate_skill",
            )

        # Determine patrol target list
        rooms_to_visit = self._resolve_rooms(params, context, max_rooms)

        # Execute patrol loop
        return self._run_patrol(
            rooms_to_visit,
            navigate_skill,
            look_skill,
            context,
            timeout,
        )

    # ------------------------------------------------------------------
    # Room selection
    # ------------------------------------------------------------------

    def _resolve_rooms(
        self,
        params: dict,
        context: SkillContext,
        max_rooms: int,
    ) -> list[str]:
        """Determine the ordered list of rooms to patrol.

        Falls back to SceneGraph visited rooms when no explicit list is given.
        """
        memory: Any = context.services.get("spatial_memory")
        all_known_rooms: list[str] = (
            memory.get_visited_rooms() if memory is not None else []
        )

        explicit: list[str] | None = params.get("rooms")
        if explicit:
            # Accept any explicitly-named room; validate against SceneGraph if available
            if all_known_rooms:
                valid = [r for r in explicit if r in all_known_rooms]
            else:
                valid = list(explicit)
            return valid[:max_rooms]

        # Use spatial_memory to prefer unvisited rooms
        if memory is not None and all_known_rooms:
            unvisited = memory.get_unvisited_rooms(all_known_rooms)
            if unvisited:
                return unvisited[:max_rooms]
            # All rooms visited — revisit all
            return all_known_rooms[:max_rooms]

        return all_known_rooms[:max_rooms]

    # ------------------------------------------------------------------
    # Patrol execution
    # ------------------------------------------------------------------

    def _run_patrol(
        self,
        rooms: list[str],
        navigate_skill: Any,
        look_skill: Any | None,
        context: SkillContext,
        timeout: float,
    ) -> SkillResult:
        """Iterate rooms, navigate + observe each one, collect results."""
        deadline: float = time.monotonic() + timeout

        rooms_visited: list[str] = []
        rooms_failed: list[str] = []
        observations: dict[str, str] = {}

        for room in rooms:
            # Global timeout guard
            if time.monotonic() >= deadline:
                logger.warning("[PATROL] Timeout reached — stopping patrol early")
                break

            # Fall detection — abort entire patrol
            if _robot_has_fallen(context):
                logger.error("[PATROL] Robot fell — aborting patrol")
                break

            # --- Navigate ---
            logger.info("[PATROL] Navigating to %s...", room)
            nav_result: SkillResult = navigate_skill.execute(
                {"room": room}, context
            )

            if not nav_result.success:
                logger.warning(
                    "[PATROL] Navigation to %s failed: %s",
                    room,
                    nav_result.error_message,
                )
                rooms_failed.append(room)
                continue

            # --- Observe ---
            if look_skill is not None:
                logger.info("[PATROL] Observing %s...", room)
                look_result: SkillResult = look_skill.execute({}, context)
                description = _extract_description(look_result)
            else:
                description = "(no look skill available)"

            observations[room] = description
            rooms_visited.append(room)

            # Persist observation to spatial memory when available
            memory: Any = context.services.get("spatial_memory")
            if memory is not None:
                try:
                    pos = context.base.get_position()
                    memory.visit(room, float(pos[0]), float(pos[1]))
                    memory.observe(room, [], description=description)
                except Exception as exc:
                    logger.warning("[PATROL] spatial_memory update failed: %s", exc)

        # --- Build report ---
        total = len(rooms)
        visited_count = len(rooms_visited)

        spatial_summary: str = ""
        memory = context.services.get("spatial_memory")
        if memory is not None:
            try:
                spatial_summary = memory.get_room_summary()
            except Exception as exc:
                logger.warning("[PATROL] get_room_summary failed: %s", exc)

        overall_success: bool = visited_count > 0

        return SkillResult(
            success=overall_success,
            result_data={
                "rooms_visited": rooms_visited,
                "rooms_failed": rooms_failed,
                "observations": observations,
                "coverage": f"{visited_count}/{total} rooms visited",
                "spatial_summary": spatial_summary,
            },
            error_message="" if overall_success else "No rooms were successfully patrolled",
            diagnosis_code="" if overall_success else "navigation_failed",
        )
