"""ExploreSkill — autonomous house exploration with spatial memory.

Systematically visits rooms, records observations, builds semantic map.
Uses NavStackClient for movement and SpatialMemory for persistence.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

# Default exploration waypoints (matching go2_room.xml house layout)
_DEFAULT_EXPLORATION = [
    (10.0, 5.0, "hallway", "corridor"),
    (3.0, 2.5, "living_room", "room"),
    (3.0, 7.5, "dining_room", "room"),
    (17.0, 2.5, "kitchen", "room"),
    (17.0, 7.5, "study", "room"),
    (3.5, 12.0, "master_bedroom", "room"),
    (16.0, 12.0, "guest_bedroom", "room"),
    (8.5, 12.0, "bathroom", "room"),
]


@skill(
    aliases=["explore", "探索", "逛逛", "看看房子", "explore house", "look around"],
    direct=False,
)
class ExploreSkill:
    """Autonomously explore the house, visiting each room and recording observations."""

    name: str = "explore"
    description: str = (
        "Explore the house autonomously. Visits unvisited rooms, records what is found. "
        "Reports back a summary of discovered rooms and objects."
    )
    parameters: dict = {
        "strategy": {
            "type": "string",
            "required": False,
            "default": "systematic",
            "enum": ["systematic", "nearest"],
            "description": "Exploration strategy: systematic (room by room) or nearest (closest unvisited)",
        },
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"exploration": "updated"}
    failure_modes: list[str] = ["no_base", "navigation_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.base is None:
            return SkillResult(success=False, diagnosis_code="no_base")

        memory = _get_or_create_memory(context)
        strategy = params.get("strategy", "systematic")

        # Initialize exploration waypoints if not set
        if not memory.get_all_locations():
            for x, y, name, cat in _DEFAULT_EXPLORATION:
                memory.add_location(name, x, y, category=cat)
            memory.set_exploration_waypoints(
                [(x, y, name) for x, y, name, _ in _DEFAULT_EXPLORATION]
            )

        # Get nav client
        nav = context.services.get("nav")
        visited = []
        failed = []

        # Visit unvisited rooms
        while True:
            if strategy == "nearest":
                target = _nearest_unvisited(memory, context.base)
            else:
                target = memory.get_next_exploration_target()

            if target is None:
                break  # all explored

            tx, ty, tname = target
            logger.info("[EXPLORE] Navigating to %s (%.1f, %.1f)", tname, tx, ty)

            # Navigate
            ok = False
            if nav and nav.is_available:
                ok = nav.navigate_to(tx, ty, timeout=30.0)
            else:
                # Dead-reckoning fallback
                ok = _dead_reckoning_to(context.base, tx, ty)

            if ok:
                pos = context.base.get_position()
                memory.visit(tname, pos[0], pos[1])
                visited.append(tname)
                logger.info("[EXPLORE] Visited %s", tname)
            else:
                memory.visit(tname, tx, ty)  # mark as visited even if imprecise
                failed.append(tname)
                logger.warning("[EXPLORE] Failed to reach %s", tname)

        return SkillResult(
            success=True,
            result_data={
                "visited": visited,
                "failed": failed,
                "total_known": len(memory.get_all_locations()),
                "total_visited": len(memory.get_visited_locations()),
                "summary": memory.summary_for_llm(),
            },
        )


@skill(
    aliases=["remember", "记住", "mark", "标记"],
    direct=False,
)
class RememberLocationSkill:
    """Remember the current location with a custom name."""

    name: str = "remember_location"
    description: str = "Save the current location with a name for future navigation."
    parameters: dict = {
        "name": {
            "type": "string",
            "required": True,
            "description": "Name for this location (e.g., 'charging_station', 'my_spot')",
        },
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"memory": "updated"}
    failure_modes: list[str] = ["no_base"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.base is None:
            return SkillResult(success=False, diagnosis_code="no_base")

        memory = _get_or_create_memory(context)
        pos = context.base.get_position()
        name = params.get("name", "unnamed")

        memory.add_location(name, pos[0], pos[1], category="landmark",
                           tags=["user_defined"])
        memory.visit(name, pos[0], pos[1])

        return SkillResult(
            success=True,
            result_data={
                "location": name,
                "position": [round(pos[0], 1), round(pos[1], 1)],
            },
        )


@skill(
    aliases=["where am i", "在哪", "我在哪", "位置", "location"],
    direct=True,
)
class WhereAmISkill:
    """Report the robot's current location using spatial memory."""

    name: str = "where_am_i"
    description: str = "Report which room or location the robot is currently in."
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {}
    failure_modes: list[str] = ["no_base"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.base is None:
            return SkillResult(success=False, diagnosis_code="no_base")

        memory = _get_or_create_memory(context)
        pos = context.base.get_position()
        loc_name = memory.current_location_name(pos[0], pos[1])

        return SkillResult(
            success=True,
            result_data={
                "current_location": loc_name or "unknown",
                "position": [round(pos[0], 1), round(pos[1], 1)],
                "memory_summary": memory.summary_for_llm(),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_create_memory(context: SkillContext) -> Any:
    """Get SpatialMemory from context services, or create one."""
    from vector_os_nano.core.spatial_memory import SpatialMemory
    memory = context.services.get("spatial_memory")
    if memory is None:
        memory = SpatialMemory()
        context.services["spatial_memory"] = memory
    return memory


def _nearest_unvisited(memory: Any, base: Any) -> tuple[float, float, str] | None:
    """Find nearest unvisited location."""
    import math
    pos = base.get_position()
    unvisited = memory.get_unvisited_locations()
    if not unvisited:
        return None
    best = min(unvisited, key=lambda l: math.sqrt((l.x - pos[0])**2 + (l.y - pos[1])**2))
    return (best.x, best.y, best.name)


def _dead_reckoning_to(base: Any, tx: float, ty: float) -> bool:
    """Simple turn-and-walk fallback."""
    import math
    pos = base.get_position()
    heading = base.get_heading()
    dx, dy = tx - pos[0], ty - pos[1]
    dist = math.sqrt(dx**2 + dy**2)
    if dist < 0.5:
        return True
    target_angle = math.atan2(dy, dx)
    turn = target_angle - heading
    while turn > math.pi: turn -= 2 * math.pi
    while turn < -math.pi: turn += 2 * math.pi
    if abs(turn) > 0.1:
        base.walk(0, 0, 0.8 if turn > 0 else -0.8, abs(turn) / 0.8)
    base.walk(0.4, 0, 0, dist / 0.4)
    return True
