"""Autonomous exploration + spatial memory skills.

ExploreSkill: robot explores autonomously using a grid pattern,
auto-names discovered areas, builds a complete spatial map.
No human input needed — just say "explore".
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)


def _get_memory(context: SkillContext) -> Any:
    from vector_os_nano.core.spatial_memory import SpatialMemory
    mem = context.services.get("spatial_memory")
    if mem is None:
        mem = SpatialMemory()
        context.services["spatial_memory"] = mem
    return mem


def _get_position(context: SkillContext) -> tuple[float, float] | None:
    if context.base is not None:
        try:
            pos = context.base.get_position()
            return (pos[0], pos[1])
        except Exception:
            pass
    nav = context.services.get("nav")
    if nav is not None:
        odom = nav.get_state_estimation()
        if odom:
            return (odom.x, odom.y)
    return None


@skill(
    aliases=["explore", "探索", "自动探索", "逛逛", "看看房子",
             "explore house", "look around", "map the house", "建图"],
    direct=False,
)
class ExploreSkill:
    """Autonomously explore the environment with a grid pattern.

    The robot navigates to a series of waypoints in expanding circles,
    records each location, and builds a spatial map automatically.
    """

    name: str = "explore"
    description: str = (
        "Autonomously explore the environment. The robot navigates in a pattern, "
        "discovers and remembers areas. Just say 'explore' to start."
    )
    parameters: dict = {
        "radius": {
            "type": "number",
            "required": False,
            "default": 10.0,
            "description": "Exploration radius in meters from starting position",
        },
        "step": {
            "type": "number",
            "required": False,
            "default": 5.0,
            "description": "Distance between exploration waypoints in meters",
        },
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"exploration": "updated"}
    failure_modes: list[str] = ["no_nav"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        nav = context.services.get("nav")
        if nav is None or not nav.is_available:
            return SkillResult(success=False, diagnosis_code="no_nav",
                             error_message="Navigation stack not available")

        pos = _get_position(context)
        if pos is None:
            return SkillResult(success=False, diagnosis_code="no_position")

        memory = _get_memory(context)
        radius = float(params.get("radius", 10.0))
        step = float(params.get("step", 5.0))

        # Record starting position
        start_x, start_y = pos
        memory.add_location("start", start_x, start_y, category="landmark")
        memory.visit("start", start_x, start_y)

        # Generate exploration waypoints in a star pattern from current position
        # 8 directions × expanding distances
        directions = [0, 45, 90, 135, 180, 225, 270, 315]
        waypoints = []
        for dist in [step, step * 2]:
            if dist > radius:
                break
            for angle_deg in directions:
                angle = math.radians(angle_deg)
                wx = start_x + dist * math.cos(angle)
                wy = start_y + dist * math.sin(angle)
                name = f"area_{angle_deg}d_{int(dist)}m"
                waypoints.append((wx, wy, name))

        visited = []
        total = len(waypoints)
        logger.info("[EXPLORE] Starting autonomous exploration: %d waypoints", total)

        for i, (wx, wy, name) in enumerate(waypoints):
            logger.info("[EXPLORE] [%d/%d] Navigating to %s (%.1f, %.1f)",
                       i + 1, total, name, wx, wy)

            # Navigate
            nav.navigate_to(wx, wy, timeout=20.0)

            # Record actual position after navigation
            actual = _get_position(context)
            if actual:
                ax, ay = actual
                # Auto-name based on relative direction from start
                dx = ax - start_x
                dy = ay - start_y
                dist_from_start = math.sqrt(dx**2 + dy**2)
                direction = _direction_name(dx, dy)
                auto_name = f"{direction}_{int(dist_from_start)}m"

                memory.add_location(auto_name, ax, ay, category="explored")
                memory.visit(auto_name, ax, ay)
                visited.append(auto_name)

                logger.info("[EXPLORE]   Reached (%.1f, %.1f) = %s", ax, ay, auto_name)

        # Return to start
        logger.info("[EXPLORE] Returning to start")
        nav.navigate_to(start_x, start_y, timeout=20.0)

        final_pos = _get_position(context)
        all_locations = memory.get_all_locations()

        return SkillResult(
            success=True,
            result_data={
                "waypoints_planned": total,
                "areas_discovered": len(visited),
                "discovered": visited,
                "all_known_locations": [l.name for l in all_locations],
                "total_visited": len(memory.get_visited_locations()),
                "summary": memory.summary_for_llm(),
                "start": [round(start_x, 1), round(start_y, 1)],
                "final": [round(final_pos[0], 1), round(final_pos[1], 1)] if final_pos else None,
            },
        )


@skill(
    aliases=["remember", "记住", "mark", "标记", "save location", "保存位置"],
    direct=False,
)
class RememberLocationSkill:
    """Save the current position with a name for future navigation."""

    name: str = "remember_location"
    description: str = (
        "Save the robot's current position with a custom name. "
        "Later you can navigate back with navigate(room=name)."
    )
    parameters: dict = {
        "name": {
            "type": "string",
            "required": True,
            "description": "Name for this location",
        },
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"memory": "updated"}
    failure_modes: list[str] = ["no_position"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        pos = _get_position(context)
        if pos is None:
            return SkillResult(success=False, diagnosis_code="no_position")

        name = params.get("name", "unnamed")
        memory = _get_memory(context)
        memory.add_location(name, pos[0], pos[1], category="landmark", tags=["user_defined"])
        memory.visit(name, pos[0], pos[1])

        # Register in navigate room map for immediate use
        from vector_os_nano.skills.navigate import _ROOM_ALIASES, _ROOM_CENTERS
        _ROOM_ALIASES[name.lower()] = name
        _ROOM_CENTERS[name] = (pos[0], pos[1])

        return SkillResult(
            success=True,
            result_data={
                "saved": name,
                "position": [round(pos[0], 1), round(pos[1], 1)],
            },
        )


@skill(
    aliases=["where am i", "where", "在哪", "我在哪", "位置", "location"],
    direct=True,
)
class WhereAmISkill:
    """Report the robot's current position and known locations."""

    name: str = "where_am_i"
    description: str = "Report current position, nearby locations, and spatial memory summary."
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {}
    failure_modes: list[str] = ["no_position"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        pos = _get_position(context)
        if pos is None:
            return SkillResult(success=False, diagnosis_code="no_position")

        memory = _get_memory(context)
        loc_name = memory.current_location_name(pos[0], pos[1])
        nearest = memory.nearest_location(pos[0], pos[1])

        return SkillResult(
            success=True,
            result_data={
                "position": [round(pos[0], 1), round(pos[1], 1)],
                "current_location": loc_name,
                "nearest": nearest.name if nearest else None,
                "known_locations": [l.name for l in memory.get_all_locations()],
                "memory": memory.summary_for_llm(),
            },
        )


def _direction_name(dx: float, dy: float) -> str:
    """Convert delta (dx, dy) to a compass direction name."""
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    directions = [
        (337.5, "east"), (22.5, "east"),
        (22.5, "northeast"), (67.5, "northeast"),
        (67.5, "north"), (112.5, "north"),
        (112.5, "northwest"), (157.5, "northwest"),
        (157.5, "west"), (202.5, "west"),
        (202.5, "southwest"), (247.5, "southwest"),
        (247.5, "south"), (292.5, "south"),
        (292.5, "southeast"), (337.5, "southeast"),
    ]
    for lo, name in zip(directions[::2], directions[1::2]):
        lo_val = lo[0]
        name_str = name[1] if isinstance(name, tuple) else name
    # Simpler approach
    if angle < 22.5 or angle >= 337.5:
        return "east"
    elif angle < 67.5:
        return "northeast"
    elif angle < 112.5:
        return "north"
    elif angle < 157.5:
        return "northwest"
    elif angle < 202.5:
        return "west"
    elif angle < 247.5:
        return "southwest"
    elif angle < 292.5:
        return "south"
    else:
        return "southeast"
