"""ExploreSkill -- autonomous exploration using TARE planner (or dead-reckoning).

Triggers /start_exploration via ROS2 if available, then monitors position
over the configured duration and reports which rooms were visited.

Fallback: when ROS2 is unavailable, visits each room in the room map via
the NavigateSkill dead-reckoning path so the robot still explores usefully.

All ROS2 imports are lazy (inside try/except) so this module is importable
without a ROS2 installation.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.navigate import (
    _ROOM_CENTERS,
    _detect_current_room,
    _distance,
    _navigate_to_waypoint,
)

logger = logging.getLogger(__name__)

_DEFAULT_DURATION: float = 60.0      # seconds
_POSITION_SAMPLE_INTERVAL: float = 2.0   # seconds between position checks
_VISIT_RADIUS: float = 2.5           # metres -- close enough to count as "visited"


@skill(
    aliases=[
        "explore",
        "探索",
        "自主探索",
        "explore the house",
        "look around",
        "四处看看",
    ],
    direct=False,
)
class ExploreSkill:
    """Start autonomous exploration of the house using TARE planner."""

    name: str = "explore"
    description: str = (
        "Start autonomous exploration of the house using TARE planner. "
        "Monitors which rooms are visited and reports a summary."
    )
    parameters: dict = {
        "duration": {
            "type": "number",
            "required": False,
            "default": _DEFAULT_DURATION,
            "description": "Exploration duration in seconds (default 60).",
        },
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"explored": True}
    failure_modes: list[str] = ["no_base", "exploration_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Run exploration for the given duration.

        Args:
            params: Optional ``duration`` (seconds).
            context: SkillContext with base attached.

        Returns:
            SkillResult with rooms_visited list in result_data.
        """
        if context.base is None:
            logger.error("[EXPLORE] No base connected")
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        duration: float = float(params.get("duration", _DEFAULT_DURATION))
        duration = max(5.0, duration)

        logger.info("[EXPLORE] Starting exploration for %.0f s", duration)

        # Attempt to trigger TARE via ROS2
        ros2_active = _try_start_tare_exploration()

        if ros2_active:
            return self._monitor_exploration(context.base, duration)
        else:
            logger.info("[EXPLORE] ROS2 unavailable -- using dead-reckoning fallback")
            return self._dead_reckoning_exploration(context.base, duration)

    # ------------------------------------------------------------------
    # Exploration monitoring (ROS2 / TARE path)
    # ------------------------------------------------------------------

    def _monitor_exploration(self, base: Any, duration: float) -> SkillResult:
        """Sample position every few seconds to track which rooms were visited."""
        visited: set[str] = set()
        deadline = time.time() + duration

        while time.time() < deadline:
            try:
                pos = base.get_position()
                room = _detect_current_room(float(pos[0]), float(pos[1]))
                if room not in visited:
                    visited.add(room)
                    logger.info("[EXPLORE] Entered room: %s", room)
            except Exception as exc:
                logger.warning("[EXPLORE] Position read error: %s", exc)

            time.sleep(_POSITION_SAMPLE_INTERVAL)

        return _build_result(visited, duration, mode="tare")

    # ------------------------------------------------------------------
    # Dead-reckoning fallback
    # ------------------------------------------------------------------

    def _dead_reckoning_exploration(
        self,
        base: Any,
        duration: float,
    ) -> SkillResult:
        """Visit each room in the room map via turn+walk dead-reckoning.

        Respects the time budget -- stops when duration is exceeded.
        """
        visited: set[str] = set()
        deadline = time.time() + duration

        # Determine starting room
        try:
            pos = base.get_position()
            start_room = _detect_current_room(float(pos[0]), float(pos[1]))
            visited.add(start_room)
        except Exception:
            start_room = "hallway"

        # Visit rooms in a fixed order (hallway first as hub, then others)
        visit_order = ["hallway"] + [
            r for r in _ROOM_CENTERS if r != "hallway" and r != start_room
        ]

        for room in visit_order:
            if time.time() >= deadline:
                logger.info("[EXPLORE] Time budget exhausted")
                break

            target = _ROOM_CENTERS[room]

            # Skip if already close
            try:
                pos = base.get_position()
                if _distance(pos[0], pos[1], target[0], target[1]) < _VISIT_RADIUS:
                    visited.add(room)
                    continue
            except Exception:
                pass

            logger.info("[EXPLORE] Navigating to %s", room)
            ok = _navigate_to_waypoint(base, target[0], target[1], room)
            if ok:
                visited.add(room)
                logger.info("[EXPLORE] Visited: %s", room)
            else:
                logger.warning("[EXPLORE] Navigation to %s failed", room)
                break  # robot likely fell -- abort

        return _build_result(visited, duration, mode="dead_reckoning")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_start_tare_exploration() -> bool:
    """Publish True to /start_exploration via ROS2.

    Returns True if published successfully, False if ROS2 is unavailable.
    All imports are lazy so this never raises ImportError at module load time.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Bool

        if not rclpy.ok():
            return False

        node = Node("_explore_skill_oneshot")
        try:
            pub = node.create_publisher(Bool, "/start_exploration", 1)
            msg = Bool()
            msg.data = True
            pub.publish(msg)
            logger.info("[EXPLORE] Published True to /start_exploration")
            return True
        finally:
            node.destroy_node()

    except Exception as exc:
        logger.debug("[EXPLORE] ROS2 start_exploration publish skipped: %s", exc)
        return False


def _build_result(
    visited: set[str],
    duration: float,
    mode: str,
) -> SkillResult:
    """Build a SkillResult summarising the exploration run."""
    rooms_list = sorted(visited)
    total_rooms = len(_ROOM_CENTERS)
    coverage = round(len(visited) / total_rooms * 100.0, 1) if total_rooms else 0.0

    logger.info(
        "[EXPLORE] Done. Visited %d/%d rooms in %.0f s (mode=%s): %s",
        len(visited), total_rooms, duration, mode, rooms_list,
    )

    return SkillResult(
        success=True,
        result_data={
            "rooms_visited": rooms_list,
            "rooms_visited_count": len(visited),
            "total_rooms": total_rooms,
            "coverage_percent": coverage,
            "duration_s": duration,
            "mode": mode,
        },
    )
