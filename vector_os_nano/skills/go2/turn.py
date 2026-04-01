"""TurnSkill — rotate the Go2 quadruped by a given angle.

Maps direction + angle to a timed yaw velocity command via
context.base.walk(0, 0, vyaw, duration).
"""
from __future__ import annotations

import logging
import math

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

_VYAW_SPEED: float = 0.5  # rad/s

_DEFAULT_DIRECTION: str = "left"
_DEFAULT_ANGLE: float = 90.0  # degrees


@skill(
    aliases=["turn", "rotate", "转", "转弯", "转向", "左转", "右转"],
    direct=False,
)
class TurnSkill:
    """Rotate the Go2 quadruped left or right by a given angle."""

    name: str = "turn"
    description: str = "Rotate the quadruped left or right by a given angle in degrees."
    parameters: dict = {
        "direction": {
            "type": "string",
            "required": False,
            "default": _DEFAULT_DIRECTION,
            "enum": ["left", "right"],
            "description": "Turn direction: left (counter-clockwise) or right (clockwise)",
        },
        "angle": {
            "type": "number",
            "required": False,
            "default": _DEFAULT_ANGLE,
            "description": "Rotation angle in degrees",
        },
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"is_moving": False}
    failure_modes: list[str] = ["no_base", "turn_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Execute the turn command.

        Args:
            params: direction, angle.
            context: SkillContext with base (Go2) attached.

        Returns:
            SkillResult(success=True) when rotation completes.
            SkillResult(success=False, diagnosis_code="no_base") if base missing.
        """
        if context.base is None:
            logger.error("[TURN] No base connected")
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        direction: str = params.get("direction", _DEFAULT_DIRECTION)
        angle_deg: float = float(params.get("angle", _DEFAULT_ANGLE))
        angle_rad: float = math.radians(abs(angle_deg))

        # Left = positive yaw (counter-clockwise), right = negative yaw (clockwise)
        sign: float = 1.0 if direction == "left" else -1.0
        vyaw: float = sign * _VYAW_SPEED
        duration: float = angle_rad / _VYAW_SPEED

        logger.info(
            "[TURN] direction=%s angle=%.1fdeg vyaw=%.2f duration=%.2fs",
            direction, angle_deg, vyaw, duration,
        )

        ok = context.base.walk(0.0, 0.0, vyaw, duration)

        if not ok:
            return SkillResult(
                success=False,
                error_message="Turn command failed",
                diagnosis_code="turn_failed",
            )

        return SkillResult(
            success=True,
            result_data={
                "direction": direction,
                "angle_deg": angle_deg,
                "angle_rad": round(angle_rad, 4),
            },
        )
