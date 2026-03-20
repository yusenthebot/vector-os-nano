"""Skill protocol, registry, and execution context for Vector OS Nano SDK.

The Skill Protocol defines the interface that all skills must implement.
Users extend this to add custom robot capabilities.

The SkillRegistry discovers and manages registered skills, and serialises
them for LLM planner consumption.

The SkillContext bundles everything a skill needs during execution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from vector_os.core.types import SkillResult

logger = logging.getLogger(__name__)


@runtime_checkable
class Skill(Protocol):
    """Abstract skill interface. Implement this to add new capabilities.

    Attributes must be set as class attributes (or instance attributes on
    __init__) — they are accessed by the registry and executor.

    Example::

        class PickSkill:
            name = "pick"
            description = "Pick up a detected object"
            parameters = {
                "object_id": {"type": "string", "description": "Object to pick"}
            }
            preconditions = ["gripper_empty", "object_visible(obj_001)"]
            postconditions = ["gripper_holding_any"]
            effects = {"gripper_state": "holding"}

            def execute(self, params: dict, context: SkillContext) -> SkillResult:
                ...
    """

    name: str
    description: str
    parameters: dict        # JSON Schema sub-object for parameters
    preconditions: list[str]
    postconditions: list[str]
    effects: dict

    def execute(self, params: dict, context: "SkillContext") -> SkillResult:
        """Execute the skill with given parameters and context.

        Args:
            params: validated parameter dict matching self.parameters schema.
            context: SkillContext providing access to arm, gripper, world model, etc.

        Returns:
            SkillResult indicating success or failure with optional result_data.
        """
        ...


@dataclass
class SkillContext:
    """Everything a skill needs during execution.

    Required fields:
        arm: object implementing ArmProtocol (move_joints, get_joint_positions, etc.)
        gripper: object implementing GripperProtocol (open, close, etc.)
        perception: object implementing PerceptionProtocol, or None
        world_model: WorldModel instance (read/write)
        calibration: Calibration object, or None

    Optional extensions (for future multi-arm / mobile base support):
        arms: dict mapping arm names to arm objects
        base: mobile base object, or None
        config: runtime configuration overrides
    """

    arm: Any
    gripper: Any
    perception: Any
    world_model: Any  # WorldModel — avoids circular import at type level
    calibration: Any
    config: dict = field(default_factory=dict)

    # Future extensions
    arms: dict | None = None     # Multi-arm: {"left": arm, "right": arm}
    base: Any | None = None      # Mobile base


class SkillRegistry:
    """Discovers and manages skills for task execution.

    Skills are registered by name. Duplicate registrations overwrite the
    previous skill with the same name. The registry is the single source
    of truth for what capabilities the robot has at runtime.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill. Overwrites any existing skill with the same name.

        Args:
            skill: object satisfying the Skill protocol.
        """
        self._skills[skill.name] = skill
        logger.debug("SkillRegistry: registered skill %r", skill.name)

    def get(self, name: str) -> Skill | None:
        """Retrieve a skill by name. Returns None if not found."""
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        """Return all registered skill names as a list."""
        return list(self._skills.keys())

    def to_schemas(self) -> list[dict]:
        """Serialize all skill schemas for LLM planner context.

        Each schema dict contains:
            name, description, parameters, preconditions, postconditions, effects

        Returns:
            List of dicts — one per registered skill.
        """
        schemas: list[dict] = []
        for skill in self._skills.values():
            schema = {
                "name": skill.name,
                "description": skill.description,
                "parameters": skill.parameters,
                "preconditions": list(skill.preconditions),
                "postconditions": list(skill.postconditions),
                "effects": dict(skill.effects),
            }
            schemas.append(schema)
        return schemas
