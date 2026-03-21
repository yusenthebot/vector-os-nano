"""Vector OS Nano — Python SDK for intelligent robot arm control.

Quick start:

    from vector_os_nano import Agent, SO101, Skill, SkillResult

    arm = SO101(port="/dev/ttyACM0")
    agent = Agent(arm=arm, llm_api_key="sk-...")
    agent.execute("pick up the red cup")

The Agent, SO101, Skill, and SkillResult names are the four public entry
points. Everything else is importable from sub-packages but not part of
the stable public API in v0.1.
"""
from __future__ import annotations

from vector_os_nano.version import __version__
from vector_os_nano.core.agent import Agent
from vector_os_nano.core.skill import Skill
from vector_os_nano.core.types import ExecutionResult, SkillResult

try:
    from vector_os_nano.hardware.so101.arm import SO101Arm as SO101
except ImportError:
    SO101 = None  # type: ignore[assignment, misc]

__all__ = [
    "__version__",
    "Agent",
    "ExecutionResult",
    "SO101",
    "Skill",
    "SkillResult",
]
