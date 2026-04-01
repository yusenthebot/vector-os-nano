"""SimStartTool — start/stop robot simulations at runtime.

Allows V to spin up MuJoCo simulations mid-conversation without
requiring --sim or --sim-go2 flags at startup.

Supported simulations:
  arm   — SO-101 6-DOF arm (MuJoCoArm)
  go2   — Unitree Go2 quadruped (MuJoCoGo2)
"""
from __future__ import annotations

from typing import Any

from vector_os_nano.vcli.tools.base import (
    PermissionResult,
    ToolContext,
    ToolResult,
    tool,
)


@tool(
    name="start_simulation",
    description="Start a robot simulation (arm or go2 quadruped). No restart needed.",
    read_only=False,
    permission="ask",
)
class SimStartTool:
    """Start a MuJoCo simulation and register its skills into the tool registry."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "sim_type": {
                "type": "string",
                "enum": ["arm", "go2"],
                "description": "Which simulation to start: 'arm' (SO-101) or 'go2' (Unitree Go2)",
            },
            "gui": {
                "type": "boolean",
                "description": "Open the MuJoCo viewer window (default: true)",
                "default": True,
            },
        },
        "required": ["sim_type"],
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        sim_type: str = params["sim_type"]
        gui: bool = params.get("gui", True)
        app = context.app_state
        if app is None:
            return ToolResult(content="No app state available", is_error=True)

        # Check if already running
        current_agent = app.get("agent")
        if current_agent is not None:
            current_arm = getattr(current_agent, "_arm", None)
            current_base = getattr(current_agent, "_base", None)
            if sim_type == "arm" and current_arm is not None:
                return ToolResult(content=f"Arm sim already running: {type(current_arm).__name__}")
            if sim_type == "go2" and current_base is not None:
                return ToolResult(content=f"Go2 sim already running: {type(current_base).__name__}")

        try:
            if sim_type == "arm":
                agent = self._start_arm(gui=gui)
            elif sim_type == "go2":
                agent = self._start_go2(gui=gui)
            else:
                return ToolResult(content=f"Unknown sim type: {sim_type}", is_error=True)
        except Exception as exc:
            return ToolResult(content=f"Failed to start {sim_type} sim: {exc}", is_error=True)

        # Update app state
        app["agent"] = agent

        # Register skill tools
        registry = app.get("registry")
        if registry is not None:
            from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills
            for skill_tool in wrap_skills(agent):
                registry.register(skill_tool)

        # Rebuild system prompt
        engine = app.get("engine")
        if engine is not None:
            from vector_os_nano.vcli.prompt import build_system_prompt
            engine._system_prompt = build_system_prompt(agent=agent, cwd=context.cwd)

        hw_name = type(getattr(agent, "_arm", None) or getattr(agent, "_base", None)).__name__
        skill_count = len(agent._skill_registry.list_skills()) if hasattr(agent, "_skill_registry") else 0
        return ToolResult(
            content=f"Started {sim_type} simulation: {hw_name}, {skill_count} skills registered."
        )

    @staticmethod
    def _start_arm(gui: bool = True) -> Any:
        from vector_os_nano.core.agent import Agent  # type: ignore[import]
        from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm  # type: ignore[import]
        arm = MuJoCoArm(gui=gui)
        arm.connect()
        return Agent(arm=arm)

    @staticmethod
    def _start_go2(gui: bool = True) -> Any:
        from vector_os_nano.core.agent import Agent  # type: ignore[import]
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # type: ignore[import]
        base = MuJoCoGo2(gui=gui)
        base.connect()
        agent = Agent(base=base)
        # Register Go2-specific skills (walk, turn, explore, navigate, etc.)
        try:
            from vector_os_nano.skills.go2 import get_go2_skills  # type: ignore[import]
            for skill in get_go2_skills():
                agent.register_skill(skill)
        except Exception:
            pass  # Go2 skills optional
        return agent

    def check_permissions(
        self, params: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        return PermissionResult(behavior="ask", reason=f"Start {params.get('sim_type', '?')} simulation?")
