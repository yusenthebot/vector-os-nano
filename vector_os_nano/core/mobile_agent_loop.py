"""MobileAgentLoop — multi-step task execution for mobile robots.

Unlike AgentLoop (designed for stationary arm manipulation), MobileAgentLoop
handles spatial reasoning, multi-room planning, VLM-based verification, and
persistent spatial memory.

Execution flow:
    1. PLAN:  Ask LLM to decompose goal into ordered sub-tasks
    2. EXECUTE: For each sub-task, dispatch to the appropriate skill
    3. OBSERVE: After navigation steps, auto-capture scene via VLM
    4. RECORD: Update spatial memory with observations
    5. CHECK:  Evaluate if the goal is complete
    6. REPORT: Generate a summary of what happened

No ROS2 dependency — pure Python + skill registry + VLM.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubTaskResult:
    """Result of one sub-task within a mobile goal."""

    step: int
    action: str             # skill name
    params: dict
    success: bool
    result_data: dict
    observation: str        # VLM scene description (empty if no VLM)
    duration_sec: float


@dataclass(frozen=True)
class MobileGoalResult:
    """Final result of a multi-step mobile goal execution."""

    success: bool
    goal: str
    steps_completed: int
    steps_total: int
    duration_sec: float
    sub_results: list[SubTaskResult]
    summary: str
    rooms_visited: list[str]


# ---------------------------------------------------------------------------
# Plan schema
# ---------------------------------------------------------------------------


@dataclass
class SubTask:
    """One step in a decomposed plan."""

    action: str         # skill name: "navigate", "look", "walk", "turn", ...
    params: dict        # skill parameters
    reason: str = ""    # why this step is needed


_PLAN_PROMPT = """\
You are a robot dog task planner. Decompose the user's goal into a sequence of \
actions the robot can execute.

Available actions:
- navigate(room): Go to a room. room must be one of: living_room, dining_room, \
kitchen, study, master_bedroom, guest_bedroom, bathroom, hallway
- look(): Look around and describe what the robot sees (uses camera + VLM)
- walk(direction, distance): Short movement. direction: forward/backward/left/right
- turn(direction, angle): Turn in place. direction: left/right. angle: degrees
- stand(): Stand up
- sit(): Sit down
- where_am_i(): Report current room

Rules:
1. After navigate(), ALWAYS add look() to observe the destination
2. Keep plans short — max 12 steps
3. If the goal mentions "all rooms" or "patrol", visit each room + look
4. For "go to X and check Y", navigate(X) then look()

{memory_context}

Respond in JSON array format:
[{{"action": "navigate", "params": {{"room": "kitchen"}}, "reason": "go to kitchen"}}, ...]
"""


# ---------------------------------------------------------------------------
# MobileAgentLoop
# ---------------------------------------------------------------------------


class MobileAgentLoop:
    """Multi-step task execution engine for mobile robots.

    Requires:
        agent_ref: Reference to the Agent instance (for skill_registry, config, etc.)
        config: Agent config dict (for LLM settings).

    Optional services (from agent_ref or injected):
        "vlm":            Go2VLMPerception instance for scene understanding
        "spatial_memory": SpatialMemory instance for persistent room tracking
    """

    def __init__(self, agent_ref: Any, config: dict) -> None:
        self._agent = agent_ref
        self._config = config
        loop_cfg = config.get("agent", {}).get("mobile_loop", {})
        self._max_steps: int = loop_cfg.get("max_steps", 20)

    def run(
        self,
        goal: str,
        max_steps: int | None = None,
        on_step: Callable[[str, int, int], None] | None = None,
        on_message: Callable[[str], None] | None = None,
    ) -> MobileGoalResult:
        """Execute a multi-step mobile task.

        Args:
            goal: Natural language goal (e.g. "patrol the house", "go to kitchen").
            max_steps: Override max steps (default from config).
            on_step: Callback(action_name, step_index, total_steps).
            on_message: Callback(str) for status messages.

        Returns:
            MobileGoalResult with full execution trace.
        """
        start_time = time.monotonic()
        cap = max_steps or self._max_steps

        # 1. PLAN
        plan = self._plan(goal)
        if not plan:
            # Fallback: single-step interpretation
            plan = self._fallback_plan(goal)

        plan = plan[:cap]
        total = len(plan)

        if on_message:
            actions = ", ".join(s.action for s in plan)
            on_message(f"Plan ({total} steps): {actions}")

        # 2. EXECUTE each sub-task
        sub_results: list[SubTaskResult] = []
        rooms_visited: list[str] = []

        for i, task in enumerate(plan):
            if on_step:
                on_step(task.action, i, total)

            step_start = time.monotonic()
            result, observation = self._execute_subtask(task, i)
            duration = time.monotonic() - step_start

            sub = SubTaskResult(
                step=i,
                action=task.action,
                params=task.params,
                success=result.success,
                result_data=result.result_data or {},
                observation=observation,
                duration_sec=duration,
            )
            sub_results.append(sub)

            # Track rooms
            room = (result.result_data or {}).get("room")
            if room and room not in rooms_visited:
                rooms_visited.append(room)

            # Abort if robot fell
            if self._robot_fell():
                if on_message:
                    on_message("Robot fell — aborting patrol.")
                break

        # 3. REPORT
        elapsed = time.monotonic() - start_time
        completed = sum(1 for s in sub_results if s.success)
        summary = self._summarize(goal, sub_results, rooms_visited)

        if on_message:
            on_message(summary)

        return MobileGoalResult(
            success=completed > 0,
            goal=goal,
            steps_completed=completed,
            steps_total=total,
            duration_sec=elapsed,
            sub_results=sub_results,
            summary=summary,
            rooms_visited=rooms_visited,
        )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan(self, goal: str) -> list[SubTask]:
        """Ask LLM to decompose goal into sub-tasks."""
        agent = self._agent
        if agent._llm is None:
            return []

        # Build memory context
        memory_ctx = ""
        try:
            from vector_os_nano.core.spatial_memory import SpatialMemory
            sm = getattr(agent, "_spatial_memory", None)
            if sm is not None:
                memory_ctx = f"Spatial memory:\n{sm.get_room_summary()}"
        except Exception:
            pass

        prompt = _PLAN_PROMPT.format(memory_context=memory_ctx)

        try:
            result = agent._llm.chat(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": goal},
                ],
            )
            text = result if isinstance(result, str) else str(result)
            return self._parse_plan(text)
        except Exception as exc:
            logger.warning("[MobileLoop] LLM planning failed: %s", exc)
            return []

    def _parse_plan(self, text: str) -> list[SubTask]:
        """Parse LLM plan response into SubTask list."""
        # Try direct JSON parse
        stripped = text.strip()
        data = None

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            # Try extracting from markdown fence
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", stripped, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

            # Try finding array in text
            if data is None:
                match = re.search(r"\[.*\]", stripped, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        pass

        if not isinstance(data, list):
            logger.warning("[MobileLoop] Could not parse plan: %.200s", text)
            return []

        tasks: list[SubTask] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", ""))
            params = item.get("params", {})
            if not isinstance(params, dict):
                params = {}
            reason = str(item.get("reason", ""))
            if action:
                tasks.append(SubTask(action=action, params=params, reason=reason))

        return tasks

    def _fallback_plan(self, goal: str) -> list[SubTask]:
        """Generate a simple plan without LLM when planning fails.

        Heuristic: if goal contains a room name, navigate + look.
        Otherwise, just look.
        """
        from vector_os_nano.skills.navigate import _resolve_room, _ROOM_CENTERS

        # Check if goal mentions a room
        for word in goal.lower().replace(",", " ").split():
            room = _resolve_room(word)
            if room:
                return [
                    SubTask("navigate", {"room": room}, f"go to {room}"),
                    SubTask("look", {}, "observe the scene"),
                ]

        # Check Chinese room names
        for alias in ["厨房", "卧室", "客厅", "书房", "餐厅", "卫生间", "走廊", "客房"]:
            if alias in goal:
                room = _resolve_room(alias)
                if room:
                    return [
                        SubTask("navigate", {"room": room}, f"go to {room}"),
                        SubTask("look", {}, "observe the scene"),
                    ]

        # Check for "patrol" / "巡逻" keywords
        patrol_keywords = ["patrol", "巡逻", "巡视", "all rooms", "every room", "所有房间"]
        if any(kw in goal.lower() for kw in patrol_keywords):
            tasks: list[SubTask] = []
            for room in _ROOM_CENTERS:
                tasks.append(SubTask("navigate", {"room": room}, f"go to {room}"))
                tasks.append(SubTask("look", {}, f"observe {room}"))
            return tasks

        # Default: just look
        return [SubTask("look", {}, "observe current scene")]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_subtask(
        self, task: SubTask, step_idx: int,
    ) -> tuple[SkillResult, str]:
        """Execute one sub-task and optionally observe the scene.

        Returns (skill_result, observation_text).
        """
        agent = self._agent
        registry = agent._skill_registry
        context = agent._build_context()

        skill = registry.get(task.action)
        if skill is None:
            logger.warning("[MobileLoop] Skill %r not found, skipping", task.action)
            return (
                SkillResult(
                    success=False,
                    error_message=f"Unknown skill: {task.action}",
                    diagnosis_code="skill_not_found",
                ),
                "",
            )

        try:
            result = skill.execute(task.params, context)
        except Exception as exc:
            logger.error("[MobileLoop] Skill %r raised: %s", task.action, exc)
            return (
                SkillResult(success=False, error_message=str(exc)),
                "",
            )

        # Auto-observe after navigation if VLM is available
        observation = ""
        if task.action in ("navigate",) and result.success:
            observation = self._auto_observe(context)

        return result, observation

    def _auto_observe(self, context: Any) -> str:
        """Capture scene via VLM after arriving at a new location.

        Returns scene description string, or empty string on failure.
        """
        vlm = getattr(self._agent, "_vlm", None)
        base = getattr(context, "base", None)
        if vlm is None or base is None:
            return ""

        try:
            frame = base.get_camera_frame()
            scene = vlm.describe_scene(frame)
            room_id = vlm.identify_room(frame)

            # Record in spatial memory
            sm = getattr(self._agent, "_spatial_memory", None)
            if sm is not None:
                obj_names = [o.name for o in scene.objects]
                sm.observe(room_id.room, obj_names, scene.summary)

            logger.info(
                "[MobileLoop] Auto-observe: room=%s objects=%s",
                room_id.room,
                [o.name for o in scene.objects],
            )
            return scene.summary
        except Exception as exc:
            logger.warning("[MobileLoop] Auto-observe failed: %s", exc)
            return ""

    def _robot_fell(self) -> bool:
        """Check if the robot has fallen over."""
        base = self._agent._base
        if base is None:
            return False
        try:
            pos = base.get_position()
            return pos[2] < 0.12
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _summarize(
        self,
        goal: str,
        results: list[SubTaskResult],
        rooms: list[str],
    ) -> str:
        """Generate a concise human-readable summary."""
        total = len(results)
        succeeded = sum(1 for r in results if r.success)
        failed = total - succeeded

        observations = [
            r.observation for r in results if r.observation
        ]

        parts = [f"Goal: {goal}"]
        parts.append(f"Steps: {succeeded}/{total} succeeded")

        if rooms:
            parts.append(f"Rooms visited: {', '.join(rooms)}")

        if failed > 0:
            failed_actions = [
                f"{r.action}({r.params})" for r in results if not r.success
            ]
            parts.append(f"Failed: {', '.join(failed_actions)}")

        if observations:
            parts.append("Observations:")
            for obs in observations:
                parts.append(f"  - {obs}")

        return "\n".join(parts)
