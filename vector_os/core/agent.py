"""Vector OS Nano Agent — the main entry point.

This is the ONE class users interact with. It wires together:
- Hardware (arm + gripper)
- Perception pipeline
- LLM planner
- Skill registry
- World model
- Task executor
"""
from __future__ import annotations

import logging
from typing import Any

from vector_os.core.config import load_config
from vector_os.core.executor import TaskExecutor
from vector_os.core.skill import Skill, SkillContext, SkillRegistry
from vector_os.core.types import ExecutionResult
from vector_os.core.world_model import WorldModel

logger = logging.getLogger(__name__)


class Agent:
    """Natural language robot arm control.

    Usage::

        from vector_os import Agent, SO101
        arm = SO101(port="/dev/ttyACM0")
        agent = Agent(arm=arm, llm_api_key="sk-...")
        agent.execute("pick up the red cup")

    All hardware arguments are optional — Agent degrades gracefully when
    components are absent (useful for unit testing and partial deployments).

    Args:
        arm: Object implementing ArmProtocol.  None disables motion.
        gripper: Object implementing GripperProtocol.  If None and arm is an
            SO101Arm (has ``_bus``), a SO101Gripper is created automatically.
        perception: Object implementing PerceptionProtocol.  None disables
            vision-based skills.
        llm: An LLMProvider instance.  If None and llm_api_key is given, a
            ClaudeProvider is created automatically.
        llm_api_key: Convenience shorthand — ignored when llm= is provided.
        skills: Additional Skill instances to register alongside the built-in
            defaults.
        config: Configuration override — a dict, a path to a YAML file, or
            None to use the built-in defaults.
    """

    def __init__(
        self,
        arm: Any = None,
        gripper: Any = None,
        perception: Any = None,
        llm: Any = None,
        llm_api_key: str | None = None,
        skills: list[Skill] | None = None,
        config: dict | str | None = None,
    ) -> None:
        # ---- Configuration --------------------------------------------------
        if isinstance(config, dict):
            self._config: dict = config
        else:
            self._config = load_config(config)

        # ---- Hardware -------------------------------------------------------
        self._arm = arm
        self._gripper = gripper

        # Auto-create SO101Gripper when arm shares a serial bus and no gripper
        # was explicitly supplied.
        if gripper is None and arm is not None and hasattr(arm, "_bus") and arm._bus is not None:
            try:
                from vector_os.hardware.so101.gripper import SO101Gripper  # lazy

                self._gripper = SO101Gripper(arm._bus)
            except Exception as exc:
                logger.warning("Could not auto-create SO101Gripper: %s", exc)

        # ---- Perception -----------------------------------------------------
        self._perception = perception

        # ---- LLM provider ---------------------------------------------------
        # llm= takes priority over llm_api_key=
        self._llm = llm
        if llm is None and llm_api_key:
            try:
                from vector_os.llm.claude import ClaudeProvider  # lazy

                llm_cfg = self._config.get("llm", {})
                self._llm = ClaudeProvider(
                    api_key=llm_api_key,
                    model=llm_cfg.get("model", "anthropic/claude-sonnet-4-6"),
                    api_base=llm_cfg.get("api_base", "https://openrouter.ai/api/v1"),
                )
            except Exception as exc:
                logger.warning("Could not create ClaudeProvider: %s", exc)

        # ---- World model ----------------------------------------------------
        self._world_model = WorldModel()

        # ---- Skill registry -------------------------------------------------
        self._skill_registry = SkillRegistry()

        # Register built-in skills
        from vector_os.skills import get_default_skills  # lazy-ish (already imported by wave 2)

        for skill in get_default_skills():
            self._skill_registry.register(skill)

        # Register caller-supplied custom skills
        if skills:
            for skill in skills:
                self._skill_registry.register(skill)

        # ---- Executor -------------------------------------------------------
        self._executor = TaskExecutor()

        # ---- Lazy-init state ------------------------------------------------
        self._ik_solver: Any = None
        self._calibration: Any = None

    # -------------------------------------------------------------------------
    # Public API — primary entry point
    # -------------------------------------------------------------------------

    def execute(self, instruction: str) -> ExecutionResult:
        """Execute a natural language instruction.

        Without an LLM, only single-word commands are supported (home, scan,
        pick, place, detect) via ``_execute_direct``.

        With an LLM the flow is:
            1. Build world state snapshot
            2. Ask the LLM to decompose the instruction into a TaskPlan
            3. Execute the plan with the TaskExecutor
            4. Retry on failure (up to ``agent.max_planning_retries`` times)
            5. Return the final ExecutionResult

        Args:
            instruction: Human-readable command string.

        Returns:
            ExecutionResult describing success/failure with trace.
        """
        if self._llm is None:
            return self._execute_direct(instruction)

        max_retries: int = (
            self._config.get("agent", {}).get("max_planning_retries", 3)
        )

        last_result: ExecutionResult | None = None

        for attempt in range(max_retries):
            world_state = self._world_model.to_dict()
            skill_schemas = self._skill_registry.to_schemas()

            plan = self._llm.plan(instruction, world_state, skill_schemas, None)

            if plan.requires_clarification:
                return ExecutionResult(
                    success=False,
                    status="clarification_needed",
                    clarification_question=plan.clarification_question,
                )

            context = self._build_context()
            result = self._executor.execute(plan, self._skill_registry, context)

            if result.success:
                return result

            last_result = result
            logger.warning(
                "[Agent] Attempt %d/%d failed: %s",
                attempt + 1,
                max_retries,
                result.failure_reason,
            )

        # All attempts exhausted
        if last_result is not None:
            return last_result

        return ExecutionResult(
            success=False,
            status="failed",
            failure_reason="All planning attempts exhausted",
        )

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def home(self) -> bool:
        """Move arm to home position.

        Returns:
            True when the home skill succeeds, False otherwise.
        """
        return self.execute("home").success

    def stop(self) -> None:
        """Emergency stop — halts arm motion immediately."""
        if self._arm is not None:
            self._arm.stop()

    def connect(self) -> None:
        """Connect to all hardware peripherals."""
        if self._arm is not None:
            self._arm.connect()
        if self._perception is not None and hasattr(self._perception, "connect"):
            self._perception.connect()

    def disconnect(self) -> None:
        """Disconnect from all hardware peripherals."""
        if self._arm is not None:
            self._arm.disconnect()
        if self._perception is not None and hasattr(self._perception, "disconnect"):
            self._perception.disconnect()

    def register_skill(self, skill: Skill) -> None:
        """Register a custom skill, immediately available to the planner.

        Overwrites any existing skill with the same name.

        Args:
            skill: Object satisfying the Skill protocol.
        """
        self._skill_registry.register(skill)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def world(self) -> WorldModel:
        """The current world model."""
        return self._world_model

    @property
    def skills(self) -> list[str]:
        """Names of all currently registered skills."""
        return self._skill_registry.list_skills()

    # -------------------------------------------------------------------------
    # Context manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> "Agent":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.disconnect()

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _execute_direct(self, instruction: str) -> ExecutionResult:
        """Execute a command without an LLM.

        Parses simple commands of the form ``<verb> [argument]`` and delegates
        to the matching skill.

        Supported verbs: home, scan, pick, place, detect

        Args:
            instruction: Command string.

        Returns:
            ExecutionResult from the skill, or a failure if no skill matches.
        """
        parts = instruction.strip().lower().split(None, 1)
        command = parts[0] if parts else ""
        arg = parts[1] if len(parts) > 1 else ""

        context = self._build_context()
        skill = self._skill_registry.get(command)

        if skill is None:
            return ExecutionResult(
                success=False,
                status="failed",
                failure_reason=(
                    f"Unknown command: {command!r}. No LLM configured."
                ),
            )

        params: dict[str, Any] = {}
        if command == "pick":
            params["object_label"] = arg or "object"
        elif command == "detect":
            params["query"] = arg or "all objects"
        elif command == "place" and arg:
            try:
                coords = [float(v) for v in arg.split()]
                if len(coords) >= 3:
                    params = {"x": coords[0], "y": coords[1], "z": coords[2]}
            except ValueError:
                pass

        skill_result = skill.execute(params, context)

        return ExecutionResult(
            success=skill_result.success,
            status="completed" if skill_result.success else "failed",
            steps_completed=1 if skill_result.success else 0,
            steps_total=1,
            failure_reason=(
                skill_result.error_message if not skill_result.success else None
            ),
        )

    def _build_context(self) -> SkillContext:
        """Build a SkillContext from current agent state.

        IK solver and calibration are initialised lazily on first call.

        Returns:
            SkillContext bundling all resources skills need during execution.
        """
        # Lazy-init IK solver
        if self._ik_solver is None and self._arm is not None:
            try:
                from vector_os.hardware.so101.ik_solver import IKSolver  # lazy

                self._ik_solver = IKSolver()
                if hasattr(self._arm, "set_ik_solver"):
                    self._arm.set_ik_solver(self._ik_solver)
            except Exception as exc:
                logger.debug("IK solver not available: %s", exc)

        # Lazy-init calibration
        if self._calibration is None:
            try:
                from vector_os.perception.calibration import Calibration  # lazy

                cal_file: str = self._config.get("calibration", {}).get("file", "")
                if cal_file:
                    self._calibration = Calibration.load(cal_file)
                else:
                    self._calibration = Calibration()
            except Exception as exc:
                logger.debug("Calibration not available: %s", exc)

        return SkillContext(
            arm=self._arm,
            gripper=self._gripper,
            perception=self._perception,
            world_model=self._world_model,
            calibration=self._calibration,
            config=self._config,
        )
