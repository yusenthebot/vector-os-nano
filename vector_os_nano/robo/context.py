"""Shared CLI context — lazy-initialized agent, skills, and hardware refs.

Passed through Click's ctx.obj so every command group can access the
skill registry, agent pipeline, and hardware without re-initializing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

# Imported at module level so tests can patch it via
# "vector_os_nano.robo.context.create_backend".
try:
    from vector_os_nano.vcli.backends import create_backend  # noqa: F401
except ImportError:
    create_backend = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class RoboContext:
    """Shared state threaded through all Click commands via ctx.obj."""

    console: Console = field(default_factory=Console)
    verbose: bool = False

    # Lazy-initialized — set by connect_*() methods
    _agent: Any = field(default=None, repr=False)
    _base: Any = field(default=None, repr=False)
    _arm: Any = field(default=None, repr=False)
    _gripper: Any = field(default=None, repr=False)
    _perception: Any = field(default=None, repr=False)
    _skill_registry: Any = field(default=None, repr=False)
    _skill_context: Any = field(default=None, repr=False)
    _config: dict = field(default_factory=dict)

    # VectorEngine + Session (lazy-initialized on first use)
    _engine: Any = field(default=None, repr=False)
    _session: Any = field(default=None, repr=False)

    # ----------------------------------------------------------------
    # Skill system access
    # ----------------------------------------------------------------

    @property
    def skill_registry(self) -> Any:
        """Return initialized SkillRegistry with all skills registered."""
        if self._skill_registry is None:
            self._init_skills()
        return self._skill_registry

    @property
    def skill_context(self) -> Any:
        """Return SkillContext wired to current hardware."""
        if self._skill_context is None:
            self._init_skill_context()
        return self._skill_context

    @property
    def agent(self) -> Any:
        """Return Agent instance (lazy-init)."""
        return self._agent

    @property
    def base(self) -> Any:
        return self._base

    @property
    def arm(self) -> Any:
        return self._arm

    # ----------------------------------------------------------------
    # Initialization helpers
    # ----------------------------------------------------------------

    def _init_skills(self) -> None:
        """Initialize SkillRegistry with all built-in skills."""
        from vector_os_nano.core.skill import SkillRegistry

        self._skill_registry = SkillRegistry()

        # Register SO-101 arm skills
        try:
            from vector_os_nano.skills import get_default_skills
            for s in get_default_skills():
                self._skill_registry.register(s)
        except Exception as exc:
            logger.debug("Arm skills not available: %s", exc)

        # Register Go2 quadruped skills
        try:
            from vector_os_nano.skills.go2 import get_go2_skills
            for s in get_go2_skills():
                self._skill_registry.register(s)
        except Exception as exc:
            logger.debug("Go2 skills not available: %s", exc)

    def _init_skill_context(self) -> None:
        """Build SkillContext from connected hardware."""
        from vector_os_nano.core.skill import SkillContext

        bases = {}
        arms = {}
        grippers = {}
        services: dict[str, Any] = {}

        if self._base is not None:
            bases["go2"] = self._base
            # Wire scene graph if available
            if hasattr(self._base, "_scene_graph") and self._base._scene_graph is not None:
                services["spatial_memory"] = self._base._scene_graph

        if self._arm is not None:
            arms["so101"] = self._arm

        if self._gripper is not None:
            grippers["so101"] = self._gripper

        self._skill_context = SkillContext(
            arms=arms,
            grippers=grippers,
            bases=bases,
            services=services,
            config=self._config,
        )

    # ----------------------------------------------------------------
    # VectorEngine access (lazy-init)
    # ----------------------------------------------------------------

    def get_engine(self) -> Any:
        """Return VectorEngine (lazy-init on first call).

        Returns None if initialization fails (e.g., no API key configured).
        """
        if self._engine is None:
            self._init_engine()
        return self._engine

    def get_session(self) -> Any:
        """Return Session (lazy-init on first call)."""
        if self._session is None:
            try:
                from vector_os_nano.vcli.session import create_session
                self._session = create_session()
            except ImportError:
                pass
        return self._session

    def _init_engine(self) -> None:
        """Initialize VectorEngine with all tools and robot context."""
        try:
            from vector_os_nano.vcli.engine import VectorEngine
            from vector_os_nano.vcli.tools import discover_all_tools
            from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
            from vector_os_nano.vcli.config import load_config
            from vector_os_nano.vcli.prompt import build_system_prompt
            from vector_os_nano.vcli.robot_context import RobotContextProvider
            from vector_os_nano.vcli.permissions import PermissionContext
            # Re-import from module attribute so tests can patch it.
            import vector_os_nano.robo.context as _self_mod
            _create_backend = _self_mod.create_backend

            # 1. Build categorized registry
            registry = CategorizedToolRegistry()
            for t in discover_all_tools():
                name = t.name
                if name in ("file_read", "file_write", "file_edit", "bash", "glob", "grep"):
                    registry.register(t, category="code")
                elif name in ("robot_status", "start_simulation", "web_fetch"):
                    registry.register(t, category="system")
                else:
                    registry.register(t, category="default")

            # 2. Register optional tools (graceful degradation)
            try:
                from vector_os_nano.vcli.tools.scene_graph_tool import SceneGraphQueryTool
                registry.register(SceneGraphQueryTool(), category="robot")
            except ImportError:
                pass
            try:
                from vector_os_nano.vcli.tools.ros2_tools import (
                    Ros2TopicsTool,
                    Ros2NodesTool,
                    Ros2LogTool,
                )
                registry.register(Ros2TopicsTool(), category="diag")
                registry.register(Ros2NodesTool(), category="diag")
                registry.register(Ros2LogTool(), category="diag")
            except ImportError:
                pass
            try:
                from vector_os_nano.vcli.tools.nav_tools import NavStateTool, TerrainStatusTool
                registry.register(NavStateTool(), category="diag")
                registry.register(TerrainStatusTool(), category="diag")
            except ImportError:
                pass
            try:
                from vector_os_nano.vcli.tools.reload_tool import SkillReloadTool
                registry.register(SkillReloadTool(), category="system")
            except ImportError:
                pass

            # 3. Wrap robot skills as tools
            try:
                from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills
                if self._agent is not None:
                    for t in wrap_skills(self._agent):
                        registry.register(t, category="robot")
            except ImportError:
                pass

            # 4. Build system prompt with robot context
            sg = None
            if self._base is not None and hasattr(self._base, "_scene_graph"):
                sg = self._base._scene_graph
            robot_ctx = RobotContextProvider(base=self._base, scene_graph=sg)
            system_prompt = build_system_prompt(
                agent=self._agent,
                robot_context=robot_ctx,
            )

            # 5. Create LLM backend
            cfg = load_config()
            api_key = cfg.get("openrouter_api_key") or cfg.get("anthropic_api_key", "")
            model = cfg.get("model", "claude-haiku-4-5")
            provider = cfg.get("provider", "openrouter")
            base_url = cfg.get("base_url", "")
            backend = _create_backend(provider, api_key, model, base_url or None)

            # 6. Create engine
            permissions = PermissionContext()
            self._engine = VectorEngine(
                backend=backend,
                registry=registry,
                system_prompt=system_prompt,
                permissions=permissions,
            )

        except Exception as exc:
            logger.warning("Engine init failed: %s", exc)

    # ----------------------------------------------------------------
    # Hardware connection
    # ----------------------------------------------------------------

    def connect_go2_proxy(self) -> bool:
        """Connect to Go2 via ROS2 proxy (requires running sim/real)."""
        try:
            from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy

            proxy = Go2ROS2Proxy()
            proxy.connect()
            self._base = proxy
            self._skill_context = None  # invalidate cached context
            logger.info("Go2 proxy connected")
            return True
        except Exception as exc:
            logger.warning("Go2 proxy connection failed: %s", exc)
            return False

    def connect_scene_graph(self, layout_path: str | None = None) -> bool:
        """Initialize SceneGraph and wire to proxy."""
        try:
            from vector_os_nano.core.scene_graph import SceneGraph
            import os

            sg = SceneGraph()

            # Load room layout if available
            if layout_path and os.path.isfile(layout_path):
                n = sg.load_layout(layout_path)
                logger.info("SceneGraph loaded %d rooms from %s", n, layout_path)

            # Wire to proxy
            if self._base is not None and hasattr(self._base, "_scene_graph"):
                self._base._scene_graph = sg

            self._skill_context = None  # invalidate
            return True
        except Exception as exc:
            logger.warning("SceneGraph init failed: %s", exc)
            return False

    def execute_skill(self, skill_name: str, params: dict | None = None) -> Any:
        """Execute a skill by name, return SkillResult."""
        skill = self.skill_registry.get(skill_name)
        if skill is None:
            from vector_os_nano.core.types import SkillResult
            return SkillResult(
                success=False,
                error_message=f"Unknown skill: {skill_name}",
                diagnosis_code="unknown_skill",
            )
        return skill.execute(params or {}, self.skill_context)

    def get_status(self) -> dict[str, Any]:
        """Collect hardware/connection status for display."""
        status: dict[str, Any] = {
            "base": None,
            "arm": None,
            "gripper": None,
            "skills": 0,
            "ros2": False,
        }

        if self._base is not None:
            try:
                pos = self._base.get_position()
                status["base"] = {
                    "name": getattr(self._base, "name", "go2"),
                    "position": [round(p, 2) for p in pos],
                    "connected": True,
                }
            except Exception:
                status["base"] = {"name": "go2", "connected": False}

        if self._arm is not None:
            status["arm"] = {"name": "so101", "connected": True}

        if self._gripper is not None:
            status["gripper"] = {"connected": True}

        status["skills"] = len(self.skill_registry.list_skills())

        try:
            import rclpy
            status["ros2"] = rclpy.ok()
        except Exception:
            pass

        return status
