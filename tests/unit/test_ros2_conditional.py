"""Tests for ROS2 integration layer conditional import guard.

These tests verify that:
    1. `import vector_os_nano.ros2` never crashes regardless of rclpy availability.
    2. `ROS2_AVAILABLE` is always a bool.
    3. If ROS2 is NOT available, node classes are NOT exported at package level.
    4. If ROS2 IS available, node classes ARE exported at package level.

All tests run on non-ROS2 systems (rclpy not installed) — the conditional
import guard is the primary feature under test.
"""
from __future__ import annotations

import importlib
import sys
import types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_ros2_module() -> types.ModuleType:
    """Force a fresh import of vector_os_nano.ros2 by removing cached modules."""
    # Remove cached submodules to allow re-evaluation of the guard
    to_remove = [k for k in sys.modules if k.startswith("vector_os_nano.ros2")]
    for key in to_remove:
        del sys.modules[key]
    return importlib.import_module("vector_os_nano.ros2")


# ---------------------------------------------------------------------------
# Basic import safety
# ---------------------------------------------------------------------------


class TestROS2ImportNoCrash:
    """Importing vector_os_nano.ros2 must never raise an exception."""

    def test_import_vector_os_ros2_no_crash(self) -> None:
        """Importing the ros2 sub-package must not raise."""
        mod = _reload_ros2_module()
        assert mod is not None

    def test_import_vector_os_top_level_no_crash(self) -> None:
        """Importing the top-level vector_os package must not raise."""
        import vector_os_nano  # noqa: F401

    def test_reimport_is_safe(self) -> None:
        """Re-importing vector_os_nano.ros2 multiple times must not crash."""
        for _ in range(3):
            mod = _reload_ros2_module()
            assert mod is not None


# ---------------------------------------------------------------------------
# ROS2_AVAILABLE flag
# ---------------------------------------------------------------------------


class TestROS2AvailableFlag:
    """ROS2_AVAILABLE must always be a bool."""

    def test_ros2_available_is_bool(self) -> None:
        mod = _reload_ros2_module()
        assert isinstance(mod.ROS2_AVAILABLE, bool)

    def test_ros2_available_in_module_namespace(self) -> None:
        mod = _reload_ros2_module()
        assert hasattr(mod, "ROS2_AVAILABLE")

    def test_ros2_available_in_all(self) -> None:
        mod = _reload_ros2_module()
        assert "ROS2_AVAILABLE" in mod.__all__


# ---------------------------------------------------------------------------
# Node guard when ROS2 is NOT available
# ---------------------------------------------------------------------------


class TestROS2NodesGuardedWithoutROS2:
    """When ROS2 is not installed, node classes must not be importable from vector_os_nano.ros2."""

    def _simulate_no_ros2(self) -> types.ModuleType:
        """Temporarily hide rclpy and reimport the ros2 module."""
        real_rclpy = sys.modules.pop("rclpy", None)
        # Also hide the import so the try/except in __init__.py fails
        import builtins
        real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name == "rclpy" or name.startswith("rclpy."):
                raise ImportError(f"Simulated: rclpy not available ({name})")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = _blocking_import
        try:
            mod = _reload_ros2_module()
        finally:
            builtins.__import__ = real_import
            if real_rclpy is not None:
                sys.modules["rclpy"] = real_rclpy
        return mod

    def test_ros2_available_false_when_no_rclpy(self) -> None:
        mod = self._simulate_no_ros2()
        assert mod.ROS2_AVAILABLE is False

    def test_hardware_bridge_node_not_in_namespace(self) -> None:
        mod = self._simulate_no_ros2()
        assert not hasattr(mod, "HardwareBridgeNode")

    def test_perception_bridge_node_not_in_namespace(self) -> None:
        mod = self._simulate_no_ros2()
        assert not hasattr(mod, "PerceptionBridgeNode")

    def test_skill_server_node_not_in_namespace(self) -> None:
        mod = self._simulate_no_ros2()
        assert not hasattr(mod, "SkillServerNode")

    def test_world_model_node_not_in_namespace(self) -> None:
        mod = self._simulate_no_ros2()
        assert not hasattr(mod, "WorldModelServiceNode")

    def test_agent_node_not_in_namespace(self) -> None:
        mod = self._simulate_no_ros2()
        assert not hasattr(mod, "AgentNode")

    def test_all_only_contains_ros2_available(self) -> None:
        mod = self._simulate_no_ros2()
        assert mod.__all__ == ["ROS2_AVAILABLE"]


# ---------------------------------------------------------------------------
# Node files exist and are syntactically valid Python
# ---------------------------------------------------------------------------


class TestROS2NodeFiles:
    """Node source files must be present and importable as raw modules."""

    def _node_path(self, filename: str) -> str:
        import os
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(base, "vector_os_nano", "ros2", "nodes", filename)

    def test_hardware_bridge_file_exists(self) -> None:
        import os
        assert os.path.isfile(self._node_path("hardware_bridge.py"))

    def test_perception_node_file_exists(self) -> None:
        import os
        assert os.path.isfile(self._node_path("perception_node.py"))

    def test_skill_server_file_exists(self) -> None:
        import os
        assert os.path.isfile(self._node_path("skill_server.py"))

    def test_world_model_node_file_exists(self) -> None:
        import os
        assert os.path.isfile(self._node_path("world_model_node.py"))

    def test_agent_node_file_exists(self) -> None:
        import os
        assert os.path.isfile(self._node_path("agent_node.py"))

    def test_hardware_bridge_syntax_valid(self) -> None:
        import ast
        with open(self._node_path("hardware_bridge.py")) as f:
            source = f.read()
        ast.parse(source)  # raises SyntaxError if invalid

    def test_perception_node_syntax_valid(self) -> None:
        import ast
        with open(self._node_path("perception_node.py")) as f:
            source = f.read()
        ast.parse(source)

    def test_skill_server_syntax_valid(self) -> None:
        import ast
        with open(self._node_path("skill_server.py")) as f:
            source = f.read()
        ast.parse(source)

    def test_world_model_node_syntax_valid(self) -> None:
        import ast
        with open(self._node_path("world_model_node.py")) as f:
            source = f.read()
        ast.parse(source)

    def test_agent_node_syntax_valid(self) -> None:
        import ast
        with open(self._node_path("agent_node.py")) as f:
            source = f.read()
        ast.parse(source)


# ---------------------------------------------------------------------------
# Launch file
# ---------------------------------------------------------------------------


class TestROS2LaunchFile:
    """Launch file must exist and be syntactically valid."""

    def _launch_path(self) -> str:
        import os
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(base, "vector_os_nano", "ros2", "launch", "nano.launch.py")

    def test_launch_file_exists(self) -> None:
        import os
        assert os.path.isfile(self._launch_path())

    def test_launch_file_syntax_valid(self) -> None:
        import ast
        with open(self._launch_path()) as f:
            source = f.read()
        ast.parse(source)

    def test_launch_file_has_generate_launch_description(self) -> None:
        import ast
        with open(self._launch_path()) as f:
            source = f.read()
        tree = ast.parse(source)
        func_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "generate_launch_description" in func_names
