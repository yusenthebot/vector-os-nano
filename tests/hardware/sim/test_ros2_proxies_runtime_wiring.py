"""Unit tests for T4: Ros2Runtime wiring into Go2/Piper ROS2 proxies.

Strategy
--------
- Inject fake rclpy + ROS2 message modules into sys.modules so no live DDS
  is needed.
- Patch ``vector_os_nano.hardware.ros2.runtime.get_ros2_runtime`` to return a
  Mock runtime so we can assert add_node / remove_node calls.
- Reset the singleton (``_runtime = None``) in teardown to isolate tests.
- Control ``VECTOR_SHARED_EXECUTOR`` env var per test.
"""
from __future__ import annotations

import sys
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal fake ROS2 module tree
# ---------------------------------------------------------------------------


def _make_fake_rclpy() -> MagicMock:
    """Return a minimal fake rclpy module."""
    fake_rclpy = MagicMock(name="rclpy")
    fake_rclpy.ok.return_value = True

    fake_node_cls = MagicMock(name="rclpy.node.Node")
    fake_node_instance = MagicMock(name="node_instance")
    fake_node_cls.return_value = fake_node_instance

    fake_node_mod = MagicMock(name="rclpy.node")
    fake_node_mod.Node = fake_node_cls

    fake_qos = MagicMock(name="rclpy.qos")
    fake_qos.QoSProfile = MagicMock(return_value=MagicMock())
    fake_qos.ReliabilityPolicy = MagicMock()
    fake_qos.ReliabilityPolicy.RELIABLE = "RELIABLE"

    fake_rclpy.node = fake_node_mod
    fake_rclpy.qos = fake_qos

    return fake_rclpy


def _make_fake_nav_msgs() -> MagicMock:
    mod = MagicMock(name="nav_msgs")
    mod.msg = MagicMock()
    mod.msg.Odometry = MagicMock
    return mod


def _make_fake_geometry_msgs() -> MagicMock:
    mod = MagicMock(name="geometry_msgs")
    mod.msg = MagicMock()
    mod.msg.Twist = MagicMock
    mod.msg.PointStamped = MagicMock
    return mod


def _make_fake_sensor_msgs() -> MagicMock:
    mod = MagicMock(name="sensor_msgs")
    mod.msg = MagicMock()
    mod.msg.Image = MagicMock
    mod.msg.JointState = MagicMock
    return mod


def _make_fake_std_msgs() -> MagicMock:
    mod = MagicMock(name="std_msgs")
    mod.msg = MagicMock()
    mod.msg.Float64MultiArray = MagicMock
    mod.msg.Float64 = MagicMock
    return mod


def _make_fake_visualization_msgs() -> MagicMock:
    mod = MagicMock(name="visualization_msgs")
    mod.msg = MagicMock()
    mod.msg.MarkerArray = MagicMock
    return mod


def _inject_ros2_modules() -> dict[str, Any]:
    """Inject fake ROS2 modules and return originals for cleanup."""
    fake_rclpy = _make_fake_rclpy()
    originals = {}

    fake_modules: dict[str, Any] = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_rclpy.node,
        "rclpy.qos": fake_rclpy.qos,
        "nav_msgs": _make_fake_nav_msgs(),
        "nav_msgs.msg": _make_fake_nav_msgs().msg,
        "geometry_msgs": _make_fake_geometry_msgs(),
        "geometry_msgs.msg": _make_fake_geometry_msgs().msg,
        "sensor_msgs": _make_fake_sensor_msgs(),
        "sensor_msgs.msg": _make_fake_sensor_msgs().msg,
        "std_msgs": _make_fake_std_msgs(),
        "std_msgs.msg": _make_fake_std_msgs().msg,
        "visualization_msgs": _make_fake_visualization_msgs(),
        "visualization_msgs.msg": _make_fake_visualization_msgs().msg,
        "mujoco": MagicMock(name="mujoco"),
        "yaml": MagicMock(name="yaml"),
    }

    for name, mod in fake_modules.items():
        originals[name] = sys.modules.get(name)
        sys.modules[name] = mod

    return originals


def _restore_modules(originals: dict[str, Any]) -> None:
    for name, mod in originals.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def _make_runtime_mock() -> MagicMock:
    rt = MagicMock(name="Ros2Runtime")
    rt.add_node = MagicMock()
    rt.remove_node = MagicMock()
    return rt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_runtime_singleton():
    """Reset _runtime singleton before and after every test."""
    # Ensure the runtime module is freshly loaded each time
    runtime_mod = sys.modules.get("vector_os_nano.hardware.ros2.runtime")
    if runtime_mod is not None:
        runtime_mod._runtime = None
    yield
    runtime_mod = sys.modules.get("vector_os_nano.hardware.ros2.runtime")
    if runtime_mod is not None:
        runtime_mod._runtime = None


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove VECTOR_SHARED_EXECUTOR from env so tests start from clean state."""
    monkeypatch.delenv("VECTOR_SHARED_EXECUTOR", raising=False)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _unload_proxy_modules() -> None:
    """Remove cached proxy module from sys.modules to force re-import."""
    for key in list(sys.modules.keys()):
        if "go2_ros2_proxy" in key or "piper_ros2_proxy" in key:
            del sys.modules[key]


# ---------------------------------------------------------------------------
# T1: Go2ROS2Proxy — shared runtime path (VECTOR_SHARED_EXECUTOR=1)
# ---------------------------------------------------------------------------


def test_go2_proxy_connect_uses_shared_runtime_when_env_on(monkeypatch):
    """Go2ROS2Proxy.connect() must call runtime.add_node when env=1."""
    monkeypatch.setenv("VECTOR_SHARED_EXECUTOR", "1")
    originals = _inject_ros2_modules()
    _unload_proxy_modules()

    runtime_mock = _make_runtime_mock()

    try:
        with patch(
            "vector_os_nano.hardware.ros2.runtime.get_ros2_runtime",
            return_value=runtime_mock,
        ):
            from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy

            proxy = Go2ROS2Proxy()
            proxy.connect()

            assert runtime_mock.add_node.called, (
                "add_node should have been called on the shared runtime"
            )
            added_node = runtime_mock.add_node.call_args[0][0]
            assert added_node is proxy._node, (
                "add_node must be called with proxy._node"
            )
            assert getattr(proxy, "_shared_runtime_used", False) is True, (
                "_shared_runtime_used must be True after shared-path connect"
            )
    finally:
        _unload_proxy_modules()
        _restore_modules(originals)


# ---------------------------------------------------------------------------
# T2: PiperROS2Proxy — shared runtime path (VECTOR_SHARED_EXECUTOR=1)
# ---------------------------------------------------------------------------


def test_piper_proxy_connect_uses_shared_runtime_when_env_on(monkeypatch):
    """PiperROS2Proxy.connect() must call runtime.add_node when env=1."""
    monkeypatch.setenv("VECTOR_SHARED_EXECUTOR", "1")
    originals = _inject_ros2_modules()
    _unload_proxy_modules()

    runtime_mock = _make_runtime_mock()

    # PiperROS2Proxy loads a MuJoCo model on connect — stub it out
    fake_mj = sys.modules["mujoco"]
    fake_model = MagicMock(name="MjModel")
    fake_model.nv = 30
    fake_model.jnt_qposadr = [0] * 20
    fake_model.jnt_dofadr = [0] * 20
    fake_mj.MjModel.from_xml_path.return_value = fake_model
    fake_mj.MjData.return_value = MagicMock(name="MjData")
    fake_mj.mj_name2id.return_value = 0
    fake_mj.mjtObj = MagicMock()
    fake_mj.mjtObj.mjOBJ_JOINT = 1
    fake_mj.mjtObj.mjOBJ_SITE = 6

    fake_base = MagicMock(name="base_proxy")

    try:
        with patch(
            "vector_os_nano.hardware.ros2.runtime.get_ros2_runtime",
            return_value=runtime_mock,
        ), patch("os.path.exists", return_value=True):
            from vector_os_nano.hardware.sim.piper_ros2_proxy import PiperROS2Proxy

            proxy = PiperROS2Proxy(
                base_proxy=fake_base,
                scene_xml_path="/fake/scene.xml",
            )
            proxy.connect()

            assert runtime_mock.add_node.called, (
                "add_node should have been called on the shared runtime"
            )
            added_node = runtime_mock.add_node.call_args[0][0]
            assert added_node is proxy._node, (
                "add_node must be called with proxy._node"
            )
            assert getattr(proxy, "_shared_runtime_used", False) is True
    finally:
        _unload_proxy_modules()
        _restore_modules(originals)


# ---------------------------------------------------------------------------
# T3: PiperGripperROS2Proxy — shared runtime path (VECTOR_SHARED_EXECUTOR=1)
# ---------------------------------------------------------------------------


def test_piper_gripper_proxy_connect_uses_shared_runtime_when_env_on(monkeypatch):
    """PiperGripperROS2Proxy.connect() must call runtime.add_node when env=1."""
    monkeypatch.setenv("VECTOR_SHARED_EXECUTOR", "1")
    originals = _inject_ros2_modules()
    _unload_proxy_modules()

    runtime_mock = _make_runtime_mock()

    try:
        with patch(
            "vector_os_nano.hardware.ros2.runtime.get_ros2_runtime",
            return_value=runtime_mock,
        ):
            from vector_os_nano.hardware.sim.piper_ros2_proxy import (
                PiperGripperROS2Proxy,
            )

            proxy = PiperGripperROS2Proxy()
            proxy.connect()

            assert runtime_mock.add_node.called, (
                "add_node should have been called on the shared runtime"
            )
            added_node = runtime_mock.add_node.call_args[0][0]
            assert added_node is proxy._node, (
                "add_node must be called with proxy._node"
            )
            assert getattr(proxy, "_shared_runtime_used", False) is True
    finally:
        _unload_proxy_modules()
        _restore_modules(originals)


# ---------------------------------------------------------------------------
# T4: Go2ROS2Proxy — legacy spin path (VECTOR_SHARED_EXECUTOR=0)
# ---------------------------------------------------------------------------


def test_go2_proxy_connect_uses_legacy_spin_when_env_zero(monkeypatch):
    """Go2ROS2Proxy.connect() must NOT call runtime.add_node when env=0."""
    monkeypatch.setenv("VECTOR_SHARED_EXECUTOR", "0")
    originals = _inject_ros2_modules()
    _unload_proxy_modules()

    runtime_mock = _make_runtime_mock()

    started_threads: list[threading.Thread] = []

    def _capture_start(self: threading.Thread) -> None:
        started_threads.append(self)
        # Do NOT call real start — avoid spawning daemon threads in tests
        # Just record that it was requested

    try:
        with patch(
            "vector_os_nano.hardware.ros2.runtime.get_ros2_runtime",
            return_value=runtime_mock,
        ), patch.object(threading.Thread, "start", _capture_start):
            from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy

            proxy = Go2ROS2Proxy()
            proxy.connect()

            assert not runtime_mock.add_node.called, (
                "add_node must NOT be called when VECTOR_SHARED_EXECUTOR=0"
            )
            assert len(started_threads) >= 1, (
                "At least one Thread.start() must have been called (legacy spin)"
            )
            assert getattr(proxy, "_shared_runtime_used", True) is False, (
                "_shared_runtime_used must be False in legacy path"
            )
    finally:
        _unload_proxy_modules()
        _restore_modules(originals)


# ---------------------------------------------------------------------------
# T5: disconnect() calls remove_node on shared runtime
# ---------------------------------------------------------------------------


def test_go2_proxy_disconnect_calls_remove_node(monkeypatch):
    """Go2ROS2Proxy.disconnect() must call runtime.remove_node after connect."""
    monkeypatch.setenv("VECTOR_SHARED_EXECUTOR", "1")
    originals = _inject_ros2_modules()
    _unload_proxy_modules()

    runtime_mock = _make_runtime_mock()

    try:
        with patch(
            "vector_os_nano.hardware.ros2.runtime.get_ros2_runtime",
            return_value=runtime_mock,
        ):
            from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy

            proxy = Go2ROS2Proxy()
            proxy.connect()
            node_before_disconnect = proxy._node

            proxy.disconnect()

            assert runtime_mock.remove_node.called, (
                "remove_node must be called during disconnect"
            )
            removed_node = runtime_mock.remove_node.call_args[0][0]
            assert removed_node is node_before_disconnect
    finally:
        _unload_proxy_modules()
        _restore_modules(originals)
