# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Integration tests for SimStartTool._start_go2 perception + calibration wire-up.

These tests verify that T4 (sim_tool wire-up) correctly assigns agent._perception
and agent._calibration when with_arm=True and an API key is present, without
launching any real subprocess or importing MuJoCo.

All external calls (subprocess.Popen, Go2ROS2Proxy.connect, PiperROS2Proxy.connect,
PiperGripperROS2Proxy.connect, Go2VLMPerception.__init__, time.sleep, atexit.register,
load_config) are monkey-patched to no-ops to prevent OOM or hanging.
"""
from __future__ import annotations

import unittest.mock as mock

# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

_FAKE_POSITION = (0.0, 0.0, 0.28)


def _make_fake_proc():
    """Return a fake subprocess.Popen-like object."""
    proc = mock.MagicMock()
    proc.pid = 12345
    proc.poll.return_value = None  # still running
    return proc


def _make_fake_base(proc):
    """Return a fake Go2ROS2Proxy instance (returned by Go2ROS2Proxy() call)."""
    base = mock.MagicMock()
    base._sim_subprocess = proc
    base._sim_log_fh = mock.MagicMock()
    base.get_position.return_value = _FAKE_POSITION
    base.connect.return_value = None
    return base


def _make_fake_piper_arm():
    arm = mock.MagicMock()
    arm.connect.return_value = None
    return arm


def _make_fake_piper_gripper():
    gripper = mock.MagicMock()
    gripper.connect.return_value = None
    return gripper


def _patch_start_go2_internals(monkeypatch, *, with_arm: bool, api_key: str = "test-key"):
    """Apply all necessary monkey-patches to _start_go2 so no real subprocess is launched.

    Returns (fake_proc, fake_base) so tests can inspect subprocess handles.
    """
    fake_proc = _make_fake_proc()
    fake_base = _make_fake_base(fake_proc)
    fake_piper_arm = _make_fake_piper_arm()
    fake_piper_gripper = _make_fake_piper_gripper()

    # --- subprocess.Popen → returns our fake proc ---
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: fake_proc)

    # --- atexit.register → no-op ---
    import atexit
    monkeypatch.setattr(atexit, "register", lambda *a, **kw: None)

    # --- time.sleep → no-op ---
    import time
    monkeypatch.setattr(time, "sleep", lambda *a: None)

    # --- os.killpg / os.getpgid → no-op (cleanup path) ---
    import os
    monkeypatch.setattr(os, "killpg", lambda *a: None)
    monkeypatch.setattr(os, "getpgid", lambda *a: 1)

    # --- open() for log file --- handled via mocking Popen env already
    # We don't need to mock open() because log_fh is passed to Popen which is mocked.
    # But _start_go2 calls open("/tmp/vector_vnav.log", "w") before Popen.
    monkeypatch.setattr("builtins.open", mock.mock_open())

    # --- Go2ROS2Proxy: constructor returns fake_base ---
    from vector_os_nano.hardware.sim import go2_ros2_proxy
    monkeypatch.setattr(go2_ros2_proxy, "Go2ROS2Proxy", lambda: fake_base)

    # --- PiperROS2Proxy + PiperGripperROS2Proxy ---
    if with_arm:
        try:
            from vector_os_nano.hardware.sim import piper_ros2_proxy as _prox
            monkeypatch.setattr(
                _prox, "PiperROS2Proxy",
                lambda base_proxy=None, scene_xml_path=None: fake_piper_arm,
            )
            monkeypatch.setattr(
                _prox, "PiperGripperROS2Proxy",
                lambda: fake_piper_gripper,
            )
        except (ImportError, AttributeError):
            pass

    # --- Go2VLMPerception.__init__ → no-op (prevents HTTP client creation) ---
    try:
        from vector_os_nano.perception import vlm_go2 as _vg

        def _noop_vlm_init(self, config=None):  # noqa: ANN001
            self._api_key = (config or {}).get("api_key", "")
            self._cost_lock = __import__("threading").Lock()
            self._cost_usd = 0.0

        monkeypatch.setattr(_vg.Go2VLMPerception, "__init__", _noop_vlm_init)
    except (ImportError, AttributeError):
        pass

    # --- load_config → returns config with api_key ---
    from vector_os_nano.core import config as _cfg_mod
    monkeypatch.setattr(
        _cfg_mod,
        "load_config",
        lambda *a, **kw: {"llm": {"api_key": api_key}} if api_key else {},
    )

    # --- os.path.exists for cfg_path → True ---
    _real_exists = os.path.exists
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda p: True if "user.yaml" in str(p) else _real_exists(p),
    )

    # --- Ensure OPENROUTER_API_KEY env var matches the api_key we want ---
    monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
    monkeypatch.setenv("VECTOR_VLM_URL", "")  # clear local URL override

    # --- _build_room_scene_xml → returns a dummy path ---
    try:
        from vector_os_nano.hardware.sim import mujoco_go2 as _mg2
        monkeypatch.setattr(
            _mg2,
            "_build_room_scene_xml",
            lambda **kw: "/tmp/fake_scene.xml",
        )
    except (ImportError, AttributeError):
        pass

    # --- SceneGraph → no-op persist ---
    try:
        from vector_os_nano.core import scene_graph as _sg_mod
        fake_sg = mock.MagicMock()
        fake_sg.stats.return_value = {"rooms": 0, "objects": 0}
        monkeypatch.setattr(_sg_mod, "SceneGraph", lambda **kw: fake_sg)
    except (ImportError, AttributeError):
        pass

    return fake_proc, fake_base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimToolPerceptionWire:

    # ------------------------------------------------------------------
    # T4-1: with_arm=True + api_key present → both wired
    # ------------------------------------------------------------------

    def test_start_go2_with_arm_and_key_wires_perception_and_calibration(
        self, monkeypatch
    ):
        """with_arm=True + api_key → agent._perception is Go2Perception,
        agent._calibration is Go2Calibration, agent._vlm is not None."""
        _patch_start_go2_internals(monkeypatch, with_arm=True, api_key="test-key")

        from vector_os_nano.vcli.tools.sim_tool import SimStartTool
        from vector_os_nano.perception.go2_perception import Go2Perception
        from vector_os_nano.perception.go2_calibration import Go2Calibration

        agent = SimStartTool._start_go2(gui=False, with_arm=True)

        assert isinstance(agent._perception, Go2Perception), (
            f"Expected Go2Perception, got {type(agent._perception)}"
        )
        assert isinstance(agent._calibration, Go2Calibration), (
            f"Expected Go2Calibration, got {type(agent._calibration)}"
        )
        # _vlm (Go2VLMPerception) should also be set
        assert agent._vlm is not None, "agent._vlm should be set when api_key is present"

    # ------------------------------------------------------------------
    # T4-2: with_arm=True, no api_key → perception / calibration not wired
    # ------------------------------------------------------------------

    def test_start_go2_with_arm_no_key_leaves_perception_none(self, monkeypatch):
        """with_arm=True, api_key empty → _perception / _calibration are None,
        agent still builds, skills still registered."""
        _patch_start_go2_internals(monkeypatch, with_arm=True, api_key="")

        from vector_os_nano.vcli.tools.sim_tool import SimStartTool

        agent = SimStartTool._start_go2(gui=False, with_arm=True)

        assert getattr(agent, "_perception", None) is None, (
            "_perception should be None when no api_key"
        )
        assert getattr(agent, "_calibration", None) is None, (
            "_calibration should be None when no api_key"
        )
        # Ensure agent built successfully — skills should be registered
        assert hasattr(agent, "_skill_registry"), "agent should have _skill_registry"

    # ------------------------------------------------------------------
    # T4-3: with_arm=False → perception / calibration not wired
    # ------------------------------------------------------------------

    def test_start_go2_without_arm_does_not_wire_perception(self, monkeypatch):
        """with_arm=False → _perception / _calibration are None/unset, no piper."""
        _patch_start_go2_internals(monkeypatch, with_arm=False, api_key="test-key")

        from vector_os_nano.vcli.tools.sim_tool import SimStartTool

        agent = SimStartTool._start_go2(gui=False, with_arm=False)

        assert getattr(agent, "_perception", None) is None, (
            "_perception should not be wired when with_arm=False"
        )
        assert getattr(agent, "_calibration", None) is None, (
            "_calibration should not be wired when with_arm=False"
        )

    # ------------------------------------------------------------------
    # T4-4: QwenVLMDetector ctor failure → logged, both None, agent alive
    # ------------------------------------------------------------------

    def test_start_go2_perception_ctor_failure_logs_and_continues(
        self, monkeypatch, caplog
    ):
        """If QwenVLMDetector.__init__ raises, _perception and _calibration
        are both None, agent still builds successfully, warning is logged."""
        _patch_start_go2_internals(monkeypatch, with_arm=True, api_key="test-key")

        # Patch QwenVLMDetector to raise
        from vector_os_nano.perception import vlm_qwen as _vq
        monkeypatch.setattr(
            _vq.QwenVLMDetector,
            "__init__",
            lambda self, **kw: (_ for _ in ()).throw(
                RuntimeError("Qwen unavailable")
            ),
        )

        import logging
        from vector_os_nano.vcli.tools.sim_tool import SimStartTool

        with caplog.at_level(logging.WARNING):
            agent = SimStartTool._start_go2(gui=False, with_arm=True)

        assert getattr(agent, "_perception", None) is None, (
            "_perception should be None after ctor failure"
        )
        assert getattr(agent, "_calibration", None) is None, (
            "_calibration should be None after ctor failure"
        )
        # Warning should have been logged
        assert any(
            "Perception wire-up failed" in rec.message or "perception" in rec.message.lower()
            for rec in caplog.records
        ), f"Expected warning log about perception failure. Got: {[r.message for r in caplog.records]}"

    # ------------------------------------------------------------------
    # T4-5: both agent._vlm (Go2VLMPerception) AND agent._perception coexist
    # ------------------------------------------------------------------

    def test_start_go2_vlm_go2_coexists_with_perception(self, monkeypatch):
        """with_arm=True + key → agent._vlm (Go2VLMPerception) AND
        agent._perception (Go2Perception) are both set."""
        _patch_start_go2_internals(monkeypatch, with_arm=True, api_key="test-key")

        from vector_os_nano.vcli.tools.sim_tool import SimStartTool
        from vector_os_nano.perception.go2_perception import Go2Perception

        agent = SimStartTool._start_go2(gui=False, with_arm=True)

        assert agent._vlm is not None, "agent._vlm (Go2VLMPerception) should be set"
        assert isinstance(agent._perception, Go2Perception), (
            "agent._perception should be Go2Perception"
        )
        # Both must be distinct objects serving different roles
        assert agent._vlm is not agent._perception, (
            "_vlm and _perception should be distinct objects"
        )
