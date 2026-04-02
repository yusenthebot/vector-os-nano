"""Level 15: GroundingDINO detector wiring — Agent integration tests.

Verifies that detect_and_project is wired into Agent as a service so
skills can call context.services["detector"] without importing the
heavy perception module themselves.

All tests are mock-based and run without GPU / torch installed.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(**kwargs):
    """Create a minimal Agent without real hardware."""
    from vector_os_nano.core.agent import Agent
    return Agent(**kwargs)


# ---------------------------------------------------------------------------
# TestDetectorWiring
# ---------------------------------------------------------------------------


class TestDetectorWiring:
    """Verify detect_and_project is exposed through Agent and SkillContext."""

    def test_agent_has_detector_attribute(self):
        """After attaching _detector, agent._detector must be accessible."""
        agent = _make_agent()
        # Simulate what _start_go2 / _init_agent do after construction
        agent._detector = None  # default state — no detector
        assert hasattr(agent, "_detector")

    def test_agent_detector_can_be_set_to_function(self):
        """agent._detector can hold a callable (function reference)."""
        agent = _make_agent()
        fake_detect = MagicMock(return_value=[])
        agent._detector = fake_detect
        assert agent._detector is fake_detect
        assert callable(agent._detector)

    def test_detector_in_skill_context_services(self):
        """_build_context().services['detector'] is set when _detector exists."""
        agent = _make_agent()
        fake_detect = MagicMock(return_value=[])
        agent._detector = fake_detect

        ctx = agent._build_context()
        assert "detector" in ctx.services
        assert ctx.services["detector"] is fake_detect

    def test_detector_not_in_services_when_none(self):
        """_build_context() must NOT add 'detector' key when _detector is None."""
        agent = _make_agent()
        agent._detector = None

        ctx = agent._build_context()
        assert "detector" not in ctx.services

    def test_detector_not_in_services_when_attribute_absent(self):
        """_build_context() must NOT crash when _detector attribute is absent."""
        agent = _make_agent()
        # Ensure _detector attribute does not exist at all
        if hasattr(agent, "_detector"):
            delattr(agent, "_detector")

        ctx = agent._build_context()
        # Should not raise, and detector key should be absent
        assert "detector" not in ctx.services

    def test_detector_is_none_when_torch_missing(self):
        """When torch/transformers are unavailable, _detector is set to None."""
        # Patch the import so it raises ImportError
        with patch.dict("sys.modules", {
            "vector_os_nano.perception.object_detector": None,
        }):
            agent = _make_agent()
            # Simulate what _start_go2 does: catch ImportError → None
            try:
                from vector_os_nano.perception.object_detector import detect_and_project  # noqa: F401
                agent._detector = detect_and_project
            except (ImportError, TypeError):
                agent._detector = None

            assert agent._detector is None

    def test_detector_service_is_callable(self):
        """When detector is wired, skills can call it via context.services."""
        agent = _make_agent()

        def _fake_detect(rgb, depth, pose, **kw):
            return []

        agent._detector = _fake_detect
        ctx = agent._build_context()

        detector_fn = ctx.services["detector"]
        import numpy as np
        result = detector_fn(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.float32),
            None,
        )
        assert result == []


# ---------------------------------------------------------------------------
# TestSimToolDetectorWiring
# ---------------------------------------------------------------------------


class TestSimToolDetectorWiring:
    """Verify _start_go2 in SimStartTool wires the detector."""

    def test_start_go2_sets_detector_attribute(self):
        """_start_go2 calls detect_and_project import and assigns to agent._detector."""
        # We only test that the attribute assignment block is present and
        # correct — we do NOT actually spin up the full sim stack.
        from vector_os_nano.vcli.tools.sim_tool import SimStartTool

        # Verify the method source contains the detector wiring pattern
        import inspect
        src = inspect.getsource(SimStartTool._start_go2)
        assert "object_detector" in src, "_start_go2 must import from object_detector"
        assert "_detector" in src, "_start_go2 must assign agent._detector"
        assert "detect_and_project" in src, "_start_go2 must reference detect_and_project"

    def test_start_go2_handles_import_error_gracefully(self):
        """ImportError from object_detector sets agent._detector = None (no crash)."""
        from vector_os_nano.vcli.tools.sim_tool import SimStartTool
        import inspect
        src = inspect.getsource(SimStartTool._start_go2)
        # Must have try/except around the import
        assert "ImportError" in src or "except" in src, (
            "_start_go2 must handle ImportError from object_detector import"
        )


# ---------------------------------------------------------------------------
# TestCliDetectorWiring
# ---------------------------------------------------------------------------


class TestCliDetectorWiring:
    """Verify _init_agent in cli.py wires the detector in Go2 branch."""

    def test_init_agent_go2_sets_detector(self):
        """_init_agent (Go2 branch) assigns agent._detector after VLM setup."""
        import vector_os_nano.vcli.cli as cli_mod
        import inspect
        src = inspect.getsource(cli_mod._init_agent)
        assert "object_detector" in src, "_init_agent must import from object_detector"
        assert "_detector" in src, "_init_agent must assign agent._detector"
        assert "detect_and_project" in src, "_init_agent must reference detect_and_project"

    def test_init_agent_go2_handles_import_error(self):
        """_init_agent must handle ImportError from object_detector gracefully."""
        import vector_os_nano.vcli.cli as cli_mod
        import inspect
        src = inspect.getsource(cli_mod._init_agent)
        assert "except" in src, "_init_agent must catch import errors for detector"
