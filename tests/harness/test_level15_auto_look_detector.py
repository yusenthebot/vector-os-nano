"""Level 15: ExploreSkill auto-look + GroundingDINO detector integration tests.

Tests that the _do_auto_look closure inside ExploreSkill.execute() calls the
GroundingDINO detector service when available, passes detected_objects with
world coords to SceneGraph.observe_with_viewpoint(), and falls back gracefully
to plain VLM object names when the detector is absent or raises.

All tests use mock detector and mock VLM — no GPU required.
Threading is exercised by calling _auto_look directly (the module-level
callback set by set_auto_look), which avoids spinning up a real background
exploration thread.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vector_os_nano.core.scene_graph import SceneGraph
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.perception.object_detector import Detection, RobotPose
from vector_os_nano.skills.go2 import explore as explore_mod
from vector_os_nano.skills.go2.explore import ExploreSkill, set_auto_look


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vlm(room: str = "living_room", confidence: float = 0.9) -> MagicMock:
    """Return a mock VLM with predictable scene + room identification."""
    vlm = MagicMock()

    scene = MagicMock()
    scene.summary = "A bright living room"
    obj1 = MagicMock()
    obj1.name = "sofa"
    obj1.confidence = 0.88
    obj2 = MagicMock()
    obj2.name = "table"
    obj2.confidence = 0.75
    scene.objects = [obj1, obj2]
    vlm.describe_scene.return_value = scene

    room_id = MagicMock()
    room_id.room = room
    room_id.confidence = confidence
    vlm.identify_room.return_value = room_id

    return vlm


def _make_detections() -> list[Detection]:
    """Two Detection objects with non-zero world coordinates."""
    return [
        Detection(
            label="sofa",
            confidence=0.91,
            bbox_u1=50.0, bbox_v1=80.0,
            bbox_u2=200.0, bbox_v2=160.0,
            world_x=4.2, world_y=1.8, world_z=0.3,
            depth_m=2.5,
        ),
        Detection(
            label="table",
            confidence=0.83,
            bbox_u1=120.0, bbox_v1=100.0,
            bbox_u2=220.0, bbox_v2=140.0,
            world_x=3.7, world_y=2.1, world_z=0.1,
            depth_m=1.9,
        ),
    ]


def _make_base(
    position: tuple[float, float, float] = (1.0, 2.0, 0.28),
    heading: float = 0.0,
    depth_frame: np.ndarray | None = None,
) -> MagicMock:
    """Mock base with camera + depth frames."""
    base = MagicMock()
    base.get_position.return_value = position
    base.get_heading.return_value = heading
    base.get_camera_frame.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
    if depth_frame is None:
        depth_frame = np.full((240, 320), 2.0, dtype=np.float32)
    base.get_depth_frame.return_value = depth_frame
    return base


def _make_base_no_depth(
    position: tuple[float, float, float] = (0.0, 0.0, 0.28),
    heading: float = 0.0,
) -> MagicMock:
    """Mock base WITHOUT get_depth_frame attribute (spec-constrained)."""
    spec_attrs = ["get_position", "get_heading", "get_camera_frame", "set_velocity"]
    base = MagicMock(spec=spec_attrs)
    base.get_position.return_value = position
    base.get_heading.return_value = heading
    base.get_camera_frame.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
    return base


def _make_context(
    vlm: Any = None,
    detector: Any = None,
    spatial_memory: Any = None,
    base: Any = None,
) -> SkillContext:
    services: dict[str, Any] = {}
    if vlm is not None:
        services["vlm"] = vlm
    if detector is not None:
        services["detector"] = detector
    if spatial_memory is not None:
        services["spatial_memory"] = spatial_memory

    b = base if base is not None else _make_base()
    return SkillContext(base=b, services=services)


def _install_auto_look(
    vlm: Any,
    detector: Any,
    spatial_memory: Any,
    base: Any,
) -> None:
    """Run ExploreSkill.execute() to install _do_auto_look, then cancel immediately."""
    # Patch the background-thread + nav/bridge launchers so execute() returns fast
    # without touching the filesystem or real threads.
    import threading

    ctx = _make_context(vlm=vlm, detector=detector,
                        spatial_memory=spatial_memory, base=base)

    original_start_bridge = explore_mod._start_bridge_on_go2
    original_launch_nav = explore_mod._launch_nav_stack

    def _fake_bridge(_b: Any) -> bool:
        return False  # no bridge → skip nav stack check

    explore_mod._start_bridge_on_go2 = _fake_bridge

    try:
        # Reset global exploration state
        explore_mod._explore_running = False
        explore_mod._explore_cancel.clear()
        explore_mod._explore_visited.clear()

        ExploreSkill().execute({}, ctx)
        # The background thread is running; cancel it immediately
        explore_mod.cancel_exploration()
    finally:
        explore_mod._start_bridge_on_go2 = original_start_bridge
        explore_mod._launch_nav_stack = original_launch_nav


# ---------------------------------------------------------------------------
# TestAutoLookWithDetector
# ---------------------------------------------------------------------------


class TestAutoLookWithDetector:
    """_do_auto_look calls detector when it is wired via context.services."""

    def test_auto_look_calls_detector(self):
        """Detector function is invoked during auto-look when available."""
        vlm = _make_vlm()
        detector_fn = MagicMock(return_value=_make_detections())
        base = _make_base()

        _install_auto_look(vlm, detector_fn, None, base)

        # Call the installed auto-look callback directly
        assert explore_mod._auto_look is not None
        result = explore_mod._auto_look("living_room")

        assert result is not None
        detector_fn.assert_called_once()

    def test_auto_look_calls_detector_with_correct_args(self):
        """Detector is called with (rgb, depth, RobotPose) where pose matches base."""
        vlm = _make_vlm()
        rgb = np.ones((240, 320, 3), dtype=np.uint8) * 77
        depth = np.full((240, 320), 3.5, dtype=np.float32)
        base = _make_base(position=(5.0, 6.0, 0.28), heading=1.57,
                          depth_frame=depth)
        base.get_camera_frame.return_value = rgb

        detector_fn = MagicMock(return_value=_make_detections())
        _install_auto_look(vlm, detector_fn, None, base)

        explore_mod._auto_look("living_room")

        detector_fn.assert_called_once()
        args = detector_fn.call_args[0]
        called_rgb, called_depth, called_pose = args

        assert np.array_equal(called_rgb, rgb)
        assert np.array_equal(called_depth, depth)
        assert isinstance(called_pose, RobotPose)
        assert called_pose.x == pytest.approx(5.0)
        assert called_pose.y == pytest.approx(6.0)
        assert called_pose.heading == pytest.approx(1.57)

    def test_auto_look_passes_detected_objects_to_scene_graph(self):
        """After auto-look with detector, SceneGraph objects have world coords."""
        vlm = _make_vlm(room="kitchen")
        detections = _make_detections()
        detector_fn = MagicMock(return_value=detections)
        scene_graph = SceneGraph()
        base = _make_base(position=(2.0, 3.0, 0.28), heading=0.0)

        _install_auto_look(vlm, detector_fn, scene_graph, base)
        explore_mod._auto_look("kitchen")

        objects = scene_graph.find_objects_in_room("kitchen")
        assert objects, "No objects stored in scene graph"

        objects_with_coords = [o for o in objects if o.x != 0.0 or o.y != 0.0]
        assert objects_with_coords, (
            "No objects have non-zero world coordinates — "
            "detected_objects not passed to observe_with_viewpoint"
        )

        cats = {o.category for o in objects_with_coords}
        assert "sofa" in cats or "table" in cats

    def test_auto_look_world_coords_match_detector(self):
        """ObjectNode x/y values exactly match Detection.world_x/world_y."""
        vlm = _make_vlm(room="bedroom")
        detections = [
            Detection(
                label="bed",
                confidence=0.92,
                bbox_u1=0, bbox_v1=0, bbox_u2=100, bbox_v2=100,
                world_x=7.1, world_y=3.3, world_z=0.5,
                depth_m=2.0,
            )
        ]
        detector_fn = MagicMock(return_value=detections)
        scene_graph = SceneGraph()
        base = _make_base()

        _install_auto_look(vlm, detector_fn, scene_graph, base)
        explore_mod._auto_look("bedroom")

        beds = scene_graph.find_objects_by_category("bed")
        assert beds, "bed not found in scene graph"
        assert beds[0].x == pytest.approx(7.1, abs=1e-3)
        assert beds[0].y == pytest.approx(3.3, abs=1e-3)

    def test_auto_look_result_contains_world_coords(self):
        """result dict objects entries include world_x, world_y, depth_m keys."""
        vlm = _make_vlm()
        detector_fn = MagicMock(return_value=_make_detections())
        base = _make_base()

        _install_auto_look(vlm, detector_fn, None, base)
        result = explore_mod._auto_look("living_room")

        assert result is not None
        for obj in result["objects"]:
            assert "world_x" in obj
            assert "world_y" in obj
            assert "depth_m" in obj

    def test_auto_look_falls_back_without_detector(self):
        """No detector in services -> VLM names only, no crash, no world_x keys."""
        vlm = _make_vlm()
        base = _make_base()

        # No detector passed
        _install_auto_look(vlm, None, None, base)
        result = explore_mod._auto_look("living_room")

        assert result is not None
        assert result["summary"] == "A bright living room"
        for obj in result["objects"]:
            assert "world_x" not in obj

    def test_auto_look_falls_back_on_detector_exception(self):
        """If detector raises, auto-look logs warning and returns VLM objects."""
        vlm = _make_vlm()
        detector_fn = MagicMock(side_effect=RuntimeError("GPU OOM"))
        base = _make_base()

        _install_auto_look(vlm, detector_fn, None, base)
        result = explore_mod._auto_look("living_room")

        assert result is not None, "auto-look must not return None on detector error"
        # VLM objects returned without world coords
        for obj in result["objects"]:
            assert "world_x" not in obj

    def test_auto_look_falls_back_on_empty_detections(self):
        """Detector returns [] -> result uses VLM object names (no world_x)."""
        vlm = _make_vlm()
        detector_fn = MagicMock(return_value=[])
        base = _make_base()

        _install_auto_look(vlm, detector_fn, None, base)
        result = explore_mod._auto_look("living_room")

        assert result is not None
        # Empty detector → VLM fallback
        names = {o["name"] for o in result["objects"]}
        assert "sofa" in names or "table" in names
        for obj in result["objects"]:
            assert "world_x" not in obj

    def test_auto_look_falls_back_without_depth_frame(self):
        """Base without get_depth_frame -> detector skipped, VLM objects returned."""
        vlm = _make_vlm()
        detector_fn = MagicMock(return_value=_make_detections())
        base = _make_base_no_depth()

        _install_auto_look(vlm, detector_fn, None, base)
        result = explore_mod._auto_look("living_room")

        assert result is not None
        detector_fn.assert_not_called()
        for obj in result["objects"]:
            assert "world_x" not in obj

    def test_auto_look_room_id_still_from_vlm(self):
        """Room identification always comes from VLM identify_room."""
        vlm = _make_vlm(room="study", confidence=0.87)
        detector_fn = MagicMock(return_value=_make_detections())
        base = _make_base()

        _install_auto_look(vlm, detector_fn, None, base)
        result = explore_mod._auto_look("study")

        assert result is not None
        assert result["room"] == "study"
        assert result["room_confidence"] == pytest.approx(0.87, abs=1e-3)
        vlm.identify_room.assert_called_once()
        vlm.describe_scene.assert_called_once()

    def test_auto_look_detector_zero_world_coords_excluded_from_detected_objects(self):
        """Detections with world_x==0 and world_y==0 are filtered from detected_objects."""
        vlm = _make_vlm(room="hallway")
        detections = [
            Detection(
                label="lamp",
                confidence=0.80,
                bbox_u1=0, bbox_v1=0, bbox_u2=50, bbox_v2=50,
                world_x=0.0, world_y=0.0, world_z=0.0,  # no depth hit
                depth_m=0.0,
            ),
            Detection(
                label="chair",
                confidence=0.88,
                bbox_u1=0, bbox_v1=0, bbox_u2=80, bbox_v2=80,
                world_x=5.5, world_y=2.2, world_z=0.3,
                depth_m=1.8,
            ),
        ]
        detector_fn = MagicMock(return_value=detections)
        scene_graph = SceneGraph()
        base = _make_base()

        _install_auto_look(vlm, detector_fn, scene_graph, base)
        explore_mod._auto_look("hallway")

        objects = scene_graph.find_objects_in_room("hallway")
        objects_with_nonzero_coords = [o for o in objects if o.x != 0.0 or o.y != 0.0]
        # lamp has (0,0) and should not appear with world coords
        lamps_with_coords = [o for o in objects_with_nonzero_coords if o.category == "lamp"]
        assert not lamps_with_coords, (
            "lamp with zero world coords should not be in detected_objects"
        )
        # chair should have coords
        chairs = scene_graph.find_objects_by_category("chair")
        assert chairs and chairs[0].x == pytest.approx(5.5, abs=1e-3)
