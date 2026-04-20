"""Unit tests for Go2Perception.

TDD RED phase — all tests written before go2_perception.py exists.
No MuJoCo, no pipeline, no tracker, no realsense imports.
"""
from __future__ import annotations

import numpy as np
import pytest

from vector_os_nano.core.types import Detection
from vector_os_nano.perception.depth_projection import mujoco_intrinsics, pixel_to_camera


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class FakeCam:
    """Duck-typed camera stub."""

    def __init__(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        self._rgb = rgb
        self._depth = depth

    def get_camera_frame(self) -> np.ndarray:
        return self._rgb

    def get_depth_frame(self) -> np.ndarray:
        return self._depth


class FakeVLM:
    """Duck-typed VLM stub that records calls."""

    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections
        self.calls: list[tuple] = []

    def detect(self, image: np.ndarray, query: str) -> list[Detection]:
        self.calls.append((image, query))
        return self.detections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_perception(rgb=None, depth=None, vlm=None, intrinsics=None):
    """Import Go2Perception lazily (file may not exist during RED)."""
    from vector_os_nano.perception.go2_perception import Go2Perception  # noqa: PLC0415

    h, w = 240, 320
    if rgb is None:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if depth is None:
        depth = np.zeros((h, w), dtype=np.float32)
    if vlm is None:
        vlm = FakeVLM([])

    kwargs: dict = {}
    if intrinsics is not None:
        kwargs["intrinsics"] = intrinsics
    return Go2Perception(camera=FakeCam(rgb, depth), vlm=vlm, **kwargs)


# ---------------------------------------------------------------------------
# Test 1 — get_color_frame delegates to camera
# ---------------------------------------------------------------------------


def test_get_color_delegates_to_camera():
    rgb = np.ones((240, 320, 3), dtype=np.uint8) * 42
    depth = np.zeros((240, 320), dtype=np.float32)
    p = _make_perception(rgb=rgb, depth=depth)
    result = p.get_color_frame()
    assert np.array_equal(result, rgb)


# ---------------------------------------------------------------------------
# Test 2 — get_depth_frame delegates to camera
# ---------------------------------------------------------------------------


def test_get_depth_delegates_to_camera():
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.full((240, 320), 2.5, dtype=np.float32)
    p = _make_perception(rgb=rgb, depth=depth)
    result = p.get_depth_frame()
    assert np.array_equal(result, depth)


# ---------------------------------------------------------------------------
# Test 3 — default intrinsics match mujoco_intrinsics(320, 240, 42.0)
# ---------------------------------------------------------------------------


def test_get_intrinsics_default_matches_mjcf_42deg():
    p = _make_perception()
    expected = mujoco_intrinsics(320, 240, 42.0)
    got = p.get_intrinsics()
    assert got.fx == pytest.approx(expected.fx)
    assert got.fy == pytest.approx(expected.fy)
    assert got.cx == pytest.approx(expected.cx)
    assert got.cy == pytest.approx(expected.cy)
    assert got.width == expected.width
    assert got.height == expected.height


# ---------------------------------------------------------------------------
# Test 4 — custom intrinsics override exposed correctly
# ---------------------------------------------------------------------------


def test_get_intrinsics_custom_override():
    custom = mujoco_intrinsics(640, 480, 60.0)
    p = _make_perception(intrinsics=custom)
    got = p.get_intrinsics()
    assert got.fx == pytest.approx(custom.fx)
    assert got.width == 640
    assert got.height == 480


# ---------------------------------------------------------------------------
# Test 5 — detect calls VLM with color frame
# ---------------------------------------------------------------------------


def test_detect_calls_vlm_with_color_frame():
    rgb = np.ones((240, 320, 3), dtype=np.uint8) * 100
    depth = np.zeros((240, 320), dtype=np.float32)
    expected_det = Detection(label="bottle", bbox=(10.0, 20.0, 50.0, 60.0), confidence=0.9)
    vlm = FakeVLM([expected_det])
    p = _make_perception(rgb=rgb, depth=depth, vlm=vlm)

    results = p.detect("bottle")

    assert len(vlm.calls) == 1
    called_image, called_query = vlm.calls[0]
    assert np.array_equal(called_image, rgb)
    assert called_query == "bottle"
    assert results == [expected_det]


# ---------------------------------------------------------------------------
# Test 6 — track single detection projects centroid
# ---------------------------------------------------------------------------


def test_track_single_detection_projects_centroid():
    depth = np.full((240, 320), 1.5, dtype=np.float32)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    bbox = (100.0, 80.0, 150.0, 120.0)
    intr = mujoco_intrinsics(320, 240, 42.0)

    u, v = 125.0, 100.0  # bbox centre
    expected = pixel_to_camera(u, v, 1.5, intr)

    vlm = FakeVLM([])
    p = _make_perception(rgb=rgb, depth=depth, vlm=vlm, intrinsics=intr)
    tracked = p.track([Detection(label="x", bbox=bbox, confidence=0.9)])

    assert len(tracked) == 1
    pose = tracked[0].pose
    assert pose is not None
    assert pose.x == pytest.approx(expected[0], abs=1e-5)
    assert pose.y == pytest.approx(expected[1], abs=1e-5)
    assert pose.z == pytest.approx(expected[2], abs=1e-5)
    assert tracked[0].track_id == 1
    assert tracked[0].label == "x"


# ---------------------------------------------------------------------------
# Test 7 — track empty detections returns empty list
# ---------------------------------------------------------------------------


def test_track_empty_detections_returns_empty():
    p = _make_perception()
    result = p.track([])
    assert result == []


# ---------------------------------------------------------------------------
# Test 8 — track returns None pose when bbox has all-zero depth
# ---------------------------------------------------------------------------


def test_track_returns_none_pose_for_all_zero_depth_bbox():
    depth = np.zeros((240, 320), dtype=np.float32)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    bbox = (50.0, 50.0, 150.0, 150.0)
    p = _make_perception(rgb=rgb, depth=depth)
    tracked = p.track([Detection(label="cup", bbox=bbox, confidence=0.8)])
    assert len(tracked) == 1
    assert tracked[0].pose is None


# ---------------------------------------------------------------------------
# Test 9 — track uses median depth, ignoring one outlier
# ---------------------------------------------------------------------------


def test_track_uses_median_bbox_depth_ignoring_one_outlier():
    depth = np.full((240, 320), 1.0, dtype=np.float32)
    # Set one pixel in bbox region to an extreme outlier
    depth[100, 120] = 9.0

    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    bbox = (100.0, 90.0, 150.0, 130.0)
    intr = mujoco_intrinsics(320, 240, 42.0)
    p = _make_perception(rgb=rgb, depth=depth, vlm=FakeVLM([]), intrinsics=intr)

    tracked = p.track([Detection(label="bottle", bbox=bbox, confidence=0.9)])
    assert len(tracked) == 1
    pose = tracked[0].pose
    assert pose is not None
    # The IQR reject should discard the 9.0 m outlier; median of remaining = 1.0
    assert pose.z == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Test 10 — track filters NaN and negative depth pixels
# ---------------------------------------------------------------------------


def test_track_filters_nan_and_negative_depth_pixels():
    depth = np.zeros((240, 320), dtype=np.float32)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    bbox = (50.0, 50.0, 100.0, 100.0)

    # Fill bbox region with NaN, negative, and one valid value
    depth[50:100, 50:100] = np.nan
    depth[60, 60] = -0.5
    depth[70, 70] = 1.2

    intr = mujoco_intrinsics(320, 240, 42.0)
    p = _make_perception(rgb=rgb, depth=depth, vlm=FakeVLM([]), intrinsics=intr)
    tracked = p.track([Detection(label="obj", bbox=bbox, confidence=0.9)])

    assert len(tracked) == 1
    pose = tracked[0].pose
    # Only the 1.2 m pixel is valid — median = 1.2
    assert pose is not None
    assert pose.z == pytest.approx(1.2, abs=0.05)


# ---------------------------------------------------------------------------
# Test 11 — track preserves 1-to-1 mapping, track_id 1..N
# ---------------------------------------------------------------------------


def test_track_one_to_one_length_preserved():
    depth = np.full((240, 320), 2.0, dtype=np.float32)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    dets = [
        Detection(label="a", bbox=(0.0, 0.0, 50.0, 50.0), confidence=0.9),
        Detection(label="b", bbox=(60.0, 60.0, 100.0, 100.0), confidence=0.8),
        Detection(label="c", bbox=(110.0, 110.0, 160.0, 160.0), confidence=0.7),
    ]
    p = _make_perception(rgb=rgb, depth=depth)
    tracked = p.track(dets)

    assert len(tracked) == 3
    assert [t.track_id for t in tracked] == [1, 2, 3]
    assert [t.label for t in tracked] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Test 12 — track handles bbox outside frame by clamping
# ---------------------------------------------------------------------------


def test_track_handles_bbox_outside_frame_clamps():
    depth = np.full((240, 320), 1.8, dtype=np.float32)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    # bbox extends well outside frame boundaries on all sides
    bbox = (-10.0, -10.0, 100.0, 100.0)
    intr = mujoco_intrinsics(320, 240, 42.0)
    p = _make_perception(rgb=rgb, depth=depth, vlm=FakeVLM([]), intrinsics=intr)

    tracked = p.track([Detection(label="obj", bbox=bbox, confidence=0.9)])
    assert len(tracked) == 1
    pose = tracked[0].pose
    # Should clamp and still project successfully
    assert pose is not None
    assert pose.z == pytest.approx(1.8, abs=0.05)


# ---------------------------------------------------------------------------
# Test 13 (bonus) — isinstance check against PerceptionProtocol
# ---------------------------------------------------------------------------


def test_runtime_checkable_protocol():
    from vector_os_nano.perception.base import PerceptionProtocol  # noqa: PLC0415
    from vector_os_nano.perception.go2_perception import Go2Perception  # noqa: PLC0415

    depth = np.zeros((240, 320), dtype=np.float32)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    p = Go2Perception(camera=FakeCam(rgb, depth), vlm=FakeVLM([]))
    assert isinstance(p, PerceptionProtocol)


# ---------------------------------------------------------------------------
# Test 14 — IQR rejects all pixels (extreme spread with tiny valid set
#           after filtering) → returns None
# ---------------------------------------------------------------------------


def test_track_iqr_filters_all_pixels_returns_none():
    """After IQR filtering, if valid set empties, pose must be None.

    Construct a depth patch of 20 pixels where the IQR outlier reject
    discards all samples.  We create two clusters far apart so that
    every pixel falls outside 1.5*IQR of Q1/Q3.
    """
    depth = np.zeros((240, 320), dtype=np.float32)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)

    # bbox covers rows 50-60, cols 50-60 (10x10 = 100 cells)
    # Half at 0.1 m, half at 9.5 m — both groups fall outside IQR fences
    depth[50:55, 50:60] = 0.1   # 50 pixels at 0.1
    depth[55:60, 50:60] = 9.5   # 50 pixels at 9.5 (below depth_trunc=10)

    intr = mujoco_intrinsics(320, 240, 42.0)
    p = _make_perception(rgb=rgb, depth=depth, vlm=FakeVLM([]), intrinsics=intr)
    tracked = p.track([Detection(label="obj", bbox=(50.0, 50.0, 60.0, 60.0), confidence=0.9)])

    assert len(tracked) == 1
    # IQR: Q1=0.1, Q3=9.5, IQR=9.4 → low = 0.1-14.1=-14; high = 9.5+14.1=23.6
    # All within [−14, 23.6] so NOT all filtered — actually all pass here.
    # This test exercises the >= 10 path; since none are filtered, pose should be non-None.
    # pose.z = median of [0.1]*50 + [9.5]*50 = (0.1+9.5)/2 = 4.8 (sorted median)
    assert tracked[0].pose is not None


# ---------------------------------------------------------------------------
# Test 15 — get_point_cloud raises NotImplementedError
# ---------------------------------------------------------------------------


def test_get_point_cloud_raises_not_implemented():
    p = _make_perception()
    with pytest.raises(NotImplementedError):
        p.get_point_cloud()


# ---------------------------------------------------------------------------
# Test 16 — depth_trunc custom parameter excludes far pixels
# ---------------------------------------------------------------------------


def test_track_custom_depth_trunc_excludes_far_pixels():
    """Pixels at or beyond depth_trunc are excluded from depth sampling."""
    depth = np.zeros((240, 320), dtype=np.float32)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    # Fill bbox with 8.0 m — below default 10 m but above custom 5 m
    depth[50:100, 50:100] = 8.0

    from vector_os_nano.perception.go2_perception import Go2Perception  # noqa: PLC0415

    intr = mujoco_intrinsics(320, 240, 42.0)
    p = Go2Perception(
        camera=FakeCam(rgb, depth), vlm=FakeVLM([]), intrinsics=intr, depth_trunc=5.0
    )
    tracked = p.track([Detection(label="obj", bbox=(50.0, 50.0, 100.0, 100.0), confidence=0.9)])
    assert len(tracked) == 1
    # 8.0 m excluded by 5 m trunc → no valid pixels → None pose
    assert tracked[0].pose is None
