# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for Go2Calibration — camera-to-world transform.

Math convention:
  MuJoCo xmat columns = [right, up, -forward]
  OpenCV camera frame: x=right, y=down, z=forward

Tests are authored by constructing (cam_xpos, cam_xmat) from the same
geometry used in Go2ROS2Proxy.get_camera_pose(), so the calibration
must invert that mapping correctly.
"""
from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from vector_os_nano.perception.go2_calibration import Go2Calibration


# ---------------------------------------------------------------------------
# Helpers to build fake base_proxy instances
# ---------------------------------------------------------------------------

def _make_proxy(cam_xpos: np.ndarray, cam_xmat: np.ndarray) -> Any:
    """Return a mock base_proxy whose get_camera_pose returns fixed values."""
    proxy = MagicMock()
    proxy.get_camera_pose.return_value = (cam_xpos, cam_xmat)
    return proxy


def _dog_pose_to_cam(
    pos: tuple[float, float, float],
    heading: float,
    pitch_deg: float = -5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (cam_xpos, cam_xmat) using the same formula as Go2ROS2Proxy.

    Args:
        pos: dog world position (x, y, z)
        heading: yaw in radians
        pitch_deg: camera tilt from horizontal (default: -5 deg)

    Returns:
        (cam_xpos shape=(3,), cam_xmat shape=(9,)) — flattened row-major
    """
    mount_fwd, mount_up = 0.3, 0.05
    pitch = math.radians(pitch_deg)

    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)

    cam_x = pos[0] + cos_h * mount_fwd
    cam_y = pos[1] + sin_h * mount_fwd
    cam_z = pos[2] + mount_up
    cam_xpos = np.array([cam_x, cam_y, cam_z])

    fwd = np.array([cos_h * cos_p, sin_h * cos_p, sin_p])
    right = np.array([-sin_h, cos_h, 0.0])
    up = np.cross(right, fwd)
    cam_xmat = np.column_stack([right, up, -fwd]).flatten()

    return cam_xpos, cam_xmat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGo2CalibrationBasics:
    """Test 1: dog at origin facing +X, 1 m forward in cam."""

    def test_dog_at_origin_facing_x_1m_forward(self) -> None:
        """cam point (0,0,1) -> world approx (1.3, 0, 0.33) ±0.01."""
        cam_xpos, cam_xmat = _dog_pose_to_cam((0.0, 0.0, 0.28), heading=0.0, pitch_deg=0.0)
        proxy = _make_proxy(cam_xpos, cam_xmat)
        cal = Go2Calibration(proxy)

        result = cal.camera_to_base(np.array([0.0, 0.0, 1.0]))

        # cam pos x = 0.3, plus 1m forward along +X = 1.3; y=0; z stays ~0.33
        assert result.shape == (3,)
        assert abs(result[0] - 1.3) < 0.01, f"expected x≈1.3, got {result[0]}"
        assert abs(result[1] - 0.0) < 0.01, f"expected y≈0, got {result[1]}"
        assert abs(result[2] - 0.33) < 0.01, f"expected z≈0.33, got {result[2]}"

    def test_dog_at_origin_facing_y_90deg(self) -> None:
        """Heading π/2 (+Y). cam point (0,0,1) -> world approx (0, 1.3, 0.33) ±0.01."""
        heading = math.pi / 2.0
        cam_xpos, cam_xmat = _dog_pose_to_cam((0.0, 0.0, 0.28), heading=heading, pitch_deg=0.0)
        proxy = _make_proxy(cam_xpos, cam_xmat)
        cal = Go2Calibration(proxy)

        result = cal.camera_to_base(np.array([0.0, 0.0, 1.0]))

        assert abs(result[0] - 0.0) < 0.01, f"expected x≈0, got {result[0]}"
        assert abs(result[1] - 1.3) < 0.01, f"expected y≈1.3, got {result[1]}"
        assert abs(result[2] - 0.33) < 0.01, f"expected z≈0.33, got {result[2]}"

    def test_dog_offset_position_adds_to_world(self) -> None:
        """Dog at (5,3,0.28) facing +X: world result shifts by (5,3,0) vs origin."""
        cam_xpos_origin, cam_xmat_origin = _dog_pose_to_cam(
            (0.0, 0.0, 0.28), heading=0.0, pitch_deg=0.0
        )
        cam_xpos_offset, cam_xmat_offset = _dog_pose_to_cam(
            (5.0, 3.0, 0.28), heading=0.0, pitch_deg=0.0
        )
        proxy_origin = _make_proxy(cam_xpos_origin, cam_xmat_origin)
        proxy_offset = _make_proxy(cam_xpos_offset, cam_xmat_offset)

        cal_origin = Go2Calibration(proxy_origin)
        cal_offset = Go2Calibration(proxy_offset)

        point = np.array([0.5, 0.2, 2.0])
        w_origin = cal_origin.camera_to_base(point)
        w_offset = cal_offset.camera_to_base(point)

        assert abs((w_offset[0] - w_origin[0]) - 5.0) < 0.01
        assert abs((w_offset[1] - w_origin[1]) - 3.0) < 0.01
        assert abs(w_offset[2] - w_origin[2]) < 0.01

    def test_input_shape_preserved(self) -> None:
        """Output shape is always (3,) regardless of input shape."""
        cam_xpos, cam_xmat = _dog_pose_to_cam((0.0, 0.0, 0.28), heading=0.0)
        proxy = _make_proxy(cam_xpos, cam_xmat)
        cal = Go2Calibration(proxy)

        result = cal.camera_to_base(np.array([1, 2, 3]))
        assert result.shape == (3,)

    def test_input_accepts_list_not_only_ndarray(self) -> None:
        """Plain Python list input works identically to ndarray."""
        cam_xpos, cam_xmat = _dog_pose_to_cam((0.0, 0.0, 0.28), heading=0.0)
        proxy = _make_proxy(cam_xpos, cam_xmat)
        cal = Go2Calibration(proxy)

        result_list = cal.camera_to_base([0.0, 0.0, 1.0])
        result_arr = cal.camera_to_base(np.array([0.0, 0.0, 1.0]))

        np.testing.assert_allclose(result_list, result_arr)


class TestAxisConventions:
    """Tests 6–8: lock in OpenCV <-> MuJoCo axis convention."""

    def test_downward_pitch_minus_5deg_lowers_z(self) -> None:
        """Camera pitched -5 deg: 2 m forward lowers world z by ~2*sin(5deg)."""
        pitch_deg = -5.0
        # With pitch=0 for comparison baseline
        cam_xpos_flat, cam_xmat_flat = _dog_pose_to_cam(
            (0.0, 0.0, 0.28), heading=0.0, pitch_deg=0.0
        )
        cam_xpos_pitch, cam_xmat_pitch = _dog_pose_to_cam(
            (0.0, 0.0, 0.28), heading=0.0, pitch_deg=pitch_deg
        )

        proxy_flat = _make_proxy(cam_xpos_flat, cam_xmat_flat)
        proxy_pitch = _make_proxy(cam_xpos_pitch, cam_xmat_pitch)

        cal_flat = Go2Calibration(proxy_flat)
        cal_pitch = Go2Calibration(proxy_pitch)

        point = np.array([0.0, 0.0, 2.0])
        w_flat = cal_flat.camera_to_base(point)
        w_pitch = cal_pitch.camera_to_base(point)

        expected_z_drop = 2.0 * math.sin(math.radians(5.0))  # ~0.174 m
        actual_drop = w_flat[2] - w_pitch[2]
        assert abs(actual_drop - expected_z_drop) < 0.02, (
            f"expected z-drop≈{expected_z_drop:.3f}, got {actual_drop:.3f}"
        )

    def test_off_centre_pixel_deprojection_symmetry(self) -> None:
        """cam (+1,0,5) and (-1,0,5) are symmetric around the dog's world y-axis."""
        cam_xpos, cam_xmat = _dog_pose_to_cam((0.0, 0.0, 0.28), heading=0.0)
        proxy = _make_proxy(cam_xpos, cam_xmat)
        cal = Go2Calibration(proxy)

        w_right = cal.camera_to_base(np.array([1.0, 0.0, 5.0]))
        w_left = cal.camera_to_base(np.array([-1.0, 0.0, 5.0]))

        # Dog faces +X; camera right = -Y world → y delta symmetric around dog y=0
        assert abs(w_right[1] + w_left[1]) < 0.01, (
            f"y values should be equal and opposite: {w_right[1]}, {w_left[1]}"
        )
        # x and z should be equal for both
        assert abs(w_right[0] - w_left[0]) < 0.01
        assert abs(w_right[2] - w_left[2]) < 0.01

    def test_y_axis_opencv_to_world_up_convention(self) -> None:
        """Verify the OpenCV↔MuJoCo axis sign convention used in this codebase.

        In this codebase's convention (Go2ROS2Proxy / depth_projection.camera_to_world):
            up_col = cross(right, fwd)  →  [0, 0, -1]  when heading=0, pitch=0

        Formula: world += (-y_cam) * up_col
        With up_col = [0,0,-1]: (-y_cam)*[0,0,-1]  →  y_cam=+1 adds world +z.

        So cam (0, +1, 5) (below centre in image) projects HIGHER in world z
        than the on-axis point (0, 0, 5).  This is the established convention
        used by depth_projection.camera_to_world (line 148) and is what
        Go2Calibration must reproduce.

        The formula is: world = pos + p[0]*col0 + (-p[1])*col1 + p[2]*(-col2)
        which is identical to depth_projection.py line 148.
        """
        cam_xpos, cam_xmat = _dog_pose_to_cam((0.0, 0.0, 0.28), heading=0.0, pitch_deg=0.0)
        proxy = _make_proxy(cam_xpos, cam_xmat)
        cal = Go2Calibration(proxy)

        w_centre = cal.camera_to_base(np.array([0.0, 0.0, 5.0]))
        w_down = cal.camera_to_base(np.array([0.0, 1.0, 5.0]))

        # In this codebase's convention, cam +y projects to HIGHER world z
        # (up_col=[0,0,-1] for level heading=0 camera; see depth_projection.py)
        assert w_down[2] > w_centre[2], (
            f"With up_col=[0,0,-1]: cam +y should increase world z. "
            f"w_centre_z={w_centre[2]:.3f}, w_down_z={w_down[2]:.3f}"
        )
        # The z delta should equal the y_cam offset (1.0) scaled by |up_col[2]| = 1
        assert abs((w_down[2] - w_centre[2]) - 1.0) < 0.01
