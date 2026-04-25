# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 13: Depth projection and D435 camera tests.

Tests the depth-to-world projection pipeline:
  pixel (u,v) + depth → camera frame → world frame

Also tests D435 intrinsics computation and center depth sampling.
All pure Python — no MuJoCo or ROS2 required.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from vector_os_nano.perception.depth_projection import (
    CameraIntrinsics,
    camera_to_world,
    center_depth,
    d435_intrinsics,
    depth_to_world,
    pixel_to_camera,
    project_center_to_world,
)


# ---------------------------------------------------------------------------
# D435 Intrinsics
# ---------------------------------------------------------------------------


class TestD435Intrinsics:
    """Test RealSense D435 intrinsic parameter computation."""

    def test_default_resolution(self):
        intr = d435_intrinsics()
        assert intr.width == 320
        assert intr.height == 240

    def test_principal_point_at_center(self):
        intr = d435_intrinsics(640, 480)
        assert intr.cx == 320.0
        assert intr.cy == 240.0

    def test_focal_length_from_fov(self):
        """69 deg HFOV at 320px → fx ≈ 230."""
        intr = d435_intrinsics(320, 240)
        expected_fx = 160.0 / math.tan(math.radians(69.0 / 2))
        assert abs(intr.fx - expected_fx) < 0.01

    def test_square_pixels(self):
        intr = d435_intrinsics()
        assert intr.fx == intr.fy

    def test_higher_resolution_scales(self):
        lo = d435_intrinsics(320, 240)
        hi = d435_intrinsics(640, 480)
        assert abs(hi.fx / lo.fx - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Pixel → Camera Frame
# ---------------------------------------------------------------------------


class TestPixelToCamera:
    """Test pixel to camera-frame 3D projection."""

    def test_center_pixel_straight_ahead(self):
        """Center pixel at 2m depth → (0, 0, 2) in camera frame."""
        intr = d435_intrinsics(320, 240)
        x, y, z = pixel_to_camera(intr.cx, intr.cy, 2.0, intr)
        assert abs(x) < 0.001
        assert abs(y) < 0.001
        assert abs(z - 2.0) < 0.001

    def test_right_pixel_positive_x(self):
        """Pixel to the right of center → positive x_cam."""
        intr = d435_intrinsics(320, 240)
        x, y, z = pixel_to_camera(intr.cx + 50, intr.cy, 2.0, intr)
        assert x > 0
        assert abs(y) < 0.001

    def test_bottom_pixel_positive_y(self):
        """Pixel below center → positive y_cam (OpenCV: Y down)."""
        intr = d435_intrinsics(320, 240)
        x, y, z = pixel_to_camera(intr.cx, intr.cy + 50, 2.0, intr)
        assert abs(x) < 0.001
        assert y > 0

    def test_depth_preserved(self):
        intr = d435_intrinsics()
        _, _, z = pixel_to_camera(100, 100, 3.5, intr)
        assert abs(z - 3.5) < 0.001


# ---------------------------------------------------------------------------
# Camera Frame → World Frame
# ---------------------------------------------------------------------------


class TestCameraToWorld:
    """Test camera-frame to world-frame transformation."""

    def test_straight_ahead_heading_zero(self):
        """Camera looking along +X (heading=0), object 2m ahead."""
        wx, wy, wz = camera_to_world(
            0.0, 0.0, 2.0,  # cam: straight ahead 2m
            robot_x=5.0, robot_y=3.0, robot_z=0.28,
            robot_heading=0.0,
        )
        # mount_forward=0.3, so total forward = 0.3 + 2.0 = 2.3
        assert abs(wx - (5.0 + 2.3)) < 0.01
        assert abs(wy - 3.0) < 0.01

    def test_heading_90_degrees(self):
        """Heading = pi/2 (facing +Y), object 2m ahead."""
        wx, wy, wz = camera_to_world(
            0.0, 0.0, 2.0,
            robot_x=5.0, robot_y=3.0, robot_z=0.28,
            robot_heading=math.pi / 2,
        )
        assert abs(wx - 5.0) < 0.05  # X stays ~same
        assert wy > 3.0 + 2.0        # Y increases by ~2.3m

    def test_object_to_right(self):
        """Object 1m to the right in camera frame, heading=0."""
        wx, wy, wz = camera_to_world(
            1.0, 0.0, 2.0,  # cam: 1m right, 2m forward
            robot_x=0.0, robot_y=0.0, robot_z=0.28,
            robot_heading=0.0,
        )
        # Right in camera = -Y in world (heading=0, right is -Y)
        # Actually: cam_x=right → body_right=1.0 → world_y = sin(0)*forward + cos(0)*right = 1.0
        # Wait, let me reconsider: heading=0 means facing +X
        # cam_x = right → perpendicular to heading → -Y direction? No...
        # body_right=1.0, heading=0: world_y = sin(0)*(forward) + cos(0)*(right) = 0 + 1 = 1
        assert wy > 0  # right of heading=0 is +Y in our convention

    def test_height_preserved(self):
        """Object at camera height should be near robot_z + mount_up."""
        _, _, wz = camera_to_world(
            0.0, 0.0, 2.0,
            robot_x=0, robot_y=0, robot_z=0.28,
            robot_heading=0,
        )
        # cam_y=0 → body_up=0 → wz = 0.28 + 0.05 = 0.33 (mount_up=0.05 per MJCF)
        assert abs(wz - 0.33) < 0.01


# ---------------------------------------------------------------------------
# Center Depth
# ---------------------------------------------------------------------------


class TestCenterDepth:
    """Test center region depth sampling."""

    def test_uniform_depth(self):
        """Uniform 2m depth frame → center_depth = 2.0."""
        frame = np.full((240, 320), 2.0, dtype=np.float32)
        assert abs(center_depth(frame) - 2.0) < 0.01

    def test_invalid_depth_ignored(self):
        """Zeros and out-of-range values are excluded."""
        frame = np.zeros((240, 320), dtype=np.float32)
        frame[110:130, 150:170] = 3.0  # valid center patch
        d = center_depth(frame)
        assert abs(d - 3.0) < 0.1

    def test_all_invalid_returns_zero(self):
        frame = np.zeros((240, 320), dtype=np.float32)
        assert center_depth(frame) == 0.0

    def test_out_of_range_excluded(self):
        frame = np.full((240, 320), 15.0, dtype=np.float32)  # >10m
        assert center_depth(frame) == 0.0

    def test_median_is_robust(self):
        """Median ignores outliers better than mean."""
        frame = np.full((240, 320), 2.0, dtype=np.float32)
        # Add some noise
        frame[115:125, 155:165] = 8.0  # outlier in center
        d = center_depth(frame)
        assert abs(d - 2.0) < 0.5  # median still near 2.0


# ---------------------------------------------------------------------------
# Full Pipeline: depth_to_world
# ---------------------------------------------------------------------------


class TestDepthToWorld:
    """Test the full pixel → world projection."""

    def test_center_pixel_projects_ahead(self):
        """Center pixel at 2m depth projects to a point ahead of robot."""
        intr = d435_intrinsics(320, 240)
        frame = np.full((240, 320), 2.0, dtype=np.float32)

        result = depth_to_world(
            frame, intr.cx, intr.cy, intr,
            robot_x=5.0, robot_y=3.0, robot_z=0.28,
            robot_heading=0.0,
        )
        assert result is not None
        wx, wy, wz = result
        assert wx > 5.0  # ahead of robot
        assert abs(wy - 3.0) < 0.1

    def test_invalid_depth_returns_none(self):
        intr = d435_intrinsics()
        frame = np.zeros((240, 320), dtype=np.float32)
        result = depth_to_world(frame, 160, 120, intr, 0, 0, 0.28, 0)
        assert result is None

    def test_out_of_bounds_pixel_returns_none(self):
        intr = d435_intrinsics(320, 240)
        frame = np.full((240, 320), 2.0, dtype=np.float32)
        assert depth_to_world(frame, -1, 120, intr, 0, 0, 0.28, 0) is None
        assert depth_to_world(frame, 160, 300, intr, 0, 0, 0.28, 0) is None


class TestProjectCenterToWorld:
    """Test the convenience center-projection function."""

    def test_projects_to_point_ahead(self):
        intr = d435_intrinsics(320, 240)
        frame = np.full((240, 320), 3.0, dtype=np.float32)
        result = project_center_to_world(
            frame, intr, robot_x=0, robot_y=0, robot_z=0.28, robot_heading=0,
        )
        assert result is not None
        wx, wy, wz = result
        # 3m depth + 0.3m mount offset = ~3.3m ahead
        assert abs(wx - 3.3) < 0.1

    def test_no_valid_depth_returns_none(self):
        intr = d435_intrinsics()
        frame = np.zeros((240, 320), dtype=np.float32)
        result = project_center_to_world(frame, intr, 0, 0, 0.28, 0)
        assert result is None

    def test_heading_rotates_projection(self):
        """Heading=pi/2 → projects along +Y instead of +X."""
        intr = d435_intrinsics(320, 240)
        frame = np.full((240, 320), 2.0, dtype=np.float32)
        result = project_center_to_world(
            frame, intr, robot_x=0, robot_y=0, robot_z=0.28,
            robot_heading=math.pi / 2,
        )
        assert result is not None
        wx, wy, wz = result
        assert abs(wx) < 0.1  # X stays near 0
        assert wy > 2.0       # Y projects forward
