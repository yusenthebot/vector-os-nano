# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Go2Calibration — pose-driven camera-to-world transform for the Unitree Go2.

Axis conventions
----------------
MuJoCo xmat columns:
    col 0 = right   (+X in camera frame)
    col 1 = up      (+Y in world, but -Y in OpenCV camera frame)
    col 2 = -forward  (OpenCV +Z points forward, so col 2 = negative forward)

OpenCV camera frame:
    x = right   (+x_cam → +col0 direction in world)
    y = down    (+y_cam → -col1 direction in world, because col1 = world UP)
    z = forward (+z_cam → -col2 direction in world, because col2 = -forward)

Transform formula::

    world = cam_pos
            + p[0] * xmat[:, 0]   # right component
            + (-p[1]) * xmat[:, 1]  # down -> negate up column
            + p[2] * (-xmat[:, 2])  # forward -> negate -forward column

Usage::

    cal = Go2Calibration(base_proxy)
    world_xyz = cal.camera_to_base(np.array([x_cam, y_cam, depth]))
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Go2Calibration:
    """Transform OpenCV-frame camera points to world coordinates.

    Reads the live camera pose from ``base_proxy.get_camera_pose()`` on
    every call, so the transform stays valid as the robot moves.

    Args:
        base_proxy: Any object implementing ``get_camera_pose() ->
            (cam_xpos: array-like (3,), cam_xmat: array-like (9,))``.
            Compatible with both ``MuJoCoGo2`` and ``Go2ROS2Proxy``.
    """

    def __init__(self, base_proxy: Any) -> None:
        self._base = base_proxy

    def camera_to_base(self, point_camera: Any) -> np.ndarray:
        """Project an OpenCV-frame camera point into world coordinates.

        Args:
            point_camera: A 3-element array-like ``(x, y, z)`` in OpenCV
                camera frame where x=right, y=down, z=forward (depth).
                Accepts ``np.ndarray`` or plain Python ``list``/``tuple``.

        Returns:
            World-frame XYZ as ``np.ndarray`` of shape ``(3,)``.

        Notes:
            The transform reads ``base_proxy.get_camera_pose()`` fresh on
            every call.  Camera pose uses MuJoCo xmat convention where
            columns are [right, up, -forward].
        """
        p = np.asarray(point_camera, dtype=np.float64).reshape(3)
        cam_xpos, cam_xmat = self._base.get_camera_pose()

        pos = np.asarray(cam_xpos, dtype=np.float64).reshape(3)
        xmat = np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3)

        # MuJoCo xmat cols = [right, up, -forward]
        # OpenCV point components: (right, down, forward)
        #   right  (+p[0]) maps to  col0 direction
        #   down   (+p[1]) maps to -col1 direction (col1 = world up)
        #   forward(+p[2]) maps to -col2 direction (col2 = -forward)
        world = (
            pos
            + p[0] * xmat[:, 0]
            + (-p[1]) * xmat[:, 1]
            + p[2] * (-xmat[:, 2])
        )
        return world
