# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Go2Perception — PerceptionProtocol implementation for Go2 robot.

Composes a duck-typed camera (get_camera_frame / get_depth_frame) and a
duck-typed VLM (detect(image, query) -> list[Detection]).

Single-shot tracking: bbox-center pixel + IQR-cleaned median depth
in the bounding box patch → Pose3D in CAMERA frame (Z forward, X right,
Y down — OpenCV convention).

No hard imports of MuJoCo, ROS2, or any specific VLM implementation.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from vector_os_nano.core.types import (
    CameraIntrinsics,
    Detection,
    Pose3D,
    TrackedObject,
)
from vector_os_nano.perception.depth_projection import (
    mujoco_intrinsics,
    pixel_to_camera,
)

class Go2Perception:
    """Go2 perception backend.

    Implements PerceptionProtocol via structural subtyping.

    Not thread-safe — callers should invoke ``detect()`` / ``track()``
    from a single thread (skill executor is sequential by design).

    Args:
        camera: Any object exposing ``get_camera_frame() -> np.ndarray``
                and ``get_depth_frame() -> np.ndarray``.
        vlm:    Any object exposing ``detect(image, query) -> list[Detection]``.
        intrinsics: Camera intrinsics.  Defaults to
                    ``mujoco_intrinsics(320, 240, 42.0)`` (Go2 MJCF fovy).
        depth_trunc: Maximum valid depth in metres (exclusive). Defaults to 10.0.
    """

    def __init__(
        self,
        camera: Any,
        vlm: Any,
        intrinsics: CameraIntrinsics | None = None,
        depth_trunc: float = 10.0,
    ) -> None:
        self._camera = camera
        self._vlm = vlm
        self._intrinsics: CameraIntrinsics = (
            intrinsics if intrinsics is not None else mujoco_intrinsics(320, 240, 42.0)
        )
        self._depth_trunc = depth_trunc
        # M2: no mutable frame cache — Go2Perception is not thread-safe by design.
        # Callers that want per-call results should hold on to the returned lists.

    # ------------------------------------------------------------------
    # PerceptionProtocol — frame accessors
    # ------------------------------------------------------------------

    def get_color_frame(self) -> np.ndarray:
        """Return latest RGB image as (H, W, 3) uint8 from the camera."""
        return self._camera.get_camera_frame()

    def get_depth_frame(self) -> np.ndarray:
        """Return latest depth image as (H, W) float32 (metres) from the camera."""
        return self._camera.get_depth_frame()

    def get_intrinsics(self) -> CameraIntrinsics:
        """Return pinhole camera intrinsic parameters."""
        return self._intrinsics

    # ------------------------------------------------------------------
    # PerceptionProtocol — VLM detection
    # ------------------------------------------------------------------

    def detect(self, query: str) -> list[Detection]:
        """Run VLM detection for objects matching *query*.

        Calls ``vlm.detect(rgb_frame, query)`` and returns the result.
        Caches the last detections internally.
        """
        rgb = self.get_color_frame()
        return self._vlm.detect(rgb, query)

    # ------------------------------------------------------------------
    # PerceptionProtocol — depth-based tracking
    # ------------------------------------------------------------------

    def track(self, detections: list[Detection]) -> list[TrackedObject]:
        """Project each detection's bbox centre through the depth frame.

        For each :class:`Detection` a :class:`TrackedObject` is returned
        with a :class:`Pose3D` in camera frame (or ``None`` if depth is
        unavailable / invalid in the bbox region).

        Args:
            detections: List of 2D detections (pixel-space bboxes).

        Returns:
            List of :class:`TrackedObject`, same length as *detections*.
            ``track_id`` is assigned 1-based in iteration order.
        """
        depth = self.get_depth_frame()
        out: list[TrackedObject] = []
        for i, det in enumerate(detections):
            pose = self._project_bbox_to_camera_frame(
                det.bbox, depth, self._intrinsics, self._depth_trunc
            )
            out.append(
                TrackedObject(
                    track_id=i + 1,
                    label=det.label,
                    bbox_2d=det.bbox,
                    pose=pose,
                    bbox_3d=None,
                    confidence=det.confidence,
                    mask=None,
                )
            )
        return out

    # ------------------------------------------------------------------
    # PerceptionProtocol — point cloud (not implemented in v2.3)
    # ------------------------------------------------------------------

    def get_point_cloud(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Not implemented in v2.3.  Raises :class:`NotImplementedError`."""
        raise NotImplementedError(
            "Go2Perception does not expose point clouds in v2.3"
        )

    # ------------------------------------------------------------------
    # PerceptionProtocol — caption / visual_query (stubs for protocol)
    # ------------------------------------------------------------------

    def caption(self, length: str = "normal") -> str:
        """Not implemented in v2.3."""
        raise NotImplementedError("Go2Perception.caption not implemented in v2.3")

    def visual_query(self, question: str) -> str:
        """Not implemented in v2.3."""
        raise NotImplementedError(
            "Go2Perception.visual_query not implemented in v2.3"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _project_bbox_to_camera_frame(
        bbox: tuple[float, float, float, float],
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        depth_trunc: float,
    ) -> Pose3D | None:
        """Project bbox centre through depth to a camera-frame Pose3D.

        Steps:
        1. Clamp integer bbox slice to frame boundaries.
        2. Extract centre pixel (float, unclamped) for precision.
        3. Collect valid depth samples from the clamped patch
           (filter NaN, <= 0, >= depth_trunc).
        4. IQR outlier reject (when patch has >= 10 valid samples).
        5. Median depth → ``pixel_to_camera`` → ``Pose3D``.

        Returns ``None`` when no valid depth samples remain.
        """
        x1, y1, x2, y2 = bbox
        h, w = depth.shape[:2]

        # Integer-clamped slice boundaries
        x1i = max(0, min(int(round(x1)), w - 1))
        y1i = max(0, min(int(round(y1)), h - 1))
        x2i = max(x1i + 1, min(int(round(x2)), w))
        y2i = max(y1i + 1, min(int(round(y2)), h))
        if x2i <= x1i or y2i <= y1i:
            return None

        # Float centre for projection precision (not clamped)
        u = (x1 + x2) * 0.5
        v = (y1 + y2) * 0.5

        # Collect valid depth pixels in the bbox patch
        patch = depth[y1i:y2i, x1i:x2i]
        valid = patch[np.isfinite(patch) & (patch > 0) & (patch < depth_trunc)]
        if valid.size == 0:
            return None

        # IQR outlier reject on the z-samples (only when enough pixels)
        if valid.size >= 10:
            q1, q3 = np.percentile(valid, [25, 75])
            iqr = q3 - q1
            if iqr > 1e-6:
                low = q1 - 1.5 * iqr
                high = q3 + 1.5 * iqr
                valid = valid[(valid >= low) & (valid <= high)]
                if valid.size == 0:
                    return None

        d = float(np.median(valid))
        x_cam, y_cam, z_cam = pixel_to_camera(u, v, d, intrinsics)
        return Pose3D(x=x_cam, y=y_cam, z=z_cam)
