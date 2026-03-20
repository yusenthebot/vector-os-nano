"""PerceptionPipeline — orchestrates camera, VLM, tracker, and 3D.

Ported from:
  vector_ws/src/track_anything/track_anything/track_3d.py (rgbd_callback logic)

The ROS2 pub/sub loop is replaced by direct method calls.

Pipeline steps (mirrors track_3d.py rgbd_callback):
  1. Acquire aligned RGBD frames from camera
  2. On detect() call: run VLM -> get bboxes -> init_track
  3. On track() call: process_image -> get masks -> project to 3D

Performance constants from track_3d.py:
  depth_scale  = 1000.0  (mm -> metres)
  depth_trunc  = 10.0    (metres)
  bbox_max_points = 2000 (sampling for fast bbox3d)
"""
from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Optional

import numpy as np

from vector_os.core.types import (
    BBox3D,
    CameraIntrinsics,
    Detection,
    Pose3D,
    TrackedObject,
)
from vector_os.perception.base import PerceptionProtocol
from vector_os.perception.pointcloud import (
    pointcloud_to_bbox3d_fast,
    rgbd_to_pointcloud_fast,
)

logger = logging.getLogger(__name__)

# Performance constants from track_3d.py
_DEPTH_SCALE = 1000.0
_DEPTH_TRUNC = 10.0
_BBOX_MAX_POINTS = 2000
_TRACKING_TIMEOUT_S = 20.0
_MASK_OPEN_KERNEL_SIZE = 5
_MASK_ERODE_KERNEL_SIZE = 7


class PerceptionPipeline:
    """Full perception pipeline: camera -> VLM -> tracker -> 3D.

    Satisfies PerceptionProtocol structurally.

    Usage (with real hardware):
        cam = RealSenseCamera()
        cam.connect()
        vlm = VLMDetector()
        tracker = EdgeTAMTracker()
        pipeline = PerceptionPipeline(camera=cam, vlm=vlm, tracker=tracker)

        detections = pipeline.detect("red cup")
        tracked = pipeline.track(detections)

    Usage (without hardware — synthetic frames):
        pipeline = PerceptionPipeline()
        pipeline.set_synthetic_frames(color, depth, intrinsics)
        ...
    """

    def __init__(
        self,
        camera: object | None = None,
        vlm: object | None = None,
        tracker: object | None = None,
        depth_scale: float = _DEPTH_SCALE,
        depth_trunc: float = _DEPTH_TRUNC,
        bbox_max_points: int = _BBOX_MAX_POINTS,
    ) -> None:
        self._camera = camera
        self._vlm = vlm
        self._tracker = tracker

        self._depth_scale = depth_scale
        self._depth_trunc = depth_trunc
        self._bbox_max_points = bbox_max_points

        self._lock = Lock()
        self._tracked_objects: list[TrackedObject] = []
        self._tracking_loss_time: float | None = None

        # Synthetic frame support (for testing without hardware)
        self._synthetic_color: np.ndarray | None = None
        self._synthetic_depth: np.ndarray | None = None
        self._synthetic_intrinsics: CameraIntrinsics | None = None

    # ------------------------------------------------------------------
    # Synthetic frame injection (for unit testing)
    # ------------------------------------------------------------------

    def set_synthetic_frames(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> None:
        """Inject synthetic frames for testing without a real camera."""
        self._synthetic_color = color
        self._synthetic_depth = depth
        self._synthetic_intrinsics = intrinsics

    # ------------------------------------------------------------------
    # PerceptionProtocol implementation
    # ------------------------------------------------------------------

    def get_color_frame(self) -> np.ndarray:
        """Return latest RGB color frame."""
        if self._synthetic_color is not None:
            return self._synthetic_color
        if self._camera is None:
            raise RuntimeError("No camera configured and no synthetic frames set")
        return self._camera.get_color_frame()  # type: ignore[union-attr]

    def get_depth_frame(self) -> np.ndarray:
        """Return latest depth frame (uint16 mm)."""
        if self._synthetic_depth is not None:
            return self._synthetic_depth
        if self._camera is None:
            raise RuntimeError("No camera configured and no synthetic frames set")
        return self._camera.get_depth_frame()  # type: ignore[union-attr]

    def get_intrinsics(self) -> CameraIntrinsics:
        """Return camera intrinsics."""
        if self._synthetic_intrinsics is not None:
            return self._synthetic_intrinsics
        if self._camera is None:
            raise RuntimeError("No camera configured and no synthetic intrinsics set")
        return self._camera.get_intrinsics()  # type: ignore[union-attr]

    def get_point_cloud(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Project depth to 3D point cloud using current frame.

        Args:
            mask: Optional (H, W) binary mask.

        Returns:
            (N, 3) float64 array of 3D points in camera frame.
        """
        depth = self.get_depth_frame()
        color = self.get_color_frame()
        intrinsics = self.get_intrinsics()
        points, _ = rgbd_to_pointcloud_fast(
            depth, color, intrinsics,
            depth_scale=self._depth_scale,
            depth_trunc=self._depth_trunc,
            mask=mask,
        )
        return points

    def detect(self, query: str) -> list[Detection]:
        """Run VLM detection on the current frame.

        Args:
            query: Natural language object description.

        Returns:
            List of Detection with pixel-space bboxes.

        Raises:
            RuntimeError: If no VLM is configured.
        """
        if self._vlm is None:
            raise RuntimeError("No VLM configured. Pass vlm=VLMDetector() to PerceptionPipeline")
        color = self.get_color_frame()
        detections: list[Detection] = self._vlm.detect(color, query)  # type: ignore[union-attr]
        logger.info("VLM detect('%s') -> %d detections", query, len(detections))
        return detections

    def track(self, detections: list[Detection]) -> list[TrackedObject]:
        """Initialize tracker from detections and return TrackedObject list.

        Ported from track_3d.py track_init_callback + rgbd_callback flow:
          1. Convert Detection bboxes to tracker format
          2. init_track() with validated bboxes
          3. For each tracked object: project mask to 3D bbox

        Args:
            detections: List of Detection from detect().

        Returns:
            List of TrackedObject with 2D bbox, optional 3D bbox, mask.

        Raises:
            RuntimeError: If no tracker is configured.
        """
        if self._tracker is None:
            raise RuntimeError("No tracker configured. Pass tracker=EdgeTAMTracker() to PerceptionPipeline")

        color = self.get_color_frame()
        depth = self.get_depth_frame()
        intrinsics = self.get_intrinsics()
        h, w = color.shape[:2]

        # Convert Detection bboxes to tracker format (x1, y1, x2, y2) ints
        bboxes = []
        labels = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1 = max(0, min(int(round(x1)), w - 1))
            y1 = max(0, min(int(round(y1)), h - 1))
            x2 = max(0, min(int(round(x2)), w - 1))
            y2 = max(0, min(int(round(y2)), h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            bboxes.append((x1, y1, x2, y2))
            labels.append(det.label)

        if not bboxes:
            self._tracked_objects = []
            return []

        raw_tracks = self._tracker.init_track(color, bboxes=bboxes)  # type: ignore[union-attr]
        tracked_objects = self._build_tracked_objects(
            raw_tracks, labels, depth, intrinsics
        )
        with self._lock:
            self._tracked_objects = tracked_objects
        return tracked_objects

    def get_tracked_objects(self) -> list[TrackedObject]:
        """Return the last set of tracked objects (from the most recent track() call)."""
        with self._lock:
            return list(self._tracked_objects)

    def update(self) -> list[TrackedObject]:
        """Process the current frame and update tracking.

        Call this each frame after an initial track() to propagate masks.
        Ported from track_3d.py rgbd_callback (process_image path).

        Returns:
            Updated TrackedObject list. Empty if tracking is lost.
        """
        if self._tracker is None:
            return []

        color = self.get_color_frame()
        depth = self.get_depth_frame()
        intrinsics = self.get_intrinsics()

        raw_tracks = self._tracker.process_image(color)  # type: ignore[union-attr]

        if not raw_tracks:
            if self._tracking_loss_time is None:
                self._tracking_loss_time = time.monotonic()
                logger.warning("Tracking lost, starting timeout")
            elapsed = time.monotonic() - self._tracking_loss_time
            if elapsed > _TRACKING_TIMEOUT_S:
                logger.warning("Tracking lost for %.0fs, stopping", elapsed)
                self._tracker.stop()  # type: ignore[union-attr]
                self._tracking_loss_time = None
                with self._lock:
                    self._tracked_objects = []
            return []

        self._tracking_loss_time = None
        # Preserve labels from last known tracked objects
        label_map = {t.track_id: t.label for t in self._tracked_objects}
        labels = [label_map.get(r.get("track_id", 0), "object") for r in raw_tracks]

        tracked_objects = self._build_tracked_objects(
            raw_tracks, labels, depth, intrinsics
        )
        with self._lock:
            self._tracked_objects = tracked_objects
        return tracked_objects

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tracked_objects(
        self,
        raw_tracks: list[dict],
        labels: list[str],
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> list[TrackedObject]:
        """Convert raw tracker output dicts to TrackedObject list with 3D data."""
        result: list[TrackedObject] = []
        for i, raw in enumerate(raw_tracks):
            track_id = int(raw.get("track_id", i + 1))
            mask = raw.get("mask")
            bbox_2d_raw = raw.get("bbox", [0, 0, 0, 0])
            score = float(raw.get("score", 1.0))
            label = labels[i] if i < len(labels) else "object"

            bbox_2d: tuple[float, float, float, float] = (
                float(bbox_2d_raw[0]),
                float(bbox_2d_raw[1]),
                float(bbox_2d_raw[2]),
                float(bbox_2d_raw[3]),
            )

            # Refine mask morphologically (same as track_3d.py _refine_mask)
            refined_mask: np.ndarray | None = None
            if mask is not None:
                refined_mask = self._refine_mask(mask)

            # Project mask to 3D
            bbox_3d: BBox3D | None = None
            pose: Pose3D | None = None
            if refined_mask is not None and np.any(refined_mask):
                points, _ = rgbd_to_pointcloud_fast(
                    depth,
                    np.zeros((*depth.shape, 3), dtype=np.uint8),  # color not needed here
                    intrinsics,
                    depth_scale=self._depth_scale,
                    depth_trunc=self._depth_trunc,
                    mask=refined_mask,
                )
                if len(points) >= 4:
                    sampled = self._sample_points(points, self._bbox_max_points)
                    bbox_3d = pointcloud_to_bbox3d_fast(sampled)
                    if bbox_3d is not None:
                        pose = bbox_3d.center

            result.append(TrackedObject(
                track_id=track_id,
                label=label,
                bbox_2d=bbox_2d,
                pose=pose,
                bbox_3d=bbox_3d,
                confidence=score,
                mask=refined_mask,
            ))

        return result

    @staticmethod
    def _refine_mask(mask: np.ndarray) -> np.ndarray:
        """Morphological mask refinement (ported from track_3d.py _refine_mask).

        Opening removes noise; erosion shrinks edges to filter boundary noise.
        Falls back gracefully if cv2 not installed.
        """
        try:
            import cv2
        except ImportError:
            return (mask > 0).astype(np.uint8)

        mask_u8 = mask.astype(np.uint8) * 255
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (_MASK_OPEN_KERNEL_SIZE, _MASK_OPEN_KERNEL_SIZE),
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (_MASK_ERODE_KERNEL_SIZE, _MASK_ERODE_KERNEL_SIZE),
        )
        mask_u8 = cv2.erode(mask_u8, erode_kernel, iterations=1)
        return (mask_u8 > 0).astype(np.uint8)

    @staticmethod
    def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
        """Uniformly subsample points for performance (same as track_3d.py)."""
        if max_points <= 0 or len(points) <= max_points:
            return points
        idx = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int32)
        return points[idx]
