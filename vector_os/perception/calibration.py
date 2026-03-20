"""Camera-to-arm coordinate transformation.

Ported from:
  vector_ws/src/so101_perception/scripts/calibrate_hand_eye.py

Provides:
  - Calibration.solve_affine() / solve_affine_and_store() — least-squares affine
  - Calibration.solve_rbf() — RBF nonlinear interpolation (scipy, falls back to affine)
  - Calibration.camera_to_base() — apply best available transform (RBF > affine > identity)
  - Calibration.get_error_stats() — mean/max/per-point errors
  - Calibration.save() / Calibration.load() — persist matrix + optional RBF sidecar

No ROS2 or hardware dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy scipy import — may not be installed
_rbf_class: Any | None = None
_scipy_checked = False


def _get_rbf_class() -> Any | None:
    """Return scipy RBFInterpolator class or None if unavailable."""
    global _rbf_class, _scipy_checked
    if _scipy_checked:
        return _rbf_class
    _scipy_checked = True
    try:
        from scipy.interpolate import RBFInterpolator
        _rbf_class = RBFInterpolator
        logger.debug("scipy RBFInterpolator available")
    except ImportError:
        _rbf_class = None
        logger.debug("scipy not available — RBF disabled")
    return _rbf_class


class Calibration:
    """Camera-to-arm base-frame coordinate transformation.

    The internal representation is a 4x4 homogeneous transform matrix T such
    that:
        p_base = T[:3, :3] @ p_cam + T[:3, 3]

    When RBF is fitted, camera_to_base() uses the RBF interpolant instead.

    Default (no calibration file): identity — camera frame == base frame.
    """

    def __init__(self, calibration_file: str | None = None) -> None:
        self._matrix: np.ndarray = np.eye(4, dtype=np.float64)
        self._rbf_fitted: bool = False
        self._rbf_x: np.ndarray | None = None   # (N, 3) training camera points
        self._rbf_y: np.ndarray | None = None   # (N, 3) training base points
        self._rbf_interp: Any | None = None      # scipy RBFInterpolator instance
        self._cal_points_cam: np.ndarray | None = None
        self._cal_points_base: np.ndarray | None = None

        if calibration_file is not None:
            loaded = Calibration.load(calibration_file)
            self._matrix = loaded._matrix
            self._rbf_fitted = loaded._rbf_fitted
            self._rbf_x = loaded._rbf_x
            self._rbf_y = loaded._rbf_y
            self._rbf_interp = loaded._rbf_interp
            self._cal_points_cam = loaded._cal_points_cam
            self._cal_points_base = loaded._cal_points_base
            logger.info("Calibration loaded from %s", calibration_file)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def camera_to_base(self, point_camera: np.ndarray) -> np.ndarray:
        """Transform a point from camera frame to arm base frame.

        Uses best available method: RBF > affine > identity.

        Args:
            point_camera: 3D point (3,) in camera coordinates.

        Returns:
            3D point (3,) in arm base frame.
        """
        p = np.asarray(point_camera, dtype=np.float64).reshape(3)

        if self._rbf_fitted and self._rbf_interp is not None:
            result = self._rbf_interp(p.reshape(1, 3))
            return result.reshape(3)

        R = self._matrix[:3, :3]
        t = self._matrix[:3, 3]
        return R @ p + t

    # ------------------------------------------------------------------
    # Solve — affine
    # ------------------------------------------------------------------

    @staticmethod
    def solve_affine(
        points_camera: np.ndarray,
        points_base: np.ndarray,
    ) -> np.ndarray:
        """Solve affine transform T from paired 3D point correspondences.

        Uses least-squares (numpy.linalg.lstsq) to solve:
            points_base ≈ points_camera @ A.T + b

        Returns a 4x4 homogeneous matrix [R|t; 0 0 0 1].

        Args:
            points_camera: (N, 3) array of points in camera frame.
            points_base: (N, 3) array of corresponding points in base frame.

        Returns:
            4x4 float64 homogeneous transform matrix.
        """
        pts_cam = np.asarray(points_camera, dtype=np.float64)
        pts_base = np.asarray(points_base, dtype=np.float64)

        _warn_if_no_z_variation(pts_cam, "solve_affine")

        return _solve_affine_matrix(pts_cam, pts_base)

    def solve_affine_and_store(
        self,
        points_camera: np.ndarray,
        points_base: np.ndarray,
    ) -> None:
        """Solve affine and store result + calibration data in this instance.

        Equivalent to solve_affine() but mutates self.

        Args:
            points_camera: (N, 3) camera-frame points.
            points_base: (N, 3) base-frame points.
        """
        pts_cam = np.asarray(points_camera, dtype=np.float64)
        pts_base = np.asarray(points_base, dtype=np.float64)
        _warn_if_no_z_variation(pts_cam, "solve_affine_and_store")
        self._matrix = _solve_affine_matrix(pts_cam, pts_base)
        self._cal_points_cam = pts_cam
        self._cal_points_base = pts_base
        self._rbf_fitted = False
        self._rbf_interp = None
        logger.info(
            "Affine calibration fitted with %d points", len(pts_cam)
        )

    # ------------------------------------------------------------------
    # Solve — RBF
    # ------------------------------------------------------------------

    def solve_rbf(
        self,
        points_camera: np.ndarray,
        points_base: np.ndarray,
    ) -> None:
        """Fit RBF interpolation for nonlinear correction.

        Uses scipy.interpolate.RBFInterpolator when available.
        Falls back to affine if scipy is not installed.

        Args:
            points_camera: (N, 3) camera-frame points.
            points_base: (N, 3) base-frame points.
        """
        pts_cam = np.asarray(points_camera, dtype=np.float64)
        pts_base = np.asarray(points_base, dtype=np.float64)

        _warn_if_no_z_variation(pts_cam, "solve_rbf")

        # Always compute affine as baseline / fallback (call _solve_affine_matrix
        # directly to avoid double Z-variation warning)
        self._matrix = _solve_affine_matrix(pts_cam, pts_base)
        self._cal_points_cam = pts_cam
        self._cal_points_base = pts_base

        rbf_cls = _get_rbf_class()
        if rbf_cls is None:
            logger.warning(
                "scipy not available — solve_rbf falling back to affine"
            )
            self._rbf_fitted = False
            self._rbf_interp = None
            self._rbf_x = None
            self._rbf_y = None
            return

        try:
            interp = rbf_cls(pts_cam, pts_base, kernel="thin_plate_spline")
            self._rbf_interp = interp
            self._rbf_x = pts_cam
            self._rbf_y = pts_base
            self._rbf_fitted = True
            logger.info(
                "RBF calibration fitted with %d points", len(pts_cam)
            )
        except Exception as exc:
            logger.warning(
                "RBF fitting failed (%s) — falling back to affine", exc
            )
            self._rbf_fitted = False
            self._rbf_interp = None

    # ------------------------------------------------------------------
    # Error statistics
    # ------------------------------------------------------------------

    def get_error_stats(self) -> dict:
        """Return mean/max/per-point errors from calibration data.

        Returns a dict with keys:
            mean_m (float | None): mean residual in metres
            max_m (float | None): max residual in metres
            per_point_m (list[float] | None): per-point residuals
            num_points (int): number of calibration points (0 if none)
        """
        if self._cal_points_cam is None or self._cal_points_base is None:
            return {
                "mean_m": None,
                "max_m": None,
                "per_point_m": None,
                "num_points": 0,
            }

        pts_cam = self._cal_points_cam
        pts_base = self._cal_points_base

        predicted = np.array(
            [self.camera_to_base(p) for p in pts_cam], dtype=np.float64
        )
        errors = np.linalg.norm(predicted - pts_base, axis=1)
        return {
            "mean_m": float(np.mean(errors)),
            "max_m": float(np.max(errors)),
            "per_point_m": errors.tolist(),
            "num_points": len(errors),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save calibration matrix (and optional RBF training data) to files.

        Primary: <path>.npy  — 4x4 affine matrix
        Sidecar: <path>_rbf.npz — RBF training data when fitted

        Args:
            path: File path (typically ends in .npy).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self._matrix)
        logger.info("Calibration matrix saved to %s", path)

        if self._rbf_fitted and self._rbf_x is not None and self._rbf_y is not None:
            sidecar = p.with_name(p.stem + "_rbf.npz")
            np.savez(str(sidecar), x=self._rbf_x, y=self._rbf_y)
            logger.info("RBF training data saved to %s", sidecar)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        """Load calibration from .npy file (+ optional RBF sidecar).

        Args:
            path: File path previously created by save().

        Returns:
            Calibration instance with loaded matrix (and RBF if sidecar exists).

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If file does not contain a valid 4x4 matrix.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        matrix = np.load(str(p))
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected (4, 4) matrix, got shape {matrix.shape}")
        cal = cls()
        cal._matrix = matrix.astype(np.float64)

        # Attempt to load RBF sidecar
        sidecar = p.with_name(p.stem + "_rbf.npz")
        if sidecar.exists():
            data = np.load(str(sidecar))
            rbf_x = data["x"]
            rbf_y = data["y"]
            rbf_cls = _get_rbf_class()
            if rbf_cls is not None:
                try:
                    interp = rbf_cls(rbf_x, rbf_y, kernel="thin_plate_spline")
                    cal._rbf_interp = interp
                    cal._rbf_x = rbf_x
                    cal._rbf_y = rbf_y
                    cal._rbf_fitted = True
                    cal._cal_points_cam = rbf_x
                    cal._cal_points_base = rbf_y
                    logger.info("RBF sidecar loaded from %s", sidecar)
                except Exception as exc:
                    logger.warning("Failed to refit RBF from sidecar: %s", exc)
            else:
                # Store data even if scipy unavailable
                cal._rbf_x = rbf_x
                cal._rbf_y = rbf_y
                cal._cal_points_cam = rbf_x
                cal._cal_points_base = rbf_y

        return cal


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _solve_affine_matrix(pts_cam: np.ndarray, pts_base: np.ndarray) -> np.ndarray:
    """Core least-squares affine solve. Both inputs must be float64 (N,3)."""
    n = pts_cam.shape[0]
    A = np.hstack([pts_cam, np.ones((n, 1))])  # (N, 4)
    coeff, _, _, _ = np.linalg.lstsq(A, pts_base, rcond=None)  # (4, 3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = coeff[:3, :].T
    T[:3, 3] = coeff[3, :]
    return T


def _warn_if_no_z_variation(pts_cam: np.ndarray, method: str) -> None:
    """Log a warning when Z coordinates have no variation."""
    if pts_cam.ndim == 2 and pts_cam.shape[0] >= 2:
        z_std = float(np.std(pts_cam[:, 2]))
        if z_std < 1e-6:
            logger.warning(
                "%s: all camera Z values are the same (std=%.2e). "
                "Include Z variation for better calibration accuracy.",
                method,
                z_std,
            )
