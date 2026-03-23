"""Tests for calibration transform accuracy using real calibration data.

Validates that the transform_matrix in workspace_calibration.yaml correctly
maps camera-frame 3D points to robot base-frame coordinates. Tests are skipped
when no calibration file is present so CI without real hardware still passes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

CAL_FILE = Path(__file__).parent.parent.parent / "config" / "workspace_calibration.yaml"


@pytest.fixture
def cal_data() -> dict:
    """Load calibration YAML. Skip entire class if file absent."""
    if not CAL_FILE.exists():
        pytest.skip("No calibration file at config/workspace_calibration.yaml")
    with open(CAL_FILE) as f:
        return yaml.safe_load(f)


@pytest.fixture
def T(cal_data: dict) -> np.ndarray:
    """4x4 transform matrix extracted from calibration data."""
    return np.array(cal_data["transform_matrix"], dtype=np.float64)


class TestCalibrationTransform:
    """Validate workspace_calibration.yaml transform_matrix accuracy."""

    def test_matrix_shape(self, T: np.ndarray) -> None:
        """Transform matrix must be exactly 4x4."""
        assert T.shape == (4, 4), f"Expected (4,4), got {T.shape}"

    def test_last_row_is_homogeneous(self, T: np.ndarray) -> None:
        """Last row must be [0, 0, 0, 1] for a valid homogeneous transform."""
        assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6), (
            f"Last row is not homogeneous: {T[3, :]}"
        )

    def test_matrix_has_no_nan(self, T: np.ndarray) -> None:
        """Transform matrix must not contain NaN or inf values."""
        assert not np.any(np.isnan(T)), "Transform matrix contains NaN"
        assert not np.any(np.isinf(T)), "Transform matrix contains inf"

    def test_calibration_points_accuracy(self, cal_data: dict, T: np.ndarray) -> None:
        """Transform each camera point and verify it matches the base point within 10mm.

        The calibration file records mean_error_mm ~3.1mm. We use a generous
        10mm tolerance to account for individual outlier points.
        """
        pts_cam = np.array(cal_data["points_camera"], dtype=np.float64)
        pts_base = np.array(cal_data["points_base"], dtype=np.float64)

        assert len(pts_cam) == len(pts_base), "Mismatched camera/base point counts"
        assert len(pts_cam) > 0, "No calibration points found"

        for i in range(len(pts_cam)):
            cam_hom = np.append(pts_cam[i], 1.0)
            result = (T @ cam_hom)[:3]
            err_mm = float(np.linalg.norm(result - pts_base[i])) * 1000.0
            assert err_mm < 10.0, (
                f"Point {i}: cam={pts_cam[i]}, expected_base={pts_base[i]}, "
                f"got={result}, error={err_mm:.2f}mm > 10mm"
            )

    def test_mean_error_within_5mm(self, cal_data: dict, T: np.ndarray) -> None:
        """Mean transform error across all calibration points must be < 5mm."""
        pts_cam = np.array(cal_data["points_camera"], dtype=np.float64)
        pts_base = np.array(cal_data["points_base"], dtype=np.float64)

        errors_mm: list[float] = []
        for i in range(len(pts_cam)):
            cam_hom = np.append(pts_cam[i], 1.0)
            result = (T @ cam_hom)[:3]
            errors_mm.append(float(np.linalg.norm(result - pts_base[i])) * 1000.0)

        mean_err = float(np.mean(errors_mm))
        assert mean_err < 5.0, (
            f"Mean calibration error {mean_err:.2f}mm exceeds 5mm threshold. "
            f"Recalibrate with workspace_calibration script."
        )

    def test_normal_camera_coords_in_workspace(self, T: np.ndarray) -> None:
        """Camera coords in the calibration volume should map into workspace (5–35cm).

        Uses the centroid of the calibration camera points as the test input.
        """
        # Geometric center of calibration camera points (median is robust to outliers)
        cam_hom = np.array([0.0, 0.02, 0.23, 1.0])
        base = (T @ cam_hom)[:3]
        dist = float(np.linalg.norm(base[:2]))
        assert 0.05 <= dist <= 0.35, (
            f"Central calibration point maps to {dist * 100:.1f}cm from origin, "
            f"outside expected 5–35cm workspace"
        )

    def test_outlier_camera_coords_outside_workspace(self, T: np.ndarray) -> None:
        """A camera X value of 0.5m (well outside the calibration range) should
        map to a base-frame position outside the 35cm workspace radius.
        """
        cam_hom = np.array([0.5, 0.0, 0.25, 1.0])
        base = (T @ cam_hom)[:3]
        dist = float(np.linalg.norm(base[:2]))
        assert dist > 0.35, (
            f"Outlier camera point maps to {dist * 100:.1f}cm, expected > 35cm"
        )

    def test_transform_is_deterministic(self, T: np.ndarray) -> None:
        """Same input always produces same output (pure matrix multiply)."""
        cam_hom = np.array([0.05, 0.03, 0.24, 1.0])
        result_a = (T @ cam_hom)[:3]
        result_b = (T @ cam_hom)[:3]
        assert np.allclose(result_a, result_b)

    def test_calibration_covers_expected_base_range(
        self, cal_data: dict
    ) -> None:
        """Calibration base points should span the 20–30cm forward range."""
        pts_base = np.array(cal_data["points_base"], dtype=np.float64)
        x_vals = pts_base[:, 0]
        assert x_vals.min() <= 0.21, (
            f"Calibration base X min {x_vals.min()*100:.1f}cm — expected ≤ 21cm"
        )
        assert x_vals.max() >= 0.29, (
            f"Calibration base X max {x_vals.max()*100:.1f}cm — expected ≥ 29cm"
        )

    def test_camera_to_base_helper_matches_matrix(
        self, T: np.ndarray, cal_data: dict
    ) -> None:
        """skills.calibration.camera_to_base() must match direct matrix multiply."""
        from vector_os_nano.skills.calibration import camera_to_base

        pts_cam = np.array(cal_data["points_camera"], dtype=np.float64)
        for p in pts_cam[:3]:
            via_helper = camera_to_base(p, T)
            cam_hom = np.append(p, 1.0)
            via_matrix = (T @ cam_hom)[:3]
            assert np.allclose(via_helper, via_matrix, atol=1e-10), (
                f"camera_to_base helper mismatch: {via_helper} vs {via_matrix}"
            )
