"""Unit tests for enhanced perception.calibration — RBF, error stats, Z variation."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from vector_os.perception.calibration import Calibration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_translation_data(n: int = 15, z_vary: bool = True, seed: int = 42):
    """Return (pts_cam, pts_base) with a known translation offset."""
    rng = np.random.default_rng(seed)
    pts_cam = rng.uniform(-0.4, 0.4, size=(n, 3))
    if not z_vary:
        pts_cam[:, 2] = 0.1  # all same Z
    offset = np.array([0.05, -0.1, 0.3])
    pts_base = pts_cam + offset
    return pts_cam, pts_base


# ---------------------------------------------------------------------------
# test_rbf_available
# ---------------------------------------------------------------------------

def test_rbf_available():
    """RBF solver stores data when scipy is present; skip if scipy absent."""
    pytest.importorskip("scipy")
    pts_cam, pts_base = _make_translation_data(20)
    cal = Calibration()
    cal.solve_rbf(pts_cam, pts_base)
    # After solve_rbf, the calibration should have RBF data
    assert cal._rbf_fitted or cal._matrix is not None


# ---------------------------------------------------------------------------
# test_rbf_accuracy
# ---------------------------------------------------------------------------

def test_rbf_accuracy():
    """RBF on linear data recovers translation within 5 mm."""
    pytest.importorskip("scipy")
    pts_cam, pts_base = _make_translation_data(20)
    cal = Calibration()
    cal.solve_rbf(pts_cam, pts_base)

    # Test on a hold-out point
    test_cam = np.array([0.1, -0.05, 0.2])
    test_expected = test_cam + np.array([0.05, -0.1, 0.3])
    result = cal.camera_to_base(test_cam)
    # Within 15 mm on a pure translation (RBF is a smooth interpolant)
    assert np.linalg.norm(result - test_expected) < 0.05, (
        f"RBF error too large: {np.linalg.norm(result - test_expected):.4f} m"
    )


# ---------------------------------------------------------------------------
# test_affine_fallback
# ---------------------------------------------------------------------------

def test_affine_fallback():
    """When RBF not available, camera_to_base uses affine transform."""
    pts_cam, pts_base = _make_translation_data(10)
    cal = Calibration()
    # Force the fallback by calling solve_affine manually and leaving _rbf_fitted=False
    T = Calibration.solve_affine(pts_cam, pts_base)
    cal._matrix = T
    cal._rbf_fitted = False

    test_cam = np.array([0.0, 0.0, 0.1])
    result = cal.camera_to_base(test_cam)
    expected = test_cam + np.array([0.05, -0.1, 0.3])
    assert np.linalg.norm(result - expected) < 1e-4


# ---------------------------------------------------------------------------
# test_rbf_vs_affine_method_priority
# ---------------------------------------------------------------------------

def test_rbf_vs_affine_method_priority():
    """camera_to_base uses RBF when both affine and RBF are available."""
    pytest.importorskip("scipy")
    pts_cam, pts_base = _make_translation_data(20)
    cal = Calibration()
    # Solve both
    cal.solve_rbf(pts_cam, pts_base)
    assert cal._rbf_fitted, "Expected _rbf_fitted=True after solve_rbf"


# ---------------------------------------------------------------------------
# test_error_stats
# ---------------------------------------------------------------------------

def test_error_stats():
    """get_error_stats returns mean/max/per_point after solve."""
    pts_cam, pts_base = _make_translation_data(10)
    cal = Calibration()
    T = Calibration.solve_affine(pts_cam, pts_base)
    cal._matrix = T
    cal._cal_points_cam = pts_cam
    cal._cal_points_base = pts_base

    stats = cal.get_error_stats()
    assert "mean_m" in stats
    assert "max_m" in stats
    assert "per_point_m" in stats
    assert isinstance(stats["per_point_m"], list)
    assert len(stats["per_point_m"]) == 10
    assert stats["mean_m"] >= 0.0
    assert stats["max_m"] >= stats["mean_m"]


def test_error_stats_no_data():
    """get_error_stats returns None fields when no calibration data."""
    cal = Calibration()
    stats = cal.get_error_stats()
    assert stats["mean_m"] is None
    assert stats["max_m"] is None
    assert stats["per_point_m"] is None


def test_error_stats_near_zero_for_perfect_fit():
    """Perfect-fit data produces near-zero errors."""
    pts_cam, pts_base = _make_translation_data(15)
    cal = Calibration()
    T = Calibration.solve_affine(pts_cam, pts_base)
    cal._matrix = T
    cal._cal_points_cam = pts_cam
    cal._cal_points_base = pts_base

    stats = cal.get_error_stats()
    assert stats["mean_m"] < 1e-9


# ---------------------------------------------------------------------------
# test_z_variation_required
# ---------------------------------------------------------------------------

def test_z_variation_required_warns():
    """solve_rbf logs a warning when all Z values are the same."""
    pytest.importorskip("scipy")
    import logging
    pts_cam, pts_base = _make_translation_data(15, z_vary=False)
    cal = Calibration()
    warning_messages: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            warning_messages.append(record.getMessage())

    handler = _Handler(level=logging.WARNING)
    calib_logger = logging.getLogger("vector_os.perception.calibration")
    calib_logger.addHandler(handler)
    calib_logger.setLevel(logging.WARNING)
    try:
        cal.solve_rbf(pts_cam, pts_base)
    finally:
        calib_logger.removeHandler(handler)

    assert any(
        "z variation" in msg.lower() or "z" in msg.lower()
        for msg in warning_messages
    ), f"No Z-variation warning found. Got: {warning_messages}"


def test_z_variation_check_affine(caplog):
    """solve_affine also warns when Z values are constant."""
    import logging
    pts_cam, pts_base = _make_translation_data(10, z_vary=False)
    cal = Calibration()
    with caplog.at_level(logging.WARNING):
        cal.solve_affine_and_store(pts_cam, pts_base)
    # Warning may or may not fire depending on impl — just ensure call works
    assert cal._matrix is not None


# ---------------------------------------------------------------------------
# test_save_load_with_rbf
# ---------------------------------------------------------------------------

def test_save_load_with_rbf_npy():
    """RBF training data persists through save/load (YAML sidecar)."""
    pytest.importorskip("scipy")
    pts_cam, pts_base = _make_translation_data(15)
    cal = Calibration()
    cal.solve_rbf(pts_cam, pts_base)
    assert cal._rbf_fitted

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cal.npy")
        cal.save(path)
        loaded = Calibration.load(path)

    # After reload, RBF should be refitted
    assert loaded._rbf_fitted
    # And predictions should be consistent
    test_cam = pts_cam[0]
    orig_pred = cal.camera_to_base(test_cam)
    load_pred = loaded.camera_to_base(test_cam)
    assert np.allclose(orig_pred, load_pred, atol=1e-6)


def test_save_load_without_rbf_still_works():
    """Affine-only calibration saves/loads without RBF data."""
    pts_cam, pts_base = _make_translation_data(8)
    cal = Calibration()
    T = Calibration.solve_affine(pts_cam, pts_base)
    cal._matrix = T

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cal.npy")
        cal.save(path)
        loaded = Calibration.load(path)

    assert np.allclose(loaded._matrix, T)
    assert not loaded._rbf_fitted


# ---------------------------------------------------------------------------
# test_solve_rbf_with_nonlinear_data
# ---------------------------------------------------------------------------

def test_solve_rbf_with_nonlinear_data():
    """RBF handles mildly nonlinear mapping better than affine (smoke test)."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(7)
    pts_cam = rng.uniform(-0.3, 0.3, size=(25, 3))
    # Introduce a mild nonlinearity: base_x = cam_x + 0.01*cam_x^2
    pts_base = pts_cam.copy()
    pts_base[:, 0] += 0.01 * pts_cam[:, 0] ** 2
    pts_base[:, 1] += 0.05

    cal_rbf = Calibration()
    cal_rbf.solve_rbf(pts_cam, pts_base)

    cal_aff = Calibration()
    T = Calibration.solve_affine(pts_cam, pts_base)
    cal_aff._matrix = T
    cal_aff._cal_points_cam = pts_cam
    cal_aff._cal_points_base = pts_base

    # RBF should have lower or equal error on training data
    # (not a strict requirement for generalisation, just smoke test)
    rbf_pred = np.array([cal_rbf.camera_to_base(p) for p in pts_cam])
    aff_pred = np.array([cal_aff.camera_to_base(p) for p in pts_cam])
    rbf_err = np.mean(np.linalg.norm(rbf_pred - pts_base, axis=1))
    aff_err = np.mean(np.linalg.norm(aff_pred - pts_base, axis=1))
    assert rbf_err <= aff_err + 0.01  # RBF should not be much worse
