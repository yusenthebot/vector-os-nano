"""Camera-to-base calibration transform for Vector OS skills.

Loads the affine transform matrix produced by the workspace calibration
script (workspace_calibration.yaml) and applies it to convert 3D positions
from the camera frame to the robot base_link frame.

No ROS2 imports — pure Python + numpy.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Default calibration file location (same as skill_node_v2.py)
_DEFAULT_CALIB_FILE = str(
    Path.home() / "Desktop" / "vector_ws" / "config" / "workspace_calibration.yaml"
)


def load_calibration(calib_file: str | None = None) -> np.ndarray:
    """Load camera→base_link 4x4 affine transform matrix from YAML.

    If the file does not exist, returns a 4x4 identity matrix and logs a
    warning. The identity fallback allows skills to run without a calibration
    file (positions will be in camera frame, which is wrong, but at least the
    pipeline doesn't crash).

    Args:
        calib_file: path to workspace_calibration.yaml.  When None, the
            default path under ~/Desktop/vector_ws/config/ is used.

    Returns:
        4x4 numpy float64 array representing the homogeneous transform.
    """
    path = calib_file or _DEFAULT_CALIB_FILE
    if not Path(path).exists():
        logger.warning(
            "No calibration file at %s — using identity transform. "
            "Run calibrate_workspace first for accurate picks.",
            path,
        )
        return np.eye(4)

    with open(path) as f:
        data = yaml.safe_load(f)

    T = np.array(data["transform_matrix"], dtype=np.float64)
    logger.info(
        "Loaded workspace calibration from %s (mean error: %s mm)",
        path,
        data.get("mean_error_mm", "?"),
    )
    return T


def camera_to_base(cam_pos: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform a 3D position from camera frame to base_link frame.

    Args:
        cam_pos: (3,) array — position in camera frame [x, y, z] in metres.
        T: 4x4 homogeneous transform matrix (camera→base).

    Returns:
        (3,) array — position in base_link frame [x, y, z] in metres.
    """
    p_hom = np.array([cam_pos[0], cam_pos[1], cam_pos[2], 1.0], dtype=np.float64)
    p_base = T @ p_hom
    return p_base[:3]
