# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Single source of truth for SO-101 joint configuration.

All encoder-to-radian mappings, joint names, and motor IDs live here.
Every other module that needs these values must import from this file
instead of maintaining its own copy.

Ported verbatim from vector_ws/src/so101_hardware/so101_hardware/joint_config.py.
No ROS2 imports — pure Python.
"""

import math

# ---------------------------------------------------------------------------
# Primary configuration table
# ---------------------------------------------------------------------------

JOINT_CONFIG: dict = {
    "shoulder_pan": {
        "id": 1, "enc_min": 488,  "enc_max": 2952,
        "rad_min": -1.91986, "rad_max": 1.91986,
    },
    "shoulder_lift": {
        "id": 2, "enc_min": 1050, "enc_max": 3425,
        "rad_min": -1.74533, "rad_max": 1.74533,
    },
    "elbow_flex": {
        "id": 3, "enc_min": 1417, "enc_max": 3580,
        "rad_min": -1.69,    "rad_max": 1.69,
    },
    "wrist_flex": {
        "id": 4, "enc_min": 947,  "enc_max": 3197,
        "rad_min": -1.65806, "rad_max": 1.65806,
    },
    "wrist_roll": {
        "id": 5, "enc_min": 200,  "enc_max": 2961,
        "rad_min": -2.74385, "rad_max": 2.84121,
    },
    "gripper": {
        "id": 6, "enc_min": 1000, "enc_max": 3037,
        "rad_min": -1.0, "rad_max": 1.74533,
    },
}

# ---------------------------------------------------------------------------
# Derived name lists
# ---------------------------------------------------------------------------

ARM_JOINT_NAMES: list = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll",
]

ALL_JOINT_NAMES: list = ARM_JOINT_NAMES + ["gripper"]

# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def enc_to_rad(joint_name: str, enc: int) -> float:
    """Convert encoder ticks to radians for the given joint.

    Values outside [enc_min, enc_max] are clamped before conversion.
    """
    c = JOINT_CONFIG[joint_name]
    enc_clamped = max(c["enc_min"], min(c["enc_max"], enc))
    ratio = (enc_clamped - c["enc_min"]) / (c["enc_max"] - c["enc_min"])
    return c["rad_min"] + ratio * (c["rad_max"] - c["rad_min"])


def rad_to_enc(joint_name: str, rad: float) -> int:
    """Convert radians to encoder ticks for the given joint.

    NaN and Inf inputs are mapped to the midpoint encoder value.
    Values outside [rad_min, rad_max] are clamped before conversion.
    """
    c = JOINT_CONFIG[joint_name]
    if math.isnan(rad) or math.isinf(rad):
        return (c["enc_min"] + c["enc_max"]) // 2
    rad_clamped = max(c["rad_min"], min(c["rad_max"], rad))
    ratio = (rad_clamped - c["rad_min"]) / (c["rad_max"] - c["rad_min"])
    return int(c["enc_min"] + ratio * (c["enc_max"] - c["enc_min"]))
