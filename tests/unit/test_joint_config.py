"""Unit tests for SO-101 joint configuration.

Tests enc_to_rad / rad_to_enc conversions, boundary clamping, and NaN/Inf
safety for all six joints. Ported from vector_ws test_mapping.py.

Run with: pytest tests/unit/test_joint_config.py -v
"""

import math

import pytest

from vector_os.hardware.so101.joint_config import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    JOINT_CONFIG,
    enc_to_rad,
    rad_to_enc,
)

# Tolerance constants
EPSILON = 1e-6
ROUND_TRIP_ENC_TOL = 1  # +-1 encoder count acceptable for int round-trip
MID_RAD_TOL = 2e-3      # ~1 enc count expressed in radians (floor-div bias for odd enc ranges)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=list(JOINT_CONFIG.keys()))
def joint(request):
    """Parametrize over all six joints."""
    return request.param


@pytest.fixture(params=ARM_JOINT_NAMES)
def arm_joint(request):
    """Parametrize over the five arm joints."""
    return request.param


# ---------------------------------------------------------------------------
# 1. Boundary tests: enc_min -> rad_min, enc_max -> rad_max
# ---------------------------------------------------------------------------

class TestEncToRadBoundaries:
    """enc_min maps to rad_min and enc_max maps to rad_max for each joint."""

    def test_enc_min_to_rad_min(self, joint):
        cfg = JOINT_CONFIG[joint]
        result = enc_to_rad(joint, cfg["enc_min"])
        assert abs(result - cfg["rad_min"]) < EPSILON, (
            f"{joint}: enc_min {cfg['enc_min']} -> expected rad_min "
            f"{cfg['rad_min']:.6f}, got {result:.6f}"
        )

    def test_enc_max_to_rad_max(self, joint):
        cfg = JOINT_CONFIG[joint]
        result = enc_to_rad(joint, cfg["enc_max"])
        assert abs(result - cfg["rad_max"]) < EPSILON, (
            f"{joint}: enc_max {cfg['enc_max']} -> expected rad_max "
            f"{cfg['rad_max']:.6f}, got {result:.6f}"
        )


# ---------------------------------------------------------------------------
# 2. Midpoint test
# ---------------------------------------------------------------------------

class TestEncToRadMidpoint:
    """Midpoint encoder maps to midpoint radian within tolerance."""

    def test_enc_midpoint_to_rad_midpoint(self, joint):
        cfg = JOINT_CONFIG[joint]
        mid_enc = (cfg["enc_min"] + cfg["enc_max"]) // 2
        mid_rad = (cfg["rad_min"] + cfg["rad_max"]) / 2.0
        result = enc_to_rad(joint, mid_enc)
        # Allow MID_RAD_TOL because integer floor-division of odd enc ranges
        # introduces a half-count bias.
        assert abs(result - mid_rad) <= MID_RAD_TOL, (
            f"{joint}: mid_enc {mid_enc} -> expected ~{mid_rad:.6f}, got {result:.6f}"
        )


# ---------------------------------------------------------------------------
# 3. Round-trip test: rad -> enc -> rad within tolerance
# ---------------------------------------------------------------------------

class TestRadToEncRoundtrip:
    """rad_to_enc(enc_to_rad(enc)) recovers the original radian value.

    We sample the encoder range at ~20 evenly-spaced points and verify
    the worst-case round-trip error stays within +-1 encoder count expressed
    back in radians.
    """

    def test_roundtrip_worst_case(self, joint):
        cfg = JOINT_CONFIG[joint]
        step = max(1, (cfg["enc_max"] - cfg["enc_min"]) // 20)
        worst_err = 0
        worst_enc = cfg["enc_min"]
        for enc in range(cfg["enc_min"], cfg["enc_max"] + 1, step):
            rad = enc_to_rad(joint, enc)
            enc_back = rad_to_enc(joint, rad)
            err = abs(enc_back - enc)
            if err > worst_err:
                worst_err = err
                worst_enc = enc
        assert worst_err <= ROUND_TRIP_ENC_TOL, (
            f"{joint}: worst round-trip error {worst_err} counts at enc={worst_enc} "
            f"(tolerance {ROUND_TRIP_ENC_TOL})"
        )

    def test_rad_min_to_enc_min(self, joint):
        cfg = JOINT_CONFIG[joint]
        result = rad_to_enc(joint, cfg["rad_min"])
        assert abs(result - cfg["enc_min"]) <= ROUND_TRIP_ENC_TOL, (
            f"{joint}: rad_min {cfg['rad_min']:.5f} -> expected enc_min "
            f"{cfg['enc_min']}, got {result}"
        )

    def test_rad_max_to_enc_max(self, joint):
        cfg = JOINT_CONFIG[joint]
        result = rad_to_enc(joint, cfg["rad_max"])
        assert abs(result - cfg["enc_max"]) <= ROUND_TRIP_ENC_TOL, (
            f"{joint}: rad_max {cfg['rad_max']:.5f} -> expected enc_max "
            f"{cfg['enc_max']}, got {result}"
        )


# ---------------------------------------------------------------------------
# 4. Encoder clamping tests
# ---------------------------------------------------------------------------

class TestEncClamping:
    """Out-of-range encoder values are clamped to the configured limits."""

    def test_enc_below_min_clamped_to_rad_min(self, joint):
        cfg = JOINT_CONFIG[joint]
        result = enc_to_rad(joint, cfg["enc_min"] - 500)
        assert abs(result - cfg["rad_min"]) < EPSILON, (
            f"{joint}: enc below min not clamped to rad_min, got {result:.6f}"
        )

    def test_enc_above_max_clamped_to_rad_max(self, joint):
        cfg = JOINT_CONFIG[joint]
        result = enc_to_rad(joint, cfg["enc_max"] + 500)
        assert abs(result - cfg["rad_max"]) < EPSILON, (
            f"{joint}: enc above max not clamped to rad_max, got {result:.6f}"
        )

    def test_rad_below_min_clamped_to_enc_min(self, joint):
        cfg = JOINT_CONFIG[joint]
        result = rad_to_enc(joint, cfg["rad_min"] - 10.0)
        assert abs(result - cfg["enc_min"]) <= ROUND_TRIP_ENC_TOL, (
            f"{joint}: rad below rad_min not clamped to enc_min, got {result}"
        )

    def test_rad_above_max_clamped_to_enc_max(self, joint):
        cfg = JOINT_CONFIG[joint]
        result = rad_to_enc(joint, cfg["rad_max"] + 10.0)
        assert abs(result - cfg["enc_max"]) <= ROUND_TRIP_ENC_TOL, (
            f"{joint}: rad above rad_max not clamped to enc_max, got {result}"
        )


# ---------------------------------------------------------------------------
# 5. NaN / Inf safety tests
# ---------------------------------------------------------------------------

class TestNanSafety:
    """NaN and Inf radian inputs return the midpoint encoder value."""

    def _expected_mid_enc(self, joint: str) -> int:
        cfg = JOINT_CONFIG[joint]
        return (cfg["enc_min"] + cfg["enc_max"]) // 2

    def test_nan_returns_midpoint(self, joint):
        mid = self._expected_mid_enc(joint)
        result = rad_to_enc(joint, float("nan"))
        assert abs(result - mid) <= ROUND_TRIP_ENC_TOL, (
            f"{joint}: NaN input should yield mid enc {mid}, got {result}"
        )

    def test_pos_inf_returns_midpoint(self, joint):
        mid = self._expected_mid_enc(joint)
        result = rad_to_enc(joint, float("inf"))
        assert abs(result - mid) <= ROUND_TRIP_ENC_TOL, (
            f"{joint}: +Inf input should yield mid enc {mid}, got {result}"
        )

    def test_neg_inf_returns_midpoint(self, joint):
        mid = self._expected_mid_enc(joint)
        result = rad_to_enc(joint, float("-inf"))
        assert abs(result - mid) <= ROUND_TRIP_ENC_TOL, (
            f"{joint}: -Inf input should yield mid enc {mid}, got {result}"
        )


# ---------------------------------------------------------------------------
# 6. Name list integrity
# ---------------------------------------------------------------------------

class TestNameLists:
    """ARM_JOINT_NAMES and ALL_JOINT_NAMES are correct and ordered."""

    def test_arm_joint_names_length(self):
        assert len(ARM_JOINT_NAMES) == 5

    def test_all_joint_names_length(self):
        assert len(ALL_JOINT_NAMES) == 6

    def test_gripper_not_in_arm_joint_names(self):
        assert "gripper" not in ARM_JOINT_NAMES

    def test_gripper_in_all_joint_names(self):
        assert "gripper" in ALL_JOINT_NAMES

    def test_arm_joints_in_order(self):
        expected = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        assert ARM_JOINT_NAMES == expected

    def test_all_joint_names_includes_arm_joints(self):
        for name in ARM_JOINT_NAMES:
            assert name in ALL_JOINT_NAMES

    def test_joint_config_contains_all_joints(self):
        for name in ALL_JOINT_NAMES:
            assert name in JOINT_CONFIG

    def test_motor_ids_are_unique(self):
        ids = [cfg["id"] for cfg in JOINT_CONFIG.values()]
        assert len(ids) == len(set(ids)), "Motor IDs must be unique"

    def test_gripper_motor_id_is_6(self):
        assert JOINT_CONFIG["gripper"]["id"] == 6


# ---------------------------------------------------------------------------
# 7. Specific constant spot-checks (regression against vector_ws values)
# ---------------------------------------------------------------------------

class TestConstantValues:
    """Verify key encoder/radian constants match vector_ws source exactly."""

    def test_shoulder_pan_enc_range(self):
        cfg = JOINT_CONFIG["shoulder_pan"]
        assert cfg["enc_min"] == 488
        assert cfg["enc_max"] == 2952

    def test_shoulder_pan_rad_range(self):
        cfg = JOINT_CONFIG["shoulder_pan"]
        assert abs(cfg["rad_min"] - (-1.91986)) < EPSILON
        assert abs(cfg["rad_max"] - 1.91986) < EPSILON

    def test_wrist_roll_asymmetric_rad_range(self):
        """wrist_roll has asymmetric rad limits (not centered on 0)."""
        cfg = JOINT_CONFIG["wrist_roll"]
        assert abs(cfg["rad_min"] - (-2.74385)) < EPSILON
        assert abs(cfg["rad_max"] - 2.84121) < EPSILON

    def test_gripper_enc_range(self):
        cfg = JOINT_CONFIG["gripper"]
        assert cfg["enc_min"] == 1000
        assert cfg["enc_max"] == 3037
