# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 21: Terrain classification and threshold coordination tests."""
import os
import re
import yaml
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NAV_STACK = os.path.expanduser("~/Desktop/vector_navigation_stack")

_LAUNCH_VNAV = os.path.join(_REPO, "scripts", "launch_vnav.sh")
_GO2_CFG = os.path.join(
    _NAV_STACK, "src/base_autonomy/local_planner/config/unitree/unitree_go2.yaml"
)
_TARE_CFG = os.path.join(_REPO, "config", "tare_go2_indoor.yaml")
_BRIDGE = os.path.join(_REPO, "scripts", "go2_vnav_bridge.py")


def _load_local_planner_config():
    with open(_GO2_CFG) as f:
        data = yaml.safe_load(f)
    return data.get("localPlanner", {}).get("ros__parameters", {})


def _load_tare_config():
    with open(_TARE_CFG) as f:
        data = yaml.safe_load(f)
    return data.get("tare_planner_node", {}).get("ros__parameters", {})


def _read_launch_vnav():
    with open(_LAUNCH_VNAV) as f:
        return f.read()


class TestObstacleThresholdCoordination:
    """All three pipeline components must use coordinated obstacle thresholds."""

    def test_terrain_analysis_threshold(self):
        """terrainAnalysis obstacleHeightThre should be 0.15 (from --ros-args)."""
        src = _read_launch_vnav()
        match = re.search(r'obstacleHeightThre:=(\S+)', src)
        assert match, "obstacleHeightThre not found in launch_vnav.sh --ros-args"
        val = float(match.group(1))
        assert val == 0.15, f"terrainAnalysis obstacleHeightThre should be 0.15, got {val}"

    def test_local_planner_threshold(self):
        """localPlanner obstacleHeightThre should be 0.15."""
        cfg = _load_local_planner_config()
        assert cfg["obstacleHeightThre"] == 0.15

    def test_thresholds_match(self):
        """terrainAnalysis and localPlanner thresholds must be equal."""
        lp = _load_local_planner_config()["obstacleHeightThre"]
        src = _read_launch_vnav()
        match = re.search(r'obstacleHeightThre:=(\S+)', src)
        ta = float(match.group(1))
        assert lp == ta, f"Mismatch: localPlanner={lp}, terrainAnalysis={ta}"

    def test_tare_terrain_collision_threshold(self):
        """TARE kTerrainCollisionThreshold should be >= localPlanner threshold."""
        lp = _load_local_planner_config()["obstacleHeightThre"]
        tare = _load_tare_config()
        tct = tare.get("kTerrainCollisionThreshold", 0.3)
        assert tct >= lp, f"TARE collision threshold ({tct}) < localPlanner ({lp})"

    def test_threshold_above_go2_foot_height(self):
        """Threshold must be above Go2 foot height (~0.08m) to avoid foot detection."""
        cfg = _load_local_planner_config()
        assert cfg["obstacleHeightThre"] > 0.08

    def test_threshold_below_furniture(self):
        """Threshold must be below typical furniture legs (~0.3m)."""
        cfg = _load_local_planner_config()
        assert cfg["obstacleHeightThre"] < 0.3


class TestTerrainAnalysisGoParameters:
    """Verify Go2-specific terrainAnalysis parameters are set."""

    def test_clear_dynamic_obstacles(self):
        src = _read_launch_vnav()
        assert "clearDyObs:=true" in src

    def test_limit_ground_lift(self):
        src = _read_launch_vnav()
        assert "limitGroundLift:=true" in src

    def test_max_ground_lift_tight(self):
        src = _read_launch_vnav()
        match = re.search(r'maxGroundLift:=(\S+)', src)
        assert match
        assert float(match.group(1)) <= 0.1, "maxGroundLift should be tight for gait oscillation"

    def test_max_rel_z(self):
        src = _read_launch_vnav()
        match = re.search(r'maxRelZ:=(\S+)', src)
        assert match
        assert float(match.group(1)) >= 0.3


class TestBridgePointcloudIntensity:
    """Verify bridge pointcloud intensity calculation."""

    def test_intensity_is_height_above_ground(self):
        """Bridge should compute intensity = z - ground_z."""
        with open(_BRIDGE) as f:
            src = f.read()
        # Look for intensity calculation
        assert "ground_z" in src, "Bridge should compute ground_z for intensity"
