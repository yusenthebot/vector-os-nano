"""Level 22: Path planning parameter verification tests."""
import os
import yaml
import pytest

_NAV_STACK = os.path.expanduser("~/Desktop/vector_navigation_stack")
_GO2_CFG = os.path.join(
    _NAV_STACK, "src/base_autonomy/local_planner/config/unitree/unitree_go2.yaml"
)
_INSTALLED_CFG = os.path.join(
    _NAV_STACK, "install/local_planner/share/local_planner/config/unitree/unitree_go2.yaml"
)


def _load_local_planner_config(path=_GO2_CFG):
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("localPlanner", {}).get("ros__parameters", {})


class TestGo2VehicleDimensions:
    """Verify vehicle dimensions match Go2 body with safety margin."""

    def test_config_file_exists(self):
        assert os.path.isfile(_GO2_CFG)

    def test_vehicle_width_go2(self):
        cfg = _load_local_planner_config()
        assert cfg["vehicleWidth"] == 0.35, f"vehicleWidth should be 0.35 for Go2, got {cfg['vehicleWidth']}"

    def test_vehicle_length_go2(self):
        cfg = _load_local_planner_config()
        assert cfg["vehicleLength"] == 0.45, f"vehicleLength should be 0.45 for Go2, got {cfg['vehicleLength']}"

    def test_vehicle_width_reasonable(self):
        """Width must be > Go2 body (0.09m) and < old value (0.5m)."""
        cfg = _load_local_planner_config()
        assert 0.1 < cfg["vehicleWidth"] < 0.5

    def test_vehicle_length_reasonable(self):
        """Length must be > Go2 body (0.38m) and < old value (0.6m)."""
        cfg = _load_local_planner_config()
        assert 0.38 < cfg["vehicleLength"] < 0.6

    def test_installed_config_matches_source(self):
        """After colcon build, installed config must match source."""
        if not os.path.isfile(_INSTALLED_CFG):
            pytest.skip("Nav stack not built yet")
        src = _load_local_planner_config(_GO2_CFG)
        inst = _load_local_planner_config(_INSTALLED_CFG)
        assert src["vehicleWidth"] == inst["vehicleWidth"]
        assert src["vehicleLength"] == inst["vehicleLength"]


class TestPathPlanningParams:
    """Verify other critical path planning parameters."""

    def test_obstacle_height_threshold(self):
        cfg = _load_local_planner_config()
        assert cfg["obstacleHeightThre"] == 0.15

    def test_path_scale(self):
        cfg = _load_local_planner_config()
        assert cfg["pathScale"] == 0.75

    def test_max_speed(self):
        cfg = _load_local_planner_config()
        assert cfg["maxSpeed"] == 0.875
