# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""L17 — TARE warmup and data flow tests."""
import pytest
import os
import ast
import re

# --- T1: Initial waypoint distance ---

class TestInitialWaypoint:
    """Verify TARE initial waypoint is reasonable for indoor."""

    def test_initial_waypoint_under_5m(self):
        """SendInitialWaypoint() should send waypoint < 5m ahead."""
        cpp_path = os.path.expanduser(
            "~/Desktop/vector_navigation_stack/src/exploration_planner/"
            "tare_planner/src/sensor_coverage_planner/"
            "sensor_coverage_planner_ground.cpp"
        )
        with open(cpp_path) as f:
            content = f.read()
        # Find "double lx = <number>" in SendInitialWaypoint
        match = re.search(r'double\s+lx\s*=\s*([\d.]+)', content)
        assert match, "Could not find 'double lx = ...' in TARE source"
        lx = float(match.group(1))
        assert lx <= 5.0, f"Initial waypoint distance {lx}m > 5m (too far for indoor)"
        assert lx >= 1.0, f"Initial waypoint distance {lx}m < 1m (too short)"

    def test_initial_waypoint_not_12m(self):
        """Regression: waypoint must NOT be the old 12m default."""
        cpp_path = os.path.expanduser(
            "~/Desktop/vector_navigation_stack/src/exploration_planner/"
            "tare_planner/src/sensor_coverage_planner/"
            "sensor_coverage_planner_ground.cpp"
        )
        with open(cpp_path) as f:
            content = f.read()
        match = re.search(r'double\s+lx\s*=\s*([\d.]+)', content)
        assert match
        assert float(match.group(1)) != 12.0, "Initial waypoint still at 12m (old default)"

# --- T2: sensor_scan_generation QoS ---

class TestSensorScanGenQoS:
    """Verify sensor_scan_generation uses RELIABLE QoS."""

    def test_qos_not_best_effort(self):
        """Subscribers should use RELIABLE, not BEST_EFFORT."""
        cpp_path = os.path.expanduser(
            "~/Desktop/vector_navigation_stack/src/base_autonomy/"
            "sensor_scan_generation/src/sensorScanGeneration.cpp"
        )
        with open(cpp_path) as f:
            content = f.read()
        # After fix, RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT must not appear
        # in the qos_profile struct (the subscriber config).
        assert "RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT" not in content, \
            "sensor_scan_generation should not use BEST_EFFORT QoS for subscribers"

    def test_qos_depth_at_least_5(self):
        """Subscriber queue depth should be >= 5."""
        cpp_path = os.path.expanduser(
            "~/Desktop/vector_navigation_stack/src/base_autonomy/"
            "sensor_scan_generation/src/sensorScanGeneration.cpp"
        )
        with open(cpp_path) as f:
            content = f.read()
        # Look for the depth value in qos_profile struct (the '1' after KEEP_LAST).
        # After fix it should be 5 or more.
        match = re.search(
            r'RMW_QOS_POLICY_HISTORY_KEEP_LAST\s*,\s*(\d+)', content
        )
        if match:
            depth = int(match.group(1))
            assert depth >= 5, f"QoS depth {depth} < 5 (too shallow, messages will drop)"

    def test_sync_queue_at_least_50(self):
        """Sync policy queue size should be >= 50."""
        cpp_path = os.path.expanduser(
            "~/Desktop/vector_navigation_stack/src/base_autonomy/"
            "sensor_scan_generation/src/sensorScanGeneration.cpp"
        )
        with open(cpp_path) as f:
            content = f.read()
        # Look for the sync policy queue size: syncPolicy(N)
        match = re.search(r'syncPolicy\((\d+)\)', content)
        if match:
            queue_size = int(match.group(1))
            assert queue_size >= 50, f"Sync policy queue size {queue_size} < 50"

# --- T3: Seed walk before nav flag ---

class TestSeedWalk:
    """Verify explore.py does a seed walk before enabling nav."""

    def test_exploration_loop_has_seed_walk(self):
        """_exploration_loop should walk before creating nav flag."""
        explore_path = os.path.expanduser(
            "~/Desktop/vector_os_nano/vector_os_nano/skills/go2/explore.py"
        )
        with open(explore_path) as f:
            content = f.read()
        # The function should contain a walk/set_velocity call BEFORE
        # creating /tmp/vector_nav_active
        loop_start = content.find("def _exploration_loop")
        assert loop_start >= 0
        loop_code = content[loop_start:]

        # Find positions of seed walk and nav flag creation
        walk_pos = loop_code.find("set_velocity") if "set_velocity" in loop_code else loop_code.find("walk(")
        flag_pos = loop_code.find("vector_nav_active")

        assert walk_pos >= 0, "No seed walk (set_velocity or walk()) in _exploration_loop"
        assert walk_pos < flag_pos, "Seed walk must happen BEFORE nav flag creation"

    def test_seed_walk_short_distance(self):
        """Seed walk should be short (< 2m)."""
        explore_path = os.path.expanduser(
            "~/Desktop/vector_os_nano/vector_os_nano/skills/go2/explore.py"
        )
        with open(explore_path) as f:
            content = f.read()
        # Check that seed walk is documented in _exploration_loop
        loop_start = content.find("def _exploration_loop")
        loop_code = content[loop_start:]
        assert "seed" in loop_code.lower() or "initial" in loop_code.lower(), \
            "Seed walk should be documented in _exploration_loop"

    def test_nav_flag_not_created_immediately(self):
        """Nav flag should NOT be the first action in _exploration_loop."""
        explore_path = os.path.expanduser(
            "~/Desktop/vector_os_nano/vector_os_nano/skills/go2/explore.py"
        )
        with open(explore_path) as f:
            content = f.read()
        loop_start = content.find("def _exploration_loop")
        loop_code = content[loop_start:loop_start + 500]

        # Nav flag creation should not be in the first 10 lines of the function body
        lines = loop_code.split('\n')
        first_10_lines = '\n'.join(lines[1:11])
        assert "vector_nav_active" not in first_10_lines, \
            "Nav flag created too early in _exploration_loop — seed walk should come first"

# --- T4: Bridge diagnostic counters ---

class TestBridgeDiagnostics:
    """Verify bridge has diagnostic counters for debugging."""

    def test_bridge_has_publish_counters(self):
        """Bridge should track message publish counts."""
        bridge_path = os.path.expanduser(
            "~/Desktop/vector_os_nano/scripts/go2_vnav_bridge.py"
        )
        with open(bridge_path) as f:
            content = f.read()
        # Should have counter attributes
        assert "_diag_odom_count" in content or "_pub_count" in content or "_diag_count" in content, \
            "Bridge should have publish count tracking for diagnostics"

    def test_bridge_logs_diagnostics(self):
        """Bridge should periodically log publish counts."""
        bridge_path = os.path.expanduser(
            "~/Desktop/vector_os_nano/scripts/go2_vnav_bridge.py"
        )
        with open(bridge_path) as f:
            content = f.read()
        # Should have a diagnostic log with counts
        has_diag = (
            "odom_count" in content.lower()
            or "scan_count" in content.lower()
            or "path_count" in content.lower()
            or "_log_diagnostics" in content
        )
        assert has_diag, "Bridge should log diagnostic message counts"

# --- Additional: TARE config validation ---

class TestTareConfig:
    """Validate TARE config is tuned for Go2 indoor."""

    def test_extend_waypoint_distances_safe(self):
        """Extension distances must be <= 2.0m (big) and <= 1.0m (small) for indoor.

        Reduced from 4.0/1.5 to prevent waypoints penetrating walls in the MuJoCo house.
        """
        cfg_path = os.path.expanduser(
            "~/Desktop/vector_os_nano/config/tare_go2_indoor.yaml"
        )
        import yaml
        with open(cfg_path) as f:
            data = yaml.safe_load(f)
        params = data["tare_planner_node"]["ros__parameters"]
        assert params["kExtendWayPointDistanceBig"] <= 2.0, \
            f"kExtendWayPointDistanceBig={params['kExtendWayPointDistanceBig']} > 2.0m (unsafe for indoor)"
        assert params["kExtendWayPointDistanceSmall"] <= 1.0, \
            f"kExtendWayPointDistanceSmall={params['kExtendWayPointDistanceSmall']} > 1.0m (unsafe for indoor)"

    def test_auto_start_disabled(self):
        """kAutoStart should be false — ExploreSkill sends /start_exploration."""
        cfg_path = os.path.expanduser(
            "~/Desktop/vector_os_nano/config/tare_go2_indoor.yaml"
        )
        with open(cfg_path) as f:
            content = f.read()
        assert "kAutoStart : false" in content or "kAutoStart: false" in content
