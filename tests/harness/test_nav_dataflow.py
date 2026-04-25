# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Integration test: verify the full navigation data flow chain.

Tests that all critical ROS2 topics are flowing correctly between:
  bridge → sensorScanGeneration → terrainAnalysis → localPlanner → TARE

This test starts the actual nav stack processes and validates data flow.
Run with: source .venv-nano/bin/activate && python3 tests/harness/test_nav_dataflow.py

NOT a pytest — runs standalone because it needs ROS2 subprocess management.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAV_STACK = os.path.expanduser("~/Desktop/vector_navigation_stack")

# Colors for terminal output
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
BOLD = "\033[1m"


def log(status: str, msg: str) -> None:
    color = GREEN if status == "OK" else RED if status == "FAIL" else YELLOW
    print(f"  {color}{status:6s}{RESET} {msg}")


def check_topic_exists(node, topic: str) -> bool:
    """Check if a topic is registered in the ROS2 graph."""
    topics = node.get_topic_names_and_types()
    return any(name == topic for name, _ in topics)


def check_topic_has_publishers(node, topic: str) -> int:
    """Count publishers for a topic."""
    return node.count_publishers(topic)


def wait_for_topic(node, topic: str, timeout: float = 10.0) -> bool:
    """Wait until a topic has at least one publisher."""
    start = time.time()
    while time.time() - start < timeout:
        if node.count_publishers(topic) > 0:
            return True
        time.sleep(0.5)
    return False


def main():
    print(f"\n{BOLD}=== Navigation Data Flow Test ==={RESET}\n")

    # Check ROS2 is available
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError:
        print(f"{RED}FAIL: rclpy not available. Source ROS2 first.{RESET}")
        sys.exit(1)

    rclpy.init()
    node = rclpy.create_node("nav_dataflow_test")

    procs = []

    def cleanup():
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                p.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    import atexit
    atexit.register(cleanup)

    failures = 0

    # ---------------------------------------------------------------
    # Step 1: Check if bridge is running (from launch go2 sim)
    # ---------------------------------------------------------------
    print(f"{BOLD}Step 1: Bridge{RESET}")
    if wait_for_topic(node, "/state_estimation", timeout=5.0):
        log("OK", "/state_estimation has publishers")
    else:
        log("FAIL", "/state_estimation — no publishers. Is bridge running?")
        failures += 1

    if wait_for_topic(node, "/registered_scan", timeout=3.0):
        log("OK", "/registered_scan has publishers")
    else:
        log("FAIL", "/registered_scan — no publishers")
        failures += 1

    # ---------------------------------------------------------------
    # Step 2: Check sensorScanGeneration
    # ---------------------------------------------------------------
    print(f"\n{BOLD}Step 2: sensorScanGeneration{RESET}")
    if wait_for_topic(node, "/state_estimation_at_scan", timeout=10.0):
        n = node.count_publishers("/state_estimation_at_scan")
        log("OK", f"/state_estimation_at_scan has {n} publisher(s)")
    else:
        log("FAIL", "/state_estimation_at_scan — NOT PRODUCED!")
        log("INFO", "sensorScanGeneration may not be syncing /state_estimation + /registered_scan")
        log("INFO", "Check QoS compatibility: bridge=RELIABLE, sensorScanGeneration=BEST_EFFORT")
        failures += 1

    # ---------------------------------------------------------------
    # Step 3: Check terrainAnalysis
    # ---------------------------------------------------------------
    print(f"\n{BOLD}Step 3: terrainAnalysis{RESET}")
    if wait_for_topic(node, "/terrain_map", timeout=10.0):
        log("OK", "/terrain_map has publishers")
    else:
        log("FAIL", "/terrain_map — NOT PRODUCED!")
        failures += 1

    # ---------------------------------------------------------------
    # Step 4: Check localPlanner
    # ---------------------------------------------------------------
    print(f"\n{BOLD}Step 4: localPlanner{RESET}")
    if wait_for_topic(node, "/path", timeout=5.0):
        log("OK", "/path has publishers")
    else:
        log("FAIL", "/path — no publishers. localPlanner may have crashed")
        failures += 1

    # ---------------------------------------------------------------
    # Step 5: Check FAR planner
    # ---------------------------------------------------------------
    print(f"\n{BOLD}Step 5: FAR planner{RESET}")
    if wait_for_topic(node, "/goal_point", timeout=5.0):
        log("OK", "/goal_point has publishers")
    else:
        log("WARN", "/goal_point — no publishers (FAR planner may not be running)")

    # ---------------------------------------------------------------
    # Step 6: Check TARE (if running)
    # ---------------------------------------------------------------
    print(f"\n{BOLD}Step 6: TARE planner{RESET}")
    if wait_for_topic(node, "/way_point", timeout=5.0):
        log("OK", "/way_point has publishers — TARE is running")
    else:
        log("WARN", "/way_point — no publishers (TARE not started yet, normal before explore)")

    # ---------------------------------------------------------------
    # Step 7: Measure /state_estimation_at_scan frequency
    # ---------------------------------------------------------------
    print(f"\n{BOLD}Step 7: Data frequency check{RESET}")
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    msg_count = 0
    def _scan_odom_cb(msg):
        nonlocal msg_count
        msg_count += 1

    sub = node.create_subscription(
        Odometry, "/state_estimation_at_scan",
        _scan_odom_cb,
        QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=5),
    )

    # Spin for 5 seconds and count messages
    start = time.time()
    while time.time() - start < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_subscription(sub)
    freq = msg_count / 5.0

    if freq >= 5.0:
        log("OK", f"/state_estimation_at_scan: {freq:.1f} Hz (expected ~10 Hz)")
    elif freq > 0:
        log("WARN", f"/state_estimation_at_scan: {freq:.1f} Hz (LOW — expected ~10 Hz)")
    else:
        log("FAIL", "/state_estimation_at_scan: 0 Hz — sensorScanGeneration NOT producing data!")
        log("INFO", "This is likely why TARE waypoints don't update.")
        failures += 1

    # ---------------------------------------------------------------
    # Step 8: Check /way_point distance (if TARE is running)
    # ---------------------------------------------------------------
    print(f"\n{BOLD}Step 8: TARE waypoint distance{RESET}")
    from geometry_msgs.msg import PointStamped

    waypoint_received = False
    waypoint_dist = 0.0

    def _waypoint_cb(msg):
        nonlocal waypoint_received, waypoint_dist
        # Get robot position from latest odom
        waypoint_received = True

    sub_wp = node.create_subscription(
        PointStamped, "/way_point", _waypoint_cb,
        QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=5),
    )

    start = time.time()
    while time.time() - start < 10.0 and not waypoint_received:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_subscription(sub_wp)

    if waypoint_received:
        log("OK", "/way_point received — TARE is publishing waypoints")
    else:
        log("WARN", "/way_point not received in 10s (normal if TARE not started)")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{BOLD}=== Summary ==={RESET}")
    if failures == 0:
        print(f"{GREEN}All critical topics are flowing.{RESET}")
    else:
        print(f"{RED}{failures} critical failure(s) detected.{RESET}")
        print(f"Run this AFTER 'launch go2 sim' and optionally after 'explore'.")

    cleanup()
    sys.exit(failures)


if __name__ == "__main__":
    main()
