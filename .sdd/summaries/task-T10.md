# Task T10 — Real rclpy Integration Tests (Coexist Regression Guard)

**Status:** DONE
**Agent:** Gamma
**Date:** 2026-04-19

## rclpy version detected

rclpy is installed (ROS2 Jazzy, `/opt/ros/jazzy/lib/python3.12/site-packages/rclpy`). No `__version__` attribute exposed; import + init + shutdown verified functional.

## Test output

```
tests/hardware/sim/test_ros2_proxies_coexist.py::test_three_stub_nodes_spin_concurrently_no_already_spinning PASSED
tests/hardware/sim/test_ros2_proxies_coexist.py::test_runtime_shutdown_tears_down_cleanly PASSED
tests/hardware/sim/test_ros2_proxies_coexist.py::test_add_node_after_shutdown_raises_or_reinits PASSED
3 passed in 0.46s
```

Wave 1+2 regression: 10/10 pass (no regressions).

## Ros2Runtime gaps found

None requiring escalation. One behaviour documented during test authoring:

- `Ros2Runtime.shutdown()` correctly skips `rclpy.shutdown()` when it did not own the init (`_we_inited_rclpy == False`). This is the correct guard — if the fixture (or another caller) inited rclpy, the runtime must not tear it down. No bug; working as designed.
- `test_runtime_shutdown_tears_down_cleanly` was updated to reflect this: it asserts `is_running == False` (the primary executor contract) rather than `rclpy.ok() == False` (which depends on init ownership).

## pyproject.toml marker addition

YES — added `"ros2: requires rclpy installed and a live DDS context"` to `[tool.pytest.ini_options] markers` list. Without this, `pytest -m ros2` raised `PytestUnknownMarkWarning`.

## Design decision documented in tests

The `_isolate_runtime` fixture calls `rclpy.init()` before each test so `Node()` constructors succeed. This means `Ros2Runtime.add_node()` sees `rclpy.ok()==True` and sets `_we_inited_rclpy=False`, which is the correct code path when rclpy is externally managed. All three tests exercise real DDS pub/sub or real executor lifecycle.
