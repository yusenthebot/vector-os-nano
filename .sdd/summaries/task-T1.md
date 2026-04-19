# Task T1 — Ros2Runtime singleton

**Status:** DONE
**Agent:** Alpha (Sonnet 4.6)
**Date:** 2026-04-19

## Files Created

1. `vector_os_nano/hardware/ros2/__init__.py` — 1 line (docstring)
2. `vector_os_nano/hardware/ros2/runtime.py` — 173 LoC
3. `tests/hardware/ros2/__init__.py` — empty
4. `tests/hardware/ros2/test_runtime.py` — 226 LoC

## Test Output

```
5 passed in 0.11s
```

Full hardware suite (including pre-existing mujoco_piper tests): **22 passed, 0 failed**.

## Ruff Output

```
All checks passed!
```

## LoC Summary

| File | LoC |
|------|-----|
| runtime.py | 173 |
| test_runtime.py | 226 |
| __init__.py (x2) | 2 |
| **Total** | **401** |

## Deviations / Surprises

1. **Test 2 init-count bug**: `rclpy.ok()` on a MagicMock always returns False even after `rclpy.init()` is called, so the second `add_node()` would call `rclpy.init()` again. Fixed in runtime.py by guarding on `self._we_inited_rclpy` in addition to `rclpy.ok()` — both must be false to trigger init. This guard is semantically correct: once WE called init, we should not call it again regardless of what `ok()` reports.

2. **Atexit ordering noise**: After each test, the atexit handler registered by `add_node()` fires with the real rclpy module (monkeypatch already reverted). Added an explicit `runtime.shutdown()` call in the fixture teardown before resetting `_runtime = None`, which eliminates the stale atexit registration and the resulting stderr exception from `rclpy.shutdown()` on an uninitialised context.

3. **Unused import cleanup**: Initial test file included `threading`, `types`, and `call` from unittest.mock. Removed after ruff flagged them.
