# Task T8 — MobilePickSkill

## Files

- `vector_os_nano/skills/mobile_pick.py` — 168 LoC (with docstring and comments)
- `tests/skills/test_mobile_pick.py` — 17 tests (8 primary + 9 helper/guard coverage tests)

## Tests

17/17 pass. 8 primary tests as specified in task contract:

1. already_reachable_skips_navigate
2. calls_navigate_then_wait_then_pick_in_order
3. nav_failed_returns_nav_failed
4. wait_stable_timeout_returns_wait_stable_timeout
5. propagates_pick_ik_unreachable_failure
6. object_not_found_returns_object_not_found
7. no_base_returns_no_base
8. skip_navigate_param_honoured

9 additional tests cover: no_arm, no_gripper, no_world_model guards; _ang_diff wrap-around (positive and negative); _dist_xy; _wait_stable (true path and timeout path).

## Coverage

98% (94 stmts, 2 missed: line 57 triple-wrap guard in _ang_diff, line 86 dt<=0 guard in _wait_stable — both defensive unreachable branches under normal clock).

## Ruff

Clean on both files.

## Deviations from spec

- WorldModel.update_object() does not exist; corrected to add_object() in test helper (observed via RED run). Implementation unchanged.
- _MonotonicCounter class defined in tests but unused after _wait_stable was patched in primary tests; removed the dead assignment (ruff caught it).
- `from typing import TYPE_CHECKING` block with empty body removed by ruff auto-fix (was a noop import guard).

## Wave regression

79/79 skills suite tests pass (Wave 1 + Wave 2 + T8).

## Verdict

DONE
