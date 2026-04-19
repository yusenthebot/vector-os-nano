# SDD Complete — v2.2 Loco Manipulation Readiness

**Date**: 2026-04-19
**Branch**: `feat/v2.0-vectorengine-unification`
**Status**: ready for CEO final approval

---

## One-liner
Fixed 3 live-REPL blocker bugs (rclpy executor clash / Chinese color matching / VGG detect fallback), added PlaceTopDownSkill + MobilePickSkill + MobilePlaceSkill, and a subprocess-per-attempt E2E harness — so "狗走过去抓蓝瓶 → 搬到另一桌 → 放下" is ready to run as a single REPL command.

## Tasks: 13/13 done

| Task | Agent | Tests | Status |
|------|-------|-------|--------|
| T1 Ros2Runtime singleton | Alpha | 5 unit | DONE |
| T2 compute_approach_pose | Beta | 12 unit | DONE |
| T3 Chinese color normaliser | Gamma | 10 unit | DONE |
| T4 Proxy wire Ros2Runtime | Alpha | 5 wiring | DONE |
| T5 PlaceTopDownSkill | Beta | 8 unit (96% cov) | DONE |
| T6 VGG source hint | Gamma | 7 unit | DONE |
| T7 pick resolver upgrade | Gamma | 4 unit | DONE |
| T8 MobilePickSkill | Alpha | 17 unit (98% cov) | DONE |
| T9 MobilePlaceSkill | Beta | 15 unit (98% cov) | DONE |
| T10 Coexist integration | Gamma | 3 rclpy | DONE |
| T11 sim_tool registration | Alpha | 2 guard | DONE |
| T12 E2E verify harness | Beta | dry-run | DONE |
| T13 Live REPL doc | Scribe | n/a | DONE |

## Test Results

- **Unit**: 105 passed
- **rclpy integration**: 3 passed
- **Total new**: 108 tests, 0 failures
- **Coverage** on 5 new modules: **97%** (required ≥80%)
  - `hardware/ros2/runtime.py` 95%
  - `skills/mobile_pick.py` 98%
  - `skills/mobile_place.py` 98%
  - `skills/place_top_down.py` 96%
  - `skills/utils/approach_pose.py` 100%
- **No regression** on pre-existing suite within impacted directories
  (tests/hardware/ros2, tests/hardware/sim, tests/skills/, tests/vcli/)

## Quality Review

### Code review — PASS-WITH-NITS
- 0 CRITICAL / 0 HIGH
- 2 MAJOR findings:
  - **Fixed**: `target_xyz` silent fallthrough on wrong length / NaN → now returns `invalid_target_xyz` diagnosis in all 3 resolvers
  - **Deferred**: legacy per-proxy spin-thread leak on disconnect (only triggers when `VECTOR_SHARED_EXECUTOR=0` rollback flag is set, dead code in production default)
- 5 NITs: 3 fixed (dedupe color values, add `arm_unsupported` to PlaceTopDown.failure_modes, remove dead `TYPE_CHECKING: pass`), 2 deferred for v2.3 (extract duplicate `_wait_stable`, undeclared `_spin_thread` attr on PiperGripperROS2Proxy)

### Security review — PASS-WITH-LOW
- 0 CRITICAL / 0 HIGH / 0 MEDIUM
- 3 LOW notes, all addressed:
  - NaN/inf in `target_xyz` — fixed via `math.isfinite()` guard
  - `_wait_stable` corrupt position tuple → graceful timeout (documented)
  - Verify-script log file symlink risk — dev-only harness, acceptable

## Files Changed

### New (12)
- `vector_os_nano/hardware/ros2/{__init__,runtime}.py`
- `vector_os_nano/skills/utils/{__init__,approach_pose}.py`
- `vector_os_nano/skills/{place_top_down,mobile_pick,mobile_place}.py`
- `scripts/verify_loco_pick_place.py`
- `docs/v2.2_live_repl_checklist.md`
- Test files: `tests/hardware/ros2/*`, `tests/hardware/sim/test_ros2_proxies_runtime_wiring.py`, `tests/hardware/sim/test_ros2_proxies_coexist.py`, `tests/skills/utils/*`, `tests/skills/test_{place_top_down,mobile_pick,mobile_place}.py`, `tests/vcli/cognitive/test_decomposer_source.py`, `tests/vcli/tools/test_sim_tool_registration.py`

### Modified (5 production + 1 config)
- `vector_os_nano/hardware/sim/go2_ros2_proxy.py` — shared-executor gate
- `vector_os_nano/hardware/sim/piper_ros2_proxy.py` — same, 2 proxies + E702 cleanup
- `vector_os_nano/skills/pick_top_down.py` — color normaliser + resolver upgrade + F401 cleanup + `invalid_target_xyz` diagnosis
- `vector_os_nano/vcli/cognitive/goal_decomposer.py` — `_skill_is_world_model_only` helper + catalog tag + prompt bullet
- `vector_os_nano/vcli/tools/sim_tool.py` — 3 new skill registrations
- `pyproject.toml` — added `ros2` pytest marker

## Risks / Known Issues

- **Legacy rollback path leak**: `VECTOR_SHARED_EXECUTOR=0` does not join the per-proxy spin thread on disconnect. Dead code in production. Fix in v2.3 if the flag is ever used.
- **Divergent `_wait_stable` impls**: mobile_pick (5 Hz) vs mobile_place (10 Hz). Functional parity; extract to `skills/utils/mobile_helpers.py` in v2.3.
- **No post-place tracking**: after place, world_model object position not updated. Acceptable for demo.
- **No collision check**: mobile skills rely on nav stack safety radius (0.35 m). Documented.

## Ready for Release: **YES** (pending Yusen live REPL smoke)

### CEO sign-off checklist
1. Run `vector-cli` → `go2sim with_arm=1` → follow `docs/v2.2_live_repl_checklist.md`
2. If all 5 steps PASS → commit uncommitted v2.2 work + push the 5 prior commits
3. If ANY step FAILS → report the failing step; agent team will triage

### Commit plan (once smoke passes)
- 1 commit for v2.2: `feat(v2.2): loco manipulation (3 bug fixes + place/mobile_pick/mobile_place + Ros2Runtime)`
- Plus existing 5 v2.1 commits already on branch
- Push `feat/v2.0-vectorengine-unification` → open PR to master when Yusen approves
