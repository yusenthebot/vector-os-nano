# Spec — v2.2 Loco Manipulation Readiness

**Status**: DRAFT (awaiting CEO approval)
**Version**: v2.2 (builds on v2.1 Phase A+B+C on branch `feat/v2.0-vectorengine-unification`)
**Owner**: Architect (Opus)
**Date**: 2026-04-19

---

## 1. Problem Statement

v2.1 implemented **static pick** (PickTopDownSkill: dog stands still, Piper arm
grasps a known object). Live REPL validation on 2026-04-19 found 3 blocking
bugs preventing the skill from running in a real `go2sim with_arm=1` session.
Separately, the system has **no place skill** for the Piper and **no
composition** for "dog walks to object, then picks" — both required for
loco manipulation.

Goal: unblock the live REPL path, add the missing primitives, and compose
them into mobile pick/place so the next session can run an end-to-end
"狗走到桌 A 抓蓝瓶 → 搬到桌 B 放下" demo.

---

## 2. In Scope

1. **Bug 1 fix** — rclpy executor sharing: single `MultiThreadedExecutor`
   shared by all ROS2 proxies in the main process.
2. **Bug 2 fix** — object label matching: Chinese color keyword normaliser
   + single-candidate fallback, so "抓绿色" resolves to "green bottle".
3. **Bug 3 fix** — VGG `source` metadata honoured: skills declaring
   `source: world_model.*` must NOT trigger `detect_*` fallback.
4. **PlaceTopDownSkill** — drop-a-held-object skill, Piper-specific, mirror
   of `pick_top_down` (pre-place → descend → open → lift).
5. **compute_approach_pose utility** — shared (x, y, yaw) planner for
   stopping the dog at a reachable distance facing the target.
6. **MobilePickSkill** — compose navigate → wait_stable → pick_top_down.
7. **MobilePlaceSkill** — compose navigate → wait_stable → place_top_down.
8. **E2E harness** — `scripts/verify_loco_pick_place.py` executing the
   full pick-and-place loop in fresh subprocesses.

## 3. Out of Scope

- Perception-driven grasp (RealSense + SAM3D → pose) — future Phase E.
- Real Piper hardware driver — future.
- Angled/side grasps — top-down only for now.
- Force feedback / torque-controlled grasp — position control only.
- MPC gait for with-arm mode — still sinusoidal (v2.1 decision stands).
- Multi-object clutter reasoning — single target only.
- Receptacle detection / collision avoidance against furniture during place.
- VGG auto-decomposition of "去厨房拿杯子" into nav+pick — VGG layer adds
  it later; mobile_pick is a direct skill for determinism.

## 4. User Stories

- **US-1** As Yusen, when I say `go2sim with_arm=1` and then "抓前面绿色",
  the arm executes `pick_top_down` against the green bottle without
  `Executor is already spinning` errors, `Cannot locate target object`
  errors, or `No perception backend` fallbacks.
- **US-2** As Yusen, after picking a bottle, when I say "放到前面" or
  `place_top_down target_xyz=(11.0, 3.2, 0.25)`, the arm releases the
  object at that location.
- **US-3** As Yusen, from dog spawn (far from table), when I say
  "mobile_pick object_label='blue bottle'", the dog walks to a reachable
  approach pose, waits for stability, and picks the bottle — one command,
  one skill.
- **US-4** As Yusen, running `verify_loco_pick_place.py`, the full
  navigate→pick→navigate→place loop succeeds 3/3 times in fresh
  subprocesses.

## 5. Non-Functional Requirements

- **Reliability**: mobile_pick success rate ≥ 80% across 5 fresh runs
  (same scene, dog spawned at origin).
- **Latency**: Bug fixes must not regress static pick latency (<6 s from
  IK call to grasped_heuristic).
- **Determinism**: E2E harness must NOT rely on wall-clock `time.sleep()`
  for synchronisation; use odom-based stability detection.
- **Safety**: Dog must NOT collide with pick_table during approach
  (nav stack safety radius handles this; mobile_pick merely stops ≥ 55 cm
  away).
- **Backward compat**: existing static pick (30 unit + 30 E2E) must pass
  unchanged. Existing navigate / explore / walk skills unchanged.

## 6. Tech Stack (no new entries)

- MuJoCo 3.6+ (existing go2_piper MJCF)
- ROS2 Jazzy (existing topics + one shared executor)
- Pinocchio-free IK (existing damped-least-squares in PiperROS2Proxy)
- pytest (unit) + subprocess harness (E2E)

No new external dependencies.

## 7. Interface Definitions

### 7.1 New Python modules

| Module | Kind | Public symbols |
|---|---|---|
| `vector_os_nano/hardware/ros2/runtime.py` | singleton helper | `Ros2Runtime`, `get_ros2_runtime() -> Ros2Runtime`, `Ros2Runtime.add_node(node)`, `.remove_node(node)`, `.shutdown()` |
| `vector_os_nano/skills/utils/approach_pose.py` | util | `compute_approach_pose(object_xyz, dog_pose, clearance=0.55, approach_direction=None) -> (x, y, yaw)` |
| `vector_os_nano/skills/place_top_down.py` | skill | `PlaceTopDownSkill` (name `place_top_down`, aliases 放/放下/放到/put/drop) |
| `vector_os_nano/skills/mobile_pick.py` | skill | `MobilePickSkill` (name `mobile_pick`, aliases 去拿/去抓/go-grab) |
| `vector_os_nano/skills/mobile_place.py` | skill | `MobilePlaceSkill` (name `mobile_place`, aliases 拿去放/送到) |

### 7.2 Modified Python modules

| Module | Change |
|---|---|
| `hardware/sim/go2_ros2_proxy.py` | Replace own `rclpy.spin()` thread with `Ros2Runtime.add_node(self._node)`. On `disconnect()`, `Ros2Runtime.remove_node`. |
| `hardware/sim/piper_ros2_proxy.py` | Same replacement for `PiperROS2Proxy` + `PiperGripperROS2Proxy`. |
| `skills/pick_top_down.py` | `_resolve_target` gains Chinese color normaliser + single-candidate fallback. |
| `vcli/cognitive/harness.py` or `goal_decomposer.py` | Respect `parameters[*].source` metadata: if any param lists `world_model.*`, skip auto-injection of `detect_*` fallback step. |
| `vcli/tools/sim_tool.py` | When `with_arm=True`, register `MobilePickSkill` + `MobilePlaceSkill` + `PlaceTopDownSkill` (like current PickTopDownSkill registration). |

### 7.3 ROS2 interfaces

**No new topics / services / actions.** Existing `/piper/joint_cmd`,
`/piper/joint_state`, `/piper/gripper_cmd`, `/odom_to_base`, `/way_point`
are sufficient. This keeps us out of CEO's new-interface gate.

### 7.4 Skill metadata (VGG contract)

`PickTopDownSkill`, `PlaceTopDownSkill`, `MobilePickSkill`,
`MobilePlaceSkill` declare `parameters[*].source` with values from:
- `"world_model.objects.object_id"` — exact id lookup
- `"world_model.objects.label"` — label lookup (with color normaliser)
- `"static"` — literal value, no perception needed
- `"explicit"` — caller provides xyz directly

VGG decomposer reads these and, if no `source: perception.*` appears,
skips inserting a `detect_*` pre-step.

### 7.5 `Ros2Runtime` contract

```python
class Ros2Runtime:
    """Process-singleton holder for rclpy executor + nodes."""

    def add_node(self, node: rclpy.node.Node) -> None:
        """Register node with shared executor. Idempotent.
        Starts the singleton spin thread on first call. Thread-safe."""

    def remove_node(self, node: rclpy.node.Node) -> None:
        """Unregister. Does NOT destroy the node (caller owns it)."""

    def shutdown(self) -> None:
        """Stop executor, join spin thread, call rclpy.shutdown().
        Called at process exit or sim_tool teardown."""
```

Initialisation:
- First `add_node()` call invokes `rclpy.init()` if not initialised,
  creates `MultiThreadedExecutor(num_threads=4)`, starts daemon spin thread.
- Subsequent `add_node()` calls simply `executor.add_node(node)`.
- `get_ros2_runtime()` returns the process singleton (lazy construct).

### 7.6 `compute_approach_pose` contract

```python
def compute_approach_pose(
    object_xyz: tuple[float, float, float],
    dog_pose: tuple[float, float, float],     # (x, y, yaw)
    clearance: float = 0.55,                   # metres
    approach_direction: str | None = None,     # "from_dog" | "from_normal" | None=auto
) -> tuple[float, float, float]:               # (x, y, yaw) world frame
```

Algorithm:
- Compute unit vector `v` from object to dog in world XY plane.
- Approach pose XY = `object_xyz[:2] + clearance * v`.
- Approach yaw = `atan2(object_y - approach_y, object_x - approach_x)`
  (dog faces object).
- `approach_direction="from_dog"` (default): `v` = dog-to-object direction
  reversed. Best when dog already roughly faces object.
- `approach_direction="from_normal"`: not implemented in v2.2 (needs
  table-normal inference); reserved for v2.3 so caller API is stable.

### 7.7 `MobilePickSkill` contract

Parameters (same VGG metadata pattern as PickTopDownSkill):
- `object_id` (str, optional, `source: world_model.objects.object_id`)
- `object_label` (str, optional, `source: world_model.objects.label`)
- `skip_navigate` (bool, default false, `source: static`) — if dog already
  reachable, skip nav step (debug use).

Flow:
1. Resolve target via WorldModel (reuse pick_top_down resolver).
2. `compute_approach_pose(target_xyz, dog_current_pose, clearance=0.55)`.
3. Check if already within reach (`distance < 0.6 m` AND `|yaw_err| < 20°`)
   → skip nav.
4. `base.navigate_to(ax, ay, timeout=20.0)`; if fail → return nav_failed.
5. `wait_stable(base, max_speed=0.05, duration=1.0, timeout=5.0)`.
6. Delegate to `PickTopDownSkill.execute(params, context)`.
7. Propagate its `SkillResult` to caller; add `mobile_pick.nav_distance`
   to `result_data`.

Failure modes: `no_base`, `no_arm`, `no_gripper`, `no_world_model`,
`object_not_found`, `nav_failed`, `wait_stable_timeout`, plus all of
pick_top_down's failure_modes.

### 7.8 `PlaceTopDownSkill` contract

Parameters:
- `target_xyz` (tuple, optional, `source: explicit`) — world XYZ for drop
- `receptacle_id` (str, optional, `source: world_model.objects.object_id`)
- `drop_height` (float, default 0.05, `source: static`) — Z above surface

Preconditions: `gripper_holding_any`
Effects: `gripper_state=open`, `held_object=None`

Flow:
1. Resolve target (explicit xyz OR receptacle object + drop_height).
2. IK `pre_place = target_xyz + (0, 0, pre_drop_height)`.
3. IK `place = target_xyz + (0, 0, drop_height)`.
4. Move pre-place → descend → open → lift (no return-to-home; mirrors
   pick's rationale for orientation preservation).

Failure modes: `no_arm`, `no_gripper`, `ik_unreachable`, `move_failed`,
`gripper_empty`, `receptacle_not_found`.

### 7.9 `MobilePlaceSkill` contract

Parameters: `receptacle_id` | `target_xyz` + `target_room`

Flow: compute_approach_pose(target_xyz) → navigate → wait_stable →
PlaceTopDownSkill.execute.

Failure modes: `no_base`, `gripper_empty`, `receptacle_not_found`,
`nav_failed`, `wait_stable_timeout` + place_top_down's set.

---

## 8. Test Contracts (becomes RED tests in Phase 4)

### 8.1 `Ros2Runtime` (unit)

```python
def test_ros2_runtime_singleton_returns_same_instance()
def test_ros2_runtime_add_node_does_not_raise_when_rclpy_uninitialised()
def test_ros2_runtime_add_then_remove_node_ok()
def test_ros2_runtime_supports_three_concurrent_nodes()  # key: no "already spinning"
def test_ros2_runtime_shutdown_joins_thread()
```

### 8.2 ROS2 proxies (integration, marked @pytest.mark.ros2)

```python
def test_go2_proxy_uses_shared_runtime()          # monkeypatch, assert add_node called
def test_piper_proxy_uses_shared_runtime()
def test_three_proxies_coexist_no_executor_error()  # the actual live-REPL bug reproducer
```

### 8.3 Color normaliser + label fallback (unit)

```python
def test_resolve_target_matches_chinese_color_suffix()    # "绿色" → "green bottle"
def test_resolve_target_matches_english_label_exact()     # "green bottle" → ok (unchanged)
def test_resolve_target_single_pickable_fallback()        # 1 object + empty label → return it
def test_resolve_target_none_for_missing_label()          # "紫色" + no purple object → None
def test_resolve_target_prefers_color_exact_over_substr() # "red" prefers "red can" over "redish bottle"
```

### 8.4 VGG source metadata (unit)

```python
def test_decomposer_skips_detect_when_source_is_world_model()
def test_decomposer_injects_detect_when_source_is_perception()  # existing behaviour unchanged
def test_decomposer_honours_static_source_no_detect()
```

### 8.5 `compute_approach_pose` (unit)

```python
def test_approach_pose_dog_east_of_object_stops_east()
def test_approach_pose_yaw_faces_object()
def test_approach_pose_clearance_exact_distance()      # ||approach - object|| == clearance
def test_approach_pose_zero_dog_offset_raises_or_default()  # degenerate case
```

### 8.6 `PlaceTopDownSkill` (unit, mocked arm/gripper)

```python
def test_place_top_down_happy_path_calls_open_after_descent()
def test_place_top_down_ik_unreachable_returns_failure()
def test_place_top_down_gripper_empty_precondition_not_enforced_in_mock()  # precondition in VGG layer
def test_place_top_down_receptacle_id_resolution()
def test_place_top_down_explicit_xyz_bypasses_receptacle()
```

### 8.7 `MobilePickSkill` (unit, mocked base+arm)

```python
def test_mobile_pick_already_reachable_skips_navigate()
def test_mobile_pick_calls_navigate_then_pick_in_order()
def test_mobile_pick_propagates_pick_failure()
def test_mobile_pick_nav_failed_returns_nav_failed()
def test_mobile_pick_wait_stable_timeout_aborts_before_pick()
```

### 8.8 `MobilePlaceSkill` (unit)

Same structure as mobile_pick, with receptacle_id / target_xyz resolution.

### 8.9 E2E (subprocess harness, marked @pytest.mark.e2e)

```python
# scripts/verify_loco_pick_place.py
# Runs in fresh subprocess per attempt to avoid MuJoCo state pollution.

--repeat 3 --mode pick_only      # baseline
--repeat 3 --mode pick_and_place # full loop
```

Pass criteria:
- mode=pick_only: 3/3 `mobile_pick('blue bottle')` → `grasped_heuristic=True`
- mode=pick_and_place: 3/3 `mobile_pick → mobile_place(target_xyz=...)` →
  bottle z-height within 3 cm of target_z, gripper open at end.

### 8.10 Live REPL smoke (manual)

Yusen-run checklist (documented in `docs/v2.2_live_repl_checklist.md`):
1. `vector-cli` → `go2sim with_arm=1` → no executor errors in stderr
2. `抓前面绿色` → success, held
3. `放下` / `放到 (x, y, z)` → success, released
4. `去拿红色罐头` → mobile_pick runs end-to-end

---

## 9. Acceptance Criteria

| # | Criterion | Verify command |
|---|-----------|----------------|
| AC-1 | Static pick still passes (no regression) | `pytest tests/hardware/sim/test_mujoco_piper.py tests/skills/test_pick_top_down.py` → 30/30 |
| AC-2 | New unit tests pass | `pytest tests/hardware/ros2/test_runtime.py tests/skills/test_place_top_down.py tests/skills/test_mobile_pick.py tests/skills/test_mobile_place.py tests/skills/utils/test_approach_pose.py tests/vcli/cognitive/test_decomposer_source.py` → all pass |
| AC-3 | Three proxies coexist | `pytest tests/hardware/sim/test_ros2_proxies_coexist.py` → pass (reproduces Bug 1 fixed) |
| AC-4 | Label normaliser unblocks Bug 2 | `pytest tests/skills/test_pick_top_down.py::test_resolve_target_matches_chinese_color_suffix` → pass |
| AC-5 | VGG source respect unblocks Bug 3 | `pytest tests/vcli/cognitive/test_decomposer_source.py::test_decomposer_skips_detect_when_source_is_world_model` → pass |
| AC-6 | E2E pick_only 3/3 | `.venv-nano/bin/python scripts/verify_loco_pick_place.py --repeat 3 --mode pick_only` → 3 passes, 0 fails |
| AC-7 | E2E pick_and_place 3/3 | `.venv-nano/bin/python scripts/verify_loco_pick_place.py --repeat 3 --mode pick_and_place` → 3 passes |
| AC-8 | Live REPL smoke (Yusen) | Yusen runs the 4-step checklist in `docs/v2.2_live_repl_checklist.md` and reports success |
| AC-9 | Coverage ≥ 80% on new modules | `pytest --cov=vector_os_nano.skills.mobile_pick --cov=vector_os_nano.skills.mobile_place --cov=vector_os_nano.skills.place_top_down --cov=vector_os_nano.skills.utils.approach_pose --cov=vector_os_nano.hardware.ros2.runtime --cov-report=term-missing` → each ≥ 80 |
| AC-10 | No new lint warnings | `ruff check vector_os_nano/skills/mobile_pick.py vector_os_nano/skills/mobile_place.py vector_os_nano/skills/place_top_down.py vector_os_nano/skills/utils/approach_pose.py vector_os_nano/hardware/ros2/runtime.py` → 0 errors |

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Shared executor refactor breaks existing nav stack | med | high | Keep old `rclpy.spin` path behind feature flag `VECTOR_SHARED_EXECUTOR=1` during transition; default on after AC-3 passes |
| VGG `source` metadata change affects other skills that DO want detect | low | med | Only skip detect when ALL required params list `world_model.*` or `static`; ANY `perception.*` keeps old behaviour |
| `compute_approach_pose` puts dog inside table geometry | med | high | nav stack has safety radius 0.35 m; if approach pose is blocked, nav returns False → mobile_pick returns nav_failed (no crash). Future: bisect clearance |
| wait_stable timeout on bumpy floor | low | med | 5 s timeout, max_speed threshold 0.05 m/s (generous) |
| Chinese color normaliser misses edge cases ("浅绿" = light green) | med | low | Unit test suite covers common cases; unknown colors pass through unchanged (existing substring match) |
| MuJoCo MjModel isolated IK still races under concurrent mobile_pick calls | low | high | Each skill call runs sequentially via VectorEngine; document this invariant in MobilePickSkill docstring |

---

## 11. Open Questions (CEO decisions)

### Q1: Skill registration scope
Should MobilePick/Place be registered ONLY in `with_arm=True` sim mode
(current PickTopDownSkill pattern), or always-on with a runtime `no_arm`
error path?

**Recommendation**: only in `with_arm=True`. Keeps prompt small; explicit
error on missing arm is confusing UX.

### Q2: Color normaliser alphabet
Which Chinese color words to map?

**Recommendation**: `红/红色 → red`, `绿/绿色 → green`, `蓝/蓝色 → blue`,
`黄/黄色 → yellow`, `白/白色 → white`, `黑/黑色 → black`. Six colors
covers current 3 objects + near-future common cases. Unknown colors
fall through unchanged.

### Q3: Mobile pick clearance distance
Default approach distance (`clearance=0.55`)?

**Recommendation**: 0.55 m. Piper reach envelope at top-down extends
~0.45 m in front of dog; 0.55 m gives 0.1 m margin. Configurable via
`config.skills.mobile_pick.clearance`.

### Q4: wait_stable implementation
New module or inline helper in MobilePickSkill?

**Recommendation**: inline helper for now; promote to
`vector_os_nano/skills/utils/stability.py` if mobile_place or future
skills need it.

### Q5: Where to compose the flow — VGG or direct skill?
Should "狗走到桌子拿瓶子" decompose via VGG (navigate + pick) or route
directly to MobilePickSkill?

**Recommendation**: alias MobilePickSkill with `"去拿"/"去抓"` for direct
routing. Let VGG still handle compound sentences like "先去厨房再拿杯子"
where two rooms are involved. Demo-quality deterministic path wins now;
VGG composition is a future improvement.

---

## 12. Dependencies on Prior Work

- ✅ v2.1 Phase A (Piper mounted, dual-mode sim) — committed `0bcbc9e`
- ✅ v2.1 Phase B (MuJoCoPiper + PiperROS2Proxy ArmProtocol) — committed
- ✅ v2.1 Phase C (PickTopDownSkill + ROS2 bridge) — committed `5a673df`
- ✅ Go2ROS2Proxy.navigate_to (FAR path) — existing
- ✅ World model populated via `_populate_pickables_from_mjcf` — existing
- Branch: `feat/v2.0-vectorengine-unification` (5 commits ahead of origin)

---

## 13. Success Metrics

- All 10 acceptance criteria pass
- Yusen successfully runs full live-REPL demo "去桌子拿蓝瓶 → 送到另一桌"
  in one session with no manual intervention
- No regression on static pick (v2.1) or navigate (v1.8) tests
- progress.md updated; 4 uncommitted v2.1 commits + v2.2 commits ready
  to push as a single branch merge
