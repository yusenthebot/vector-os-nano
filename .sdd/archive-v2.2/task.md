# Task Decomposition — v2.2 Loco Manipulation Readiness

**Status**: draft (awaiting QA review)
**Reads**: `.sdd/spec.md`, `.sdd/plan.md`
**Total tasks**: 13
**Waves**: 4 execution waves, ~4 hrs wall-clock with 3 parallel Sonnet agents

---

## Conventions

- All tasks are TDD: write RED tests first, GREEN minimal impl, REFACTOR under full suite.
- Each task runs in an **isolated subagent** (fresh context) spawned by sdd-execute.
- Package: `vector_os_nano` (single monorepo).
- Python venv: `~/Desktop/vector_os_nano/.venv-nano/bin/python`.
- No task requires ROS2 sim start (integration tests use mocked rclpy).
- Wave gate = all tasks in wave pass unit tests + ruff check + no regression on pre-existing suite.

---

## Wave 1 — Foundations (3 parallel, no dependencies)

### T1 — Ros2Runtime singleton
- **Status**: [ ] pending
- **Agent**: Alpha
- **Depends**: none
- **Files to create**:
  - `vector_os_nano/hardware/ros2/__init__.py` (empty)
  - `vector_os_nano/hardware/ros2/runtime.py` (~80 LoC)
  - `tests/hardware/ros2/__init__.py` (empty)
  - `tests/hardware/ros2/test_runtime.py` (~150 LoC, 5 tests)
- **Input**: `.sdd/spec.md` §7.5 (Ros2Runtime contract), `.sdd/plan.md` §3.1
- **TDD**:
  - **RED** — write 5 tests:
    1. `test_ros2_runtime_singleton_returns_same_instance` — two `get_ros2_runtime()` calls return the same object.
    2. `test_ros2_runtime_add_node_initialises_rclpy_once` — monkeypatch `rclpy.init`, call `add_node()` twice, assert init called once.
    3. `test_ros2_runtime_add_then_remove_node_ok` — add/remove lifecycle no raise.
    4. `test_ros2_runtime_supports_three_concurrent_nodes` — add 3 MagicMock nodes, assert `executor.add_node` called 3 times on the same executor instance (reproduces Bug 1 regression guard).
    5. `test_ros2_runtime_shutdown_joins_thread` — `shutdown()` sets `is_running=False`, mocks `thread.join` called.
  - Mock `rclpy` / `rclpy.executors.MultiThreadedExecutor` via `sys.modules` injection (match existing `tests/hardware/sim/test_mujoco_piper.py` lazy-mock pattern).
  - **GREEN** — implement `Ros2Runtime` with `add_node`, `remove_node`, `shutdown`, `is_running`, `get_ros2_runtime()` module singleton + thread lock. Register `atexit` on first init.
  - **REFACTOR** — clean, docstrings, ruff clean.
- **Acceptance**: AC-2 (partial, Ros2Runtime tests)
- **Verify**:
  ```bash
  .venv-nano/bin/python -m pytest tests/hardware/ros2/test_runtime.py -v
  .venv-nano/bin/ruff check vector_os_nano/hardware/ros2/runtime.py
  ```

### T2 — compute_approach_pose util
- **Status**: [ ] pending
- **Agent**: Beta
- **Depends**: none
- **Files to create**:
  - `vector_os_nano/skills/utils/__init__.py` (create if missing, else empty)
  - `vector_os_nano/skills/utils/approach_pose.py` (~60 LoC)
  - `tests/skills/utils/__init__.py` (empty)
  - `tests/skills/utils/test_approach_pose.py` (~120 LoC, 6 tests)
- **Input**: `.sdd/spec.md` §7.6, `.sdd/plan.md` §3.2
- **TDD**:
  - **RED** — 6 tests:
    1. `test_approach_pose_dog_east_of_object_stops_east` — object at (0,0), dog at (2,0) → approach x in (0, 2), approach_x ≈ 0.55.
    2. `test_approach_pose_dog_north_of_object_stops_north` — object (0,0), dog (0,2) → approach (0, 0.55, yaw=-π/2).
    3. `test_approach_pose_yaw_faces_object` — from 4 cardinal dog positions, yaw points at object (±1e-6).
    4. `test_approach_pose_clearance_is_exact_distance` — ||approach − object|| == clearance.
    5. `test_approach_pose_degenerate_dog_equals_object_raises_value_error`.
    6. `test_approach_pose_custom_clearance_propagated` — clearance=0.3 and 1.0 produce matching distances.
  - `from_normal` NotImplementedError NOT tested in Wave 1 (contract says v2.3 reserved; leave it for a xfail later).
  - **GREEN** — implement pure math function. Single file. No imports beyond `math`.
  - **REFACTOR** — docstrings, type hints, ruff clean.
- **Acceptance**: AC-2 (partial)
- **Verify**:
  ```bash
  .venv-nano/bin/python -m pytest tests/skills/utils/test_approach_pose.py -v
  .venv-nano/bin/ruff check vector_os_nano/skills/utils/approach_pose.py
  ```

### T3 — Chinese color normaliser helper
- **Status**: [ ] pending
- **Agent**: Gamma
- **Depends**: none
- **Files to modify**:
  - `vector_os_nano/skills/pick_top_down.py` (add module-level `_CN_COLOR_MAP` + `_normalise_color_keyword` function; do NOT touch `_resolve_target` yet — that is T7)
  - `tests/skills/test_pick_top_down.py` (append 5 new tests; existing 13 must stay green)
- **Input**: `.sdd/spec.md` §11 Q2 (color alphabet), `.sdd/plan.md` §3.3
- **TDD**:
  - **RED** — 5 new tests:
    1. `test_normalise_color_keyword_chinese_suffix` — `"绿色"` → `"green"`, `"红色瓶子"` → `"red 瓶子"`.
    2. `test_normalise_color_keyword_single_char` — `"红"` → `"red"`.
    3. `test_normalise_color_keyword_returns_none_if_no_match` — `"bottle"` → None, `"紫色"` → None.
    4. `test_normalise_color_keyword_all_six_colors` — parametrized for red/green/blue/yellow/white/black.
    5. `test_normalise_color_keyword_mixed_input_with_color_substring` — `"抓前面绿色瓶子"` → `"抓前面green瓶子"`.
  - **GREEN** — add `_CN_COLOR_MAP` dict + `_normalise_color_keyword(label: str) -> str | None` at module level (above `PickTopDownSkill`).
  - **REFACTOR** — re-run full `test_pick_top_down.py` (old 13 + new 5) all green, ruff clean.
- **Acceptance**: AC-4 (partial — helper only; full resolver update is T7)
- **Verify**:
  ```bash
  .venv-nano/bin/python -m pytest tests/skills/test_pick_top_down.py -v
  .venv-nano/bin/ruff check vector_os_nano/skills/pick_top_down.py
  ```

### Wave 1 Gate
- All 3 tasks' tests green
- Ruff clean on all touched files
- Existing `test_pick_top_down.py` unchanged (13 regress tests still pass after T3)
- De-sloppify: no dead code, no debug prints, no TODO comments left behind

---

## Wave 2 — Integration (3 parallel, depends on Wave 1)

### T4 — Wire Ros2Runtime into 3 proxies
- **Status**: [ ] pending
- **Agent**: Alpha
- **Depends**: T1
- **Files to modify**:
  - `vector_os_nano/hardware/sim/go2_ros2_proxy.py` (connect + disconnect blocks)
  - `vector_os_nano/hardware/sim/piper_ros2_proxy.py` (`PiperROS2Proxy.connect/disconnect` + `PiperGripperROS2Proxy.connect/disconnect`)
- **Files to create**: none (integration test lands in T10)
- **Input**: `.sdd/plan.md` §3.7, §3.8
- **TDD**:
  - **RED** — since proxies require real rclpy to test directly, regression is enforced via Wave 3's T10 integration test. For this task, the "red" gate is that the existing `tests/hardware/sim/test_go2_ros2_proxy.py` (if any mocked tests exist) must still pass.
    - Additionally: write a targeted unit-mock test `test_proxies_use_shared_runtime_when_flag_on` in `tests/hardware/sim/test_ros2_proxies_runtime_wiring.py` — monkeypatch `os.environ["VECTOR_SHARED_EXECUTOR"]="1"`, mock `get_ros2_runtime`, assert `.add_node(node)` called on connect and `.remove_node(node)` on disconnect.
  - **GREEN** — replace spin-thread blocks per `plan.md` §3.7:
    ```python
    if os.environ.get("VECTOR_SHARED_EXECUTOR", "1") == "1":
        from vector_os_nano.hardware.ros2.runtime import get_ros2_runtime
        get_ros2_runtime().add_node(self._node)
    else:
        # legacy per-proxy spin
        self._spin_thread = threading.Thread(target=lambda: rclpy.spin(self._node), daemon=True)
        self._spin_thread.start()
    ```
    Apply to all 3 proxy `connect()` methods. Mirror in `disconnect()` with `remove_node`.
  - **REFACTOR** — ruff clean, remove stale comments.
- **Acceptance**: AC-3 via T10, no regression on existing proxy tests
- **Verify**:
  ```bash
  .venv-nano/bin/python -m pytest tests/hardware/sim/ -v -k "not mujoco"  # exclude slow mujoco tests
  .venv-nano/bin/ruff check vector_os_nano/hardware/sim/go2_ros2_proxy.py vector_os_nano/hardware/sim/piper_ros2_proxy.py
  ```

### T5 — PlaceTopDownSkill
- **Status**: [ ] pending
- **Agent**: Beta
- **Depends**: none (can start with Wave 1; grouped here for throughput balance)
- **Files to create**:
  - `vector_os_nano/skills/place_top_down.py` (~180 LoC)
  - `tests/skills/test_place_top_down.py` (~250 LoC, 8 tests)
- **Input**: `.sdd/spec.md` §7.8, `.sdd/plan.md` §3.4
- **TDD**:
  - **RED** — 8 tests with mocked arm + gripper + WorldModel:
    1. `test_place_top_down_happy_path_calls_open_after_descent` — assert call order: ik(pre) → ik(place) → move(pre) → move(place) → open() → move(pre_lift).
    2. `test_place_top_down_ik_unreachable_returns_ik_unreachable`.
    3. `test_place_top_down_move_failed_returns_move_failed`.
    4. `test_place_top_down_no_arm_returns_no_arm`.
    5. `test_place_top_down_no_gripper_returns_no_gripper`.
    6. `test_place_top_down_explicit_xyz_bypasses_receptacle` — given both, xyz wins.
    7. `test_place_top_down_receptacle_id_resolves_xyz_from_world_model`.
    8. `test_place_top_down_missing_target_returns_missing_target`.
  - Mock pattern follows `tests/skills/test_pick_top_down.py`.
  - **GREEN** — implement `PlaceTopDownSkill`:
    - @skill(aliases=["put","drop","放","放下","放到","put down"], direct=False)
    - preconditions=["gripper_holding_any"], effects={"gripper_state":"open","held_object":None}
    - `_resolve_target` — explicit xyz OR `receptacle_id` via `wm.get_object(id)` + `drop_height`
    - Motion: ik(pre) → ik(place) → move(pre) → move(place) → open() → move(pre_lift); no return home
  - **REFACTOR** — docstrings, ruff clean, coverage ≥ 80%.
- **Acceptance**: AC-2 (partial)
- **Verify**:
  ```bash
  .venv-nano/bin/python -m pytest tests/skills/test_place_top_down.py -v
  .venv-nano/bin/python -m pytest tests/skills/test_place_top_down.py --cov=vector_os_nano.skills.place_top_down --cov-report=term-missing
  .venv-nano/bin/ruff check vector_os_nano/skills/place_top_down.py
  ```

### T6 — VGG source-aware decomposer hint
- **Status**: [ ] pending
- **Agent**: Gamma
- **Depends**: none
- **Files to modify**:
  - `vector_os_nano/vcli/cognitive/goal_decomposer.py` (skill catalog stringifier + system prompt)
- **Files to create**:
  - `tests/vcli/cognitive/test_decomposer_source.py` (3 tests)
- **Input**: `.sdd/spec.md` §7.4, `.sdd/plan.md` §3.9
- **TDD**:
  - **RED** — 3 tests:
    1. `test_skill_catalog_tags_world_model_source` — mock a skill with params all `source: world_model.*`, build catalog, assert `"(source: world_model)"` appears in its description line.
    2. `test_skill_catalog_does_not_tag_when_any_perception_source` — mock a skill with one `source: perception.*` param, assert NO tag.
    3. `test_system_prompt_includes_no_detect_hint` — build prompt with at least one world_model-source skill present, assert prompt contains the hint string about skipping detect_*.
  - Use existing `GoalDecomposer` test patterns (check `tests/vcli/cognitive/test_goal_decomposer.py` for pattern — else use bare instantiation).
  - **GREEN**:
    - Helper `_skill_is_world_model_only(skill) -> bool` — returns True iff all declared params have `source` starting with `"world_model."` or `"static"` or `"explicit"`, AND at least one param uses `"world_model."`.
    - When stringifying skill in catalog, append ` (source: world_model)` when helper returns True.
    - Add to system prompt a sentence: `"If a skill's description ends with (source: world_model), do NOT prepend detect_* steps; the target object is already tracked in the world model."`
  - **REFACTOR** — ruff clean, existing decomposer tests untouched.
- **Acceptance**: AC-5
- **Verify**:
  ```bash
  .venv-nano/bin/python -m pytest tests/vcli/cognitive/test_decomposer_source.py -v
  .venv-nano/bin/python -m pytest tests/vcli/cognitive/ -v  # full decomposer suite green
  .venv-nano/bin/ruff check vector_os_nano/vcli/cognitive/goal_decomposer.py
  ```

### T7 — pick_top_down _resolve_target upgrade
- **Status**: [ ] pending
- **Agent**: Gamma (continues from T3)
- **Depends**: T3 (normaliser helper must exist)
- **Files to modify**:
  - `vector_os_nano/skills/pick_top_down.py` (`_resolve_target` method)
  - `tests/skills/test_pick_top_down.py` (append 4 new resolver tests)
- **Input**: `.sdd/plan.md` §3.3 (resolver upgrade algorithm)
- **TDD**:
  - **RED** — 4 new resolver tests (on top of T3's 5 helper tests):
    1. `test_resolve_target_matches_chinese_color_after_normalise` — world_model has `"green bottle"`, label=`"抓前面绿色"`, resolver returns green bottle.
    2. `test_resolve_target_single_pickable_fallback_when_label_unmatched` — 1 pickable object, label=`"紫色"`, resolver returns that object (logs fallback).
    3. `test_resolve_target_no_fallback_when_multiple_pickables_and_unmatched_label` — 3 pickables, label=`"紫色"`, resolver returns None.
    4. `test_resolve_target_prefers_explicit_label_match_over_color_normalise` — label=`"green bottle"` directly matches, normaliser path not invoked.
  - **GREEN** — extend `_resolve_target` per `plan.md` §3.3:
    - After existing obj_id + label passes: if label present, try `_normalise_color_keyword(label)` → if non-None, re-query `wm.get_objects_by_label(normalised)`.
    - Last-resort single-pickable fallback: if exactly one object has `object_id.startswith("pickable_")` AND a label/id was requested, return it.
  - **REFACTOR** — run full test suite (13 old + 5 T3 + 4 new = 22), ruff clean.
- **Acceptance**: AC-1 (no regress), AC-4 (full Bug 2 fix)
- **Verify**:
  ```bash
  .venv-nano/bin/python -m pytest tests/skills/test_pick_top_down.py -v
  .venv-nano/bin/ruff check vector_os_nano/skills/pick_top_down.py
  ```

### Wave 2 Gate
- T4 / T5 / T6 / T7 all green
- `pytest tests/skills/ tests/vcli/cognitive/ tests/hardware/ros2/ tests/hardware/sim/` passes (no new red, no regress)
- Coverage on new modules (place_top_down) ≥ 80%
- Ruff clean across all touched files

---

## Wave 3 — Composition + Integration (3 parallel, depends on Wave 2)

### T8 — MobilePickSkill
- **Status**: [ ] pending
- **Agent**: Alpha
- **Depends**: T2 (approach_pose), T7 (pick resolver). T4 not strictly required (uses base.navigate_to, not Ros2Runtime directly) but recommended so bench tests reflect final runtime.
- **Files to create**:
  - `vector_os_nano/skills/mobile_pick.py` (~150 LoC)
  - `tests/skills/test_mobile_pick.py` (~230 LoC, 8 tests)
- **Input**: `.sdd/spec.md` §7.7, `.sdd/plan.md` §3.5
- **TDD**:
  - **RED** — 8 tests with mocked base + arm + gripper + WorldModel:
    1. `test_mobile_pick_already_reachable_skips_navigate` — dog spawn within 0.10 m of approach pose → `base.navigate_to` NOT called.
    2. `test_mobile_pick_calls_navigate_then_wait_stable_then_pick_in_order` — assert call order via side_effect list.
    3. `test_mobile_pick_nav_failed_returns_nav_failed`.
    4. `test_mobile_pick_wait_stable_timeout_returns_wait_stable_timeout` — mock `base.get_position` to oscillate forever, assert no pick attempted.
    5. `test_mobile_pick_propagates_pick_ik_unreachable_failure`.
    6. `test_mobile_pick_object_not_found_returns_object_not_found`.
    7. `test_mobile_pick_no_base_returns_no_base`.
    8. `test_mobile_pick_skip_navigate_param_honoured` — skip_navigate=True, even if far away, skip nav.
  - Mock `time.sleep` via monkeypatch to zero for fast tests.
  - **GREEN** — implement per `plan.md` §3.5:
    - `self._pick = PickTopDownSkill()` composition
    - Resolve target via `self._pick._resolve_target(params, wm)`
    - compute_approach_pose → reachability check → navigate_to → _wait_stable inline → delegate pick
    - Inline `_wait_stable(base, max_speed=0.05, settle_duration=1.0, timeout=5.0)` — position-delta based, 5 Hz poll
  - **REFACTOR** — coverage ≥ 80%, ruff clean.
- **Acceptance**: AC-2 (partial), AC-9 (partial)
- **Verify**:
  ```bash
  .venv-nano/bin/python -m pytest tests/skills/test_mobile_pick.py -v
  .venv-nano/bin/python -m pytest tests/skills/test_mobile_pick.py --cov=vector_os_nano.skills.mobile_pick --cov-report=term-missing
  .venv-nano/bin/ruff check vector_os_nano/skills/mobile_pick.py
  ```

### T9 — MobilePlaceSkill
- **Status**: [ ] pending
- **Agent**: Beta
- **Depends**: T2 (approach_pose), T5 (place_top_down)
- **Files to create**:
  - `vector_os_nano/skills/mobile_place.py` (~140 LoC)
  - `tests/skills/test_mobile_place.py` (~210 LoC, 7 tests)
- **Input**: `.sdd/spec.md` §7.9, `.sdd/plan.md` §3.6
- **TDD**:
  - **RED** — 7 tests (parallel to T8):
    1. `test_mobile_place_already_reachable_skips_navigate`.
    2. `test_mobile_place_calls_navigate_then_wait_then_place_in_order`.
    3. `test_mobile_place_nav_failed_returns_nav_failed`.
    4. `test_mobile_place_wait_stable_timeout_aborts_before_place`.
    5. `test_mobile_place_propagates_place_failure`.
    6. `test_mobile_place_no_base_returns_no_base`.
    7. `test_mobile_place_explicit_target_xyz_vs_receptacle_id` — explicit wins.
  - **GREEN** — mirror MobilePickSkill with `self._place = PlaceTopDownSkill()`.
    - Target resolution: explicit `target_xyz` OR `receptacle_id` → query WorldModel → `(obj.x, obj.y, obj.z + drop_height_default)`.
  - Prefer EXTRACT `_wait_stable` and approach-reachability fast-path to a private module helper shared by mobile_pick and mobile_place to avoid duplication — acceptable to create `vector_os_nano/skills/utils/mobile_helpers.py` with `_wait_stable(base, ...)` and `_already_reachable(dog_pose, approach, tol_xy, tol_yaw)`. T8 and T9 both import from it; T8 may land first and T9 extracts in its own task.
  - **REFACTOR** — coverage ≥ 80%, ruff clean.
- **Acceptance**: AC-2 (partial), AC-9 (partial)
- **Verify**:
  ```bash
  .venv-nano/bin/python -m pytest tests/skills/test_mobile_place.py -v
  .venv-nano/bin/python -m pytest tests/skills/test_mobile_place.py --cov=vector_os_nano.skills.mobile_place --cov-report=term-missing
  .venv-nano/bin/ruff check vector_os_nano/skills/mobile_place.py
  ```

### T10 — ROS2 proxies coexist integration test
- **Status**: [ ] pending
- **Agent**: Gamma
- **Depends**: T1, T4
- **Files to create**:
  - `tests/hardware/sim/test_ros2_proxies_coexist.py` (~120 LoC, 3 tests, @pytest.mark.ros2)
- **Input**: `.sdd/spec.md` §8.2, `.sdd/plan.md` §5.2
- **TDD**:
  - **RED** — 3 tests using REAL rclpy (not mocks) — skip if rclpy unavailable:
    1. `test_three_stub_nodes_spin_concurrently_no_already_spinning` — create 3 stub `rclpy.node.Node` objects via `Ros2Runtime.add_node`, publish a dummy message on each, subscriber on each receives within 1 s. No `RuntimeError: Executor is already spinning` in captured logs.
    2. `test_runtime_shutdown_tears_down_cleanly` — add nodes, call `shutdown`, assert no pending threads, `rclpy.ok()` is False.
    3. `test_runtime_reentrant_add_after_shutdown_reinits` — edge case; optional if time-boxed, else xfail.
  - Decorate file with `pytest.importorskip("rclpy")`.
  - **GREEN** — pure test-only task; no production code changes expected (Ros2Runtime from T1 + proxy wiring from T4 should suffice).
  - If an implementation gap is found → escalate to Alpha for T1/T4 hotfix; do NOT patch in T10.
  - **REFACTOR** — ruff clean on new test file.
- **Acceptance**: AC-3 (full Bug 1 regression guard)
- **Verify**:
  ```bash
  .venv-nano/bin/python -m pytest tests/hardware/sim/test_ros2_proxies_coexist.py -v -m ros2
  ```

### Wave 3 Gate
- T8 / T9 / T10 all green
- Full `pytest tests/` suite green (3200+ tests)
- Coverage on mobile_pick/mobile_place/place_top_down ≥ 80%
- Ruff clean
- De-sloppify pass: remove any debug prints, stale TODOs added during Wave 2-3

---

## Wave 4 — Wiring + E2E + Docs (3 sequential then parallel)

### T11 — sim_tool registration wiring
- **Status**: [ ] pending
- **Agent**: Alpha
- **Depends**: T5, T8, T9
- **Files to modify**:
  - `vector_os_nano/vcli/tools/sim_tool.py` (register 3 new skills when `with_arm=True`)
- **Files to create**: none
- **Input**: `.sdd/plan.md` §3.10
- **TDD**:
  - **RED** — augment `tests/vcli/tools/test_sim_tool.py` (or create if absent) with 1 test:
    - `test_start_go2_with_arm_registers_place_mobile_pick_mobile_place` — monkeypatch subprocess launch + proxy connect, assert `registry.register` called with each of: PickTopDownSkill, PlaceTopDownSkill, MobilePickSkill, MobilePlaceSkill instances.
  - **GREEN** — add 3 import + register lines after existing PickTopDownSkill block (per `plan.md` §3.10).
  - **REFACTOR** — ruff clean.
- **Acceptance**: AC-2, AC-8 (enables Yusen live REPL)
- **Verify**:
  ```bash
  .venv-nano/bin/python -m pytest tests/vcli/tools/test_sim_tool.py -v
  .venv-nano/bin/ruff check vector_os_nano/vcli/tools/sim_tool.py
  ```

### T12 — E2E verify_loco_pick_place.py
- **Status**: [ ] pending
- **Agent**: Beta
- **Depends**: T11
- **Files to create**:
  - `scripts/verify_loco_pick_place.py` (~300 LoC, modelled on `scripts/verify_pick_top_down.py`)
- **Input**: `.sdd/spec.md` §8.9, `.sdd/plan.md` §5.3
- **Deliverables**:
  - CLI: `--repeat N`, `--mode {pick_only, pick_and_place}`, `--objects blue,green,red`
  - Each attempt: fresh subprocess spawns bridge with VECTOR_SIM_WITH_ARM=1, main loads proxies, runs skill, asserts result_data, tears down
  - Pass criteria (per spec §8.9):
    - pick_only: `grasped_heuristic=True` after mobile_pick
    - pick_and_place: bottle final z within 3 cm of target_z, gripper open
  - Summary stats at end: `N/N passed, avg latency Xs`
  - Exit code 0 on all pass, 1 on any fail
- **TDD notes**: E2E harness is manually run; include a `--dry-run` flag that exits 0 without spawning subprocess for CI smoke.
- **Acceptance**: AC-6, AC-7
- **Verify**:
  ```bash
  .venv-nano/bin/python scripts/verify_loco_pick_place.py --dry-run
  # Full run (~3 min, requires working sim):
  .venv-nano/bin/python scripts/verify_loco_pick_place.py --repeat 3 --mode pick_only
  .venv-nano/bin/python scripts/verify_loco_pick_place.py --repeat 3 --mode pick_and_place
  ```

### T13 — Live REPL checklist doc
- **Status**: [ ] pending
- **Agent**: Gamma (can run parallel to T11/T12 — doc only)
- **Depends**: none (doc is independent)
- **Files to create**:
  - `docs/v2.2_live_repl_checklist.md` (~80 lines)
- **Input**: `.sdd/spec.md` §8.10
- **Deliverables**: numbered human checklist Yusen runs in `vector-cli`:
  1. `go2sim with_arm=1` → rviz opens, no executor errors in stderr
  2. `抓前面绿色` → success, `is_holding=True`
  3. `放到 (11.0, 3.2, 0.25)` → success, gripper opens
  4. `去拿红色罐头` → mobile_pick runs end-to-end (navigate + wait + pick)
  5. Each step: expected stdout snippets, common failure modes, fallback commands.
- **Acceptance**: AC-8 (Yusen acceptance criterion)
- **Verify**: doc exists + rendered correctly:
  ```bash
  cat docs/v2.2_live_repl_checklist.md | head -50
  ```

### Wave 4 Gate
- T11 unit test green
- T12 `--dry-run` green (full E2E validated in Phase 5 QA)
- T13 doc exists, accurate, reviewed by Scribe
- De-sloppify pass on whole diff (ensure no uncommitted debug logs)

---

## Dependency Graph

```
Wave 1 (parallel):
    T1 (Alpha)  ─┐
    T2 (Beta)   ─┼──> Wave 1 gate
    T3 (Gamma)  ─┘

Wave 2 (parallel, Wave 1 satisfied):
    T4 (Alpha, needs T1) ─┐
    T5 (Beta, free)       ─┼──> Wave 2 gate
    T6 (Gamma, free)      ─┤
    T7 (Gamma, needs T3)  ─┘      # Gamma runs T6 or T7 first, then the other; OR Gamma runs one, Beta picks up the other after T5

Wave 3 (parallel, Wave 2 satisfied):
    T8 (Alpha, needs T2+T7) ─┐
    T9 (Beta, needs T2+T5)  ─┼──> Wave 3 gate
    T10 (Gamma, needs T1+T4)─┘

Wave 4:
    T11 (Alpha, needs T5+T8+T9) ──> triggers T12
    T12 (Beta, needs T11)       ──┐
    T13 (Gamma, free — doc)     ──┴──> Wave 4 gate
```

Gamma in Wave 2 gets two tasks (T6 + T7) since T7 must wait for T3 (same agent) — they chain naturally. Alternatively Beta picks up T7 after T5 if parallelism is key; either way the wave completes together.

---

## Execution Waves Summary

| Wave | Tasks | Agents | Dur (parallel) | Gate |
|------|-------|--------|----------------|------|
| 1 | T1, T2, T3 | Alpha, Beta, Gamma | ~1.5 hr | unit tests + ruff |
| 2 | T4, T5, T6, T7 | Alpha, Beta, Gamma (Gamma chains T6→T7) | ~1.5 hr | unit+integration tests, no regress |
| 3 | T8, T9, T10 | Alpha, Beta, Gamma | ~2 hr | full pytest + coverage ≥ 80% |
| 4 | T11, T12, T13 | Alpha, Beta, Gamma | ~1 hr | E2E dry-run + doc review |
| — | QA + release | code-reviewer + security-reviewer | — | Phase 5 |

Total parallel wall-clock: ~6 hr including de-sloppify and gate checks.

---

## Task Totals

- 13 tasks
- ~42 new unit tests + 3 integration tests
- 10 new files + 5 modified
- Coverage target: ≥ 80% on 5 new modules (Ros2Runtime, approach_pose, place_top_down, mobile_pick, mobile_place)

---

## Notes for sdd-execute / subagent runner

- Each task handed to a fresh agent via `Agent({ subagent_type: "vr-alpha" | "vr-beta" | "vr-gamma", prompt: <task-specific context> })`.
- Subagent prompt MUST include: task ID, files to touch, spec/plan section references, TDD order (RED first), verify command, done-definition.
- De-sloppify skill runs after each wave (before gate).
- Gate failure → re-spawn the failing agent with failure context; 2 retries max before escalating to vr-lead.
- progress.md + agents/devlog/status.md updated by vr-scribe after each wave gate.
