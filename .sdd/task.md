# v2.4 SysNav Simulation Integration — Task Breakdown

**Status**: APPROVED async (CEO auto-approval 2026-04-25)
**Prereq**: spec.md + plan.md
**Agents**: Alpha / Beta / Gamma serial-dispatch (post-OOM rule:
forbid full pytest in subagents; narrow `pytest tests/unit/<file>.py`
only; forbid imports of `pipeline`, `track_anything`, `mujoco`-with-
go2-room.xml, `realsense`, `tracker`, `ultralytics`).

---

## Execution summary

| Metric | Value |
|---|---|
| Code tasks | 8 (T0–T8) |
| Waves | 6 |
| New files | 9 impl + 9 test files + 1 launch + 1 smoke + 1 doc |
| Modified files | 4 |
| Unit tests target | ≥ 50 new |
| Integration tests target | ≥ 5 new |
| Coverage floor | 90 % on new modules |
| Estimated wall-clock | 3–4 days serial |

Test-first culture is non-negotiable for this cycle. Each task lists
RED test names that **must be authored and verified failing before**
any implementation starts.

---

## Subagent prompt template

```
[context]
Task {ID} of v2.4 SysNav Simulation Integration.
Spec: .sdd/spec.md §{relevant}
Plan: .sdd/plan.md §{relevant}
Depends on: {tasks landed at SHAs}

[safety]
- DO NOT run `pytest tests/` or `pytest tests/integration/`.
- Run only the narrow file(s) listed under "Verify".
- DO NOT import: pipeline, track_anything, realsense, tracker,
  ultralytics. mujoco is allowed only via the inline tiny-MJCF
  fixture in tests/unit/hardware/sim/sensors/conftest.py.
- Stick to the files listed in "Files".
- TDD strict: write RED tests first, confirm failure, then GREEN,
  then REFACTOR. Do not skip RED.

[deliverable]
{files list}
RED tests: {names}
Verify: {commands}

[completion]
- All listed tests pass narrow-scope.
- ruff check clean on modified files.
- Coverage ≥ 90 % on the module via:
    pytest <test_file> --cov=<module> --cov-report=term-missing
- Commit `[alpha|beta|gamma] {type}(v2.4): <desc>`.
- Report SHA + test pass count + coverage % to dispatcher.
```

---

## Wave 0 — Environment Probe

### T0 — Probe MuJoCo + OpenCV versions; mj_ray smoke

**Agent**: Dispatcher (no subagent)
**Output**: `agents/devlog/v24-sysnav-env-probe.md`

Steps:
1. `python -c "import mujoco; print(mujoco.__version__)"` — record.
2. `python -c "import cv2; print(cv2.__version__)"` — record.
3. mj_ray smoke: 1024 rays against a single-body MJCF → wall time.
   Target: ≤ 5 ms; if > 20 ms, R1 mitigation (`mj_multiRay`) is required.
4. Cube-face render smoke: `mujoco.Renderer(model, 480, 480)` × 6 faces
   → wall time. Target: ≤ 80 ms total on RTX 5080; document VRAM use.
5. Confirm SysNav workspace presence at `~/Desktop/SysNav` and topic
   types `tare_planner.msg.ObjectNodeList` build status.

**Gate**: probe report posted; no abort unless mj_ray > 50 ms (then
escalate).

---

## Wave 1 — Foundational sensors (3 parallel tasks, **SERIAL** dispatch)

### T1 — `MuJoCoLivox360` virtual lidar

**Agent**: Alpha
**Wave**: 1
**Depends**: T0
**Package**: `vector_os_nano/hardware/sim/sensors/`

**Files**:
- NEW `vector_os_nano/hardware/sim/sensors/__init__.py`
- NEW `vector_os_nano/hardware/sim/sensors/lidar360.py` (~180 LoC)
- NEW `tests/unit/hardware/sim/sensors/__init__.py`
- NEW `tests/unit/hardware/sim/sensors/conftest.py` (tiny MJCF fixture)
- NEW `tests/unit/hardware/sim/sensors/test_lidar360.py`

**RED tests (write FIRST, confirm failure)**:

1. `test_ray_dirs_polar_grid_shape` — h=360, v=16 → (5760, 3) unit vectors.
2. `test_ray_dirs_azimuth_endpoints_excluded` — last azimuth ≠ first.
3. `test_ray_dirs_elevation_range_within_minus7_to_52_deg` — Mid-360 spec.
4. `test_step_returns_empty_when_no_geom` — empty MJCF → 0 hits.
5. `test_step_against_known_wall_returns_correct_xyz` — wall at x=3, ray azimuth=0 → hit `(3.0, 0.0, body_z)` ± 5 cm.
6. `test_step_clamps_at_max_range` — max_range=2.0, wall at x=3 → no hit recorded.
7. `test_intensity_field_defaults_to_one_point_zero` — every hit `intensity == 1.0`.
8. `test_rate_limit_returns_cached_when_called_too_fast` — two `step()` calls within 1/rate_hz return same array.
9. `test_to_pointcloud2_field_layout_x_y_z_intensity_float32` — offsets 0,4,8,12; point_step 16; row_step matches.
10. `test_to_pointcloud2_dense_and_little_endian` — `is_dense == True`, `is_bigendian == False`.
11. `test_to_pointcloud2_frame_id_is_map` — header.frame_id="map".
12. `test_step_in_world_frame_after_body_translated` — translate body to (5,0,0); same ray to wall at (3,0,0) does NOT hit (target now behind).

**GREEN**: implement per plan §3.2 / §6.1 / §6.3.

**Verify**:
```
.venv-nano/bin/python -m pytest tests/unit/hardware/sim/sensors/test_lidar360.py -v
.venv-nano/bin/python -m pytest tests/unit/hardware/sim/sensors/test_lidar360.py \
  --cov=vector_os_nano.hardware.sim.sensors.lidar360 --cov-fail-under=90
```

### T2 — `GroundTruthOdomPublisher`

**Agent**: Beta
**Wave**: 1
**Depends**: T0
**Package**: `vector_os_nano/hardware/sim/sensors/`

**Files**:
- NEW `vector_os_nano/hardware/sim/sensors/gt_odom.py` (~120 LoC)
- NEW `tests/unit/hardware/sim/sensors/test_gt_odom.py`

**RED tests**:

1. `test_position_matches_body_xpos` — body at (1, 2, 0.5) → odom.pose.position == (1, 2, 0.5).
2. `test_orientation_quaternion_normalised` — set xquat to non-unit; output magnitude 1.
3. `test_first_call_twist_is_zero` — no prior state → linear/angular all zero.
4. `test_twist_is_finite_difference_after_translation` — body at (0,0,0) then (0.1, 0, 0) after dt=0.1 → vx ≈ 1.0 ± 0.05.
5. `test_frame_id_default_map` and `test_child_frame_id_default_sensor`.
6. `test_rate_limit_step_returns_cached_msg` — two `step()` within 1/rate_hz identical.
7. `test_dt_clamp_against_zero_division` — back-to-back calls at same monotonic time → no ZeroDivisionError, twist zero.
8. `test_orientation_unchanged_returns_zero_angular_twist` — body translates without rotation → angular = (0,0,0).

**GREEN**: implement per plan §3.4.

**Verify**:
```
.venv-nano/bin/python -m pytest tests/unit/hardware/sim/sensors/test_gt_odom.py -v --cov=vector_os_nano.hardware.sim.sensors.gt_odom --cov-fail-under=90
```

### T3 — G3 xmat REP-103 fix + regression test

**Agent**: Gamma
**Wave**: 1
**Depends**: — (parallel-safe with T1/T2 since file is independent)

**Files**:
- MOD `vector_os_nano/hardware/sim/go2_ros2_proxy.py` (lines 337–339)
- MOD `tests/integration/test_go2_camera_pose.py` (update expectations)
- NEW `tests/integration/test_xmat_rep103_regression.py` (additional)

**RED tests** (in regression file):

1. `test_right_at_heading_zero_is_minus_y` — `right == (0, -1, 0)`.
2. `test_up_at_heading_zero_is_plus_z` — `up == (0, 0, 1)`.
3. `test_right_at_heading_pi_over_2_is_plus_x` — heading=π/2 → `right == (1, 0, 0)` ± 1e-9.
4. `test_xmat_columns_orthonormal` — det ≈ 1, every column unit-length.
5. `test_existing_camera_pose_test_still_passes_with_new_values` — re-run the v2.3 expected fixture with G3-corrected expectations.

**GREEN**: change two lines in `go2_ros2_proxy.py:337-339`:
```python
right = np.array([sin_h, -cos_h, 0.0])
up = np.cross(right, fwd)
```

Update the existing `test_go2_camera_pose.py` expected values to
match REP-103.

**Verify**:
```
.venv-nano/bin/python -m pytest tests/integration/test_go2_camera_pose.py tests/integration/test_xmat_rep103_regression.py -v
```

Wave 1 gate runs all three:
```
.venv-nano/bin/python -m pytest tests/unit/hardware/sim/sensors/ tests/integration/test_go2_camera_pose.py tests/integration/test_xmat_rep103_regression.py
```

---

## Wave 2 — Pano camera + ROS subscriber

### T4 — `MuJoCoPano360` virtual 360-degree RGBD

**Agent**: Alpha
**Wave**: 2
**Depends**: T0, T1 (renderer pattern reused)
**Package**: `vector_os_nano/hardware/sim/sensors/`

**Files**:
- NEW `vector_os_nano/hardware/sim/sensors/pano360.py` (~250 LoC)
- NEW `tests/unit/hardware/sim/sensors/test_pano360.py`

**RED tests**:

1. `test_lut_shape_matches_output_resolution` — out_w=1920, out_h=640 → LUT (640, 1920) for face_idx, fx, fy.
2. `test_lut_face_indices_in_range_zero_to_five` — every entry ∈ [0, 5].
3. `test_lut_central_pixel_maps_to_front_face_centre` — (out_w/2, out_h/2) → face=front, fx≈face_size/2, fy≈face_size/2.
4. `test_lut_left_quarter_maps_to_left_face` — column 0 → face=left.
5. `test_step_returns_rgb_uint8_with_correct_shape` — output (out_h, out_w, 3) uint8.
6. `test_step_returns_depth_float32_with_correct_shape` — output (out_h, out_w) float32.
7. `test_step_uniform_red_world_returns_red_pano` — synthetic uniform red sphere → output >95 % red.
8. `test_depth_clipped_at_max_range` — far wall, max_range=5 → depth values capped at 5.
9. `test_rate_limit_returns_cached_when_called_too_fast` — two step calls within 1/rate_hz identical.
10. `test_image_pano_aspect_ratio_3_to_1` — width / height == 3.0 ± 0.01.

**GREEN**: implement per plan §3.3 / §6.2.

**Verify**:
```
.venv-nano/bin/python -m pytest tests/unit/hardware/sim/sensors/test_pano360.py -v --cov=vector_os_nano.hardware.sim.sensors.pano360 --cov-fail-under=90
```

### T5 — `LiveSysnavBridge` rclpy subscriber

**Agent**: Beta
**Wave**: 2
**Depends**: T0
**Package**: `vector_os_nano/integrations/sysnav_bridge/`

**Files**:
- NEW `vector_os_nano/integrations/sysnav_bridge/live_bridge.py` (~200 LoC)
- MOD `vector_os_nano/integrations/sysnav_bridge/__init__.py` (re-export)
- NEW `tests/unit/integrations/sysnav_bridge/__init__.py`
- NEW `tests/unit/integrations/sysnav_bridge/conftest.py`
- NEW `tests/unit/integrations/sysnav_bridge/test_live_bridge.py`

**RED tests**:

1. `test_start_returns_false_when_rclpy_missing` — patch import → False, no exception.
2. `test_start_returns_false_when_tare_planner_msg_missing` — same pattern → False.
3. `test_start_returns_true_with_stubs` — both stubs available → True; `_active` flag set.
4. `test_start_creates_subscription_with_correct_topic_and_type` — assert subscription args.
5. `test_callback_dispatches_one_add_object_per_node` — 3 nodes → 3 `add_object` calls.
6. `test_callback_uses_existing_object_node_to_state` — passes `prior` from `world_model.get_object`.
7. `test_callback_logs_warning_on_malformed_node_no_crash` — node missing position → WARN log, processing continues for next.
8. `test_status_false_node_maps_to_unknown_state` — uses object_node_to_state already.
9. `test_stop_is_idempotent` — `stop(); stop()` → no errors, single shutdown.
10. `test_disconnect_warning_fires_after_threshold` — no callback for `on_disconnect_after_s` → WARN logged once.
11. `test_disconnect_warning_resets_after_message` — message arrives → next disconnect rearms.
12. `test_world_model_observed_through_repeated_callbacks` — same sysnav_id called twice with different positions → `add_object` upserts.

**GREEN**: implement per plan §3.5.

**Verify**:
```
.venv-nano/bin/python -m pytest tests/unit/integrations/sysnav_bridge/test_live_bridge.py -v --cov=vector_os_nano.integrations.sysnav_bridge.live_bridge --cov-fail-under=90
```

Wave 2 gate:
```
.venv-nano/bin/python -m pytest tests/unit/hardware/sim/sensors/ tests/unit/integrations/sysnav_bridge/
```

---

## Wave 3 — Bridge wiring + CLI tool

### T6 — `go2_vnav_bridge.py` wiring + integration tests

**Agent**: Alpha
**Wave**: 3
**Depends**: T1, T2, T4

**Files**:
- MOD `scripts/go2_vnav_bridge.py` (~100 LoC of new wiring)
- NEW `tests/integration/test_lidar360_against_world.py`
- NEW `tests/integration/test_pano360_against_world.py`
- NEW `tests/integration/test_gt_odom_against_walk.py`

**RED tests**:

1. `test_lidar_publishes_at_least_one_point_against_room` — load `go2_room.xml` (this integration may import mujoco directly; not subagent-only), tick lidar → PointCloud2 has ≥ 1000 points within trunk radius.
2. `test_lidar_hits_known_wall_position_within_tolerance` — known wall at world (10, 0, 1) → at least 1 lidar return within 5 cm.
3. `test_pano_image_shape_after_render` — full pipeline tick → image (640, 1920, 3) uint8.
4. `test_pano_depth_correlates_with_distance_to_wall` — wall at known distance → depth pixel ≈ that distance.
5. `test_gt_odom_after_simulated_walk` — drive Go2 forward via `set_velocity` for 1 s → odom.pose.position.x ≈ 0.5 m.

**GREEN**: implement publisher wiring per plan §3.7.

**Verify**:
```
.venv-nano/bin/python -m pytest tests/integration/test_lidar360_against_world.py tests/integration/test_pano360_against_world.py tests/integration/test_gt_odom_against_walk.py -v
```

### T7 — `sysnav_sim_tool` CLI + tests

**Agent**: Beta
**Wave**: 3
**Depends**: T5, T6

**Files**:
- NEW `vector_os_nano/vcli/tools/sysnav_sim_tool.py` (~150 LoC)
- MOD `vector_os_nano/vcli/tools/__init__.py` (register tool)
- NEW `tests/unit/vcli/test_sysnav_sim_tool.py`
- NEW `tests/integration/test_sysnav_sim_smoke.py`

**RED tests** (unit):

1. `test_preflight_when_tare_planner_msg_present` — stub import → returns True.
2. `test_preflight_when_tare_planner_msg_absent` — stub ImportError → returns False, WARN logged.
3. `test_run_starts_sim_then_bridge` — mocks `SimStartTool` and `LiveSysnavBridge`; assert order.
4. `test_run_continues_when_bridge_start_fails` — bridge.start() returns False → tool still completes, agent.perception stays None.
5. `test_stop_cleans_bridge_then_sim` — assert teardown order reverse of run.
6. `test_double_run_idempotent` — calling run twice does not double-construct.
7. `test_help_text_lists_required_topics` — tool.description mentions /object_nodes_list and /target_object_instruction.
8. `test_tool_registered_in_init` — importing `vector_os_nano.vcli.tools` exposes `SysnavSimTool`.

**Integration test** (`test_sysnav_sim_smoke.py`):

1. `test_mock_object_nodes_list_publisher_to_world_model` — boots a tiny rclpy node that publishes 3 stub `ObjectNodeList` messages → after 2 s, `world_model.get_objects()` has ≥ 3 sysnav_-prefixed entries with correct labels.

**GREEN**: implement per plan §3.6.

**Verify**:
```
.venv-nano/bin/python -m pytest tests/unit/vcli/test_sysnav_sim_tool.py tests/integration/test_sysnav_sim_smoke.py -v
```

Wave 3 gate:
```
.venv-nano/bin/python -m pytest tests/unit tests/integration -k "sysnav or lidar360 or pano360 or gt_odom or xmat or pick_top_down or mobile_pick"
```

(Filter avoids accidentally pulling in unrelated heavy MuJoCo paths.)

---

## Wave 4 — Smoke + docs

### T8 — `smoke_sysnav_sim.py` + `docs/sysnav_simulation.md`

**Agent**: Gamma
**Wave**: 4
**Depends**: all prior

**Files**:
- NEW `scripts/smoke_sysnav_sim.py`
- NEW `docs/sysnav_simulation.md`
- MOD `progress.md` (v2.4 final section)
- MOD `agents/devlog/status.md` (v2.4 final state)

`smoke_sysnav_sim.py` contract:

- `--check-deps` mode: imports rclpy + tare_planner.msg, exits 0/1.
- `--no-sysnav` mode: starts sim only, asserts 4 topics publishing
  within 5 s.
- Default mode: starts sim, asserts SysNav workspace running, asserts
  `/object_nodes_list` carries ≥ 1 node within 30 s, asserts
  `world_model` populated.

`docs/sysnav_simulation.md` covers:

- Bringup order (3 terminals).
- Topic matrix (input/output).
- Performance targets (mj_ray latency, pano FPS).
- Troubleshooting (no /object_nodes_list, GPU OOM, frame mismatch).
- Cross-reference to `docs/sysnav_integration.md` (real-robot + license
  boundary).

**Verify**:
```
.venv-nano/bin/python scripts/smoke_sysnav_sim.py --check-deps
```

---

## Wave 5 — QA (parallel subagents)

### code-reviewer

Focus:
- Lidar pose math (body→world quaternion correctness).
- Pano LUT precompute (off-by-one / boundary cases).
- LiveSysnavBridge resource cleanup (`stop()` must join thread, kill
  rclpy node).
- Rate-limit thread-safety.
- New CLI tool concurrency (no race vs. SimStartTool).

### security-reviewer

Focus:
- New rclpy subscriber: malformed PointCloud2/Image payloads must
  not crash the process.
- LiveSysnavBridge logging — ensure nothing logs PII / secret env vars.
- `sysnav_sim_tool` does not allow argument injection through user
  input (CLI args validation).
- Topic queue depths: avoid unbounded buffering of large PointCloud2.

Gate: 0 CRITICAL / 0 unaddressed HIGH. MAJOR/LOW deferred with explicit
note in `qa_status` history.

---

## Wave 6 — CEO smoke

Yusen runs (with SysNav workspace started in another terminal):

```
vector-cli sysnav-sim
> 抓起蓝色瓶子
```

Pass: bottle picked, no phantom navigation, world_model entries match
ground truth within 0.1 m.

On pass: tag `v2.4.0-rc1`, push to remote.

---

## Dispatch order (dispatcher script)

```
W0  T0   dispatcher
W1  T1   Alpha   →   T2  Beta   →   T3  Gamma
W1  gate: pytest tests/unit/hardware/sim/sensors tests/integration/test_go2_camera_pose.py tests/integration/test_xmat_rep103_regression.py
W2  T4   Alpha   →   T5  Beta
W2  gate: pytest tests/unit/hardware/sim/sensors tests/unit/integrations/sysnav_bridge
W3  T6   Alpha   →   T7  Beta
W3  gate: pytest -k "sysnav or lidar360 or pano360 or gt_odom or xmat or pick_top_down or mobile_pick"
W4  T8   Gamma
W5  QA   code-reviewer + security-reviewer (parallel subagents)
W6  CEO  live-REPL smoke + tag
```

Coverage gate after each wave: `--cov-fail-under=90` on the new
modules touched in that wave.

---

## Test-first discipline reminders

1. RED before GREEN. Each task lists exact RED test names.
2. Coverage ≥ 90 % per module — measured per wave gate.
3. Dispatcher runs the wave gate, NOT the subagent (avoid full pytest
   in subagent prompts per `feedback_no_parallel_agents.md`).
4. New integration tests use `go2_room.xml` only when the test name
   ends with `_against_world` or `_against_walk` — keep unit tests on
   inline tiny MJCFs.
5. `LiveSysnavBridge` and `SysnavSimTool` tests do NOT spawn real
   rclpy nodes; they patch import sites.
