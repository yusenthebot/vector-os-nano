# v2.4 SysNav Simulation Integration — Specification

**Status**: APPROVED (CEO auto-approval 2026-04-25)
**Scope**: MuJoCo-sim only. Real-robot bringup deferred to v2.5+.
**Branch**: `feat/v2.0-vectorengine-unification` (continuation;
v2.4-perception-overhaul archived)

---

## 1. Overview

Re-route Vector OS Nano's grounding + scene-graph layer to the
**SysNav** sibling project (CMU Robotics Institute, PolyForm-NC) by
making MuJoCo publish the ROS2 topics SysNav already knows how to
consume — `/registered_scan` (Livox PointCloud2),
`/state_estimation` (Odometry), `/camera/image` and `/camera/depth`
(360-degree equirectangular). Three new MuJoCo virtual sensors plus
one launch harness will let SysNav's `semantic_mapping_node` +
`detection_node` + `vlm_reasoning_node` run unchanged on top of our
sim. Their `/object_nodes_list` output is already adapted by the
`sysnav_bridge` package landed in commit `886ec4d`.

This **replaces** the v2.4 perception overhaul (YOLOE + SAM3 +
hand-rolled pointcloud projection + sanity gates) — that effort was
duplicating capabilities SysNav already provides (SAM2 + YOLO/RFDETR
+ cloud_image_fusion + voxel voting). Net outcome: less code in this
repo, more capability via a clean ROS2 contract.

Target user flow after v2.4:

```
Terminal 1: cd ~/Desktop/SysNav && ./system_real_robot_with_exploration_planner_go2.sh
Terminal 2: vector-cli sysnav-sim
              → MuJoCo go2_room.xml + Go2 + Piper
              → publishes /registered_scan /camera/image /camera/depth /state_estimation
            SysNav publishes /object_nodes_list
            sysnav_bridge → world_model populated continuously
            user> 抓起蓝色瓶子
              → MobilePickSkill resolves against SysNav-populated world_model
              → navigate + PickTopDownSkill
```

## 2. Background & Motivation

### Why pivot from v2.4 perception overhaul (YOLOE + SAM3) to SysNav

The previous v2.4 spec (now archived under
`.sdd/archive-v2.4-perception-overhaul/`) planned to build:

| v2.4 perception module | SysNav already provides |
|---|---|
| `YoloeDetector` | `detection_node.py` runs YOLO/YOLOE/RFDETR |
| `Sam3Segmenter` | SAM2 integrated in `semantic_mapping_node.py` |
| `pointcloud_projection.py` | `cloud_image_fusion.py` (547 LoC) does mask × depth → world points |
| `sanity_gates.py` | `VoxelFeatureManager` voxel voting + spatial regularisation |
| `Go2Perception` (rewrite) | `semantic_mapping_node` + `vlm_reasoning_node` |

Reimplementing these in vector_os_nano adds ~1500 LoC + 32 tests of
work that would always lag SysNav's mainline. Ji Zhang's lab (the
SysNav authors) is the same lab Vector OS Nano develops within;
keeping the dependency boundary clean (Apache-2.0 ↔ PolyForm-NC via
ROS2 topics, never source copy) lets us focus engineering on
manipulation, agent reasoning, and Go2-specific control — not on
re-implementing scene graphs.

### Why v2.3 grounding failed (root cause carried forward)

The 2026-04-20 live REPL smoke produced a +5.8 m phantom-bottle goal
for `抓起蓝色瓶子`. Root cause was confirmed two-fold:

1. `QwenVLMDetector` resized frames to 160 × 120 before sending to
   the API; bbox coordinates returned in that thumbnail space were
   then applied to the full-resolution depth frame — landing on an
   upper-left wall pixel ~6 m away (`vlm_qwen.py:148`).
2. `Go2ROS2Proxy.get_camera_pose` computed `right = (-sin h, cos h, 0)`
   which is body-LEFT under ROS REP-103 (`go2_ros2_proxy.py:338`);
   contributes ~1.4 m mirrored lateral error.

The v2.4-perception-overhaul branch planned to fix both. With the
SysNav pivot, bug 1 is moot (Qwen path is deleted in this cycle). Bug
2 still exists but is contained: SysNav publishes its own SLAM-derived
poses; our `Go2ROS2Proxy.get_camera_pose` is only consumed by the now-
obsolete grasp pipeline target-pose computation. We carry the fix
forward in this cycle as G3 below.

## 3. Goals

### MUST (blocking for release)

- **G1** `MuJoCoLivox360` virtual lidar — `mj_ray` × ≥4096 pts/frame
  at ≥10 Hz, output `sensor_msgs/PointCloud2` already in `map` frame
  (sim has ground truth, no SLAM step needed). Topic
  `/registered_scan`.
- **G2** `MuJoCoPano360` virtual 360-degree RGBD camera — 6 × 90-degree
  cube-face renders stitched into 1920 × 640 equirectangular at ≥5 Hz.
  Both RGB and aligned depth. Topics `/camera/image` and
  `/camera/depth`.
- **G3** `Go2ROS2Proxy.get_camera_pose` — fix `right` to
  `(sin h, -cos h, 0)`, recompute `up` accordingly. Update
  `tests/integration/test_go2_camera_pose.py` expected values.
- **G4** `GroundTruthOdomPublisher` — derive Odometry from MuJoCo
  `data.qpos[0:7]` at ≥50 Hz, frame_id `map`, child_frame_id `sensor`,
  topic `/state_estimation`. Skips SLAM entirely in sim.
- **G5** `sysnav_sim.launch.py` — bringup launch that starts only
  `semantic_mapping_node`, `detection_node`, `vlm_reasoning_node`
  from the SysNav workspace. Skips `arise_slam_mid360`,
  `livox_ros_driver2`, `tare_planner`, `unitree_webrtc_ros`.
- **G6** Sysnav-bridge **live ROS2 subscriber** — extend
  `vector_os_nano/integrations/sysnav_bridge/` with a
  `LiveSysnavBridge` class that subscribes to
  `/object_nodes_list` and calls
  `world_model.add_object(object_node_to_state(...))` per node.
- **G7** `vector-cli sysnav-sim` end-to-end — single command starts
  MuJoCo + 3 virtual sensors + bridge subscriber. Pre-flight checks
  detect whether SysNav workspace is sourced; logs WARNING and falls
  back to no-perception mode if not.
- **G8** Cleanup: delete `vlm_qwen.py`, `go2_perception.py`,
  `go2_calibration.py`, plus tests and `verify_perception_pick.py`.
  *(landed already this cycle, before spec approval)*
- **G9** `docs/sysnav_simulation.md` — full bringup guide with topic
  matrix, performance targets, troubleshooting.

### SHOULD

- **S1** Coverage ≥ 90 % on each new module (`MuJoCoLivox360`,
  `MuJoCoPano360`, `GroundTruthOdomPublisher`, `LiveSysnavBridge`).
- **S2** `mj_ray` lidar latency budget ≤ 50 ms / frame on RTX 5080.
- **S3** Pano camera latency budget ≤ 100 ms / frame at 1920 × 640.
- **S4** Total sim → SysNav loop latency ≤ 300 ms (frame published →
  ObjectNodeList received).
- **S5** End-to-end smoke: place a blue bottle in `go2_room.xml`,
  start sysnav-sim, within 10 seconds `world_model` contains an entry
  whose world XYZ is within 0.1 m of the MJCF body pose.

### MAY

- **M1** SysNav `cloud_image_fusion.CAMERA_PARA` configuration
  generator — emit a YAML matching our virtual-camera mount.
- **M2** GSO scene swap (replace 3 capsule cylinders with realistic
  meshes) — keep optional for v2.4; main blocker is in
  `pickable_assets/` curation, not perception. **Deferred to v2.5**.
- **M3** RViz config for joint vector_os_nano + SysNav visualisation.

## 4. Non-Goals (explicitly out of scope)

- Real Go2 + real Livox + real Ricoh Theta integration. v2.5+.
- Replacing `vector_navigation_stack` with SysNav's `tare_planner`.
- VLM reasoning in our process — SysNav's `vlm_reasoning_node` owns
  it, we publish `/target_object_instruction` to drive it.
- 6-DoF grasp pose / FoundationPose. Manipulation stays top-down.
- GSO realistic-mesh scene swap (deferred to v2.5).
- `arise_slam_mid360` SLAM in sim (we publish ground-truth odom).
- `tare_planner` C++ exploration — current `vector_navigation_stack`
  + FAR planner remains authoritative.

## 5. User Scenarios

### US1 — `抓起蓝色瓶子` against SysNav-populated world_model

1. User runs `vector-cli sysnav-sim` (Terminal 2 already has SysNav
   launched per `docs/sysnav_simulation.md`).
2. MuJoCo loads `go2_room.xml`; 3 sensor publishers start.
3. SysNav `detection_node` starts running YOLO on `/camera/image`,
   `semantic_mapping_node` clusters lidar points and publishes
   `/object_nodes_list` (~2-3 s after first frames).
4. `LiveSysnavBridge` callback writes ObjectState entries into
   `world_model`. Each cylinder body in `go2_room.xml` becomes one
   `bottle` / `can` ObjectState with reasonable XYZ.
5. User says `抓起蓝色瓶子`.
6. `MobilePickSkill._resolve_target` picks the matching ObjectState
   (existing colour-keyword normaliser in `pick_top_down.py`).
7. Approach + grasp succeed within 10 s of the user prompt.

### US2 — SysNav not running → graceful degrade

1. User runs `vector-cli sysnav-sim` but Terminal 1 SysNav is not
   started.
2. `LiveSysnavBridge.start()` discovers no `/object_nodes_list`
   publisher within 5 s timeout, logs WARNING.
3. `world_model` remains empty (no auto-detect path on Go2 anymore).
4. User says `抓起蓝色瓶子` → `object_not_found` with the existing
   "known pickable objects: []" message — no crash, no phantom.

### US3 — fast-moving object handling

1. User pushes a bottle while `sysnav-sim` is running.
2. SysNav publishes the same `object_id` with updated XYZ;
   `LiveSysnavBridge` calls `add_object` (which upserts by id).
3. `world_model` reflects the new pose within one publish cycle
   (~500 ms).
4. `MobilePickSkill` retargets next time it is invoked.

### US4 — synthetic "moving" status

1. SysNav's voxel-voting flags an object as `moving` (status flag
   `False` in `ObjectNode.status` per `single_object_new.py`).
2. `object_node_to_state` maps `status=False → state="unknown"`.
3. `MobilePick` filters objects in `unknown` state when picking
   (existing behaviour — no change required).

## 6. Technical Constraints

- **Hardware**: Yusen confirms full hardware available. We optimise
  for sim first; real-robot scripts unchanged.
- **GPU budget**: RTX 5080 16 GB shared between MuJoCo render +
  pano cube renders + lidar ray cast + SysNav's SAM2 + YOLO inference.
  Target: ≤ 12 GB total, 5 Hz pano OK during heavy SysNav inference.
- **MuJoCo 3.6+** — `mj_ray` API stable; offscreen GLFW context
  required for cube-face renders.
- **No new Python deps in vector_os_nano**. Pano stitching uses
  numpy + opencv (already in `[perception]` extras).
- **ROS2 Jazzy** — must coexist with running SysNav workspace. Topic
  domain ID consistent.
- Topics published with `RELIABLE` QoS at moderate depth (≥ 5).
- Sensor publishers must run inside the existing
  `go2_vnav_bridge.py` subprocess so MuJoCo state is shared with the
  physics thread; no extra subprocess fan-out.

## 7. Interface Definitions

### 7.1 New `hardware/sim/sensors/lidar360.py`

```python
class MuJoCoLivox360:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_name: str = "trunk",       # mount on Go2 trunk
        offset: tuple[float, float, float] = (0.0, 0.0, 0.10),
        h_resolution: int = 360,         # rays per spin
        v_layers: int = 16,              # Mid-360 has 59-deg vert
        max_range: float = 30.0,
        rate_hz: float = 10.0,
    ) -> None: ...

    def step(self) -> np.ndarray:        # (N, 4) xyz + intensity
        ...

    def to_pointcloud2(
        self, points: np.ndarray, stamp: builtin_interfaces.Time
    ) -> sensor_msgs.PointCloud2: ...
```

### 7.2 New `hardware/sim/sensors/pano360.py`

```python
class MuJoCoPano360:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_name: str = "trunk",
        offset: tuple[float, float, float] = (0.0, 0.0, 0.185),
        out_w: int = 1920,
        out_h: int = 640,                # 120 deg vfov cropped
        rate_hz: float = 5.0,
    ) -> None: ...

    def step(self) -> tuple[np.ndarray, np.ndarray]:    # rgb, depth
        ...
```

### 7.3 New `hardware/sim/sensors/gt_odom.py`

```python
class GroundTruthOdomPublisher:
    """Reads MuJoCo qpos for the Go2 free-joint body and emits
    nav_msgs/Odometry directly. Skips SLAM in sim."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_name: str = "trunk",
        rate_hz: float = 50.0,
        frame_id: str = "map",
        child_frame_id: str = "sensor",
    ) -> None: ...

    def step(self) -> nav_msgs.Odometry: ...
```

### 7.4 New `integrations/sysnav_bridge/live_bridge.py`

```python
class LiveSysnavBridge:
    """rclpy subscriber that fans /object_nodes_list updates into
    the agent's WorldModel via the existing object_node_to_state
    adapter. Imports rclpy / tare_planner.msg lazily to keep tests
    runnable without a sourced SysNav workspace."""

    def __init__(
        self,
        world_model: WorldModel,
        node_name: str = "vector_os_nano_sysnav_bridge",
        topic: str = "/object_nodes_list",
        on_disconnect_after_s: float = 5.0,
    ) -> None: ...

    def start(self) -> bool:    # False if rclpy or tare_planner.msg missing
        ...

    def stop(self) -> None: ...
```

### 7.5 New `vcli/tools/sysnav_sim_tool.py`

CLI entry that wraps `SimStartTool` plus pre-flight check for the
SysNav workspace and `LiveSysnavBridge.start()`.

```
vector> sysnav-sim
  ├─ MuJoCo subprocess up
  ├─ /registered_scan /camera/image /camera/depth /state_estimation publishing
  ├─ Pre-flight: SysNav nodes seen on /object_nodes_list topic? [yes / no]
  └─ LiveSysnavBridge.start() → continuous WorldModel updates
```

## 8. Test Contracts

A core principle of this cycle: **test breadth over implementation
breadth**. The new code paths are infrastructure; they must be
exercised through unit tests with synthetic MuJoCo state and ROS2
mocks before any GPU smoke is attempted.

### 8.1 Unit tests (target ≥ 50 new tests, ≥ 90 % coverage)

| File | Tests |
|---|---|
| `tests/unit/hardware/sim/sensors/test_lidar360.py` | 12+ |
| `tests/unit/hardware/sim/sensors/test_pano360.py` | 10+ |
| `tests/unit/hardware/sim/sensors/test_gt_odom.py` | 8+ |
| `tests/unit/integrations/sysnav_bridge/test_live_bridge.py` | 12+ |
| `tests/unit/vcli/test_sysnav_sim_tool.py` | 8+ |

Mandatory test cases per module:

- **Lidar360**: ray pattern symmetry; range clipping at `max_range`;
  intensity defaults to 1.0; PointCloud2 fields layout (`x` `y` `z`
  `intensity` 32-bit float at correct offsets); empty model degenerate;
  rate-limit honoured (no double-step within 1/rate); points expressed
  in `map` frame given known body pose.
- **Pano360**: 6 cube-face mosaic correctness against synthetic
  uniform-colour scene; equirectangular pixel <-> spherical angle
  invariant on a known pole pixel; depth and RGB grid alignment;
  out-of-range depth clipped; rate-limit honoured.
- **GT odom**: position == body pose; quaternion normalisation;
  twist computed via finite difference between successive `step()`
  calls; first-call twist is zeros (no prior); frame_id correctness.
- **LiveSysnavBridge**: degrade-to-noop when `rclpy` import fails;
  degrade when `tare_planner.msg` missing; with both available,
  callback dispatches one `add_object` per node; status=False object
  marked unknown; no crash on malformed payload (skip + WARN log);
  `stop()` is idempotent; `on_disconnect_after_s` triggers WARN log
  exactly once.
- **sysnav_sim_tool**: pre-flight discovery of running SysNav nodes;
  graceful path when SysNav not present; bridge `start()` failure
  does not abort sim startup; `stop()` cleans up subscribers and
  shutdowns rclpy.

### 8.2 Integration tests (≥ 5 new)

| File | Test |
|---|---|
| `tests/integration/test_lidar360_against_world.py` | Ray-cast hits a known wall body in `go2_room.xml` at expected world XYZ within 5 cm |
| `tests/integration/test_pano360_against_world.py` | A coloured marker placed at θ = 0° (in front of robot) appears at the expected image column |
| `tests/integration/test_gt_odom_against_walk.py` | After `MuJoCoGo2.set_velocity(0.5, 0)` for 1 s, odom reports x ≈ 0.5 m forward |
| `tests/integration/test_sysnav_sim_smoke.py` | Mocked `/object_nodes_list` publisher → bridge → `world_model.get_objects` returns expected entries |
| `tests/integration/test_xmat_rep103_regression.py` | After G3 fix, `Go2ROS2Proxy.get_camera_pose` returns `right == (0, -1, 0)` and `up == (0, 0, 1)` at heading 0 |

### 8.3 E2E smoke (Yusen-run, GPU + SysNav workspace required)

`scripts/smoke_sysnav_sim.py`:
- Starts MuJoCo `go2_room.xml`.
- Verifies all four topics publishing within 5 s.
- Asserts `/object_nodes_list` carries ≥ 1 node within 30 s when
  SysNav workspace is sourced (skipped with `pytest.skip` otherwise).
- Asserts a known cylinder pose appears in `world_model` within 0.1 m.

### 8.4 Regression contract

Existing 70 tests across `test_pick_top_down.py`,
`test_mobile_pick.py`, `test_sysnav_bridge_mapping.py` must remain
green throughout v2.4. CI runs all three plus the new unit tests.

### 8.5 Coverage gate

`pytest --cov=vector_os_nano.hardware.sim.sensors
--cov=vector_os_nano.integrations.sysnav_bridge
--cov-fail-under=90` blocks the wave-5 QA gate.

## 9. Acceptance Criteria

1. All G1–G9 met.
2. ≥ 50 new unit tests + ≥ 5 integration tests, all green on CI.
3. Coverage ≥ 90 % on new modules.
4. Existing 70 manipulation tests preserved green.
5. E2E smoke (8.3) passes when SysNav is running, gracefully skips
   when not.
6. `ruff check vector_os_nano/ tests/` clean.
7. Yusen REPL smoke approves.
8. `progress.md` updated with the v2.4 narrative.

## 10. Open Questions

| # | Question | Default |
|---|---|---|
| O1 | Pano stitching — 6 cube faces or single fisheye? | 6 cube faces (more robust geometry) |
| O2 | Lidar ray pattern — uniform polar grid or Mid-360 spinning? | Polar grid (deterministic for tests) |
| O3 | `/registered_scan` already in `map` frame — do we still need to publish `/aft_mapped_to_init_incremental` for SysNav's bagfile platform? | No — sim platform only |
| O4 | Pano camera mount offset — match SysNav `cloud_image_fusion` defaults (z=0.265 m, x=-0.12 m, y=-0.075 m for "go2 4090") or pick a Vector OS Nano-specific tuple? | Match SysNav defaults; emit YAML override (M1) if mismatch |
| O5 | LiveSysnavBridge — single-threaded `rclpy.spin_once` polled from MuJoCo step thread, or its own `MultiThreadedExecutor`? | Own executor (matches Go2ROS2Proxy v2.3 fix) |
| O6 | M2 GSO scene swap — keep deferred to v2.5? | Yes — focus this cycle on infrastructure |

CEO has authorised proceeding with all defaults unless a change is
flagged at /sdd plan time.

## 11. Carry-forward debt

- `_normalise_color_keyword` private API (since v2.3 H3) — deferred.
- `_wait_stable` extract → `mobile_helpers.py` — deferred.
- `VECTOR_SHARED_EXECUTOR=0` rollback path — deferred.
- `coverage` × `numpy 2.4` C-tracer conflict — settrace workaround
  persists in `test_lidar360.py`; document.
- Real-robot bringup path — v2.5.
- Replacing `vector_navigation_stack` with SysNav's `tare_planner` —
  v3.0 architectural decision.
