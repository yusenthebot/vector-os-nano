# v2.4 SysNav Simulation Integration — Technical Plan

**Status**: APPROVED async (CEO auto-approval)
**Depends on**: spec.md (`.sdd/spec.md`)
**Branch**: `feat/v2.0-vectorengine-unification`

---

## 1. Architecture

```
                    Vector OS Nano main process (Apache 2.0)
        ┌────────────────────────────────────────────────────┐
        │  vector-cli sysnav-sim                              │
        │   ├─ MuJoCo subprocess (existing go2_vnav_bridge)  │
        │   │    └─ + 3 new sensor publishers               │
        │   │                                                │
        │   └─ LiveSysnavBridge (new, rclpy subscriber)      │
        │        └─> world_model.add_object(ObjectState)     │
        │             via existing object_node_to_state()    │
        └────────────────────────────────────────────────────┘
                              ▲
              ROS2 topics      │ /object_nodes_list
                              │
                              │ /registered_scan, /camera/image,
                              ▼ /camera/depth, /state_estimation
        ┌────────────────────────────────────────────────────┐
        │  SysNav Workspace (sibling, PolyForm-NC)           │
        │   semantic_mapping_node + detection_node + vlm_*    │
        └────────────────────────────────────────────────────┘
```

Isolation invariants preserved from v2.3:
- Skills do **not** import sensors or rclpy.
- `integrations/sysnav_bridge/` defers ROS2 imports to runtime so
  unit tests run on a CPU-only Python interpreter.
- `hardware/sim/sensors/` may import `mujoco` (we already do) but not
  `rclpy` at module load — rclpy publishers are constructed lazily by
  `go2_vnav_bridge.py` and passed in.

## 2. Technical Decisions

| # | Decision | Choice | Reason |
|---|---|---|---|
| D1 | Sim sensor location | `vector_os_nano/hardware/sim/sensors/` (new package) | Isolates publisher logic from `MuJoCoGo2` |
| D2 | Lidar pattern | Uniform polar grid (h_resolution × v_layers) | Deterministic — tests can assert exact pixel-to-ray mapping |
| D3 | Pano stitching | 6 cube faces × 90° render → equirectangular reproject | Robust geometry, Apache-friendly numpy + opencv math |
| D4 | Stitch backend | OpenCV `cv2.remap` with precomputed LUT | Single LUT init, ~5 ms per frame on RTX 5080 |
| D5 | Odometry source in sim | Read MuJoCo qpos directly → `/state_estimation`; **skip** `arise_slam_mid360` | Sim has ground truth; SLAM in sim is wasted compute |
| D6 | rclpy in tests | Lazy import, mock-friendly factory | 0 ROS2 dep at test collection |
| D7 | Bridge executor | Own `MultiThreadedExecutor` (matches v2.3 Go2 proxy refactor) | Aligns with `Ros2Runtime` pattern already in repo |
| D8 | LiveSysnavBridge resilience | Soft-degrade: missing `tare_planner.msg`, `rclpy`, or topic → log WARNING + run as no-op | Already a v2.3 Q decision pattern |
| D9 | Sysnav-sim entry point | New CLI tool `sysnav_sim_tool.py`, distinct from `sim_tool.py` | Keeps go2sim minimal; sysnav-sim adds bridge wiring |
| D10 | Coordinate frames | All sim publishes in `map` frame; SysNav's frame conventions match | Avoid TF chaos; `frame_id="map"` everywhere |
| D11 | Cleanup of v2.3 perception | Delete `vlm_qwen.py`, `go2_perception.py`, `go2_calibration.py` and their tests | Already landed pre-spec; no orphan paths |
| D12 | xmat fix (G3) | Two-line patch in `go2_ros2_proxy.py` + update `test_go2_camera_pose.py` | Carries the v2.3.1 hot-fix forward |
| D13 | GSO scene swap | **Out of scope** for v2.4 | Defer to v2.5; SysNav perception works on capsule cylinders too (lower fidelity but unblocks integration) |
| D14 | Pano camera mount | Match SysNav `cloud_image_fusion` defaults: x=-0.12, y=-0.075, z=0.265, roll=-π/2, yaw=-π/2 | Avoids editing SysNav config; emit our launch with these as the canonical mount |
| D15 | Lidar mount | Trunk-top (z = 0.10 m above body) | Matches typical Mid-360 mount on Go2 |

## 3. Module Design

### 3.1 `hardware/sim/sensors/__init__.py` (NEW)

```python
"""MuJoCo virtual ROS2 sensors.

Module load is rclpy-free; publishers are wired by
`go2_vnav_bridge.py` after rclpy is initialised.
"""
from .lidar360 import MuJoCoLivox360
from .pano360 import MuJoCoPano360
from .gt_odom import GroundTruthOdomPublisher

__all__ = ["MuJoCoLivox360", "MuJoCoPano360", "GroundTruthOdomPublisher"]
```

### 3.2 `lidar360.py` — Virtual Mid-360 (~180 LoC)

Algorithm:

1. Precompute ray directions in body frame (`h_resolution × v_layers`,
   default 360 × 16 = 5760 rays).
2. Each `step()`:
   - Read body world pose from `data.xpos[body_id]`,
     `data.xquat[body_id]`.
   - Rotate ray directions to world frame.
   - Call `mujoco.mj_ray()` for each ray (vectorised loop).
   - Build `(N, 4)` array of `(x, y, z, intensity=1.0)` for hits inside
     `max_range`.
3. `to_pointcloud2()`:
   - Use the existing `ros2_bag_utils.create_point_cloud` helper
     pattern (we'll port the relevant bits) — but simpler: hand-build
     `PointCloud2` with fields `x y z intensity` (float32 each).

Rate-limit:
- Track last `step()` wall time; subsequent calls within `1/rate_hz`
  return cached cloud.

Tests use `mujoco.MjModel` constructed from a tiny inline MJCF:

```xml
<mujoco>
  <worldbody>
    <body name="trunk" pos="0 0 0.5"><freejoint/></body>
    <geom name="wall" type="box" pos="3 0 0.5" size="0.1 5 5"/>
  </worldbody>
</mujoco>
```

### 3.3 `pano360.py` — 360-degree RGBD (~250 LoC)

Algorithm:

1. Build 6 MuJoCo cameras in MJCF runtime (via `mujoco.MjvCamera`):
   front / back / left / right / up / down at `body_offset`.
2. For each face: `mujoco.Renderer(...).render()` → RGB and depth
   buffers at 480 × 480.
3. Stitch into 1920 × 640 equirectangular via precomputed `cv2.remap`
   LUT mapping each output (u, v) to a (face, fx, fy) sample.
4. Crop top 30° / bottom 30° (per SysNav `cloud_image_fusion.py:68`
   "cropped 30 degree (160 pixels) in top and 30 degree (160 pixels) in
   bottom").

Rate-limit identical to lidar.

LUT precompute is one-time at __init__; runtime cost is dominated by
the 6 GPU renders (~80 ms total on RTX 5080).

Test inline MJCF places coloured boxes around the camera; assert the
red box at θ=0° appears in the image columns matching θ=0°.

### 3.4 `gt_odom.py` — Ground-truth odometry (~120 LoC)

Algorithm:

1. Read `data.xpos[body_id]` and `data.xquat[body_id]`.
2. Differentiate position to estimate twist:

   ```python
   dt = now - self._last_t
   linear = (xpos - self._last_xpos) / max(dt, 1e-3)
   ```

3. Build `nav_msgs/Odometry`. First call returns zero twist.

Tests assert:
- Position equals body pose.
- Twist after teleporting body returns finite-diff velocity.
- Quaternion normalised.
- Frame fields correct.

### 3.5 `integrations/sysnav_bridge/live_bridge.py` — Live ROS2 subscriber (~200 LoC)

```python
class LiveSysnavBridge:
    def __init__(self, world_model, ...): ...

    def start(self) -> bool:
        try:
            import rclpy
            from rclpy.executors import MultiThreadedExecutor
            from tare_planner.msg import ObjectNodeList
        except ImportError as exc:
            logger.warning("[sysnav_bridge] %s — degrading to no-op", exc)
            self._active = False
            return False

        rclpy.init(args=[])
        self._node = rclpy.create_node(self._name)
        self._sub = self._node.create_subscription(
            ObjectNodeList, self._topic, self._callback, 50,
        )
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True,
        )
        self._spin_thread.start()
        self._active = True
        return True

    def _callback(self, msg):
        for node in msg.nodes:
            try:
                state = object_node_to_state(node, prior=self._world_model.get_object(...))
                self._world_model.add_object(state)
            except Exception as exc:
                logger.warning("[sysnav_bridge] skipping malformed node: %s", exc)
```

Disconnect detection:
- Track `last_msg_t`. If `now - last_msg_t > on_disconnect_after_s`,
  log WARNING once. Reset on next message.

### 3.6 `vcli/tools/sysnav_sim_tool.py` — CLI entry (~150 LoC)

Wraps `SimStartTool` (with_arm=True) and adds:

1. Pre-flight: import `tare_planner.msg`; if missing, WARN and proceed
   with `LiveSysnavBridge` disabled.
2. Construct `LiveSysnavBridge` after MuJoCo subprocess is up.
3. Wait up to 5 s for first `/object_nodes_list` message; log status.
4. Register `stop_sysnav_sim` tool to clean up.

### 3.7 `go2_vnav_bridge.py` patches (~100 LoC of new wiring)

Where the bridge subprocess currently constructs the rclpy node, add:

```python
self._lidar = MuJoCoLivox360(model, data, ...)
self._pano  = MuJoCoPano360(model, data, ...)
self._odom  = GroundTruthOdomPublisher(model, data, ...)

self._lidar_pub = node.create_publisher(PointCloud2, '/registered_scan', 10)
self._image_pub = node.create_publisher(Image, '/camera/image', 5)
self._depth_pub = node.create_publisher(Image, '/camera/depth', 5)
self._odom_pub  = node.create_publisher(Odometry, '/state_estimation', 50)
```

Per `mj_step` callback:

```python
if self._lidar.due():  self._lidar_pub.publish(self._lidar.step_pc2(stamp))
if self._pano.due():   img, depth = self._pano.step()
                       self._image_pub.publish(self._cv_bridge.cv2_to_imgmsg(img, 'rgb8'))
                       self._depth_pub.publish(self._cv_bridge.cv2_to_imgmsg(depth, '32FC1'))
if self._odom.due():   self._odom_pub.publish(self._odom.step_msg(stamp))
```

`due()` is a simple rate-limit method on each sensor.

### 3.8 `go2_ros2_proxy.py:338` — xmat fix (G3)

```python
# OLD (wrong — ROS-LEFT under REP-103):
# right = np.array([-sin_h, cos_h, 0.0])
# up = np.cross(fwd, right)

# NEW (REP-103 compliant):
right = np.array([sin_h, -cos_h, 0.0])
up = np.cross(right, fwd)
```

Update `tests/integration/test_go2_camera_pose.py` expected values:
at heading=0: `right = (0, -1, 0)`, `up = (0, 0, 1)`.

## 4. Data Flow

```
MuJoCo step (~1000 Hz)
   │
   ├─ Lidar.step()       (10 Hz)  ──▶ /registered_scan PointCloud2
   ├─ Pano.step()        (5 Hz)   ──▶ /camera/image    Image (rgb8)
   │                                ──▶ /camera/depth    Image (32FC1)
   └─ GTodom.step()      (50 Hz)  ──▶ /state_estimation Odometry

(SysNav consumes; meanwhile inside Vector OS Nano main process:)

LiveSysnavBridge subscriber (10 Hz typical from SysNav side)
   │
   └─▶ world_model.add_object(ObjectState)
         │ via object_node_to_state() — already tested
         ▼
       Existing skills: DetectSkill / PickTopDownSkill / MobilePickSkill
```

## 5. Directory Structure

```
vector_os_nano/
├── hardware/sim/
│   ├── go2_vnav_bridge.py            (MODIFIED — sensor publishers)
│   ├── go2_ros2_proxy.py              (MODIFIED — xmat G3)
│   └── sensors/                       (NEW package)
│       ├── __init__.py
│       ├── lidar360.py
│       ├── pano360.py
│       └── gt_odom.py
├── integrations/sysnav_bridge/
│   ├── __init__.py                    (UPDATED — re-export LiveSysnavBridge)
│   ├── topic_interfaces.py            (existing)
│   └── live_bridge.py                 (NEW)
└── vcli/tools/
    ├── sim_tool.py                    (UPDATED — Qwen wire-up gone, see G8)
    └── sysnav_sim_tool.py             (NEW)

tests/
├── unit/hardware/sim/sensors/         (NEW dir)
│   ├── test_lidar360.py
│   ├── test_pano360.py
│   └── test_gt_odom.py
├── unit/integrations/sysnav_bridge/   (NEW dir)
│   └── test_live_bridge.py
├── unit/vcli/test_sysnav_sim_tool.py
└── integration/
    ├── test_lidar360_against_world.py     (NEW)
    ├── test_pano360_against_world.py      (NEW)
    ├── test_gt_odom_against_walk.py        (NEW)
    ├── test_sysnav_sim_smoke.py            (NEW)
    ├── test_xmat_rep103_regression.py      (NEW)
    └── test_sysnav_bridge_mapping.py       (existing — kept green)

scripts/
└── smoke_sysnav_sim.py                 (NEW)

docs/
├── sysnav_integration.md               (existing — bringup)
└── sysnav_simulation.md                (NEW — sim-specific)
```

## 6. Key Implementation Details

### 6.1 Lidar polar grid

```python
def _build_ray_dirs(h_resolution: int, v_layers: int) -> np.ndarray:
    azimuths = np.linspace(-np.pi, np.pi, h_resolution, endpoint=False)
    elevations = np.linspace(np.deg2rad(-7.0), np.deg2rad(52.0), v_layers)
    az_grid, el_grid = np.meshgrid(azimuths, elevations)
    cx = np.cos(el_grid) * np.cos(az_grid)
    cy = np.cos(el_grid) * np.sin(az_grid)
    cz = np.sin(el_grid)
    return np.stack([cx, cy, cz], axis=-1).reshape(-1, 3)
```

Body→world rotation via quaternion. Total rays default = 5760.

### 6.2 Pano cube-face → equirectangular LUT

Precompute mapping from output `(u, v)` to a cube face and face-pixel
coordinates:

```python
def _build_equirec_lut(out_w: int, out_h: int, face_size: int):
    """For each output pixel, return (face_idx, fx, fy) sample point."""
    # u, v in output → spherical angles
    phi = (np.linspace(0, 1, out_w) - 0.5) * 2 * np.pi    # azimuth
    theta = (0.5 - np.linspace(0, 1, out_h)) * np.pi        # elevation
    # ...standard cube-map sampling math...
    return face_idx_map, fx_map, fy_map
```

Apply via `cv2.remap` per face. Combine 6 face renders into one image.
This is the SysNav-equivalent of what their Unity binary does
internally.

### 6.3 PointCloud2 hand-build

```python
def _build_pointcloud2(points: np.ndarray, stamp, frame_id) -> PointCloud2:
    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * points.shape[0]
    msg.is_dense = True
    msg.data = points.astype(np.float32).tobytes()
    return msg
```

### 6.4 LiveSysnavBridge prior-state lookup

```python
def _callback(self, msg):
    for node in msg.nodes:
        canonical_id = f"sysnav_{int(node.object_id[0])}" if node.object_id else "sysnav_unknown"
        prior = self._world_model.get_object(canonical_id)
        try:
            state = object_node_to_state(node, prior=prior)
            self._world_model.add_object(state)
        except Exception as exc:
            logger.warning("[sysnav_bridge] dropping malformed node: %s", exc)
```

## 7. Test Strategy

### 7.1 Coverage philosophy

This cycle adds infrastructure (sensor sims + ROS subscriber). Bugs
here cascade into all downstream behaviour, so we over-invest in
unit tests:

- ≥ 50 unit tests across 5 new modules.
- Each module tests at least: happy path, empty input, rate-limit,
  resilience (missing dep / malformed input), one geometric invariant.
- Mocks: small inline MJCF strings — no go2.xml dependency for unit
  tests (avoids OOM cascade from `feedback_no_parallel_agents.md`).

### 7.2 Mock infrastructure

`tests/unit/hardware/sim/sensors/conftest.py`:

```python
@pytest.fixture
def tiny_mujoco_model_data():
    """Single-body MJCF for sensor tests — no Go2 robot, no perception
    dependency cascade. Returns (model, data) ready for sensors."""
    xml = ...
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data
```

### 7.3 ROS2 mocking

`tests/unit/integrations/sysnav_bridge/conftest.py`:

```python
class _StubObjectNodeList:
    def __init__(self, nodes):
        self.nodes = nodes
        self.header = _StubHeader()

@pytest.fixture
def patch_rclpy(monkeypatch):
    """Stub out rclpy + tare_planner.msg so LiveSysnavBridge.start()
    can be exercised in isolation."""
    ...
```

### 7.4 Coverage gate

```bash
pytest \
  tests/unit/hardware/sim/sensors \
  tests/unit/integrations/sysnav_bridge \
  tests/unit/vcli/test_sysnav_sim_tool.py \
  --cov=vector_os_nano.hardware.sim.sensors \
  --cov=vector_os_nano.integrations.sysnav_bridge \
  --cov=vector_os_nano.vcli.tools.sysnav_sim_tool \
  --cov-fail-under=90
```

## 8. Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | `mujoco.mj_ray` Python loop @ 5760 rays/frame too slow | Benchmark in T1 RED; if > 50 ms, vectorise via batched call (mujoco 3.6 supports `mj_multiRay`) |
| R2 | 6 cube-face render saturates GPU when SAM2 also running | Drop pano rate to 3 Hz under load; runtime VRAM probe in pre-flight |
| R3 | SysNav `cloud_image_fusion` rejects our pano due to mount mismatch | M1 emit YAML override; first launch logs SysNav's complained-about params |
| R4 | rclpy version drift between Vector OS Nano venv and SysNav workspace | Docs spec `source /opt/ros/jazzy/setup.bash` in both terminals; pre-flight checks `/object_nodes_list` topic type |
| R5 | LiveSysnavBridge spin-thread leaks on agent shutdown | Match `Ros2Runtime` graceful-stop pattern; `stop()` joins with 2 s timeout |
| R6 | Pano LUT precompute hits memory if out resolution upsizes | Cap `out_w * out_h * 4 bytes < 50 MB` (default 1920×640×4 = 4.9 MB OK) |
| R7 | Sensor publishers consume MuJoCo state from Python while physics thread mutates → segfault | Sensors run in same subprocess as physics step; gate via existing thread lock pattern from `MuJoCoGo2.set_velocity` |
| R8 | Tests fail on dev box without `mujoco` available | We already require mujoco for existing tests; this risk is constant. Lidar/pano unit tests skip when import fails |

## 9. Execution Plan (for task.md)

| Wave | Task | Agent | Depends |
|---|---|---|---|
| **W0** | T0 env probe — mujoco + opencv versions, mj_ray smoke | dispatcher | — |
| **W1** | T1 lidar360 + tests | Alpha | T0 |
| **W1** | T2 gt_odom + tests | Beta | T0 |
| **W1** | T3 G3 xmat fix + regression test | Gamma | — |
| **W2** | T4 pano360 + tests | Alpha | T0, T1 (cube-face renderer pattern shared) |
| **W2** | T5 LiveSysnavBridge + tests | Beta | T0 |
| **W3** | T6 go2_vnav_bridge wiring + integration tests | Alpha | T1, T2, T4 |
| **W3** | T7 sysnav_sim_tool CLI + tests | Beta | T5, T6 |
| **W4** | T8 smoke_sysnav_sim.py + docs | Gamma | all |
| **W5** | QA: code-reviewer + security-reviewer | parallel subagents | all |
| **W6** | CEO live-REPL smoke | Yusen | all |

Estimated wall-clock: **3–4 days** at serial subagent cadence.

## 10. Carry-forward debt (not addressed in v2.4)

- GSO realistic-mesh scene swap → v2.5
- `_normalise_color_keyword` promote to public → v2.5
- `_wait_stable` extract → v2.5
- `arise_slam_mid360` for real-robot bringup → v2.5
- Replace `vector_navigation_stack` with SysNav `tare_planner` → v3.0
- Real Go2 + real Livox + real Ricoh integration → v3.0
- 6-DoF FoundationPose grasp → v3.0
