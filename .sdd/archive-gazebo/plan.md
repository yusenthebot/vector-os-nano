# Gazebo Harmonic Integration — Technical Plan

**Version**: 1.0
**Author**: Architect (Opus)
**Date**: 2026-04-11
**Spec**: `.sdd/spec.md` v1.1 (approved)
**Status**: DRAFT

---

## 1. Architecture Overview

```
  vector-cli (REPL)
       │
       ▼
  SimStartTool._start_gazebo_go2()
       │
       ├── subprocess: ros2 launch gazebo/launch/go2_sim.launch.py
       │     ├── gz sim apartment.sdf (Gz Sim Harmonic, DART physics)
       │     ├── spawn Go2 SDF model (12 DOF + sensors)
       │     ├── gz_ros2_control (rl_quadruped_controller)
       │     └── ros_gz_bridge (topic relay per bridge.yaml)
       │
       └── GazeboGo2Proxy(Go2ROS2Proxy).connect()
             ├── subscribes: /state_estimation, /camera/image, /camera/depth
             ├── publishes:  /cmd_vel_nav, /goal_point, /way_point
             └── Agent(base=proxy) + skills + VLM + SceneGraph
```

### Key: What changes vs what stays the same

| Component | Change? | Detail |
|-----------|---------|--------|
| GazeboGo2Proxy | **NEW** | ~40 lines, extends Go2ROS2Proxy |
| Go2 SDF model | **NEW** | URDF→SDF from quadruped_ros2_control + sensor xacro |
| apartment.sdf | **NEW** | Custom world with PBR textured furniture |
| bridge.yaml | **NEW** | ros_gz_bridge topic mapping |
| go2_sim.launch.py | **NEW** | ROS2 launch: gz + spawn + bridge + controller |
| launch/stop scripts | **NEW** | Preflight + startup + teardown |
| sim_tool.py | **MODIFY** | Add "gazebo" backend + _start_gazebo_go2() |
| sim/__init__.py | **MODIFY** | Add GazeboGo2Proxy import |
| Go2ROS2Proxy | NO CHANGE | Parent class unchanged |
| VGG cognitive layer | NO CHANGE | Works through BaseProtocol |
| Nav stack (FAR/TARE) | NO CHANGE | Consumes same ROS2 topics |
| Skills (explore, navigate, etc.) | NO CHANGE | Same primitives API |

---

## 2. Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Locomotion controller | `rl_quadruped_controller` from quadruped_ros2_control | Same obs/action format as our go2_jit.pt. TorchScript C++ inference at 1000 Hz via ros2_control. Gazebo does joint-level physics — no velocity hacks. |
| Fallback controller | `unitree_guide_controller` (Phase 1 only) | Lighter deps than OCS2. Works out of box. OCS2 MPC deferred to Phase 2 (Pinocchio build complexity). |
| Go2 model source | `descriptions/unitree/go2_description` from quadruped_ros2_control | 12 DOF URDF with ros2_control config verified on Jazzy. We add sensor xacro on top. |
| World construction | Custom SDF + Gazebo Fuel models | AWS house is Gazebo Classic format (migration cost). Custom SDF with Fuel furniture gives control over layout + PBR materials. |
| Gazebo ↔ ROS2 bridge | `ros_gz_bridge` (already installed) | Standard ros_gz approach. YAML config maps gz-transport topics to ROS2 topics. |
| Process management | `ros2 launch` via subprocess in sim_tool.py | Same pattern as `_start_go2()` which uses subprocess.Popen for launch_explore.sh. Process group with atexit cleanup. |
| Libtorch dependency | Download pre-built, cache in ~/.vector_os_nano/deps/ | rl_quadruped_controller needs libtorch 2.5.0. One-time download. Script in launch_gazebo.sh checks and downloads if missing. |

---

## 3. Module Design

### Module A: GazeboGo2Proxy

**File**: `vector_os_nano/hardware/sim/gazebo_go2_proxy.py`
**Responsibility**: Health check + proxy identity for Gazebo backend
**Input**: ROS2 topics from ros_gz_bridge
**Output**: Same BaseProtocol interface as Go2ROS2Proxy
**Dependencies**: Go2ROS2Proxy (parent), rclpy

```python
# Pseudocode — actual implementation in Phase 4
class GazeboGo2Proxy(Go2ROS2Proxy):
    _NODE_NAME = "gazebo_go2_proxy"

    @property
    def name(self) -> str: return "gazebo_go2"

    @property
    def supports_lidar(self) -> bool: return True

    def connect(self) -> None:
        if not self.is_gazebo_running():
            raise ConnectionError(...)
        super().connect()

    @staticmethod
    def is_gazebo_running() -> bool:
        # Check for /clock topic (gz sim always publishes this)
        # subprocess: ros2 topic list | grep /clock
        ...
```

### Module B: Go2 SDF Model with Sensors

**Directory**: `gazebo/models/go2/`
**Responsibility**: Go2 robot model with Livox MID-360 + D435 for Gazebo Harmonic
**Source**: `quadruped_ros2_control/descriptions/unitree/go2_description/`

Files:
- `gazebo/models/go2/model.urdf.xacro` — base Go2 + our sensor additions
- `gazebo/models/go2/sensors.xacro` — Livox MID-360 gpu_lidar + D435 camera/depth
- `gazebo/models/go2/ros2_control.yaml` — joint controller config (rl or guide)
- `gazebo/models/go2/model.sdf` — generated from xacro (or direct SDF)

Sensor config (matching MuJoCo/Isaac Sim):

| Sensor | gz-sensors type | Mount (relative to base_link) | Key params |
|--------|----------------|-------------------------------|------------|
| Livox MID-360 | `gpu_lidar` | pos="0.3 0 0.2", pitch=-20deg | horizontal_samples=360, vertical_samples=30, range=12m, rate=10Hz |
| D435 RGB | `camera` | pos="0.3 0 0.05", pitch=-5deg | width=640, height=480, hfov=1.2rad(~69deg), rate=30Hz |
| D435 Depth | `depth_camera` | co-located with RGB | width=640, height=480, near=0.3, far=10.0, rate=30Hz |
| IMU | `imu` | pos="0 0 0" (base_link center) | rate=200Hz |

### Module C: Apartment World SDF

**File**: `gazebo/worlds/apartment.sdf`
**Responsibility**: Textured indoor environment for nav + VLN testing

Layout (inspired by MuJoCo house, upgraded for VLN):

```
  ┌──────────────┬───────────┬──────────┐
  │              │           │          │
  │  Bedroom     D  Hallway  D Kitchen  │
  │  (carpet)    │           │ (tile)   │
  │              │           │          │
  ├──────D───────┤           ├────D─────┤
  │              │           │          │
  │  Bathroom    │           │  Living  │
  │  (tile)      │           │  Room    │
  │              │           │ (wood)   │
  └──────────────┴───────────┴──────────┘
  D = door opening (0.9m)
  Total: ~65m2, 5 rooms + hallway
```

Materials:
- Floor: PBR textures per room (wood, tile, carpet) from Gazebo Fuel materials
- Walls: Painted drywall texture, 2.5m height
- Ceiling: White flat

Furniture (Gazebo Fuel models with textures):
- Living room: sofa, coffee_table, bookshelf, TV_stand
- Kitchen: table, chairs(x2), counter, cabinet
- Bedroom: bed, nightstand, dresser, desk
- Bathroom: bathtub (or shower stall), sink, toilet
- Hallway: coat_rack, shoe_shelf

Graspable objects (dynamic, with mass/inertia):
- red_cup (0.15kg), blue_bottle (0.3kg), phone (0.2kg), book (0.5kg)
- bowl (0.2kg), remote (0.15kg), apple (0.2kg), keys (0.05kg)
- Placed on: coffee_table, kitchen_table, nightstand, counter, desk

Lighting:
- 1 directional light (sun, through windows)
- 1 point light per room (ceiling fixture)
- Ambient: 0.3 (enough for camera exposure)

### Module D: ros_gz_bridge Configuration

**File**: `gazebo/config/bridge.yaml`

```yaml
# ros_gz_bridge topic mapping
- topic_name: /state_estimation
  ros_type_name: nav_msgs/msg/Odometry
  gz_type_name: gz.msgs.Odometry
  direction: GZ_TO_ROS

- topic_name: /registered_scan
  ros_type_name: sensor_msgs/msg/PointCloud2
  gz_type_name: gz.msgs.PointCloudPacked
  direction: GZ_TO_ROS

- topic_name: /camera/image
  ros_type_name: sensor_msgs/msg/Image
  gz_type_name: gz.msgs.Image
  direction: GZ_TO_ROS

- topic_name: /camera/depth
  ros_type_name: sensor_msgs/msg/Image
  gz_type_name: gz.msgs.Image
  direction: GZ_TO_ROS

- topic_name: /cmd_vel_nav
  ros_type_name: geometry_msgs/msg/Twist
  gz_type_name: gz.msgs.Twist
  direction: ROS_TO_GZ

- topic_name: /clock
  ros_type_name: rosgraph_msgs/msg/Clock
  gz_type_name: gz.msgs.Clock
  direction: GZ_TO_ROS

- topic_name: /imu/data
  ros_type_name: sensor_msgs/msg/Imu
  gz_type_name: gz.msgs.IMU
  direction: GZ_TO_ROS
```

Note: Exact Gazebo topic names depend on model namespace. The launch file may need to remap from `/model/go2/...` to the bridge input names. This is configured in the SDF model `<topic>` tags or via launch remapping.

### Module E: ROS2 Launch File

**File**: `gazebo/launch/go2_sim.launch.py`

```python
# Pseudocode structure
def generate_launch_description():
    world = LaunchConfiguration('world', default='apartment')
    gui = LaunchConfiguration('gui', default='true')

    return LaunchDescription([
        # 1. Launch Gz Sim with world
        IncludeLaunchDescription(
            ros_gz_sim: GzServer + GzGui(if gui)
            world_sdf_file = gazebo/worlds/{world}.sdf
        ),

        # 2. Spawn Go2 model
        Node(ros_gz_sim, 'create',
            arguments=['-file', go2_model_path,
                       '-name', 'go2',
                       '-x', '0', '-y', '0', '-z', '0.35']),

        # 3. Start ros2_control controller manager
        Node(controller_manager, spawner,
            arguments=['rl_quadruped_controller']),

        # 4. Start ros_gz_bridge
        Node(ros_gz_bridge, 'parameter_bridge',
            parameters=[bridge.yaml]),

        # 5. Optional: joint_state_broadcaster
        Node(controller_manager, spawner,
            arguments=['joint_state_broadcaster']),
    ])
```

### Module F: CLI Backend Registration

**File**: `vector_os_nano/vcli/tools/sim_tool.py` (modify)

Changes:
1. Add `"gazebo"` to backend enum
2. Add `_start_gazebo_go2()` — follows exact same pattern as `_start_isaac_go2()`:
   - Launch Gazebo via subprocess (ros2 launch)
   - Wait for /state_estimation topic (timeout 30s)
   - Create GazeboGo2Proxy, connect
   - Build Agent with skills + VLM + SceneGraph (identical code)
3. Route in execute(): `if backend == "gazebo": agent = self._start_gazebo_go2()`

### Module G: Launch/Stop Scripts

**Files**: `scripts/launch_gazebo.sh`, `scripts/stop_gazebo.sh`

launch_gazebo.sh:
- Args: `--world apartment|empty`, `--headless`, `--controller rl|guide`
- Preflight: `gz sim --version`, `ros2 pkg list | grep ros_gz`
- Launch: `ros2 launch gazebo go2_sim.launch.py world:=$WORLD gui:=$GUI`
- Wait: poll `ros2 topic list | grep state_estimation` (timeout 30s)
- Print: topic list + connection info

stop_gazebo.sh:
- `pkill -f "gz sim"`
- `pkill -f "ros_gz_bridge"`
- `pkill -f "controller_manager"`

---

## 4. Data Flow

```
Gz Sim (DART physics)
  │
  ├── Go2 Articulation (12 joints)
  │     ├── joint states → gz_ros2_control → rl_quadruped_controller
  │     │                                      ├── reads: joint pos/vel, IMU, commands
  │     │                                      ├── runs: TorchScript policy inference
  │     │                                      └── writes: joint position targets
  │     └── odometry → ros_gz_bridge → /state_estimation (50Hz)
  │
  ├── gpu_lidar (Livox MID-360)
  │     └── PointCloudPacked → ros_gz_bridge → /registered_scan (10Hz)
  │
  ├── camera (D435 RGB)
  │     └── Image → ros_gz_bridge → /camera/image (30Hz)
  │
  ├── depth_camera (D435 Depth)
  │     └── Image → ros_gz_bridge → /camera/depth (30Hz)
  │
  └── /cmd_vel_nav (Twist) ← ros_gz_bridge ← Go2ROS2Proxy.set_velocity()
        (Note: with rl_quadruped_controller, velocity commands go through
         the controller's command interface, not direct /cmd_vel)

VGG Cognitive Layer
  │
  ├── walk/turn/stop → Go2ROS2Proxy.set_velocity() → /cmd_vel_nav
  ├── navigate_to()  → /goal_point → FAR planner → /way_point
  ├── explore()      → TARE planner → /way_point
  ├── capture_image() → proxy.get_camera_frame() (from /camera/image subscription)
  └── describe_scene() → VLM(camera_frame) → room/object identification
```

---

## 5. Directory Structure

```
vector_os_nano/
├── gazebo/                                    # NEW — Gazebo-specific assets
│   ├── config/
│   │   └── bridge.yaml                        # ros_gz_bridge topic mapping
│   ├── launch/
│   │   └── go2_sim.launch.py                  # ROS2 launch file
│   ├── models/
│   │   └── go2/
│   │       ├── model.urdf.xacro               # Go2 + sensors (from quadruped_ros2_control)
│   │       ├── sensors.xacro                  # Livox MID-360 + D435 definitions
│   │       ├── ros2_control.yaml              # Controller config
│   │       └── meshes/                        # Go2 visual/collision meshes
│   └── worlds/
│       ├── apartment.sdf                      # Primary: 5-room furnished apartment
│       ├── empty_room.sdf                     # Minimal: single room for quick testing
│       └── models/                            # Furniture + object SDF models
│           ├── sofa/
│           ├── table/
│           ├── red_cup/
│           └── ...
├── scripts/
│   ├── launch_gazebo.sh                       # NEW — preflight + start
│   └── stop_gazebo.sh                         # NEW — cleanup
├── vector_os_nano/
│   └── hardware/
│       └── sim/
│           ├── __init__.py                    # MODIFY — add GazeboGo2Proxy import
│           └── gazebo_go2_proxy.py            # NEW — ~40 lines
│   └── vcli/
│       └── tools/
│           └── sim_tool.py                    # MODIFY — add gazebo backend
└── tests/
    └── unit/
        ├── test_gazebo_proxy.py               # NEW — proxy unit tests
        ├── test_gazebo_bridge.py              # NEW — bridge config tests
        ├── test_gazebo_world.py               # NEW — SDF validation tests
        ├── test_gazebo_model.py               # NEW — Go2 model tests
        └── test_gazebo_backend.py             # NEW — CLI backend tests
```

---

## 6. Key Implementation Details

### 6.1 Controller Integration Path

The `rl_quadruped_controller` uses ros2_control hardware interface. The control flow:

1. Gazebo loads Go2 SDF with `<plugin filename="gz_ros2_control-system" ...>`
2. Controller manager starts `rl_quadruped_controller`
3. Controller reads IMU + joint states from hardware interface
4. Controller runs TorchScript policy at 1000 Hz (separate thread)
5. Controller writes joint position targets back to hardware interface
6. Gazebo DART physics applies PD control to achieve targets

For velocity commands from VGG (set_velocity → /cmd_vel_nav):
- The rl_quadruped_controller subscribes to a command topic
- Maps (vx, vy, vyaw) → observation vector cmd field
- Policy outputs joint targets that achieve the velocity

If rl_quadruped_controller is not available (libtorch missing):
- Fall back to `unitree_guide_controller` (no ML deps)
- Uses KDL-based kinematics for basic trot gait

### 6.2 Gazebo Topic Namespacing

Gazebo Harmonic uses `/model/<model_name>/...` namespace by default. The SDF model must set explicit topic names matching our bridge.yaml:

```xml
<!-- In Go2 SDF model -->
<sensor name="lidar" type="gpu_lidar">
  <topic>/lidar/points</topic>
  ...
</sensor>
```

Or use ros_gz_bridge remapping in launch file.

### 6.3 Go2 Spawn Position

Spawn at `(0, 0, 0.35)` in the hallway. z=0.35 is slightly above standing height (0.32) to allow settling. The apartment world hallway center is at (0, 0).

### 6.4 Velocity Command Path (rl_quadruped_controller)

The RL controller receives velocity commands via its command interface:
- ros2_control command: `velocity_command` (3D: vx, vy, vyaw)
- The controller incorporates these into the observation vector `obs[0:3]`
- Policy generates joint targets that achieve the commanded velocity

Our `set_velocity()` publishes to `/cmd_vel_nav` (Twist). The bridge relays to Gazebo. The controller reads from its configured command topic. We may need to align topic names or add a relay node.

---

## 7. Test Strategy

### Unit Tests (per module, TDD)

| Test File | Module | Test Count (est.) | Coverage |
|-----------|--------|-------------------|----------|
| `test_gazebo_proxy.py` | GazeboGo2Proxy | 10-15 | Proxy identity, health check, connect/disconnect, inheritance |
| `test_gazebo_bridge.py` | bridge.yaml | 10-12 | YAML valid, all topic mappings present, types correct |
| `test_gazebo_world.py` | apartment.sdf | 12-15 | SDF valid, rooms, doors, furniture, objects, lighting |
| `test_gazebo_model.py` | Go2 SDF model | 10-12 | SDF valid, joints, sensors (lidar, camera, depth, imu) |
| `test_gazebo_backend.py` | sim_tool.py changes | 8-10 | Backend enum, method exists, correct proxy type |
| `test_gazebo_launch.py` | Launch file | 6-8 | File exists, args declared, nodes present |

**Estimated total: 56-72 new tests**

### Integration Tests (post-wave)

Require running Gazebo (skip in CI, run locally):
- `test_gz_topics_publish.py` — launch Gazebo, verify topics appear within 10s
- `test_gz_proxy_connects.py` — GazeboGo2Proxy.connect() succeeds with live Gazebo
- `test_gz_velocity_moves.py` — set_velocity() → Go2 position changes

### Test Approach

All unit tests mock external dependencies (subprocess, rclpy, gz process). Tests validate:
- Configuration correctness (YAML, SDF, launch file structure)
- Code contracts (proxy interface, backend routing)
- File existence and format validity

No Gazebo process needed for unit tests.

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| libtorch 2.5.0 download + linking | Blocks rl_quadruped_controller | Phase 1: start with unitree_guide_controller. Add RL as enhancement once libtorch is cached. |
| rl_quadruped_controller cmd_vel path unclear | Go2 may not respond to velocity commands | Test with guide_controller first. Debug RL command interface separately. |
| quadruped_ros2_control build complexity | Many packages to build | Build only needed packages: go2_description, gz_quadruped_hardware, one controller |
| Gazebo Fuel model download reliability | World missing furniture at runtime | Bundle critical models locally in gazebo/worlds/models/. Fuel models as enhancement. |
| SDF/URDF sensor topic namespacing | Bridge can't find topics | Set explicit `<topic>` tags in SDF. Test topic flow early. |
| PBR texture quality insufficient for VLM | Room identification fails | Start with solid colors per room (tile=white, carpet=brown, wood=light). Add PBR textures incrementally. |
| Apartment SDF build time | Custom world is labor-intensive | Start with empty_room.sdf (quick validation). Build apartment incrementally. |

---

## 9. Phasing (within this spec's scope)

| Wave | Tasks | Priority | Gate |
|------|-------|----------|------|
| Wave 1 | GazeboGo2Proxy + bridge.yaml + Go2 model + empty_room.sdf + CLI backend | Core infra | Unit tests pass, proxy code works |
| Wave 2 | Launch file + launch/stop scripts | Startup flow | Can launch Gazebo + spawn Go2 via script |
| Wave 3 | apartment.sdf + furniture + objects | World building | SDF validates, Gazebo loads world |
| Wave 4 | Controller integration (guide first, RL after) | Locomotion | Go2 stands and walks in Gazebo |
| Integration | Full pipeline test | End-to-end | vector sim start --backend gazebo works |

---

## 10. Dependencies to Clone/Build

```bash
# 1. Clone quadruped_ros2_control (for Go2 description + controllers)
cd ~/Desktop
git clone https://github.com/legubiao/quadruped_ros2_control.git
cd quadruped_ros2_control

# 2. Build only needed packages
colcon build --packages-select \
  go2_description \
  gz_quadruped_hardware \
  unitree_guide_controller \
  --symlink-install

# 3. (Later) Build RL controller — requires libtorch
# wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.0%2Bcu121.zip
# colcon build --packages-select rl_quadruped_controller
```

All ros-jazzy-ros-gz packages already installed. No new apt installs needed.
