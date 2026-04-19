# Spec: Gazebo Harmonic Integration for Vector OS Nano

**Version**: 1.1
**Author**: Architect (Opus)
**Date**: 2026-04-11
**Status**: DRAFT — awaiting CEO approval

---

## 1. Problem Statement

Vector OS Nano needs a simulation backend that:
- Works natively with ROS2 Jazzy (no Docker, no DDS bridging, no Python version conflicts)
- Provides pre-made complex indoor environments for navigation testing
- Supports the existing nav stack (FAR/TARE) and VGG cognitive layer without code changes
- Enables VLN (Vision-Language Navigation) development with camera feeds
- Leaves room for future loco-manipulation (Go2 + mounted arm)

Isaac Sim proved too complex (two-process architecture, cmd_vel chain broken, Isaac Lab uninstallable). Gazebo Harmonic 8.10.0 is already installed on the system with full ros_gz integration.

## 2. Goal

Add `gazebo` as a third simulation backend alongside `isaac` and `mujoco`. A user types `vector sim start --backend gazebo` and gets Go2 spawned in a furnished indoor world with lidar, camera, and depth publishing standard ROS2 topics — identical to what the nav stack and VGG already consume.

## 3. Non-Goals

- Custom RL training inside Gazebo (MJX is better for this)
- Photorealistic rendering (Gazebo is for fast iteration, not final demo)
- Custom RL policy training (MJX is better; we deploy pre-trained policies)
- Arm integration (Phase 3 — future spec)
- Gazebo Classic (old `gazebo_ros`) compatibility — Gz Sim Harmonic only

## 4. Architecture

```
                      ┌──────────────────────────────────┐
                      │   vector-cli / VGG Cognitive Layer │
                      └──────────┬───────────────────────┘
                                 │ BaseProtocol interface
                      ┌──────────▼───────────────────────┐
                      │       GazeboGo2Proxy              │
                      │   (extends Go2ROS2Proxy)           │
                      └──────────┬───────────────────────┘
                                 │ ROS2 topics
          ┌──────────────────────┼──────────────────────────┐
          │                      │                          │
    ros_gz_bridge          Gz Sim Harmonic           Go2 SDF model
    (topic relay)          (DART physics)            (12 DOF + sensors)
          │                      │                          │
    /state_estimation         Sensors                  SDF World
    /registered_scan          (gz-sensors)             (furnished house)
    /cmd_vel_nav              - GPU lidar
    /camera/image             - RGBD camera
    /camera/depth
```

### 4.1 Key Design Decisions

**D1: Extend Go2ROS2Proxy (not new class)**
Same pattern as IsaacSimProxy. GazeboGo2Proxy is ~30 lines — overrides `name`, `supports_lidar`, and `connect()` with a Gazebo health check. All motion/nav/perception inherited.

**D2: ros_gz_bridge for topic relay**
Gazebo uses gz-transport internally. ros_gz_bridge translates to ROS2 topics. Configured via YAML — maps Gazebo sensor topics to the exact topic names the nav stack expects (`/state_estimation`, `/registered_scan`, `/cmd_vel_nav`, `/camera/image`, `/camera/depth`).

**D3: Go2 model + locomotion controller from quadruped_ros2_control**

legubiao/quadruped_ros2_control (489 stars, Apache 2.0, ROS2 Jazzy) provides:
- `descriptions/unitree/go2_description` -- Go2 URDF with ros2_control hardware interfaces
- `gz_quadruped_hardware` -- Gazebo Harmonic hardware interface plugin
- 4 controllers (analyzed below). We add our own sensor xacro (Livox MID-360 + D435).

**Controller Analysis:**

| Controller | Method | Key Details |
|---|---|---|
| `rl_quadruped_controller` | **RL policy (TorchScript)** | 48-dim obs (gravity, ang_vel, cmd, joint_pos/vel, last_action) -> 12-dim joint position targets -> PD (kp/kd). LibTorch C++ inference at 1000 Hz. Separate RL thread option. Same architecture as our Glowing-Torch policy. |
| `ocs2_quadruped_controller` | **Model Predictive Control** | Pinocchio dynamics + CppAD auto-diff -> trajectory optimization. Trot, flying_trot, stance modes. Most robust. Heavy deps: Pinocchio (custom build), CppAD, ocs2_ros2. First launch compiles shared libs. |
| `unitree_guide_controller` | **FSM + KDL** | State machine (passive/stand/trot). Simplest. Author notes instability vs original. |
| `leg_pd_controller` | **Basic PD** | Just tracks joint targets. No locomotion behavior. |

**Decision: `rl_quadruped_controller` primary, `ocs2_quadruped_controller` fallback.**

RL controller is natural -- same obs/action architecture as our existing go2_jit.pt policy. TorchScript deployment, ros2_control hardware interface, can swap policies without code changes. OCS2 MPC as robust fallback (no trained policy needed, multiple gaits) but heavy dependency (Pinocchio custom build).

**D4: Indoor world -- textured mesh environment for VLN**

MuJoCo cuboid geometry is insufficient for VLN. The world needs textured walls/floors so VLM can distinguish rooms, realistic furniture meshes (not boxes), and visually distinct graspable objects.

Strategy:
1. **Walls + floor plan**: Custom SDF, 5+ rooms, 60m2+, door openings >= 0.8m. PBR materials (wood/tile/carpet).
2. **Furniture**: Gazebo Fuel models (OpenRobotics textured meshes) + 3DGEMS (270+ SDF models).
3. **Graspable objects**: Dynamic Fuel models (cups, cans, bottles) on surfaces with mass/inertia.
4. **Lighting**: Directional + point lights per room. PBR metallic/roughness on key surfaces.

Gazebo Harmonic supports PBR via Ogre 2.2. Not photorealistic, but with textured meshes + proper lighting, sufficient for VLM room/object identification.

**D5: Gazebo process managed by launch file (not subprocess hack)**
Use ROS2 launch file to start Gazebo + spawn Go2 + start bridge. CLI calls `subprocess.Popen(ros2 launch ...)`. Clean shutdown via SIGTERM.

## 5. Components

### 5.1 GazeboGo2Proxy

**File**: `vector_os_nano/hardware/sim/gazebo_go2_proxy.py`

```python
class GazeboGo2Proxy(Go2ROS2Proxy):
    _NODE_NAME = "gazebo_go2_proxy"

    @property
    def name(self) -> str:
        return "gazebo_go2"

    @property
    def supports_lidar(self) -> bool:
        return True

    def connect(self) -> None:
        if not self.is_gazebo_running():
            raise ConnectionError("Gazebo not running")
        super().connect()

    @staticmethod
    def is_gazebo_running() -> bool:
        # Check for /clock topic (Gazebo publishes this)
        ...
```

### 5.2 Go2 SDF Model

**Directory**: `gazebo/models/go2/`

Base Go2 URDF from quadruped_ros2_control, converted to SDF or loaded via `spawn_entity`. Sensor additions:

| Sensor | Type | Mount | Parameters |
|--------|------|-------|------------|
| Livox MID-360 | gpu_lidar | base_link + (0.3, 0, 0.2)m, -20 deg pitch | 360 HFoV, -7/+52 VFoV, 12m range, 10 Hz |
| RealSense D435 RGB | camera | base_link + (0.3, 0, 0.05)m, -5 deg pitch | 640x480, 42 deg FoV, 30 Hz |
| RealSense D435 Depth | depth_camera | co-located with RGB | 640x480, aligned, 30 Hz |
| IMU | imu | base_link center | 200 Hz |

### 5.3 Indoor World SDF

**File**: `gazebo/worlds/apartment.sdf`

Layout (matching MuJoCo house for test parity):
- 5+ rooms: living room, kitchen, bedroom, bathroom, hallway
- 60m2+ total floor area
- Door openings (0.8m wide) between rooms
- Furniture: tables, chairs, shelves, counters, bed, sofa
- 8+ graspable objects placed on furniture surfaces:
  - cups, bottles, phones, books, bowls, remotes, keys, fruit
- Static lighting (uniform, no dynamic shadows needed for nav)
- Ground plane with friction (mu=0.8)

### 5.4 ros_gz_bridge Configuration

**File**: `gazebo/config/bridge.yaml`

Topic mapping:

| Gazebo Topic | ROS2 Topic | Type | Direction |
|---|---|---|---|
| `/model/go2/odometry` | `/state_estimation` | nav_msgs/msg/Odometry | GZ→ROS |
| `/lidar/points` | `/registered_scan` | sensor_msgs/msg/PointCloud2 | GZ→ROS |
| `/camera/image` | `/camera/image` | sensor_msgs/msg/Image | GZ→ROS |
| `/camera/depth` | `/camera/depth` | sensor_msgs/msg/Image | GZ→ROS |
| `/cmd_vel` | `/cmd_vel_nav` | geometry_msgs/msg/Twist | ROS→GZ |
| `/clock` | `/clock` | rosgraph_msgs/msg/Clock | GZ→ROS |
| `/imu` | `/imu/data` | sensor_msgs/msg/Imu | GZ→ROS |

### 5.5 ROS2 Launch File

**File**: `gazebo/launch/go2_sim.launch.py`

Arguments:
- `world` (default: `apartment`) — which world SDF to load
- `gui` (default: `true`) — show Gazebo GUI
- `use_rviz` (default: `false`) — start RViz alongside

Launches:
1. `gz sim <world>.sdf` via ros_gz_sim
2. Spawn Go2 model via `ros_gz_sim create`
3. Start ros_gz_bridge with bridge.yaml
4. Optionally start RViz with pre-configured display

### 5.6 CLI Backend Registration

**File**: `vector_os_nano/vcli/tools/sim_tool.py` (modify)

Add `"gazebo"` to backend enum. Add `_start_gazebo_go2()` static method:
- Launch Gazebo via `subprocess.Popen(ros2 launch gazebo go2_sim.launch.py ...)`
- Wait for `/state_estimation` topic to appear (timeout 30s)
- Create GazeboGo2Proxy, connect, build Agent
- Same pattern as `_start_go2()` but without MuJoCo subprocess

### 5.7 Launch/Stop Scripts

**Files**: `scripts/launch_gazebo.sh`, `scripts/stop_gazebo.sh`

launch_gazebo.sh:
- Preflight: check gz sim installed, check ROS2 sourced
- Accept `--world` and `--headless` args
- Launch via `ros2 launch`
- Wait for topics to appear
- Print connection info

stop_gazebo.sh:
- Kill gz sim processes
- Kill ros_gz_bridge

## 6. Dependencies

### Already Installed
- `ros-jazzy-ros-gz` (bridge, sim, image, interfaces)
- `ros-jazzy-gz-ros2-control`
- `ros-jazzy-gz-sensors-vendor`
- Gazebo Sim 8.10.0

### To Clone (vendored or submodule)
- `legubiao/quadruped_ros2_control` → extract `go2_description` only
- Gazebo Fuel models for furniture (downloaded at build time)

### No New apt Installs Required

## 7. Interface Definitions

### 7.1 ROS2 Topics (consumed by Vector nav stack — no changes needed)

| Topic | Type | Publisher | Subscriber | QoS |
|-------|------|-----------|------------|-----|
| `/state_estimation` | Odometry | ros_gz_bridge | Go2ROS2Proxy, nav stack | RELIABLE |
| `/registered_scan` | PointCloud2 | ros_gz_bridge | FAR, TARE, localPlanner | BEST_EFFORT |
| `/cmd_vel_nav` | Twist | Go2ROS2Proxy | ros_gz_bridge → Gazebo | RELIABLE |
| `/camera/image` | Image | ros_gz_bridge | VLM perception | RELIABLE |
| `/camera/depth` | Image | ros_gz_bridge | depth projection | RELIABLE |
| `/goal_point` | PointStamped | Go2ROS2Proxy | FAR planner | RELIABLE |
| `/way_point` | PointStamped | FAR/TARE | Go2ROS2Proxy, localPlanner | RELIABLE |

### 7.2 Python API (no changes to BaseProtocol)

GazeboGo2Proxy inherits all methods from Go2ROS2Proxy:
- `connect()` / `disconnect()`
- `get_position()` / `get_heading()` / `get_odometry()`
- `get_camera_frame()` / `get_depth_frame()` / `get_rgbd_frame()`
- `set_velocity()` / `walk()` / `stop()` / `stand()` / `sit()`
- `navigate_to()` / `cancel_navigation()` / `stop_navigation()`

## 8. Test Contracts

### T1: GazeboGo2Proxy unit tests
- `test_proxy_inherits_go2ros2proxy` — isinstance(GazeboGo2Proxy(), Go2ROS2Proxy)
- `test_proxy_name` — proxy.name == "gazebo_go2"
- `test_proxy_supports_lidar` — proxy.supports_lidar is True
- `test_is_gazebo_running_no_process` — returns False when gz not running
- `test_is_gazebo_running_with_clock` — returns True when /clock topic exists
- `test_connect_raises_when_not_running` — ConnectionError raised
- `test_connect_delegates_to_parent` — super().connect() called

### T2: Bridge configuration tests
- `test_bridge_yaml_valid` — YAML loads without error
- `test_bridge_maps_state_estimation` — /state_estimation mapping present
- `test_bridge_maps_registered_scan` — /registered_scan mapping present
- `test_bridge_maps_camera` — /camera/image + /camera/depth present
- `test_bridge_maps_cmd_vel` — /cmd_vel_nav ROS→GZ direction
- `test_bridge_topic_types_correct` — all msg types match nav stack expectations

### T3: SDF world validation tests
- `test_apartment_sdf_valid` — gz sdf -k apartment.sdf returns 0
- `test_apartment_has_ground_plane` — ground_plane model present
- `test_apartment_has_walls` — wall models create enclosed rooms
- `test_apartment_has_doors` — door openings >= 0.8m width
- `test_apartment_has_furniture` — >= 5 furniture models
- `test_apartment_has_graspable_objects` — >= 8 small dynamic objects
- `test_apartment_room_count` — >= 5 distinct rooms

### T4: Go2 model tests
- `test_go2_sdf_valid` — model.sdf valid SDF
- `test_go2_has_lidar` — gpu_lidar sensor present
- `test_go2_has_camera` — camera sensor present
- `test_go2_has_depth` — depth_camera sensor present
- `test_go2_has_imu` — imu sensor present
- `test_go2_joint_count` — 12 revolute joints (hip, thigh, calf x4)
- `test_go2_lidar_config` — 360 HFoV, 12m range, mount at (0.3, 0, 0.2)

### T5: Launch file tests
- `test_launch_file_exists` — go2_sim.launch.py exists
- `test_launch_accepts_world_arg` — world argument declared
- `test_launch_accepts_gui_arg` — gui argument declared
- `test_launch_includes_bridge` — ros_gz_bridge node present
- `test_launch_includes_spawn` — spawn_entity action present

### T6: CLI backend integration tests
- `test_sim_tool_accepts_gazebo_backend` — "gazebo" in backend enum
- `test_start_gazebo_go2_method_exists` — _start_gazebo_go2 callable
- `test_gazebo_backend_creates_proxy` — returns Agent with GazeboGo2Proxy base

### T7: Topic compatibility tests (match existing nav stack)
- `test_state_estimation_type` — nav_msgs/Odometry
- `test_registered_scan_type` — sensor_msgs/PointCloud2
- `test_cmd_vel_type` — geometry_msgs/Twist
- `test_camera_image_type` — sensor_msgs/Image
- `test_topic_names_match_go2ros2proxy` — same topics as parent class subscribes to

## 9. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| AC1 | `ros2 launch gazebo go2_sim.launch.py` spawns Go2 in apartment world | Visual: Go2 visible in Gazebo GUI, standing in a room |
| AC2 | `/state_estimation` publishes at >= 20 Hz | `ros2 topic hz /state_estimation` >= 20 |
| AC3 | `/registered_scan` publishes PointCloud2 with >= 1000 points | `ros2 topic echo /registered_scan --once` field count |
| AC4 | `/camera/image` publishes 640x480 RGB frames | `ros2 topic echo /camera/image --once` width=640 height=480 |
| AC5 | `vector sim start --backend gazebo` connects and reports skills | CLI output shows "Started go2 simulation: GazeboGo2Proxy" |
| AC6 | `walk forward 2 seconds` moves Go2 in Gazebo | Position delta > 0.3m after walk command |
| AC7 | All existing tests pass (no regressions) | `pytest tests/ -x` exit code 0, >= 1150 tests |
| AC8 | Apartment world has >= 5 rooms with furniture and graspable objects | Visual inspection + SDF validation |
| AC9 | VLM can describe Gazebo camera frame | `describe what you see` returns room/furniture description |

## 10. Risks

| Risk | Mitigation |
|------|-----------|
| libtorch 2.5.0 C++ dependency for RL controller | Pre-download and cache; fallback to ocs2_quadruped_controller (MPC, no ML deps) |
| RL policy obs/action format mismatch with go2_jit.pt | Verify joint ordering + observation scaling match Glowing-Torch convention; write adapter if needed |
| Pinocchio custom build for OCS2 fallback | Document exact build steps; make OCS2 optional (not required for Phase 1) |
| ros_gz_bridge latency on pointcloud | Use BEST_EFFORT QoS, reduce scan rate to 10 Hz if needed |
| Go2 URDF from quadruped_ros2_control may need sensor additions | We add our own sensor xacro on top of base URDF |
| Gazebo rendering insufficient for VLM room identification | Use PBR textured meshes + proper lighting; VLM needs object/room distinction, not photorealism |
| Fuel furniture models vary in quality | Curate a fixed set of tested models; include fallback primitive meshes with textures |

## 11. Phasing

| Phase | Scope | Gate |
|-------|-------|------|
| Phase 1 (this spec) | Gazebo + Go2 + apartment world + bridge + CLI | AC1-AC9 |
| Phase 2 (future) | VLN pipeline (semantic labels, instruction following) | Separate spec |
| Phase 3 (future) | Loco-manipulation (arm mount, grasping) | Separate spec |
