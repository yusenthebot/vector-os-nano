# Vector OS Nano SDK — Progress

**Last updated:** 2026-04-11
**Version:** v1.7.0-dev
**Branch:** sim-with-gazebo

## VGG: Verified Goal Graph — Complete Framework

Cognitive layer — ALL actionable commands flow through VGG. LLM decomposes complex tasks into verifiable sub-goal trees. Simple commands get 1-step GoalTrees without LLM call.

```
User input
  ↓
should_use_vgg?
  ├─ Action → VGG
  │    ├─ Simple (skill match) → 1-step GoalTree (fast, no LLM)
  │    └─ Complex (multi-step) → LLM decomposition → GoalTree
  │    ↓
  │  VGG Harness: 3-layer feedback loop
  │    Layer 1: step retry (alt strategies)
  │    Layer 2: continue past failure
  │    Layer 3: re-plan with failure context
  │    ↓
  │  GoalExecutor → verify → trace → stats
  │
  └─ Conversation → tool_use path (LLM direct)
```

### Cognitive Layer (vcli/cognitive/)

| Component | Purpose |
|-----------|---------|
| GoalDecomposer | LLM → GoalTree; template + skill fast path |
| GoalVerifier | Safe sandbox for verify expressions |
| StrategySelector | Rule + stats-driven strategy selection |
| GoalExecutor | Execute + verify + fallback + stats recording |
| VGGHarness | 3-layer feedback loop (retry → continue → re-plan) |
| CodeExecutor | RestrictedPython sandbox (velocity clamped) |
| StrategyStats | Persistent success rate tracking |
| ExperienceCompiler | Traces → parameterized templates |
| TemplateLibrary | Store + match + instantiate templates |
| ObjectMemory | Time-aware object tracking with exponential confidence decay |
| predict | Rule-based state prediction from room topology |
| VisualVerifier | VLM-based visual verification fallback |

### Primitives API (vcli/primitives/)

30 functions across 4 categories:
- **locomotion** (8): get_position, get_heading, walk_forward, turn, stop, stand, sit, set_velocity
- **navigation** (5): nearest_room, publish_goal, wait_until_near, get_door_chain, navigate_to_room
- **perception** (6): capture_image, describe_scene, detect_objects, identify_room, measure_distance, scan_360
- **world** (11): query_rooms, query_doors, query_objects, get_visited_rooms, path_between, world_stats, last_seen, certainty, find_object, objects_in_room, room_coverage

### CLI Integration

- Async execution — CLI never blocks during navigation/explore
- GoalTree plan shown before execution
- Step-by-step [idx/total] progress feedback
- VGG only active after sim start (requires functioning robot)

Design spec: `docs/vgg-design-spec.md`

## Sensor Configuration

- **Lidar**: Livox MID-360, -20 deg downward tilt (match real Go2)
- **Terrain Analysis**: VFoV -30/+35 deg (matched to MID-360)
- **VLM**: OpenRouter (google/gemma-4-31b-it)
- **Ceiling filter**: points > 1.8m filtered from /registered_scan (fixes V-Graph)

## Navigation Pipeline

```
Explore: TARE → room detection → SceneGraph doors
Navigate: FAR V-Graph → door-chain fallback (nav stack waypoints)
Path follower: TRACK/TURN modes, cylinder body safety
Stuck recovery: boxed-in detection → 3-4s sustained reverse
```

## Vector CLI

```bash
vector                    # Interactive REPL (VGG cognitive layer)
vector go2 stand          # One-shot Go2 commands
vector sim start          # Simulation lifecycle
vector ros nodes          # ROS2 diagnostics
vector chat               # LLM agent mode
```

## 备注：开发测试流程

所有测试和验证只通过 vector-cli 启动，直接对话交互。不单独脚本调用 MuJoCo/ROS2。

## Test Coverage: 630+ VGG tests, 1150+ total

| Suite | Tests | Status |
|-------|-------|--------|
| Locomotion L0-L4 | 26 | pass |
| Agent+Go2 | 5 | pass |
| VLM+Scene L0-L9 | 200+ | pass |
| Nav L17-L33 | 247 | pass |
| Sim-to-Real L34-L38 | 120+ | pass |
| Nav fixes L39-L40 | 27 | pass |
| VGG Phase 1 L41-L46 | 187 | pass |
| VGG Phase 2 L47-L50 | 87 | pass |
| VGG CLI L51 | 25 | pass |
| Door-chain L52 | 18 | pass |
| Ceiling filter L53 | 21 | pass |
| VGG Integration L54 | 29 | pass |
| CLI Scenarios L55 | 52 | pass |
| VGG Harness L56 | 24 | pass |
| ObjectMemory L57 | 39 | pass |
| predict L58 | 35 | pass |
| VisualVerifier L59 | 28 | pass |
| Namespace Integration L60 | 21 | pass |
| Auto-Observe L61 | 36 | pass |
| Other | 80+ | pass |

## Phase 3: Active World Model

```
ObjectMemory: SceneGraph → TrackedObject (指数衰减: conf * exp(-0.001 * elapsed))
  ↓
GoalVerifier namespace: last_seen(), certainty(), find_object(), objects_in_room(), room_coverage(), predict_navigation()
  ↓
VisualVerifier: verify 失败 → VLM 拍照二次确认 (感知步骤才触发)
  ↓
Auto-Observe: 探索时每个新 viewpoint → VLM 自动识别物体 → SceneGraph + ObjectMemory
```

## Foxglove 可视化 — feat/web-viz

Foxglove Studio + foxglove_bridge 替代 RViz (ADR-004 revised)。
Three.js 自建方案已废弃 — 开发成本高、效果调试困难。

```
ROS2 Topics → foxglove_bridge (ws://8765) → Foxglove Studio
  Dashboard: foxglove/vector-os-dashboard.json
  6 面板: 3D 透视 + 3D 俯视 + 摄像头 + SceneGraph JSON + Raw Messages + 速度曲线
  14 topic: registered_scan(turbo) + free_paths + path + markers + camera + scene_graph JSON...
  启动: vector CLI → "打开可视化" 或 ./foxglove/launch_foxglove.sh
  CLI tool: open_foxglove (start/stop/status)
  JSON topic: /vector_os/scene_graph (0.5Hz, rooms+doors+objects+stats)
```

已知限制: V-Graph 线段在 Foxglove 中只能显示为散点或很细的 marker（PointCloud2 无法渲染为 LineSegments，MarkerArray 线宽由 FAR 源码决定）。MuJoCo sim 的点云密度和视觉效果不如真实 LiDAR。

## Gazebo Harmonic 集成 (2026-04-11) — 主仿真后端

Gazebo Sim 8.10.0 (Harmonic) 原生 ROS2 Jazzy，无 Docker。Go2 URDF 来自 quadruped_ros2_control (489 stars)。

**状态: 运行中** — 3 controllers active, 传感器 topic 全部发布。

```
Host (Ubuntu 24.04 + ROS2 Jazzy)
  vector-cli → VGG → GazeboGo2Proxy (BaseProtocol)
       │ ROS2 topics (native, no Docker)
  Gz Sim Harmonic (DART physics)
    Go2 URDF (12 DOF) + unitree_guide_controller
    MID-360 gpu_lidar + D435 RGB/Depth + IMU
    ros_gz_bridge → ROS2 topics
```

### 架构

| 组件 | 文件 | 说明 |
|------|------|------|
| Proxy | `vector_os_nano/hardware/sim/gazebo_go2_proxy.py` | BaseProtocol (继承 Go2ROS2Proxy) |
| Go2 模型 | `gazebo/models/go2/` | SDF + sensors.xacro + ros2_control.yaml |
| 传感器 | `gazebo/models/go2/sensors.xacro` | MID-360 + D435 (注入 Go2 URDF) |
| Bridge | `gazebo/config/bridge.yaml` | ros_gz_bridge topic 映射 |
| Launch | `gazebo/launch/go2_sim.launch.py` | Gz + spawn + bridge + controllers |
| 世界 | `gazebo/worlds/apartment.sdf` | 5 房间公寓 + 14 家具 + 8 可抓取物体 |
| CLI | `vector_os_nano/vcli/tools/sim_tool.py` | `backend=gazebo` |
| 脚本 | `scripts/launch_gazebo.sh` / `stop_gazebo.sh` | 一键启动/停止 |
| 控制器 | `~/Desktop/quadruped_ros2_control/` | unitree_guide_controller (FSM) |

### 传感器配置

| 传感器 | 挂载 | 参数 | 实测 |
|--------|------|------|------|
| Livox MID-360 | trunk + (0.15, 0, 0.15)m, -20 deg | 360x30, 0.1-12m, 10Hz | 8 Hz, 10800 pts |
| RealSense D435 RGB | trunk + (0.27, 0, 0.12)m, +5 deg | 640x480, 30Hz | 22 Hz, rgb8 |
| RealSense D435 Depth | co-located | 640x480, 0.3-10m | publishing |
| IMU | trunk center | 200Hz | publishing |

### 世界

| 世界 | 说明 | 用途 |
|------|------|------|
| empty_room | 5x6m 单房间 | 快速验证 |
| apartment | 65m2 五房间 (客厅+厨房+卧室+浴室+走廊, 14 家具, 8 可抓取物体) | 导航 + VLN |

### 测试: 120 passed, 0 failed

| 文件 | 数量 | 覆盖 |
|------|------|------|
| test_gazebo_proxy.py | 21 | Proxy identity, health check, lifecycle |
| test_gazebo_bridge.py | 12 | YAML 有效性, topic 映射 |
| test_gazebo_model.py | 15 | SDF 有效性, joints, sensors |
| test_gazebo_launch.py | 16 | Launch file 结构 |
| test_gazebo_scripts.py | 14 | Shell 脚本有效性 |
| test_gazebo_backend.py | 6 | CLI backend 路由 |
| test_gazebo_world.py | 23 | empty_room + apartment SDF |
| test_gazebo_controller.py | 13 | ros2_control config |

### 启动方式

```bash
# 依赖 (首次)
cd ~/Desktop/quadruped_ros2_control
colcon build --packages-up-to unitree_guide_controller --symlink-install

# 启动
source /opt/ros/jazzy/setup.bash
source ~/Desktop/quadruped_ros2_control/install/setup.bash
ros2 launch gazebo/launch/go2_sim.launch.py world:=apartment gui:=true

# 或一键脚本
bash scripts/launch_gazebo.sh --world apartment
bash scripts/stop_gazebo.sh

# vector-cli 连接
vector sim start --backend gazebo
```

### 已验证

- [x] Gz Sim 8.10.0 启动 (无 Docker)
- [x] Go2 URDF 从 quadruped_ros2_control 加载 (12 DOF)
- [x] unitree_guide_controller active
- [x] joint_state_broadcaster active (887 Hz)
- [x] imu_sensor_broadcaster active
- [x] MID-360 gpu_lidar 发布 /registered_scan (8 Hz, 10800 pts)
- [x] D435 RGB 发布 /camera/image (22 Hz, 640x480 rgb8)
- [x] D435 Depth 发布 /camera/depth
- [x] IMU 发布 /imu/data
- [x] apartment.sdf 加载 (5 rooms, 14 furniture, 8 objects)
- [x] gz sdf -k apartment.sdf → Valid

### 待完成

- [ ] CLI `vector sim start --backend gazebo` 实际行走测试
- [ ] rl_quadruped_controller 集成 (需要 libtorch 2.5.0)
- [ ] 连接 Vector nav stack (FAR/TARE) — topic 已对齐
- [ ] VLM 感知测试 (camera frame → room identification)

## Isaac Sim 集成 (2026-04-10) — 已归档

归档在 `web-viz-Isaac-sim` 分支。Isaac Sim 5.1.0 Docker 可运行但 cmd_vel 链路未通。
详见归档分支代码和 `docker/isaac-sim/`。

## FAR V-Graph 调参记录 (2026-04-09)

根本原因: `new_intensity_thred: 2.0` — bridge 的 intensity = height above ground，ceiling filter 1.8m 导致所有点 < 2.0，FAR 不认为有"新"障碍物 → V-Graph 不建。改为 0.5 后数据层有改善。

调参尝试:
- `is_static_env: true` → 让 FAR 行为更差（obs 归零），不要用
- 降低 `connect_votes_size` 10→3, `node_finalize_thred` 6→3 → 效果不明显
- 增大 `dynamic_obs_dacay_time` 2→30, `new_points_decay_time` 1→15 → 效果更差

最终配置: 只改 `new_intensity_thred: 0.5`，其余保持原始值。

实际状态: FAR 数据层在工作 — `/free_paths` 18k-60k 点持续发布，`/global_path` 5-58 poses。V-Graph edges 存在但 RViz 看不到（PointCloud2 用 2px Points 渲染，不是线段）。

结论: 不是 FAR 不工作，是 RViz 不适合显示 V-Graph。→ 自建 Web 可视化。

## Known Limitations

- VGG complex decomposition quality depends on LLM model
- Async skills (explore, patrol) report "launched" not "completed" in VGG
- Real-world room detection needs SLAM + spatial understanding
