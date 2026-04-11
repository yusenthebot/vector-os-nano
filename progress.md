# Vector OS Nano SDK — Progress

**Last updated:** 2026-04-11
**Version:** v1.6.0-dev
**Branch:** feat/web-viz

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

## Isaac Sim 集成 (2026-04-10) — 默认仿真后端

Isaac Sim 5.1.0 (Docker) 作为主仿真后端。光追渲染 + RTX 传感器 + 复杂室内环境。

**状态: 已验证运行** — GUI 可视化 + ROS2 topic 50Hz 稳定 + 传感器已附加。

```
Host (Ubuntu 24.04 + ROS2 Jazzy)
  vector-cli → VGG → IsaacSimProxy (BaseProtocol)
       ↕ DDS (CycloneDDS, --network host, same Jazzy!)
Docker (Isaac Sim 5.1 + Ubuntu 24.04 + ROS2 Jazzy)
  isaac_sim_physics.py → PhysX + sensors → shared state files
  ros2_publisher.py    → reads state → publishes ROS2 topics
```

**关键架构: 两进程**
- Isaac Sim Python 3.11 (物理 + 传感器) — 写状态到 /tmp/isaac_state/
- 系统 Python 3.12 (ROS2 Jazzy) — 读状态发布 topic
- 原因: Isaac Sim 内置 Python 3.11 与 ROS2 Jazzy 的 rclpy (Python 3.12) C extension 不兼容

### 架构

| 组件 | 文件 | 说明 |
|------|------|------|
| Docker | `docker/isaac-sim/` | Dockerfile + compose + CycloneDDS + entrypoint |
| Bridge | `docker/isaac-sim/bridge/isaac_sim_bridge.py` | ROS2 node: odom/tf/joints/scan/camera/joy/speed |
| 场景 | `docker/isaac-sim/bridge/go2_scene.py` | 6 场景: flat/room/apartment/navigation/hospital |
| 传感器 | `docker/isaac-sim/bridge/go2_sensors.py` | Livox MID-360 + D435 配置 |
| Lidar 配置 | `docker/isaac-sim/bridge/lidar_configs/Livox_MID360.json` | 自定义 RTX lidar |
| Host Proxy | `vector_os_nano/hardware/sim/isaac_sim_proxy.py` | BaseProtocol (继承 Go2ROS2Proxy) |
| Arm Proxy | `vector_os_nano/hardware/sim/isaac_sim_arm_proxy.py` | ArmProtocol via ROS2 |
| CLI | `vector_os_nano/vcli/tools/sim_tool.py` | `backend=isaac` 默认 |
| 启动 | `scripts/launch_isaac.sh` / `stop_isaac.sh` | 一键启动/停止 |

### 传感器配置 (匹配 MuJoCo)

| 传感器 | 挂载 | 参数 |
|--------|------|------|
| Livox MID-360 | base_link + (0.3, 0, 0.2)m, -20 deg | VFoV -7~+52 deg, 360 deg HFoV, 30 rings, 12m range |
| RealSense D435 | base_link + (0.3, 0, 0.05)m, -5 deg | 640x480, FoV 42 deg, RGB + depth aligned |

### 场景

| 场景 | 说明 | 用途 |
|------|------|------|
| flat | 平地 + Go2 | 基础测试 |
| room | 4x5m 单房间 + 桌椅 | 简单导航 |
| apartment | 3 房间 + 门 | 多房间导航 |
| navigation | 60m2 五房间公寓 (走廊+客厅+厨房+卧室+浴室, 8+ 可抓取物体) | 完整导航 + 操作测试 |
| hospital | Isaac Sim 内置医院 | 最复杂导航 |

### 测试: 263 passed, 51 skipped, 0 failed

| 文件 | 数量 | 覆盖 |
|------|------|------|
| test_isaac_sim_proxy.py | 62 | Protocol, Docker 检查, 状态, 运动, 导航 |
| test_isaac_arm_proxy.py | 51 skip | TDD stubs |
| test_docker_config.py | 49 | Dockerfile, compose, DDS, 脚本 |
| test_topic_compat.py | 44 | Topic 名/类型/QoS/传感器匹配 |
| test_backend_switch.py | 45 | CLI backend 路由, 兼容性 |
| test_isaac_e2e_chain.py | 63 | 全链路: Docker→DDS→Proxy→Primitives→VGG |

### 启动方式

```bash
# 构建 (首次)
docker build --network host -t vector-isaac-sim:latest docker/isaac-sim/

# 启动
ISAAC_SCENE=navigation docker compose -f docker/isaac-sim/docker-compose.yaml up

# 验证
ros2 topic list | grep state_estimation

# vector-cli 连接
vector sim start    # 默认 isaac 后端
```

### 已验证

- [x] Docker 构建成功 (Isaac Sim 5.1.0 + ROS2 Jazzy)
- [x] GPU passthrough (RTX 5080, driver 580.126.20, CUDA 13.0)
- [x] Isaac Sim GUI 可视化 (ISAAC_HEADLESS=false)
- [x] 物理循环运行 (200 Hz PhysX)
- [x] Livox MID-360 RTX lidar 附加
- [x] RealSense D435 RGB+Depth 相机附加
- [x] ROS2 topic 全部发布 (12 topics, /state_estimation 50 Hz)
- [x] Host ROS2 Jazzy 可见容器内 topic (DDS 直通)
- [x] 真实 Go2 USD 模型从 Nucleus 加载 (12 DOF articulation)
- [x] RL locomotion policy 加载 (JIT, 45-dim obs, 12-dim action)
- [x] Go2 站立稳定 (z=0.319, headless 验证)
- [x] Go2 前进 (dx=0.47m/3s, headless 验证)
- [x] 关节映射 Lab↔Sim 验证正确

### 待完成

- [ ] cmd_vel → RL policy 传递链路 debug (ROS2 → shared file → physics)
- [ ] 键盘遥控 debug (Isaac Sim 5.1 keyboard event API)
- [ ] Isaac Lab 安装 (isaaclab.sh install 需要修 setuptools/pip)
- [ ] 自训练 Go2 policy (Isaac Lab + RSL-RL)
- [ ] Wire RTX lidar annotator → real scan data
- [ ] Wire RenderProduct camera → real image data
- [ ] 连接 Vector nav stack (FAR/TARE)
- [ ] 开发模式: 挂载代码 volume 避免 rebuild

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
