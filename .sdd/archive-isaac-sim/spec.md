# Spec: Isaac Sim 集成 — 高保真仿真后端

**Version:** 0.1
**Date:** 2026-04-10
**Status:** Approved + Executed (Wave 1 done, Go2 walking verified)
**Author:** Architect (Opus)

---

## 一句话

在 Docker 中运行 Isaac Sim 5.1，通过 DDS 桥接到 Jazzy host，为 Vector OS Nano 提供光追级视觉仿真 + 复杂室内环境 + 机械臂操作测试 + Go2 locomotion RL 训练。

---

## 背景

MuJoCo 仿真的局限已成为瓶颈：
- **视觉质量差** — OpenGL 渲染，物体只有简单几何体 + 纯色材质，无法做 demo 或视觉 sim-to-real
- **环境简单** — go2_room.xml 只有几面墙和基本障碍物，没有家具、电器、日常物品
- **传感器仿真弱** — 点云通过 mj_ray 模拟，无 RTX lidar，无真实相机噪声模型
- **无 GPU 并行** — 单环境 CPU 物理，无法做大规模 RL 训练

Isaac Sim 5.1 解决以上全部问题：RTX 光追渲染、USD 资产生态、GPU PhysX、OmniGraph ROS2 Bridge。

---

## 目标

1. **视觉仿真优先** — 在 Isaac Sim 中搭建真实感室内环境（客厅/厨房/卧室），用于 demo、manipulation 测试、VLM 感知验证
2. **Go2 导航集成** — Isaac Sim Go2 通过 ROS2 topic 接入 Vector nav stack (FAR/TARE/pathFollower)
3. **机械臂操作** — SO-101 或等效 arm 在 Isaac Sim 中做 pick/place，photorealistic 物体识别
4. **RL 训练能力** — Isaac Lab Go2 locomotion policy 训练（rough terrain, domain randomization）
5. **与现有系统无缝切换** — 通过 BaseProtocol/ArmProtocol，CLI 一个参数切换 MuJoCo / Isaac Sim

## 非目标

- 替代 MuJoCo — MuJoCo 继续用于快速迭代、单元测试、CI（轻量级）
- 修改 VGG 认知层 — VGG 通过 Protocol 接口操作机器人，不感知具体后端
- Isaac Sim 原生安装 — 只用 Docker 容器，不污染 host 环境
- 实时 sim-to-real 部署 — 训练好的 policy 导出 ONNX 后再做部署（本 spec 不涵盖部署）

---

## 架构

```
┌─────────────────────────────────────────────────────┐
│  Host: Ubuntu 24.04 + ROS2 Jazzy                    │
│                                                      │
│  vector-cli ─── VGG ─── Primitives                  │
│       │                      │                       │
│       ▼                      ▼                       │
│  IsaacSimProxy (BaseProtocol / ArmProtocol)          │
│       │                                              │
│       ▼  DDS (CycloneDDS / FastDDS, same domain)     │
├───────┼──────────────────────────────────────────────┤
│       ▼                                              │
│  Docker: Isaac Sim 5.1 + ROS2 Humble                 │
│                                                      │
│  isaac_sim_bridge.py (ROS2 Humble node)              │
│    ├── Pub: /state_estimation (Odometry, 50Hz)       │
│    ├── Pub: /registered_scan (PointCloud2, 10Hz)     │
│    ├── Pub: /camera/image (Image, 30Hz)              │
│    ├── Pub: /camera/depth (Image, 30Hz)              │
│    ├── Pub: /tf (odom→base_link, 50Hz)               │
│    ├── Sub: /cmd_vel (Twist)                         │
│    └── Sub: /joint_commands (Float64MultiArray)      │
│                                                      │
│  OmniGraph: Isaac Sim sensors → ROS2 topics          │
│  USD Scene: 室内环境 + Go2 + arm + 可抓取物体         │
└──────────────────────────────────────────────────────┘
```

### DDS 跨版本通信

Humble 和 Jazzy 的 DDS wire protocol 是兼容的。两侧使用相同 DDS 实现 + domain ID 即可互通：
- Docker 内: CycloneDDS (Isaac Sim 默认)
- Host: CycloneDDS (ROS2 Jazzy 默认)
- `ROS_DOMAIN_ID=0` 两侧一致
- Docker 网络: `--network host` 消除网络隔离

---

## 功能

### F1: Docker 环境 — `docker/isaac-sim/`

Isaac Sim 5.1 容器化环境，一键构建 + 启动。

**Dockerfile:**
- 基于 `nvcr.io/nvidia/isaac-sim:4.5.0` (Isaac Sim 5.1 对应的容器 tag)
- 安装 ROS2 Humble (ros-humble-ros-base + bridge 包)
- 安装 CycloneDDS
- 复制 Vector OS Nano 的 USD 场景、bridge 脚本、配置
- ENTRYPOINT: 启动 isaac_sim_bridge.py

**docker-compose.yaml:**
- GPU passthrough (`runtime: nvidia`, `NVIDIA_VISIBLE_DEVICES=all`)
- `network_mode: host` (DDS 直通)
- Volume mounts: USD 场景、配置、log 目录
- 环境变量: `ROS_DOMAIN_ID`, `FASTRTPS_DEFAULT_PROFILES_FILE`

**启动脚本 `scripts/launch_isaac.sh`:**
```bash
docker compose -f docker/isaac-sim/docker-compose.yaml up -d
# 等待 Isaac Sim 初始化 (首次 ~5min, 后续 ~30s)
# 健康检查: ROS2 topic list 包含 /state_estimation
```

**验收标准:**
- [ ] `./scripts/launch_isaac.sh` 一键启动 Isaac Sim 容器
- [ ] Host 上 `ros2 topic list` 能看到 Isaac Sim 发布的 topic
- [ ] `ros2 topic echo /state_estimation` 收到 Odometry 消息
- [ ] `docker compose down` 干净关闭
- [ ] RTX 5080 GPU 正确 passthrough (nvidia-smi 容器内可见)

### F2: Isaac Sim ROS2 Bridge — `isaac_sim_bridge.py`

在容器内运行的 ROS2 Humble node，将 Isaac Sim 仿真数据发布为标准 ROS2 topic。

**发布 Topic (与现有 MuJoCo bridge 兼容):**

| Topic | Type | Rate | 说明 |
|-------|------|------|------|
| `/state_estimation` | nav_msgs/Odometry | 50 Hz | Go2 位姿 + 速度 |
| `/registered_scan` | sensor_msgs/PointCloud2 | 10 Hz | RTX lidar 点云 |
| `/camera/image` | sensor_msgs/Image | 30 Hz | RGB 相机 |
| `/camera/depth` | sensor_msgs/Image | 30 Hz | 深度图 |
| `/camera/camera_info` | sensor_msgs/CameraInfo | 30 Hz | 相机内参 |
| `/tf` | tf2_msgs/TFMessage | 50 Hz | odom→base_link |
| `/joint_states` | sensor_msgs/JointState | 50 Hz | 12 DOF 关节状态 |

**订阅 Topic:**

| Topic | Type | 说明 |
|-------|------|------|
| `/cmd_vel` | geometry_msgs/Twist | 速度命令 (nav stack) |
| `/cmd_vel_nav` | geometry_msgs/Twist | Agent 速度命令 |
| `/joint_commands` | std_msgs/Float64MultiArray | 机械臂关节命令 |

**验收标准:**
- [ ] Topic 名称和消息类型与 MuJoCo bridge 完全一致 (drop-in)
- [ ] Odometry 频率 >= 40 Hz，延迟 < 50ms
- [ ] PointCloud2 包含 >= 10000 点/帧
- [ ] 相机图像分辨率 >= 640x480, RGB
- [ ] /cmd_vel 控制 Go2 运动，响应延迟 < 100ms

### F3: IsaacSimProxy — Host 侧 Protocol 实现

在 host (Jazzy) 上运行的 Python 类，实现 `BaseProtocol`，通过 ROS2 topic 控制 Docker 中的 Isaac Sim Go2。

**文件:** `vector_os_nano/hardware/sim/isaac_sim_proxy.py`

```python
class IsaacSimProxy:
    """BaseProtocol implementation for Isaac Sim Go2 (via ROS2 DDS).

    Same pattern as Go2ROS2Proxy but for Isaac Sim backend.
    Topic names identical — difference is the Docker-side publisher.
    """
    name = "isaac_go2"

    def connect(self) -> None:
        # Create rclpy node, subscribe /state_estimation, /camera/*, /registered_scan
        # Publish /cmd_vel_nav, /goal_point, /way_point
        # Wait for first odom message (Isaac Sim ready)

    def set_velocity(self, vx, vy, vyaw) -> None:
        # Publish Twist to /cmd_vel_nav

    def get_position(self) -> list[float]:
        # Return cached position from /state_estimation

    def get_odometry(self) -> Odometry:
        # Return cached odom

    def get_lidar_scan(self) -> LaserScan | None:
        # Convert cached PointCloud2 → LaserScan
```

**关键设计: 与 Go2ROS2Proxy 共享逻辑**

IsaacSimProxy 和 Go2ROS2Proxy 订阅相同 topic、发布相同 topic。区别仅在于：
- Go2ROS2Proxy: MuJoCo 进程发布 topic
- IsaacSimProxy: Isaac Sim Docker 发布 topic

可以复用同一个类，通过 name 属性区分。或者直接复用 `Go2ROS2Proxy`，只改 name。

**验收标准:**
- [ ] `IsaacSimProxy` 实现完整 `BaseProtocol`
- [ ] `isinstance(proxy, BaseProtocol)` 返回 True
- [ ] 所有 locomotion primitives 通过 Isaac Sim 执行成功
- [ ] 现有 Go2 skill 不需要任何修改即可在 Isaac Sim 上运行

### F4: IsaacSimArmProxy — 机械臂 Protocol 实现

**文件:** `vector_os_nano/hardware/sim/isaac_sim_arm_proxy.py`

```python
class IsaacSimArmProxy:
    """ArmProtocol implementation for arm in Isaac Sim.

    Controls arm via /joint_commands topic.
    Reads joint state from /arm/joint_states.
    """
    name = "isaac_arm"
    dof = 6
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
```

机械臂 topic:

| Topic | Type | 方向 | 说明 |
|-------|------|------|------|
| `/arm/joint_states` | JointState | Pub (from Isaac) | 关节位置/速度/力 |
| `/arm/joint_commands` | Float64MultiArray | Sub (to Isaac) | 目标关节角度 |
| `/arm/end_effector_pose` | PoseStamped | Pub (from Isaac) | 末端位姿 |

IK 计算: 在 host 侧用 pinocchio/casadi (已有 .venv-nano 中)，或者在 Isaac Sim 内用 Lula IK。

**验收标准:**
- [ ] `IsaacSimArmProxy` 实现完整 `ArmProtocol`
- [ ] pick/place skill 在 Isaac Sim 中成功执行
- [ ] 物体被抓取后位置正确更新
- [ ] IK 求解成功率 >= 95%（标准测试物体位置）

### F5: USD 室内场景 — `isaac_sim/scenes/`

预构建的 photorealistic 室内环境 USD 文件。

**场景 1: 单房间 (apartment_single.usd)**
- 4m x 5m 客厅
- 沙发、茶几、电视柜、书架
- 桌上有可抓取物体: 杯子、遥控器、书、水瓶
- PBR 材质、area light、window 天光
- Go2 spawn point + arm 工作台

**场景 2: 多房间 (apartment_multi.usd)**
- 3 房间: 客厅 + 厨房 + 卧室
- 门连接 (与 SceneGraph door 模型一致)
- 每个房间 3-5 个可交互物体
- Go2 可在房间间自由导航

**场景 3: 操作台 (manipulation_bench.usd)**
- 桌面 + 机械臂
- 10 种标准测试物体 (不同形状、大小、材质)
- 固定相机 + 可移动相机
- 用于 manipulation 回归测试

**资产来源:**
- NVIDIA Isaac Sim 内置资产 (Nucleus)
- USD 社区资产 (兼容 CC0/CC-BY)

**验收标准:**
- [ ] 3 个场景在 Isaac Sim 中加载并渲染 (>= 30 FPS)
- [ ] 物体可抓取 (rigid body + 碰撞体)
- [ ] Go2 可在环境中行走不穿模
- [ ] 相机渲染质量满足 VLM 识别需求
- [ ] 场景文件总大小 < 500 MB

### F6: CLI 集成 — `vector sim start --backend isaac`

在 vector-cli 中支持 Isaac Sim 后端选择。

**SimStartTool 扩展:**
```python
@tool(name="start_simulation")
class SimStartTool:
    input_schema = {
        "properties": {
            "sim_type": {"enum": ["arm", "go2"]},
            "backend": {
                "type": "string",
                "enum": ["mujoco", "isaac"],
                "default": "mujoco",
                "description": "Simulation backend: 'mujoco' (fast, lightweight) or 'isaac' (photorealistic)",
            },
            # ...
        }
    }
```

**CLI 命令:**
```bash
vector sim start --backend isaac          # Isaac Sim Go2 (Docker must be running)
vector sim start --backend mujoco         # MuJoCo Go2 (default, unchanged)
vector sim start --backend isaac --arm    # Isaac Sim arm manipulation
```

**验收标准:**
- [ ] `vector sim start --backend isaac` 连接到运行中的 Isaac Sim Docker
- [ ] Isaac Sim 未启动时报清晰错误 ("Isaac Sim container not running")
- [ ] 切换 backend 不需要重启 vector-cli
- [ ] VGG 在 Isaac Sim 后端上正常工作 (decompose + execute + verify)

### F7: Nav Stack 集成

确保 Vector nav stack (FAR/TARE/pathFollower) 能通过 Isaac Sim 的 topic 正常工作。

**关键: Topic 兼容性**

Isaac Sim bridge 发布的 topic 名称和消息类型与 MuJoCo bridge 完全一致。Nav stack 不感知后端差异。

**额外需要的 topic:**
- `/joy` (sensor_msgs/Joy) — pathFollower 要求的 fake joystick，Isaac bridge 需要发布
- `/speed` (std_msgs/Float32) — pathFollower 速度参数

**launch_explore_isaac.sh:**
```bash
# 1. 确认 Isaac Sim Docker 正在运行
# 2. 启动 host 侧 nav stack (与 launch_explore.sh 相同的 FAR/TARE/pathFollower)
# 3. 连接 IsaacSimProxy
```

**验收标准:**
- [ ] TARE exploration 在 Isaac Sim 多房间环境中正常工作
- [ ] FAR planner 生成有效路径 (/free_paths, /global_path)
- [ ] pathFollower 控制 Isaac Sim Go2 沿路径行走
- [ ] SceneGraph 在 Isaac Sim 环境中正确构建 (room/door/object detection)

### F8: 测试策略

**单元测试 (不需要 Docker/Isaac Sim):**

| 测试文件 | 测试内容 | 数量 |
|----------|---------|------|
| test_isaac_sim_proxy.py | IsaacSimProxy Protocol 合规 (mock ROS2) | 25+ |
| test_isaac_arm_proxy.py | IsaacSimArmProxy Protocol 合规 | 20+ |
| test_docker_config.py | Dockerfile, compose 配置正确性 | 10+ |
| test_topic_compat.py | Topic 名称/类型与 MuJoCo bridge 一致 | 15+ |
| test_backend_switch.py | CLI backend 切换逻辑 | 10+ |

**集成测试 (需要 Docker + Isaac Sim):**

| 测试文件 | 测试内容 | 数量 |
|----------|---------|------|
| test_isaac_integration.py | DDS 跨容器通信 | 10+ |
| test_isaac_locomotion.py | Go2 walk/turn/stop 通过 Isaac Sim | 10+ |
| test_isaac_manipulation.py | arm pick/place 通过 Isaac Sim | 10+ |
| test_isaac_perception.py | 相机/点云/VLM 在 Isaac Sim 场景 | 10+ |
| test_isaac_nav_stack.py | FAR/TARE 在 Isaac Sim 环境 | 10+ |

**集成测试标记:**
```python
import pytest

# 需要 Docker + Isaac Sim 运行
isaac_sim = pytest.mark.skipif(
    not _isaac_sim_available(),
    reason="Isaac Sim Docker not running"
)

@isaac_sim
def test_go2_walk_forward():
    proxy = IsaacSimProxy()
    proxy.connect()
    ...
```

**目标: 100+ 新测试 (80+ 单元 + 20+ 集成)**

---

## 依赖

### 新增外部依赖 (需 CEO 批准)

| 依赖 | 版本 | 类型 | 原因 |
|------|------|------|------|
| Isaac Sim | 5.1.0 | Docker 容器 | 核心仿真引擎 |
| NVIDIA Container Toolkit | latest | Host apt | GPU passthrough |
| CycloneDDS | Jazzy 默认 | 已有 | DDS 跨版本通信 |

### Host 侧不新增 pip 依赖

IsaacSimProxy 仅使用 rclpy (已有) + 标准 ROS2 消息类型 (已有)。

### Docker 内依赖 (容器隔离，不影响 host)

- Isaac Sim 5.1.0 runtime
- ROS2 Humble (ros-humble-ros-base)
- ros-humble-foxglove-bridge (可选)

---

## 兼容性风险

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| Isaac Sim 5.1 官方不支持 Ubuntu 24.04 | 低 | Docker 隔离，容器内是 Ubuntu 22.04。Host 只需 NVIDIA driver + Container Toolkit |
| DDS Humble↔Jazzy 通信故障 | 中 | CycloneDDS 两侧一致 + `--network host` + 相同 domain ID。已有社区验证 |
| Isaac Sim 首次启动慢 (shader 编译 5-10 min) | 低 | 只影响首次。可 pre-warm cache 到 Docker volume |
| Isaac Sim GPU 显存占用大 (>8 GB) | 中 | RTX 5080 有 16 GB VRAM。限制场景复杂度，不同时运行 MuJoCo GUI |
| ROS2 消息版本差异 (Humble msg vs Jazzy msg) | 低 | 标准消息 (Odometry, PointCloud2, Image) 的 wire format 在 Humble/Jazzy 之间不变 |
| SO-101 arm 无 Isaac Sim USD 模型 | 中 | 用 URDF→USD 转换工具 (`convert_urdf.py`)。已有 `so101_sim.urdf` |

---

## 开发计划

### Wave 1: Docker + Bridge (基础设施)
- Dockerfile + docker-compose.yaml
- isaac_sim_bridge.py (Humble node, 容器内)
- launch_isaac.sh / stop_isaac.sh
- 健康检查 + DDS 连通性测试
- **50+ 测试**

### Wave 2: Protocol + CLI (Host 集成)
- IsaacSimProxy (BaseProtocol)
- IsaacSimArmProxy (ArmProtocol)
- SimStartTool backend 参数
- VGG 验证 (decompose + execute on Isaac Sim)
- **30+ 测试**

### Wave 3: 场景 + Nav Stack (功能完整)
- 3 个 USD 场景 (single room / multi room / manipulation bench)
- launch_explore_isaac.sh (nav stack on Isaac Sim)
- SceneGraph 在 Isaac Sim 中验证
- Manipulation E2E (pick cup in photorealistic kitchen)
- **20+ 测试**

### Wave 4: RL 训练 (高级功能)
- Isaac Lab Go2 locomotion environment 配置
- 训练脚本 + rough terrain curriculum
- Policy 导出 (ONNX)
- 在 Vector OS Nano 中加载训练好的 policy

---

## 决定需 CEO 审批

1. **新增 Isaac Sim 5.1 依赖** — Docker 容器 (~20 GB image)，需要 NVIDIA Container Toolkit
2. **DDS 跨版本通信方案** — Humble (Docker) ↔ Jazzy (Host)，业界有验证但我们是首次用
3. **场景资产来源** — NVIDIA Nucleus 内置资产 vs 社区 USD。许可证: CC0/CC-BY
4. **开发优先级** — Wave 1-3 约需 3 个 session。Wave 4 (RL) 视 Wave 3 结果决定是否启动
