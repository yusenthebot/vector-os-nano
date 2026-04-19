# Plan: Isaac Sim 集成 — Wave 1

**Version:** 0.1
**Date:** 2026-04-10
**Spec:** spec.md (approved)
**Scope:** Wave 1 — Docker + Bridge + Proxy + Tests

---

## 前置条件修复

### P0: 修复 Docker daemon
Docker 因 iptables NAT 模块缺失崩溃 (Ubuntu 24.04 nftables 兼容)。
- `sudo modprobe iptable_nat iptable_filter` 加载内核模块
- 写 `/etc/modules-load.d/docker.conf` 使其持久化
- `sudo systemctl restart docker`

### P1: 安装 NVIDIA Container Toolkit
GPU passthrough 需要 nvidia-container-toolkit。
- 添加 NVIDIA 仓库 key + apt source
- `sudo apt install nvidia-container-toolkit`
- `sudo nvidia-ctk runtime configure --runtime=docker`
- `sudo systemctl restart docker`
- 验证: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`

---

## Wave 1 任务

### T1: Docker 基础设施 — `docker/isaac-sim/`

**文件:**
- `docker/isaac-sim/Dockerfile` — 基于 Isaac Sim 容器, 加 ROS2 Humble + CycloneDDS + bridge 脚本
- `docker/isaac-sim/docker-compose.yaml` — GPU passthrough, host network, volumes
- `docker/isaac-sim/cyclonedds.xml` — DDS 配置 (确保跨容器通信)
- `scripts/launch_isaac.sh` — 一键启动 + 健康检查
- `scripts/stop_isaac.sh` — 干净关闭

**依赖:**
- Isaac Sim 容器 tag: `nvcr.io/nvidia/isaac-sim:4.5.0` (Isaac Sim 5.1)
- 如果 4.5.0 不存在，回退到 `nvcr.io/nvidia/isaac-sim:4.2.0` 或最新可用

### T2: Isaac Sim Bridge — `docker/isaac-sim/bridge/`

**文件:**
- `docker/isaac-sim/bridge/isaac_sim_bridge.py` — 主 bridge 脚本 (容器内运行)
  - 用 Isaac Sim Python API 创建 scene、spawn Go2、step physics
  - 用 ROS2 Humble 发布 topic
  - Topic 名称与 MuJoCo bridge 完全一致
- `docker/isaac-sim/bridge/go2_scene.py` — Go2 场景创建 (flat ground + room)
- `docker/isaac-sim/bridge/sensor_publisher.py` — 传感器数据 → ROS2 topic

**发布 topic:**
| Topic | Type | Rate |
|-------|------|------|
| /state_estimation | Odometry | 50 Hz |
| /registered_scan | PointCloud2 | 10 Hz |
| /camera/image | Image | 30 Hz |
| /camera/depth | Image | 30 Hz |
| /tf | TFMessage | 50 Hz |
| /joint_states | JointState | 50 Hz |
| /joy | Joy | 2 Hz (fake) |
| /speed | Float32 | 2 Hz |

**订阅:**
| Topic | Type |
|-------|------|
| /cmd_vel | Twist |
| /cmd_vel_nav | Twist |

### T3: IsaacSimProxy — Host 侧

**文件:**
- `vector_os_nano/hardware/sim/isaac_sim_proxy.py`

**设计决定:** Go2ROS2Proxy 已经实现了完整的 BaseProtocol 通过 ROS2 topic。
IsaacSimProxy 继承 Go2ROS2Proxy，覆盖:
- `name` → "isaac_go2"
- `connect()` → 加 Docker 健康检查 (确认容器运行 + topic 可达)
- `supports_lidar` → True (Isaac Sim RTX lidar vs MuJoCo mj_ray)

**复用的方法 (不改):**
set_velocity, walk, stand, sit, get_position, get_heading, get_odometry,
get_camera_frame, get_depth_frame, navigate_to, _publish_markers...
全部通过 ROS2 topic 工作，不感知后端。

### T4: IsaacSimArmProxy — Host 侧

**文件:**
- `vector_os_nano/hardware/sim/isaac_sim_arm_proxy.py`

**实现 ArmProtocol:**
- connect(): 创建 rclpy node, subscribe /arm/joint_states, publish /arm/joint_commands
- get_joint_positions(): 从 /arm/joint_states 缓存读
- move_joints(): 发布目标到 /arm/joint_commands, 等待到达
- fk/ik(): host 侧用 pinocchio 计算 (已有依赖)

### T5: CLI 集成

**修改:**
- `vector_os_nano/vcli/tools/sim_tool.py` — 添加 `backend` 参数
- `vector_os_nano/robo/groups/sim.py` — 添加 `--backend` CLI flag

**流程:**
```
vector sim start --backend isaac
  → 检查 Docker 容器是否运行 (docker ps | grep isaac-sim)
  → IsaacSimProxy().connect()
  → 等待 /state_estimation 第一条消息
  → Agent(base=proxy) + 注册 Go2 skills
```

### T6: 测试

**单元测试 (mock, 不需要 Docker):**

`tests/unit/test_isaac_sim_proxy.py` (25+):
- Protocol 合规 (isinstance BaseProtocol)
- connect/disconnect 生命周期
- set_velocity 发布 Twist
- get_position/heading/odometry 返回缓存值
- walk() blocking 行为
- Docker 健康检查逻辑

`tests/unit/test_isaac_arm_proxy.py` (20+):
- Protocol 合规 (isinstance ArmProtocol)
- connect/disconnect
- get_joint_positions 缓存
- move_joints 发布 + 等待
- fk/ik 数学正确性

`tests/unit/test_docker_config.py` (10+):
- Dockerfile 语法验证
- docker-compose.yaml 结构验证
- 环境变量正确
- Volume mount 路径存在

`tests/unit/test_topic_compat.py` (15+):
- Isaac bridge topic 名称 == MuJoCo bridge topic 名称
- 消息类型匹配
- QoS 兼容

`tests/unit/test_backend_switch.py` (10+):
- SimStartTool schema 包含 backend 参数
- backend=mujoco 走旧路径
- backend=isaac 走新路径
- isaac 容器未运行时报错

**集成测试 (需要 Docker + Isaac Sim):**
标记 `@pytest.mark.isaac_sim` — CI 跳过，本地手动跑。

`tests/integration/test_isaac_integration.py` (10+):
- Docker 容器启动
- DDS topic 可达 (ros2 topic list)
- Odometry 数据正确
- cmd_vel 控制 Go2

---

## 文件变更清单

### 新增文件
```
docker/isaac-sim/
  Dockerfile
  docker-compose.yaml
  cyclonedds.xml
  bridge/
    isaac_sim_bridge.py
    go2_scene.py
    sensor_publisher.py
    requirements.txt

scripts/
  launch_isaac.sh
  stop_isaac.sh

vector_os_nano/hardware/sim/
  isaac_sim_proxy.py
  isaac_sim_arm_proxy.py

tests/unit/
  test_isaac_sim_proxy.py
  test_isaac_arm_proxy.py
  test_docker_config.py
  test_topic_compat.py
  test_backend_switch.py

tests/integration/
  test_isaac_integration.py
```

### 修改文件
```
vector_os_nano/hardware/sim/__init__.py  — export IsaacSimProxy, IsaacSimArmProxy
vector_os_nano/vcli/tools/sim_tool.py   — backend 参数
vector_os_nano/robo/groups/sim.py       — --backend flag
```

---

## 执行顺序

1. **P0 + P1**: 修 Docker + 装 nvidia-container-toolkit (需要 sudo)
2. **T1**: Docker 基础设施 (Dockerfile, compose, scripts)
3. **T6a**: 单元测试先行 (test_docker_config, test_topic_compat)
4. **T3 + T4**: IsaacSimProxy + IsaacSimArmProxy
5. **T6b**: Proxy 单元测试 (test_isaac_sim_proxy, test_isaac_arm_proxy)
6. **T5**: CLI 集成 (SimStartTool + robo groups)
7. **T6c**: CLI 测试 (test_backend_switch)
8. **T2**: Isaac Sim Bridge (容器内脚本)
9. **T6d**: 集成测试 (需 Docker 运行)
10. **验证**: 端到端 — `vector sim start --backend isaac` → VGG 执行
