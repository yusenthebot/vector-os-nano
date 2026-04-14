# v2.0 Plan — Wave 3: 全栈升级

**Status**: APPROVED
**Branch**: `feat/v2.0-vectorengine-unification`

## 前置完成

- Wave 1: 架构统一 (-17,700 行) — done
- Wave 2: Abort 信号 (14 tests) — done

## Wave 3 任务分解

三组并行，按子系统分：Alpha=导航可靠性, Beta=反馈可观测性, Gamma=引擎质量。

### Group A: 导航可靠性 (Alpha)

| ID | 任务 | 文件 | 要点 |
|----|------|------|------|
| A1 | Nav stack 健康监控 | `vcli/cli.py` | 后台线程每 5s poll proc, 崩溃自动重启 + console.print 告警 |
| A2 | 单一 TARE 启动 | `skills/go2/explore.py` | 删 _start_tare() 的 Popen, 改为只发 `/start_exploration` 信号; TARE 由 launch_nav_only.sh 独占管理 |
| A3 | 卡死检测上限 30s | `scripts/go2_vnav_bridge.py` | stuck_count 累计超 15 (30s) 时强制 abort + emit event |
| A4 | navigate_to 全局超时 | `hardware/sim/go2_ros2_proxy.py` | Phase 2 循环增加 deadline check, 超时 return False |
| A5 | door-chain 超时分配 | `skills/navigate.py` | 总 timeout 减已用时间, 平均分给剩余 waypoint |

### Group B: 反馈可观测性 (Beta)

| ID | 任务 | 文件 | 要点 |
|----|------|------|------|
| B1 | 导航进度回调 | `hardware/sim/go2_ros2_proxy.py`, `skills/navigate.py` | navigate_to 接受 on_progress callback, 每 2s 调: (dist_remaining, elapsed, waypoint_label) |
| B2 | 导航进度 CLI 显示 | `vcli/cli.py` 或 `vcli/tools/skill_wrapper.py` | on_progress → console.print 距离/时间 |
| B3 | 探索进度频率 5s | `skills/go2/explore.py` | _exploration_loop 的报告间隔 30s → 5s |
| B4 | Camera 时间戳 | `hardware/sim/go2_ros2_proxy.py` | 加 _last_camera_ts, get_camera_frame 返回 (frame, age_sec), VLM warn if >1s |

### Group C: 引擎质量 (Gamma)

| ID | 任务 | 文件 | 要点 |
|----|------|------|------|
| C1 | 导航常量参数化 | NEW `config/nav.yaml`, `go2_ros2_proxy.py`, `go2_vnav_bridge.py`, `navigate.py` | 提取 ceiling_height, arrival_radius, far_probe_timeout, waypoint_timeout, stall_threshold 到 yaml |
| C2 | GoalDecomposer prompt caching | `vcli/cognitive/goal_decomposer.py` | system prompt 加 cache_control ephemeral, world context 作为 user message |
| C3 | World context 缓存 | `vcli/engine.py` | 缓存 _build_world_context, motor skill 后失效 |
| C4 | 智能 session 压缩 | `vcli/engine.py` 或 `vcli/session.py` | 旧消息 LLM 摘要 (1句话) 替代截断 |
| C5 | 日志轮转 | `vcli/cli.py` | nav stack 日志用 RotatingFileHandler (5MB × 3) |
| C6 | VGG init 失败反馈 | `vcli/engine.py` | init_vgg 每组件单独 try/except, 失败原因 console.print |

## 依赖关系

```
A1 (health) ──→ A2 (single TARE, 因为 health monitor 管 TARE 进程)
A4 (timeout) ──→ A5 (door-chain, 因为 per-waypoint timeout 依赖全局 timeout)
B1 (callback) ──→ B2 (CLI display)
C1 (nav.yaml) ──→ A3, A4 (常量从 yaml 读)
其余全部独立，可并行
```

## 执行顺序

```
第一批 (并行):
  Alpha: A1 → A2
  Beta:  B3, B4
  Gamma: C1, C5, C6

第二批 (依赖第一批):
  Alpha: A3, A4 → A5
  Beta:  B1 → B2
  Gamma: C2, C3

第三批:
  Gamma: C4 (session 压缩, 最低优先级)

最后: 全量测试 + 文档更新
```

## 测试计划

| Group | 新增测试 | 验证方式 |
|-------|---------|---------|
| A | nav_health_test (进程 kill → 重启), stall_timeout_test, navigate_timeout_test | pytest + subprocess mock |
| B | progress_callback_test, camera_timestamp_test, explore_progress_test | pytest |
| C | nav_yaml_load_test, prompt_cache_test, context_cache_test, session_compress_test | pytest |

预估新增: ~60 个测试

## Agent 分工

| Agent | Tasks | 预估改动 |
|-------|-------|---------|
| Alpha | A1-A5 (导航可靠性) | ~200 行改, 4 文件 |
| Beta | B1-B4 (反馈可观测性) | ~150 行改, 4 文件 |
| Gamma | C1-C6 (引擎质量) | ~250 行改, 6 文件 |
