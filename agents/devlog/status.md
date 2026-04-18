# Agent Status

**Updated:** 2026-04-17

## Current: v2.1 Phase A — Piper 挂载 + 双模式仿真

Branch: `feat/v2.0-vectorengine-unification` (uncommitted on top of c83e96d)

### Delivered this session
- **Piper 资产**: MuJoCo Menagerie 官方 `agilex_piper` → `mjcf/piper/`
- **Go2+Piper 合成 MJCF**: `mjcf/go2_piper/go2_piper.xml` (via `MjSpec.attach`, 29 bodies / 21 joints / 19 actuators)
- **挂载位置**: trunk 顶面居中，雷达后方 15cm，Z-up X-forward (4 视角 render 验证无穿模)
- **双模式**: `sim_tool` 加 `with_arm` 参数 + `VECTOR_SIM_WITH_ARM` env var；`_build_room_scene_xml` 自动读
- **MPC 维度守卫**: `_init_mpc_stack` 检测 Pinocchio nq ≠ MuJoCo nq → 抛出 → 外层 fallback sinusoidal (修复了 physics 线程崩溃)
- **SimStopTool**: `stop_simulation` tool，killpg subprocess + 卸载 skill tools + rebuild prompt

### 关键调试 (4 轮 Hypothesis Loop)
真正根因：用户有 convex_mpc，Pinocchio 模型 nq=19，MuJoCo 加 Piper 后 nq=27，`pin.forwardKinematics` 每次 MPC tick 抛 `ValueError: expected 19, got 27` → `mujoco_go2_physics` 线程挂掉 → cmd_vel 写入但无消费者 → 狗永远不动

详细四轮过程见 `progress.md` v2.1 段。

### 未 commit 的改动
```
M scripts/go2_vnav_bridge.py          (debug log + 注释清理)
M vector_os_nano/hardware/sim/go2_ros2_proxy.py  (walk() 加 logger)
M vector_os_nano/hardware/sim/mujoco_go2.py  (MPC guard + thread-ID gate + Piper stow)
M vector_os_nano/vcli/tools/__init__.py  (SimStopTool 注册)
M vector_os_nano/vcli/tools/sim_tool.py  (with_arm + SimStopTool + _shutdown_agent)
?? vector_os_nano/hardware/sim/mjcf/go2/scene_room_piper.xml
?? vector_os_nano/hardware/sim/mjcf/go2_piper/
?? vector_os_nano/hardware/sim/mjcf/piper/  (Menagerie LICENSE 已带)
```

### 测试状态
| 场景 | 状态 |
|---|---|
| Python 独立 walk (no-arm, MPC) | ✓ 走 0.9m/3s |
| Python 独立 walk (with-arm, sinusoidal) | ✓ 走 0.75m/3s |
| Yusen 机器 go2sim → 询问模式 → no-arm | ✓ 启动 OK |
| Yusen 机器 走两米 (with-arm) | ✓ 能走，步态粗糙（sinusoidal 本身的限制） |
| 关闭仿真 tool | ⏸ 已实现，待下 session 验证 |

---

## 历史：v2.0.1 V-Graph 跨房间修复 — 完成 (2026-04-16)

Branch: `feat/v2.0-vectorengine-unification`
Commit: c83e96d pushed. FAR V-Graph 75 vertices + 131 edges verified.

详细见 progress.md 的 v2.0.1 段。
