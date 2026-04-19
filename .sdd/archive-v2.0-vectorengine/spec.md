# v2.0 Spec: VectorEngine 统一架构 + 全栈升级

**Status**: APPROVED
**Branch**: `feat/v2.0-vectorengine-unification`

## 完成状态

- [x] Wave 1: 架构统一 (MCP → VectorEngine, 删 17,700 行)
- [x] Wave 2: Abort 信号 (P0 stop, 全局中断)
- [ ] Wave 3: 全栈可靠性 + 可观测性 + 引擎质量 (进行中)

## Wave 3 Spec: 全栈升级

### 问题清单 (审计发现)

**CRITICAL — 卡死/静默失败:**
1. Nav stack 崩溃后 CLI 不知道 — 无健康监控
2. TARE 被 CLI 和 ExploreSkill 重复启动 — DDS 冲突
3. 卡死检测无上限 — 机器人贴墙振荡永远不触发
4. navigate_to 无全局超时 — FAR 不响应时 poll 永远等

**HIGH — 用户体验:**
5. 导航无进度反馈 — 30-120s 黑盒
6. 探索进度 30s 才报一次 — 用户以为卡了
7. door-chain 超时不分配 — 5 waypoint × 30s = 150s
8. LLM 分解 2-5s 无反馈
9. Camera 帧无时间戳 — VLM 用旧帧

**MEDIUM — 引擎质量:**
10. 12+ 硬编码常量
11. 日志无限增长
12. World context 每次重查硬件 (50ms)
13. Session 50 条截断丢历史
14. GoalDecomposer 未用 prompt caching
15. VGG init 静默失败

### Goals

**MUST (可靠性):**
- Nav stack 健康监控 + 自动重启
- 单一 TARE 启动 (消除重复)
- 卡死检测 30s 上限
- navigate_to 全局超时
- door-chain 超时动态分配

**MUST (可观测性):**
- 导航进度每 2s 回调
- 探索进度每 5s 报告
- Camera 时间戳 + 过期警告

**SHOULD (引擎质量):**
- GoalDecomposer prompt caching
- World context 缓存 (motor skill 后失效)
- 智能 session 压缩 (摘要替代截断)
- 常量参数化 → config/nav.yaml
- 日志轮转
- VGG init 失败日志 + 用户通知

### 文件改动范围

| 文件 | 改动 |
|------|------|
| `vcli/cli.py` | nav health monitor 线程, 探索进度频率 |
| `vcli/engine.py` | world context 缓存, VGG init 日志, session 压缩 |
| `vcli/cognitive/goal_decomposer.py` | prompt caching |
| `skills/go2/explore.py` | 删 _start_tare(), 改用信号, 进度 5s |
| `skills/navigate.py` | door-chain 超时分配, 进度回调 |
| `hardware/sim/go2_ros2_proxy.py` | navigate_to 全局超时, 进度回调, camera 时间戳 |
| `scripts/go2_vnav_bridge.py` | 卡死上限 30s, 日志轮转 |
| NEW `config/nav.yaml` | 参数化常量 |

### 验收标准

- [ ] FAR 进程被 kill 后 5s 内自动重启
- [ ] 探索期间只有一个 TARE 进程 (pgrep 验证)
- [ ] 机器人贴墙 30s 后自动 abort + 通知用户
- [ ] navigate_to 超过 timeout 参数后返回 False
- [ ] 导航期间每 2s 打印距离进度
- [ ] 探索期间每 5s 打印发现进度
- [ ] Camera 帧超过 1s 旧时日志 warning
- [ ] config/nav.yaml 中可配置所有导航参数
- [ ] Session 压缩用摘要保留关键上下文
