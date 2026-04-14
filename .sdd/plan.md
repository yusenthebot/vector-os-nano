# v2.0 Plan: VectorEngine 统一架构 + VGG 全面升级

**Status**: DRAFT
**Branch**: `feat/v2.0-vectorengine-unification`
**Base**: v1.8.0 (legacy 已清理)

## 架构变更

### 变更前

```
vector-cli → VectorEngine (新)
vector-os-mcp → Agent.execute() (旧) → llm/ → executor → skill
                                        ↑ 完全独立的执行路径
```

### 变更后

```
vector-cli  ─┐
             ├→ VectorEngine → VGG / tool_use → skill.execute()
vector-os-mcp┘
```

MCP 创建 VectorEngine 实例（与 CLI 同一引擎），通过 `engine.run_turn()` 处理 NL，通过 VGG fast-path 处理直接技能调用。

### MCP 适配方案

| MCP 旧调用 | 新实现 |
|-----------|--------|
| `agent.execute(instruction)` | `engine.run_turn(instruction, session)` |
| `agent.run_goal(goal)` | `engine.vgg_decompose(goal)` + `engine.vgg_execute_sync(tree)` |
| `agent.execute_skill(name, args)` | `engine.run_turn("name args", session)` (VGG fast-path) |
| `agent._skill_registry` | 直接持有 `SkillRegistry` |
| `agent._arm/_perception/...` | 持有 `HardwareRefs` dataclass |
| `agent.world` | 持有 `WorldModel` 引用 |

新增 `HardwareRefs` (frozen dataclass):
```python
@dataclass(frozen=True)
class HardwareRefs:
    arm: Any = None
    gripper: Any = None
    base: Any = None
    perception: Any = None
    calibration: Any = None
    world_model: Any = None
    skill_registry: Any = None
    config: dict = field(default_factory=dict)
```

新增 `VectorEngine.vgg_execute_sync()`:
```python
def vgg_execute_sync(self, goal_tree: GoalTree) -> ExecutionTrace:
    """同步执行 GoalTree（MCP 用）。阻塞到完成或 abort。"""
```

## 任务分解

### Wave 1: 统一架构 (Phase 1) — BLOCKING

依赖关系：1.1 → 1.2/1.3/1.4 并行 → 1.5 → 1.6 → 1.7 → 1.8

| ID | 任务 | 文件 | 预估行数 | 依赖 |
|----|------|------|---------|------|
| 1.1 | 新增 `HardwareRefs` + MCP engine 工厂 | `mcp/server.py` | +80 改 | 无 |
| 1.2 | 重写 `handle_tool_call` 用 engine | `mcp/tools.py` | ~120 改 | 1.1 |
| 1.3 | 重写 resources 用 HardwareRefs | `mcp/resources.py` | ~30 改 | 1.1 |
| 1.4 | 新增 `vgg_execute_sync` 方法 | `vcli/engine.py` | +30 新 | 无 |
| 1.5 | 瘦身 `core/agent.py` (删 execute/handle/plan) | `core/agent.py` | -700 删 | 1.2 |
| 1.6 | 删除旧管线 + llm/ | 6 文件 + 1 模块 | -2400 删 | 1.5 |
| 1.7 | 删除旧测试 (21 测试 + 18 stubs) | tests/ | -1500 删 | 1.6 |
| 1.8 | 更新 `__init__.py` 导出 | 2 文件 | ~10 改 | 1.7 |

**Wave 1 验收**:
```bash
python3 -c "from vector_os_nano import Agent; print('ok')"
python3 -c "from vector_os_nano.vcli.engine import VectorEngine; print('ok')"
python3 -c "from vector_os_nano.mcp.server import VectorMCPServer; print('ok')"
pytest tests/ --co -q  # 收集成功，无 ImportError
# MCP --sim-headless 启动 + natural_language tool 返回结果
```

### Wave 2: Abort 信号 (Phase 2) — BLOCKING

全部并行开发，最后集成测试。

| ID | 任务 | 文件 | 预估行数 | 依赖 |
|----|------|------|---------|------|
| 2.1 | 新增全局 abort 模块 | NEW `vcli/cognitive/abort.py` | +40 新 | Wave 1 |
| 2.2 | Engine P0 stop 绕过 + clear/request | `vcli/engine.py` | +35 改 | 2.1 |
| 2.3 | VGGHarness 每步检查 abort | `vcli/cognitive/vgg_harness.py` | +15 改 | 2.1 |
| 2.4 | GoalExecutor abort + async skill wait | `vcli/cognitive/goal_executor.py` | +40 改 | 2.1 |
| 2.5 | StopSkill 调 request_abort() | `skills/go2/stop.py` | +5 改 | 2.1 |
| 2.6 | ExploreSkill 挂 abort → _explore_cancel | `skills/go2/explore.py` | +10 改 | 2.1 |
| 2.7 | NavigateSkill dead_reckoning 检查 abort | `skills/navigate.py` | +10 改 | 2.1 |
| 2.8 | go2_ros2_proxy.navigate_to() abort | `hardware/sim/go2_ros2_proxy.py` | +10 改 | 2.1 |
| 2.9 | Abort 测试套件 | NEW `tests/harness/test_abort.py` | +120 新 | 2.2-2.8 |

**Wave 2 验收**:
```
"stop" 中断 VGG → <100ms 响应
"探索然后去厨房" → explore 完成后 navigate
explore 中 "stop" → 探索取消
navigate 中 "stop" → 导航取消
新命令覆盖旧任务
```

### Wave 3: VGG 框架升级 (Phase 3) — NON-BLOCKING, 并行

每个任务独立，可分配给 Alpha/Beta/Gamma 并行。

| ID | 任务 | 文件 | 预估行数 |
|----|------|------|---------|
| 3.1 | VGG init 失败日志 + 用户反馈 | `vcli/engine.py` init_vgg() | +20 改 |
| 3.2 | GoalDecomposer: skill 可用性检查 + few-shot | `vcli/cognitive/goal_decomposer.py` | +60 改 |
| 3.3 | GoalExecutor: error_category + 上下文注入 | `vcli/cognitive/goal_executor.py`, `types.py` | +30 改 |
| 3.4 | StrategySelector: 阈值调整 + 模糊匹配 + 冷却 | `vcli/cognitive/strategy_selector.py` | +40 改 |
| 3.5 | VisualVerifier: 基于 verify 表达式触发 | `vcli/cognitive/visual_verifier.py` | +15 改 |
| 3.6 | ObjectMemory: 双向同步 | `vcli/cognitive/object_memory.py`, `goal_executor.py` | +25 改 |
| 3.7 | Wave 3 测试 | `tests/harness/test_vgg_upgrade.py` | +100 新 |

### Wave 4: 引擎质量 (Phase 4) — NON-BLOCKING

| ID | 任务 | 文件 | 预估行数 |
|----|------|------|---------|
| 4.1 | IntentRouter 中英文混合 | `vcli/intent_router.py` | +25 改 |
| 4.2 | Engine room context 追踪 | `vcli/engine.py`, `vcli/prompt.py` | +30 改 |
| 4.3 | VGG CLI 实时进度 | `vcli/engine.py` | +20 改 |
| 4.4 | Wave 4 测试 | tests/ | +50 新 |

## 文档更新计划

| 文档 | 时机 | 改动 |
|------|------|------|
| `progress.md` | 每个 Wave 完成后 | 更新版本号、测试数、架构描述 |
| `docs/cli-tool-system.md` | Wave 1 后 | 删除旧 Agent 引用，加 MCP 统一说明 |
| `docs/vgg-design-spec.md` | Wave 3 后 | 更新 VGG 组件描述（abort、error category、few-shot） |
| `README.md` | Wave 2 后 | 更新入口点（只剩 vector-cli + vector-os-mcp） |
| `.sdd/status.json` | 每个任务完成后 | 更新进度 |

## 测试策略

| Wave | 新测试 | 验证方式 |
|------|--------|---------|
| 1 | MCP E2E (natural_language, run_goal, direct skill, resources) | pytest + 手动 MCP 启动 |
| 2 | Abort 场景 (stop 中断 VGG, explore, navigate, 新命令覆盖) | pytest + threading |
| 3 | VGG 单元测试 (decomposer, executor, strategy, verifier) | pytest |
| 4 | IntentRouter 中英文 + context 解析 | pytest |

全量回归：每个 Wave 完成后 `pytest tests/ -x`

## Agent 分工建议

| Wave | 分工 |
|------|------|
| 1 | Alpha: 1.1+1.2+1.3 (MCP 重写), Beta: 1.4+1.5 (engine+agent 瘦身), Gamma: 1.6+1.7+1.8 (清理) |
| 2 | Alpha: 2.1+2.2+2.3+2.4 (abort 核心), Beta: 2.5+2.6+2.7+2.8 (技能集成), Gamma: 2.9 (测试) |
| 3 | Alpha: 3.1+3.2, Beta: 3.3+3.4, Gamma: 3.5+3.6+3.7 |
| 4 | Alpha: 4.1+4.2, Beta: 4.3+4.4 |
