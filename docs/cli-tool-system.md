# Vector CLI — Tool Call System Architecture

## Overview

Vector CLI is an AI-powered robot development environment. The user speaks natural language; the AI agent uses tools to control the robot, edit code, and diagnose issues — all in one session.

```
User: "探索的时候狗在转角撞墙"
  ↓
AI Agent (VectorEngine)
  ├── file_read("go2_vnav_bridge.py")     → reads path follower code
  ├── file_edit(old="0.6", new="0.4")     → fixes speed constant
  ├── skill_reload("walk")                → hot-reloads without restart
  ├── explore()                           → re-runs exploration
  └── responds: "改了转弯速度，重新探索中"
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  vector-cli (vcli/cli.py)                                       │
│                                                                 │
│  User Input ──→ VectorEngine.run_turn()                         │
│                    │                                            │
│                    ├── DynamicSystemPrompt                       │
│                    │     ├── ROLE_PROMPT (cached)                │
│                    │     ├── TOOL_INSTRUCTIONS (cached)          │
│                    │     ├── Hardware / Skills / World (static)  │
│                    │     └── [Robot State] (refreshed each turn) │
│                    │           Position, Room, SceneGraph,       │
│                    │           Nav state, Exploring status       │
│                    │                                            │
│                    ├── CategorizedToolRegistry                   │
│                    │     ├── code:   file_read/write/edit,       │
│                    │     │          bash, glob, grep             │
│                    │     ├── robot:  22 wrapped skills +         │
│                    │     │          scene_graph_query             │
│                    │     ├── diag:   ros2_topics, ros2_nodes,    │
│                    │     │          ros2_log, nav_state,          │
│                    │     │          terrain_status                │
│                    │     └── system: robot_status, sim_start,    │
│                    │                web_fetch, skill_reload       │
│                    │                                            │
│                    ├── LLM Backend (Anthropic / OpenRouter)      │
│                    ├── PermissionContext (7-layer check)         │
│                    └── Session (JSONL, crash-safe)               │
└─────────────────────────────────────────────────────────────────┘
```

## Tool Call Flow

```
1. User types natural language
2. VectorEngine serializes:
   - system_prompt (DynamicSystemPrompt refreshes robot state)
   - messages (session history)
   - tools (CategorizedToolRegistry.to_anthropic_schemas())
3. LLM returns tool_use blocks
4. Engine partitions tools:
   - read-only + concurrency-safe → parallel (ThreadPoolExecutor, 10 workers)
   - write / motor → sequential
5. For each tool:
   a. PermissionContext.check() → allow / deny / ask
   b. If "ask" → user prompted in CLI
   c. tool.execute(params, context) → ToolResult
   d. If motor skill → post-execution state appended to result
6. Results appended to session
7. Loop back to step 2 until LLM returns end_turn
8. Final text rendered in CLI panel
```

## CategorizedToolRegistry

### Design

Extends `ToolRegistry` (backward-compatible). Adds category-based grouping so tools can be organized, enabled/disabled at runtime, and (future) routed by intent.

```python
class CategorizedToolRegistry(ToolRegistry):
    """Tool registry with category-based organization."""

    _categories: dict[str, list[str]]   # category → [tool_names]
    _disabled: set[str]                 # disabled category names

    def register(self, tool, category="default") -> None
    def enable_category(self, category: str) -> None
    def disable_category(self, category: str) -> None
    def to_anthropic_schemas(self) -> list[dict]     # filters out disabled
    def list_categories(self) -> dict[str, list[str]]
```

### Categories

| Category | Tools | Purpose |
|----------|-------|---------|
| `code` | file_read, file_write, file_edit, bash, glob, grep | Code reading and editing |
| `robot` | 22 wrapped skills + scene_graph_query + world_query | Robot control + spatial queries |
| `diag` | ros2_topics, ros2_nodes, ros2_log, nav_state, terrain_status | ROS2 diagnostics |
| `system` | robot_status, start_simulation, web_fetch, skill_reload | System management |

### Scalability Strategy

1. **v1 (current)**: All categories enabled. ~30 tools sent to LLM per turn.
2. **v1.1 (future)**: Intent routing — analyze user message, select relevant categories. "改代码" → code+system only. "去厨房" → robot+diag only.
3. **v2 (future)**: Deferred schemas — send tool names only, LLM requests full schema on demand.

### Adding a New Tool

```python
# 1. Create file: vcli/tools/my_tool.py
@tool(name="my_tool", description="...", read_only=True, permission="allow")
class MyTool:
    input_schema = { "type": "object", "properties": { ... } }
    def execute(self, params, context) -> ToolResult: ...

# 2. Add to vcli/tools/__init__.py:
#    - Import in discover_all_tools()
#    - Add name to _TOOL_CATEGORIES["my_category"]

# Done. No engine, backend, or permission changes needed.
```

## Tool Protocol

Every tool implements this interface (Protocol-based, no inheritance required):

```python
class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]    # JSON Schema for LLM

    def execute(params, context) -> ToolResult
    def check_permissions(params, context) -> PermissionResult
    def is_read_only(params) -> bool
    def is_concurrency_safe(params) -> bool
```

The `@tool` decorator auto-injects defaults for permissions, read_only, and concurrency.

## Skill Wrapping

Robot skills (`@skill` decorator) are automatically wrapped as LLM tools:

```
@skill(aliases=["stand", "站"]) class StandSkill → SkillWrapperTool("stand")
@skill(aliases=["navigate"])    class NavigateSkill → SkillWrapperTool("navigate")
```

`SkillWrapperTool` adds:
- **Motor detection**: Skills with "move", "navigate", "arm" in effects → requires permission
- **Post-execution state**: After motor skills, appends position/room to result
- **Recovery hints**: On failure, includes suggested next action based on diagnosis_code

```
Success: "Skill 'navigate' succeeded. Data: {room: kitchen}
          State: pos=(16.8, 2.3) room=kitchen"

Failure: "Skill 'navigate' failed. (room_not_explored)
          Suggested: Room not explored yet. Run the explore skill first.
          Current state: {position: [10.0, 5.0], room: hallway}"
```

## DynamicSystemPrompt

Problem: System prompt is built once at startup. Robot state goes stale after the first message.

Solution: `DynamicSystemPrompt` is a list subclass that refreshes the robot context block on each `__iter__()` call. VectorEngine iterates the system prompt on every API call, so the LLM always sees current state.

```python
class DynamicSystemPrompt(list):
    def __iter__(self):
        # Refresh [Robot State] block from RobotContextProvider
        block = self._provider.get_context_block()
        self[self._context_idx] = block
        return super().__iter__()
```

The LLM sees on every turn:
```
[Robot State]
Position: (10.2, 5.3, 0.28) — hallway
Heading: 23 deg (NNE)
SceneGraph: 8 rooms (6 visited), 7 doors, 12 objects
Exploring: no
Nav stack: running
```

## RobotContextProvider

Collects real-time state from multiple sources:

| Field | Source | Updates |
|-------|--------|---------|
| Position (x, y, z) | `base.get_position()` | Every turn |
| Heading (deg + compass) | `base.get_heading()` | Every turn |
| Current room | `scene_graph.nearest_room()` | Every turn |
| SceneGraph summary | `scene_graph.stats()` + `get_room_summary()` | Every turn |
| Exploring? | `explore.is_exploring()` | Every turn |
| Nav stack running? | `explore.is_nav_stack_running()` | Every turn |

Graceful degradation: no base → "No hardware connected". No SceneGraph → omits room data.

## Permission System

7-layer check (highest to lowest priority):

1. `no_permission` flag → allow all
2. `deny_tools` blacklist → deny
3. `tool.check_permissions()` → deny → deny
4. `session_allow` (user said "always") → allow
5. `is_read_only(params)` → allow
6. `tool.check_permissions()` → ask → prompt user
7. default → ask

Motor skills (navigate, walk, pick) → always "ask".
Read-only tools (file_read, grep, ros2_topics) → always "allow".

## Complete Tool Inventory (17 built-in + 22 skills)

### Built-in Tools

| Tool | Category | R/O | Permission | Description |
|------|----------|-----|------------|-------------|
| file_read | code | yes | allow | Read file with line numbers |
| file_write | code | no | ask | Create/overwrite file |
| file_edit | code | no | ask | Search & replace in file |
| bash | code | no | ask | Execute shell command |
| glob | code | yes | allow | Find files by pattern |
| grep | code | yes | allow | Search file contents |
| world_query | robot | yes | allow | Query world model objects |
| scene_graph_query | robot | yes | allow | Query rooms/doors/objects/paths |
| robot_status | system | yes | allow | Hardware connection status |
| start_simulation | system | no | ask | Launch MuJoCo sim |
| web_fetch | system | yes | allow | Fetch URL |
| skill_reload | system | no | ask | Hot-reload skill module |
| ros2_topics | diag | yes | allow | List/hz/echo ROS2 topics |
| ros2_nodes | diag | yes | allow | List/info ROS2 nodes |
| ros2_log | diag | yes | allow | Read robot log files |
| nav_state | diag | yes | allow | Navigation/exploration status |
| terrain_status | diag | yes | allow | Terrain map file info |

### Wrapped Robot Skills (22)

Walk, Turn, Stand, Sit, Lie Down, Stop, Explore, Navigate, Patrol,
Look, Describe Scene, Where Am I, Home, Scan, Wave, Pick, Place,
Handover, Detect, Describe, Gripper Open, Gripper Close

## Session Persistence

JSONL format with atomic write + fsync:
```
{"type":"user","content":"去厨房","ts":"..."}
{"type":"assistant","text":"","tool_use":[{"name":"navigate","input":{"room":"kitchen"}}],"ts":"..."}
{"type":"tool_result","results":[{"content":"Skill 'navigate' succeeded..."}],"ts":"..."}
{"type":"assistant","text":"到了厨房，你要我看看有什么吗？","ts":"..."}
```

Auto-compacted at 50 entries to prevent context overflow.

## File Map

```
vcli/
├── cli.py                # Entry point, REPL loop, slash commands
├── engine.py             # VectorEngine — multi-turn tool_use agent loop
├── prompt.py             # System prompt builder (static + dynamic blocks)
├── robot_context.py      # RobotContextProvider (live robot state)
├── dynamic_prompt.py     # DynamicSystemPrompt (refreshes on each turn)
├── session.py            # JSONL session persistence
├── config.py             # ~/.vector/config.yaml loader
├── permissions.py        # 7-layer permission checker
├── backends/
│   ├── __init__.py       # LLMBackend Protocol + create_backend factory
│   ├── anthropic.py      # Anthropic Messages API (streaming)
│   └── openai_compat.py  # OpenRouter / Ollama / vLLM
└── tools/
    ├── base.py           # Tool Protocol, @tool decorator, ToolRegistry,
    │                     # CategorizedToolRegistry
    ├── __init__.py       # discover_all_tools(), discover_categorized_tools()
    ├── file_tools.py     # file_read, file_write, file_edit
    ├── bash_tool.py      # bash
    ├── search_tools.py   # glob, grep
    ├── robot.py          # world_query, robot_status
    ├── sim_tool.py       # start_simulation
    ├── web_tool.py       # web_fetch
    ├── skill_wrapper.py  # SkillWrapperTool + wrap_skills() + recovery hints
    ├── scene_graph_tool.py  # scene_graph_query (7 query types)
    ├── ros2_tools.py     # ros2_topics, ros2_nodes, ros2_log
    ├── nav_tools.py      # nav_state, terrain_status
    └── reload_tool.py    # skill_reload (hot reload)
```
