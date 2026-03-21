# ADR-002: Skill Manifest Protocol

- **Status:** proposed
- **Date:** 2026-03-21
- **Author:** CEO/CTO (Yusen) + Lead Architect (Opus)

## Context

The current system has two execution paths:
1. Direct commands (home, scan, pick X) -> parsed in Agent._execute_direct()
2. LLM planning -> full pipeline

Problems:
- Simple commands like "close grip" require LLM (slow, costs money) or hardcoded if/else
- LLM doesn't always understand intent ("pick proteinbar" vs "pick protein bar")
- Adding a new skill requires editing code in 3 places (skill class, LLM prompt, direct parser)
- No way for the LLM to know WHEN to use which skill (description too brief)

## Decision

Introduce a **Skill Manifest** — a YAML configuration file that serves as the single source of truth for:
1. What skills exist and what they do
2. How to invoke them (aliases for direct execution)
3. When to use them (rich descriptions for LLM context)
4. Common task chains (auto_steps for compound operations)

## Manifest Schema

```yaml
# config/skill_manifest.yaml

version: "1.0"

skills:
  pick:
    description: "Pick up an object from the workspace"
    when_to_use: "User wants to grab, pick up, take, or grasp an object"
    aliases:
      - "grab"
      - "take"
      - "grasp"
      - "拿起"
      - "抓取"
      - "拿"
    requires_perception: true
    requires_arm: true
    auto_steps: ["scan", "detect", "pick"]
    parameters:
      object_label:
        type: string
        description: "Name of the object to pick"
        required: false
    examples:
      - "pick up the red cup"
      - "grab the battery"
      - "拿起电池"

  place:
    description: "Place a held object at a target position"
    when_to_use: "User wants to put down, drop, place, or release an object at a specific location"
    aliases:
      - "put"
      - "put down"
      - "drop"
      - "放下"
      - "放置"
    requires_arm: true
    auto_steps: ["place"]
    parameters:
      x: {type: float, description: "Target X in metres", required: false}
      y: {type: float, description: "Target Y in metres", required: false}
      z: {type: float, description: "Target Z in metres", required: false}

  home:
    description: "Move arm to home position and open gripper"
    when_to_use: "User wants to reset, go home, or move arm to safe position"
    aliases:
      - "go home"
      - "reset"
      - "回家"
      - "归位"
      - "复位"
    requires_arm: true
    direct_execute: true

  scan:
    description: "Move arm to scan position for workspace observation"
    when_to_use: "User wants to look at the workspace, observe, or prepare for detection"
    aliases:
      - "look"
      - "observe"
      - "look around"
      - "看看"
      - "扫描"
    requires_arm: true
    direct_execute: true

  detect:
    description: "Detect objects in the workspace using VLM"
    when_to_use: "User wants to find, search, identify, or list objects"
    aliases:
      - "find"
      - "search"
      - "what do you see"
      - "list objects"
      - "看到什么"
      - "找"
      - "检测"
    requires_perception: true
    auto_steps: ["scan", "detect"]
    parameters:
      query:
        type: string
        description: "What to detect"
        required: true
        default: "all objects"

  gripper_open:
    description: "Open the gripper / release"
    when_to_use: "User wants to open, release, or let go"
    aliases:
      - "open"
      - "open grip"
      - "open gripper"
      - "open claw"
      - "release"
      - "let go"
      - "张开"
      - "松开"
    requires_arm: true
    direct_execute: true
    is_primitive: true

  gripper_close:
    description: "Close the gripper / grip"
    when_to_use: "User wants to close, grip, clench, or hold tight"
    aliases:
      - "close"
      - "close grip"
      - "close gripper"
      - "grip"
      - "clench"
      - "夹紧"
      - "合上"
    requires_arm: true
    direct_execute: true
    is_primitive: true
```

## Execution Flow

```
User input: "grab the battery"
    |
    v
[1. Alias Matcher]
    Match "grab" -> skill "pick"
    direct_execute? No -> need LLM for parameters
    auto_steps defined? Yes -> ["scan", "detect", "pick"]
    |
    v
[2. If direct_execute=true OR auto_steps covers it]
    Execute auto_steps directly, no LLM needed
    |
[3. Else -> LLM Planner]
    Send manifest as context (rich descriptions + examples)
    LLM generates task plan
    |
    v
[4. Executor]
    Run task chain
```

```
User input: "close grip"
    |
    v
[1. Alias Matcher]
    Match "close grip" -> skill "gripper_close"
    direct_execute? Yes
    |
    v
[2. Direct Execute]
    gripper.close() -- no LLM, no planning, instant
```

## Implementation Plan

### Phase 1: Manifest Loading
- Load skill_manifest.yaml at Agent startup
- Parse into SkillManifest dataclass
- Merge with code-registered skills

### Phase 2: Alias Matcher
- New class: SkillMatcher
- Input: user text -> Output: matched skill + params
- Fuzzy matching for typos (Levenshtein distance)
- Handles both English and Chinese aliases

### Phase 3: Auto-Steps Execution
- If matched skill has auto_steps, execute the chain directly
- Skip LLM entirely for common patterns (scan->detect->pick)
- Fall back to LLM for complex/ambiguous instructions

### Phase 4: LLM Context Enhancement
- Replace current skill schemas with manifest descriptions
- Include examples in LLM prompt
- Include when_to_use hints

## Comparison with MCP

| Feature | MCP | Skill Manifest |
|---------|-----|----------------|
| Discovery | Dynamic (server announces tools) | Static YAML + code registration |
| Invocation | JSON-RPC tool call | Direct Python or task chain |
| Description | Tool description string | Rich: description + when_to_use + examples |
| Aliases | None | Multi-language command aliases |
| Chaining | LLM decides | auto_steps pre-defined chains |
| Direct execution | Always via LLM | direct_execute bypasses LLM |

## Risks

- Alias conflicts between skills (e.g., "open" could mean gripper or a door)
- Manifest gets out of sync with code-registered skills
- Auto-steps may not cover all edge cases

## Mitigations

- Priority: code-registered skills override manifest
- Validation at startup: warn if manifest skill has no matching code class
- Auto-steps are hints, not constraints -- LLM can override for complex cases
