# VGG Phase 1 — Task List

## Execution Status
- Total tasks: 6
- Completed: 0
- In progress: 0
- Pending: 6

## Wave 1 — Foundation (parallel)

### T1: types.py + goal_verifier.py [Alpha]
- Status: pending
- Files: vcli/cognitive/types.py, vcli/cognitive/goal_verifier.py
- Tests: tests/harness/test_level41_vgg_types_verifier.py
- ACs: AC-7 to AC-12
- Details:
  - types.py: GoalTree, SubGoal, StepRecord, ExecutionTrace (frozen dataclasses)
  - goal_verifier.py: GoalVerifier class with whitelist sandbox
  - Whitelist: nearest_room, get_position, get_heading, get_visited_rooms, query_rooms, describe_scene, detect_objects, world_stats
  - Block: import, exec, eval, open, os, sys, __ prefixed names
  - Block: assignment statements (only expressions allowed)
  - 5 second timeout per verify call
  - Exception → return False

### T2: primitives (all 4 modules) [Beta]
- Status: pending
- Files: vcli/primitives/__init__.py, locomotion.py, navigation.py, perception.py, world.py
- Tests: tests/harness/test_level42_primitives.py
- ACs: AC-13 to AC-20
- Details:
  - PrimitiveContext dataclass in __init__.py (base, scene_graph, vlm, nav_client, skill_registry)
  - Module-global _ctx: PrimitiveContext | None for each module
  - init_primitives(ctx) sets the global context
  - Each function: type hints, docstring, wraps existing interface
  - No hardware → raise RuntimeError("No hardware connected") for motor primitives
  - No hardware → return sensible defaults for query primitives (empty lists, None)
  - walk_forward: poll get_position until distance reached or timeout
  - turn: poll get_heading until angle reached or timeout
  - measure_distance: extract from LaserScan at given angle
  - scan_360: return full LaserScan as (angle, distance) pairs

### T3: goal_decomposer.py [Gamma]
- Status: pending
- Files: vcli/cognitive/goal_decomposer.py, vcli/cognitive/__init__.py
- Tests: tests/harness/test_level43_goal_decomposer.py
- ACs: AC-1 to AC-6
- Details:
  - GoalDecomposer.__init__(backend: LLMBackend)
  - decompose(task, world_context) → GoalTree
  - System prompt: defines GoalTree JSON schema, lists available strategies + verify functions
  - Parse LLM response as JSON → validate against schema
  - Max 8 sub_goals (truncate if more)
  - Validate: each verify expression is parseable Python
  - Validate: each strategy references known strategy name
  - Fallback: if JSON parse fails → single SubGoal wrapping the entire task as skill match
  - cognitive/__init__.py: export all classes

## Wave 2 — Core Logic (parallel, after Wave 1)

### T4: strategy_selector.py [Alpha]
- Status: pending
- Files: vcli/cognitive/strategy_selector.py
- Tests: tests/harness/test_level44_strategy_selector.py
- ACs: AC-21 to AC-24
- Details:
  - StrategySelector.__init__(skill_registry, primitives)
  - select(sub_goal) → StrategyResult(executor_type, executor_name, params)
  - Rule priority:
    1. sub_goal.strategy non-empty → use that directly
    2. name contains "reach"/"navigate" → navigate skill
    3. name contains "observe"/"look"/"scan" → look skill
    4. name contains "detect"/"find" → describe_scene skill with query
    5. name contains "stand"/"sit"/"stop" → corresponding skill
    6. name contains "walk"/"turn"/"move" → corresponding primitive
    7. Try skill_registry.match(sub_goal.description) → matched skill
    8. Fallback: return fallback result (log warning)
  - StrategyResult dataclass: executor_type ("skill"|"primitive"|"fallback"), name, params

### T5: goal_executor.py [Beta]
- Status: pending
- Files: vcli/cognitive/goal_executor.py
- Tests: tests/harness/test_level45_goal_executor.py
- ACs: AC-25 to AC-30
- Details:
  - GoalExecutor.__init__(strategy_selector, verifier, skill_registry, primitives)
  - execute(goal_tree, on_step=None) → ExecutionTrace
  - Topological sort sub_goals by depends_on
  - For each sub_goal:
    1. strategy = selector.select(sub_goal)
    2. Execute strategy:
       - "skill": find skill in registry, call execute(params, context)
       - "primitive": call named primitive function
       - "fallback": log and skip
    3. verify = verifier.verify(sub_goal.verify)
    4. If verify True → StepRecord(success=True), continue
    5. If verify False AND fail_action exists → execute fail_action, re-verify
    6. If still False → StepRecord(success=False), abort remaining goals
    7. If timeout exceeded → StepRecord(success=False, error="timeout")
    8. Call on_step(step_record) if provided
  - Build ExecutionTrace from all StepRecords

## Wave 3 — Integration (after Wave 2)

### T6: VectorEngine + IntentRouter integration [Alpha]
- Status: pending
- Files: vcli/intent_router.py (modify), vcli/engine.py (modify), vcli/cli.py (modify)
- Tests: tests/harness/test_level46_vgg_integration.py
- ACs: AC-31 to AC-35
- Details:
  - IntentRouter.route() returns categories + complexity flag
  - Complexity rules:
    - Keywords: "然后", "并且", "如果", "检查所有", "每个", "all rooms", "then", "and then", "if"
    - Multi-clause: contains Chinese comma or "，" separating action phrases
    - Simple override: single-word commands always simple
  - VectorEngine: if complex → GoalDecomposer → GoalExecutor pipeline
  - VectorEngine: if simple → existing tool_use path (unchanged)
  - cli.py: construct PrimitiveContext from agent, pass to engine
  - cli.py: display ExecutionTrace steps with timing (like existing tool display)
