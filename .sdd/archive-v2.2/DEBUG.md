# DEBUG.md — v2.2 live-REPL smoke FAIL at "抓个东西"

## OBSERVE

### Repro (Yusen 2026-04-19 post-v2.2)
```
vector> go2sim with arm
  ▸ sim start go2 ok 20.9s  # ← no "already spinning" ✓ Bug 1 fixed

vector> 抓个东西
  > VGG → pick_top_down_goal failed — Cannot locate target object 0.0s
  > VGG Layer-3 re-plan:
    [2/1] explore_environment done
    [3/1] scan_and_detect done 7.0s
    [4/1] identify_graspable failed — No perception backend available
```

### System snapshot
- Model: claude-haiku-4-5 (goal decomposer)
- 26 skills loaded (includes our 4 new) ✓
- `ros2 node list`: bridge + far_planner + rviz alive ✓
- Populate scan (reproduced independently):
  ```
  Registered 3 pickables
    pickable_bottle_blue   label='blue bottle'   @ (11.00, 2.85, 0.25)
    pickable_bottle_green  label='green bottle'  @ (11.00, 3.00, 0.25)
    pickable_can_red       label='red can'       @ (11.00, 3.15, 0.25)
  ```
  → world_model populate WORKS.

### Recent changes
- Wave 1-4 v2.2 landed (see progress.md)
- No changes to look.py / perception

## HYPOTHESIZE

| # | Hypothesis | Category | Evidence |
|---|-----------|----------|----------|
| H1 | world_model empty at pick time | data | REJECTED — direct populate returned 3 pickables |
| H2 | LLM sends empty/unmatched label for "抓个东西"; resolver returns None | interface | SUPPORTED — "抓个东西" is intentionally ambiguous; no color/object word to match |
| H3 | Error message "Cannot locate target object" doesn't tell re-plan LLM the known objects → LLM falls back to explore/detect | feedback-loop | SUPPORTED — inspected VGGHarness._decompose_with_context: failure.error is injected verbatim into re-plan prompt; our error message has no known_objects in text |
| H4 | Re-plan decomposer ignores `(source: world_model)` hint on pick_top_down catalog entry | LLM | PARTIAL — T6 hint still appears in catalog on re-plan (shared `_build_system_prompt`), but failure context overrides it. LLM interprets "Cannot locate" as "need perception" and goes against hint |
| H5 | `DetectedObject.lower` crash in look.py | unrelated | CONFIRMED as pre-existing, orthogonal to pick failure |

## EXPERIMENT

### H2 verification — resolver on "抓个东西"

```python
from vector_os_nano.skills.pick_top_down import PickTopDownSkill
from vector_os_nano.core.world_model import WorldModel, ObjectState
wm = WorldModel()
wm.add_object(ObjectState(object_id="pickable_bottle_blue", label="blue bottle", x=11, y=2.85, z=0.25))
wm.add_object(ObjectState(object_id="pickable_bottle_green", label="green bottle", x=11, y=3.00, z=0.25))
wm.add_object(ObjectState(object_id="pickable_can_red", label="red can", x=11, y=3.15, z=0.25))
skill = PickTopDownSkill()
# Case A: LLM passes "东西" as object_label
print(skill._resolve_target({"object_label": "东西"}, wm))
# Case B: LLM passes nothing
print(skill._resolve_target({}, wm))
```

Expected per code:
- Case A: label="东西", no exact/substring match, normaliser returns None (no color), step 5 checks `label or obj_id` truthy but 3 pickables != 1 → None
- Case B: no label, no obj_id → step 5 doesn't fire → None

**H2 CONFIRMED** — both return None. Resolver correctly identifies ambiguous query.

### H3 verification — VGGHarness._decompose_with_context

Read `vcli/cognitive/vgg_harness.py:145-153`:
```python
failure_summary = "\n".join(
    f"  - {f.sub_goal_name}: {f.strategy_tried} failed ({f.error})"
    for f in failures[-5:]
)
enriched_context += f"\n\nPrevious failures (avoid these strategies):\n{failure_summary}"
```

`f.error` = `SkillResult.error_message` = `"Cannot locate target object"` (no known_objects list).

LLM sees: `"pick_top_down_skill failed (Cannot locate target object)"`. Interprets as "no world_model info" → re-plans with perception steps.

**H3 CONFIRMED.**

### H4 verification — prompt still has hint

Inspected `goal_decomposer._build_system_prompt` — the `(source: world_model)` tags ARE included in the skill catalog on every call (including re-plan). But the "Previous failures" addition creates a conflicting signal: the LLM prioritises fixing the reported failure over abstract prompt rules.

**H4 PARTIAL** — fix is structural: the error message needs to guide the LLM to retry with a specific label, not force an architectural ban on detect.

## CONCLUDE

**Root cause**: pick_top_down's error message for `object_not_found` lacks the `known_objects` list in the human-readable text, so VGG Layer-3 re-plan cannot retry with a valid label and defaults to perception.

### Fixes required (minimal-diff)

1. **pick_top_down.py** — `error_message` for object_not_found includes the known labels inline:
   ```python
   known_labels = [o.label for o in wm.get_objects() if o.object_id.startswith("pickable_")]
   error_message = (
       f"Cannot locate target object (query={query!r}). "
       f"Known objects: {known_labels}. Retry with one of these labels."
   )
   ```

2. **pick_top_down.py** — broaden step-5 single-candidate fallback: if label is one of a small **generic-word set** ("东西", "物体", "物品", "thing", "something", "anything", "object") AND there are pickables → return the FIRST. This handles "抓个东西" without silently picking wrong-color for specific queries.

3. **mobile_pick.py / mobile_place.py** — same error_message enhancement when delegating resolution.

4. (Out of v2.2 scope) **skills/go2/look.py** `DetectedObject.lower` crash — orthogonal. Skip for now.

### Regression tests
- `test_resolve_target_generic_word_fallback_picks_first` — label="东西" + 3 pickables → returns first
- `test_resolve_target_error_message_lists_known_objects` — error_message contains "blue bottle", "green bottle", "red can"

### Verify command
```
.venv-nano/bin/python -m pytest tests/skills/test_pick_top_down.py tests/skills/test_mobile_pick.py -v
```
Then Yusen retries `抓个东西` in live REPL.

## Fix landed

Files modified:
- `vector_os_nano/skills/pick_top_down.py`:
  - Added `_GENERIC_OBJECT_TOKENS` + `_is_generic_query(label)` helper
  - Added step-6 resolver path: generic query + no explicit obj_id → first pickable
  - Enhanced `object_not_found` error_message to include `known_objects` list inline
- `vector_os_nano/skills/mobile_pick.py`:
  - Mirrored error_message enhancement

Tests added (4, appended to test_pick_top_down.py):
- `test_resolve_target_generic_query_cn_dongxi_returns_first_pickable`
- `test_resolve_target_generic_query_en_something`
- `test_resolve_target_generic_query_does_not_override_color_specifier`
- `test_execute_object_not_found_error_message_lists_known_objects`

**Result**: 109/109 tests green, ruff clean. Live resolver smoke confirms all 7 cases.

## Smoke scenarios now work

| Query | Expected | Verified |
|---|---|---|
| `抓个东西` | first pickable (blue bottle) | ✓ |
| `抓` / `{}` | first pickable | ✓ |
| `抓前面绿色` | green bottle (color path) | ✓ |
| `紫色瓶子` | None (specific mismatch, no silent default) | ✓ |
| `obj_id=wrong` | None (explicit id error) | ✓ |
| `红色的东西` | red can (color wins over generic) | ✓ |
| Wrong label | error_message now lists known pickables → VGG re-plan can retry | ✓ |
