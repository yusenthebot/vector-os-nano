# Task T7 — `_resolve_target` Fallback Passes

## Files Modified

1. `vector_os_nano/skills/pick_top_down.py`
2. `tests/skills/test_pick_top_down.py`

## Diff Summary

### `pick_top_down.py` — `_resolve_target` only

- Steps 1-3 unchanged.
- **Step 4 added (~12 lines)**: Chinese color normaliser pass. Calls `_normalise_color_keyword(label)` to get normalised string, then iterates English color values from `_CN_COLOR_MAP` to find which appeared in the normalised string, and calls `get_objects_by_label(en_color)` for each. Returns on first match with INFO log.
- **Step 5 added (~9 lines)**: Single-pickable fallback. Triggered only when `label` was given but nothing matched (explicit `obj_id` failures are not substituted — preserves `test_object_not_found_lists_known`). If exactly one `pickable_*` object exists, returns it with INFO log.
- Total: ~25 lines added, 0 removed from `_resolve_target`.

### `tests/skills/test_pick_top_down.py` — 4 tests appended

- `test_resolve_target_matches_chinese_color_after_normalise`
- `test_resolve_target_single_pickable_fallback_when_label_unmatched`
- `test_resolve_target_no_fallback_when_multiple_pickables_and_unmatched`
- `test_resolve_target_prefers_explicit_label_match_over_color_normalise`

## Implementation Note

`_normalise_color_keyword("抓前面绿色")` returns `"抓前面green"`. Calling `get_objects_by_label("抓前面green")` against a label "green bottle" yields no match (substring check fails both ways with Chinese chars present). Fix: iterate `_CN_COLOR_MAP.values()`, check which English color appears in the normalised string, and query with that color token directly (e.g., "green"). This matches "green bottle" via substring.

Step 5 fallback is restricted to label-only queries. Explicit `object_id` misses remain `object_not_found`.

## Test Output

```
27 passed in 4.48s
```

## Ruff Output

```
All checks passed!
```

## Status

DONE
