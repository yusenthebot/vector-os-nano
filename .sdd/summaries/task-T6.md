# Task T6 — VGG source-aware decomposer hint

## Status: DONE

## Files Modified (1)

`vector_os_nano/vcli/cognitive/goal_decomposer.py`

- Lines 27-62: Added `_skill_is_world_model_only(skill)` module-level helper.
  Returns True iff skill.parameters has at least one `source` starting with `world_model.`
  and no `source` starting with `perception.`.
- Line 21: Removed pre-existing unused `from dataclasses import field` import (ruff fix).
- `__init__`: stored `self._skill_registry = skill_registry` (line ~270).
- `_build_system_prompt`: replaced the `strategies_block` one-liner with a
  `_strategy_line(name, desc)` closure that appends ` (source: world_model)` when
  the helper returns True for the matching skill object looked up from `_skill_registry`.
- Rules section: added rule #9 — the no-detect-* hint bullet (one line added after rule 8).

## Files Created (2)

`tests/vcli/cognitive/__init__.py` — empty, required for pytest collection.

`tests/vcli/cognitive/test_decomposer_source.py` — 7 tests:
- `TestSkillIsWorldModelOnly` (5 unit tests for the helper)
- `TestSkillCatalogIncludesSourceTag` (2 integration tests: catalog tag + prompt hint)

## Diff Summary (line ranges)

| Change | File | Lines |
|--------|------|-------|
| Helper function `_skill_is_world_model_only` | goal_decomposer.py | 27-62 |
| Store `_skill_registry` in `__init__` | goal_decomposer.py | ~272 |
| `_strategy_line` closure + `strategies_block` rewrite | goal_decomposer.py | `_build_system_prompt` (first 25 lines) |
| Rule #9 bullet | goal_decomposer.py | inside `_build_system_prompt` text f-string |

## Test Output

```
tests/vcli/cognitive/test_decomposer_source.py::TestSkillIsWorldModelOnly::test_skill_is_world_model_only_tags_world_model_source_skill PASSED
tests/vcli/cognitive/test_decomposer_source.py::TestSkillIsWorldModelOnly::test_skill_is_world_model_only_false_when_any_perception_source PASSED
tests/vcli/cognitive/test_decomposer_source.py::TestSkillIsWorldModelOnly::test_skill_is_world_model_only_false_when_no_world_model_source PASSED
tests/vcli/cognitive/test_decomposer_source.py::TestSkillIsWorldModelOnly::test_skill_is_world_model_only_false_when_no_parameters_attr PASSED
tests/vcli/cognitive/test_decomposer_source.py::TestSkillIsWorldModelOnly::test_skill_is_world_model_only_false_when_parameters_not_dict PASSED
tests/vcli/cognitive/test_decomposer_source.py::TestSkillCatalogIncludesSourceTag::test_skill_catalog_includes_source_tag PASSED
tests/vcli/cognitive/test_decomposer_source.py::TestSkillCatalogIncludesSourceTag::test_system_prompt_includes_no_detect_hint PASSED

7 passed in 0.08s

Full suite (36 tests including existing harness/test_level43_goal_decomposer.py): 36 passed
Ruff: All checks passed
```
