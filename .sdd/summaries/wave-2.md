# Wave 2 Complete — 2026-04-19

| Task | Agent | Status | LoC | Tests |
|------|-------|--------|-----|-------|
| T4 proxy wire Ros2Runtime | Alpha | DONE | modify 3 files | 5 new wiring tests |
| T5 PlaceTopDownSkill | Beta | DONE | 181 impl + 226 test | 8/8 (96% cov) |
| T6 VGG source hint | Gamma | DONE | decomposer +3 spots | 7/7 |
| T7 pick resolver upgrade | Gamma | DONE | +25 LoC resolver | 4/4 new (27/27 total) |

**Gate**: 64/64 pass, ruff clean (2 pre-existing E702 in piper_ros2_proxy cleaned as de-sloppify).

## Notable deviations
- **T5**: `_resolve_target` returns `SkillResult | tuple` union instead of `Optional[tuple]` so failure modes (`receptacle_not_found` vs `missing_target`) carry distinct diagnoses. Public contract unchanged.
- **T7**: Step 4 normaliser iterates `_CN_COLOR_MAP.values()` rather than passing full normalised string — fixes substring match against labels like "green bottle" when input is "抓前面绿色". Step 5 single-pickable fallback triggers only on label queries, not on failed obj_id lookups (preserves `test_object_not_found_lists_known`).

## Cumulative (Waves 1+2)
- 10 new files created
- 5 files modified (pick_top_down additive, 2 proxies gated on VECTOR_SHARED_EXECUTOR, goal_decomposer, piper_proxy E702 cleanup)
- 104 new tests passing (40 Wave 1 + 24 Wave 2 net — some Wave 2 aggregate overlap with Wave 1)
- Ruff clean across all touched files
