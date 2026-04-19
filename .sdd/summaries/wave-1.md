# Wave 1 Complete — 2026-04-19

| Task | Agent | Status | LoC | Tests |
|------|-------|--------|-----|-------|
| T1 Ros2Runtime | Alpha | DONE | 173 (impl) + 226 (test) | 5/5 |
| T2 approach_pose | Beta | DONE | 66 (impl) + 121 (test) | 12/12 |
| T3 color normaliser | Gamma | DONE | +40 (impl) + append (test) | 10/10 new (23/23 total on pick_top_down) |

**Gate**: 40/40 pass, ruff clean, pre-existing F401 also cleaned.
Files created: `hardware/ros2/{__init__,runtime}.py`, `skills/utils/{__init__,approach_pose}.py`, `tests/hardware/ros2/*`, `tests/skills/utils/*`.
Files modified: `skills/pick_top_down.py` (additive + F401 cleanup), `tests/skills/test_pick_top_down.py` (append 5 tests).
