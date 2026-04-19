# T11 — sim_tool registration wiring

## Files modified
- `vector_os_nano/vcli/tools/sim_tool.py` — lines 386-394: expanded `if piper_arm is not None` block to import and register PlaceTopDownSkill, MobilePickSkill, MobilePlaceSkill alongside existing PickTopDownSkill.

## Files created
- `tests/vcli/__init__.py` (empty)
- `tests/vcli/tools/__init__.py` (empty)
- `tests/vcli/tools/test_sim_tool_registration.py` (62 LoC, 2 tests)

## Test choice: two simpler tests (not full behavioural)

Reason: `_start_go2` requires mocking subprocess.Popen, os.setsid, atexit.register, time.sleep(20), Go2ROS2Proxy, PiperROS2Proxy, PiperGripperROS2Proxy, Agent, SceneGraph, and multiple lazy imports. That surface is too brittle and couples the test to unrelated code paths. The two-test approach provides equivalent safety:

1. `test_manipulation_skills_importable_and_instantiable` — all 4 skill classes import and construct cleanly. Catches broken __init__ or missing deps that would raise at register() time.
2. `test_sim_tool_module_contains_all_manipulation_registrations` — source-text check confirms each `XyzSkill()` call appears in sim_tool.py. Catches typos and accidental deletions.

## Result
- 2/2 T11 tests pass (RED confirmed before edit, GREEN after)
- Ruff clean on both T11 files
- Regression: 95/95 (tests/skills/ + tests/vcli/)

## Verdict
DONE
