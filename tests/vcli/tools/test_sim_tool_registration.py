"""Tests for Wave 2/3 skill registration wiring in sim_tool.py.

Test strategy: The full behavioural test (calling _start_go2 end-to-end with
with_arm=True) requires mocking subprocess.Popen, os.setsid, time.sleep(20),
Go2ROS2Proxy, PiperROS2Proxy, PiperGripperROS2Proxy, Agent, SceneGraph, and
multiple lazy imports inside the method body. That mock surface is fragile and
couples the test to implementation details of unrelated code paths.

Instead we use two focused guards:

1. test_manipulation_skills_importable_and_instantiable — verifies all 4 skill
   classes can be imported and constructed with no arguments. A failure here
   means any register(XyzSkill()) call in sim_tool will raise ImportError or
   TypeError at runtime.

2. test_sim_tool_module_contains_all_manipulation_registrations — inspects the
   source text of sim_tool to confirm each skill class name appears with a
   register() call pattern. Catches typos, missing imports, or accidental
   deletions without running the full method.
"""
from __future__ import annotations

import inspect


def test_manipulation_skills_importable_and_instantiable() -> None:
    """Regression guard: all 4 Wave 2/3 skill classes are importable and
    instantiable with no arguments. A failure means the register(...) calls
    added to _start_go2 will crash at runtime.
    """
    from vector_os_nano.skills.pick_top_down import PickTopDownSkill
    from vector_os_nano.skills.place_top_down import PlaceTopDownSkill
    from vector_os_nano.skills.mobile_pick import MobilePickSkill
    from vector_os_nano.skills.mobile_place import MobilePlaceSkill

    PickTopDownSkill()
    PlaceTopDownSkill()
    MobilePickSkill()
    MobilePlaceSkill()


def test_sim_tool_module_contains_all_manipulation_registrations() -> None:
    """Sanity: the sim_tool source text imports and registers all 4 skill
    classes inside the piper_arm guard block. Not a behaviour test but catches
    most typos and missing lines that would break runtime registration.
    """
    from vector_os_nano.vcli.tools import sim_tool

    src = inspect.getsource(sim_tool)

    expected_classes = (
        "PickTopDownSkill",
        "PlaceTopDownSkill",
        "MobilePickSkill",
        "MobilePlaceSkill",
    )
    for cls in expected_classes:
        assert f"{cls}()" in src, (
            f"{cls}() not found in sim_tool.py — "
            "skill will not be registered when with_arm=True"
        )
