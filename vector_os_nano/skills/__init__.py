"""Built-in robot skills.

Each skill implements the Skill protocol (vector_os_nano.core.skill).
Skills are pure Python — no ROS2 imports.

Built-in skills (Task 6):
- pick.py   : PickSkill
- place.py  : PlaceSkill
- home.py   : HomeSkill
- scan.py   : ScanSkill
- detect.py : DetectSkill

Usage::

    from vector_os_nano.skills import get_default_skills
    from vector_os_nano.core.skill import SkillRegistry

    registry = SkillRegistry()
    for skill in get_default_skills():
        registry.register(skill)
"""
from __future__ import annotations

from vector_os_nano.skills.detect import DetectSkill
from vector_os_nano.skills.home import HomeSkill
from vector_os_nano.skills.pick import PickSkill
from vector_os_nano.skills.place import PlaceSkill
from vector_os_nano.skills.scan import ScanSkill

__all__ = [
    "DetectSkill",
    "HomeSkill",
    "PickSkill",
    "PlaceSkill",
    "ScanSkill",
    "get_default_skills",
]


def get_default_skills() -> list:
    """Return one instance of each built-in skill.

    Convenience factory for bootstrapping a SkillRegistry with all
    standard SO-101 capabilities.

    Returns:
        List of skill instances: [HomeSkill, ScanSkill, DetectSkill,
                                   PickSkill, PlaceSkill]
    """
    return [
        HomeSkill(),
        ScanSkill(),
        DetectSkill(),
        PickSkill(),
        PlaceSkill(),
    ]
