"""Go2 quadruped skill package.

Exports all Go2 skills and the get_go2_skills() factory.
"""
from vector_os_nano.skills.go2.walk import WalkSkill
from vector_os_nano.skills.go2.turn import TurnSkill
from vector_os_nano.skills.go2.stance import StandSkill, SitSkill, LieDownSkill
from vector_os_nano.skills.go2.explore import ExploreSkill
from vector_os_nano.skills.go2.where_am_i import WhereAmISkill
from vector_os_nano.skills.go2.stop import StopSkill
from vector_os_nano.skills.go2.look import LookSkill, DescribeSceneSkill
from vector_os_nano.skills.go2.patrol import PatrolSkill
from vector_os_nano.skills.navigate import NavigateSkill


def get_go2_skills() -> list:
    """Return one instance of each Go2 skill."""
    return [
        WalkSkill(), TurnSkill(),
        StandSkill(), SitSkill(), LieDownSkill(),
        NavigateSkill(),
        ExploreSkill(),
        WhereAmISkill(),
        StopSkill(),
        LookSkill(),
        DescribeSceneSkill(),
        PatrolSkill(),
    ]


__all__ = [
    "WalkSkill",
    "TurnSkill",
    "StandSkill",
    "SitSkill",
    "LieDownSkill",
    "NavigateSkill",
    "ExploreSkill",
    "WhereAmISkill",
    "StopSkill",
    "LookSkill",
    "DescribeSceneSkill",
    "PatrolSkill",
    "get_go2_skills",
]
