"""Core agent engine — pure Python, no ROS2 imports.

Contains:
- types.py   : Shared data types (Pose3D, Detection, SkillResult, etc.)
- config.py  : Configuration loading and validation
- agent.py   : Agent class (main entry point) — Task 7
- planner.py : LLM task planner — Task 4/7
- executor.py: Deterministic task executor — Task 3
- world_model.py: World state management — Task 3
- skill.py   : Skill protocol, registry, predicates — Task 3
"""
