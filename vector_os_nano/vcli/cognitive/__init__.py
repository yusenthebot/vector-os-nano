"""VGG cognitive layer — types, verifier, decomposer, strategy selector, and executor.

Public surface::

    from vector_os_nano.vcli.cognitive import (
        SubGoal,
        GoalTree,
        StepRecord,
        ExecutionTrace,
        GoalVerifier,
        GoalDecomposer,
        GoalExecutor,
        StrategySelector,
        StrategyResult,
    )
"""
from __future__ import annotations

from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult, StrategySelector
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)

__all__ = [
    "ExecutionTrace",
    "GoalDecomposer",
    "GoalExecutor",
    "GoalTree",
    "GoalVerifier",
    "StepRecord",
    "StrategyResult",
    "StrategySelector",
    "SubGoal",
]
