"""StrategySelector — maps SubGoals to execution strategies.

Priority order:
1. sub_goal.strategy is non-empty → resolve explicitly
2. Name/description keyword rules (navigate, observe, detect, etc.)
3. skill_registry.match(description) if a registry is injected
4. Fallback result
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vector_os_nano.vcli.cognitive.types import SubGoal


# ---------------------------------------------------------------------------
# Primitive function names (used in _resolve_explicit)
# ---------------------------------------------------------------------------

_PRIMITIVE_NAMES: frozenset[str] = frozenset({
    "walk_forward",
    "turn",
    "scan_360",
    "stop",
})

# Strategy names that end with "_skill" map to skill executor_type.
# All others that appear in KNOWN_STRATEGIES without "_skill" suffix are
# either primitives (above) or treated as skills by default.

_SKILL_SUFFIX = "_skill"


@dataclass(frozen=True)
class StrategyResult:
    """Result of strategy selection."""

    executor_type: str   # "skill" | "primitive" | "fallback"
    name: str            # skill name or primitive function name
    params: dict         # parameters to pass


class StrategySelector:
    """Maps SubGoals to execution strategies using rule-based matching."""

    def __init__(self, skill_registry: Any = None) -> None:
        self._skill_registry = skill_registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, sub_goal: SubGoal) -> StrategyResult:
        """Select best execution strategy for a sub-goal.

        Priority:
        1. sub_goal.strategy is non-empty → use directly
        2. Name-based rules (reach→navigate, observe→look, etc.)
        3. Try skill_registry.match(description)
        4. Fallback
        """
        # Priority 1: Explicit strategy from GoalTree
        if sub_goal.strategy:
            return self._resolve_explicit(sub_goal.strategy, sub_goal.strategy_params)

        # Priority 2: Name/description keyword rules
        name_lower = sub_goal.name.lower()
        desc_lower = sub_goal.description.lower()
        combined = name_lower + " " + desc_lower

        # Navigation
        if any(kw in combined for kw in ("reach", "navigate", "go_to", "到", "去")):
            room = sub_goal.strategy_params.get("room", sub_goal.description)
            return StrategyResult("skill", "navigate", {"room": room})

        # Observation
        if any(kw in combined for kw in ("observe", "look", "scan", "看", "观察")):
            return StrategyResult("skill", "look", {})

        # Detection
        if any(kw in combined for kw in ("detect", "find", "check", "检测", "找", "检查")):
            query = sub_goal.strategy_params.get("query", sub_goal.description)
            return StrategyResult("skill", "describe_scene", {"query": query})

        # Stance — stand / sit
        if any(kw in combined for kw in ("stand",)):
            return StrategyResult("skill", "stand", {})
        if any(kw in combined for kw in ("sit",)):
            return StrategyResult("skill", "sit", {})

        # Stop (primitive before walk to avoid 'stop' being caught by nothing)
        if any(kw in combined for kw in ("stop",)):
            return StrategyResult("primitive", "stop", {})

        # Movement primitives
        if any(kw in combined for kw in ("walk", "forward", "前进")):
            dist = sub_goal.strategy_params.get("distance", 1.0)
            return StrategyResult("primitive", "walk_forward", {"distance_m": dist})
        if any(kw in combined for kw in ("turn", "rotate", "转")):
            angle = sub_goal.strategy_params.get("angle", 1.57)
            return StrategyResult("primitive", "turn", {"angle_rad": angle})

        # Priority 3: Skill registry alias match
        if self._skill_registry is not None:
            match = self._skill_registry.match(sub_goal.description)
            if match is not None:
                return StrategyResult("skill", match.skill_name, {})

        # Priority 4: Fallback
        return StrategyResult("fallback", "unmatched", {"sub_goal": sub_goal.name})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_explicit(self, strategy: str, params: dict) -> StrategyResult:
        """Resolve an explicit strategy name to a StrategyResult.

        Mapping rules:
        - Ends with "_skill" → executor_type="skill", name=strategy minus suffix.
        - In _PRIMITIVE_NAMES → executor_type="primitive".
        - Otherwise → executor_type="skill" (assume skill by name convention).
        """
        if strategy.endswith(_SKILL_SUFFIX):
            skill_name = strategy[: -len(_SKILL_SUFFIX)]
            return StrategyResult("skill", skill_name, params)

        if strategy in _PRIMITIVE_NAMES:
            return StrategyResult("primitive", strategy, params)

        # Treat as a skill with the strategy name as-is
        return StrategyResult("skill", strategy, params)
