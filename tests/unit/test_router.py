"""Unit tests for vector_os_nano.llm.router.ModelRouter.

TDD — covers all routing logic and complexity heuristics.
Run with: pytest tests/unit/test_router.py -v
"""
from __future__ import annotations

import pytest

from vector_os_nano.llm.router import ModelRouter, ModelSelection

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONFIG: dict = {
    "llm": {
        "model": "anthropic/claude-haiku-4-5",
        "models": {
            "classify": "anthropic/claude-haiku-4-5",
            "plan_simple": "anthropic/claude-haiku-4-5",
            "plan_complex": "anthropic/claude-sonnet-4-6",
            "chat": "anthropic/claude-haiku-4-5",
            "summarize": "anthropic/claude-haiku-4-5",
        },
    }
}

SAMPLE_WORLD_STATE: dict = {
    "objects": [
        {"object_id": "banana_1", "label": "banana", "x": 0.15, "y": 0.10, "z": 0.02},
        {"object_id": "mug_1", "label": "mug", "x": 0.20, "y": -0.05, "z": 0.02},
    ],
    "robot": {"gripper_state": "open", "held_object": None},
}

_HAIKU = "anthropic/claude-haiku-4-5"
_SONNET = "anthropic/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Fallback and default behaviour
# ---------------------------------------------------------------------------


class TestDefaultModelFallback:
    """No 'models' key in config — every method should return the global default."""

    def setup_method(self) -> None:
        config = {"llm": {"model": _HAIKU}}
        self.router = ModelRouter(config)

    def test_classify_falls_back(self) -> None:
        sel = self.router.for_classify()
        assert sel.model == _HAIKU

    def test_plan_falls_back(self) -> None:
        sel = self.router.for_plan("pick the mug", SAMPLE_WORLD_STATE)
        assert sel.model == _HAIKU

    def test_chat_falls_back(self) -> None:
        sel = self.router.for_chat()
        assert sel.model == _HAIKU

    def test_summarize_falls_back(self) -> None:
        sel = self.router.for_summarize()
        assert sel.model == _HAIKU


class TestEmptyConfig:
    """Empty dict config — everything should return the hardcoded fallback."""

    def setup_method(self) -> None:
        self.router = ModelRouter({})

    def test_classify_returns_default(self) -> None:
        sel = self.router.for_classify()
        assert sel.model == "anthropic/claude-haiku-4-5"

    def test_plan_returns_default(self) -> None:
        sel = self.router.for_plan("pick it up", {})
        assert sel.model == "anthropic/claude-haiku-4-5"

    def test_chat_returns_default(self) -> None:
        sel = self.router.for_chat()
        assert sel.model == "anthropic/claude-haiku-4-5"

    def test_summarize_returns_default(self) -> None:
        sel = self.router.for_summarize()
        assert sel.model == "anthropic/claude-haiku-4-5"


class TestPartialConfig:
    """Only some stages configured — unset ones fall back to global default."""

    def setup_method(self) -> None:
        config = {
            "llm": {
                "model": _HAIKU,
                "models": {
                    "plan_complex": _SONNET,
                    # classify, plan_simple, chat, summarize intentionally absent
                },
            }
        }
        self.router = ModelRouter(config)

    def test_plan_complex_uses_configured_model(self) -> None:
        sel = self.router.for_plan(
            "put the mug on the left side of the banana",
            SAMPLE_WORLD_STATE,
        )
        assert sel.model == _SONNET

    def test_classify_falls_back_to_default(self) -> None:
        assert self.router.for_classify().model == _HAIKU

    def test_chat_falls_back_to_default(self) -> None:
        assert self.router.for_chat().model == _HAIKU

    def test_summarize_falls_back_to_default(self) -> None:
        assert self.router.for_summarize().model == _HAIKU


# ---------------------------------------------------------------------------
# Individual stage selectors
# ---------------------------------------------------------------------------


class TestClassifyModel:
    def test_returns_configured_classify_model(self) -> None:
        router = ModelRouter(SAMPLE_CONFIG)
        sel = router.for_classify()
        assert sel.model == _HAIKU
        assert sel.reason == "classify"


class TestChatModel:
    def test_returns_configured_chat_model(self) -> None:
        router = ModelRouter(SAMPLE_CONFIG)
        sel = router.for_chat()
        assert sel.model == _HAIKU
        assert sel.reason == "chat"


class TestSummarizeModel:
    def test_returns_configured_summarize_model(self) -> None:
        router = ModelRouter(SAMPLE_CONFIG)
        sel = router.for_summarize()
        assert sel.model == _HAIKU
        assert sel.reason == "summarize"


# ---------------------------------------------------------------------------
# Plan routing — simple vs complex
# ---------------------------------------------------------------------------


class TestPlanSimple:
    def test_short_single_object_returns_simple_model(self) -> None:
        router = ModelRouter(SAMPLE_CONFIG)
        sel = router.for_plan("pick the mug", SAMPLE_WORLD_STATE)
        assert sel.model == _HAIKU
        assert sel.reason == "simple_task"


class TestPlanComplexSpatial:
    def test_spatial_word_triggers_complex(self) -> None:
        router = ModelRouter(SAMPLE_CONFIG)
        # Spatial word ("left") + multi-object mention ("mug" + "banana") → score 2
        sel = router.for_plan(
            "put the mug on the left side of the banana", SAMPLE_WORLD_STATE
        )
        assert sel.model == _SONNET
        assert sel.reason == "complex_task"

    def test_right_spatial_word_triggers_complex(self) -> None:
        router = ModelRouter(SAMPLE_CONFIG)
        # spatial + multi-object ("mug" + "banana" both in world state)
        sel = router.for_plan(
            "move the mug to the right of the banana", SAMPLE_WORLD_STATE
        )
        assert sel.model == _SONNET

    def test_between_triggers_complex(self) -> None:
        router = ModelRouter(SAMPLE_CONFIG)
        sel = router.for_plan(
            "place the mug between the banana and the cup", SAMPLE_WORLD_STATE
        )
        assert sel.model == _SONNET


class TestPlanComplexMultiAction:
    def test_then_keyword_triggers_complex(self) -> None:
        router = ModelRouter(SAMPLE_CONFIG)
        # "then" + length > 50 chars
        sel = router.for_plan(
            "pick the banana then put it on the left side", SAMPLE_WORLD_STATE
        )
        assert sel.model == _SONNET
        assert sel.reason == "complex_task"

    def test_after_that_triggers_complex(self) -> None:
        router = ModelRouter(SAMPLE_CONFIG)
        sel = router.for_plan(
            "pick the mug, after that put it down gently on the table",
            SAMPLE_WORLD_STATE,
        )
        assert sel.model == _SONNET

    def test_first_then_pattern_triggers_complex(self) -> None:
        router = ModelRouter(SAMPLE_CONFIG)
        # Multi-action ("then") + multi-object mention ("banana" + "mug") → score 2
        sel = router.for_plan(
            "first detect the banana then put it next to the mug", SAMPLE_WORLD_STATE
        )
        assert sel.model == _SONNET


class TestPlanComplexMultiObject:
    def test_two_object_labels_triggers_complex(self) -> None:
        """Instruction mentions both 'banana' and 'mug' — multi-object indicator."""
        router = ModelRouter(SAMPLE_CONFIG)
        # Two labels match world state: "banana" and "mug" — score >= 2
        sel = router.for_plan(
            "put the banana next to the mug", SAMPLE_WORLD_STATE
        )
        assert sel.model == _SONNET
        assert sel.reason == "complex_task"


class TestPlanComplexManyObjects:
    def test_four_objects_in_world_triggers_complex(self) -> None:
        """4+ visible objects in world state → large planning space → complex."""
        world = {
            "objects": [
                {"object_id": "a", "label": "apple"},
                {"object_id": "b", "label": "bottle"},
                {"object_id": "c", "label": "cup"},
                {"object_id": "d", "label": "dice"},
            ]
        }
        router = ModelRouter(SAMPLE_CONFIG)
        # Instruction is short but world has 4 objects; combine with length rule
        # to reach score 2: 4 objects (score 1) + long enough instruction (score 1)
        sel = router.for_plan(
            "pick something up from the workspace and place it somewhere else",
            world,
        )
        assert sel.model == _SONNET


class TestPlanComplexLongInstruction:
    def test_long_instruction_plus_spatial_triggers_complex(self) -> None:
        """Length > 50 chars acts as one indicator; pair with spatial for score 2."""
        router = ModelRouter(SAMPLE_CONFIG)
        long_spatial = (
            "carefully pick the mug from the right side of the table and set it down"
        )
        assert len(long_spatial) > 50
        sel = router.for_plan(long_spatial, SAMPLE_WORLD_STATE)
        assert sel.model == _SONNET


# ---------------------------------------------------------------------------
# ModelSelection immutability
# ---------------------------------------------------------------------------


class TestModelSelectionFrozen:
    def test_cannot_mutate_model(self) -> None:
        sel = ModelSelection(model=_HAIKU, reason="test")
        with pytest.raises((AttributeError, TypeError)):
            sel.model = _SONNET  # type: ignore[misc]

    def test_cannot_mutate_reason(self) -> None:
        sel = ModelSelection(model=_HAIKU, reason="test")
        with pytest.raises((AttributeError, TypeError)):
            sel.reason = "other"  # type: ignore[misc]

    def test_model_selection_equality(self) -> None:
        a = ModelSelection(model=_HAIKU, reason="classify")
        b = ModelSelection(model=_HAIKU, reason="classify")
        assert a == b


# ---------------------------------------------------------------------------
# Chinese instruction complexity
# ---------------------------------------------------------------------------


class TestEstimateComplexityChinese:
    def test_chinese_multi_action_triggers_complex(self) -> None:
        """先.*再 (multi-action) + label match for 香蕉 → score 2 → complex."""
        world = {
            "objects": [
                {"label": "香蕉"},
                {"label": "杯子"},
            ]
        }
        # "先扫描再抓香蕉" — multi-action (先.*再) score 1 + object label "香蕉" matched
        # + object label "香蕉" counted, second object "杯子" not mentioned → objects = 1
        # Need a second indicator: add 杯子 to instruction for multi-object score
        result = ModelRouter.estimate_complexity(
            "先扫描香蕉再把它放到杯子旁边", world
        )
        assert result == "complex"

    def test_chinese_then_triggers_complex(self) -> None:
        """然后 is a multi-action indicator."""
        result = ModelRouter.estimate_complexity(
            "拿起香蕉然后放到桌子上",
            {"objects": [{"label": "香蕉"}]},
        )
        assert result == "complex"

    def test_chinese_spatial_triggers_complex(self) -> None:
        """旁边 is spatial; pair with multi-object mention."""
        world = {
            "objects": [
                {"label": "苹果"},
                {"label": "杯子"},
            ]
        }
        result = ModelRouter.estimate_complexity(
            "把苹果放在杯子旁边", world
        )
        assert result == "complex"


# ---------------------------------------------------------------------------
# Empty world state
# ---------------------------------------------------------------------------


class TestEstimateComplexityEmptyWorld:
    def test_simple_instruction_empty_world_returns_simple(self) -> None:
        result = ModelRouter.estimate_complexity("pick the mug", {})
        assert result == "simple"

    def test_empty_instruction_empty_world_returns_simple(self) -> None:
        result = ModelRouter.estimate_complexity("", {})
        assert result == "simple"

    def test_very_short_instruction_no_indicators_returns_simple(self) -> None:
        result = ModelRouter.estimate_complexity("go home", {})
        assert result == "simple"
