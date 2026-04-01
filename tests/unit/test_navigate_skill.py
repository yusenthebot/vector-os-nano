"""Tests for unified NavigateSkill -- hardware-agnostic navigation."""
import pytest
from unittest.mock import MagicMock
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.core.types import SkillResult


def _make_context(with_nav=False):
    base = MagicMock()
    base.walk.return_value = True
    base.get_position.return_value = [10.0, 3.0, 0.27]
    base.get_heading.return_value = 0.0
    services = {}
    if with_nav:
        nav = MagicMock()
        nav.is_available = True
        nav.navigate_to.return_value = True
        services["nav"] = nav
    return SkillContext(
        bases={"go2": base},
        world_model=WorldModel(),
        services=services,
    )


class TestNavigateSkillMetadata:
    def test_skill_name(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        skill = NavigateSkill()
        assert skill.name == "navigate"

    def test_aliases(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        skill = NavigateSkill()
        assert "去" in skill.__class__.__skill_aliases__
        assert "navigate" in skill.__class__.__skill_aliases__

    def test_has_required_metadata(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        skill = NavigateSkill()
        assert hasattr(skill, "description")
        assert hasattr(skill, "parameters")
        assert "room" in skill.parameters
        assert hasattr(skill, "failure_modes")
        assert "no_base" in skill.failure_modes
        assert "unknown_room" in skill.failure_modes
        assert "navigation_failed" in skill.failure_modes


class TestNavigateWithNavStack:
    def test_uses_nav_client_when_available(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=True)
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert result.success
        # Should have called nav client, not dead-reckoning walk
        ctx.services["nav"].navigate_to.assert_called_once()

    def test_nav_client_receives_coordinates(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=True)
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        args = ctx.services["nav"].navigate_to.call_args[0]
        # Kitchen center is (17.0, 2.5)
        assert 15 < args[0] < 19
        assert 1 < args[1] < 4

    def test_nav_client_failure_returns_navigation_failed(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=True)
        ctx.services["nav"].navigate_to.return_value = False
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert not result.success
        assert "navigation_failed" in result.diagnosis_code

    def test_nav_unavailable_falls_back_to_dead_reckoning(self):
        """nav service present but is_available=False -> falls back to dead-reckoning."""
        from vector_os_nano.skills.navigate import NavigateSkill
        nav = MagicMock()
        nav.is_available = False
        nav.navigate_to.return_value = True
        ctx = SkillContext(
            bases={"go2": MagicMock(
                walk=MagicMock(return_value=True),
                get_position=MagicMock(return_value=[10.0, 3.0, 0.27]),
                get_heading=MagicMock(return_value=0.0),
            )},
            world_model=WorldModel(),
            services={"nav": nav},
        )
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        # Dead-reckoning path; nav.navigate_to must NOT be called
        nav.navigate_to.assert_not_called()

    def test_nav_unknown_room_not_sent_to_nav_client(self):
        """Unknown room should fail before reaching nav client."""
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=True)
        skill = NavigateSkill()
        result = skill.execute({"room": "mars_base"}, ctx)
        assert not result.success
        assert "unknown_room" in result.diagnosis_code
        ctx.services["nav"].navigate_to.assert_not_called()


class TestNavigateDeadReckoning:
    def test_fallback_without_nav_stack(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=False)
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert result.success
        # Should have called base.walk (dead-reckoning)
        assert ctx.base.walk.called

    def test_unknown_room(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=False)
        skill = NavigateSkill()
        result = skill.execute({"room": "nonexistent_room"}, ctx)
        assert not result.success
        assert "unknown_room" in result.diagnosis_code

    def test_chinese_room_name(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=False)
        skill = NavigateSkill()
        result = skill.execute({"room": "厨房"}, ctx)
        assert result.success

    def test_no_base(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = SkillContext(world_model=WorldModel())
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert not result.success
        assert "no_base" in result.diagnosis_code

    def test_already_at_destination(self):
        """Robot already near target room center -- should succeed immediately."""
        from vector_os_nano.skills.navigate import NavigateSkill
        # Position (3.0, 2.5) is the living_room center
        base = MagicMock()
        base.walk.return_value = True
        base.get_position.return_value = [3.0, 2.5, 0.27]
        base.get_heading.return_value = 0.0
        ctx = SkillContext(bases={"go2": base}, world_model=WorldModel())
        skill = NavigateSkill()
        result = skill.execute({"room": "living_room"}, ctx)
        assert result.success
        assert result.result_data.get("note") == "already here"

    def test_dead_reckoning_result_data(self):
        """Dead-reckoning success should include room and position in result_data."""
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=False)
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert result.success
        assert result.result_data is not None
        assert "room" in result.result_data
        assert result.result_data["room"] == "kitchen"

    def test_alias_english_shorthand(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=False)
        skill = NavigateSkill()
        # "bedroom" is an alias for master_bedroom
        result = skill.execute({"room": "bedroom"}, ctx)
        assert result.success
        assert result.result_data["room"] == "master_bedroom"

    def test_empty_room_param(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=False)
        skill = NavigateSkill()
        result = skill.execute({"room": ""}, ctx)
        assert not result.success
        assert "unknown_room" in result.diagnosis_code

    def test_missing_room_param(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context(with_nav=False)
        skill = NavigateSkill()
        result = skill.execute({}, ctx)
        assert not result.success
        assert "unknown_room" in result.diagnosis_code


class TestNavigateFromGo2Package:
    def test_go2_skills_includes_navigate(self):
        from vector_os_nano.skills.go2 import get_go2_skills
        skills = get_go2_skills()
        names = {s.name for s in skills}
        assert "navigate" in names

    def test_go2_navigate_is_unified(self):
        """The NavigateSkill from go2 package should be the unified one."""
        from vector_os_nano.skills.go2 import NavigateSkill as Go2Nav
        from vector_os_nano.skills.navigate import NavigateSkill as UnifiedNav
        assert Go2Nav is UnifiedNav

    def test_go2_navigate_module_removed(self):
        """go2/navigate.py shim has been removed -- import must fail."""
        import importlib
        import pytest
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("vector_os_nano.skills.go2.navigate")
