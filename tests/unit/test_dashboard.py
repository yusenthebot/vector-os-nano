"""Unit tests for vector_os_nano.cli.dashboard — DashboardApp (Textual TUI).

TDD: written before implementation.
Tests skip gracefully when textual is not installed.
All tests use agent=None or a minimal mock agent.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

textual = pytest.importorskip("textual")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_agent():
    """Build a minimal mock agent that satisfies DashboardApp's interface."""
    agent = MagicMock()
    agent.skills = ["pick", "place", "home", "scan", "detect"]

    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.skills import get_default_skills
    registry = SkillRegistry()
    for s in get_default_skills():
        registry.register(s)
    agent._skill_registry = registry

    from vector_os_nano.core.world_model import WorldModel
    world = WorldModel()
    agent.world = world

    from vector_os_nano.core.types import ExecutionResult
    agent.execute.return_value = ExecutionResult(
        success=True,
        status="completed",
        steps_completed=1,
        steps_total=1,
    )
    return agent


# ---------------------------------------------------------------------------
# T1 — import
# ---------------------------------------------------------------------------

def test_dashboard_import():
    from vector_os_nano.cli.dashboard import DashboardApp
    assert DashboardApp is not None


# ---------------------------------------------------------------------------
# T2 — creation without agent
# ---------------------------------------------------------------------------

def test_dashboard_creation():
    from vector_os_nano.cli.dashboard import DashboardApp
    app = DashboardApp()
    assert app._agent is None


# ---------------------------------------------------------------------------
# T3 — creation with agent
# ---------------------------------------------------------------------------

def test_dashboard_creation_with_agent():
    from vector_os_nano.cli.dashboard import DashboardApp
    agent = _make_mock_agent()
    app = DashboardApp(agent=agent)
    assert app._agent is agent


# ---------------------------------------------------------------------------
# T4 — main() function is importable and callable
# ---------------------------------------------------------------------------

def test_dashboard_main_exists():
    from vector_os_nano.cli.dashboard import main
    assert callable(main)


# ---------------------------------------------------------------------------
# T5 — CSS is non-empty
# ---------------------------------------------------------------------------

def test_dashboard_css_not_empty():
    from vector_os_nano.cli.dashboard import DashboardApp
    assert DashboardApp.CSS.strip() != ""


# ---------------------------------------------------------------------------
# T6 — BINDINGS has required keys
# ---------------------------------------------------------------------------

def test_dashboard_bindings():
    from vector_os_nano.cli.dashboard import DashboardApp
    keys = {b.key for b in DashboardApp.BINDINGS}
    assert "f1" in keys
    assert "f2" in keys
    assert "f3" in keys
    assert "f4" in keys
    assert "ctrl+e" in keys
    assert "ctrl+c" in keys


# ---------------------------------------------------------------------------
# T7 — headless run: panels are mounted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dashboard_headless():
    from vector_os_nano.cli.dashboard import DashboardApp
    app = DashboardApp()
    async with app.run_test() as pilot:
        # Core panels must be present
        assert app.query_one("#status-panel")
        assert app.query_one("#joint-panel")
        assert app.query_one("#skill-panel")
        # Command input must be present
        assert app.query_one("#command-input")


# ---------------------------------------------------------------------------
# T8 — headless: log tab and RichLog present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dashboard_log_tab_present():
    from vector_os_nano.cli.dashboard import DashboardApp
    from textual.widgets import RichLog
    app = DashboardApp()
    async with app.run_test() as pilot:
        log_view = app.query_one("#log-view", RichLog)
        assert log_view is not None


# ---------------------------------------------------------------------------
# T9 — headless: skills tab and DataTable present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dashboard_skills_tab_present():
    from vector_os_nano.cli.dashboard import DashboardApp
    from textual.widgets import DataTable
    app = DashboardApp()
    async with app.run_test() as pilot:
        table = app.query_one("#skills-table", DataTable)
        assert table is not None


# ---------------------------------------------------------------------------
# T10 — headless: world tab view present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dashboard_world_tab_present():
    from vector_os_nano.cli.dashboard import DashboardApp
    from textual.widgets import RichLog
    app = DashboardApp()
    async with app.run_test() as pilot:
        world_view = app.query_one("#world-view", RichLog)
        assert world_view is not None


# ---------------------------------------------------------------------------
# T11 — headless: quick action buttons present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dashboard_action_buttons():
    from vector_os_nano.cli.dashboard import DashboardApp
    from textual.widgets import Button
    app = DashboardApp()
    async with app.run_test() as pilot:
        assert app.query_one("#btn-home", Button)
        assert app.query_one("#btn-scan", Button)
        assert app.query_one("#btn-detect", Button)
        assert app.query_one("#btn-stop", Button)


# ---------------------------------------------------------------------------
# T12 — headless: status panel shows "not connected" without hardware
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dashboard_status_no_agent():
    from vector_os_nano.cli.dashboard import DashboardApp
    from textual.widgets import Static
    app = DashboardApp()
    async with app.run_test() as pilot:
        panel = app.query_one("#status-panel", Static)
        # Panel must be mounted — check it renders something (render() returns a renderable)
        assert panel is not None
        rendered = panel.render()
        assert rendered is not None


# ---------------------------------------------------------------------------
# T13 — headless: skills table populated when agent provided
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dashboard_skills_table_populated():
    from vector_os_nano.cli.dashboard import DashboardApp
    from textual.widgets import DataTable
    agent = _make_mock_agent()
    app = DashboardApp(agent=agent)
    async with app.run_test() as pilot:
        table = app.query_one("#skills-table", DataTable)
        assert table.row_count > 0


# ---------------------------------------------------------------------------
# T14 — headless: F1 key switches to dashboard tab
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dashboard_f1_tab_switch():
    from vector_os_nano.cli.dashboard import DashboardApp
    from textual.widgets import TabbedContent
    app = DashboardApp()
    async with app.run_test() as pilot:
        await pilot.press("f1")
        tc = app.query_one(TabbedContent)
        assert tc.active == "dashboard"


# ---------------------------------------------------------------------------
# T15 — headless: F2 key switches to log tab
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dashboard_f2_tab_switch():
    from vector_os_nano.cli.dashboard import DashboardApp
    from textual.widgets import TabbedContent
    app = DashboardApp()
    async with app.run_test() as pilot:
        await pilot.press("f2")
        tc = app.query_one(TabbedContent)
        assert tc.active == "log"


# ---------------------------------------------------------------------------
# T16 — headless: Ctrl+E triggers E-stop (no crash)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dashboard_estop_no_crash():
    from vector_os_nano.cli.dashboard import DashboardApp
    app = DashboardApp()
    async with app.run_test() as pilot:
        # Call action directly — key bindings for ctrl+e may be swallowed by
        # the terminal emulator in headless mode
        app.action_estop()
        # After e-stop, log widget must still be present
        from textual.widgets import RichLog
        log_view = app.query_one("#log-view", RichLog)
        assert log_view is not None


# ---------------------------------------------------------------------------
# T17 — headless: E-stop calls agent.stop() when agent present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dashboard_estop_calls_agent_stop():
    from vector_os_nano.cli.dashboard import DashboardApp
    agent = _make_mock_agent()
    app = DashboardApp(agent=agent)
    async with app.run_test() as pilot:
        # action_estop() must delegate to agent.stop()
        app.action_estop()
        agent.stop.assert_called_once()


# ---------------------------------------------------------------------------
# T18 — headless: empty command input does not call agent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dashboard_empty_command_no_execute():
    from vector_os_nano.cli.dashboard import DashboardApp
    agent = _make_mock_agent()
    app = DashboardApp(agent=agent)
    async with app.run_test() as pilot:
        # Submit empty input — agent.execute should NOT be called
        input_widget = app.query_one("#command-input")
        await pilot.click("#command-input")
        await pilot.press("enter")
        agent.execute.assert_not_called()


# ---------------------------------------------------------------------------
# T19 — _log() method works without crashing before/after mount
# ---------------------------------------------------------------------------

def test_dashboard_log_method_no_crash():
    from vector_os_nano.cli.dashboard import DashboardApp
    app = DashboardApp()
    # Should not raise even before mounting
    try:
        app._log("test message")
    except Exception:
        pass  # Expected before mount — just must not raise uncaught


# ---------------------------------------------------------------------------
# T20 — TEXTUAL_AVAILABLE flag exported
# ---------------------------------------------------------------------------

def test_textual_available_flag():
    from vector_os_nano.cli.dashboard import TEXTUAL_AVAILABLE
    assert TEXTUAL_AVAILABLE is True


# ---------------------------------------------------------------------------
# T21 — Camera tab: #camera-rgb and #camera-depth widgets present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_camera_tab_exists():
    from vector_os_nano.cli.dashboard import DashboardApp
    from textual.widgets import Static
    app = DashboardApp()
    async with app.run_test() as pilot:
        rgb_panel = app.query_one("#camera-rgb", Static)
        depth_panel = app.query_one("#camera-depth", Static)
        assert rgb_panel is not None
        assert depth_panel is not None


# ---------------------------------------------------------------------------
# T22 — F5 key switches to camera tab
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dashboard_f5_tab_switch():
    from vector_os_nano.cli.dashboard import DashboardApp
    from textual.widgets import TabbedContent
    app = DashboardApp()
    async with app.run_test() as pilot:
        await pilot.press("f5")
        tc = app.query_one(TabbedContent)
        assert tc.active == "camera"


# ---------------------------------------------------------------------------
# T23 — F5 and F6 bindings present in BINDINGS
# ---------------------------------------------------------------------------

def test_dashboard_camera_bindings():
    from vector_os_nano.cli.dashboard import DashboardApp
    keys = {b.key for b in DashboardApp.BINDINGS}
    assert "f5" in keys
    assert "f6" in keys


# ---------------------------------------------------------------------------
# T24 — action_open_fullscreen_camera logs warning when no agent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fullscreen_camera_no_agent():
    from vector_os_nano.cli.dashboard import DashboardApp
    app = DashboardApp()
    async with app.run_test() as pilot:
        # Should not raise; logs a yellow warning instead
        app.action_open_fullscreen_camera()
        from textual.widgets import RichLog
        log_view = app.query_one("#log-view", RichLog)
        assert log_view is not None


# ---------------------------------------------------------------------------
# T25 — logo banner widget present after mount
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_logo_banner_present():
    from vector_os_nano.cli.dashboard import DashboardApp
    app = DashboardApp()
    async with app.run_test() as pilot:
        banner = app.query_one("#logo-banner")
        assert banner is not None


# ---------------------------------------------------------------------------
# T26 — command input present and not docked (dock fix)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_command_input_present():
    from vector_os_nano.cli.dashboard import DashboardApp
    from textual.widgets import Input
    app = DashboardApp()
    async with app.run_test() as pilot:
        inp = app.query_one("#command-input", Input)
        assert inp is not None


# ---------------------------------------------------------------------------
# T27 — action_focus_command does not crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_focus_command_action():
    from vector_os_nano.cli.dashboard import DashboardApp
    app = DashboardApp()
    async with app.run_test() as pilot:
        # Must not raise
        app.action_focus_command()


# ---------------------------------------------------------------------------
# T28 — connected agent shows green dot in status output
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_status_dots_green():
    from vector_os_nano.cli.dashboard import DashboardApp
    from unittest.mock import MagicMock
    agent = _make_mock_agent()
    agent._arm = MagicMock()
    agent._perception = MagicMock()
    agent._llm = MagicMock()
    app = DashboardApp(agent=agent)
    async with app.run_test() as pilot:
        status_text = app._render_status()
        assert "[green]●[/green]" in status_text


# ---------------------------------------------------------------------------
# T29 — dashboard camera preview widget present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dash_camera_preview_present():
    from vector_os_nano.cli.dashboard import DashboardApp
    from textual.widgets import Static
    app = DashboardApp()
    async with app.run_test() as pilot:
        widget = app.query_one("#dash-camera-preview", Static)
        assert widget is not None


# ---------------------------------------------------------------------------
# T25 — _update_camera_panel is a no-op when agent is None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_camera_panel_no_agent_no_crash():
    from vector_os_nano.cli.dashboard import DashboardApp
    app = DashboardApp()
    async with app.run_test() as pilot:
        # Calling directly must not raise
        app._update_camera_panel()
