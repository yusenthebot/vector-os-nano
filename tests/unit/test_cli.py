"""Unit tests for vector_os.cli.simple — SimpleCLI.

TDD: written before implementation.
No ROS2 imports. All tests use mocks or no-agent mode.
"""
from __future__ import annotations

import importlib
import io
import sys
import types
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_agent(skills=("pick", "place", "home"), held=None):
    """Build a minimal mock agent that satisfies SimpleCLI's interface."""
    agent = MagicMock()
    agent.skills = list(skills)

    # world model
    world = MagicMock()
    from vector_os.core.world_model import ObjectState, RobotState, WorldModel
    obj1 = ObjectState(object_id="obj_001", label="cup", confidence=0.92)
    world.get_objects.return_value = [obj1]
    robot = RobotState(gripper_state="open", held_object=held)
    world.get_robot.return_value = robot
    world.to_dict.return_value = {"objects": {}, "robot": {}}
    agent.world = world

    # execute returns a successful ExecutionResult
    from vector_os.core.types import ExecutionResult
    result = ExecutionResult(
        success=True,
        status="completed",
        steps_completed=1,
        steps_total=1,
    )
    agent.execute.return_value = result

    return agent


# ---------------------------------------------------------------------------
# T1 — creation without agent
# ---------------------------------------------------------------------------

def test_cli_creation():
    from vector_os.cli.simple import SimpleCLI
    cli = SimpleCLI()
    assert cli._agent is None
    assert cli._verbose is False
    assert cli._running is False


# ---------------------------------------------------------------------------
# T2 — creation with agent
# ---------------------------------------------------------------------------

def test_cli_creation_with_agent():
    from vector_os.cli.simple import SimpleCLI
    agent = _make_mock_agent()
    cli = SimpleCLI(agent=agent, verbose=True)
    assert cli._agent is agent
    assert cli._verbose is True


# ---------------------------------------------------------------------------
# T3 — help command doesn't crash
# ---------------------------------------------------------------------------

def test_cli_handle_help(capsys):
    from vector_os.cli.simple import SimpleCLI
    cli = SimpleCLI()
    cli._handle_input("help")
    out = capsys.readouterr().out
    assert "help" in out.lower() or "commands" in out.lower() or "Available" in out


# ---------------------------------------------------------------------------
# T4 — quit sets running to False
# ---------------------------------------------------------------------------

def test_cli_handle_quit():
    from vector_os.cli.simple import SimpleCLI
    cli = SimpleCLI()
    cli._running = True
    cli._handle_input("quit")
    assert cli._running is False


# ---------------------------------------------------------------------------
# T5 — exit sets running to False
# ---------------------------------------------------------------------------

def test_cli_handle_exit():
    from vector_os.cli.simple import SimpleCLI
    cli = SimpleCLI()
    cli._running = True
    cli._handle_input("exit")
    assert cli._running is False


# ---------------------------------------------------------------------------
# T6 — q also sets running to False
# ---------------------------------------------------------------------------

def test_cli_handle_q():
    from vector_os.cli.simple import SimpleCLI
    cli = SimpleCLI()
    cli._running = True
    cli._handle_input("q")
    assert cli._running is False


# ---------------------------------------------------------------------------
# T7 — unknown command without agent prints informative message
# ---------------------------------------------------------------------------

def test_cli_handle_unknown(capsys):
    from vector_os.cli.simple import SimpleCLI
    cli = SimpleCLI(agent=None)
    cli._handle_input("frobnicate")
    out = capsys.readouterr().out
    assert "frobnicate" in out or "Unknown" in out or "help" in out.lower()


# ---------------------------------------------------------------------------
# T8 — status without agent prints informative message
# ---------------------------------------------------------------------------

def test_cli_handle_status_no_agent(capsys):
    from vector_os.cli.simple import SimpleCLI
    cli = SimpleCLI(agent=None)
    cli._handle_input("status")
    out = capsys.readouterr().out
    assert out.strip() != ""


# ---------------------------------------------------------------------------
# T9 — skills command shows skill names
# ---------------------------------------------------------------------------

def test_cli_handle_skills(capsys):
    from vector_os.cli.simple import SimpleCLI
    agent = _make_mock_agent(skills=("pick", "place", "home"))
    cli = SimpleCLI(agent=agent)
    cli._handle_input("skills")
    out = capsys.readouterr().out
    assert "pick" in out
    assert "place" in out
    assert "home" in out


# ---------------------------------------------------------------------------
# T10 — all expected command keys present in COMMANDS dict
# ---------------------------------------------------------------------------

def test_cli_commands_dict():
    from vector_os.cli.simple import SimpleCLI
    expected = {"pick", "place", "home", "scan", "detect", "status", "skills",
                "world", "help", "quit"}
    assert expected.issubset(set(SimpleCLI.COMMANDS.keys()))


# ---------------------------------------------------------------------------
# T11 — main() is importable and is callable
# ---------------------------------------------------------------------------

def test_cli_main_function_exists():
    from vector_os.cli.simple import main
    assert callable(main)


# ---------------------------------------------------------------------------
# T12 — status with agent prints object and gripper info
# ---------------------------------------------------------------------------

def test_cli_handle_status_with_agent(capsys):
    from vector_os.cli.simple import SimpleCLI
    agent = _make_mock_agent()
    cli = SimpleCLI(agent=agent)
    cli._handle_input("status")
    out = capsys.readouterr().out
    assert "cup" in out or "obj_001" in out or "Objects" in out


# ---------------------------------------------------------------------------
# T13 — world command outputs JSON-like text
# ---------------------------------------------------------------------------

def test_cli_handle_world(capsys):
    from vector_os.cli.simple import SimpleCLI
    agent = _make_mock_agent()
    cli = SimpleCLI(agent=agent)
    cli._handle_input("world")
    out = capsys.readouterr().out
    # Should be valid-looking JSON (curly braces present)
    assert "{" in out


# ---------------------------------------------------------------------------
# T14 — world without agent prints informative message
# ---------------------------------------------------------------------------

def test_cli_handle_world_no_agent(capsys):
    from vector_os.cli.simple import SimpleCLI
    cli = SimpleCLI(agent=None)
    cli._handle_input("world")
    out = capsys.readouterr().out
    assert out.strip() != ""


# ---------------------------------------------------------------------------
# T15 — unknown command with agent calls agent.execute()
# ---------------------------------------------------------------------------

def test_cli_unknown_command_calls_agent_execute(capsys):
    from vector_os.cli.simple import SimpleCLI
    agent = _make_mock_agent()
    cli = SimpleCLI(agent=agent)
    cli._handle_input("pick the cup")
    agent.execute.assert_called_once_with("pick the cup")


# ---------------------------------------------------------------------------
# T16 — _print_result success path
# ---------------------------------------------------------------------------

def test_cli_print_result_success(capsys):
    from vector_os.cli.simple import SimpleCLI
    from vector_os.core.types import ExecutionResult
    cli = SimpleCLI()
    result = ExecutionResult(
        success=True,
        status="completed",
        steps_completed=2,
        steps_total=2,
    )
    cli._print_result(result)
    out = capsys.readouterr().out
    assert "OK" in out or "2" in out


# ---------------------------------------------------------------------------
# T17 — _print_result clarification_needed path
# ---------------------------------------------------------------------------

def test_cli_print_result_clarification(capsys):
    from vector_os.cli.simple import SimpleCLI
    from vector_os.core.types import ExecutionResult
    cli = SimpleCLI()
    result = ExecutionResult(
        success=False,
        status="clarification_needed",
        clarification_question="Which cup?",
    )
    cli._print_result(result)
    out = capsys.readouterr().out
    assert "Which cup?" in out or "clarification" in out.lower() or "Question" in out


# ---------------------------------------------------------------------------
# T18 — _print_result failure path
# ---------------------------------------------------------------------------

def test_cli_print_result_failure(capsys):
    from vector_os.cli.simple import SimpleCLI
    from vector_os.core.types import ExecutionResult
    cli = SimpleCLI()
    result = ExecutionResult(
        success=False,
        status="failed",
        failure_reason="IK solver failed",
    )
    cli._print_result(result)
    out = capsys.readouterr().out
    assert "IK solver failed" in out or "FAILED" in out


# ---------------------------------------------------------------------------
# T19 — _print_result verbose shows trace steps
# ---------------------------------------------------------------------------

def test_cli_print_result_verbose_trace(capsys):
    from vector_os.cli.simple import SimpleCLI
    from vector_os.core.types import ExecutionResult, StepTrace
    cli = SimpleCLI(verbose=True)
    trace = [StepTrace(step_id="s1", skill_name="home", status="success", duration_sec=0.5)]
    result = ExecutionResult(
        success=True,
        status="completed",
        steps_completed=1,
        steps_total=1,
        trace=trace,
    )
    cli._print_result(result)
    out = capsys.readouterr().out
    assert "home" in out


# ---------------------------------------------------------------------------
# T20 — banner is printed on run() start
# ---------------------------------------------------------------------------

def test_cli_print_banner(capsys):
    from vector_os.cli.simple import SimpleCLI
    cli = SimpleCLI()
    cli._print_banner()
    out = capsys.readouterr().out
    assert "Vector OS" in out


# ---------------------------------------------------------------------------
# T21 — run() exits cleanly on EOF
# ---------------------------------------------------------------------------

def test_cli_run_exits_on_eof():
    from vector_os.cli.simple import SimpleCLI
    cli = SimpleCLI()
    with patch("builtins.input", side_effect=EOFError):
        cli.run()
    assert cli._running is False


# ---------------------------------------------------------------------------
# T22 — run() exits cleanly on KeyboardInterrupt
# ---------------------------------------------------------------------------

def test_cli_run_exits_on_keyboard_interrupt():
    from vector_os.cli.simple import SimpleCLI
    cli = SimpleCLI()
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        cli.run()
    assert cli._running is False


# ---------------------------------------------------------------------------
# T23 — readline is importable (side-effect: enables line editing)
# ---------------------------------------------------------------------------

def test_readline_importable():
    import readline  # noqa: F401
    assert True


# ---------------------------------------------------------------------------
# T24 — skills without agent prints informative message
# ---------------------------------------------------------------------------

def test_cli_handle_skills_no_agent(capsys):
    from vector_os.cli.simple import SimpleCLI
    cli = SimpleCLI(agent=None)
    cli._handle_input("skills")
    out = capsys.readouterr().out
    assert out.strip() != ""


# ---------------------------------------------------------------------------
# T25 — held object shown in status when agent has held object
# ---------------------------------------------------------------------------

def test_cli_handle_status_shows_held_object(capsys):
    from vector_os.cli.simple import SimpleCLI
    agent = _make_mock_agent(held="obj_001")
    # override robot state to show held object
    from vector_os.core.world_model import RobotState
    agent.world.get_robot.return_value = RobotState(
        gripper_state="closed", held_object="obj_001"
    )
    cli = SimpleCLI(agent=agent)
    cli._handle_input("status")
    out = capsys.readouterr().out
    assert "obj_001" in out or "closed" in out
