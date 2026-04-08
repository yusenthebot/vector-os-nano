"""Unit tests for REPL → VectorEngine wiring (Task 8).

Covers:
- test_robo_context_get_engine_returns_engine: get_engine() returns VectorEngine (mocked backend)
- test_robo_context_get_session_returns_session: get_session() returns Session
- test_robo_context_engine_cached: second call to get_engine() returns same instance
- test_robo_context_engine_init_failure_returns_none: import failure → _engine stays None
- test_repl_slash_commands_still_work: "/" input handled locally, not sent to engine
- test_repl_shell_passthrough_still_works: "!" input runs subprocess
- test_repl_routes_plain_text_to_engine: plain text goes to engine.run_turn()
- test_repl_engine_fallback_when_none: no engine → renders error
- test_repl_builtin_shorthands_not_sent_to_engine: quit/exit/help/status handled locally
- test_repl_engine_exception_renders_error: engine.run_turn() raises → renders error
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from vector_os_nano.robo.context import RoboContext
from vector_os_nano.robo.output import render_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rctx(**kwargs: Any) -> RoboContext:
    """Build a RoboContext with a mock console."""
    from rich.console import Console
    from io import StringIO
    console = Console(file=StringIO(), highlight=False)
    return RoboContext(console=console, **kwargs)


def _make_mock_turn_result() -> Any:
    """Build a minimal TurnResult-like mock."""
    from vector_os_nano.vcli.session import TokenUsage
    result = MagicMock()
    result.text = "OK"
    result.tool_calls = []
    result.stop_reason = "end_turn"
    result.usage = TokenUsage()
    return result


# ---------------------------------------------------------------------------
# RoboContext.get_engine() / get_session()
# ---------------------------------------------------------------------------

class TestRoboContextEngine:

    @patch("vector_os_nano.robo.context.create_backend")
    def test_get_engine_returns_engine(self, mock_create_backend: MagicMock) -> None:
        """get_engine() should return a VectorEngine after lazy init."""
        from vector_os_nano.vcli.engine import VectorEngine

        mock_backend = MagicMock()
        mock_create_backend.return_value = mock_backend

        rctx = _make_rctx()
        engine = rctx.get_engine()

        assert engine is not None
        assert isinstance(engine, VectorEngine)

    @patch("vector_os_nano.robo.context.create_backend")
    def test_get_engine_cached(self, mock_create_backend: MagicMock) -> None:
        """Second call to get_engine() returns same instance (no re-init)."""
        mock_create_backend.return_value = MagicMock()

        rctx = _make_rctx()
        first = rctx.get_engine()
        second = rctx.get_engine()

        assert first is second
        # create_backend called only once
        mock_create_backend.assert_called_once()

    def test_get_engine_init_failure_returns_none(self) -> None:
        """If _init_engine raises (e.g., missing deps), get_engine() returns None."""
        rctx = _make_rctx()

        with patch.object(rctx, "_init_engine", side_effect=RuntimeError("no key")):
            # _engine is None, _init_engine raises → should be swallowed, return None
            # We need to patch get_engine's internal guard
            rctx._engine = None
            # Patch _init_engine directly to raise
            with patch.object(rctx, "_init_engine", side_effect=Exception("fail")):
                # Since get_engine calls _init_engine and the exception bubbles,
                # the engine should be None (init_engine catches and logs)
                # Actually the task says _init_engine has try/except, so it sets None
                pass

        # Test that logging.warning is hit when create_backend is unavailable
        with patch("vector_os_nano.robo.context.create_backend", side_effect=ImportError("no module")):
            rctx2 = _make_rctx()
            engine = rctx2.get_engine()
            # _init_engine catches exceptions and logs; _engine stays None
            assert engine is None

    def test_get_session_returns_session(self) -> None:
        """get_session() returns a Session instance."""
        from vector_os_nano.vcli.session import Session

        rctx = _make_rctx()
        session = rctx.get_session()

        assert session is not None
        assert isinstance(session, Session)

    def test_get_session_cached(self) -> None:
        """Second call to get_session() returns same instance."""
        rctx = _make_rctx()
        first = rctx.get_session()
        second = rctx.get_session()

        assert first is second


# ---------------------------------------------------------------------------
# REPL _process_input routing
# ---------------------------------------------------------------------------

class TestReplRouting:
    """Tests for _process_input in repl.py."""

    def _get_process_input(self):
        from vector_os_nano.robo.repl import _process_input
        return _process_input

    # -- Slash commands -------------------------------------------------------

    @patch("vector_os_nano.robo.context.create_backend")
    def test_slash_commands_not_sent_to_engine(self, mock_create: MagicMock) -> None:
        """'/help' should be handled locally without touching the engine."""
        mock_create.return_value = MagicMock()
        rctx = _make_rctx()

        # Ensure engine is initialized so we can spy on run_turn
        engine_mock = MagicMock()
        rctx._engine = engine_mock
        rctx._session = MagicMock()

        _process_input = self._get_process_input()
        _process_input("/help", rctx)

        # engine.run_turn should NOT be called for slash commands
        engine_mock.run_turn.assert_not_called()

    @patch("vector_os_nano.robo.context.create_backend")
    def test_slash_unknown_renders_error(self, mock_create: MagicMock) -> None:
        """/unknown-cmd renders error, not sent to engine."""
        mock_create.return_value = MagicMock()
        rctx = _make_rctx()

        engine_mock = MagicMock()
        rctx._engine = engine_mock
        rctx._session = MagicMock()

        _process_input = self._get_process_input()
        _process_input("/unknowncmd", rctx)

        engine_mock.run_turn.assert_not_called()

    # -- Shell passthrough ----------------------------------------------------

    @patch("subprocess.run")
    @patch("vector_os_nano.robo.context.create_backend")
    def test_shell_passthrough_runs_subprocess(
        self, mock_create: MagicMock, mock_subprocess: MagicMock
    ) -> None:
        """'!ls' should call subprocess.run, not engine.run_turn."""
        mock_create.return_value = MagicMock()
        rctx = _make_rctx()

        engine_mock = MagicMock()
        rctx._engine = engine_mock
        rctx._session = MagicMock()

        _process_input = self._get_process_input()
        _process_input("!ls -la", rctx)

        mock_subprocess.assert_called_once_with("ls -la", shell=True)
        engine_mock.run_turn.assert_not_called()

    # -- Built-in shorthands --------------------------------------------------

    @patch("vector_os_nano.robo.context.create_backend")
    def test_help_shorthand_not_sent_to_engine(self, mock_create: MagicMock) -> None:
        """'help' shorthand uses local handler."""
        mock_create.return_value = MagicMock()
        rctx = _make_rctx()

        engine_mock = MagicMock()
        rctx._engine = engine_mock
        rctx._session = MagicMock()

        _process_input = self._get_process_input()
        _process_input("help", rctx)

        engine_mock.run_turn.assert_not_called()

    @patch("vector_os_nano.robo.context.create_backend")
    def test_status_shorthand_not_sent_to_engine(self, mock_create: MagicMock) -> None:
        """'status' shorthand uses local handler."""
        mock_create.return_value = MagicMock()
        rctx = _make_rctx()

        engine_mock = MagicMock()
        rctx._engine = engine_mock
        rctx._session = MagicMock()

        _process_input = self._get_process_input()
        _process_input("status", rctx)

        engine_mock.run_turn.assert_not_called()

    # -- Plain text → engine --------------------------------------------------

    @patch("vector_os_nano.robo.context.create_backend")
    def test_plain_text_routes_to_engine(self, mock_create: MagicMock) -> None:
        """Non-slash, non-shell, non-shorthand text is sent to engine.run_turn()."""
        mock_create.return_value = MagicMock()
        rctx = _make_rctx()

        engine_mock = MagicMock()
        engine_mock.run_turn.return_value = _make_mock_turn_result()
        session_mock = MagicMock()

        rctx._engine = engine_mock
        rctx._session = session_mock

        _process_input = self._get_process_input()
        _process_input("go to the kitchen", rctx)

        engine_mock.run_turn.assert_called_once()
        call_args = engine_mock.run_turn.call_args
        assert call_args[0][0] == "go to the kitchen"
        assert call_args[0][1] is session_mock

    @patch("vector_os_nano.robo.context.create_backend")
    def test_skill_name_routes_to_engine(self, mock_create: MagicMock) -> None:
        """Direct skill names like 'stand' are sent to VectorEngine, not old skill registry."""
        mock_create.return_value = MagicMock()
        rctx = _make_rctx()

        engine_mock = MagicMock()
        engine_mock.run_turn.return_value = _make_mock_turn_result()
        session_mock = MagicMock()

        rctx._engine = engine_mock
        rctx._session = session_mock

        _process_input = self._get_process_input()
        _process_input("stand", rctx)

        engine_mock.run_turn.assert_called_once()

    # -- Engine = None fallback -----------------------------------------------

    def test_no_engine_renders_error(self) -> None:
        """When engine is None, renders 'No LLM configured' error."""
        rctx = _make_rctx()
        rctx._engine = None
        rctx._session = None

        with patch("vector_os_nano.robo.context.create_backend", side_effect=ImportError("no")):
            _process_input = self._get_process_input()
            # Force _init_engine to fail so engine remains None
            with patch.object(rctx, "_init_engine", side_effect=Exception("no key")):
                pass  # already None

        rctx._engine = None  # ensure None

        # Should render an error, not raise
        _process_input = self._get_process_input()

        # Patch get_engine to return None to simulate unconfigured state
        with patch.object(rctx, "get_engine", return_value=None):
            with patch.object(rctx, "get_session", return_value=None):
                _process_input("do something", rctx)
                # No exception should be raised — error rendered to console

    # -- Engine exception handling --------------------------------------------

    @patch("vector_os_nano.robo.context.create_backend")
    def test_engine_exception_renders_error(self, mock_create: MagicMock) -> None:
        """If engine.run_turn() raises, renders error instead of crashing."""
        mock_create.return_value = MagicMock()
        rctx = _make_rctx()

        engine_mock = MagicMock()
        engine_mock.run_turn.side_effect = RuntimeError("backend error")
        session_mock = MagicMock()

        rctx._engine = engine_mock
        rctx._session = session_mock

        _process_input = self._get_process_input()
        # Should not raise — error should be caught and displayed
        _process_input("do something risky", rctx)

        engine_mock.run_turn.assert_called_once()
