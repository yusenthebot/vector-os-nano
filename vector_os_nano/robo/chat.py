"""LLM-powered agent chat mode — thin wrapper around the shared REPL.

Reuses RoboContext.get_engine() / get_session() so both "vector chat"
and the plain REPL share the same VectorEngine instance.
This is the "vector chat" command.
"""
from __future__ import annotations

import logging

from vector_os_nano.robo.context import RoboContext
from vector_os_nano.robo.output import render_error

logger = logging.getLogger(__name__)


def run_chat(rctx: RoboContext) -> None:
    """Launch the LLM-powered agent chat loop.

    Uses the same VectorEngine that the REPL uses (initialized via
    RoboContext.get_engine()), so both modes share tool registry,
    session, and system prompt configuration.
    """
    engine = rctx.get_engine()
    if engine is None:
        render_error(
            rctx.console,
            "No LLM configured. Run: vector config openrouter_api_key <key>",
        )
        return

    # Delegate to the unified REPL loop (engine already wired into rctx)
    from vector_os_nano.robo.repl import run_repl
    run_repl(rctx)
