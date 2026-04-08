"""Unified interactive REPL for Vector CLI.

Merges the best of SimpleCLI and VectorCLI into one prompt_toolkit-based
loop.  Supports direct skill execution, slash commands, shell passthrough,
and LLM fallback via Agent.execute().

Usage::

    vector          # Opens REPL
    vector repl     # Same
"""
from __future__ import annotations

import logging
import os
import subprocess
import shlex
from typing import Any

from rich.console import Console

from vector_os_nano.robo.context import RoboContext
from vector_os_nano.robo.output import (
    render_error,
    render_info,
    render_skill_result,
    render_skills_list,
    render_status,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slash command handlers
# ---------------------------------------------------------------------------

_SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show available commands",
    "/status": "Hardware and connection status",
    "/skills": "List all registered skills",
    "/config": "Show configuration",
    "/model": "Show or switch LLM model (/model <name>)",
    "/connect": "Connect to Go2 proxy",
    "/clear_memory": "Clear scene graph data",
    "/reset": "Reset robot pose (sim only)",
    "/quit": "Exit REPL",
}


def _handle_slash(text: str, rctx: RoboContext) -> bool:
    """Handle /commands. Returns True if handled, False otherwise."""
    parts = text.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/help":
        rctx.console.print("[bold cyan]Commands:[/bold cyan]")
        for k, v in _SLASH_COMMANDS.items():
            rctx.console.print(f"  {k:20s} {v}")
        rctx.console.print()
        rctx.console.print("[bold cyan]Interaction:[/bold cyan]")
        rctx.console.print("  <skill>              Direct skill execution (stand, walk, etc.)")
        rctx.console.print("  <natural language>   LLM-powered execution")
        rctx.console.print("  !<cmd>               Shell passthrough")
        return True

    if cmd == "/status":
        render_status(rctx.console, rctx.get_status())
        return True

    if cmd == "/skills":
        render_skills_list(rctx.console, rctx.skill_registry)
        return True

    if cmd == "/config":
        try:
            from vector_os_nano.vcli.config import load_config
            cfg = load_config()
            for k, v in sorted(cfg.items()):
                display = str(v)
                if "key" in k.lower() and v:
                    display = display[:8] + "..." if len(display) > 8 else "***"
                rctx.console.print(f"  {k}: {display}")
        except ImportError:
            render_error(rctx.console, "vcli.config not available")
        return True

    if cmd == "/model":
        try:
            from vector_os_nano.vcli.config import load_config, save_config
            cfg = load_config()
            if arg:
                cfg["model"] = arg
                save_config(cfg)
                render_info(rctx.console, f"Model set to: {arg}")
            else:
                rctx.console.print(f"Current model: [bold]{cfg.get('model', 'not set')}[/bold]")
        except ImportError:
            render_error(rctx.console, "vcli.config not available")
        return True

    if cmd == "/connect":
        if rctx.connect_go2_proxy():
            layout = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "config", "room_layout.yaml",
            )
            rctx.connect_scene_graph(layout)
            render_info(rctx.console, "Connected to Go2 proxy")
        else:
            render_error(rctx.console, "Failed to connect to Go2 proxy")
        return True

    if cmd == "/clear_memory":
        if rctx.base and hasattr(rctx.base, "_scene_graph") and rctx.base._scene_graph:
            rctx.base._scene_graph.__init__()
            render_info(rctx.console, "Scene graph cleared")
        else:
            render_error(rctx.console, "No scene graph to clear")
        return True

    if cmd == "/reset":
        try:
            with open("/tmp/vector_reset_pose", "w") as f:
                f.write("1")
            render_info(rctx.console, "Reset pose requested")
        except OSError as exc:
            render_error(rctx.console, f"Reset failed: {exc}")
        return True

    if cmd in ("/quit", "/exit", "/q"):
        raise SystemExit(0)

    return False


# ---------------------------------------------------------------------------
# Input processing
# ---------------------------------------------------------------------------

def _process_input(text: str, rctx: RoboContext) -> None:
    """Process one line of REPL input."""
    text = text.strip()
    if not text:
        return

    # Slash commands
    if text.startswith("/"):
        if _handle_slash(text, rctx):
            return
        render_error(rctx.console, f"Unknown command: {text.split()[0]}")
        return

    # Shell passthrough
    if text.startswith("!"):
        cmd = text[1:].strip()
        if cmd:
            try:
                subprocess.run(cmd, shell=True)
            except Exception as exc:
                render_error(rctx.console, str(exc))
        return

    # Built-in shorthand
    if text.lower() in ("quit", "exit", "q"):
        raise SystemExit(0)

    if text.lower() in ("help", "?"):
        _handle_slash("/help", rctx)
        return

    if text.lower() == "status":
        _handle_slash("/status", rctx)
        return

    # VectorEngine handles everything else (skill dispatch, code editing, diagnostics)
    engine = rctx.get_engine()
    session = rctx.get_session()
    if engine is not None and session is not None:
        try:
            engine.run_turn(
                text,
                session,
                on_text=lambda t: rctx.console.print(t, end=""),
                ask_permission=lambda name, params: (
                    input(f"  Allow {name}? [y/n/a] ").strip().lower() or "y"
                ),
            )
            rctx.console.print()  # newline after streamed text
        except Exception as exc:
            render_error(rctx.console, f"Engine error: {exc}")
        return

    # Fallback: no engine available (no API key configured)
    render_error(rctx.console, "No LLM configured. Set API key with /config or use vector chat.")


# ---------------------------------------------------------------------------
# REPL loop
# ---------------------------------------------------------------------------

def run_repl(rctx: RoboContext) -> None:
    """Run the interactive REPL loop."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

        history_path = os.path.expanduser("~/.vector/repl_history")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)

        session: Any = PromptSession(
            history=FileHistory(history_path),
            auto_suggest=AutoSuggestFromHistory(),
        )

        rctx.console.print("[dim]Type /help for commands, /quit to exit[/dim]\n")

        while True:
            try:
                text = session.prompt("vector> ")
                _process_input(text, rctx)
            except KeyboardInterrupt:
                rctx.console.print()
                continue
            except EOFError:
                break

    except ImportError:
        # Fallback: plain input() loop (no history, no completion)
        rctx.console.print("[dim]Type /help for commands, /quit to exit[/dim]\n")
        while True:
            try:
                text = input("vector> ")
                _process_input(text, rctx)
            except (KeyboardInterrupt, EOFError):
                break

    rctx.console.print("\n[dim]Goodbye[/dim]")
