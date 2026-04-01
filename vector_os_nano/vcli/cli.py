"""Vector CLI — interactive REPL entry point.

Ties together VectorEngine, Session, PermissionContext, and all tools
into an interactive agent loop with V's personality.

    python -m vector_os_nano.vcli.cli [options]
    # or via console_scripts: vector-cli [options]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory

from vector_os_nano.vcli.backends import create_backend
from vector_os_nano.vcli.engine import VectorEngine, TurnResult
from vector_os_nano.vcli.session import (
    Session,
    create_session,
    get_latest_session,
    list_sessions,
    load_session,
)
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.prompt import build_system_prompt
from vector_os_nano.vcli.tools import ToolRegistry, discover_all_tools

logger = logging.getLogger(__name__)
console = Console()

VERSION = "0.1.0"

TEAL = "#00b4b4"
DIM_TEAL = "#006666"

EXIT_COMMANDS: frozenset[str] = frozenset({"quit", "exit", "q"})

# Path to the braille logo shipped with the old CLI
_LOGO_PATH = Path(__file__).resolve().parent.parent / "cli" / "logo_braille.txt"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vector-cli",
        description="Vector CLI — Agentic CLI for Vector OS Nano",
    )
    parser.add_argument("--sim", action="store_true", help="Start with MuJoCo arm simulation")
    parser.add_argument("--sim-go2", action="store_true", help="Start with Go2 quadruped simulation")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Model to use (default: claude-sonnet-4-6)")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Resume session")
    parser.add_argument("--api-key", default=None, help="API key (or set ANTHROPIC_API_KEY / OPENROUTER_API_KEY)")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--no-permission", action="store_true", help="Allow all tools without prompts")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--system-prompt", default=None, help="Path to custom system prompt file")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Input classification
# ---------------------------------------------------------------------------


def is_slash_command(text: str) -> bool:
    return text.strip().startswith("/")


def is_exit_command(text: str) -> bool:
    return text.strip().lower() in EXIT_COMMANDS


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _load_logo() -> str:
    """Load the braille logo, fall back to plain text."""
    try:
        return _LOGO_PATH.read_text(encoding="utf-8").rstrip()
    except (FileNotFoundError, OSError):
        return "VECTOR OS NANO"


def format_banner(model: str, agent: Any = None) -> str:
    """Return banner info text (testable, no side effects)."""
    lines = [
        f"Vector CLI v{VERSION}",
        f"Model: {model}",
    ]
    if agent is not None:
        arm = getattr(agent, "_arm", None)
        base = getattr(agent, "_base", None)
        if arm is not None:
            lines.append(f"Arm: {getattr(arm, 'name', type(arm).__name__)}")
        if base is not None:
            lines.append(f"Base: {getattr(base, 'name', type(base).__name__)}")
    lines.append("/help for commands, quit to exit")
    return "\n".join(lines)


def print_banner(model: str, provider: str, agent: Any = None) -> None:
    """Print animated startup banner with braille logo."""
    logo_lines = _load_logo().splitlines()
    console.print()
    for line in logo_lines:
        console.print(f"[bold {TEAL}]{line}[/]")
        time.sleep(0.08)

    console.print(f"[dim]{'':>40}v{VERSION}[/]")
    time.sleep(0.2)

    info_parts = [f"Model: {model}", f"Provider: {provider}"]
    if agent is not None:
        arm = getattr(agent, "_arm", None)
        base = getattr(agent, "_base", None)
        if arm is not None:
            info_parts.append(f"Arm: {getattr(arm, 'name', type(arm).__name__)}")
        if base is not None:
            info_parts.append(f"Base: {getattr(base, 'name', type(base).__name__)}")

    console.print(f"[dim]  {' | '.join(info_parts)}[/]")
    console.print(f"[dim]  /help for commands, quit to exit[/]")
    console.print()


def ask_permission(tool_name: str, params: dict[str, Any]) -> str:
    """Interactive permission prompt. Returns 'y', 'n', or 'a'."""
    console.print(f"\n[yellow bold]Permission required:[/yellow bold]")
    console.print(f"  Tool: [{TEAL}]{tool_name}[/]")
    params_str = json.dumps(params, indent=2, ensure_ascii=False)
    if len(params_str) > 200:
        params_str = params_str[:200] + "..."
    console.print(f"  Params: [dim]{params_str}[/dim]")
    return Prompt.ask("  Allow? [y/n/a=always]", choices=["y", "n", "a"], default="y")


# ---------------------------------------------------------------------------
# Hardware init
# ---------------------------------------------------------------------------


def _init_agent(args: argparse.Namespace) -> Any:
    if not (args.sim or args.sim_go2):
        return None
    try:
        from vector_os_nano.core.agent import Agent  # type: ignore[import]

        if args.sim:
            from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm  # type: ignore[import]
            arm = MuJoCoArm()
            arm.connect()
            return Agent(arm=arm)

        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # type: ignore[import]
        base = MuJoCoGo2()
        base.connect()
        return Agent(base=base)
    except Exception as exc:
        console.print(f"[yellow]Warning: Could not init simulation: {exc}[/yellow]")
        return None


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def _handle_slash_command(
    cmd: str,
    args_rest: list[str],
    registry: ToolRegistry,
    session: Session | None = None,
) -> bool:
    """Handle /command. Returns True to continue REPL, False to exit."""
    if cmd == "help":
        console.print(f"[bold {TEAL}]Commands:[/]")
        console.print("  /help      show this message")
        console.print("  /quit      exit")
        console.print("  /tools     list registered tools")
        console.print("  /sessions  list saved sessions")
        console.print("  /usage     token usage this session")
        console.print()
    elif cmd in ("quit", "exit", "q"):
        return False
    elif cmd == "tools":
        tool_names = registry.list_tools()
        if not tool_names:
            console.print("[dim]No tools registered.[/dim]")
        else:
            console.print(f"[bold {TEAL}]Tools ({len(tool_names)}):[/]")
            for name in tool_names:
                t = registry.get(name)
                desc = getattr(t, "description", "") if t else ""
                ro = " [dim](ro)[/]" if t and hasattr(t, "is_read_only") and t.is_read_only({}) else ""
                console.print(f"  [{TEAL}]{name}[/]{ro}  [dim]{desc}[/]")
    elif cmd == "sessions":
        sessions = list_sessions()
        if not sessions:
            console.print("[dim]No saved sessions.[/dim]")
        else:
            for s in sessions:
                console.print(f"  [{TEAL}]{s.session_id}[/]  {s.created_at}  ({s.message_count} msgs)")
    elif cmd == "usage":
        if session is not None:
            u = session.token_usage
            total = u.input_tokens + u.output_tokens
            console.print(f"[bold {TEAL}]Usage:[/] in={u.input_tokens:,} out={u.output_tokens:,} total={total:,}")
        else:
            console.print("[dim]No session.[/dim]")
    else:
        console.print(f"[yellow]Unknown: /{cmd}[/]  (try /help)")
    return True


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Resolve API key + provider
    api_key: str = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    base_url: str | None = args.base_url
    provider: str = "anthropic"

    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            provider = "openrouter"
            if not base_url:
                base_url = "https://openrouter.ai/api/v1"

    if not api_key:
        console.print(
            "[red]No API key.[/] Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY, or use --api-key."
        )
        sys.exit(1)

    model: str = args.model
    if provider == "openrouter" and "/" not in model:
        model = f"anthropic/{model}"

    # Agent (optional hardware)
    agent = _init_agent(args)

    # Tools
    registry: ToolRegistry = ToolRegistry()
    for tool in discover_all_tools():
        registry.register(tool)
    if agent is not None:
        from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills
        for skill_tool in wrap_skills(agent):
            registry.register(skill_tool)

    # Permissions
    permissions = PermissionContext(no_permission=args.no_permission)

    # Session
    session: Session
    if args.resume is not None:
        if args.resume == "latest":
            loaded = get_latest_session()
            if loaded is None:
                console.print("[dim]No previous session, starting new.[/dim]")
                session = create_session(metadata={"model": model})
            else:
                session = loaded
                console.print(f"[dim]Resumed: {session.session_id}[/dim]")
        else:
            session = load_session(args.resume)
            console.print(f"[dim]Resumed: {session.session_id}[/dim]")
    else:
        session = create_session(metadata={"model": model})

    # System prompt
    system_prompt = build_system_prompt(agent=agent, cwd=Path.cwd())

    # Backend + engine
    backend = create_backend(provider=provider, api_key=api_key, model=model, base_url=base_url)
    engine = VectorEngine(backend=backend, registry=registry, system_prompt=system_prompt, permissions=permissions)

    # Banner
    provider_display = "OpenRouter" if provider == "openrouter" else "Anthropic"
    if base_url and "localhost" in base_url:
        provider_display = f"Local ({base_url})"
    print_banner(model, provider_display, agent)
    console.print(f"[dim]Session: {session.session_id}[/dim]\n")

    # REPL
    history_dir = Path.home() / ".vector"
    history_dir.mkdir(parents=True, exist_ok=True)
    pt_session: PromptSession = PromptSession(history=FileHistory(str(history_dir / "history")))

    try:
        while True:
            try:
                raw = pt_session.prompt(HTML(f'<style fg="{TEAL}" bold="true">vector&gt;</style> '))
            except EOFError:
                break
            except KeyboardInterrupt:
                console.print("\n[dim]Use quit or Ctrl-D to exit.[/dim]")
                continue

            user_input = raw.strip()
            if not user_input:
                continue
            if is_exit_command(user_input):
                break
            if is_slash_command(user_input):
                parts = user_input.split()
                if not _handle_slash_command(parts[0][1:], parts[1:], registry, session):
                    break
                continue

            # -- Engine turn ---
            try:
                streamed_text: list[str] = []
                live_ref: Live | None = None

                def on_text(chunk: str) -> None:
                    streamed_text.append(chunk)
                    if live_ref is not None:
                        live_ref.update(
                            Panel(
                                "".join(streamed_text),
                                title="[bold]V[/]",
                                title_align="left",
                                border_style=TEAL,
                                padding=(0, 1),
                                width=min(console.width, 80),
                            )
                        )

                def on_tool_start(name: str, p: dict[str, Any]) -> None:
                    console.print(f"  [{DIM_TEAL}]{name}[/]", end="")

                def on_tool_end(name: str, result: Any) -> None:
                    tag = f"[green]ok[/]" if not result.is_error else "[red]fail[/]"
                    console.print(f" {tag}")

                # Show final panel via Live (progressive update)
                empty_panel = Panel("", title="[bold]V[/]", title_align="left", border_style=TEAL, padding=(0, 1), width=min(console.width, 80))
                with Live(empty_panel, console=console, refresh_per_second=8, transient=True) as live:
                    live_ref = live
                    turn_result: TurnResult = engine.run_turn(
                        user_message=user_input,
                        session=session,
                        agent=agent,
                        on_text=on_text,
                        on_tool_start=on_tool_start,
                        on_tool_end=on_tool_end,
                        ask_permission=lambda n, p: ask_permission(n, p),
                    )

                # Print final response in V panel (replaces transient Live)
                if turn_result.text:
                    console.print(
                        Panel(
                            turn_result.text.strip(),
                            title="[bold]V[/]",
                            title_align="left",
                            border_style=TEAL,
                            padding=(0, 1),
                            width=min(console.width, 80),
                        )
                    )

                # Token usage
                if turn_result.usage:
                    u = turn_result.usage
                    total = u.input_tokens + u.output_tokens
                    if total > 0:
                        console.print(f"[dim]  {total:,} tokens[/]")
                console.print()

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/yellow]")
            except Exception as exc:
                console.print(f"[red]Error:[/] {exc}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    finally:
        session.save()
        console.print(f"[dim]Session saved: {session.session_id}[/dim]")


if __name__ == "__main__":
    main()
