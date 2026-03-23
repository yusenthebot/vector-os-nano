"""Vector OS Nano — Interactive CLI with AI Chat.

Rich-powered interactive shell with:
- Robot command execution via unified Agent pipeline
- AI conversation with Claude Haiku (multi-turn memory)
- Beautiful terminal output with panels, spinners, tables
"""
from __future__ import annotations

import json
import logging
import readline  # noqa: F401 — enables line editing and history
import sys
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

logger = logging.getLogger(__name__)

from vector_os_nano.version import __version__ as _VERSION

# Rich console instance
_console = Console()

# Brand color
_TEAL = "#00b4b4"
_DIM_TEAL = "#006666"


class SimpleCLI:
    """Interactive CLI with AI chat and robot control."""

    COMMANDS: dict[str, str] = {
        "pick":   "Pick an object: pick <name>",
        "place":  "Place held object at location",
        "home":   "Move arm to home position",
        "scan":   "Move to scan position",
        "open":   "Open gripper",
        "close":  "Close gripper",
        "detect": "Detect objects: detect [query]",
        "stop":   "Stop all tasks",
        "status": "Show system status",
        "world":  "Show world model",
        "help":   "Show commands",
        "quit":   "Exit (also: q, Ctrl-C)",
    }

    def __init__(self, agent: Any = None, verbose: bool = False) -> None:
        self._agent = agent
        self._verbose = verbose
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main readline input loop."""
        self._running = True
        if not self._verbose:
            self._quiet_logging()
        self._print_banner()

        while self._running:
            try:
                user_input = _console.input(f"[{_TEAL}]vector>[/] ").strip()
            except (KeyboardInterrupt, EOFError):
                _console.print()
                self._running = False
                break
            except Exception:
                continue

            if not user_input:
                continue

            try:
                self._handle_input(user_input)
            except Exception as exc:
                _console.print(f"[red]Error: {exc}[/]")
                logger.exception("Unhandled error")

        _console.print(f"[dim]Goodbye.[/]")

    def _handle_input(self, text: str) -> None:
        """Route input: built-in CLI commands or Agent pipeline."""
        parts = text.split(None, 1)
        command = parts[0].lower()

        if command in ("quit", "exit", "q"):
            self._running = False
            return
        if command == "help":
            self._print_help()
            return
        if command == "status":
            self._handle_status()
            return
        if command == "skills":
            self._handle_skills()
            return
        if command == "world":
            self._handle_world()
            return

        if self._agent is not None:
            self._execute_unified(text)
        else:
            _console.print("[dim]No agent configured. Type 'help'.[/]")

    # ------------------------------------------------------------------
    # Unified execution
    # ------------------------------------------------------------------

    def _execute_unified(self, text: str) -> None:
        """Route all input through Agent's multi-stage pipeline."""
        start = time.time()
        step_count = [0]

        def _on_message(msg: str) -> None:
            _console.print()
            _console.print(Panel(
                msg,
                title="[bold]V[/]",
                title_align="left",
                border_style=_TEAL,
                padding=(0, 1),
                width=min(_console.width, 70),
            ))

        def _on_step(skill_name: str, idx: int, total: int) -> None:
            step_count[0] = total
            if idx == 0:
                _console.print(f"  [dim]Plan: {total} steps[/]")
                _console.print()
            label = skill_name.replace("_", " ")
            _console.print(f"  [dim][{idx+1}/{total}][/] [{_TEAL}]{label}[/] ", end="")

        def _on_step_done(skill_name: str, success: bool, duration: float) -> None:
            if success:
                _console.print(f"[green]OK[/] [dim]{duration:.1f}s[/]")
            else:
                _console.print(f"[red]FAIL[/]")

        result = self._agent.execute(
            text, on_message=_on_message, on_step=_on_step, on_step_done=_on_step_done,
        )
        elapsed = time.time() - start

        if result.status == "chat":
            if result.message:
                _console.print()
                _console.print(Panel(
                    result.message,
                    title="[bold]V[/]",
                    title_align="left",
                    border_style=_TEAL,
                    padding=(0, 1),
                    width=min(_console.width, 70),
                ))
                _console.print()

        elif result.status == "query":
            if result.message:
                _console.print()
                _console.print(Panel(
                    result.message,
                    title="[bold]V[/]",
                    title_align="left",
                    border_style=_TEAL,
                    padding=(0, 1),
                    width=min(_console.width, 70),
                ))
                _console.print()

        elif result.status == "clarification_needed":
            msg = result.message or result.clarification_question
            _console.print()
            _console.print(Panel(
                msg or "",
                title="[bold]V[/]",
                title_align="left",
                border_style="yellow",
                padding=(0, 1),
                width=min(_console.width, 70),
            ))
            _console.print()

        elif result.success:
            _console.print()
            _console.print(
                f"  [green bold]Done[/] [dim]{result.steps_completed}/{result.steps_total} steps, {elapsed:.1f}s[/]"
            )

            # LLM summarize
            if hasattr(self._agent, '_llm') and self._agent._llm is not None and result.trace:
                try:
                    trace_str = ", ".join(
                        f"{s.skill_name}({'OK' if s.status=='success' else 'FAIL'})"
                        for s in result.trace
                    )
                    summary = self._agent._llm.summarize(text, trace_str)
                    if summary and not summary.startswith("LLM error"):
                        _console.print()
                        _console.print(Panel(
                            summary,
                            title="[bold]V[/]",
                            title_align="left",
                            border_style=_DIM_TEAL,
                            padding=(0, 1),
                            width=min(_console.width, 70),
                        ))
                except Exception:
                    pass
            _console.print()

        else:
            # Failed
            if result.message:
                _console.print()
                _console.print(Panel(
                    result.message,
                    title="[bold]V[/]",
                    title_align="left",
                    border_style="red",
                    padding=(0, 1),
                    width=min(_console.width, 70),
                ))
            _console.print(f"\n  [red bold]Failed:[/] {result.failure_reason}")
            if result.trace:
                for step in result.trace:
                    icon = "[green]OK[/]" if step.status == "success" else "[red]FAIL[/]"
                    _console.print(f"  {icon} {step.skill_name} [dim]{step.duration_sec:.1f}s[/]")
            _console.print()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quiet_logging() -> None:
        logging.getLogger().setLevel(logging.ERROR)

    def _print_banner(self) -> None:
        import pathlib as _pl
        _logo_path = _pl.Path(__file__).parent / "logo_braille.txt"
        try:
            logo_text = _logo_path.read_text().strip()
        except FileNotFoundError:
            logo_text = "VECTOR OS NANO"

        _console.print()
        _console.print(f"[{_TEAL} bold]{logo_text}[/]")
        _console.print(f"[dim]{'':>40}v{_VERSION}[/]")
        _console.print()
        _console.print(f"  [{_TEAL}]Natural language robot arm control + AI chat.[/]")
        _console.print(f"  [dim]Robot commands execute directly. Other messages chat with AI.[/]")
        _console.print(f"  [dim]Type[/] [{_TEAL}]'help'[/] [dim]for commands.[/]")
        _console.print()

    def _print_help(self) -> None:
        table = Table(
            title="Commands",
            box=box.ROUNDED,
            border_style=_DIM_TEAL,
            title_style=f"bold {_TEAL}",
            show_header=False,
            padding=(0, 1),
        )
        table.add_column("Command", style=_TEAL, width=12)
        table.add_column("Description")
        for cmd, desc in self.COMMANDS.items():
            table.add_row(cmd, desc)
        _console.print()
        _console.print(table)
        _console.print(f"\n  [dim]Type anything else to chat with V, the AI assistant.[/]\n")

    def _handle_status(self) -> None:
        if self._agent is None:
            _console.print("[dim]No agent configured.[/]")
            return

        table = Table(
            title="System Status",
            box=box.ROUNDED,
            border_style=_DIM_TEAL,
            title_style=f"bold {_TEAL}",
        )
        table.add_column("Component", style=_TEAL)
        table.add_column("Value")

        # Mode
        if hasattr(self._agent._arm, "get_object_positions"):
            table.add_row("Mode", "MuJoCo simulation")
        elif self._agent._arm is not None:
            table.add_row("Mode", "Real hardware")
        else:
            table.add_row("Mode", "No arm")

        # Skills
        table.add_row("Skills", ", ".join(self._agent.skills))

        # Joints
        if self._agent._arm is not None:
            joints = self._agent._arm.get_joint_positions()
            names = self._agent._arm.joint_names
            joint_str = "  ".join(f"{n}={j:+.2f}" for n, j in zip(names, joints))
            table.add_row("Joints", joint_str)

        # Gripper
        if self._agent._gripper is not None:
            try:
                pos = self._agent._gripper.get_position()
                holding = self._agent._gripper.is_holding()
                state = "open" if pos > 0.5 else "closed"
                if holding:
                    state += " (holding)"
                table.add_row("Gripper", state)
            except Exception:
                pass

        # LLM
        has_llm = hasattr(self._agent, '_llm') and self._agent._llm is not None
        table.add_row("LLM", "[green]enabled[/]" if has_llm else "[red]disabled[/]")

        _console.print()
        _console.print(table)

        # Objects
        if hasattr(self._agent._arm, "get_object_positions"):
            objs = self._agent._arm.get_object_positions()
            if objs:
                obj_table = Table(
                    title=f"Objects ({len(objs)})",
                    box=box.SIMPLE,
                    border_style="dim",
                    title_style=f"bold {_TEAL}",
                )
                obj_table.add_column("Name", style=_TEAL)
                obj_table.add_column("X", justify="right")
                obj_table.add_column("Y", justify="right")
                obj_table.add_column("Z", justify="right")
                for name, pos in objs.items():
                    obj_table.add_row(
                        name,
                        f"{pos[0]:.3f}",
                        f"{pos[1]:.3f}",
                        f"{pos[2]:.3f}",
                    )
                _console.print(obj_table)
        _console.print()

    def _handle_skills(self) -> None:
        if self._agent is not None:
            _console.print(f"[{_TEAL}]Skills:[/] {', '.join(self._agent.skills)}")
        else:
            _console.print("[dim]No agent configured.[/]")

    def _handle_world(self) -> None:
        if self._agent is None:
            _console.print("[dim]No agent configured.[/]")
            return
        from rich.syntax import Syntax
        world_json = json.dumps(self._agent.world.to_dict(), indent=2, ensure_ascii=False)
        _console.print(Syntax(world_json, "json", theme="monokai"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the 'vector-os' console script."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Vector OS Nano CLI")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--no-arm", action="store_true")
    parser.add_argument("--no-perception", action="store_true")
    parser.add_argument("--llm-key", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--sim", action="store_true")

    args = parser.parse_args()

    try:
        from vector_os_nano.core.agent import Agent
    except ImportError as exc:
        print(f"Error: cannot import Agent: {exc}", file=sys.stderr)
        sys.exit(1)

    arm = None
    gripper = None
    perception = None

    if not args.no_arm and not args.sim:
        try:
            from vector_os_nano.hardware.so101 import SO101Arm
            arm = SO101Arm(port=args.port)
            arm.connect()
        except Exception as exc:
            print(f"Warning: could not connect to arm: {exc}")

    api_key = args.llm_key or os.environ.get("OPENROUTER_API_KEY")
    agent = Agent(arm=arm, gripper=gripper, perception=perception,
                  llm_api_key=api_key, config=args.config)

    cli = SimpleCLI(agent=agent, verbose=args.verbose)
    try:
        cli.run()
    finally:
        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    main()
