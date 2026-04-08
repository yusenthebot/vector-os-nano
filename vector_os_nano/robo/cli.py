"""Vector CLI — unified robot control from the terminal.

Entry point: ``vector`` command (registered in pyproject.toml).

Usage::

    vector                     # Interactive REPL
    vector go2 stand           # One-shot Go2 command
    vector arm home            # One-shot arm command
    vector status              # System status
    vector skills              # List all skills
    vector chat                # LLM-powered agent mode
"""
from __future__ import annotations

import logging
import os
import sys

import click
from rich.console import Console

from vector_os_nano.robo.context import RoboContext
from vector_os_nano.robo.output import (
    render_error,
    render_info,
    render_skills_list,
    render_status,
)

logger = logging.getLogger(__name__)

_BANNER = """[bold cyan]
 _   __          __
| | / /__ ______/ /____  ____
| |/ / -_) __/ __/ _ \\/ __/
|___/\\__/\\__/\\__/\\___/_/    CLI
[/bold cyan]"""


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option(
    "--connect", is_flag=True, default=False,
    help="Auto-connect to Go2 proxy on startup.",
)
@click.version_option(version="0.1.0", prog_name="vector")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, connect: bool) -> None:
    """Vector CLI — control your robot from the terminal."""
    # Configure logging
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s: %(message)s")

    console = Console()
    rctx = RoboContext(console=console, verbose=verbose)

    # Auto-connect to Go2 proxy if requested
    if connect:
        rctx.connect_go2_proxy()
        layout = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "room_layout.yaml",
        )
        rctx.connect_scene_graph(layout)

    ctx.ensure_object(dict)
    ctx.obj = rctx

    # No subcommand → launch REPL
    if ctx.invoked_subcommand is None:
        console.print(_BANNER)
        from vector_os_nano.robo.repl import run_repl
        run_repl(rctx)


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show hardware and connection status."""
    rctx: RoboContext = ctx.obj
    info = rctx.get_status()
    render_status(rctx.console, info)


@cli.command()
@click.pass_context
def skills(ctx: click.Context) -> None:
    """List all registered skills."""
    rctx: RoboContext = ctx.obj
    render_skills_list(rctx.console, rctx.skill_registry)


@cli.command()
@click.argument("key", required=False)
@click.argument("value", required=False)
@click.pass_context
def config(ctx: click.Context, key: str | None, value: str | None) -> None:
    """Show or set configuration values."""
    rctx: RoboContext = ctx.obj

    try:
        from vector_os_nano.vcli.config import load_config, save_config
    except ImportError:
        render_error(rctx.console, "vcli.config not available")
        return

    cfg = load_config()

    if key is None:
        # Show all
        from rich.table import Table
        table = Table(title="Configuration", show_header=True)
        table.add_column("Key", style="bold")
        table.add_column("Value")
        for k, v in sorted(cfg.items()):
            # Mask API keys
            display = str(v)
            if "key" in k.lower() and v:
                display = display[:8] + "..." if len(display) > 8 else "***"
            table.add_row(k, display)
        rctx.console.print(table)
    elif value is None:
        # Show one key
        v = cfg.get(key, "[not set]")
        rctx.console.print(f"{key} = {v}")
    else:
        # Set key
        cfg[key] = value
        save_config(cfg)
        render_info(rctx.console, f"{key} = {value}")


@cli.command()
@click.argument("name", required=False)
@click.pass_context
def model(ctx: click.Context, name: str | None) -> None:
    """Show or switch the LLM model."""
    rctx: RoboContext = ctx.obj

    try:
        from vector_os_nano.vcli.config import load_config, save_config
    except ImportError:
        render_error(rctx.console, "vcli.config not available")
        return

    cfg = load_config()

    if name is None:
        current = cfg.get("model", "not set")
        rctx.console.print(f"Current model: [bold]{current}[/bold]")
    else:
        cfg["model"] = name
        save_config(cfg)
        render_info(rctx.console, f"Model set to: {name}")


@cli.command()
@click.pass_context
def chat(ctx: click.Context) -> None:
    """Start LLM-powered agent chat mode."""
    rctx: RoboContext = ctx.obj
    from vector_os_nano.robo.chat import run_chat
    run_chat(rctx)


# ---------------------------------------------------------------------------
# Register command groups
# ---------------------------------------------------------------------------

def _register_groups() -> None:
    """Register all built-in command groups."""
    from vector_os_nano.robo.groups.go2 import go2
    from vector_os_nano.robo.groups.arm import arm
    from vector_os_nano.robo.groups.arm import gripper
    from vector_os_nano.robo.groups.perception import perception
    from vector_os_nano.robo.groups.sim import sim
    from vector_os_nano.robo.groups.ros import ros

    cli.add_command(go2)
    cli.add_command(arm)
    cli.add_command(gripper)
    cli.add_command(perception)
    cli.add_command(sim)
    cli.add_command(ros)


_register_groups()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``vector`` console script."""
    cli(standalone_mode=True)


if __name__ == "__main__":
    main()
