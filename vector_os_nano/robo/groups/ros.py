"""ROS2 diagnostics command group for Vector CLI.

Delegates to rosm CLI when available; falls back to raw ``ros2`` subprocess.
"""
from __future__ import annotations

import subprocess

import click

from vector_os_nano.robo.context import RoboContext
from vector_os_nano.robo.output import render_error, render_info


def _try_rosm(args: list[str]) -> bool:
    """Try rosm first, fall back to ros2 CLI.

    Returns True if the command ran (regardless of exit code), False if
    neither rosm nor the ros2 binary could be found.
    """
    try:
        from rosm.cli import cli as rosm_cli  # type: ignore[import]

        rosm_cli(args, standalone_mode=False)
        return True
    except ImportError:
        pass

    # Fallback: raw ros2 CLI
    try:
        subprocess.run(["ros2"] + args, check=False)
        return True
    except FileNotFoundError:
        return False


@click.group("ros")
@click.pass_context
def ros(ctx: click.Context) -> None:
    """ROS2 diagnostics (via rosm)."""


@ros.command("ps")
@click.pass_context
def ps(ctx: click.Context) -> None:
    """List ROS2 processes."""
    rctx: RoboContext = ctx.obj
    if not _try_rosm(["ps"]):
        render_error(rctx.console, "Neither rosm nor ros2 CLI found on PATH.")


@ros.command("nodes")
@click.pass_context
def nodes(ctx: click.Context) -> None:
    """List active ROS2 nodes."""
    rctx: RoboContext = ctx.obj
    if not _try_rosm(["node", "list"]):
        render_error(rctx.console, "Neither rosm nor ros2 CLI found on PATH.")


@ros.command("topics")
@click.pass_context
def topics(ctx: click.Context) -> None:
    """List active ROS2 topics."""
    rctx: RoboContext = ctx.obj
    if not _try_rosm(["topic", "list"]):
        render_error(rctx.console, "Neither rosm nor ros2 CLI found on PATH.")


@ros.command("doctor")
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Run ROS2 diagnostics."""
    rctx: RoboContext = ctx.obj
    if not _try_rosm(["doctor"]):
        render_error(rctx.console, "Neither rosm nor ros2 CLI found on PATH.")


@ros.command("dashboard")
@click.pass_context
def dashboard(ctx: click.Context) -> None:
    """Launch rosm TUI dashboard."""
    rctx: RoboContext = ctx.obj
    try:
        from rosm.cli import cli as rosm_cli  # type: ignore[import]

        render_info(rctx.console, "Launching rosm dashboard...")
        rosm_cli(["dashboard"], standalone_mode=False)
    except ImportError:
        render_error(
            rctx.console,
            "rosm is not installed. Install with: pip install rosm",
        )
