"""Go2 quadruped command group for Vector CLI.

Thin CLI wrappers over Go2 skills. Each command delegates to
RoboContext.execute_skill() and renders output via render_skill_result().
"""
from __future__ import annotations

import click

from vector_os_nano.robo.context import RoboContext
from vector_os_nano.robo.output import render_skill_result


@click.group("go2")
@click.pass_context
def go2(ctx: click.Context) -> None:
    """Go2 quadruped commands."""
    pass


# ---------------------------------------------------------------------------
# Stance commands (no parameters)
# ---------------------------------------------------------------------------


@go2.command()
@click.pass_context
def stand(ctx: click.Context) -> None:
    """Stand up."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("stand")
    render_skill_result(rctx.console, result, "stand")


@go2.command()
@click.pass_context
def sit(ctx: click.Context) -> None:
    """Sit down."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("sit")
    render_skill_result(rctx.console, result, "sit")


@go2.command("lie-down")
@click.pass_context
def lie_down(ctx: click.Context) -> None:
    """Lie down flat."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("lie down")
    render_skill_result(rctx.console, result, "lie down")


@go2.command()
@click.pass_context
def stop(ctx: click.Context) -> None:
    """Stop all motion immediately."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("stop")
    render_skill_result(rctx.console, result, "stop")


# ---------------------------------------------------------------------------
# Motion commands (with parameters)
# ---------------------------------------------------------------------------


@go2.command()
@click.option(
    "--direction",
    type=click.Choice(["forward", "backward", "left", "right"]),
    default="forward",
    show_default=True,
    help="Direction of travel.",
)
@click.option(
    "--distance",
    type=float,
    default=1.0,
    show_default=True,
    help="Distance to travel in metres.",
)
@click.option(
    "--speed",
    type=float,
    default=0.3,
    show_default=True,
    help="Translational speed in m/s (max 0.5).",
)
@click.pass_context
def walk(
    ctx: click.Context,
    direction: str,
    distance: float,
    speed: float,
) -> None:
    """Walk in a direction for a given distance."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill(
        "walk", {"direction": direction, "distance": distance, "speed": speed}
    )
    render_skill_result(rctx.console, result, "walk")


@go2.command()
@click.option(
    "--direction",
    type=click.Choice(["left", "right"]),
    default="left",
    show_default=True,
    help="Turn direction.",
)
@click.option(
    "--angle",
    type=float,
    default=90.0,
    show_default=True,
    help="Rotation angle in degrees.",
)
@click.pass_context
def turn(ctx: click.Context, direction: str, angle: float) -> None:
    """Rotate by a given angle."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("turn", {"direction": direction, "angle": angle})
    render_skill_result(rctx.console, result, "turn")


# ---------------------------------------------------------------------------
# Perception commands
# ---------------------------------------------------------------------------


@go2.command("where-am-i")
@click.pass_context
def where_am_i(ctx: click.Context) -> None:
    """Report current room / position estimate."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("where am i")
    render_skill_result(rctx.console, result, "where am i")


@go2.command()
@click.option(
    "--query",
    type=str,
    default=None,
    help="Optional focus query for scene description.",
)
@click.pass_context
def look(ctx: click.Context, query: str | None) -> None:
    """Capture and describe the current camera view."""
    rctx: RoboContext = ctx.obj
    params: dict = {}
    if query is not None:
        params["query"] = query
    result = rctx.execute_skill("look", params)
    render_skill_result(rctx.console, result, "look")


@go2.command("describe")
@click.argument("query")
@click.pass_context
def describe(ctx: click.Context, query: str) -> None:
    """Describe the scene with a specific QUERY."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("describe_scene", {"query": query})
    render_skill_result(rctx.console, result, "describe")


# ---------------------------------------------------------------------------
# Navigation / autonomy commands
# ---------------------------------------------------------------------------


@go2.command()
@click.argument("room")
@click.pass_context
def navigate(ctx: click.Context, room: str) -> None:
    """Navigate to ROOM by name."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("navigate", {"room": room})
    render_skill_result(rctx.console, result, "navigate")


@go2.command()
@click.pass_context
def explore(ctx: click.Context) -> None:
    """Autonomously explore the environment."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("explore")
    render_skill_result(rctx.console, result, "explore")


@go2.command()
@click.option(
    "--rooms",
    type=str,
    default=None,
    help="Comma-separated list of rooms to patrol (default: all known rooms).",
)
@click.option(
    "--max-rooms",
    type=int,
    default=None,
    help="Maximum number of rooms to visit.",
)
@click.option(
    "--timeout",
    type=float,
    default=300.0,
    show_default=True,
    help="Total patrol timeout in seconds.",
)
@click.pass_context
def patrol(
    ctx: click.Context,
    rooms: str | None,
    max_rooms: int | None,
    timeout: float,
) -> None:
    """Patrol a set of rooms in sequence."""
    rctx: RoboContext = ctx.obj
    params: dict = {"timeout": timeout}
    if rooms is not None:
        params["rooms"] = [r.strip() for r in rooms.split(",") if r.strip()]
    if max_rooms is not None:
        params["max_rooms"] = max_rooms
    result = rctx.execute_skill("patrol", params)
    render_skill_result(rctx.console, result, "patrol")
