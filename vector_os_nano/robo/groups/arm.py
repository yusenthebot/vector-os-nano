"""Arm and gripper command groups for Vector CLI."""
from __future__ import annotations

import click

from vector_os_nano.robo.context import RoboContext
from vector_os_nano.robo.output import render_skill_result


@click.group("arm")
@click.pass_context
def arm(ctx: click.Context) -> None:
    """SO-101 robot arm commands."""


@arm.command("home")
@click.pass_context
def home(ctx: click.Context) -> None:
    """Move arm to home position."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("home")
    render_skill_result(rctx.console, result, "home")


@arm.command("scan")
@click.pass_context
def scan(ctx: click.Context) -> None:
    """Sweep arm through scan pose."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("scan")
    render_skill_result(rctx.console, result, "scan")


@arm.command("wave")
@click.pass_context
def wave(ctx: click.Context) -> None:
    """Perform a wave gesture."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("wave")
    render_skill_result(rctx.console, result, "wave")


@arm.command("pick")
@click.argument("object_name")
@click.pass_context
def pick(ctx: click.Context, object_name: str) -> None:
    """Pick up OBJECT_NAME."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("pick", {"object": object_name})
    render_skill_result(rctx.console, result, "pick")


@arm.command("place")
@click.argument("location")
@click.pass_context
def place(ctx: click.Context, location: str) -> None:
    """Place held object at LOCATION."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("place", {"location": location})
    render_skill_result(rctx.console, result, "place")


@arm.command("handover")
@click.pass_context
def handover(ctx: click.Context) -> None:
    """Extend arm for human handover."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("handover")
    render_skill_result(rctx.console, result, "handover")


@arm.command("detect")
@click.argument("query", required=False, default=None)
@click.pass_context
def detect(ctx: click.Context, query: str | None) -> None:
    """Detect objects in scene, optionally filtered by QUERY."""
    rctx: RoboContext = ctx.obj
    params = {"query": query} if query else {}
    result = rctx.execute_skill("detect", params)
    render_skill_result(rctx.console, result, "detect")


@arm.command("describe")
@click.pass_context
def describe(ctx: click.Context) -> None:
    """Describe the current scene."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("describe")
    render_skill_result(rctx.console, result, "describe")


# ---------------------------------------------------------------------------
# Gripper group
# ---------------------------------------------------------------------------


@click.group("gripper")
@click.pass_context
def gripper(ctx: click.Context) -> None:
    """Gripper open/close commands."""


@gripper.command("open")
@click.pass_context
def gripper_open(ctx: click.Context) -> None:
    """Open the gripper."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("gripper_open")
    render_skill_result(rctx.console, result, "gripper_open")


@gripper.command("close")
@click.pass_context
def gripper_close(ctx: click.Context) -> None:
    """Close the gripper."""
    rctx: RoboContext = ctx.obj
    result = rctx.execute_skill("gripper_close")
    render_skill_result(rctx.console, result, "gripper_close")
