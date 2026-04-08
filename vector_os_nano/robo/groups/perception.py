"""Perception and VLM command group for Vector CLI."""
from __future__ import annotations

import click

from vector_os_nano.robo.context import RoboContext
from vector_os_nano.robo.output import render_skill_result


@click.group("perception")
@click.pass_context
def perception(ctx: click.Context) -> None:
    """Perception and VLM commands."""


@perception.command("detect")
@click.option("--query", "-q", default=None, help="What to detect.")
@click.pass_context
def detect(ctx: click.Context, query: str | None) -> None:
    """Detect objects using VLM."""
    rctx: RoboContext = ctx.obj
    params = {"query": query} if query else {}
    result = rctx.execute_skill("detect", params)
    render_skill_result(rctx.console, result, "detect")


@perception.command("describe")
@click.option("--query", "-q", default=None, help="Specific question about the scene.")
@click.pass_context
def describe(ctx: click.Context, query: str | None) -> None:
    """Describe the current scene using VLM."""
    rctx: RoboContext = ctx.obj
    params = {"query": query} if query else {}
    result = rctx.execute_skill("describe", params)
    render_skill_result(rctx.console, result, "describe")
