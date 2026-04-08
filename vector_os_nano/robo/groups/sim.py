"""Simulation lifecycle command group for Vector CLI."""
from __future__ import annotations

import os
import subprocess

import click

from vector_os_nano.robo.context import RoboContext
from vector_os_nano.robo.output import render_error, render_info

# Processes to kill on sim stop (by name fragment passed to pkill -f)
_SIM_PROC_PATTERNS = [
    "mujoco",
    "go2_mujoco",
    "unitree_ros2_mujoco",
    "sim.sh",
    "run.py --sim",
]

_RESET_FLAG = "/tmp/vector_reset_pose"
_SIM_SH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "sim.sh",
)


@click.group("sim")
@click.pass_context
def sim(ctx: click.Context) -> None:
    """Simulation lifecycle commands."""


@sim.command("start")
@click.option(
    "--robot",
    type=click.Choice(["go2", "arm"]),
    default="go2",
    show_default=True,
    help="Robot model to simulate.",
)
@click.option("--headless", is_flag=True, help="Run without GUI.")
@click.pass_context
def start(ctx: click.Context, robot: str, headless: bool) -> None:
    """Start MuJoCo simulation."""
    rctx: RoboContext = ctx.obj

    if robot == "arm":
        render_error(rctx.console, "Arm simulation not yet implemented.")
        return

    # robot == "go2"
    if not os.path.isfile(_SIM_SH):
        render_error(rctx.console, f"sim.sh not found at {_SIM_SH}")
        return

    cmd = ["/bin/bash", _SIM_SH]
    if headless:
        cmd.append("--headless")

    render_info(rctx.console, f"Starting Go2 simulation ({' '.join(cmd)}) ...")
    try:
        # Launch detached so the CLI returns immediately
        subprocess.Popen(
            cmd,
            cwd=os.path.dirname(_SIM_SH),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        render_info(rctx.console, "Simulation launched in background.")
    except Exception as exc:
        render_error(rctx.console, f"Failed to start simulation: {exc}")


@sim.command("reset")
@click.pass_context
def reset(ctx: click.Context) -> None:
    """Reset robot pose (writes flag file for bridge reset handler)."""
    rctx: RoboContext = ctx.obj
    try:
        open(_RESET_FLAG, "w").close()
        render_info(rctx.console, f"Reset flag written to {_RESET_FLAG}.")
    except OSError as exc:
        render_error(rctx.console, f"Could not write reset flag: {exc}")


@sim.command("stop")
@click.pass_context
def stop(ctx: click.Context) -> None:
    """Stop running simulation processes."""
    rctx: RoboContext = ctx.obj
    killed_any = False
    for pattern in _SIM_PROC_PATTERNS:
        try:
            result = subprocess.run(
                ["pkill", "-f", pattern],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                killed_any = True
        except Exception:
            pass

    if killed_any:
        render_info(rctx.console, "Simulation processes stopped.")
    else:
        render_info(rctx.console, "No simulation processes found running.")
