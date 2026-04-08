"""Rich output helpers for Vector CLI.

Standardized rendering for skill results, status panels, tables,
and error messages. All CLI output goes through these helpers.
"""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def render_skill_result(console: Console, result: Any, skill_name: str) -> None:
    """Render a SkillResult as a Rich panel."""
    if result.success:
        title = f"[green]{skill_name}[/green]"
        data = result.result_data or {}
        lines = []
        for k, v in data.items():
            lines.append(f"  {k}: {v}")
        body = "\n".join(lines) if lines else "OK"
        console.print(Panel(body, title=title, border_style="green"))
    else:
        title = f"[red]{skill_name}[/red]"
        msg = result.error_message or "Failed"
        if result.diagnosis_code:
            msg += f"  ({result.diagnosis_code})"
        console.print(Panel(msg, title=title, border_style="red"))


def render_status(console: Console, status: dict[str, Any]) -> None:
    """Render hardware/connection status as a table."""
    table = Table(title="Vector Status", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    # Base (Go2)
    base = status.get("base")
    if base and base.get("connected"):
        pos = base.get("position", [0, 0, 0])
        table.add_row(
            "Base",
            "[green]connected[/green]",
            f"{base['name']}  pos=({pos[0]}, {pos[1]}, {pos[2]})",
        )
    else:
        table.add_row("Base", "[dim]disconnected[/dim]", "")

    # Arm
    arm_info = status.get("arm")
    if arm_info and arm_info.get("connected"):
        table.add_row("Arm", "[green]connected[/green]", arm_info.get("name", ""))
    else:
        table.add_row("Arm", "[dim]disconnected[/dim]", "")

    # Gripper
    grip = status.get("gripper")
    if grip and grip.get("connected"):
        table.add_row("Gripper", "[green]connected[/green]", "")
    else:
        table.add_row("Gripper", "[dim]disconnected[/dim]", "")

    # Skills
    table.add_row("Skills", str(status.get("skills", 0)), "registered")

    # ROS2
    ros2 = status.get("ros2", False)
    if ros2:
        table.add_row("ROS2", "[green]active[/green]", "")
    else:
        table.add_row("ROS2", "[dim]inactive[/dim]", "")

    console.print(table)


def render_skills_list(console: Console, registry: Any) -> None:
    """Render all registered skills as a table."""
    table = Table(title="Registered Skills", show_header=True, header_style="bold cyan")
    table.add_column("Skill", style="bold")
    table.add_column("Aliases")
    table.add_column("Direct")
    table.add_column("Description")

    for name in sorted(registry.list_skills()):
        skill = registry.get(name)
        if skill is None:
            continue

        aliases_list = getattr(skill, "__skill_aliases__", [])
        direct = getattr(skill, "__skill_direct__", False)
        desc = getattr(skill, "description", "")

        # Truncate description
        if len(desc) > 60:
            desc = desc[:57] + "..."

        aliases_str = ", ".join(aliases_list[:4])
        if len(aliases_list) > 4:
            aliases_str += f" (+{len(aliases_list) - 4})"

        table.add_row(
            name,
            aliases_str,
            "[green]yes[/green]" if direct else "[dim]no[/dim]",
            desc,
        )

    console.print(table)


def render_error(console: Console, message: str) -> None:
    """Render an error message."""
    console.print(f"[red]Error:[/red] {message}")


def render_info(console: Console, message: str) -> None:
    """Render an info message."""
    console.print(f"[cyan]>[/cyan] {message}")
