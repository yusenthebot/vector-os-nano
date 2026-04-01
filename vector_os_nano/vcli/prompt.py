"""System prompt builder for the Vector CLI agentic harness.

Builds a multi-block system prompt with:
- Static sections (ROLE_PROMPT, TOOL_INSTRUCTIONS) with cache_control
- Dynamic sections (hardware, skills, world model, VECTOR.md)

Public API:
    build_system_prompt(agent, cwd, session) -> list[dict]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Static prompt text — these are cacheable
# ---------------------------------------------------------------------------

ROLE_PROMPT = """\
You are V, the AI agent for Vector OS Nano, created by Vector Robotics.
You control robot hardware (arms, grippers, mobile bases) through tool calls.
You also read/write files, run shell commands, and search codebases.

You communicate in whatever language the user uses.
In Chinese, call the user "主人". In English, use their name if known.

Safety first:
- Never command motions that could damage hardware, knock objects, or harm humans.
- If a command seems dangerous or ambiguous, ask for clarification before executing.
- If a skill fails, report clearly and suggest a next step.

Communication:
- Keep responses concise. One to three sentences unless detail is requested.
- Write flowing plain text. Separate ideas with commas or line breaks.
- FORBIDDEN: markdown headers (# ##), bold (**), bullet markers (- *), \
numbered lists, code blocks, horizontal rules (---), emojis.
- When executing a robot command, briefly say what you will do, then act.
- Never introduce yourself unprompted. Just answer.

Behavior:
- Be proactive. Infer intent from ambiguous requests.
- Remember previous conversation context within the session.
- When reporting results, state facts: what was done, whether it succeeded.
"""

TOOL_INSTRUCTIONS = """\
Tool usage:
- Robot tools wrap real hardware skills. Motor tools require user permission.
- Dev tools (file_read, file_write, file_edit, bash, glob, grep) work like a coding assistant.
- Read-only tools run without permission and may execute in parallel.
- If a tool returns is_error=true, report the error and suggest alternatives.

Safety:
- Check robot_status before moving joints to extreme positions.
- Always detect/scan before attempting pick operations.
- Report hardware errors immediately. Do not retry motor commands silently.
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_system_prompt(
    agent: Any = None,
    cwd: Path | None = None,
    session: Any = None,
) -> list[dict]:
    """Build system prompt as a list of text blocks.

    Static blocks carry ``cache_control`` for server-side caching.
    Dynamic blocks (hardware, skills, world, VECTOR.md) are regenerated each call.
    """
    blocks: list[dict] = []

    # -- Static (cacheable) --------------------------------------------------
    blocks.append(
        {
            "type": "text",
            "text": ROLE_PROMPT.strip(),
            "cache_control": {"type": "ephemeral"},
        }
    )
    blocks.append(
        {
            "type": "text",
            "text": TOOL_INSTRUCTIONS.strip(),
            "cache_control": {"type": "ephemeral"},
        }
    )

    # -- Dynamic: hardware state ---------------------------------------------
    if agent is not None:
        hw_text = _format_hardware(agent)
        if hw_text:
            blocks.append({"type": "text", "text": f"Current Hardware:\n{hw_text}"})

    # -- Dynamic: available skills -------------------------------------------
    if agent is not None:
        skills_text = _format_skills(agent)
        if skills_text:
            blocks.append({"type": "text", "text": f"Available Skills:\n{skills_text}"})

    # -- Dynamic: world model ------------------------------------------------
    if agent is not None:
        world_text = _format_world(agent)
        if world_text:
            blocks.append({"type": "text", "text": f"World Model:\n{world_text}"})

    # -- Dynamic: VECTOR.md --------------------------------------------------
    vector_md = _load_vector_md(cwd)
    if vector_md:
        blocks.append(
            {"type": "text", "text": f"Project Context (VECTOR.md):\n{vector_md}"}
        )

    return blocks


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_hardware(agent: Any) -> str:
    """Return a formatted string describing connected hardware, or '' if none."""
    lines: list[str] = []

    arm = getattr(agent, "_arm", None)
    if arm is not None:
        arm_name: str = getattr(arm, "name", type(arm).__name__)
        dof: int | None = getattr(arm, "dof", None)
        dof_str = f", {dof}-DOF" if dof is not None else ""
        lines.append(f"Arm: {arm_name}{dof_str}")

    gripper = getattr(agent, "_gripper", None)
    if gripper is not None:
        gripper_name: str = getattr(gripper, "name", type(gripper).__name__)
        lines.append(f"Gripper: {gripper_name}")

    base = getattr(agent, "_base", None)
    if base is not None:
        base_name: str = getattr(base, "name", type(base).__name__)
        holonomic: bool | None = getattr(base, "supports_holonomic", None)
        holonomic_str = " (holonomic)" if holonomic else ""
        lines.append(f"Base: {base_name}{holonomic_str}")

    perception = getattr(agent, "_perception", None)
    if perception is not None:
        perception_name: str = getattr(perception, "name", type(perception).__name__)
        lines.append(f"Perception: {perception_name}")

    return "\n".join(lines)


def _format_skills(agent: Any) -> str:
    """Return a formatted list of skill names + descriptions, or '' if empty."""
    registry = getattr(agent, "_skill_registry", None)
    if registry is None:
        return ""

    skill_names: list[str] = []
    try:
        skill_names = registry.list_skills()
    except Exception:
        return ""

    if not skill_names:
        return ""

    lines: list[str] = []
    for name in skill_names:
        try:
            skill = registry.get(name)
        except Exception:
            skill = None
        if skill is None:
            continue
        desc: str = getattr(skill, "description", "")
        lines.append(f"{name}: {desc}" if desc else name)

    return "\n".join(lines)


def _format_world(agent: Any) -> str:
    """Return a summary of world model objects, or '' if empty."""
    world_model = getattr(agent, "_world_model", None)
    if world_model is None:
        return ""

    objects: list[Any] = []
    try:
        objects = world_model.get_objects()
    except Exception:
        return ""

    if not objects:
        return ""

    lines: list[str] = []
    for obj in objects:
        label: str = getattr(obj, "label", str(obj))
        x = getattr(obj, "x", "?")
        y = getattr(obj, "y", "?")
        z = getattr(obj, "z", "?")
        _fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
        lines.append(f"{label}: ({_fmt(x)}, {_fmt(y)}, {_fmt(z)})")

    return "\n".join(lines)


def _load_vector_md(cwd: Path | None) -> str:
    """Load VECTOR.md from cwd and/or ~/.vector/VECTOR.md."""
    parts: list[str] = []

    if cwd is not None:
        local_path = cwd / "VECTOR.md"
        if local_path.is_file():
            try:
                parts.append(local_path.read_text(encoding="utf-8").strip())
            except OSError:
                pass

    home_path = Path.home() / ".vector" / "VECTOR.md"
    if home_path.is_file():
        try:
            content = home_path.read_text(encoding="utf-8").strip()
            if content:
                parts.append(content)
        except OSError:
            pass

    return "\n\n".join(parts)
