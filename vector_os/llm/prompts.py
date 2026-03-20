"""System prompts and tool definitions for Vector OS LLM task planner.

Provides:
- PLANNING_SYSTEM_PROMPT: template with {skills_json} and {world_state_json} placeholders
- build_planning_prompt(): fills template with actual skill schemas and world state
- build_tool_definitions(): converts skill schemas to OpenAI function-calling format
"""
from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

PLANNING_SYSTEM_PROMPT = """\
You are a robot task planner for Vector OS.

You have access to these skills:
{skills_json}

Current world state:
{world_state_json}

Your job: decompose the user's instruction into a sequence of skill calls.

Output format (JSON):
{{
  "steps": [
    {{
      "step_id": "s1",
      "skill_name": "detect",
      "parameters": {{"query": "red cup"}},
      "depends_on": [],
      "preconditions": [],
      "postconditions": ["object_visible(result_object_id)"]
    }},
    ...
  ]
}}

If the instruction is ambiguous, output:
{{
  "requires_clarification": true,
  "clarification_question": "Which cup do you mean — the red one or the blue one?"
}}

Rules:
- Only use skills from the list above
- Parameters must match the skill's parameter schema
- Include dependencies between steps (step_id references in depends_on)
- Keep plans short (prefer fewer steps)
- If the gripper is holding something, you must place it before picking something else
- Output ONLY the JSON object — no explanation, no markdown fences
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_planning_prompt(
    skill_schemas: list[dict[str, Any]],
    world_state: dict[str, Any],
) -> str:
    """Build the system prompt for the LLM task planner.

    Fills PLANNING_SYSTEM_PROMPT with:
    - skill_schemas serialized as pretty JSON
    - world_state serialized as pretty JSON

    Args:
        skill_schemas: list of skill schema dicts from SkillRegistry.to_schemas().
        world_state: current world model snapshot (dict).

    Returns:
        Formatted system prompt string ready to send as a system message.
    """
    skills_json = json.dumps(skill_schemas, indent=2, ensure_ascii=False)
    world_state_json = json.dumps(world_state, indent=2, ensure_ascii=False)
    return PLANNING_SYSTEM_PROMPT.format(
        skills_json=skills_json,
        world_state_json=world_state_json,
    )


def build_tool_definitions(skill_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert skill schemas to OpenAI function-calling tool format.

    Each skill schema::

        {
            "name": "pick",
            "description": "Pick up a detected object",
            "parameters": {
                "object_id": {"type": "string", "description": "Object to pick"},
            },
            ...
        }

    becomes an OpenAI tool definition::

        {
            "type": "function",
            "function": {
                "name": "pick",
                "description": "Pick up a detected object",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_id": {"type": "string", "description": "Object to pick"},
                    },
                    "required": ["object_id"],
                },
            },
        }

    Args:
        skill_schemas: list of skill schema dicts from SkillRegistry.to_schemas().

    Returns:
        List of OpenAI-format tool definition dicts.
    """
    tools: list[dict[str, Any]] = []
    for skill in skill_schemas:
        raw_params: dict[str, Any] = skill.get("parameters", {})

        # Determine required parameters: those that don't have a "default" key
        required = [
            name
            for name, schema in raw_params.items()
            if not isinstance(schema, dict) or "default" not in schema
        ]

        tool: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": skill["name"],
                "description": skill.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": raw_params,
                    "required": required,
                },
            },
        }
        tools.append(tool)
    return tools
