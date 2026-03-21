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
You are an action-oriented robot task planner for Vector OS.
Your default is to EXECUTE — never ask for confirmation, never ask "should I scan?".
Respond in the same language the user uses (Chinese or English).

Available skills:
{skills_json}

Current world state:
{world_state_json}

EXECUTION RULES (follow strictly):
1. ALWAYS produce a plan. NEVER ask clarifying questions unless genuinely ambiguous.
2. Typos and variations are the SAME object: "proteinbar" = "protein bar" = "protein_bar".
3. Gripper commands map directly to skills:
   - "close grip / close gripper / grip / grasp" → close_gripper skill (or gripper_close)
   - "open grip / open gripper / release / drop" → open_gripper skill (or gripper_open)
   If no dedicated gripper skill exists, use the closest available skill.
4. When the user asks to pick an object NOT in the world state objects list:
   - Automatically prepend scan → detect steps before the pick step.
   - Do NOT ask "should I scan first?" — just do it.
5. When the gripper is holding something and the user wants to pick something new:
   - Prepend a place step first.
6. Keep plans short. Prefer fewer steps.
7. Only use skills from the list above. Parameters must match the skill schema exactly.
8. Include step dependencies in depends_on (reference step_id strings).

WHEN TO USE requires_clarification (rare):
- ONLY when multiple distinct objects share the exact same label and the user must
  choose one (e.g., "pick the cup" when two cups are visible with identical labels).
- NEVER use it for typos, missing objects, or gripper commands.

Output format — JSON only, no markdown fences, no explanation:
{{
  "steps": [
    {{
      "step_id": "s1",
      "skill_name": "scan",
      "parameters": {{}},
      "depends_on": [],
      "preconditions": [],
      "postconditions": ["scan_complete"]
    }},
    {{
      "step_id": "s2",
      "skill_name": "detect",
      "parameters": {{"query": "protein bar"}},
      "depends_on": ["s1"],
      "preconditions": [],
      "postconditions": ["object_visible(result_object_id)"]
    }},
    {{
      "step_id": "s3",
      "skill_name": "pick",
      "parameters": {{"object_id": "result_object_id"}},
      "depends_on": ["s2"],
      "preconditions": ["object_visible(result_object_id)"],
      "postconditions": ["gripper_holding_any"]
    }}
  ]
}}

Clarification format (use ONLY when genuinely ambiguous as described above):
{{
  "requires_clarification": true,
  "clarification_question": "I see two red cups — which one do you mean?"
}}
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
