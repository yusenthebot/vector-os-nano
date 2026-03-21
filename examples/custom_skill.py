"""Define and register a custom skill.

Custom skills are automatically exposed to the LLM planner. When the user
says "wave 5 times", the planner selects WaveSkill and passes {"times": 5}.

Requirements:
    pip install vector-os-nano

Usage:
    python examples/custom_skill.py
"""

from vector_os_nano import Agent, SO101
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import SkillResult


class WaveSkill:
    """Wave the arm back and forth by sweeping joint 0 (shoulder pan)."""

    name = "wave"
    description = "Wave the arm back and forth"
    parameters = {"times": {"type": "integer", "description": "Number of wave cycles", "default": 3}}
    preconditions: list[str] = []   # no preconditions required
    postconditions: list[str] = []  # no world model changes
    effects: dict = {}

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        times = int(params.get("times", 3))
        if context.arm is None:
            return SkillResult(success=False, error="No arm connected")

        for _ in range(times):
            joints = list(context.arm.get_joint_positions())
            # Sweep shoulder-pan joint left then right
            joints[0] = 0.5
            context.arm.move_joints(joints, duration=0.5)
            joints[0] = -0.5
            context.arm.move_joints(joints, duration=0.5)

        # Return to centre
        joints[0] = 0.0
        context.arm.move_joints(joints, duration=0.5)

        return SkillResult(success=True)


# Register alongside all built-in skills (pick, place, home, scan, detect)
arm = SO101(port="/dev/ttyACM0")
agent = Agent(
    arm=arm,
    llm_api_key="your-key-here",
    skills=[WaveSkill()],
)

# LLM planner will select WaveSkill and pass {"times": 5}
result = agent.execute("wave 5 times")
print(f"Success: {result.success}")
