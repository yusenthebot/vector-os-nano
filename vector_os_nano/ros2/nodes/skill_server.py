"""ROS2 skill server — exposes SDK skills as ROS2 Trigger services.

Each registered skill is exposed as a /skill/<name> Trigger service.
Since custom action types require a colcon package, we use std_srvs/Trigger
and encode parameters + results as JSON in the response message field.

Service naming:
    /skill/<name>  (std_srvs/Trigger)

The response message field contains a JSON object:
    {"success": true/false, "result": <skill result data or error string>}
"""
from __future__ import annotations

import json
import logging

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

logger = logging.getLogger(__name__)


class SkillServerNode(Node):
    """Exposes each registered skill as a /skill/<name> Trigger service.

    Parameters:
        The agent and its skill registry are injected at construction time.
        This node is typically created by AgentNode which holds the full agent.

    Args:
        agent: Agent instance whose ``skills`` attribute is a SkillRegistry.
        context: SkillContext for passing to skill execution.
    """

    def __init__(self, agent: object, context: object) -> None:
        super().__init__("skill_server")

        self._agent = agent
        self._context = context

        # Create a Trigger service for each registered skill
        skill_names: list[str] = agent.skills.list_skills()  # type: ignore[attr-defined]

        if not skill_names:
            self.get_logger().warn(
                "SkillServerNode: no skills registered — no services created"
            )
        else:
            for skill_name in skill_names:
                self.create_service(
                    Trigger,
                    f"/skill/{skill_name}",
                    lambda req, resp, name=skill_name: self._handle_skill(
                        req, resp, name
                    ),
                )
                self.get_logger().info(f"Registered /skill/{skill_name}")

        self.get_logger().info(f"SkillServerNode ready: {skill_names}")

    # ------------------------------------------------------------------
    # Service handler
    # ------------------------------------------------------------------

    def _handle_skill(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
        skill_name: str,
    ) -> Trigger.Response:
        """Execute a skill and encode result as JSON in response.message."""
        skill = self._agent.skills.get(skill_name)  # type: ignore[attr-defined]
        if skill is None:
            response.success = False
            response.message = json.dumps(
                {"success": False, "result": f"Skill not found: {skill_name!r}"}
            )
            return response

        try:
            result = skill.execute({}, self._context)
            response.success = result.success
            response.message = json.dumps(
                {
                    "success": result.success,
                    "result": (
                        result.result_data
                        if hasattr(result, "result_data")
                        else str(result)
                    ),
                    "error": result.error_message
                    if hasattr(result, "error_message")
                    else None,
                }
            )
        except Exception as exc:
            self.get_logger().error(f"Skill {skill_name!r} raised: {exc}")
            response.success = False
            response.message = json.dumps(
                {"success": False, "result": None, "error": str(exc)}
            )

        return response


def main(args: list[str] | None = None) -> None:
    """Entry point: requires an agent to be passed — for standalone testing only."""
    raise RuntimeError(
        "SkillServerNode requires an agent instance. "
        "Use AgentNode or create SkillServerNode programmatically."
    )


if __name__ == "__main__":
    main()
