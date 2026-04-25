# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""ROS2 agent node — wraps the Agent class as a ROS2 node.

Services:
    /agent/execute  (std_srvs/Trigger) — execute an instruction from the
                     'instruction' ROS2 parameter. Response message contains
                     JSON-encoded ExecutionResult.

    /agent/plan     (std_srvs/Trigger) — plan only (no execution). Response
                     message contains JSON-encoded TaskPlan.

Publications:
    /agent/status   (std_msgs/String) — "idle" | "planning" | "executing" | "done"
"""
from __future__ import annotations

import json
import logging

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

logger = logging.getLogger(__name__)


class AgentNode(Node):
    """ROS2 wrapper around the Agent class.

    The Agent handles LLM planning and task execution internally.
    This node exposes the agent's capabilities over ROS2 services.

    Args:
        agent: Agent instance to wrap.
    """

    def __init__(self, agent: object) -> None:
        super().__init__("agent")

        self._agent = agent

        # Parameters
        self.declare_parameter("instruction", "")

        # Publications
        self._status_pub = self.create_publisher(String, "/agent/status", 10)

        # Services
        self.create_service(Trigger, "/agent/execute", self._execute_cb)
        self.create_service(Trigger, "/agent/plan", self._plan_cb)

        self._publish_status("idle")
        self.get_logger().info("AgentNode ready")

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------

    def _execute_cb(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Execute the instruction set in the 'instruction' parameter."""
        instruction: str = (
            self.get_parameter("instruction").get_parameter_value().string_value
        )
        if not instruction:
            response.success = False
            response.message = json.dumps(
                {"error": "No instruction set. Use 'instruction' parameter."}
            )
            return response

        self._publish_status("planning")
        try:
            result = self._agent.execute(instruction)  # type: ignore[attr-defined]
            self._publish_status("done")
            response.success = result.success if hasattr(result, "success") else True
            response.message = json.dumps(
                result.to_dict() if hasattr(result, "to_dict") else str(result)
            )
        except Exception as exc:
            self.get_logger().error(f"Agent execute failed: {exc}")
            self._publish_status("idle")
            response.success = False
            response.message = json.dumps({"error": str(exc)})

        self._publish_status("idle")
        return response

    def _plan_cb(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Plan (without executing) the instruction in the 'instruction' parameter."""
        instruction: str = (
            self.get_parameter("instruction").get_parameter_value().string_value
        )
        if not instruction:
            response.success = False
            response.message = json.dumps(
                {"error": "No instruction set. Use 'instruction' parameter."}
            )
            return response

        self._publish_status("planning")
        try:
            plan = self._agent.plan(instruction)  # type: ignore[attr-defined]
            self._publish_status("idle")
            response.success = True
            response.message = json.dumps(
                plan.to_dict() if hasattr(plan, "to_dict") else str(plan)
            )
        except Exception as exc:
            self.get_logger().error(f"Agent plan failed: {exc}")
            self._publish_status("idle")
            response.success = False
            response.message = json.dumps({"error": str(exc)})

        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish_status(self, status: str) -> None:
        self._status_pub.publish(String(data=status))


def main(args: list[str] | None = None) -> None:
    """Entry point: requires an Agent instance — for standalone testing only."""
    raise RuntimeError(
        "AgentNode requires an Agent instance. "
        "Construct AgentNode programmatically with an Agent argument."
    )


if __name__ == "__main__":
    main()
