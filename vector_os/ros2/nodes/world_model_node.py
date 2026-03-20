"""ROS2 world model service node — exposes WorldModel queries over ROS2.

Services:
    /world_model/query   (std_srvs/Trigger) — request: ignored,
                          response.message: full world state as JSON

    /world_model/predicate (std_srvs/Trigger) — check a predicate;
                          use parameter 'predicate' to set the query string

Subscriptions:
    /joint_states   (sensor_msgs/JointState) — updates robot joint state
    /perception/detections (std_msgs/String) — JSON Detection list, updates objects
"""
from __future__ import annotations

import json
import logging

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from vector_os.core.world_model import ObjectState, WorldModel

logger = logging.getLogger(__name__)


class WorldModelServiceNode(Node):
    """ROS2 wrapper around WorldModel.

    Keeps the world model up-to-date from sensor topics and exposes
    query services so other nodes can read world state without direct
    Python imports.

    Args:
        world_model: WorldModel instance to wrap. If None, creates a fresh one.
    """

    def __init__(self, world_model: WorldModel | None = None) -> None:
        super().__init__("world_model_service")

        # Parameters
        self.declare_parameter("predicate", "gripper_empty")

        self._wm: WorldModel = world_model if world_model is not None else WorldModel()

        # Subscriptions
        self.create_subscription(
            JointState, "/joint_states", self._on_joint_states, 10
        )
        self.create_subscription(
            String, "/perception/detections", self._on_detections, 10
        )

        # Services
        self.create_service(Trigger, "/world_model/query", self._query_cb)
        self.create_service(Trigger, "/world_model/predicate", self._predicate_cb)

        self.get_logger().info("WorldModelServiceNode ready")

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _on_joint_states(self, msg: JointState) -> None:
        """Update robot joint positions in the world model."""
        if not msg.name or not msg.position:
            return
        positions = tuple(float(p) for p in msg.position)
        self._wm.update_robot_state(joint_positions=positions)

    def _on_detections(self, msg: String) -> None:
        """Parse JSON Detection list and update world model objects."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Failed to parse detection JSON: {exc}")
            return

        if not isinstance(data, list):
            return

        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                obj = ObjectState(
                    object_id=item.get("object_id", item.get("label", "unknown")),
                    label=item.get("label", "unknown"),
                    x=float(item.get("x", 0.0)),
                    y=float(item.get("y", 0.0)),
                    z=float(item.get("z", 0.0)),
                    confidence=float(item.get("confidence", 1.0)),
                    state=item.get("state", "on_table"),
                )
                self._wm.add_object(obj)
            except Exception as exc:
                self.get_logger().warn(f"Failed to parse object state: {exc}")

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------

    def _query_cb(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Return full world state as JSON in response.message."""
        try:
            response.success = True
            response.message = json.dumps(self._wm.to_dict())
        except Exception as exc:
            self.get_logger().error(f"World model query failed: {exc}")
            response.success = False
            response.message = str(exc)
        return response

    def _predicate_cb(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Check the predicate configured in the 'predicate' parameter."""
        predicate: str = (
            self.get_parameter("predicate").get_parameter_value().string_value
        )
        try:
            result = self._wm.check_predicate(predicate)
            response.success = result
            response.message = json.dumps(
                {"predicate": predicate, "result": result}
            )
        except Exception as exc:
            self.get_logger().error(f"Predicate check failed: {exc}")
            response.success = False
            response.message = str(exc)
        return response

    # ------------------------------------------------------------------
    # Property access (for in-process users)
    # ------------------------------------------------------------------

    @property
    def world_model(self) -> WorldModel:
        """Direct access to the underlying WorldModel instance."""
        return self._wm


def main(args: list[str] | None = None) -> None:
    """Entry point: spin WorldModelServiceNode."""
    rclpy.init(args=args)
    node = WorldModelServiceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
