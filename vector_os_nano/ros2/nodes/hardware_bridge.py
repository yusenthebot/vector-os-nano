# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""ROS2 hardware bridge — wraps SO101Arm/SO101Gripper as a ROS2 node.

Ported from vector_ws/src/so101_hardware/so101_hardware/hardware_bridge.py.
All servo protocol logic is delegated to SO101Arm and SO101Gripper from the SDK.

Topics:
    Publish:   /joint_states   (sensor_msgs/JointState)  @ publish_rate Hz
    Subscribe: /joint_commands (sensor_msgs/JointState)  — direct position commands

Services:
    /gripper/open   (std_srvs/Trigger)
    /gripper/close  (std_srvs/Trigger)

Actions:
    /arm_controller/follow_joint_trajectory  (control_msgs/FollowJointTrajectory)
    /gripper_controller/gripper_command      (control_msgs/GripperCommand)
"""
from __future__ import annotations

import time as _time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory, GripperCommand
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from vector_os_nano.hardware.so101 import (
    JOINT_CONFIG,
    SO101Arm,
    SO101Gripper,
    SerialBus,
)
from vector_os_nano.hardware.so101.joint_config import enc_to_rad, rad_to_enc

# Gripper encoder limits (mirror of vector_ws constants)
_GRIPPER_OPEN_ENC: int = 2500
_GRIPPER_CLOSED_ENC: int = 1332


class HardwareBridgeNode(Node):
    """ROS2 lifecycle wrapper around SO101Arm and SO101Gripper.

    Publishes joint states at ``publish_rate`` Hz and accepts trajectory and
    gripper-command actions plus open/close trigger services.
    """

    def __init__(self) -> None:
        super().__init__("hardware_bridge")

        # Typed parameters
        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter("baudrate", 1_000_000)
        self.declare_parameter("publish_rate", 30.0)

        port: str = self.get_parameter("serial_port").get_parameter_value().string_value
        baud: int = self.get_parameter("baudrate").get_parameter_value().integer_value
        rate: float = self.get_parameter("publish_rate").get_parameter_value().double_value

        # SDK hardware objects — single SerialBus shared between arm and gripper
        self._arm = SO101Arm(port=port, baudrate=baud)
        self._gripper = SO101Gripper(self._arm._bus)

        # Concurrent callback group for action execute callbacks
        self._cb_group = ReentrantCallbackGroup()

        # Publisher: /joint_states
        self._joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._timer = self.create_timer(1.0 / rate, self._publish_joint_states)

        # Subscriber: /joint_commands
        self.create_subscription(
            JointState, "/joint_commands", self._on_joint_command, 10
        )

        # Services: gripper open / close
        self.create_service(Trigger, "/gripper/open", self._gripper_open_cb)
        self.create_service(Trigger, "/gripper/close", self._gripper_close_cb)

        # Action: FollowJointTrajectory
        self._traj_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
            execute_callback=self._execute_trajectory,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )

        # Action: GripperCommand
        self._gripper_action_server = ActionServer(
            self,
            GripperCommand,
            "/gripper_controller/gripper_command",
            execute_callback=self._execute_gripper_command,
            callback_group=self._cb_group,
        )

        # Connect hardware
        self._arm.connect()
        self.get_logger().info(f"Hardware bridge connected on {port}")

    # ------------------------------------------------------------------
    # Publisher
    # ------------------------------------------------------------------

    def _publish_joint_states(self) -> None:
        """Read servo positions and publish JointState at the configured rate."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        try:
            positions = self._arm.get_joint_positions()
        except Exception as exc:
            self.get_logger().warn(f"Failed to read joint positions: {exc}")
            return

        for name, pos in zip(self._arm.joint_names, positions):
            msg.name.append(name)
            msg.position.append(float(pos))

        self._joint_pub.publish(msg)

    # ------------------------------------------------------------------
    # Subscriber
    # ------------------------------------------------------------------

    def _on_joint_command(self, msg: JointState) -> None:
        """Forward direct position commands to the arm."""
        if not msg.name or not msg.position:
            return
        positions = dict(zip(msg.name, msg.position))
        try:
            # Build ordered position list matching arm.joint_names
            ordered = [
                positions.get(n, 0.0) for n in self._arm.joint_names
            ]
            self._arm.move_joints(ordered)
        except Exception as exc:
            self.get_logger().warn(f"Failed to apply joint command: {exc}")

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------

    def _gripper_open_cb(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        try:
            self._gripper.open()
            response.success = True
            response.message = f"Gripper opened (enc={_GRIPPER_OPEN_ENC})"
        except Exception as exc:
            response.success = False
            response.message = f"Gripper open failed: {exc}"
        return response

    def _gripper_close_cb(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        try:
            self._gripper.close()
            response.success = True
            response.message = f"Gripper closed (enc={_GRIPPER_CLOSED_ENC})"
        except Exception as exc:
            response.success = False
            response.message = f"Gripper close failed: {exc}"
        return response

    # ------------------------------------------------------------------
    # Action: GripperCommand
    # ------------------------------------------------------------------

    def _execute_gripper_command(
        self, goal_handle: GripperCommand.Impl.ExecuteGoalRequest
    ) -> GripperCommand.Result:
        """Map GripperCommand position to open/close and report back."""
        position_cmd: float = goal_handle.request.command.position

        try:
            if position_cmd >= 0.5:
                self._gripper.open()
                action = "open"
            else:
                self._gripper.close()
                action = "close"
            _time.sleep(1.0)
            self.get_logger().info(f"GripperCommand: {action}")
            normalized = self._gripper.get_position()
        except Exception as exc:
            self.get_logger().error(f"GripperCommand failed: {exc}")
            goal_handle.abort()
            return GripperCommand.Result()

        goal_handle.succeed()
        result = GripperCommand.Result()
        result.reached_goal = True
        result.position = float(normalized)
        result.effort = 0.0
        result.stalled = False
        return result

    # ------------------------------------------------------------------
    # Action: FollowJointTrajectory
    # ------------------------------------------------------------------

    def _goal_callback(self, goal_request) -> GoalResponse:
        if not goal_request.trajectory.points:
            self.get_logger().warn("Rejecting empty trajectory goal")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info("Cancel requested for trajectory execution")
        return CancelResponse.ACCEPT

    def _execute_trajectory(
        self, goal_handle: FollowJointTrajectory.Impl.ExecuteGoalRequest
    ) -> FollowJointTrajectory.Result:
        """Step through trajectory waypoints using inter-waypoint time deltas."""
        self.get_logger().info("Executing trajectory")
        trajectory = goal_handle.request.trajectory
        joint_names = list(trajectory.joint_names)
        points = trajectory.points

        unknown = [n for n in joint_names if n not in JOINT_CONFIG]
        if unknown:
            self.get_logger().warn(
                f"Trajectory contains unknown joints (will be skipped): {unknown}"
            )

        prev_time_sec = 0.0

        for idx, point in enumerate(points):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("Trajectory cancelled")
                return FollowJointTrajectory.Result()

            current_time_sec = (
                point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            )
            delay = current_time_sec - prev_time_sec
            if delay > 0.0:
                _time.sleep(delay)
            prev_time_sec = current_time_sec

            # Build ordered position list for arm.move_joints
            positions_map: dict[str, float] = {}
            for i, name in enumerate(joint_names):
                if name not in JOINT_CONFIG:
                    continue
                if i >= len(point.positions):
                    self.get_logger().warn(
                        f"Waypoint {idx}: position index {i} out of range, skipping {name}"
                    )
                    continue
                positions_map[name] = float(point.positions[i])

            if not positions_map:
                continue

            try:
                ordered = [
                    positions_map.get(n, 0.0) for n in self._arm.joint_names
                ]
                self._arm.move_joints(ordered)
            except Exception as exc:
                self.get_logger().error(f"Waypoint {idx}: failed to write joints: {exc}")
                goal_handle.abort()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                result.error_string = str(exc)
                return result

        _time.sleep(0.5)
        goal_handle.succeed()
        self.get_logger().info(
            f"Trajectory execution complete ({len(points)} waypoints)"
        )
        return FollowJointTrajectory.Result()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        try:
            self._arm.disconnect()
        except Exception as exc:
            self.get_logger().warn(f"Error during arm disconnect: {exc}")
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    """Entry point: spin HardwareBridgeNode with MultiThreadedExecutor."""
    rclpy.init(args=args)
    node = HardwareBridgeNode()
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
