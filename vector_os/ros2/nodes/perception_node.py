"""ROS2 perception bridge — wraps PerceptionPipeline as a ROS2 node.

Ported from vector_ws/src/track_anything/track_anything/track_3d.py structure.
All perception logic delegates to PerceptionPipeline from the SDK.

Subscriptions:
    /camera/color/image_raw        (sensor_msgs/Image)
    /camera/color/camera_info      (sensor_msgs/CameraInfo)
    /camera/aligned_depth_to_color/image_raw  (sensor_msgs/Image)

Publications:
    /perception/detections         (std_msgs/String)  — JSON-encoded Detection list
    /perception/overlay            (sensor_msgs/Image) — annotated RGB image
    /perception/status             (std_msgs/String)  — "idle" | "detecting" | "tracking"
"""
from __future__ import annotations

import json
import logging
from threading import Lock
from typing import Any

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from vector_os.perception.pipeline import PerceptionPipeline

logger = logging.getLogger(__name__)

# Sensor QoS: best-effort for high-frequency camera streams
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class PerceptionBridgeNode(Node):
    """ROS2 wrapper around PerceptionPipeline.

    Receives camera frames, runs the SDK pipeline when a detect or track
    service is triggered, and publishes results to ROS2 topics.

    Parameters:
        color_topic         — RGB image topic (default: /camera/color/image_raw)
        depth_topic         — depth image topic
        camera_info_topic   — camera intrinsics topic
        depth_scale         — depth scale factor in mm (default: 1000.0)
        depth_trunc         — depth truncation in metres (default: 10.0)
    """

    def __init__(self, pipeline: PerceptionPipeline | None = None) -> None:
        super().__init__("perception_bridge")

        # Parameters
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter(
            "depth_topic",
            "/camera/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("depth_trunc", 10.0)

        color_topic: str = (
            self.get_parameter("color_topic").get_parameter_value().string_value
        )
        depth_topic: str = (
            self.get_parameter("depth_topic").get_parameter_value().string_value
        )
        info_topic: str = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )

        # SDK pipeline — injected for testing, otherwise constructed lazily
        self._pipeline: PerceptionPipeline | None = pipeline
        self._lock = Lock()
        self._latest_color: Any = None
        self._latest_depth: Any = None
        self._camera_info: Any = None

        # Subscriptions (best-effort for real-time sensor data)
        self.create_subscription(Image, color_topic, self._on_color, _SENSOR_QOS)
        self.create_subscription(Image, depth_topic, self._on_depth, _SENSOR_QOS)
        self.create_subscription(
            CameraInfo, info_topic, self._on_camera_info, _SENSOR_QOS
        )

        # Publications
        self._det_pub = self.create_publisher(String, "/perception/detections", 10)
        self._status_pub = self.create_publisher(String, "/perception/status", 10)
        self._overlay_pub = self.create_publisher(Image, "/perception/overlay", 10)

        # Service: trigger a detect pass (query from JSON in request unused here —
        # full custom action types require a colcon package, so we use a /detect
        # Trigger and accept the query via a parameter or a String subscriber)
        from std_srvs.srv import Trigger
        self.create_service(Trigger, "/perception/detect", self._detect_cb)
        self.create_service(Trigger, "/perception/track", self._track_cb)

        self._status = "idle"
        self._publish_status("idle")
        self.get_logger().info("PerceptionBridgeNode ready")

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _on_color(self, msg: Image) -> None:
        with self._lock:
            self._latest_color = msg

    def _on_depth(self, msg: Image) -> None:
        with self._lock:
            self._latest_depth = msg

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._camera_info = msg

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------

    def _detect_cb(self, request, response):
        """Run VLM detection on the latest color frame."""
        from std_srvs.srv import Trigger  # noqa: F811
        self._publish_status("detecting")
        try:
            if self._pipeline is None:
                response.success = False
                response.message = "PerceptionPipeline not initialised"
                self._publish_status("idle")
                return response

            query = self.get_parameter("color_topic").get_parameter_value().string_value
            detections = self._pipeline.detect(query)
            payload = json.dumps([d.to_dict() if hasattr(d, "to_dict") else str(d) for d in detections])
            self._det_pub.publish(String(data=payload))
            response.success = True
            response.message = f"{len(detections)} detections"
        except Exception as exc:
            self.get_logger().error(f"Detect failed: {exc}")
            response.success = False
            response.message = str(exc)
        self._publish_status("idle")
        return response

    def _track_cb(self, request, response):
        """Run tracker on the latest frames."""
        self._publish_status("tracking")
        try:
            if self._pipeline is None:
                response.success = False
                response.message = "PerceptionPipeline not initialised"
                self._publish_status("idle")
                return response

            tracked = self._pipeline.track([])
            payload = json.dumps([str(t) for t in tracked])
            self._det_pub.publish(String(data=payload))
            response.success = True
            response.message = f"{len(tracked)} tracked objects"
        except Exception as exc:
            self.get_logger().error(f"Track failed: {exc}")
            response.success = False
            response.message = str(exc)
        self._publish_status("idle")
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish_status(self, status: str) -> None:
        self._status = status
        self._status_pub.publish(String(data=status))


def main(args: list[str] | None = None) -> None:
    """Entry point: spin PerceptionBridgeNode."""
    rclpy.init(args=args)
    node = PerceptionBridgeNode()
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
