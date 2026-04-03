"""Go2 ROS2 Proxy — controls Go2 via ROS2 topics instead of direct MuJoCo.

Used when the MuJoCo simulation is managed by an external process
(e.g., launch_explore.sh) and we need to send commands via ROS2.
"""
from __future__ import annotations

import math
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


class Go2ROS2Proxy:
    """Proxy that implements the same interface as MuJoCoGo2 but via ROS2 topics.

    Publishes: /cmd_vel_nav (Twist) for velocity commands
    Subscribes: /state_estimation (Odometry) for position/heading
                /camera/image (Image) for VLM perception
    """

    def __init__(self) -> None:
        self._node: Any = None
        self._cmd_pub: Any = None
        self._position: tuple[float, float, float] = (0.0, 0.0, 0.28)
        self._heading: float = 0.0
        self._connected: bool = False
        self._last_odom: Any = None
        self._last_camera_frame: Any = None  # numpy (H, W, 3) uint8
        self._last_depth_frame: Any = None   # numpy (H, W) float32 metres

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Initialise rclpy node, publisher, and odometry subscriber."""
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            from nav_msgs.msg import Odometry
            import threading

            if not rclpy.ok():
                rclpy.init()

            self._node = Node("go2_agent_proxy")

            reliable_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                depth=5,
            )

            from geometry_msgs.msg import Twist  # noqa: F401 — ensure importable
            from sensor_msgs.msg import Image

            self._cmd_pub = self._node.create_publisher(Twist, "/cmd_vel_nav", 10)
            self._node.create_subscription(
                Odometry, "/state_estimation", self._odom_cb, reliable_qos
            )
            self._node.create_subscription(
                Image, "/camera/image", self._camera_cb, reliable_qos
            )
            self._node.create_subscription(
                Image, "/camera/depth", self._depth_cb, reliable_qos
            )

            # Scene graph marker publisher (agent sets self._scene_graph)
            self._scene_graph = None
            self._nav_goal: tuple[float, float] | None = None
            self._trajectory: list[tuple[float, float]] = []
            self._last_marker_hash: int | None = None
            self._last_marker_publish_time: float = 0.0
            try:
                from visualization_msgs.msg import MarkerArray
                self._marker_pub = self._node.create_publisher(
                    MarkerArray, "/scene_graph_markers", 5
                )
                self._node.create_timer(3.0, self._publish_markers)
            except ImportError:
                self._marker_pub = None

            # Spin in a background daemon thread so the caller is not blocked.
            self._spin_thread = threading.Thread(
                target=lambda: rclpy.spin(self._node), daemon=True
            )
            self._spin_thread.start()
            self._connected = True

            # Wait up to 5 s for the first odometry message.
            for _ in range(50):
                if self._position != (0.0, 0.0, 0.28):
                    break
                time.sleep(0.1)

            logger.info("Go2ROS2Proxy connected")
        except Exception as exc:
            logger.error("Failed to connect Go2ROS2Proxy: %s", exc)
            self._connected = False

    def disconnect(self) -> None:
        """Destroy the rclpy node and mark proxy as disconnected."""
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        self._connected = False

    # ------------------------------------------------------------------
    # Internal callback
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Any) -> None:
        """Update cached position and heading from incoming Odometry message."""
        self._last_odom = msg
        self._position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        )
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._heading = math.atan2(siny_cosp, cosy_cosp)

    def _camera_cb(self, msg: Any) -> None:
        """Cache latest camera frame from /camera/image (RGB8 240x320)."""
        try:
            import numpy as np
            frame = np.frombuffer(msg.data, dtype=np.uint8)
            frame = frame.reshape((msg.height, msg.width, 3))
            self._last_camera_frame = frame
        except Exception:
            pass

    def _depth_cb(self, msg: Any) -> None:
        """Cache latest depth frame from /camera/depth (32FC1 240x320).

        Bridge publishes depth as 32FC1 (float32, single channel) in metres.
        """
        try:
            import numpy as np
            frame = np.frombuffer(msg.data, dtype=np.float32)
            frame = frame.reshape((msg.height, msg.width))
            self._last_depth_frame = frame
        except Exception:
            pass

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_position(self) -> tuple[float, float, float]:
        """Return last known (x, y, z) position in metres."""
        return self._position

    def get_heading(self) -> float:
        """Return last known heading in radians (yaw from odometry)."""
        return self._heading

    def get_camera_frame(self, width: int = 320, height: int = 240) -> Any:
        """Return latest RGB camera frame as (H, W, 3) uint8 numpy array.

        Received from /camera/image topic published by Go2VNavBridge.
        Returns a black frame if no image has been received yet.
        """
        import numpy as np
        if self._last_camera_frame is not None:
            return self._last_camera_frame.copy()
        return np.zeros((height, width, 3), dtype=np.uint8)

    def get_depth_frame(self, width: int = 320, height: int = 240) -> Any:
        """Return latest depth frame as (H, W) float32 array in metres.

        Received from /camera/depth topic (32FC1) published by Go2VNavBridge.
        Returns a zero frame if no depth has been received yet.
        """
        import numpy as np
        if self._last_depth_frame is not None:
            return self._last_depth_frame.copy()
        return np.zeros((height, width), dtype=np.float32)

    def get_rgbd_frame(self, width: int = 320, height: int = 240) -> Any:
        """Return aligned (rgb, depth) tuple.

        Sim-to-real compatible: same interface as MuJoCoGo2.get_rgbd_frame().
        """
        return self.get_camera_frame(width, height), self.get_depth_frame(width, height)

    def get_camera_pose(self) -> tuple:
        """Compute D435 camera world pose from robot odometry + mount config.

        Returns (cam_xpos, cam_xmat) matching MuJoCoGo2.get_camera_pose().
        Camera mounted at: 0.3m forward, 0.05m up, -5deg pitch on base_link.
        """
        import numpy as np

        pos = self._position
        heading = self._heading

        # Mount offset in body frame
        mount_fwd, mount_up = 0.3, 0.05
        pitch = math.radians(-5.0)  # -5 deg downward tilt

        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)

        # Camera world position
        cam_x = pos[0] + cos_h * mount_fwd
        cam_y = pos[1] + sin_h * mount_fwd
        cam_z = pos[2] + mount_up
        cam_xpos = np.array([cam_x, cam_y, cam_z])

        # Camera rotation: MuJoCo convention columns = [right, up, -forward]
        # Body frame: forward=(cos_h, sin_h, 0), right=(-sin_h, cos_h, 0)
        # With pitch: forward rotated by pitch around right axis
        fwd = np.array([cos_h * cos_p, sin_h * cos_p, sin_p])
        right = np.array([-sin_h, cos_h, 0.0])
        up = np.cross(right, fwd)  # ensure orthogonal

        # MuJoCo xmat: columns = [right, up, -forward]
        cam_xmat = np.column_stack([right, up, -fwd]).flatten()

        return (cam_xpos, cam_xmat)

    def get_odometry(self) -> Any:
        """Return latest Odometry data as a types.Odometry dataclass."""
        from vector_os_nano.core.types import Odometry
        pos = self._position
        return Odometry(
            timestamp=time.time(),
            x=pos[0], y=pos[1], z=pos[2],
            qx=0.0, qy=0.0, qz=0.0, qw=1.0,
            vx=0.0, vy=0.0, vz=0.0, vyaw=0.0,
        )

    @property
    def name(self) -> str:
        return "go2_ros2_proxy"

    @property
    def supports_lidar(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Motion interface (matches MuJoCoGo2 public API)
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Publish a Twist command on /cmd_vel_nav (non-blocking)."""
        if self._node is None:
            return
        from geometry_msgs.msg import Twist

        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(vyaw)
        self._cmd_pub.publish(msg)

    def walk(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration: float = 1.0,
    ) -> bool:
        """Walk at the given velocity for *duration* seconds, then stop.

        Returns True (mirrors the MuJoCoGo2 API which checks upright status).
        """
        self.set_velocity(vx, vy, vyaw)
        time.sleep(duration)
        self.set_velocity(0.0, 0.0, 0.0)
        return True

    def stand(self, duration: float = 1.0) -> None:
        """Stop motion and hold position for *duration* seconds."""
        self.set_velocity(0.0, 0.0, 0.0)
        time.sleep(duration)

    def sit(self, duration: float = 1.0) -> None:
        """Best-effort sit: stop motion (cannot command sit via ROS2 velocity)."""
        self.set_velocity(0.0, 0.0, 0.0)
        time.sleep(duration)

    def _scene_graph_hash(self) -> int:
        """Compute a lightweight hash of the current scene graph state.

        Combines rooms count, viewpoints count, objects count, and robot
        position rounded to 0.5 m grid so minor drift does not trigger
        a re-publish.  Returns 0 when no scene graph is available.
        """
        sg = self._scene_graph
        rooms_count = 0
        vp_count = 0
        obj_count = 0
        if sg is not None:
            try:
                rooms = sg.get_all_rooms()
                rooms_count = len(list(rooms))
                for room in sg.get_all_rooms():
                    vp_count += len(sg.get_viewpoints_in_room(room.room_id))
                    obj_count += len(sg.find_objects_in_room(room.room_id))
            except Exception:
                pass
        pos = self._position
        rx = round(pos[0] / 0.5)
        ry = round(pos[1] / 0.5)
        return hash((rooms_count, vp_count, obj_count, rx, ry))

    def _publish_markers(self) -> None:
        """Publish scene graph visualization as MarkerArray at 3 Hz.

        Records current position into trajectory history on every call.
        Caps trajectory at 200 entries to avoid unbounded memory growth.

        Only rebuilds and publishes the MarkerArray when the scene graph
        state hash changes, or every 10 seconds as a keep-alive fallback.
        This prevents unnecessary RViz re-renders that cause flickering.
        """
        if self._marker_pub is None:
            return
        try:
            from vector_os_nano.ros2.nodes.scene_graph_viz import (
                build_scene_graph_markers,
                _TRAJECTORY_MAX_POINTS,
            )
            pos = self._position
            # Always record trajectory (position history should be continuous)
            self._trajectory.append((pos[0], pos[1]))
            if len(self._trajectory) > _TRAJECTORY_MAX_POINTS:
                del self._trajectory[: len(self._trajectory) - _TRAJECTORY_MAX_POINTS]

            # Decide whether to publish: state changed OR 10 s fallback
            now = time.time()
            current_hash = self._scene_graph_hash()
            elapsed = now - self._last_marker_publish_time
            state_changed = current_hash != self._last_marker_hash
            fallback_due = elapsed >= 10.0

            if not (state_changed or fallback_due):
                return

            ma = build_scene_graph_markers(
                scene_graph=self._scene_graph,
                robot_x=pos[0],
                robot_y=pos[1],
                robot_heading=self._heading,
                nav_goal=self._nav_goal,
                trajectory=self._trajectory,
            )
            if ma is not None:
                self._marker_pub.publish(ma)
                self._last_marker_hash = current_hash
                self._last_marker_publish_time = now
        except Exception:
            pass
