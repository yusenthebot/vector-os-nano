# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Shared helpers for nav stack debug harness (L27-L29).

Provides mock objects and simulation utilities for testing navigation
behavior without a live ROS2 / MuJoCo environment.
"""
from __future__ import annotations

import importlib
import inspect
import math
import os
from dataclasses import dataclass, field
from typing import Any

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BRIDGE = os.path.join(_REPO, "scripts", "go2_vnav_bridge.py")


def read_bridge_source() -> str:
    with open(_BRIDGE) as f:
        return f.read()


def navigate_module():
    return importlib.import_module("vector_os_nano.skills.navigate")


def navigate_source() -> str:
    mod = navigate_module()
    return inspect.getsource(mod)


def proxy_source() -> str:
    mod = importlib.import_module("vector_os_nano.hardware.sim.go2_ros2_proxy")
    return inspect.getsource(mod)


def explore_source() -> str:
    mod = importlib.import_module("vector_os_nano.skills.go2.explore")
    return inspect.getsource(mod)


# ---------------------------------------------------------------------------
# Mock base for behavioral testing
# ---------------------------------------------------------------------------

@dataclass
class MockOdometry:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.28
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    vyaw: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0


@dataclass
class StuckSimulator:
    """Simulates the bridge stuck detector logic for isolated testing.

    Reproduces the exact algorithm from go2_vnav_bridge.py _stuck_detector().
    """
    stuck_threshold: float = 0.1   # meters — same as bridge
    check_interval: float = 2.0    # seconds — same as bridge

    # Internal state
    _pos: tuple[float, float] | None = None
    _count: int = 0
    _reset_waypoint_sent: int = 0
    _backup_triggered: int = 0
    _positions_at_reset: list = field(default_factory=list)

    def tick(self, x: float, y: float, nav_enabled: bool = True) -> str:
        """Run one stuck detector cycle. Returns action taken."""
        if not nav_enabled:
            self._count = 0
            self._pos = None
            return "disabled"

        if self._pos is not None:
            dx = x - self._pos[0]
            dy = y - self._pos[1]
            moved = math.sqrt(dx * dx + dy * dy)

            if moved < self.stuck_threshold:
                self._count += 1
            else:
                self._count = 0

            if self._count == 2:  # 4s
                self._reset_waypoint_sent += 1
                self._positions_at_reset.append((x, y))
                self._pos = (x, y)
                return "reset_waypoint"
            elif self._count == 4:  # 8s
                self._backup_triggered += 1
                self._count = 0
                self._pos = (x, y)
                return "backup"

        self._pos = (x, y)
        return "ok"

    def is_same_location_loop(self, tolerance: float = 0.5) -> bool:
        """Detect if /reset_waypoint keeps firing at the same spot.

        Returns True if the last 3+ resets were within `tolerance` meters
        of each other — indicating a stuck loop that escape can't break.
        """
        if len(self._positions_at_reset) < 3:
            return False
        recent = self._positions_at_reset[-3:]
        cx = sum(p[0] for p in recent) / 3
        cy = sum(p[1] for p in recent) / 3
        return all(
            math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) < tolerance
            for p in recent
        )


@dataclass
class ClearanceCalculator:
    """Compute real wall clearance from nav stack parameters.

    Reproduces the clearance math for localPlanner to verify adequacy.
    """
    search_radius: float = 0.45
    vehicle_width: float = 0.35
    vehicle_length: float = 0.45
    go2_body_half_width: float = 0.19  # actual body extent from center

    @property
    def half_width(self) -> float:
        return self.vehicle_width / 2.0

    @property
    def planning_clearance(self) -> float:
        """Clearance between planned path edge and obstacle."""
        return self.search_radius - self.half_width

    @property
    def real_clearance(self) -> float:
        """Actual gap between Go2 body and wall."""
        return self.search_radius - self.go2_body_half_width

    @property
    def diameter(self) -> float:
        """Vehicle diagonal used in localPlanner collision check."""
        return math.sqrt(
            (self.vehicle_length / 2) ** 2 + (self.vehicle_width / 2) ** 2
        )

    def fits_doorway(self, doorway_width: float) -> bool:
        """Can the robot pass through a doorway of given width?

        Requires real_clearance > 0 on both sides.
        """
        return self.vehicle_width < doorway_width and self.real_clearance > 0.05

    def min_doorway_width(self, min_clearance: float = 0.1) -> float:
        """Minimum doorway width for safe passage with given clearance."""
        return self.go2_body_half_width * 2 + min_clearance * 2


@dataclass
class MockBase:
    """Mock base for NavigateSkill behavioral testing."""
    x: float = 10.0
    y: float = 5.0
    z: float = 0.28
    heading: float = 0.0
    _walk_log: list = field(default_factory=list)
    _navigate_log: list = field(default_factory=list)
    has_navigate_to: bool = False

    def get_position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def get_heading(self) -> float:
        return self.heading

    def walk(self, vx: float, vy: float, vyaw: float, duration: float) -> None:
        self._walk_log.append((vx, vy, vyaw, duration))
        # Simulate movement
        self.x += vx * duration * math.cos(self.heading)
        self.y += vx * duration * math.sin(self.heading)
        self.heading += vyaw * duration

    def navigate_to(self, x: float, y: float, timeout: float = 45.0) -> bool:
        if not self.has_navigate_to:
            raise AttributeError("navigate_to not available")
        self._navigate_log.append((x, y, timeout))
        # Simulate arrival
        self.x, self.y = x, y
        return True

    @property
    def waypoints_visited(self) -> list[tuple[float, float]]:
        """Extract (x, y) positions from walk log for path analysis."""
        positions = [(self.x, self.y)]  # starting position
        return positions


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Room layout constants (from navigate.py)
ROOM_CENTERS = {
    "living_room":    (3.0,  2.5),
    "dining_room":    (3.0,  7.5),
    "kitchen":        (17.0, 2.5),
    "study":          (17.0, 7.5),
    "master_bedroom": (3.5,  12.0),
    "guest_bedroom":  (16.0, 12.0),
    "bathroom":       (8.5,  12.0),
    "hallway":        (10.0, 5.0),
}

ROOM_DOORS = {
    "living_room":    (6.5,  3.0),
    "dining_room":    (6.5,  8.0),
    "kitchen":        (13.5, 3.0),
    "study":          (13.5, 8.0),
    "master_bedroom": (3.0,  10.5),
    "guest_bedroom":  (12.0, 10.5),
    "bathroom":       (8.5,  10.5),
    "hallway":        (10.0, 5.0),
}

# Wall segments (approximate) for go2_room.xml collision checking.
# Each wall: ((x1, y1), (x2, y2))
# Walls have gaps at doorway positions (~1m wide openings).
WALLS = [
    # Outer walls
    ((0, 0), (20, 0)),
    ((20, 0), (20, 14)),
    ((20, 14), (0, 14)),
    ((0, 14), (0, 0)),
    # Left interior wall (x=6.5) with door gaps at y~3 and y~8
    ((6.5, 0), (6.5, 2.5)),     # below living_room door
    ((6.5, 3.5), (6.5, 7.5)),   # between doors
    ((6.5, 8.5), (6.5, 10)),    # above dining_room door
    # Right interior wall (x=13.5) with door gaps at y~3 and y~8
    ((13.5, 0), (13.5, 2.5)),   # below kitchen door
    ((13.5, 3.5), (13.5, 7.5)), # between doors
    ((13.5, 8.5), (13.5, 10)),  # above study door
    # Bedroom partition (y=10) with door gaps
    ((0, 10), (2.5, 10)),       # left of master_bedroom door
    ((3.5, 10), (8.0, 10)),     # between master and bathroom
    ((9.0, 10), (11.5, 10)),    # between bathroom and guest
    ((12.5, 10), (20, 10)),     # right of guest_bedroom door
]


def point_to_segment_distance(
    px: float, py: float,
    x1: float, y1: float, x2: float, y2: float,
) -> float:
    """Minimum distance from point (px,py) to line segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-10:
        return distance(px, py, x1, y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return distance(px, py, proj_x, proj_y)


def min_wall_distance(x: float, y: float) -> float:
    """Minimum distance from (x, y) to any wall segment."""
    return min(
        point_to_segment_distance(x, y, *w[0], *w[1])
        for w in WALLS
    )


def path_crosses_wall(
    x1: float, y1: float, x2: float, y2: float,
) -> bool:
    """Check if a straight line from (x1,y1) to (x2,y2) crosses any wall.

    Uses simple sampling along the line.
    """
    steps = max(10, int(distance(x1, y1, x2, y2) / 0.2))
    for i in range(steps + 1):
        t = i / steps
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        if min_wall_distance(px, py) < 0.05:  # within 5cm of wall
            return True
    return False
