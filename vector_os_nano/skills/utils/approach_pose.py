# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Approach pose utility for mobile manipulation.

Given an object's world XYZ and the dog's current world pose, compute an
(x, y, yaw) approach pose so the dog stops `clearance` metres from the
object and faces it.
"""
import math


def compute_approach_pose(
    object_xyz: tuple[float, float, float],
    dog_pose: tuple[float, float, float],  # (x, y, yaw) in world frame
    clearance: float = 0.55,  # metres
    approach_direction: str | None = None,  # "from_dog" | "from_normal" | None
) -> tuple[float, float, float]:  # (approach_x, approach_y, approach_yaw) world frame
    """Compute a pose where the dog stops to grasp/place at the object.

    Algorithm for "from_dog" (default):
      1. v = unit vector from object to dog (in world XY plane)
      2. approach_xy = object_xy + clearance * v
      3. approach_yaw = atan2(obj_y - approach_y, obj_x - approach_x)
         (robot at approach_xy faces the object)

    Args:
        object_xyz: Object position in world frame (x, y, z). Z is ignored.
        dog_pose: Dog's current pose (x, y, yaw) in world frame.
        clearance: Distance from object centre to approach position, metres.
        approach_direction: Direction strategy.
            - None or "from_dog": approach from the dog's current side.
            - "from_normal": reserved for v2.3 (surface normal approach).

    Returns:
        (approach_x, approach_y, approach_yaw) in world frame.

    Raises:
        ValueError: Dog XY is identical to object XY (distance < 1e-6).
        NotImplementedError: approach_direction == "from_normal".
    """
    if approach_direction == "from_normal":
        raise NotImplementedError(
            "approach_direction='from_normal' is reserved for v2.3 and not yet implemented."
        )

    obj_x: float = object_xyz[0]
    obj_y: float = object_xyz[1]
    dog_x: float = dog_pose[0]
    dog_y: float = dog_pose[1]

    dx: float = dog_x - obj_x
    dy: float = dog_y - obj_y
    dist: float = math.sqrt(dx * dx + dy * dy)

    if dist < 1e-6:
        raise ValueError(
            f"Dog XY ({dog_x}, {dog_y}) is too close to object XY ({obj_x}, {obj_y}): "
            f"distance {dist:.2e} < 1e-6. Cannot determine approach direction."
        )

    # Unit vector from object toward dog
    ux: float = dx / dist
    uy: float = dy / dist

    # Approach position: clearance metres from object along the dog-side direction
    approach_x: float = obj_x + clearance * ux
    approach_y: float = obj_y + clearance * uy

    # Yaw at approach position must face the object
    approach_yaw: float = math.atan2(obj_y - approach_y, obj_x - approach_x)

    return (approach_x, approach_y, approach_yaw)
