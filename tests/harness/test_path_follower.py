# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for gait-stable quadruped path follower.

The Go2 MPC gait is stable when:
- Forward-biased (minimal reverse)
- Gentle turns (< 0.6 rad/s)
- Smooth transitions (rate-limited)
- Stops near walls rather than reversing
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest


def _compute_velocity(
    path: List[Tuple[float, float]],
    robot_x: float, robot_y: float, heading: float,
    obs_front: float = 2.0, obs_back: float = 2.0,
    obs_left: float = 2.0, obs_right: float = 2.0,
) -> Tuple[float, float, float]:
    """Replicate the core path follower logic. Returns (vx, vy, vyaw)."""
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    best_idx = 0
    best_dsq = float("inf")
    for i, (px, py) in enumerate(path):
        dsq = (px - robot_x)**2 + (py - robot_y)**2
        if dsq < best_dsq:
            best_dsq = dsq
            best_idx = i

    arc = 0.0
    tidx = best_idx
    for i in range(best_idx, len(path) - 1):
        sdx = path[i+1][0] - path[i][0]
        sdy = path[i+1][1] - path[i][1]
        arc += math.sqrt(sdx*sdx + sdy*sdy)
        tidx = i + 1
        if arc >= 0.5:
            break

    tx, ty = path[tidx]
    dx, dy = tx - robot_x, ty - robot_y
    dist = math.sqrt(dx*dx + dy*dy)

    desired = math.atan2(dy, dx)
    err = desired - heading
    while err > math.pi: err -= 2*math.pi
    while err < -math.pi: err += 2*math.pi
    abs_err = abs(err)

    if abs_err < 0.6:
        vx = float(np.clip(0.5 * math.cos(err), 0.1, 0.5))
        vy = float(np.clip(0.1 * math.sin(-err), -0.1, 0.1))
        vyaw = float(np.clip(err * 0.6, -0.5, 0.5))
    elif abs_err < 2.0:
        vx = 0.0
        vy = float(np.clip(0.08 * math.sin(-err), -0.08, 0.08))
        vyaw = float(np.clip(err * 0.8, -0.6, 0.6))
    else:
        vx = 0.0
        vy = 0.0
        vyaw = float(np.clip(err * 0.8, -0.6, 0.6))

    if dist < 0.3:
        s = max(0.3, dist / 0.3)
        vx *= s
        vy *= s

    _WALL_STOP = 0.35
    _WALL_SLOW = 0.60
    _WALL_PUSH = 0.45

    if obs_front < _WALL_SLOW:
        frac = max(0.0, (obs_front - _WALL_STOP) / (_WALL_SLOW - _WALL_STOP))
        vx = min(vx, 0.3 * frac)

    if obs_left < _WALL_PUSH:
        vy -= 0.08 * (1.0 - obs_left / _WALL_PUSH)
    if obs_right < _WALL_PUSH:
        vy += 0.08 * (1.0 - obs_right / _WALL_PUSH)

    vx = float(np.clip(vx, -0.05, 0.5))
    vy = float(np.clip(vy, -0.1, 0.1))
    vyaw = float(np.clip(vyaw, -0.6, 0.6))

    return vx, vy, vyaw


class TestGaitStability:
    """Velocity commands must be gentle enough for MPC gait."""

    def test_no_aggressive_reverse(self):
        """vx should never go below -0.05 (almost no backing up)."""
        for path in [[(-2, 0)], [(0, 2)], [(-1, -1)]]:
            vx, _, _ = _compute_velocity(path, 0, 0, 0)
            assert vx >= -0.05, f"vx={vx:.3f} too aggressive reverse"

    def test_gentle_yaw(self):
        """vyaw should stay within ±0.6 rad/s."""
        for path in [[(0, 2)], [(0, -2)], [(-2, 0)]]:
            _, _, vyaw = _compute_velocity(path, 0, 0, 0)
            assert -0.6 <= vyaw <= 0.6, f"vyaw={vyaw:.3f} too aggressive"

    def test_gentle_strafe(self):
        """vy should stay within ±0.1 m/s."""
        for obs_l, obs_r in [(0.2, 2.0), (2.0, 0.2), (0.3, 0.3)]:
            _, vy, _ = _compute_velocity([(2, 0)], 0, 0, 0, obs_left=obs_l, obs_right=obs_r)
            assert -0.1 <= vy <= 0.1, f"vy={vy:.3f} too aggressive"


class TestForwardMotion:
    def test_target_ahead_walks_forward(self):
        vx, _, _ = _compute_velocity([(2, 0)], 0, 0, 0)
        assert vx > 0.2

    def test_target_ahead_small_yaw(self):
        _, _, vyaw = _compute_velocity([(2, 0)], 0, 0, 0)
        assert abs(vyaw) < 0.1


class TestTurning:
    def test_target_90deg_stops_and_turns(self):
        """Target to the side: stop forward, turn in place."""
        vx, _, vyaw = _compute_velocity([(0, 2)], 0, 0, 0)
        assert vx <= 0.05, f"Should stop forward, got vx={vx:.3f}"
        assert vyaw > 0.3, f"Should turn left, got vyaw={vyaw:.3f}"

    def test_target_behind_stops_and_turns(self):
        """Target behind: stop, turn (no reverse)."""
        vx, _, vyaw = _compute_velocity([(-2, 0)], 0, 0, 0)
        assert vx >= -0.05, "Should not reverse aggressively"
        assert abs(vyaw) > 0.3, "Should turn"


class TestWallAvoidance:
    def test_wall_ahead_stops(self):
        vx, _, _ = _compute_velocity([(2, 0)], 0, 0, 0, obs_front=0.30)
        assert vx <= 0.0, f"Should stop with wall at 0.3m, got vx={vx:.3f}"

    def test_wall_left_nudges_right(self):
        _, vy, _ = _compute_velocity([(2, 0)], 0, 0, 0, obs_left=0.30)
        assert vy < 0.0, f"Should nudge right, got vy={vy:.3f}"

    def test_wall_right_nudges_left(self):
        _, vy, _ = _compute_velocity([(2, 0)], 0, 0, 0, obs_right=0.30)
        assert vy > 0.0, f"Should nudge left, got vy={vy:.3f}"

    def test_no_wall_no_strafe(self):
        _, vy, _ = _compute_velocity([(2, 0)], 0, 0, 0)
        assert abs(vy) < 0.05, "No strafe without walls"
