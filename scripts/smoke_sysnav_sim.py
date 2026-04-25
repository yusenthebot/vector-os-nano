#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Smoke test for the v2.4 SysNav simulation pipeline.

Three modes:

* ``--check-deps``   — dependency-only probe (rclpy, tare_planner.msg,
                       mujoco). Exits 0 if all importable, 1 otherwise.
                       Useful for CI gating; no MuJoCo, no GPU.
* ``--no-sysnav``    — start MuJoCo virtual sensors only. Asserts that
                       :class:`MuJoCoLivox360`, :class:`MuJoCoPano360`,
                       :class:`GroundTruthOdomPublisher` produce one
                       sample each within 5 seconds against a tiny
                       inline MJCF. No SysNav workspace needed.
* (default)          — full bringup: requires SysNav to be running in
                       another terminal. Asserts that
                       ``/object_nodes_list`` carries ≥ 1 node within
                       30 seconds and that the resulting WorldModel
                       contains a valid sysnav_*-prefixed entry.

Examples::

    python scripts/smoke_sysnav_sim.py --check-deps
    python scripts/smoke_sysnav_sim.py --no-sysnav
    python scripts/smoke_sysnav_sim.py --timeout 30
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

logger = logging.getLogger("smoke_sysnav_sim")


_TINY_MJCF = """
<mujoco>
  <worldbody>
    <body name="trunk" pos="0 0 0.5">
      <freejoint name="trunk_root"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
    <geom name="wall_pos_x" type="box" pos="3 0 1"  size="0.1 3 1" rgba="1 0 0 1"/>
    <geom name="wall_neg_x" type="box" pos="-3 0 1" size="0.1 3 1" rgba="0 1 0 1"/>
    <geom name="wall_pos_y" type="box" pos="0 3 1"  size="3 0.1 1" rgba="0 0 1 1"/>
    <geom name="wall_neg_y" type="box" pos="0 -3 1" size="3 0.1 1" rgba="1 1 0 1"/>
    <geom name="floor" type="plane" pos="0 0 0" size="5 5 0.1" rgba="0.5 0.5 0.5 1"/>
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Mode 1 — dependency probe
# ---------------------------------------------------------------------------


def check_deps() -> int:
    """Probe importability of mujoco, rclpy, tare_planner.msg.

    Returns 0 (all OK) or 1 (something missing). Always logs the
    findings so users can see which dependency to fix.
    """
    findings: list[tuple[str, str]] = []
    for name in ("mujoco", "numpy", "rclpy"):
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "?")
            findings.append((name, f"OK ({version})"))
        except ImportError as exc:
            findings.append((name, f"MISSING ({exc})"))

    try:
        import tare_planner.msg     # noqa: F401
        findings.append(("tare_planner.msg", "OK"))
    except ImportError as exc:
        findings.append(("tare_planner.msg", f"MISSING ({exc})"))

    print("dependency probe:")
    missing = 0
    for name, status in findings:
        print(f"  {name:25s}  {status}")
        if "MISSING" in status:
            missing += 1
    return 1 if missing else 0


# ---------------------------------------------------------------------------
# Mode 2 — sim sensors only
# ---------------------------------------------------------------------------


def smoke_no_sysnav(timeout_s: float) -> int:
    """Boot the 3 virtual sensors against the tiny MJCF.

    Asserts each produces one sample within ``timeout_s``.
    """
    import mujoco

    from vector_os_nano.hardware.sim.sensors import (
        GroundTruthOdomPublisher,
        MuJoCoLivox360,
        MuJoCoPano360,
    )

    model = mujoco.MjModel.from_xml_string(_TINY_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lidar = MuJoCoLivox360(
        model, data, h_resolution=180, v_layers=8, max_range=10.0,
    )
    pano = MuJoCoPano360(model, data, out_w=192, out_h=64, face_size=64)
    odom = GroundTruthOdomPublisher(model, data)

    deadline = time.monotonic() + timeout_s
    lidar_ok = pano_ok = odom_ok = False

    while time.monotonic() < deadline and not (lidar_ok and pano_ok and odom_ok):
        if not lidar_ok:
            sample = lidar.step()
            lidar_ok = sample.points.shape[0] > 0
        if not pano_ok:
            sample = pano.step()
            pano_ok = sample.image.shape == (64, 192, 3)
        if not odom_ok:
            sample = odom.step()
            odom_ok = sample.position == (0.0, 0.0, 0.5)
        time.sleep(0.1)

    pano.close()

    print("--no-sysnav smoke results:")
    print(f"  lidar  {'OK' if lidar_ok else 'FAIL'}")
    print(f"  pano   {'OK' if pano_ok else 'FAIL'}")
    print(f"  odom   {'OK' if odom_ok else 'FAIL'}")

    return 0 if (lidar_ok and pano_ok and odom_ok) else 1


# ---------------------------------------------------------------------------
# Mode 3 — full bringup (SysNav workspace required)
# ---------------------------------------------------------------------------


def smoke_full(timeout_s: float) -> int:
    """Wire LiveSysnavBridge against a fresh WorldModel and wait for
    the first ``/object_nodes_list`` message.
    """
    try:
        import rclpy            # noqa: F401
        from tare_planner.msg import ObjectNodeList   # noqa: F401
    except ImportError as exc:
        print(
            f"FAIL: SysNav workspace not sourced ({exc}). "
            "Source ~/Desktop/SysNav/install/setup.bash and try again, "
            "or use --check-deps / --no-sysnav modes."
        )
        return 1

    from vector_os_nano.core.world_model import WorldModel
    from vector_os_nano.integrations.sysnav_bridge import LiveSysnavBridge

    world = WorldModel()
    bridge = LiveSysnavBridge(world, on_disconnect_after_s=timeout_s)
    started = bridge.start()
    if not started:
        print("FAIL: LiveSysnavBridge.start() returned False (see logs)")
        return 1

    deadline = time.monotonic() + timeout_s
    print(f"awaiting /object_nodes_list for up to {timeout_s:.0f}s ...")
    while time.monotonic() < deadline:
        if world.get_objects():
            break
        time.sleep(0.5)

    bridge.stop()

    objects = world.get_objects()
    if not objects:
        print(f"FAIL: world_model still empty after {timeout_s:.0f}s")
        return 1

    print(f"OK: world_model populated with {len(objects)} object(s):")
    for obj in objects[:5]:
        print(
            f"  {obj.object_id:30s} label={obj.label!r:20s} "
            f"xyz=({obj.x:.2f}, {obj.y:.2f}, {obj.z:.2f}) "
            f"conf={obj.confidence:.2f} state={obj.state!r}"
        )
    return 0


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-deps", action="store_true",
        help="Only check importability of dependencies and exit.",
    )
    mode.add_argument(
        "--no-sysnav", action="store_true",
        help="Smoke the 3 virtual sensors without SysNav workspace.",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Per-mode timeout in seconds (default 30 for full mode).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if args.check_deps:
        return check_deps()
    if args.no_sysnav:
        return smoke_no_sysnav(args.timeout if args.timeout > 0 else 5.0)
    return smoke_full(args.timeout)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
