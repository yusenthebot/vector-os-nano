#!/usr/bin/env python3
"""End-to-end verification for MobilePickSkill and MobilePlaceSkill.

Spawns the full ROS2 bridge subprocess (VECTOR_SIM_WITH_ARM=1 + launch_explore.sh),
connects main-process proxies, populates world_model from MJCF, then calls
MobilePickSkill (and optionally MobilePlaceSkill) for each target object.

One fresh bridge subprocess per attempt — avoids MuJoCo multi-instance issues
and gives each attempt a clean physics state.

Usage:
    .venv-nano/bin/python scripts/verify_loco_pick_place.py --dry-run
    .venv-nano/bin/python scripts/verify_loco_pick_place.py --repeat 3
    .venv-nano/bin/python scripts/verify_loco_pick_place.py \\
        --mode pick_and_place --objects "blue bottle,green bottle" --repeat 1

Exit code 0 when every attempt passes, 1 if any fail.

NOTE: Full E2E requires the complete sim environment (ROS2 Jazzy, MuJoCo,
      launch_explore.sh). Use --dry-run for CI import validation only.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_OBJECTS: list[str] = ["blue bottle", "green bottle", "red can"]
_DEFAULT_REPEAT: int = 3
_DEFAULT_MODE: str = "pick_only"

# Bridge ready-wait: poll for /state_estimation and /piper/joint_state topics.
_BRIDGE_READY_TIMEOUT: float = 30.0
_BRIDGE_POLL_INTERVAL: float = 1.0

# After bridge is up, let the dog's physics settle before issuing skills.
_DOG_SETTLE_SLEEP: float = 2.0

# Cap on how long we wait for a subprocess to die after SIGTERM.
_KILL_WAIT_TIMEOUT: float = 5.0

# Place target (world XYZ) used in pick_and_place mode.
_PLACE_TARGET_XYZ: tuple[float, float, float] = (10.0, 3.0, 0.25)

# Repo root (two levels up from this script).
_REPO: Path = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Dry-run: import validation only (CI smoke test)
# ---------------------------------------------------------------------------


def _dry_run(args: argparse.Namespace) -> int:
    """Validate imports without spawning subprocess. Exit 0 on success."""
    print("[dry-run] verify_loco_pick_place — import validation")
    print(f"  mode:    {args.mode}")
    print(f"  objects: {', '.join(args.objects)}")
    print(f"  repeat:  {args.repeat}")
    print()

    # Ensure the venv site-packages are on sys.path so imports resolve.
    venv_site = _REPO / ".venv-nano" / "lib"
    if venv_site.exists():
        for pkg in sorted(venv_site.iterdir()):
            site_pkg = pkg / "site-packages"
            if site_pkg.is_dir() and str(site_pkg) not in sys.path:
                sys.path.insert(0, str(site_pkg))
    # Also ensure the repo root is importable.
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

    errors: list[str] = []

    imports = [
        ("MobilePickSkill", "vector_os_nano.skills.mobile_pick", "MobilePickSkill"),
        ("MobilePlaceSkill", "vector_os_nano.skills.mobile_place", "MobilePlaceSkill"),
        ("Go2ROS2Proxy", "vector_os_nano.hardware.sim.go2_ros2_proxy", "Go2ROS2Proxy"),
        ("PiperROS2Proxy", "vector_os_nano.hardware.sim.piper_ros2_proxy", "PiperROS2Proxy"),
        ("PiperGripperROS2Proxy", "vector_os_nano.hardware.sim.piper_ros2_proxy", "PiperGripperROS2Proxy"),
    ]

    for label, module_path, attr in imports:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, attr)
            print(f"  [OK] {label} from {module_path}")
        except Exception as exc:
            errors.append(f"  [FAIL] {label}: {exc}")
            print(errors[-1])

    # Instantiate skills (no hardware needed — skill __init__ is lightweight).
    skill_checks = [
        ("MobilePickSkill()", "vector_os_nano.skills.mobile_pick", "MobilePickSkill"),
        ("MobilePlaceSkill()", "vector_os_nano.skills.mobile_place", "MobilePlaceSkill"),
    ]
    for label, module_path, attr in skill_checks:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, attr)
            instance = cls()
            print(f"  [OK] {label} instantiated (name={instance.name!r})")
        except Exception as exc:
            errors.append(f"  [FAIL] {label}: {exc}")
            print(errors[-1])

    if errors:
        print(f"\n[dry-run] FAIL — {len(errors)} import error(s)")
        return 1

    print("\n[dry-run] PASS — all imports OK")
    return 0


# ---------------------------------------------------------------------------
# Bridge subprocess management
# ---------------------------------------------------------------------------


def _spawn_bridge(log_path: str) -> subprocess.Popen:
    """Launch launch_explore.sh with VECTOR_SIM_WITH_ARM=1.

    Uses preexec_fn=os.setsid so killpg() terminates the entire process group.
    Stderr and stdout are redirected to *log_path* for post-mortem inspection.
    """
    script = str(_REPO / "scripts" / "launch_explore.sh")
    child_env = os.environ.copy()
    child_env["VECTOR_SIM_WITH_ARM"] = "1"

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        ["bash", script, "--no-gui"],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=child_env,
        cwd=str(_REPO),
    )
    # Stash the file handle on proc so teardown can close it.
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def _wait_bridge_ready(timeout: float) -> bool:
    """Poll rclpy for /state_estimation and /piper/joint_state topics.

    Returns True when both topics appear, False on timeout.
    Imports rclpy lazily to avoid hard dep at module load.
    """
    try:
        import rclpy  # noqa: PLC0415
        from rclpy.node import Node  # noqa: PLC0415
    except ImportError:
        # rclpy not available — best-effort: just sleep and hope.
        time.sleep(timeout * 0.5)
        return True

    # Init rclpy if not already initialised (may be called from bare Python).
    if not rclpy.ok():
        rclpy.init()

    # Temporary node to query topic list.
    probe = Node("_verify_loco_probe")
    deadline = time.monotonic() + timeout
    ready = False

    try:
        while time.monotonic() < deadline:
            topics = {n for n, _ in probe.get_topic_names_and_types()}
            has_state = "/state_estimation" in topics
            has_piper = "/piper/joint_state" in topics
            if has_state and has_piper:
                ready = True
                break
            time.sleep(_BRIDGE_POLL_INTERVAL)
    finally:
        probe.destroy_node()

    return ready


def _teardown_bridge(proc: subprocess.Popen) -> None:
    """Send SIGTERM to the bridge process group; wait up to _KILL_WAIT_TIMEOUT."""
    if proc.poll() is not None:
        # Already dead.
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        proc.wait(timeout=_KILL_WAIT_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass

    # Close the log file handle that was stashed on the proc object.
    log_fh = getattr(proc, "_log_fh", None)
    if log_fh is not None:
        try:
            log_fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Proxy + world_model helpers
# ---------------------------------------------------------------------------


def _connect_proxies():
    """Construct and connect Go2ROS2Proxy, PiperROS2Proxy, PiperGripperROS2Proxy.

    Returns (base, arm, gripper) or raises on failure.
    """
    from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy  # noqa: PLC0415
    from vector_os_nano.hardware.sim.mujoco_go2 import _build_room_scene_xml  # noqa: PLC0415
    from vector_os_nano.hardware.sim.piper_ros2_proxy import (  # noqa: PLC0415
        PiperGripperROS2Proxy,
        PiperROS2Proxy,
    )

    base = Go2ROS2Proxy()
    base.connect()

    scene_xml = str(_build_room_scene_xml(with_arm=True))
    arm = PiperROS2Proxy(base_proxy=base, scene_xml_path=scene_xml)
    arm.connect()

    gripper = PiperGripperROS2Proxy()
    gripper.connect()

    return base, arm, gripper


def _disconnect_proxies(base, arm, gripper) -> None:
    """Disconnect proxies, swallowing all errors."""
    for proxy in (gripper, arm, base):
        if proxy is None:
            continue
        try:
            proxy.disconnect()
        except Exception:
            pass


def _build_world_model():
    """Build and populate WorldModel from the with_arm MJCF."""
    from vector_os_nano.core.world_model import WorldModel  # noqa: PLC0415
    from vector_os_nano.hardware.sim.mujoco_go2 import _build_room_scene_xml  # noqa: PLC0415
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool  # noqa: PLC0415

    scene_xml = str(_build_room_scene_xml(with_arm=True))
    wm = WorldModel()
    n = SimStartTool._populate_pickables_from_mjcf(wm, scene_xml)
    return wm, n


# ---------------------------------------------------------------------------
# Single-attempt runner
# ---------------------------------------------------------------------------


def _run_one_attempt(
    object_label: str,
    mode: str,
    attempt_num: int,
    total_attempts: int,
) -> tuple[bool, float, str]:
    """Run one pick (and optionally place) attempt.

    Returns (passed, duration_seconds, detail_message).

    Spawns a fresh bridge subprocess, connects proxies, calls the skill,
    then tears down regardless of outcome.
    """
    import importlib  # noqa: PLC0415

    log_path = f"/tmp/verify_loco_pick_place_{os.getpid()}_{attempt_num}.log"
    bridge: subprocess.Popen | None = None
    base = arm = gripper = None
    t0 = time.monotonic()

    try:
        # 1. Spawn bridge
        bridge = _spawn_bridge(log_path)

        # 2. Wait for topics
        ready = _wait_bridge_ready(_BRIDGE_READY_TIMEOUT)
        if not ready:
            return (
                False,
                time.monotonic() - t0,
                f"Bridge not ready within {_BRIDGE_READY_TIMEOUT}s (log: {log_path})",
            )

        # 3. Connect proxies
        base, arm, gripper = _connect_proxies()

        # 4. Populate world_model
        wm, obj_count = _build_world_model()

        # 5. Let dog settle
        time.sleep(_DOG_SETTLE_SLEEP)

        # 6. Build SkillContext
        from vector_os_nano.core.skill import SkillContext  # noqa: PLC0415

        context = SkillContext(
            base=base,
            arm=arm,
            gripper=gripper,
            world_model=wm,
            config={},
        )

        # 7. Execute MobilePickSkill
        mobile_pick_mod = importlib.import_module("vector_os_nano.skills.mobile_pick")
        mobile_pick = mobile_pick_mod.MobilePickSkill()
        pick_result = mobile_pick.execute({"object_label": object_label}, context)

        if not pick_result.success:
            duration = time.monotonic() - t0
            diag = pick_result.result_data.get("diagnosis", "unknown")
            return (
                False,
                duration,
                f"pick FAIL label={object_label!r} diag={diag} msg={pick_result.error_message!r}",
            )

        grasped = bool(pick_result.result_data.get("grasped_heuristic", False))
        if not grasped:
            duration = time.monotonic() - t0
            return (
                False,
                duration,
                f"pick FAIL label={object_label!r} grasped_heuristic=False",
            )

        # 8. pick_and_place mode: also execute MobilePlaceSkill
        if mode == "pick_and_place":
            mobile_place_mod = importlib.import_module("vector_os_nano.skills.mobile_place")
            mobile_place = mobile_place_mod.MobilePlaceSkill()
            place_params = {
                "target_xyz": list(_PLACE_TARGET_XYZ),
            }
            place_result = mobile_place.execute(place_params, context)

            if not place_result.success:
                duration = time.monotonic() - t0
                diag = place_result.result_data.get("diagnosis", "unknown")
                return (
                    False,
                    duration,
                    (
                        f"place FAIL label={object_label!r} diag={diag} "
                        f"msg={place_result.error_message!r}"
                    ),
                )

            # Relaxed assertion: gripper should be open after a successful place.
            holding = False
            try:
                holding = bool(gripper.is_holding())
            except Exception:
                pass
            if holding:
                duration = time.monotonic() - t0
                return (
                    False,
                    duration,
                    f"place FAIL label={object_label!r} gripper still holding after place",
                )

        duration = time.monotonic() - t0
        return (
            True,
            duration,
            f"PASS label={object_label!r} grasped=True mode={mode}",
        )

    except KeyboardInterrupt:
        raise
    except Exception as exc:
        duration = time.monotonic() - t0
        return (False, duration, f"EXCEPTION label={object_label!r} {type(exc).__name__}: {exc}")
    finally:
        _disconnect_proxies(base, arm, gripper)
        if bridge is not None:
            _teardown_bridge(bridge)
        # Brief gap between attempts so OS can clean up ports/sockets.
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=_DEFAULT_REPEAT,
        metavar="N",
        help="Number of attempts per object (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=["pick_only", "pick_and_place"],
        default=_DEFAULT_MODE,
        help="What flow to run (default: %(default)s)",
    )
    parser.add_argument(
        "--objects",
        type=lambda s: [o.strip() for o in s.split(",") if o.strip()],
        default=_DEFAULT_OBJECTS,
        metavar="LIST",
        help=(
            "Comma-separated labels to cycle through "
            "(default: 'blue bottle,green bottle,red can')"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exit 0 without spawning subprocess; validate imports only (CI smoke)",
    )
    args = parser.parse_args()

    if args.dry_run:
        return _dry_run(args)

    objects: list[str] = args.objects
    repeat: int = args.repeat
    mode: str = args.mode

    total = len(objects) * repeat
    print("\n===== verify_loco_pick_place =====")
    print(f"Mode:    {mode}")
    print(f"Objects: {', '.join(objects)}")
    print(f"N/obj:   {repeat}")
    print(f"Total:   {total} attempts")
    print()

    results: list[tuple[int, str, bool, float, str]] = []
    seq = 0

    try:
        for obj_label in objects:
            for rep in range(1, repeat + 1):
                seq += 1
                print(f"[{seq}/{total}] {obj_label!r} attempt {rep} ...", flush=True)
                passed, dur, detail = _run_one_attempt(
                    object_label=obj_label,
                    mode=mode,
                    attempt_num=seq,
                    total_attempts=total,
                )
                verdict = "PASS" if passed else "FAIL"
                print(f"[{seq}/{total}] {verdict} ({dur:.1f}s) — {detail}")
                results.append((seq, obj_label, passed, dur, detail))
    except KeyboardInterrupt:
        print("\n[interrupted] Ctrl-C received — partial results below.")

    # Summary
    n_pass = sum(1 for _, _, ok, _, _ in results if ok)
    n_fail = len(results) - n_pass
    avg_dur = sum(d for _, _, _, d, _ in results) / len(results) if results else 0.0
    pct = 100 * n_pass // len(results) if results else 0

    print()
    print("===== verify_loco_pick_place results =====")
    print(f"Mode:           {mode}")
    print(f"Objects:        {', '.join(objects)}")
    print(f"N per object:   {repeat}")
    print(f"Total:          {len(results)} attempts")
    print(f"PASS:           {n_pass} ({pct}%)")
    print(f"FAIL:           {n_fail}")
    print(f"Avg duration:   {avg_dur:.1f} s")
    print()
    print("Details:")
    for seq_i, label, ok, dur, detail in results:
        verdict = "PASS" if ok else "FAIL"
        print(f"  [{seq_i}/{total}] {label!r} attempt {((seq_i - 1) % repeat) + 1}: {verdict} ({dur:.1f}s)")

    return 0 if n_fail == 0 and len(results) == total else 1


if __name__ == "__main__":
    sys.exit(main())
