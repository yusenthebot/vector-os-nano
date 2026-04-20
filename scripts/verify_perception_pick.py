#!/usr/bin/env python3
"""End-to-end verification for v2.3 Go2 perception pipeline (dry-run).

Exercises the full chain:
  DetectSkill.execute(query="blue bottle")
    -> QwenVLMDetector.detect(rgb) [MOCKED]
    -> Go2Perception.track(detections) -> Pose3D (cam frame)
    -> Go2Calibration.camera_to_base -> world xyz
    -> world_model.add_object(ObjectState)
  MobilePickSkill.execute(object_label="蓝色瓶子")  # empty world_model
    -> first _resolve_target miss
    -> auto-detect via context.perception (reuses DetectSkill)
    -> retry _resolve_target -> hit
    -> fake navigate_to + wait_stable + PickTopDownSkill (STUB succeeds)
  Exit 0 if every step produced expected state.

Run:
    python3 scripts/verify_perception_pick.py --dry-run
    python3 scripts/verify_perception_pick.py            # same; dry is default
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Ensure repo root is importable.
_REPO: Path = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Synthetic frames
# ---------------------------------------------------------------------------


def _make_synthetic_frames():
    """Return (rgb, depth) numpy arrays with a blue object at bbox (120,90,160,130)."""
    import numpy as np

    H, W = 240, 320
    rgb = np.full((H, W, 3), 80, dtype=np.uint8)   # grey background
    # Paint a blue rectangle where the bbox lands
    rgb[90:130, 120:160] = [30, 30, 200]

    depth = np.full((H, W), 3.0, dtype=np.float32)  # far background
    depth[90:130, 120:160] = 1.2                      # object at 1.2 m

    return rgb, depth


# ---------------------------------------------------------------------------
# FakeBase — duck-types Go2ROS2Proxy
# ---------------------------------------------------------------------------


class FakeBase:
    """Minimal Go2ROS2Proxy stand-in for dry-run."""

    def __init__(self, rgb, depth):
        import numpy as np

        self._rgb = rgb
        self._depth = depth
        self._pos = (0.0, 0.0, 0.28)
        self._yaw = 0.0
        self.nav_calls: list[tuple[float, float]] = []
        self._np = np

    def get_position(self):
        return self._pos

    def get_heading(self):
        return self._yaw

    def get_camera_frame(self, width: int = 320, height: int = 240):
        return self._rgb.copy()

    def get_depth_frame(self, width: int = 320, height: int = 240):
        return self._depth.copy()

    def get_camera_pose(self):
        """Dog at origin, MJCF-grounded mount (0.25 m fwd, 0.1 m up, -5 deg pitch).

        Uses ``up = cross(forward, right)`` — matches Go2ROS2Proxy.get_camera_pose
        after the v2.3 sign-convention fix.
        """
        import numpy as np

        pitch = math.radians(-5.0)
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)

        # MJCF d435_camera pos="0.25 0 0.1" on trunk (dog base z ≈ 0.28) → cam z ≈ 0.38
        pos = np.array([0.25, 0.0, 0.38])

        right = np.array([0.0, 1.0, 0.0])
        fwd = np.array([cos_p, 0.0, sin_p])
        up = np.cross(fwd, right)  # points world +Z for level +X-facing camera

        # MuJoCo xmat: columns = [right, up, -forward]
        xmat = np.column_stack([right, up, -fwd])

        return pos, xmat

    def navigate_to(self, x: float, y: float, timeout: float = 20.0) -> bool:
        self.nav_calls.append((x, y))
        self._pos = (x, y, 0.28)
        return True


# ---------------------------------------------------------------------------
# Monkey-patch Qwen so no HTTP is attempted
# ---------------------------------------------------------------------------


def _monkey_patch_qwen() -> None:
    """Replace QwenVLMDetector.detect with a deterministic fake."""
    from vector_os_nano.core.types import Detection
    from vector_os_nano.perception import vlm_qwen

    def _fake_detect(self, image, query: str) -> list[Detection]:  # noqa: ARG001
        return [Detection(label=query, bbox=(120.0, 90.0, 160.0, 130.0), confidence=0.91)]

    vlm_qwen.QwenVLMDetector.detect = _fake_detect  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Monkey-patch _wait_stable so the dry-run doesn't block 5 s
# ---------------------------------------------------------------------------


def _monkey_patch_wait_stable() -> None:
    import vector_os_nano.skills.mobile_pick as mp

    mp._wait_stable = lambda *a, **k: True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Main dry-run logic
# ---------------------------------------------------------------------------


def run_dry_run() -> int:
    import os

    # Prevent QwenVLMDetector.__init__ from complaining about missing key.
    os.environ.setdefault("OPENROUTER_API_KEY", "dry-run-no-network")

    # 1. Monkey-patches before any instantiation
    _monkey_patch_qwen()
    _monkey_patch_wait_stable()

    print("[dry-run] verify_perception_pick — v2.3 perception->pick chain")
    print()

    # 2. Synthetic frames + FakeBase
    rgb, depth = _make_synthetic_frames()
    base = FakeBase(rgb, depth)

    # 3. Wire real perception + calibration over FakeBase
    from vector_os_nano.perception.go2_calibration import Go2Calibration
    from vector_os_nano.perception.go2_perception import Go2Perception
    from vector_os_nano.perception.vlm_qwen import QwenVLMDetector

    qwen = QwenVLMDetector(config={"api_key": "dry-run"})
    perception = Go2Perception(camera=base, vlm=qwen)
    calibration = Go2Calibration(base_proxy=base)

    # 4. Fake arm + gripper (MagicMock-like; PickTopDownSkill.execute stubbed below)
    from unittest.mock import MagicMock

    arm = MagicMock()
    gripper = MagicMock()
    gripper.is_holding = False

    # 5. Build WorldModel + SkillContext
    from vector_os_nano.core.skill import SkillContext
    from vector_os_nano.core.world_model import WorldModel

    wm = WorldModel()
    ctx = SkillContext(
        base=base,
        arm=arm,
        gripper=gripper,
        perception=perception,
        calibration=calibration,
        world_model=wm,
    )

    # ------------------------------------------------------------------
    # Step 1 — DetectSkill directly
    # ------------------------------------------------------------------
    from vector_os_nano.skills.detect import DetectSkill

    det_res = DetectSkill().execute({"query": "blue bottle"}, ctx)
    if not det_res.success:
        print(f"FAIL [1] DetectSkill: {det_res.error_message}")
        return 1

    count = det_res.result_data.get("count", 0)
    if count != 1:
        print(f"FAIL [1] DetectSkill: expected count=1, got {count}")
        return 1

    objects = wm.get_objects()
    if len(objects) != 1:
        print(f"FAIL [1] world_model should have 1 object, got {len(objects)}")
        return 1

    obj = objects[0]
    print(
        f"  [1] DetectSkill OK: added {obj.object_id!r} at "
        f"({obj.x:.2f}, {obj.y:.2f}, {obj.z:.2f})"
    )

    # Sanity-check world coords are plausible (object ~1.2 m in front at origin)
    if not (0.0 < obj.x < 2.5):
        print(f"FAIL [1] world x={obj.x:.2f} out of expected range (0, 2.5)")
        return 1

    # ------------------------------------------------------------------
    # Step 2 — MobilePickSkill with fresh empty world_model (forces auto-detect)
    # ------------------------------------------------------------------
    from vector_os_nano.core.types import SkillResult
    from vector_os_nano.skills.mobile_pick import MobilePickSkill

    wm2 = WorldModel()  # fresh — empty
    ctx2 = SkillContext(
        base=base,
        arm=arm,
        gripper=gripper,
        perception=perception,
        calibration=calibration,
        world_model=wm2,
    )

    mobile = MobilePickSkill()

    # Stub PickTopDownSkill.execute to succeed without real IK / arm stack
    def _fake_pick_execute(params, ctx):  # noqa: ARG001
        return SkillResult(success=True, result_data={"held": True, "lift_cm": 3.0})

    mobile._pick.execute = _fake_pick_execute  # type: ignore[method-assign]

    pick_res = mobile.execute(
        {"object_label": "蓝色瓶子", "skip_navigate": True},
        ctx2,
    )

    if not pick_res.success:
        diag = pick_res.result_data.get("diagnosis", "?")
        print(
            f"FAIL [2] MobilePickSkill: {pick_res.error_message!r} "
            f"(diagnosis={diag!r})"
        )
        return 1

    held = pick_res.result_data.get("held", False)
    print(
        f"  [2] MobilePickSkill OK: held={held}, "
        f"nav_calls={len(base.nav_calls)}, auto-detect succeeded"
    )

    # ------------------------------------------------------------------
    # Step 3 — Verify auto-detect populated wm2
    # ------------------------------------------------------------------
    wm2_objects = wm2.get_objects()
    if len(wm2_objects) < 1:
        print("FAIL [3] auto-detect did not populate wm2")
        return 1

    print(f"  [3] world_model has {len(wm2_objects)} object(s) after auto-detect")

    print()
    print("OK: perception->pick chain verified")
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run with mocked Qwen + synthetic frames (default; always true for now)",
    )
    args = parser.parse_args()  # noqa: F841  — consumed for future --live flag

    try:
        rc = run_dry_run()
    except Exception as exc:
        import traceback

        print(f"FAIL: {exc}")
        traceback.print_exc()
        rc = 1

    sys.exit(rc)


if __name__ == "__main__":
    main()
