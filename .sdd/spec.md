# v2.3 Go2 Perception Pipeline — Specification

**Status**: DRAFT — awaiting CEO approval
**Scope**: MuJoCo-sim only. Sim-to-real deferred to v2.4+.
**Branch**: `feat/v2.0-vectorengine-unification` (same branch as v2.2)

## 1. Overview

Give the Go2+Piper sim the same `context.perception.detect() + track()`
contract SO-101 already uses, so `DetectSkill` / `MobilePickSkill` work
end-to-end on live Go2. Replaces the MJCF-populate shortcut (removed in
v2.2) with a real perception loop driven by Qwen VLM and MuJoCo depth.

Target user flow after v2.3:

```
go2sim with_arm=1            # launch stack, world_model empty
抓个蓝色瓶子                   # LLM picks MobilePickSkill
  └─ _resolve_target miss    # world_model has nothing
      └─ auto-detect via context.perception  (NEW)
          ├─ Qwen detect("blue bottle") → bbox
          ├─ depth project bbox centre → 3D in camera frame
          └─ calibration.camera_to_base → world xyz → upsert
      └─ _resolve_target retry → hit
  └─ approach_pose / navigate / wait_stable
  └─ PickTopDownSkill (existing) → held=True
```

## 2. Background & Motivation

v2.2 shipped the manipulation infrastructure (`MobilePickSkill`,
`MobilePlaceSkill`, `PlaceTopDownSkill`, `Ros2Runtime`) and removed the
MJCF-populate shortcut that pretended object positions were known a
priori. The baseline is now clean — world_model starts empty — but
`context.perception` remains `None` on Go2, so `DetectSkill` returns
`no_perception` and the full pipeline breaks at the first step.

SO-101 already has this wired (`PerceptionPipeline(RealSense, Moondream,
EdgeTAM)` → `context.perception`). Go2 has `Go2VLMPerception` but it
returns `DetectedObject` (name + description, **no bboxes**), not the
`list[Detection] + list[TrackedObject]` the `PerceptionProtocol`
requires. That gap is the reason `抓个东西` doesn't work.

Additional gap: `DetectSkill` expects `context.calibration.camera_to_base(cam_xyz)`
to return a base-frame position. SO-101's calibration does this via
hand-eye RBF fitted from training points. Go2 has no calibration points
— it needs a pose-driven calibration computed from odometry + static
mount geometry.

## 3. Goals

### MUST
- **G1**: `Go2Perception` class implementing `PerceptionProtocol`
  (`get_color_frame / get_depth_frame / get_intrinsics / detect / track`)
  on top of `Go2ROS2Proxy` frames (`/camera/image`, `/camera/depth`).
- **G2**: `Go2Calibration` class exposing `camera_to_base(cam_xyz)` that
  returns **world-frame** coordinates (semantically: "the frame the
  world_model uses"). Pose-driven — reads `Go2ROS2Proxy.get_camera_pose()`
  per call.
- **G3**: `QwenVLMDetector` class calling OpenRouter
  `qwen/qwen2.5-vl-72b-instruct`, returning `list[Detection]` with
  pixel-space bboxes.
- **G4**: `sim_tool._start_go2(with_arm=True)` wires `agent._perception
  = Go2Perception(camera=base, vlm=QwenVLMDetector(...))` and
  `agent._calibration = Go2Calibration(base_proxy=base)`.
- **G5**: `DetectSkill` works live on Go2 — `detect("blue bottle")`
  populates `world_model` with at least one `ObjectState` with valid
  world-frame xyz.
- **G6**: `MobilePickSkill._resolve_target` — on world_model miss AND
  `context.perception` available — auto-invokes `DetectSkill`, then
  retries resolve. On miss without perception: existing
  `object_not_found` error is preserved.
- **G7**: Live-REPL end-to-end: `go2sim with_arm=1 → 抓个蓝色瓶子`
  succeeds without pre-populating world_model.

### SHOULD
- **S1**: Unit coverage ≥95% on `vlm_qwen.py`, `go2_perception.py`,
  `go2_calibration.py`.
- **S2**: Qwen VLM call latency budget ≤ 4 s per detect call at
  `_VLM_IMAGE_MAX_DIM=160` JPEG encoding (already used by `Go2VLMPerception`).
- **S3**: Depth denoising: bbox-masked median depth with IQR outlier
  rejection (same pattern as `PerceptionPipeline._remove_depth_outliers`).
- **S4**: E2E harness `scripts/verify_perception_pick.py` with
  `--dry-run` mode (mock Qwen response + synthetic depth) for CI.
- **S5**: Local-VLM escape hatch: honour existing `VECTOR_VLM_URL` env
  var to route Qwen calls to a local OpenAI-compatible endpoint
  (e.g. vLLM on the RTX 5080) without code change.

### MAY
- **M1**: Cost / latency telemetry surfaced via
  `engine._vgg_step_callback` so the REPL can show "VLM: 1.8s, $0.0012"
  during perception.
- **M2**: Per-call `Go2Perception` frame freshness check (warn if RGB/depth
  drift > 500 ms).

## 4. Non-Goals (explicitly out of scope)

- **NG1**: SAM3D / per-pixel masks. We use bbox centroid + median depth
  only.
- **NG2**: EdgeTAM or any temporal tracker. `track()` is single-shot
  per call (treat each VLM detect as independent; no persistence beyond
  world_model upsert).
- **NG3**: Multi-view fusion or active perception (move to see).
- **NG4**: Real Piper / real Go2 hardware integration. MuJoCo sim only.
- **NG5**: Sim-to-real: distortion coefficients, real-D435 depth scale
  (0.001), real ROS2 topic names (`/camera/color/image_raw` etc.),
  real TF tree via `tf2_ros`. All deferred to a later cycle.
- **NG6**: Confidence-weighted merge / Kalman-filter tracking in
  world_model. Upsert-by-id (current `WorldModel.add_object` behavior)
  is sufficient.
- **NG7**: Retraining / fine-tuning the VLM. Use Qwen2.5-VL-72B as-is.

## 5. User Scenarios

### Scenario 1: Fresh sim, VLM-driven grasp

- **Actor**: Yusen at the vector-cli REPL.
- **Trigger**: `go2sim with_arm=1` then `抓起蓝色瓶子`.
- **Expected Behavior**:
  1. Sim launches, world_model empty, `agent._perception ≠ None`.
  2. LLM routes to `MobilePickSkill`.
  3. Skill finds world_model empty → invokes `DetectSkill("blue bottle")`.
  4. Qwen returns 1–3 bboxes; depth projection yields world-frame xyz.
  5. `MobilePickSkill` retries resolve, gets target, navigates, picks.
- **Success Criteria**: Pick succeeds with `held=True` and lift >2 cm;
  no crashes; VLM call completes in <4 s.

### Scenario 2: No perception configured

- **Actor**: Engineer running unit tests.
- **Trigger**: `MobilePickSkill` with `context.perception = None` and
  empty world_model.
- **Expected Behavior**: Returns `object_not_found` with
  `"Known pickable objects: []"` message (unchanged v2.2 behavior).
- **Success Criteria**: No AttributeError; `diagnosis = "object_not_found"`
  (not `no_perception`).

### Scenario 3: VLM API failure

- **Actor**: Skill executor during sim run.
- **Trigger**: OpenRouter returns 503 for 2 consecutive retries.
- **Expected Behavior**: `DetectSkill` returns
  `success=False, diagnosis="no_perception"` with error string
  including retry count. `MobilePickSkill` bubbles up as
  `object_not_found` (perception unreachable → treated as "no data").
- **Success Criteria**: No hang >30 s; no infinite retry; clean error.

### Scenario 4: VLM returns bbox for wrong object

- **Actor**: Skill executor; VLM returned a `"bottle"` bbox over a coke
  can.
- **Expected Behavior**: `DetectSkill` writes the ObjectState under
  the returned label. MobilePickSkill resolves on label → picks up the
  coke can. User gets the wrong object, but no crash.
- **Success Criteria**: Graceful failure / graceful wrong pick. LLM can
  re-plan based on VGG observation of what was picked.

## 6. Technical Constraints

- **Runtime**: Ubuntu 24.04, ROS2 Jazzy (via `Ros2Runtime` shared
  executor), Python 3.12.
- **VLM**: OpenRouter `qwen/qwen2.5-vl-72b-instruct`. API key via
  `OPENROUTER_API_KEY` env var or `config/user.yaml` (existing path).
- **Camera source**: `Go2ROS2Proxy.get_rgbd_frame()` — 320×240, RGB8 +
  float32 m depth, published at 5 Hz by `Go2VNavBridge` from the
  MJCF-declared `d435_rgb` / `d435_depth` cameras.
- **Intrinsics**: `mujoco_intrinsics(320, 240, vfov_deg=42.0)` (matches
  MJCF `fovy=42`).
- **Camera mount** (MJCF `go2_piper.xml` line 188):
  `body d435_camera pos="0.25 0 0.1" quat="0.999054 0 0.0434863 0"`
  (~5° downward pitch on trunk). `Go2ROS2Proxy.get_camera_pose()`
  already produces world xmat from this geometry.
- **Depth convention**: OpenCV (x=right, y=down, z=forward) — matches
  `pixel_to_camera()` in `depth_projection.py`.
- **No new ROS2 interfaces**: no new topic / service / action /
  parameter declarations. All changes are Python-side.
- **No changes to**: Piper / gripper proxies; pick / place skills;
  MPC / MuJoCo scene; nav stack.

## 7. Interface Definitions

### 7.1 New Python modules

```
vector_os_nano/perception/vlm_qwen.py
    class QwenVLMDetector:
        def __init__(self, config: dict | None = None) -> None: ...
        def detect(self, image: np.ndarray, query: str) -> list[Detection]: ...
        @property
        def cumulative_cost_usd(self) -> float: ...

vector_os_nano/perception/go2_perception.py
    class Go2Perception:                # implements PerceptionProtocol
        def __init__(self, camera: Go2ROS2Proxy, vlm: QwenVLMDetector,
                     intrinsics: CameraIntrinsics | None = None) -> None: ...
        def get_color_frame(self) -> np.ndarray: ...
        def get_depth_frame(self) -> np.ndarray: ...
        def get_intrinsics(self) -> CameraIntrinsics: ...
        def detect(self, query: str) -> list[Detection]: ...
        def track(self, detections: list[Detection]) -> list[TrackedObject]: ...
        def get_point_cloud(self, mask: np.ndarray | None = None) -> np.ndarray: ...

vector_os_nano/perception/go2_calibration.py
    class Go2Calibration:
        def __init__(self, base_proxy: Go2ROS2Proxy) -> None: ...
        def camera_to_base(self, point_camera: np.ndarray) -> np.ndarray:
            """Return WORLD-frame xyz. Naming 'base' keeps parity with
            SO-101 Calibration where arm-base IS world. For Go2, world is
            the coordinate system world_model and MobilePickSkill use."""
```

### 7.2 Modified Python modules

| File | Change |
|---|---|
| `vcli/tools/sim_tool.py::_start_go2(with_arm=True)` | After Piper wiring: construct `QwenVLMDetector` (re-use `api_key`), `Go2Perception(camera=base, vlm=…)`, `Go2Calibration(base_proxy=base)`; assign to `agent._perception` / `agent._calibration`. Keep `agent._vlm = Go2VLMPerception` unchanged (caption / identify_room). |
| `skills/mobile_pick.py::execute` | After first `_resolve_target` miss: if `context.perception` available AND `context.calibration` available → run `DetectSkill({"query": _label_to_en_query(params)})` → retry resolve. On perception unavailable: fall through to existing `object_not_found`. |
| `skills/utils/__init__.py` | Add `_label_to_en_query(label: str) -> str` helper (CN → EN, strip trailing "的", uses `_normalise_color_keyword` already present for colours). |

### 7.3 External dependencies

- `httpx` (already pinned) — Qwen OpenRouter calls.
- `numpy` (already pinned) — depth math.
- **No new package dependencies.**

### 7.4 Environment variables (new / reused)

| Var | Purpose | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | Qwen API key | (existing) |
| `VECTOR_VLM_URL` | Route Qwen calls to local OpenAI-compatible endpoint | unset (use OpenRouter) |
| `VECTOR_VLM_MODEL` | Override Qwen model id | `qwen/qwen2.5-vl-72b-instruct` |
| `VECTOR_PERCEPTION_DRYRUN` | E2E test only: mock Qwen response + synthetic depth | 0 |

## 8. Test Contracts

### 8.1 Unit test contracts (pytest, no ROS2, no MuJoCo)

#### QwenVLMDetector
- [ ] `test_qwen_detect_parses_json_bbox_list` — mock httpx 200 with
      `[{"label":"bottle","bbox":[10,20,30,40],"confidence":0.9}]`;
      returns `[Detection(label="bottle", bbox=(10,20,30,40))]`.
- [ ] `test_qwen_detect_parses_markdown_fenced_json` — mock 200 with
      ` ```json\n[...]\n``` `; parses correctly (reuse
      `_parse_json_response` from `vlm_go2.py`).
- [ ] `test_qwen_detect_empty_response_returns_empty_list` — mock `[]`.
- [ ] `test_qwen_detect_timeout_retries_then_fails` — two httpx
      `TimeoutException` → RuntimeError after `_MAX_RETRIES=2`.
- [ ] `test_qwen_detect_4xx_does_not_retry` — mock 401 → RuntimeError
      after 1 attempt.
- [ ] `test_qwen_detect_normalised_bbox_scaled_to_pixels` — mock
      response with bbox in 0–1 range; scaled to frame WxH.
- [ ] `test_qwen_detect_cost_tracked` — accumulates on success.
- [ ] `test_qwen_detect_honours_VECTOR_VLM_URL_env` — local mode, no
      auth header, model from `VECTOR_VLM_MODEL`.

#### Go2Perception
- [ ] `test_go2_perception_get_color_delegates_to_proxy`
- [ ] `test_go2_perception_get_depth_delegates_to_proxy`
- [ ] `test_go2_perception_detect_calls_vlm_with_color_frame` — mock
      proxy returns fixed RGB; verify `vlm.detect(rgb, query)` called.
- [ ] `test_go2_perception_track_single_detection_projects_centroid` —
      synthetic depth frame (all pixels = 1.5 m); bbox (100,80,150,120);
      intrinsics fx=fy=262, cx=160, cy=120; verify
      `TrackedObject.pose = Pose3D(x≈-0.206, y≈-0.114, z=1.5)` in
      camera frame (OpenCV convention).
- [ ] `test_go2_perception_track_handles_bbox_with_no_valid_depth` —
      depth all zero; `pose = None`, `bbox_2d` populated.
- [ ] `test_go2_perception_track_uses_median_depth_in_bbox` — depth =
      1.0 m in bbox except one outlier at 9 m; median = 1.0 m.
- [ ] `test_go2_perception_track_ignores_nan_depth_pixels` — bbox
      contains NaN pixels; they're filtered before median.
- [ ] `test_go2_perception_track_returns_same_length_as_detections` —
      len(tracked) == len(detections), one-to-one.

#### Go2Calibration
- [ ] `test_calibration_dog_at_origin_facing_x_forward_1m` — dog pose
      `(0,0,0,0)`, cam point `(0, 0, 1.0)` (1 m forward in camera
      frame); result ≈ `(1.3, 0, 0.05)` accounting for 0.3 m mount
      forward + 0.05 m up + -5° pitch.
- [ ] `test_calibration_respects_heading_90deg` — dog facing +Y (yaw
      π/2); cam point 1 m forward; result `(0, 1.3, 0.05)`.
- [ ] `test_calibration_offset_dog_position_adds_to_world` — dog at
      `(5, 3, 0.28)`; same cam point; result shifted by (5, 3, 0.28+0.05).
- [ ] `test_calibration_camera_to_base_shape_preserved` — input (3,),
      output (3,).

#### MobilePickSkill (auto-detect retry)
- [ ] `test_mobile_pick_retries_via_perception_on_world_model_miss` —
      stub context with `perception.detect` returning 1 detection,
      `.track` returning 1 tracked with pose; verify `world_model.add_object`
      called and second `_resolve_target` returns the new object.
- [ ] `test_mobile_pick_no_retry_when_perception_none` — preserves
      existing `object_not_found` behavior.
- [ ] `test_mobile_pick_no_retry_when_calibration_none` — same (can't
      transform cam→world without calibration).
- [ ] `test_mobile_pick_surfaces_vlm_error_as_object_not_found` — VLM
      raises RuntimeError → skill returns `object_not_found`, not crash.

### 8.2 Integration test contracts (rclpy, no MuJoCo)

- [ ] `test_sim_tool_start_go2_with_arm_wires_perception` — mock
      Go2ROS2Proxy.connect; verify `agent._perception` is
      `Go2Perception` and `agent._calibration` is `Go2Calibration`.
- [ ] `test_sim_tool_start_go2_without_arm_leaves_perception_none` —
      non-manipulation mode; perception stays None.
- [ ] `test_sim_tool_start_go2_with_arm_no_api_key_skips_perception` —
      config has no API key; perception None, `_vlm` None; no crash.

### 8.3 E2E contract (MuJoCo-optional dry-run)

- [ ] `scripts/verify_perception_pick.py --dry-run` — stubs Qwen HTTP
      with a synthetic bbox over `pickable_blue_bottle` region; stubs
      depth with flat 1.1 m plane; verifies full chain
      `DetectSkill → world_model populate → MobilePickSkill → held=True`.

## 9. Acceptance Criteria

1. **AC-1 (G1/G4)**: After `go2sim with_arm=1`, `agent._perception` is
   a `Go2Perception` instance; `agent._calibration` is a `Go2Calibration`.
2. **AC-2 (G3)**: `QwenVLMDetector.detect(rgb, "blue bottle")` against
   live OpenRouter returns ≥1 `Detection` with `bbox ⊂ [0,W]×[0,H]`.
3. **AC-3 (G1)**: `Go2Perception.track([Detection(...)])` returns a
   list where each element has `pose: Pose3D | None`, `pose.z > 0` when
   the bbox has valid depth pixels.
4. **AC-4 (G2)**: `Go2Calibration.camera_to_base(np.array([0,0,1]))`
   when dog at origin facing +X returns xyz with `x ≈ 1.3 ± 0.05 m`,
   `y ≈ 0 ± 0.05 m`, `z ≈ 0.05 ± 0.05 m` (mount geometry verified).
5. **AC-5 (G5)**: After `DetectSkill(query="bottle")` on live Go2 with
   3 bottles in scene, `world_model.get_objects()` contains ≥1 object
   with `object_id.startswith("bottle")` or the detected label.
6. **AC-6 (G6)**: `MobilePickSkill(object_label="blue bottle")` with
   empty `world_model` but `context.perception` available — skill
   proceeds past resolve stage (does NOT return `object_not_found`).
7. **AC-7 (G6)**: Same skill call with `context.perception = None` —
   returns `diagnosis="object_not_found"` (backward compat).
8. **AC-8 (S1)**: Unit coverage ≥95 % on `vlm_qwen.py`,
   `go2_perception.py`, `go2_calibration.py`.
9. **AC-9 (G7)**: Live-REPL smoke: `go2sim with_arm=1 → 抓起蓝色瓶子`
   — pick succeeds, `held=True`, total time <30 s.
10. **AC-10 (test discipline)**: 14 + 3 + 1 = ≥18 new tests (unit +
    integration + E2E-dry-run); total suite remains green (108 existing
    + ≥18 new ≥ 126 passing).

## 10. Open Questions

None blocking. Recommended defaults below; flag in CEO review if you
want to override.

| # | Question | Recommended default | Reason |
|---|---|---|---|
| Q1 | Which Qwen model? | `qwen/qwen2.5-vl-72b-instruct` on OpenRouter | Best grounding accuracy, cheap (~$0.001/call at 160-px thumb); local 7B deferred to v2.4 |
| Q2 | Return TrackedObject.mask? | `None` | No tracker in v2.3; downstream (PickTopDown) only reads pose |
| Q3 | Depth outlier rejection inside bbox? | IQR on z-axis, fallback to raw median | Matches existing PerceptionPipeline `_remove_depth_outliers`; one-liner |
| Q4 | Where does VLM query for `MobilePick` come from? | `params["object_label"]` → `_label_to_en_query` (strips 的, applies color normaliser EN→original, lowercases) | CN label like "蓝色瓶子" → "blue bottle"; reuse v2.2 normaliser |
| Q5 | `world_model` merge policy on re-detect? | upsert-by-label (existing `add_object` behavior) | Simplest; OK for v2.3 since pick is immediate |
| Q6 | Cost visible to user? | log line only in v2.3 | MAY requirement M1; dashboard integration out of scope |
| Q7 | Perception retries on failure? | Qwen client: 2 retries on transport; DetectSkill: no retry above that | Same as Go2VLMPerception |

---

## CEO Review

**Decision Request**: Approve v2.3 spec with defaults above? Two
explicit CEO choices already made:
1. VLM = Qwen ✓ (recommended: qwen2.5-vl-72b-instruct via OpenRouter)
2. MuJoCo sim only, defer sim-to-real ✓

Once approved, Architect proceeds to `plan.md` (Phase 2).
