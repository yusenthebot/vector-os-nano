# v2.3 Go2 Perception Pipeline — Technical Plan

**Phase**: plan (draft)
**Architectural**: NO — no new ROS2 nodes / topics / services /
cross-package boundaries. Yusen reviews async; Agent Team may proceed
once Dispatcher validates.
**Prereq**: `.sdd/spec.md` approved 2026-04-19 21:20Z.

## 1. Architecture Overview

All changes are in-process Python — no new ROS2 interfaces. The sim
subprocess (`launch_explore.sh` → bridge + nav + MuJoCo) is untouched.
vector-cli gains a new perception layer that lives inside `agent`:

```
┌──────────────────────────── vector-cli process ────────────────────────────┐
│  ┌──────────────────────────── Agent ────────────────────────────────────┐ │
│  │  _base: Go2ROS2Proxy          _arm: PiperROS2Proxy                    │ │
│  │                                                                        │ │
│  │  _vlm:       Go2VLMPerception      (unchanged — describe / identify)  │ │
│  │  _perception: Go2Perception        ← NEW                              │ │
│  │                ├── camera: self._base    (Go2ROS2Proxy)               │ │
│  │                └── vlm:    QwenVLMDetector  ← NEW                     │ │
│  │  _calibration: Go2Calibration      ← NEW                              │ │
│  │                └── base_proxy: self._base                             │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                              ▲       │                                     │
│                              │       ▼                                     │
│  SkillContext(perception=_perception, calibration=_calibration, ...)      │
│       │                                                                    │
│       ▼                                                                    │
│  DetectSkill, MobilePickSkill (modified), ...                             │
└────────────────────────────────────────────────────────────────────────────┘
     │               │                             ▲
     │ subscribe     │ publish                     │ HTTPS (OpenRouter)
     ▼               ▼                             │
┌─ ROS2 DDS ──────────────────┐         ┌─────────────────────────┐
│  /camera/image (RGB8)      │         │  qwen/qwen2.5-vl-72b-    │
│  /camera/depth (32FC1)     │         │  instruct                │
│  /state_estimation (Odom)  │         └─────────────────────────┘
└─────────────────────────────┘
             ▲
             │
┌─ sim subprocess ──────────────────┐
│  Go2VNavBridge (publishes above)  │
│  MuJoCo go2_piper.xml             │
│  convex_mpc + nav stack (TARE)    │
└───────────────────────────────────┘
```

## 2. Technical Decisions

| Component | Choice | Rationale |
|---|---|---|
| VLM provider | OpenRouter REST API | No GPU management, zero-config, existing httpx infra, ~$0.001/call at 160-px thumbnail |
| VLM model | `qwen/qwen2.5-vl-72b-instruct` | Strong JSON-schema following; native grounding support; 72B gives bbox accuracy >> 7B |
| VLM prompt style | Explicit JSON schema ("return list of objects with bbox") | More reliable than Qwen's `<ref>/<box>` grounding syntax; matches existing `_parse_json_response` parser |
| Image resize | Reuse `_VLM_IMAGE_MAX_DIM=160` remote / `512` local | Existing tuning; remote payload <10 KB keeps latency 1–2 s |
| Tracker | **None** in v2.3 | Spec non-goal NG2; single-shot bbox-centre projection is enough for immediate grasp |
| Depth sampling | Median of bbox pixels, IQR outlier reject on z-axis | Matches `PerceptionPipeline._remove_depth_outliers`; robust to edge bleed |
| Calibration style | Pose-driven (not point-fitted) | Go2 has no hand-eye calibration fixture; MJCF geometry is the ground truth — read via `Go2ROS2Proxy.get_camera_pose()` per call |
| World-frame convention | `Go2Calibration.camera_to_base()` returns world xyz | Lets existing `DetectSkill` + `world_model` + `MobilePickSkill` keep using `.x / .y` as world without branching |
| Auto-detect retry | Inside `MobilePickSkill.execute`, delegates to `DetectSkill` | Composition reuse; one code path for "find missing object" |
| Failure propagation | VLM error → DetectSkill `no_perception`; MobilePick `object_not_found` | Existing failure-mode labels preserved |
| Testing — Qwen client | `pytest` + `httpx_mock` or monkey-patched `httpx.Client` | Standard pattern already used in vlm_go2 tests |
| Testing — depth math | Synthetic `np.ndarray` frames injected via `set_synthetic_frames`-style hook | SO-101 `PerceptionPipeline` already has this pattern — mirror it |
| Testing — integration | `pytest` + `rclpy.init` + monkey-patched `Go2ROS2Proxy.connect` (no real bridge) | Existing 3 rclpy-coexist integration tests set precedent |
| Lint | ruff E/F/W/I/B/N/UP/RUF (existing `pyproject.toml`) | No config changes |
| Coverage target | ≥95 % on new modules | Per spec S1 |

## 3. Module Design

### 3.1 `vector_os_nano/perception/vlm_qwen.py` (NEW)

Responsibility: One-shot object detection via Qwen2.5-VL, returning
pixel-space bboxes.

```python
class QwenVLMDetector:
    """Grounded object detection via Qwen2.5-VL on OpenRouter.

    Thread-safe (lock-guarded cost tracker).
    Satisfies the interface `detect(image, query) -> list[Detection]`.
    """

    DEFAULT_MODEL = "qwen/qwen2.5-vl-72b-instruct"
    DEFAULT_URL = "https://openrouter.ai/api/v1"
    TIMEOUT_S = 30.0
    MAX_RETRIES = 2
    JPEG_QUALITY = 50
    IMAGE_MAX_DIM_REMOTE = 160
    IMAGE_MAX_DIM_LOCAL = 512

    def __init__(self, config: dict | None = None) -> None:
        # Local mode (VECTOR_VLM_URL) → no auth; remote → OPENROUTER_API_KEY
        # Honour VECTOR_VLM_MODEL override

    def detect(self, image: np.ndarray, query: str) -> list[Detection]:
        # 1. Resize + JPEG encode (base64)
        # 2. POST /chat/completions with JSON-schema prompt:
        #    "List all <query> in the image. Return ONLY a JSON array:
        #    [{\"label\": \"...\", \"bbox\": [x1,y1,x2,y2],
        #      \"confidence\": 0.0-1.0}]
        #    Coordinates are absolute pixel values (0..W, 0..H).
        #    Empty array if none."
        # 3. Parse via _parse_json_response (reuse from vlm_go2.py)
        # 4. For each item: auto-scale bbox (if max <= 1.0, multiply by W/H)
        # 5. Return list[Detection]

    @property
    def cumulative_cost_usd(self) -> float: ...
```

Reuses `_parse_json_response` + `_parse_detected_object` helpers from
`vlm_go2.py` by importing them (they're already module-level pure
functions). `Detection` comes from `core.types`.

### 3.2 `vector_os_nano/perception/go2_perception.py` (NEW)

Responsibility: Adapt `Go2ROS2Proxy` frames + `QwenVLMDetector` into
the `PerceptionProtocol` surface expected by `DetectSkill`.

```python
class Go2Perception:
    """PerceptionProtocol impl for Go2 in MuJoCo sim.

    Composes Go2ROS2Proxy (camera frames) + QwenVLMDetector (bboxes).
    Single-shot track: bbox-centre + bbox median depth → 3D.
    """

    def __init__(
        self,
        camera: Any,                       # Go2ROS2Proxy-like (duck typed)
        vlm: QwenVLMDetector,
        intrinsics: CameraIntrinsics | None = None,
        depth_trunc: float = 10.0,
    ) -> None:
        self._camera = camera
        self._vlm = vlm
        self._intrinsics = intrinsics or mujoco_intrinsics(320, 240, 42.0)
        self._depth_trunc = depth_trunc
        self._last_detections: list[Detection] = []
        self._last_tracked: list[TrackedObject] = []

    # PerceptionProtocol
    def get_color_frame(self) -> np.ndarray:
        return self._camera.get_camera_frame()

    def get_depth_frame(self) -> np.ndarray:
        return self._camera.get_depth_frame()

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def detect(self, query: str) -> list[Detection]:
        rgb = self.get_color_frame()
        detections = self._vlm.detect(rgb, query)
        self._last_detections = detections
        return detections

    def track(self, detections: list[Detection]) -> list[TrackedObject]:
        depth = self.get_depth_frame()
        intr = self._intrinsics
        out: list[TrackedObject] = []
        for i, det in enumerate(detections):
            pose = self._project_bbox_to_camera_frame(det.bbox, depth, intr)
            out.append(TrackedObject(
                track_id=i + 1,
                label=det.label,
                bbox_2d=det.bbox,
                pose=pose,
                bbox_3d=None,                  # no pointcloud in v2.3
                confidence=det.confidence,
                mask=None,                     # no masker in v2.3
            ))
        self._last_tracked = out
        return out

    def get_point_cloud(self, mask: np.ndarray | None = None) -> np.ndarray:
        # Delegate to depth_projection; used by DescribeSkill (unchanged)
        ...

    # ------- private -------
    @staticmethod
    def _project_bbox_to_camera_frame(
        bbox: tuple[float, float, float, float],
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> Pose3D | None:
        """Bbox centre → median bbox depth → camera-frame Pose3D."""
        x1, y1, x2, y2 = bbox
        h, w = depth.shape[:2]
        # Clamp + integer
        x1i = max(0, min(int(round(x1)), w - 1))
        y1i = max(0, min(int(round(y1)), h - 1))
        x2i = max(x1i + 1, min(int(round(x2)), w))
        y2i = max(y1i + 1, min(int(round(y2)), h))
        if x2i <= x1i or y2i <= y1i:
            return None

        # Centre pixel
        u = (x1 + x2) * 0.5
        v = (y1 + y2) * 0.5

        # Sample depth inside bbox, reject NaN / outside trunc / ≤ 0
        patch = depth[y1i:y2i, x1i:x2i]
        valid = patch[np.isfinite(patch) & (patch > 0) & (patch < self._depth_trunc)]
        if valid.size == 0:
            return None

        # IQR outlier reject on z
        if valid.size >= 10:
            q1, q3 = np.percentile(valid, [25, 75])
            iqr = q3 - q1
            if iqr > 1e-6:
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                valid = valid[(valid >= low) & (valid <= high)]
                if valid.size == 0:
                    return None

        d = float(np.median(valid))
        x_cam, y_cam, z_cam = pixel_to_camera(u, v, d, intrinsics)
        return Pose3D(x=x_cam, y=y_cam, z=z_cam)
```

**Design note** — we do NOT implement `caption / visual_query` here.
`PerceptionProtocol` includes them but `DetectSkill` doesn't call them;
they remain optional. Go2VLMPerception (kept at `agent._vlm`) already
services describe_scene / identify_room callers.

### 3.3 `vector_os_nano/perception/go2_calibration.py` (NEW)

Responsibility: Transform a camera-frame point into world-frame,
using the robot's current camera pose (pose-driven, not point-fitted).

```python
class Go2Calibration:
    """Go2 camera-to-world calibration via current camera pose.

    Mirrors Calibration (SO-101) interface but is driven by the robot's
    live odometry + static MJCF mount geometry rather than paired
    training points. The method name `camera_to_base` is kept for
    interface parity — for Go2, "base" means the WORLD frame that
    world_model and MobilePickSkill operate in (SO-101's arm base IS
    world in a fixed workspace, so semantics match).
    """

    def __init__(self, base_proxy: Any) -> None:
        """
        Args:
            base_proxy: object with .get_camera_pose() -> (cam_xpos, cam_xmat)
        """
        self._base = base_proxy

    def camera_to_base(self, point_camera: np.ndarray) -> np.ndarray:
        """OpenCV-frame (x=right, y=down, z=forward) cam point → world xyz."""
        p = np.asarray(point_camera, dtype=np.float64).reshape(3)
        cam_xpos, cam_xmat = self._base.get_camera_pose()
        xmat = np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3)
        pos = np.asarray(cam_xpos, dtype=np.float64)
        # MuJoCo xmat cols = [right, up, -forward]
        cam_right   = xmat[:, 0]
        cam_up      = xmat[:, 1]
        cam_forward = -xmat[:, 2]
        # OpenCV-frame point: x=right, y=down, z=forward
        world = pos + p[0] * cam_right + (-p[1]) * cam_up + p[2] * cam_forward
        return world
```

Mathematics are identical to `depth_projection.camera_to_world` when
`cam_xpos / cam_xmat` are supplied — proven correct by existing
`test_depth_projection.py` on the SO-101 side. We do NOT import
that function (it takes scalars, not arrays) — we inline the 3-line
matrix-vector form here.

### 3.4 `vector_os_nano/vcli/tools/sim_tool.py` (MODIFIED)

One new block inside `SimStartTool._start_go2` after `piper_arm +
piper_gripper` init, guarded by `with_arm` + API-key presence:

```python
# --- v2.3 NEW: perception + calibration wiring ---
if with_arm and api_key:
    try:
        from vector_os_nano.perception.vlm_qwen import QwenVLMDetector
        from vector_os_nano.perception.go2_perception import Go2Perception
        from vector_os_nano.perception.go2_calibration import Go2Calibration
        qwen = QwenVLMDetector(config={"api_key": api_key})
        agent._perception = Go2Perception(camera=base, vlm=qwen)
        agent._calibration = Go2Calibration(base_proxy=base)
        logger.info("[sim_tool] Go2 perception + calibration wired (Qwen)")
    except Exception as exc:
        logger.warning("[sim_tool] Perception wire-up failed: %s", exc)
        agent._perception = None
        agent._calibration = None
# keep existing Go2VLMPerception for describe/identify
agent._vlm = Go2VLMPerception(config={"api_key": api_key}) if api_key else None
```

The existing `_vlm` block (Go2VLMPerception) stays but is moved
**after** perception wiring for log ordering. Both attributes coexist.

### 3.5 `vector_os_nano/skills/mobile_pick.py` (MODIFIED)

Insert an auto-detect retry block between `_resolve_target` first
attempt and the early-return `object_not_found` branch:

```python
def execute(self, params, context):
    # ... existing guards (base / arm / gripper / world_model) unchanged ...

    target = self._pick._resolve_target(params, context.world_model)

    # --- v2.3 NEW: perception-driven auto-detect retry ---
    if target is None and context.perception is not None and context.calibration is not None:
        from vector_os_nano.skills.detect import DetectSkill
        from vector_os_nano.skills.utils import label_to_en_query

        query = label_to_en_query(params.get("object_label") or params.get("object_id"))
        if query:
            logger.info("[MOBILE-PICK] world_model miss; auto-detect query=%r", query)
            det_ctx = context   # same context — DetectSkill writes world_model
            try:
                det_result = DetectSkill().execute({"query": query}, det_ctx)
            except Exception as exc:
                logger.warning("[MOBILE-PICK] auto-detect crashed: %s", exc)
                det_result = None
            if det_result is not None and det_result.success and \
               det_result.result_data.get("count", 0) > 0:
                target = self._pick._resolve_target(params, context.world_model)

    if target is None:
        # ... existing object_not_found return unchanged ...
```

No other code path in MobilePickSkill changes.

### 3.6 `vector_os_nano/skills/utils/__init__.py` (MODIFIED)

Add one helper:

```python
_CN_NOUN_MAP = {
    "瓶子": "bottle",
    "杯子": "cup",
    "碗":   "bowl",
    "盘子": "plate",
    "罐子": "can",
    "盒子": "box",
    "球":   "ball",
}

def label_to_en_query(label: str | None) -> str | None:
    """Convert Chinese/mixed label to an English VLM query.

    Strips leading/trailing spaces, removes the possessive "的",
    translates known color keywords via _normalise_color_keyword
    (existing), and maps common nouns via _CN_NOUN_MAP.

    Examples:
        "蓝色瓶子"   -> "blue bottle"
        "红色的杯子" -> "red cup"
        "bottle"     -> "bottle"
        None / ""    -> None   (caller skips auto-detect)
    """
```

Logic: split on spaces + CN characters; for each token, map color (reuse
`_normalise_color_keyword`) then noun (`_CN_NOUN_MAP`); join with space;
lowercase. No ML/tokeniser — just dict lookups.

## 4. Data Flow

### 4.1 Happy path — MobilePickSkill auto-detect

```
User: "抓起蓝色瓶子"
  │
  ▼
LLM (Haiku) → MobilePickSkill(object_label="蓝色瓶子")
  │
  ▼
_resolve_target(world_model)                    [world_model = {}]
  │  None
  ▼
label_to_en_query("蓝色瓶子")          →       "blue bottle"
  │
  ▼
DetectSkill.execute(query="blue bottle", ctx)
  │
  ├─> ctx.perception.detect("blue bottle")
  │       │
  │       ▼
  │   Go2Perception._vlm.detect(rgb_frame, "blue bottle")
  │       │
  │       ▼
  │   QwenVLMDetector.detect(rgb, "blue bottle")
  │       │
  │       ▼
  │   POST openrouter.ai/api/v1/chat/completions
  │   (qwen/qwen2.5-vl-72b-instruct, image + JSON-schema prompt)
  │       │
  │       ▼
  │   [{"label":"blue bottle","bbox":[122,88,158,136],"confidence":0.91}]
  │       │
  │       ▼
  │   list[Detection(label="blue bottle", bbox=(122,88,158,136), confidence=0.91)]
  │
  ├─> ctx.perception.track(detections)
  │       │
  │       ▼
  │   Go2Perception._project_bbox_to_camera_frame
  │     bbox_centre (140, 112), bbox depth median ≈ 1.08 m
  │     pixel_to_camera(140, 112, 1.08, intr) → cam_xyz (−0.082, −0.033, 1.08)
  │       │
  │       ▼
  │   list[TrackedObject(pose=Pose3D(-0.082, -0.033, 1.08))]
  │
  ├─> for each (det, tr):
  │     cam_xyz = [-0.082, -0.033, 1.08]
  │     ctx.calibration.camera_to_base(cam_xyz)
  │       │
  │       ▼
  │     base.get_camera_pose()  →  (cam_xpos, cam_xmat)
  │       │
  │       ▼
  │     world_xyz ≈ (1.26, 0.04, 0.28)
  │       │
  │       ▼
  │     world_model.add_object(ObjectState(
  │         object_id="blue_bottle_0", label="blue bottle",
  │         x=1.26, y=0.04, z=0.28, ...))
  │
  ▼
MobilePickSkill retry _resolve_target           [hit: blue_bottle_0]
  │
  ▼
compute_approach_pose + navigate_to + wait_stable + PickTopDownSkill
  │
  ▼
gripper holding, held=True
```

### 4.2 Error path — VLM returns empty

```
QwenVLMDetector.detect → []
  ▼
DetectSkill: count=0, diagnosis="no_detections", success=True
  ▼
MobilePickSkill: det_result.result_data.count == 0 → target still None
  ▼
return object_not_found with empty known_objects  (existing UX)
```

### 4.3 Error path — Qwen HTTP 503

```
QwenVLMDetector.detect
  → retry 1: 503 → backoff via httpx default → retry 2: 503
  → RuntimeError("VLM API failed after 2 attempts")
  ▼
Go2Perception.detect propagates RuntimeError
  ▼
DetectSkill.execute catches → success=False, diagnosis="no_perception"
  ▼
MobilePickSkill.execute: det_result.success == False → target still None
  ▼
return object_not_found (graceful)
```

## 5. Directory Structure

```
vector_os_nano/
├── perception/
│   ├── __init__.py            # (unchanged)
│   ├── base.py                # PerceptionProtocol (unchanged)
│   ├── vlm.py                 # Moondream (unchanged)
│   ├── vlm_go2.py             # Gemma/GPT — describe/identify (unchanged)
│   ├── vlm_qwen.py            # NEW — grounded detect via Qwen2.5-VL
│   ├── pipeline.py            # SO-101 PerceptionPipeline (unchanged)
│   ├── go2_perception.py      # NEW — PerceptionProtocol for Go2
│   ├── go2_calibration.py     # NEW — pose-driven Go2 calibration
│   ├── calibration.py         # SO-101 Calibration (unchanged)
│   ├── depth_projection.py    # (unchanged — reused)
│   ├── pointcloud.py          # (unchanged)
│   └── ...
├── skills/
│   ├── detect.py              # (unchanged — finally lives on Go2)
│   ├── mobile_pick.py         # MODIFIED — insert auto-detect retry
│   └── utils/
│       └── __init__.py        # MODIFIED — add label_to_en_query
├── vcli/tools/
│   └── sim_tool.py            # MODIFIED — wire _perception + _calibration
└── ...
tests/
├── unit/
│   └── perception/
│       ├── test_vlm_qwen.py            # NEW
│       ├── test_go2_perception.py      # NEW
│       └── test_go2_calibration.py     # NEW
│   └── skills/
│       └── test_mobile_pick.py         # EXTEND — auto-detect cases
│   └── utils/
│       └── test_label_to_en_query.py   # NEW
├── integration/
│   └── test_sim_tool_perception_wire.py  # NEW
scripts/
└── verify_perception_pick.py  # NEW — E2E with --dry-run
```

## 6. Key Implementation Details

### 6.1 Qwen grounding prompt

```
SYSTEM (role=user, combined with image):

"You are a vision grounding model. Find all instances of the requested
object in this image. Return ONLY a JSON array — no prose, no
markdown fences. Use pixel coordinates (0..width, 0..height).

Schema: [{\"label\": string, \"bbox\": [x1, y1, x2, y2],
         \"confidence\": 0.0..1.0}]

Return [] if no matching object is found.

Query: <USER_QUERY>"
```

We request pixel coordinates (not normalised). Parser auto-scales if
`max(bbox) ≤ 1.0` (Qwen occasionally returns normalised despite
instruction). Confidence clamped to `[0, 1]`.

### 6.2 Cost per call (Qwen2.5-VL-72B on OpenRouter, 2026-04)

```
input:  $0.40 / 1M tokens   →  _COST_PER_INPUT_TOKEN  = 0.40e-6
output: $1.20 / 1M tokens   →  _COST_PER_OUTPUT_TOKEN = 1.20e-6
typical 160-px thumbnail + schema prompt:  ~900 input / 60 output
typical call cost:  ~$0.00043
```

Tracked via `QwenVLMDetector.cumulative_cost_usd` (lock-guarded) — same
pattern as `Go2VLMPerception`. Surfaced to logs at INFO level per call.

### 6.3 Calibration geometry check (guards against MJCF drift)

On `Go2Calibration` `__init__`, we do NOT eagerly read camera pose
(it requires odometry to be flowing). Instead, the first call to
`camera_to_base` reads it. One caveat: if `get_camera_pose()` returns
the default stub (dog at `(0, 0, 0.28)`, heading 0) because no odometry
arrived, the transform still runs but yields a pose in the "dog at
origin" frame — which for an un-moved dog IS world. The bridge
publishes odometry at 20 Hz; by the time perception runs, odom is live.

### 6.4 `label_to_en_query` worked examples

| Input | Output | Notes |
|---|---|---|
| `"蓝色瓶子"` | `"blue bottle"` | color + noun mapped |
| `"红色的杯子"` | `"red cup"` | "的" stripped |
| `"bottle"` | `"bottle"` | pass-through |
| `""` | `None` | caller skips detect |
| `None` | `None` | same |
| `"奇怪的东西"` | `"奇怪的东西"` | no CN map hit; VLM gets raw string (may still work for "thing"/"object") |
| `"blue 瓶子"` | `"blue bottle"` | mixed OK |
| `"all objects"` | `"all objects"` | DetectSkill passthrough |

## 7. Test Strategy

### 7.1 Tools

| Layer | Tool | Mock strategy |
|---|---|---|
| Unit — Qwen client | pytest + monkey-patch `httpx.Client.post` | Return stubbed responses per test case |
| Unit — Go2Perception | pytest + fake `camera` object | Inject `get_camera_frame` / `get_depth_frame` returning numpy arrays |
| Unit — Go2Calibration | pytest + fake `base_proxy` | Inject `get_camera_pose()` returning synthetic (pos, xmat) |
| Unit — mobile_pick | pytest + fake SkillContext | Stub `perception.detect` / `.track`, stub world_model |
| Unit — utils | pytest | Pure function calls |
| Integration | pytest + real `rclpy.init` + monkey-patched Go2ROS2Proxy | No real bridge; verify wire-up only |
| E2E dry-run | pytest-less Python script; env flag `VECTOR_PERCEPTION_DRYRUN=1` | Monkey-patches QwenVLMDetector in the test harness |

### 7.2 Coverage targets per module

| Module | Target | Rationale |
|---|---|---|
| `vlm_qwen.py` | ≥ 95 % | External API boundary — must be fully mocked; all branches reachable |
| `go2_perception.py` | ≥ 95 % | Central to perception contract |
| `go2_calibration.py` | ≥ 98 % | Pure math — should be exhaustive |
| `skills/utils label_to_en_query` | 100 % | Tiny pure function |
| `skills/mobile_pick.py` (delta) | ≥ 90 % on new lines | Existing coverage preserved |
| `vcli/tools/sim_tool.py` (delta) | Integration test only | Wire-up is a one-off mutation |

### 7.3 Test layer → spec AC map

| AC | Layer | File(s) |
|---|---|---|
| AC-1 | Integration | `test_sim_tool_perception_wire.py` |
| AC-2 | Unit (mocked) + live smoke | `test_vlm_qwen.py`, REPL smoke |
| AC-3 | Unit | `test_go2_perception.py` |
| AC-4 | Unit | `test_go2_calibration.py` |
| AC-5 | Live smoke (dry-run alt) | `verify_perception_pick.py --dry-run` |
| AC-6 | Unit | `test_mobile_pick.py` (new cases) |
| AC-7 | Unit | `test_mobile_pick.py` (preserved) |
| AC-8 | coverage CI | pytest --cov |
| AC-9 | Live REPL (Yusen) | manual — post-merge |
| AC-10 | CI | pytest -q |

## 8. Risks & Mitigations

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | Qwen returns wrong bbox (label drift) | Wrong object grasped | `DetectSkill` stores by label — LLM can observe via VGG and re-plan; graceful degradation documented in Scenario 4 |
| R2 | Qwen latency 2–4 s blocks pick flow | UX lag | Keep 160-px thumbnail (existing tuning); log `elapsed` per call so Yusen can see |
| R3 | Camera pose in `Go2ROS2Proxy.get_camera_pose()` desyncs from MJCF | 3D position offset error ~10 cm | `test_go2_calibration.py` asserts geometry against hard-coded MJCF numbers (`0.3 fwd`, `0.05 up`, `-5° pitch`); CI fails if those drift |
| R4 | Qwen returns normalized bbox against instruction | bbox coords off by factor W/H | Parser auto-scales if `max(bbox) ≤ 1.0` — test `test_qwen_detect_normalised_bbox_scaled_to_pixels` |
| R5 | OpenRouter 429 rate limit during testing | CI flakiness | All VLM calls mocked in unit + integration; only E2E dry-run mocks VLM too; live smoke runs once per session |
| R6 | `MobilePickSkill` auto-detect recurses (DetectSkill calls back into perception which re-enters mobile_pick?) | Infinite loop | DetectSkill writes world_model directly, does NOT invoke MobilePickSkill; loop impossible by construction |
| R7 | `agent._perception` set but `agent._calibration` is None | Wrong-frame coords in world_model | `_start_go2` guard pairs them (both set or both None); `mobile_pick` retry also guards `context.calibration is not None` |
| R8 | Qwen cost drift (pricing change on OpenRouter) | Billing surprise | Log per-call cost; Yusen sees cumulative in REPL post-session |
| R9 | Bbox spans entire frame (Qwen hallucinates "blue bottle" as whole scene) | Depth projection gets wall/floor | Add bbox-area sanity check: if `bbox_area > 0.5 * W*H` → treat as failure (skipped for v2.3 unless causing grief; flag as debt for v2.4) |
| R10 | CN label with no noun-map hit | VLM gets CN query | Graceful — Qwen2.5-VL is multilingual; test with one CN-only query in `test_label_to_en_query` |

## 9. Execution Plan (for task.md)

Tentative wave structure (Phase 3 will finalise):

| Wave | Tasks | Parallelism | Agent | Gate |
|---|---|---|---|---|
| W1 | T1: QwenVLMDetector + tests; T2: Go2Calibration + tests; T3: label_to_en_query + tests | 3 independent | Alpha, Beta, Gamma | unit green |
| W2 | T4: Go2Perception + tests (uses T1 via composition, but unit test stubs vlm) | 1 | Alpha | unit green |
| W3 | T5: sim_tool wire-up + integration test; T6: mobile_pick auto-retry + tests | 2 independent after W2 | Beta, Gamma | unit+integration green |
| W4 | T7: verify_perception_pick.py E2E dry-run; T8: docs (v2.3_live_repl_checklist.md, progress.md, status.md update) | 2 independent | Alpha, Scribe | dry-run pass + docs clean |
| QA | code-review + security-review in parallel | 2 | reviewers | 0 critical/high |

## 10. Known debt (carry-forward, not addressed in v2.3)

- `VECTOR_SHARED_EXECUTOR=0` rollback path still leaks spin thread (low
  priority — only triggers when flag explicitly set).
- Divergent `_wait_stable` impls in `mobile_pick` vs `mobile_place` —
  extract to `skills/utils/mobile_helpers.py` in v2.4.
- VGG goal_decomposer sometimes emits `sub_goal` names not matching any
  registered skill (e.g. `"approach_object"`) → crash. Needs strategy
  whitelist in decomposer — separate SDD when raised.
- `bbox too large → whole frame` guard (R9) — defer to v2.4 if
  observed.

---

## Dispatcher Executive Summary (for Yusen async review)

### One-liner
v2.3 Plan: 3 new pure-Python modules + 3 small edits; zero new ROS2
interfaces / zero new dependencies; 4 waves under 8 atomic tasks; all
decisions match approved spec.

### Key Decisions made by plan (none escalated)
- Qwen grounding style: JSON schema instruction (not `<ref>/<box>`)
- Single-shot track (no temporal tracker) — spec NG2 confirmed
- Auto-detect retry inside MobilePickSkill via DetectSkill delegation —
  no new recursion risk
- Pose-driven calibration (not point-fitted) — reads live camera pose
- `_label_to_en_query` CN→EN helper sits in `skills/utils`

### Impact
- Nodes added/modified: 0 (no new ROS2 nodes; no message changes)
- Topics/services affected: 0 (consumes existing `/camera/image`,
  `/camera/depth`)
- Cross-platform impact: No (purely additive for Go2 sim)

### Risks
- Wrong bbox from VLM → graceful; LLM re-plans
- Calibration drift from MJCF edit → unit tests detect via hard-coded
  geometry assertions
- OpenRouter rate limit → all test paths mocked

### Status
- **spec.md**: approved ✓
- **plan.md**: draft ready for your review
- **Next**: if you approve async, I proceed to Phase 3 (`task.md`).
  If anything looks off, one word "改" + what and I'll revise.
