# v2.4 Perception Overhaul — Technical Plan

**Status**: DRAFT (non-architectural — CEO review async)
**Depends on**: spec.md approval (blocking for implementation)
**Branch**: `feat/v2.0-vectorengine-unification`

---

## 1. Architecture Overview

```
                       ┌─────────────────────────────┐
  go2_room.xml  ──▶────│  MuJoCo + ROS2 Bridge       │ subprocess
  (GSO meshes)         │  /camera/image  /camera/depth│
                       │  /piper/*        /cmd_vel    │
                       └──────────────┬──────────────┘
                                      │ ROS2 topics
                       ┌──────────────▼──────────────┐
  Go2ROS2Proxy  ◀──────│  main process (vector-cli)   │
  (camera, odom,       │                             │
   get_camera_pose)    │                             │
                       └──────────────┬──────────────┘
                                      │ duck-typed camera
                       ┌──────────────▼──────────────┐
       Go2Perception ─▶│ detect(q)   → YoloeDetector │  (ultralytics)
                       │ track(dets) → Sam3Segmenter │  (ultralytics)
                       │             → mask∩depth    │
                       │             → pointcloud    │  (numpy/open3d)
                       │             → sanity gates  │
                       │             → centroid      │
                       └──────────────┬──────────────┘
                                      │ Pose3D (camera frame)
                       ┌──────────────▼──────────────┐
   Go2Calibration ─────│ camera_to_world             │  (xmat FIXED)
                       └──────────────┬──────────────┘
                                      │ world xyz
                       ┌──────────────▼──────────────┐
                       │ DetectSkill → world_model   │
                       │ MobilePickSkill → navigate  │
                       │ PickTopDownSkill → grasp    │
                       └─────────────────────────────┘
```

Isolation boundaries (preserve from v2.3):
- Perception package **never** imports ROS2 or MuJoCo.
- YOLOE + SAM 3 loaded lazily on first `detect` / `track` call so
  unit tests can collect without GPU.
- Skills still access perception via `context.perception` only —
  no direct YOLOE/SAM imports in skills.

## 2. Technical Decisions

| # | Decision | Choice | Reason |
|---|---|---|---|
| D1 | Dep stack | `ultralytics ≥ 8.3.237` only | Single package covers YOLOE **and** SAM 3; bundles torch load |
| D2 | Detector | YOLOE-11S `yoloe-11s-seg.pt` | Smallest / fastest; ~5 ms inference on RTX 5080 at 640 px |
| D3 | Segmenter primary | SAM 3 via `SAM3SemanticPredictor` | Yusen direction; concept-aware; mask at native resolution |
| D4 | Segmenter fallback | SAM 2.1 via `SAM("sam2.pt")` | No HF access gate; same predictor pattern; ~80 ms |
| D5 | HF access risk | Runtime probe + graceful degrade | If `sam3.pt` absent → fall back to SAM 2.1 automatically with WARN log. |
| D6 | Pointcloud lib | `open3d` optional, `numpy` fallback | open3d adds 100 MB; numpy is enough for centroid + statistical outlier |
| D7 | Plane removal | RANSAC via numpy (no open3d required) | Avoids hard dep; 10-line implementation |
| D8 | Scene assets | 10 GSO objects, hand-picked | Manual selection from `kevinzakka/mujoco_scanned_objects`: 4 bottles, 3 mugs, 2 cans, 1 small box — all `<5 cm` diameter fits 35 mm Piper jaws |
| D9 | Scene delivery | Add `mjcf/pickable_assets/` with pre-rendered mesh dir + `include` in `go2_room.xml` | Reproducible, no submodule; ~300 MB committed OR via script download |
| D10 | Mesh storage | Git-LFS for meshes/textures | Avoid bloating git history with binary blobs. Can fall back to commit-at-small-size if LFS unavailable. |
| D11 | Qwen fate | Delete `perception/vlm_qwen.py` + tests | No runtime path touches it; describe/visual_query stays via `vlm_go2.Go2VLMPerception` |
| D12 | Test GPU gating | `@pytest.mark.gpu` + env var `VECTOR_GPU_TESTS=1` | Dispatcher/CI skip by default; Yusen's dev box enables for smoke |
| D13 | xmat fix | `right = (sin h, -cos h, 0)`; `up = cross(right, fwd)` | REP-103 compliant — right is -Y at heading 0 |
| D14 | Sanity gate defaults | See spec §7.5 | Conservative; prevents phantom goals but won't reject valid on-table objects |

## 3. Module Design

### 3.1 `perception/detectors/yoloe_detector.py` (NEW, ~130 LoC)

```python
from ultralytics import YOLOE

class YoloeDetector:
    """Open-vocabulary 2D detection via YOLOE."""

    def __init__(
        self,
        model_name: str = "yoloe-11s-seg.pt",
        confidence_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        self._model = YOLOE(model_name)
        self._device = device or ("cuda:0" if _cuda_available() else "cpu")
        self._conf = confidence_threshold
        self._last_classes: tuple[str, ...] = ()

    def detect(self, image: np.ndarray, query: str) -> list[Detection]:
        classes = _classes_from_query(query)      # e.g. "blue bottle" -> ("bottle",)
        if classes != self._last_classes:
            self._model.set_classes(list(classes))
            self._last_classes = classes
        results = self._model.predict(
            image, conf=self._conf, device=self._device, verbose=False,
        )
        return _results_to_detections(results, query)
```

Helper `_classes_from_query`: keyword → known class list. First pass
can be dictionary-based (bottle / cup / can / mug / bowl / box / toy).
Fine-grained "blue bottle" uses downstream **colour filter on mask RGB
statistics** — don't rely on YOLOE for colour distinctions.

### 3.2 `perception/segmenters/sam3_segmenter.py` (NEW, ~160 LoC)

```python
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics import SAM

class Sam3Segmenter:
    """Promptable segmentation via SAM 3 with SAM 2.1 fallback."""

    def __init__(
        self,
        primary_model: str = "sam3.pt",
        fallback_model: str = "sam2.pt",
        device: str | None = None,
    ) -> None:
        self._device = device or _auto_device()
        self._predictor = self._build_predictor(primary_model, fallback_model)

    def _build_predictor(self, primary, fallback):
        if _weights_available(primary):
            return SAM3SemanticPredictor(overrides=dict(
                conf=0.25, task="segment", mode="predict",
                model=primary, half=True,
            )), "sam3"
        logger.warning(
            "SAM 3 weights '%s' not found — falling back to SAM 2.1 '%s'",
            primary, fallback,
        )
        return SAM(fallback), "sam2"

    def segment(
        self,
        image: np.ndarray,
        prompts: list[tuple[float, float, float, float]],
    ) -> list[np.ndarray]:
        # Dispatches to SAM 3 or SAM 2.1 with a uniform (masks, bboxes)
        # return contract, upscales masks if predictor downsamples.
        ...
```

### 3.3 `perception/pointcloud_projection.py` (NEW, ~120 LoC)

```python
def mask_depth_to_pointcloud(
    mask: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics,
    depth_min: float = 0.15, depth_max: float = 5.0,
    statistical_k: int = 20, statistical_std: float = 2.0,
) -> np.ndarray:
    """
    For each True pixel in mask with valid depth, unproject to 3D
    in camera frame. Apply statistical outlier removal on the Z axis.
    """
    ys, xs = np.where(mask & (depth > depth_min) & (depth < depth_max))
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    z = depth[ys, xs].astype(np.float64)
    x = (xs - intrinsics.cx) * z / intrinsics.fx
    y = (ys - intrinsics.cy) * z / intrinsics.fy
    pts = np.stack([x, y, z], axis=1)
    return _statistical_outlier_filter(pts, statistical_k, statistical_std)


def pointcloud_centroid(pts: np.ndarray, method: str = "median") -> np.ndarray:
    if method == "median":
        return np.median(pts, axis=0)
    if method == "mean":
        return pts.mean(axis=0)
    if method == "plane_top":
        return _top_of_plane_centroid(pts)
    raise ValueError(method)
```

### 3.4 `perception/sanity_gates.py` (NEW, ~80 LoC)

```python
@dataclass(frozen=True)
class SanityGateConfig:
    depth_range: tuple[float, float] = (0.15, 5.0)
    height_range: tuple[float, float] = (-0.2, 1.5)
    mask_area_range: tuple[int, int] = (200, 50_000)
    min_pointcloud_points: int = 50


def apply_sanity_gates(
    mask_area_px: int, pointcloud: np.ndarray,
    centroid_camera: np.ndarray, centroid_world: np.ndarray | None,
    config: SanityGateConfig,
) -> tuple[bool, str]:
    """Return (accepted, reason). reason == '' when accepted."""
    if pointcloud.shape[0] < config.min_pointcloud_points:
        return False, f"pointcloud_sparse({pointcloud.shape[0]})"
    depth = float(centroid_camera[2])
    if not (config.depth_range[0] <= depth <= config.depth_range[1]):
        return False, f"depth_out_of_range({depth:.2f})"
    if centroid_world is not None:
        z_w = float(centroid_world[2])
        if not (config.height_range[0] <= z_w <= config.height_range[1]):
            return False, f"height_out_of_range({z_w:.2f})"
    if not (config.mask_area_range[0] <= mask_area_px <= config.mask_area_range[1]):
        return False, f"mask_area_out_of_range({mask_area_px})"
    return True, ""
```

### 3.5 `perception/go2_perception.py` (MODIFIED, ~280 LoC)

Constructor now accepts `detector`, `segmenter`, `gates`, `calibration`.

`detect(query)` — call `detector.detect(rgb, query)` → `list[Detection]`.

`track(detections)`:
1. `masks = segmenter.segment(rgb, [d.bbox for d in detections])`
2. For each `(det, mask)`:
   - `pts = mask_depth_to_pointcloud(mask, depth, intrinsics)`
   - `centroid_cam = pointcloud_centroid(pts)` (or None)
   - `centroid_world = calibration.camera_to_world(centroid_cam)` (only
      if calibration provided)
   - `ok, reason = apply_sanity_gates(...)`
   - Structured log: `[go2_perception] label=… area=… depth=… world=… gate=…`
   - Emit `TrackedObject(pose=Pose3D(...) if ok else None, mask=mask)`
3. Return list.

`get_point_cloud(mask)` — implemented via `mask_depth_to_pointcloud`.

`caption/visual_query` — delegate to `Go2VLMPerception` when present;
else `NotImplementedError`.

### 3.6 `hardware/sim/go2_ros2_proxy.py` (xmat fix)

Change two lines in `get_camera_pose`:
```python
right = np.array([sin_h, -cos_h, 0.0])     # was (-sin_h, cos_h, 0) — ROS-LEFT
up = np.cross(right, fwd)                    # was cross(fwd, right)
```
Update existing `test_go2_camera_pose.py` expected values. Math
reconciled with `depth_projection.camera_to_world` in same commit.

### 3.7 `vcli/tools/sim_tool.py` (wire-up)

```python
def _wire_go2_perception(agent, base_proxy):
    try:
        from vector_os_nano.perception.detectors.yoloe_detector import YoloeDetector
        from vector_os_nano.perception.segmenters.sam3_segmenter import Sam3Segmenter
        from vector_os_nano.perception.sanity_gates import SanityGateConfig
        from vector_os_nano.perception.go2_perception import Go2Perception
        from vector_os_nano.perception.go2_calibration import Go2Calibration

        det = YoloeDetector()
        seg = Sam3Segmenter()
        cal = Go2Calibration(base_proxy)
        agent._perception = Go2Perception(
            camera=base_proxy, detector=det, segmenter=seg,
            calibration=cal, gates=SanityGateConfig(),
        )
        agent._calibration = cal
        logger.info("[sim_tool] Go2Perception wired: detector=YOLOE, segmenter=%s",
                    seg.backend_name)
    except Exception as exc:
        logger.warning("[sim_tool] perception wire-up failed: %s", exc)
        agent._perception = None
        agent._calibration = None
```

Fallback: if `ultralytics` import fails at boot, log and continue with
`agent._perception = None` (v2.3 behaviour). DetectSkill will still
return `no_perception`.

### 3.8 `skills/detect.py` (CLI summary)

Add human-readable summary to `result_data`:
```python
result_data["summary"] = (
    f"{len(object_summaries)} detected: "
    + ", ".join(f"{o['label']}@world({o['position_cm']})"
                for o in object_summaries if o.get("has_3d"))
)
```
`skill_wrapper.py` already appends `result_data` to tool output —
the `summary` key will surface to the CLI/LLM naturally.

### 3.9 `skills/mobile_pick.py` (distance warning)

```python
if nav_dist > 3.0:
    logger.warning(
        "[MOBILE-PICK] far approach nav_dist=%.2f m — if target "
        "location untrusted, detect first", nav_dist,
    )
```

### 3.10 `hardware/sim/go2_room.xml` (scene swap)

1. Add `<asset>` block including `pickable_assets/*/model.xml`
   (compiler inherits textures/meshes).
2. Replace 3 `pickable_bottle_*` / `pickable_can_*` bodies with
   10 `pickable_<gso_id>` bodies arranged on the pick table.
3. Keep `pickable_*` naming for auto-registration in
   `MuJoCoGo2.connect`.

### 3.11 Removed: `perception/vlm_qwen.py` + tests

Delete `vector_os_nano/perception/vlm_qwen.py` (~400 LoC) and
`tests/unit/perception/test_vlm_qwen.py` (~20 tests). No other
module references it (verified via grep in v2.3 code review).

## 4. Data Flow

```
RGB 640×480 ────▶ YoloeDetector.detect("blue bottle")
                     │
                     ▼
                 list[Detection] (bbox, confidence)
                     │
                     ▼
                 Sam3Segmenter.segment(RGB, [bbox])
                     │
                     ▼
                 list[np.ndarray mask]   ─ each H×W bool
                     │
                     ▼
  depth 640×480 ◀──  mask ∩ depth
                     │
                     ▼
                 list[pointcloud (N,3)]
                     │
                     ▼  sanity_gates
                     │
                     ▼
                 TrackedObject(label, bbox_2d, pose, mask,
                               confidence)   -- pose=None if gated
                     │
                     ▼
                 DetectSkill iterates → calibration.camera_to_world
                     │
                     ▼
                 world_model.add_object(ObjectState)
                     │
                     ▼
                 MobilePickSkill._resolve_target retry → hit
```

## 5. Directory Structure

```
vector_os_nano/perception/
├── base.py                   (PerceptionProtocol — unchanged)
├── depth_projection.py       (unchanged; camera_to_world path)
├── go2_calibration.py        (unchanged; xmat reads come from proxy)
├── go2_perception.py         (MODIFIED — YOLOE+SAM3 backend)
├── vlm_go2.py                (unchanged; caption/visual_query)
├── vlm_qwen.py               (DELETED)
├── detectors/
│   ├── __init__.py           (NEW)
│   ├── base.py               (NEW — OpenVocabDetector protocol)
│   └── yoloe_detector.py     (NEW)
├── segmenters/
│   ├── __init__.py           (NEW)
│   └── sam3_segmenter.py     (NEW)
├── pointcloud_projection.py  (NEW)
└── sanity_gates.py           (NEW)

vector_os_nano/hardware/sim/mjcf/pickable_assets/  (NEW, Git-LFS)
├── BOTTLE_BLUE/
│   ├── model.xml
│   ├── model.obj
│   ├── model_collision_*.obj
│   └── texture.png
└── …  (9 more objects)
```

## 6. Key Implementation Details

### 6.1 YOLOE class-set caching

`set_classes` is expensive (re-compiles CLIP text embeddings). Cache
last classes; no-op on identical query. For multi-word queries, pick
a single head noun — `_classes_from_query` uses a small mapping:

```python
_KEYWORD_MAP = {
    "bottle": "bottle", "瓶子": "bottle",
    "cup": "cup", "杯子": "cup", "mug": "mug", "马克杯": "mug",
    "can": "can", "罐": "can", "bowl": "bowl", "碗": "bowl",
    "box": "box", "盒子": "box", "toy": "toy", "玩具": "toy",
    "all objects": "*", "everything": "*", "所有": "*",
}
```
`"*"` triggers the prompt-free YOLOE-26L-PF model with ≥1200 built-in
classes — delivered lazily to avoid loading both unless needed.

### 6.2 SAM 3 bbox prompt format

SAM 3 `SAM3SemanticPredictor` takes a list of bboxes in `(x1, y1, x2,
y2)` pixel coords. Image set via `predictor.set_image(img)` per
frame. Multiple bboxes in one call.

### 6.3 Mask colour filter (for "blue bottle" subquery)

When query contains a colour keyword:
```python
_COLOUR_HSV = {
    "blue": ((100, 80, 60), (130, 255, 255)),
    "green": ((40, 80, 60), (80, 255, 255)),
    "red": ((0, 80, 60), (10, 255, 255)),       # plus wrap-around
    # …
}

def mask_colour_fraction(rgb, mask, colour):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower, upper = _COLOUR_HSV[colour]
    colour_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    return (colour_mask & mask.astype(np.uint8)).sum() / max(1, mask.sum())
```
Reject tracked objects with fraction `< 0.3` when a colour keyword
is present. Zero cost when query is colour-less.

### 6.4 Statistical outlier removal (numpy-only)

```python
def _statistical_outlier_filter(pts, k=20, std=2.0):
    if pts.shape[0] < k + 1:
        return pts
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1)        # k+1 because self
    mean_d = d[:, 1:].mean(axis=1)
    thr = mean_d.mean() + std * mean_d.std()
    return pts[mean_d < thr]
```
`scipy` is already an indirect dep via `numpy 2.4` ecosystem — confirm
in `pyproject.toml` before use; if not present, add.

### 6.5 SAM3 access failure mode

At wire-up, `Sam3Segmenter.__init__` checks `Path.home() /
".cache/ultralytics/sam3.pt"` and `./sam3.pt`. On miss, attempts
`SAM("sam2.pt")` which auto-downloads (SAM 2 is open). Logs
`[WARNING] Falling back to SAM 2.1 — request SAM 3 HF access for
improved accuracy`.

### 6.6 GSO asset management

**Option A (committed meshes)**: Git-LFS tracks `*.obj` + `*.png`.
Pros: reproducible clone. Cons: LFS adds friction.

**Option B (download script)**: `scripts/fetch_gso_assets.sh` that
rsyncs selected meshes from a GitHub release tarball at first run.
Pros: no LFS. Cons: needs network on first setup.

Plan defaults to **A** for reliability. If Yusen or CI reports LFS
failures, switch to B.

### 6.7 Object selection for v2.4 scene

Manual curation — 10 objects, all compatible with 35 mm Piper jaws:

| Category | GSO ID candidates | Target diameter |
|---|---|---|
| Bottle (tall) | `Big_Bulldog_Action_Figure_Bottle`, `Rexy_Dinosaur_Bottle`, … | 4-5 cm |
| Can (short) | `Diet_Coke_Can`, `Tomato_Can`, `Sprayaway_Can` | 4.5 cm |
| Mug | `Starbucks_Espresso_Mug`, `Cooking_Tea_Mug` | 7-8 cm (handle side) |
| Small box | `Crayola_Box`, `Pokemon_Card_Box` | 5 cm |
| Toy | `Yellow_Duck`, `Dinosaur_Toy` | 5-6 cm |

Scribe agent produces final selection list after browsing the repo;
plan locks 10 slots.

## 7. Test Strategy

### Unit tests (target: ~32 new tests)

| File | Count | Key scenarios |
|---|---|---|
| `test_yoloe_detector.py` | 6 | init, class cache hit, detect empty, detect with box, confidence filter, device fallback |
| `test_sam3_segmenter.py` | 6 | SAM 3 path mock, SAM 2.1 fallback path, empty prompts, mask shape, multi-bbox, upscale |
| `test_pointcloud_projection.py` | 8 | simple cube, empty mask, all invalid depth, plane centroid, outlier removal, median vs mean, depth range clip, intrinsics correctness |
| `test_sanity_gates.py` | 6 | each gate rejection, accepted path, config override, world z None, mask area extremes, sparse pc |
| `test_go2_perception_v24.py` | 6 | new detect + track happy path, sanity gate rejects, mock segmenter fallback, logs structured entry, get_point_cloud works, calibration absent path |

### Regression tests

| File | Action |
|---|---|
| `test_go2_camera_pose.py` | Update expected `right` / `up` vectors to REP-103 convention |
| `tests/integration/test_sim_tool_wireup.py` | Assert perception is Go2Perception v2 instance |
| `tests/integration/test_mobile_pick_autodetect.py` | Assert sanity gate rejection → object_not_found, no phantom nav |

### E2E dry-run

`scripts/verify_perception_pick.py --dry-run` updated to use
mocked YoloeDetector + Sam3Segmenter returning crafted bboxes and
masks. Exit 0 within 1 s.

### GPU smoke (manual)

`scripts/debug_perception_live.py`:
- Attach to running bridge, capture 1 RGB + depth frame.
- Run YOLOE (`bottle`) → print bboxes + confidences.
- Run SAM 3 on each → save masks to `/tmp/perception_debug/`.
- Project each mask → print pointcloud stats and sanity gate results.
- Write HTML report with side-by-side RGB / mask / depth overlays.

### CI gating

- `@pytest.mark.gpu` for tests touching real weights.
- Env var `VECTOR_GPU_TESTS` selects.
- Default CI pytest excludes `gpu` mark.

## 8. Risks & Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | SAM 3 HF access denied / delayed | Medium | Medium | Automatic SAM 2.1 fallback (D5). Tested as primary path in CI. |
| R2 | ultralytics SAM 3 API changes before stable release | Low | High | Pin `ultralytics==8.3.237` initially; upgrade deliberately. |
| R3 | YOLOE small models miss small objects (<3 cm) | Medium | Medium | Use YOLOE-11M for scene with small parts if S misses; track via live smoke. |
| R4 | GSO meshes too complex for MuJoCo contact solver | Medium | Medium | V-HACD collision already pre-computed; test scene load time <5 s; reduce `solimp` if contacts misbehave. |
| R5 | Git-LFS friction (CEO / CI clone slower) | Low | Low | Fallback to download script (D6 Option B). |
| R6 | Colour filter (§6.3) too aggressive → valid bottles rejected | Medium | Low | Threshold 0.3 tuneable; fallback: skip filter if `mask.sum()<500`. |
| R7 | Qwen removal breaks a forgotten call site | Low | Medium | Grep check in task T1 + ruff unused-imports. |
| R8 | VRAM exhausted with YOLOE + SAM 3 + MuJoCo | Low | High | YOLOE-S + SAM 3 base + `half=True` <6 GB measured on similar stacks; monitor with `nvidia-smi` during smoke. |
| R9 | Sanity gate false-rejects the legitimate target when depth spotty on shiny surface | Medium | Medium | Pointcloud density threshold 50 is conservative — tune after live smoke; log rejects for audit. |
| R10 | New `_classes_from_query` misses user's language | Medium | Low | Fallback: if no mapping, send `"*"` (prompt-free YOLOE) and let SAM 3 / colour filter disambiguate. |

## 9. Execution Plan (for task.md)

Proposed wave structure — respects serial subagent discipline from
`feedback_no_parallel_agents.md` (narrow pytest, MuJoCo-forbidden
imports in subagent prompts).

| Wave | Task | Agent | Depends | Output |
|---|---|---|---|---|
| **W0** | T0 env probe: confirm ultralytics install, SAM3 access, GSO repo availability | dispatcher | — | probe log |
| **W1** | T1 YoloeDetector + base protocol + tests | Alpha | T0 | `yoloe_detector.py`, 6 tests |
| **W1** | T2 Sam3Segmenter with SAM 2 fallback + tests | Beta | T0 | `sam3_segmenter.py`, 6 tests |
| **W1** | T3 pointcloud_projection + sanity_gates + tests | Gamma | T0 | 2 modules, 14 tests |
| **W2** | T4 Go2Perception v2 rewrite + tests | Alpha | T1, T2, T3 | `go2_perception.py`, 6 tests |
| **W2** | T5 xmat fix + regression test update | Beta | — | `go2_ros2_proxy.py` + `test_go2_camera_pose.py` |
| **W2** | T6 Qwen removal + grep check | Gamma | T4 | deleted files |
| **W3** | T7 GSO asset curation + scene XML | Alpha | — | `pickable_assets/`, `go2_room.xml` |
| **W3** | T8 sim_tool wire-up + DetectSkill summary + MobilePick warning | Beta | T4 | updated files |
| **W4** | T9 debug_perception_live.py script | Alpha | T4, T7 | diagnostic tool |
| **W4** | T10 verify_perception_pick.py dry-run update + docs | Gamma | T4, T7 | updated dry-run |
| **W5** | QA: code-reviewer + security-reviewer in parallel | two subagents | all | findings |
| **W6** | Yusen live-REPL smoke; release sign-off | CEO | all | merge / push |

Estimated ~8–10 agent days serial (or ~3 calendar days with parallel
waves where independent). CEO gate at start (spec approval) and end
(release).

## 10. Known debt not addressed

- `VECTOR_SHARED_EXECUTOR=0` spin leak (carry-over v2.3)
- `coverage` / `numpy 2.4` C-tracer conflict — settrace workaround persists
- `_normalise_color_keyword` — now covered by new colour-fraction
  filter; can promote to public API in v2.5
- `_wait_stable` still duplicated — extract to `mobile_helpers.py`
  pending
- VGG literal-string goal bug (`last_seen('blue bottle')['room']`) —
  VGG decomposer scope, not perception
- Real D435 sim-to-real — v3.0
