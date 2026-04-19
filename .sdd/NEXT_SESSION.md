# Next Session — v2.3 Go2 Perception Pipeline (SO-101 pattern)

**Kickoff**: run `/sdd init` in new session.
**Branch**: continue on `feat/v2.0-vectorengine-unification` or cut
`feat/v2.3-go2-perception`.

## Problem

v2.2 ships the loco manipulation skill infrastructure (Ros2Runtime,
PlaceTopDown, MobilePick, MobilePlace, approach_pose) but world_model is
intentionally empty — the **MJCF populate shortcut** that pretended
object positions were known a priori was removed in the cleanup.

Live REPL smoke shows the symptom: `抓个东西` resolves target=None →
clean error listing "Known pickable objects: []" → VGG re-plans to
`detect` → `No perception backend available` because Go2 has no
`context.perception` set.

The fix is to build the **missing perception layer** for Go2 the same
way SO-101 does it.

## SO-101 reference pattern

```
Agent.__init__ constructs PerceptionPipeline(RealSenseCamera,
                                             VLMDetector,
                                             EdgeTAMTracker)
                        ↓ assigned to agent._perception

context.perception.detect(query)    → list[Detection] (2D bboxes)
context.perception.track(detections)→ list[TrackedObject] (3D pose)
context.calibration.camera_to_base  → base-frame xyz
                        ↓
DetectSkill writes ObjectState(x, y, z) to world_model
                        ↓
PickSkill / MobilePickSkill read by label from world_model
```

Go2 has `Go2VLMPerception` (VLM-only, no 3D, no tracking) assigned to
`agent._vlm` — different attribute, different surface. The gap is the
3D-producing layer.

## Spec seeds (for `/sdd init`)

### Goal
Produce a `Go2Perception` + `Go2Calibration` pair that gives Go2 the
same `context.perception.detect() + track()` contract SO-101 has, so
`DetectSkill` works unchanged on Go2 and downstream skills never need
to know which robot produced the `ObjectState`.

### In scope
1. `vector_os_nano/perception/go2_perception.py` — new class
   - `.detect(query: str) → list[Detection]` wraps VLM, returns normalised bboxes
   - `.track(detections) → list[TrackedObject]` uses `Go2ROS2Proxy.get_depth_frame()`
     to deproject bbox centroid into camera frame via camera intrinsics
2. `vector_os_nano/perception/go2_calibration.py` — new class
   - `.camera_to_base(camera_xyz) → base_xyz` static TF from head camera to base_link
   - `.base_to_world(base_xyz, dog_pose) → world_xyz` via odom
3. `vector_os_nano/vcli/tools/sim_tool.py::_start_go2`
   - Construct `Go2Perception` + `Go2Calibration` and assign to
     `agent._perception` / `agent._calibration` when `with_arm=True`
4. `vector_os_nano/skills/mobile_pick.py::_resolve_target`
   - If world_model resolve fails AND `context.perception` available,
     invoke `detect(query_from_label) + track → write ObjectState →
     retry resolve`
5. E2E harness extension — perception path without MJCF pre-populate
6. Live REPL checklist v2.3 — `抓个东西` end-to-end via perception

### Non-goals (future)
- SAM3D per-pixel masks → accurate centroid
- RANSAC grasp pose from pointcloud (not only top-down)
- Multi-view / active perception
- Real Piper hardware driver
- Base-arm coordinated trajectory

### Open questions for CEO at `/sdd init`
1. **VLM model**: keep `google/gemma-4-31b-it` via OpenRouter or switch
   to a local VLM that can return bboxes more reliably?
2. **Depth denoising**: raw `/camera/depth` may have NaN pixels; median
   filter inside bbox or reject outliers?
3. **Calibration source**: extract extrinsics from MJCF programmatically
   or hardcode from URDF (simpler, brittle to scene edits)?
4. **Perception → world_model merge policy**: overwrite on re-detect,
   or maintain confidence-weighted EMA? Existing `WorldModel.add_object`
   already upserts by id — same policy?
5. **Query language**: should `MobilePickSkill` pass `object_label`
   (e.g. "blue bottle") as VLM query, or translate label → query
   heuristically (strip colors, nouns)?

### Known debt to address inside v2.3 or note forward
- Legacy `VECTOR_SHARED_EXECUTOR=0` path leaks spin thread (low risk)
- Divergent `_wait_stable` impls in mobile_pick vs mobile_place
- VGG decomposer: LLM-invented sub_goal names ("approach_object") not
  matching any registered skill → executor returns "No strategy for:
  unmatched" and crashes; decomposer should validate strategy against
  skill registry before returning a GoalTree

## Starting commands

```bash
cd ~/Desktop/vector_os_nano
cat progress.md | head -80
cat agents/devlog/status.md
ls .sdd/                         # v2.2 artefacts retained
/sdd init                        # begin v2.3 spec
```
