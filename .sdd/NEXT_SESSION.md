# Next Session — v2.3.1 Hot-fix SDD (perception accuracy + UX)

**Kickoff**: run `/sdd init` with this doc as input, or jump straight
to `/sdd plan` since the spec below is already drafted.

**Branch**: continue on `feat/v2.0-vectorengine-unification`
(10 v2.3 commits landed; 2 hot-fixes already on top: `ab6167f` +
`543dfd4`).

---

## Context

v2.3 live REPL smoke on 2026-04-20 produced a **partial pass**:

- ✓ `go2sim with_arm=1` wires perception + calibration
- ✓ `[MOBILE-PICK] world_model miss; auto-detect` log fires
- ✓ VGG routes `抓起蓝色瓶子` to `mobile_pick_skill` (alias worked)
- ✗ World-model object location wildly wrong:
  - dog at `(10.287, 3.001)` heading 0 (+X facing)
  - real bottle at `(11.0, 3.0, 0.25)` — 0.7 m directly ahead
  - detected `blue bottle at (16.797, 1.62, 1.367)` — **+5.8 m fwd,
    -1.4 m side, +1.1 m up**
- ✗ `mobile_pick` navigates 6 m to a phantom goal → `far_timeout`
- ✗ CLI shows no textual summary of detected objects (user blind)

The bounding box + depth chain is producing non-physical world
coordinates. The `Go2VLMPerception` helper that VGG uses for verification
even said "I did not find any bounding box detections for blue bottle" —
so the image doesn't clearly contain the target at this viewpoint.

## Root-cause hypotheses (ranked)

| # | Hypothesis | Evidence | Falsification |
|---|-----------|----------|---------------|
| H1 | Qwen VLM hallucinated a bbox over an unrelated distant pixel | vlm_go2 parallel call said "not found"; Qwen returned a labelled bbox | log rgb/depth/bbox side-by-side at detect time |
| H2 | `/camera/depth` unit mismatch (mm vs m, or normalised) | bridge docstring claims float32 m, not live-verified | log numeric depth at bbox centre; expect 0.5-2 m for close objects |
| H3 | xmat[:, 0] = (-sin_h, cos_h, 0) is ROS-left not right; y-side flips | real bottle y=3.0, detected y=1.62 (~1.4 m mirror error) | regression test in test_go2_camera_pose.py |
| H4 | Intrinsics or bbox-scale miscomputed for 320×240 frames | existing dry-run passes with same math | A/B against synthetic frame |

## v2.3.1 Spec (draft — approved 2026-04-20, awaiting implementation)

### MUST (blocking for release)

- **G1 UX exposure**: `DetectSkill.result_data` already has objects list
  with per-object `(label, xyz, has_3d, confidence)`. Surface a concise
  summary into the message stream so LLM sees it and CLI prints it.
  Either:
  - Append a `summary` key and update the skill_wrapper to include
    it in the tool output, OR
  - Format objects into `SkillResult.content` / user-visible field.
- **G2 sanity gate**: `Go2Perception.track` must mark `pose = None`
  when
    - `z ∉ [-0.2, 1.5]` (physical bounds of workspace heights), OR
    - camera-frame distance `> 5.0 m` (out of Go2 D435 reliable
      range / likely hallucination)
  DetectSkill will then record the object with `has_3d=False` and
  not seed world_model with ObjectState.  Prevents phantom-6m
  navigation goals.
- **G3 debug log**: emit one structured INFO line per tracked
  Detection:
  ```
  [go2_perception] detect='blue bottle' bbox=(x1,y1,x2,y2)
    depth_median=1.08 cam_xyz=(-0.08,-0.04,1.08)
    world_xyz=(1.30,-0.04,0.26)
  ```
  Live smoke can grep and diff expected vs actual.
- **G4 LEFT/RIGHT flip**: in `Go2ROS2Proxy.get_camera_pose`,
  `right = (-sin_h, cos_h, 0)` is body-LEFT under ROS REP-103
  (yaw=0 → forward=+X, left=+Y, right=-Y). Swap to
  `right = (sin_h, -cos_h, 0)` and verify:
    - 30/30 top-down pick regression still passes (near-centred
      bottles)
    - new test_go2_camera_pose.py::test_right_pixel_projects_to_dog_right_side
      updated to expect world -Y (not +Y)
  `depth_projection.camera_to_world` fallback path — no change
  needed if it reads xmat from get_camera_pose.

### SHOULD

- **S1 distance warning**: `mobile_pick` warn via log when `nav_dist
  > 3.0 m` — "far approach; if target location untrusted, detect
  first". Not a blocker.
- **S2 all-invalid diagnosis**: `DetectSkill` returns
  `diagnosis="all_3d_invalid"` (not "ok") when every tracked object
  has `has_3d=False`. Signals to caller that perception found
  labels but depth projection failed every one.

### MAY

- **M1 local diagnostic script**: `scripts/debug_perception_live.py`
  attaches to a running bridge, pulls one RGB + depth frame, runs
  Qwen + Go2Perception + Go2Calibration end-to-end, prints all
  intermediate values. Useful for triage on live failure.
- **M2 Bbox-area filter**: if VLM-returned bbox area > 50% of frame,
  likely a hallucination over the whole scene — reject. (R9 from v2.3
  code review deferred; can land here if quick.)

### Non-goals

- Changing the VLM model (still Qwen2.5-VL-72B)
- Multi-frame fusion
- Real hardware
- Bridge-side depth-unit verification (assume float32 metres per
  existing Go2VNavBridge contract)
- Temporal tracker (still v2.4+)

## Planned tasks (for /sdd tasks after spec approval)

| ID | Task | Agent | Wave |
|---|---|---|---|
| T1 | LEFT/RIGHT flip fix + update test_go2_camera_pose + run 30/30 pick regression | Alpha | 1 |
| T2 | Go2Perception.track sanity gate (z range + distance) + unit tests | Beta | 1 |
| T3 | Go2Perception structured debug log + 1 unit test | Gamma | 1 |
| T4 | DetectSkill summary field + skill_wrapper plumbing + 2 tests | Alpha | 2 |
| T5 | mobile_pick distance warning + DetectSkill all_3d_invalid diagnosis + tests | Beta | 2 |
| T6 | scripts/debug_perception_live.py (MAY) | Alpha | 3 |
| T7 | Live-REPL smoke v2.3.1 (Yusen) | Yusen | gate |

Wave 1 three parallel tasks (T1/T2/T3) — each touches independent
files. Wave 2 depends on Wave 1 landing DetectSkill contract
extension. Keep serial dispatch discipline from v2.3 (no full
pytest, MuJoCo-forbidden imports in subagent prompts).

Estimated effort: ~70 minutes of agent time + 1 live-REPL smoke.

## Test safety rules (carry forward from v2.3)

- Subagents never run `pytest tests/` or `pytest tests/integration/`
- Narrow scope: `pytest tests/unit/perception/test_<file>.py`
- Forbidden imports list: `pipeline`, `track_anything`, `mujoco`,
  `realsense`, `tracker`
- Dispatcher runs narrow-scope regressions at wave gates
- Memory `feedback_no_parallel_agents.md` already enforces this

## Session starter

```bash
cd ~/Desktop/vector_os_nano
cat agents/devlog/status.md
cat .sdd/NEXT_SESSION.md                # this file
git log --oneline 51920c9..HEAD          # 10 v2.3 commits
python3 scripts/verify_perception_pick.py --dry-run   # sanity

/sdd plan                                # spec is already drafted here
```

## Open questions for CEO at /sdd init

None — spec is internally complete and approved via the 2026-04-20
smoke debrief. Architect can proceed straight to plan + tasks.

## Carried-forward v2.4 debt (not in v2.3.1 scope)

- EdgeTAM tracker for temporal consistency
- SAM3D per-pixel masks for irregular-object grasping
- `_normalise_color_keyword` promote to public utility (H3 in v2.3 review)
- `_wait_stable` extract to `skills/utils/mobile_helpers.py`
- `VECTOR_SHARED_EXECUTOR=0` spin-thread leak cleanup
- `coverage` package upgrade to resolve numpy 2.4 C-tracer conflict
- VGG decomposer strategy whitelist (avoid "No strategy for: unmatched")
- VGG `last_seen('blue bottle')['room']` literal-string goal bug
  (saw in this smoke — not v2.3 scope)
