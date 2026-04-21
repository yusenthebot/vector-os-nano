# v2.3 Go2 Perception Pipeline — Task Breakdown

**Phase**: tasks (draft)
**Prereq**: plan.md review (non-architectural — async)
**Agents**: Alpha / Beta / Gamma in parallel (Sonnet) + Scribe (Haiku)
**Reviewer**: code-reviewer + security-reviewer (parallel, Phase 5)

---

## Execution Summary

| Metric | Value |
|---|---|
| Total code tasks | 6 (T1–T6) |
| Docs task | 1 (T7) |
| Waves | 3 code + 1 docs/QA |
| Parallel peak | 3 agents (W1) |
| New files | 8 (3 impl + 5 tests) |
| Modified files | 3 |
| Target tests added | ≥ 24 (14 unit + 7 integration + 3 regression) |
| Final coverage floor | 95 % on new modules |

---

## Task 1 — QwenVLMDetector

- **ID**: T1
- **Agent**: Alpha
- **Wave**: 1
- **Depends**: none
- **Package**: `vector_os_nano/perception/`

### Scope
Implement Qwen2.5-VL grounded detection via OpenRouter; return
`list[Detection]` with pixel-space bboxes. Mirror `Go2VLMPerception`
structure (JPEG encoding, retries, cost tracking, `VECTOR_VLM_URL`
local escape hatch).

### Files
- **NEW**: `vector_os_nano/perception/vlm_qwen.py` (~180 LoC)
- **NEW**: `tests/unit/perception/test_vlm_qwen.py`

### TDD deliverables

**RED** — write these tests FIRST (they must fail since `vlm_qwen.py`
doesn't exist yet):

1. `test_detect_parses_json_bbox_list` — mock httpx 200 with
   `[{"label":"bottle","bbox":[10,20,30,40],"confidence":0.9}]` →
   returns `[Detection("bottle", (10,20,30,40), 0.9)]`.
2. `test_detect_parses_markdown_fenced_json` — response wrapped in
   ` ```json\n...\n``` ` → still parses.
3. `test_detect_empty_response_returns_empty_list` — `[]` → `[]`.
4. `test_detect_scales_normalised_bbox_to_pixels` — response with
   all coords ≤ 1.0 → scaled to frame WxH.
5. `test_detect_clamps_confidence_to_unit_interval` — confidence
   `1.5` → clamped to `1.0`; negative → `0.0`.
6. `test_detect_timeout_retries_then_fails` — 2 consecutive
   `httpx.TimeoutException` → `RuntimeError` after `MAX_RETRIES`.
7. `test_detect_4xx_does_not_retry` — httpx 401 → `RuntimeError`
   after 1 attempt; no additional `.post` calls.
8. `test_detect_tracks_cumulative_cost` — 2 successful calls → cost
   increments by `(input_tokens * rate_in + output_tokens * rate_out) * 2`.
9. `test_detect_honours_VECTOR_VLM_URL_env_no_auth_header` — env set →
   no `Authorization` header on request; model from
   `VECTOR_VLM_MODEL` if set.
10. `test_detect_raises_if_no_api_key_and_no_local_url` — unset both →
    `ValueError` in `__init__`.

**GREEN** — minimal implementation:

- `class QwenVLMDetector` with `__init__(config=None)`, `detect(image,
  query) → list[Detection]`, `cumulative_cost_usd` property.
- Reuse `_parse_json_response`, `_parse_detected_object` by importing
  from `vector_os_nano.perception.vlm_go2`.
- JPEG encoding + base64 identical to `Go2VLMPerception._encode_frame`.
- Retry loop identical to `Go2VLMPerception._call_vlm`.
- Prompt string exactly as specified in plan.md §6.1.
- Bbox scaling: if `max(x1,y1,x2,y2) ≤ 1.0`, multiply by W/H.

**REFACTOR**:
- Extract shared encode/retry into a module-level helper if > 50 LoC
  duplication with `vlm_go2.py`. Otherwise leave inline (duplication ≤
  3 lines is fine).

### Acceptance
- [ ] All 10 tests pass
- [ ] Coverage ≥ 95 % on `vlm_qwen.py`
- [ ] ruff clean
- [ ] `from vector_os_nano.perception.vlm_qwen import QwenVLMDetector`
      importable without network

### Verify
```
cd ~/Desktop/vector_os_nano
python -m pytest tests/unit/perception/test_vlm_qwen.py -v
python -m pytest tests/unit/perception/test_vlm_qwen.py \
    --cov=vector_os_nano.perception.vlm_qwen --cov-report=term-missing
ruff check vector_os_nano/perception/vlm_qwen.py tests/unit/perception/test_vlm_qwen.py
```

---

## Task 2 — Go2Calibration

- **ID**: T2
- **Agent**: Beta
- **Wave**: 1
- **Depends**: none
- **Package**: `vector_os_nano/perception/`

### Scope
Implement pose-driven camera-to-world transform. Reads live
`get_camera_pose()` from base_proxy per call. No paired-points fitting.

### Files
- **NEW**: `vector_os_nano/perception/go2_calibration.py` (~60 LoC)
- **NEW**: `tests/unit/perception/test_go2_calibration.py`

### TDD deliverables

**RED**:

1. `test_dog_at_origin_facing_x_1m_forward` — fake `base_proxy` with
   `get_camera_pose()` = cam at `(0.3, 0, 0.33)`, xmat identity in
   forward-X frame → cam point `(0, 0, 1.0)` OpenCV → world
   `(1.3, 0, 0.33)` ± 0.01.
2. `test_dog_at_origin_facing_y_90deg` — heading π/2, cam pose
   rotated → cam point `(0, 0, 1.0)` → world `(0, 1.3, 0.33)` ± 0.01.
3. `test_dog_offset_position_adds_to_world` — dog at `(5, 3, 0.28)`,
   facing +X → world shifts by `(5, 3, 0)` relative to the origin case.
4. `test_input_shape_preserved` — input `np.array([1, 2, 3])` → output
   shape `(3,)`.
5. `test_input_accepts_list_not_only_ndarray` — list `[0, 0, 1.0]`
   works identically to ndarray.
6. `test_downward_pitch_minus_5deg_lowers_z` — dog at origin, cam
   point `(0, 0, 2)` (2m forward), compare vs pitch=0: resulting
   world z should be ~`0.33 − 2·sin(5°) ≈ 0.156` (wait — we use the
   MJCF mount pitch baked into `get_camera_pose()`, so this test
   verifies against that mount baseline).
7. `test_off_centre_pixel_deprojection_symmetry` — cam point `(1, 0, 5)`
   (right+forward) and `(-1, 0, 5)` (left+forward) → world y delta
   is symmetric around dog's world y.
8. `test_y_axis_opencv_to_world_up_convention` — cam point `(0, 1, 5)`
   (down in OpenCV) → world z decreases (down in world).

Note: tests 6–8 lock in the OpenCV↔MuJoCo axis convention (`up = -y_cam`
in OpenCV). Miswiring this is the single most likely defect.

**GREEN**:
- `class Go2Calibration` with `__init__(base_proxy)` and
  `camera_to_base(point_camera) → np.ndarray(3,)`.
- Math: `world = pos + p[0]*xmat[:,0] + (-p[1])*xmat[:,1] + p[2]*(-xmat[:,2])`
  (OpenGL xmat cols = [right, up, -forward]; OpenCV p = (right, down,
  forward)).

**REFACTOR**:
- None expected — pure math.

### Acceptance
- [ ] All 8 tests pass
- [ ] Coverage ≥ 98 %
- [ ] ruff clean
- [ ] Math docstring includes axis convention explicitly

### Verify
```
python -m pytest tests/unit/perception/test_go2_calibration.py -v
python -m pytest tests/unit/perception/test_go2_calibration.py \
    --cov=vector_os_nano.perception.go2_calibration --cov-report=term-missing
```

---

## Task 3 — Go2Perception

- **ID**: T3
- **Agent**: Gamma
- **Wave**: 1
- **Depends**: none (duck-types the VLM and camera)
- **Package**: `vector_os_nano/perception/`

### Scope
PerceptionProtocol implementation for Go2. Composes any
`camera` object (with `get_camera_frame / get_depth_frame`) and any
`vlm` object (with `detect(image, query)`). No hard import of T1 —
duck-typed.

### Files
- **NEW**: `vector_os_nano/perception/go2_perception.py` (~150 LoC)
- **NEW**: `tests/unit/perception/test_go2_perception.py`

### TDD deliverables

**RED**:

1. `test_get_color_delegates_to_camera` — fake camera with fixed
   `get_camera_frame()`; `perception.get_color_frame()` returns same.
2. `test_get_depth_delegates_to_camera`.
3. `test_get_intrinsics_default_matches_mjcf_42deg` —
   `perception.get_intrinsics()` returns
   `mujoco_intrinsics(320, 240, 42.0)`.
4. `test_get_intrinsics_custom_override` — pass `intrinsics=...` to
   ctor → exposed by `get_intrinsics`.
5. `test_detect_calls_vlm_with_color_frame` — fake camera returns RGB,
   fake vlm records args → `perception.detect("bottle")` → vlm.detect
   called with RGB + "bottle"; result passed through.
6. `test_track_single_detection_projects_centroid` — synthetic depth
   frame filled with 1.5 m; bbox (100, 80, 150, 120);
   `intr = mujoco_intrinsics(320,240,42)` → returned TrackedObject has
   `pose.z == 1.5`, `pose.x ≈ (125-160)/fx * 1.5`,
   `pose.y ≈ (100-120)/fy * 1.5`.
7. `test_track_empty_detections_returns_empty`.
8. `test_track_returns_none_pose_for_all_zero_depth_bbox`.
9. `test_track_uses_median_bbox_depth_ignoring_one_outlier` — depth =
   1.0 m in bbox except one pixel at 9.0 m → pose.z = 1.0.
10. `test_track_filters_nan_and_negative_depth_pixels` — bbox contains
    NaN + -0.5 + 1.2 m pixels → uses 1.2.
11. `test_track_one_to_one_length_preserved` — 3 detections →
    3 TrackedObjects, track_id = 1..3.
12. `test_track_handles_bbox_outside_frame_clamps` — bbox
    `(-10, -10, 100, 100)` → clamps to `(0, 0, 100, 100)`, still
    projects.

**GREEN**:
- Per plan.md §3.2. `_project_bbox_to_camera_frame` is a static method.

**REFACTOR**:
- Extract the IQR reject + median into `skills/utils/depth_stats.py`?
  **NO** — only one caller in v2.3. Keep inline. If v2.4 adds a
  second caller, extract then.

### Acceptance
- [ ] All 12 tests pass
- [ ] Coverage ≥ 95 %
- [ ] ruff clean
- [ ] Structural `isinstance(go2p, PerceptionProtocol)` passes
      (`runtime_checkable`)

### Verify
```
python -m pytest tests/unit/perception/test_go2_perception.py -v
python -m pytest tests/unit/perception/test_go2_perception.py \
    --cov=vector_os_nano.perception.go2_perception --cov-report=term-missing
```

---

## Wave 1 gate

After T1, T2, T3 all complete:

```
cd ~/Desktop/vector_os_nano
python -m pytest tests/unit/perception/ -v
python -m pytest tests/ -q     # regression: all 108 prior + 30 new
ruff check vector_os_nano/perception/
```

Expected: 138 passing (108 baseline + 30 new). No regressions.

Advance to Wave 2 only when gate passes.

---

## Task 4 — sim_tool perception wire-up

- **ID**: T4
- **Agent**: Alpha
- **Wave**: 2
- **Depends**: T1, T2, T3
- **Package**: `vector_os_nano/vcli/tools/`

### Scope
Inside `SimStartTool._start_go2`, construct and assign
`agent._perception = Go2Perception(...)` and `agent._calibration =
Go2Calibration(...)` in the `with_arm=True` branch when an API key is
available. Preserve existing `agent._vlm = Go2VLMPerception(...)`.

### Files
- **MODIFIED**: `vector_os_nano/vcli/tools/sim_tool.py` (+~18 LoC)
- **NEW**: `tests/integration/test_sim_tool_perception_wire.py`

### TDD deliverables

**RED**:

1. `test_start_go2_with_arm_and_key_wires_perception_and_calibration`
   — monkey-patch `Go2ROS2Proxy.connect`, `PiperROS2Proxy.connect`,
   subprocess launch, `Go2VLMPerception.__init__`; verify
   `agent._perception is Go2Perception` and `agent._calibration is
   Go2Calibration`.
2. `test_start_go2_with_arm_no_key_leaves_perception_none` —
   `OPENROUTER_API_KEY=""` and `user.yaml` has no llm.api_key →
   `agent._perception is None`, `agent._calibration is None`, no
   crash, skills still registered.
3. `test_start_go2_without_arm_does_not_wire_perception` —
   `with_arm=False` → `_perception` / `_calibration` unset (or None).
4. `test_start_go2_perception_ctor_failure_logs_and_continues` —
   monkey-patch `QwenVLMDetector.__init__` to raise RuntimeError →
   `_perception` None, `_calibration` None, agent still built.
5. `test_start_go2_vlm_go2_coexists_with_perception` — both
   `agent._vlm` (Go2VLMPerception) AND `agent._perception`
   (Go2Perception) are set when with_arm + key.

**GREEN**:
- Per plan.md §3.4. Insert block right after `logger.info("[sim_tool]
  Piper proxies connected ...")`.
- Use try/except around perception construction so failure is
  non-fatal (logged, both set None).

**REFACTOR**:
- None — purely additive wiring.

### Acceptance
- [ ] All 5 tests pass
- [ ] No change in existing `_start_go2` tests
- [ ] Logs show: `"[sim_tool] Go2 perception + calibration wired (Qwen)"`
      when successful

### Verify
```
python -m pytest tests/integration/test_sim_tool_perception_wire.py -v
python -m pytest tests/integration/ -q   # regression
ruff check vector_os_nano/vcli/tools/sim_tool.py
```

---

## Task 5 — MobilePick auto-detect retry + label helper

- **ID**: T5
- **Agent**: Beta
- **Wave**: 2
- **Depends**: T3 (for PerceptionProtocol contract); independent of T4
- **Package**: `vector_os_nano/skills/`

### Scope
1. Add `label_to_en_query()` to `skills/utils/__init__.py`.
2. Modify `MobilePickSkill.execute` to invoke `DetectSkill` on
   world_model miss (when perception + calibration available), then
   retry `_resolve_target`.

### Files
- **MODIFIED**: `vector_os_nano/skills/utils/__init__.py` (+~40 LoC)
- **MODIFIED**: `vector_os_nano/skills/mobile_pick.py` (+~25 LoC)
- **NEW**: `tests/unit/utils/test_label_to_en_query.py`
- **EXTEND**: `tests/unit/skills/test_mobile_pick.py` (+4 cases)

### TDD deliverables

**RED** (label_to_en_query):

1. `test_pure_english_passthrough` — `"blue bottle"` → `"blue bottle"`.
2. `test_cn_color_and_noun_mapped` — `"蓝色瓶子"` → `"blue bottle"`.
3. `test_possessive_de_stripped` — `"红色的杯子"` → `"red cup"`.
4. `test_empty_returns_none` — `""` → `None`; `None` → `None`.
5. `test_mixed_cn_en_works` — `"blue 瓶子"` → `"blue bottle"`.
6. `test_unknown_cn_passes_through_as_is` — `"奇怪的东西"` →
   `"奇怪东西"` (的 stripped; no noun map, keep rest).
7. `test_all_objects_passthrough` — `"all objects"` → `"all objects"`.

**RED** (mobile_pick auto-detect):

1. `test_mobile_pick_auto_detect_on_miss_then_hit` — stub
   `context.perception.detect` returns 1 Detection; stub
   `context.perception.track` returns 1 TrackedObject with pose;
   `context.calibration.camera_to_base` returns fixed xyz; stub
   world_model empty initially; verify `DetectSkill.execute` called,
   `add_object` called, and skill proceeds past resolve (does NOT
   return `object_not_found`).
2. `test_mobile_pick_no_retry_when_perception_none` — perception None
   → existing `object_not_found` with `known_objects=[]`, retry not
   attempted.
3. `test_mobile_pick_no_retry_when_calibration_none` — perception set
   but calibration None → no retry, `object_not_found`.
4. `test_mobile_pick_vlm_returns_empty_then_object_not_found` —
   perception `.detect()` returns `[]`; retry runs but
   `_resolve_target` still None → `object_not_found`.
5. `test_mobile_pick_detect_crash_does_not_crash_skill` —
   `DetectSkill.execute` raises → caught → `object_not_found` result.

**GREEN**:
- Per plan.md §3.5 and §3.6.
- Important: do NOT call `DetectSkill` inside the retry if
  `label_to_en_query(...)` returns None (empty query).

**REFACTOR**:
- None.

### Acceptance
- [ ] 7 + 5 = 12 tests pass
- [ ] Coverage on `label_to_en_query` = 100 %
- [ ] Coverage on new lines in `mobile_pick.py` ≥ 90 %
- [ ] All existing mobile_pick tests still pass

### Verify
```
python -m pytest tests/unit/utils/test_label_to_en_query.py -v
python -m pytest tests/unit/skills/test_mobile_pick.py -v
python -m pytest tests/ -q   # regression
```

---

## Wave 2 gate

After T4 + T5 complete:

```
python -m pytest tests/ -q
colcon test --packages-select vector_os_nano  # if configured
ruff check vector_os_nano/ tests/
```

Expected: 150 passing (108 + 30 W1 + 12 W2). No regressions.

---

## Task 6 — E2E verify_perception_pick.py

- **ID**: T6
- **Agent**: Alpha
- **Wave**: 3
- **Depends**: T4, T5
- **Package**: `scripts/`

### Scope
End-to-end harness mirroring `scripts/verify_loco_pick_place.py`.
Supports `--dry-run` (CI-safe: no MuJoCo, no OpenRouter; synthetic
frames and stubbed Qwen response) and `--live` (full stack, for
Yusen's REPL-side smoke).

### Files
- **NEW**: `scripts/verify_perception_pick.py` (~200 LoC)

### TDD deliverables

**RED**:
The script is a test harness itself — no separate unit test. Acceptance
is: `python scripts/verify_perception_pick.py --dry-run` exits 0 and
prints `"OK: perception→pick chain verified"`.

Dry-run test case (encoded in the script):
- Monkey-patch `QwenVLMDetector.detect` to return
  `[Detection("blue bottle", (120, 90, 160, 130), 0.91)]`.
- Monkey-patch `Go2ROS2Proxy.get_camera_frame` to return a 240×320 RGB
  filled with blue in that bbox, rest grey.
- Monkey-patch `get_depth_frame` to return 240×320 float32 with 1.2 m
  inside the bbox region, 3.0 m elsewhere.
- Monkey-patch `get_camera_pose` to return dog-at-origin pose.
- Build a minimal `SkillContext` with a `Go2Perception` /
  `Go2Calibration` over these fakes, plus a fake `base` that records
  navigate_to calls, a fake `arm` / `gripper` that never fails.
- Call `DetectSkill.execute({"query": "blue bottle"}, ctx)` → expect
  `success=True, count=1`; `world_model.get_objects()` has 1 entry
  with world xy close to the camera projection of bbox centre through
  1.2 m forward.
- Call `MobilePickSkill.execute({"object_label": "蓝色瓶子"}, ctx)`
  with EMPTY world_model again → expect auto-detect path hits,
  `navigate_to` called, `PickTopDownSkill` invoked, success.

**GREEN**:
- Implement the harness step-by-step using existing classes as
  building blocks.
- Exit 0 on all steps passing, 1 otherwise with a diff of expected vs
  actual.

### Acceptance
- [ ] `python scripts/verify_perception_pick.py --dry-run` exits 0
- [ ] Runs in < 3 s
- [ ] No real network or MuJoCo required

### Verify
```
python scripts/verify_perception_pick.py --dry-run
echo "exit: $?"
```

---

## Wave 3 gate

After T6 complete:

```
python -m pytest tests/ -q
python scripts/verify_perception_pick.py --dry-run
```

Expected: all green.

---

## Task 7 — Docs + progress

- **ID**: T7
- **Agent**: Scribe (Haiku)
- **Wave**: 4 (parallel with QA)
- **Depends**: T6

### Scope
- Add `docs/v2.3_live_repl_checklist.md` (mirrors v2.2 checklist).
- Update `progress.md` — add v2.3 section.
- Update `agents/devlog/status.md` — reflect v2.3 done state.
- Update `.sdd/status.json` → current_phase=review.

### Files
- **NEW**: `docs/v2.3_live_repl_checklist.md`
- **MODIFIED**: `progress.md`
- **MODIFIED**: `agents/devlog/status.md`
- **MODIFIED**: `.sdd/status.json`

### Acceptance
- [ ] Checklist has 5 live-REPL smoke steps (launch, detect, mobile-
      pick, place, stop)
- [ ] progress.md v2.3 section uses same headings as v2.2 section
- [ ] Status JSON phase reflects current state

---

## Wave 4 — QA (parallel with T7)

- **QA1**: code-reviewer on all modified + new files
- **QA2**: security-reviewer (focus on `vlm_qwen.py` — external API,
  handles API keys)

### Acceptance
- [ ] 0 critical, 0 high
- [ ] All major findings addressed or deferred with note
- [ ] Security review clears API-key handling

---

## Dependency Graph

```
T1 (Alpha, Qwen client) ─┐
T2 (Beta, Calibration)  ─┼──▶ W1 gate ──▶ T4 (Alpha, sim_tool wire) ─┐
T3 (Gamma, Perception)  ─┘                T5 (Beta, mobile_pick)     ─┼──▶ W2 gate ──▶ T6 (Alpha, E2E) ──▶ W3 gate ──▶ T7 + QA (W4)
                                                                      │
                                                                  (independent)
```

All W1 tasks are independent — any ordering works; run in parallel.
W2 tasks (T4, T5) both require T3's contract but not each other; run
in parallel. W3 single task.

## Execution Waves

| Wave | Tasks | Agents | Parallelism | Exit gate |
|---|---|---|---|---|
| 1 | T1, T2, T3 | Alpha, Beta, Gamma | 3 | pytest green on new tests + no regressions |
| 2 | T4, T5 | Alpha, Beta | 2 | full pytest green |
| 3 | T6 | Alpha | 1 | dry-run exit 0 |
| 4 | T7 + QA1 + QA2 | Scribe + code-reviewer + security-reviewer | 3 | 0 critical/high + docs clean |

## Risks per task

| Task | Principal risk | Mitigation |
|---|---|---|
| T1 | httpx mocking subtlety (context manager) | Use `unittest.mock.patch.object(httpx, "Client")` or `monkeypatch.setattr("httpx.Client", ...)` — existing `test_vlm_go2.py` pattern |
| T2 | Axis convention bug (up = -y_cam) | Dedicated axis test (cases 6–8) |
| T3 | Intrinsics mismatch with MJCF fovy | Hard-coded fovy=42 assertion |
| T4 | Integration test flakiness under rclpy | Use existing `conftest.py` rclpy isolation pattern from `test_ros2_coexist.py` |
| T5 | Retry recursion risk | Verified by construction: DetectSkill does not recurse into MobilePick |
| T6 | E2E flake from real ROS2 | Dry-run stubs everything — no ROS2 required |
| T7 | Doc drift from code | Scribe reads status.json first, copies from spec/plan exact wording |

## Escalation

Any agent stuck > 2 attempts → escalate to `vr-lead` (Opus Architect)
with DEBUG.md. Architect either redefines approach or delegates.
