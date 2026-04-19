# Agent Status

**Updated:** 2026-04-19 end-of-day
**Branch:** `feat/v2.0-vectorengine-unification`

## Current state

**v2.2 Loco Manipulation Infrastructure — BASELINE READY**

Shortcut design (MJCF populate, VGG source-hint skipping detect_*, pick
single-candidate / generic fallback) has been removed. World model now
starts empty by design; skills must be driven by real perception. This
is the correct platform for the next cycle.

## Next session — v2.3 Perception Pipeline (SO-101 style, for Go2)

Goal: mirror the SO-101 perception pattern so `抓个东西` walks the full
pipeline `look → detect (VLM + depth → 3D) → mobile_pick → place`.

### Scope sketch

1. **`Go2Perception`** class under `vector_os_nano/perception/go2_perception.py`
   - `.detect(query: str) -> list[Detection]` — VLM with normalized bboxes
   - `.track(detections) -> list[TrackedObject]` — depth-based 3D centroid
   - Integrates existing `Go2VLMPerception.find_objects` + `/camera/depth`
2. **`Go2Calibration`** class — camera intrinsics + head→base TF from
   MJCF/URDF; base→world from odometry
3. **`sim_tool._start_go2`** wires `agent._perception = Go2Perception(...)`
   + `agent._calibration = Go2Calibration(...)` when `with_arm=True`
4. **`DetectSkill`** now alive on Go2 (was dead — no `context.perception`)
5. **`MobilePickSkill._resolve_target`** — world_model miss → auto invoke
   `context.perception.detect(query)` → retry; on success world_model
   gets a fresh `ObjectState` with perception-derived `(x, y, z)`
6. **E2E harness** — extend `verify_loco_pick_place.py` to exercise
   "perception → pick" path (no MJCF pre-populate)

### Known non-goals for v2.3
- SAM3D per-pixel masks (v2.4+)
- Multi-view fusion
- Real Piper hardware driver
- Arm-base coordinated motion

### Open debt carried in
- Legacy rollback spin thread (`VECTOR_SHARED_EXECUTOR=0`) leaks on
  disconnect — only if the flag is set. Cleanup in v2.3.
- Divergent `_wait_stable` impls in mobile_pick vs mobile_place —
  extract to `skills/utils/mobile_helpers.py`.
- LLM-generated sub_goal names not matching any strategy ("approach_object"
  / "grasp_object") → `"No strategy for: unmatched"`. Decomposer needs
  to constrain strategy field to registered skill names.

## Reference

- Spec: `.sdd/spec.md` (v2.2)
- Plan: `.sdd/plan.md`
- Tasks: `.sdd/task.md` (13/13 done)
- Debug: `.sdd/DEBUG.md` (hypothesis loop for "抓个东西" failure + hotfix)
- Summaries: `.sdd/summaries/wave-{1..4}.md`, `hotfix-generic-query.md`
- Executive brief: `.sdd/executive-briefs/completion-report.md`
- Live REPL checklist: `docs/v2.2_live_repl_checklist.md`

## Session starter for next time

```
cd ~/Desktop/vector_os_nano
cat agents/devlog/status.md           # this file
cat progress.md | head -80            # v2.2 state
/sdd init                             # kick off v2.3 perception SDD
```
