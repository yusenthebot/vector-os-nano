# Agent Status

**Updated:** 2026-04-19 (v2.3 implementation landed, QA pending)
**Branch:** `feat/v2.0-vectorengine-unification`

## Current state

**v2.3 Go2 Perception Pipeline — IMPLEMENTATION DONE, QA PENDING**

All 6 code tasks + 1 docs task complete. 80 new tests green
(188 cumulative). Full `抓 X` flow now runs against an empty
world_model via Qwen VLM auto-detect. Ready for Wave 4 QA
(code-review + security-review in parallel) and Yusen's
live-REPL smoke.

## Wave summary

| Wave | Tasks | Agents | Tests added | Gate |
|------|-------|--------|-------------|------|
| W1 | T1 Qwen · T2 Calibration · T3 Perception | Alpha · Beta · Gamma (serial after OOM lesson) | 44 | GREEN |
| W2 | T4 sim_tool wire · T5 MobilePick + label helper | Alpha · Beta | 31 | GREEN |
| W3 | T6 E2E dry-run harness | Alpha | 1 dry-run | GREEN |
| W4 docs | T7 live REPL checklist + progress | Dispatcher (direct) | — | DONE |
| W4 QA | code-review + security-review | subagents (pending) | — | PENDING |

## Commit chain (v2.3, feat/v2.0-vectorengine-unification)

```
37f32e7  [alpha] test(v2.3): verify_perception_pick.py E2E dry-run
2b67c6f  [beta]  feat(v2.3): MobilePick auto-detects on world_model miss
a77a2c6  [alpha] feat(v2.3): sim_tool wires Go2Perception + Go2Calibration
24ae9b1  [gamma] feat(v2.3): Go2Perception — PerceptionProtocol for Go2 sim
3ac9d58  [beta]  feat(v2.3): Go2Calibration — pose-driven camera-to-world
f59c77e  [alpha] feat(v2.3): QwenVLMDetector — grounded 2D detection
```

## Runtime safety notes

First Wave 1 dispatch (3 parallel subagents) triggered OOM crash —
each agent ran full `pytest tests/` which auto-loads MuJoCo from a
cascade through `pipeline.py`. 3 concurrent MuJoCo contexts
exhausted 64 GB RAM. Agents killed mid-flight, all files lost.

Recovery strategy applied throughout v2.3:
- Serial subagent dispatch (one at a time)
- Subagents run only their own single test file, never full pytest
- Dispatcher runs narrow-scope regressions at wave gates
- Forbidden-import list in every subagent prompt
  (pipeline, track_anything, mujoco, realsense, tracker)

Memory `feedback_no_parallel_agents.md` updated with stricter rules.

## Known issues / debt

- `VECTOR_SHARED_EXECUTOR=0` legacy spin path leaks (rollback-only).
- `pytest-cov` C-tracer conflicts with `numpy 2.4.x` — coverage
  measured via `sys.settrace` for `go2_perception`.
- Camera `xmat` up-axis convention — self-consistent between
  `get_camera_pose` Python fallback and `camera_to_world`, but may
  diverge from MuJoCo `data.cam_xmat` in live usage. Flag for v2.4
  if live smoke shows lateral offset error.

## Live REPL verification

Checklist: `docs/v2.3_live_repl_checklist.md` (5 steps + diagnosis
ladder). Requires real `OPENROUTER_API_KEY` with Qwen2.5-VL-72B
access.

## Next

1. QA: code-reviewer + security-reviewer in parallel (Wave 4)
2. Yusen final approval
3. Live REPL smoke (Yusen)
4. v2.4 seeds: EdgeTAM tracker, SAM3D masks, cam_xmat
   reconciliation, mobile_helpers.py extraction

## Reference

- Spec: `.sdd/spec.md` (v2.3)
- Plan: `.sdd/plan.md` (v2.3)
- Tasks: `.sdd/task.md` (7/7 done; 6 code + 1 docs)
- Archive: `.sdd/archive-v2.2/` (previous cycle frozen)
- E2E: `scripts/verify_perception_pick.py --dry-run`
- Live checklist: `docs/v2.3_live_repl_checklist.md`

## Session starter (next time)

```
cd ~/Desktop/vector_os_nano
cat agents/devlog/status.md           # this file
cat progress.md | head -100           # v2.3 change log
python3 scripts/verify_perception_pick.py --dry-run  # 1s sanity
```
