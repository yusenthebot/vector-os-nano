# Agent Status

**Updated:** 2026-04-25 (v2.4 redirected to SysNav simulation integration; v2.3 perception purged)
**Branch:** `feat/v2.0-vectorengine-unification`

## Current state

**v2.4 SysNav Simulation Integration — SDD APPROVED, READY FOR EXECUTION**

CEO directive 2026-04-25: auto-approve architectural decisions, start
implementation, emphasize broad test coverage. v2.4-perception-overhaul
(YOLOE + SAM3 + own pointcloud / sanity gates) is archived because
SysNav already provides equivalent capabilities. SysNav (CMU sibling
lab, PolyForm-NC) runs as a separate ROS2 workspace; we publish
ground-truth `/registered_scan`, `/camera/image`, `/camera/depth`,
`/state_estimation` from MuJoCo and consume `/object_nodes_list` via
the existing `sysnav_bridge` adapter (Apache 2.0 boundary preserved).

## Cleanup landed pre-spec (this session)

Deleted (~2570 LoC removed):
- `vector_os_nano/perception/vlm_qwen.py`
- `vector_os_nano/perception/go2_perception.py`
- `vector_os_nano/perception/go2_calibration.py`
- `tests/unit/perception/test_{vlm_qwen,go2_perception,go2_calibration}.py`
- `tests/integration/test_sim_tool_perception_wire.py`
- `scripts/verify_perception_pick.py`
- `docs/v2.3_live_repl_checklist.md`

Modified:
- `vector_os_nano/vcli/tools/sim_tool.py` — Qwen wire-up block
  replaced by a comment pointing at `sysnav_bridge`. `agent._perception`
  / `agent._calibration` set to `None` until LiveSysnavBridge
  populates `world_model` directly.

Verified post-cleanup:
- 70/70 existing tests still green
  (`test_pick_top_down.py` 33, `test_mobile_pick.py` 22, `test_sysnav_bridge_mapping.py` 15).
- Targeted import smoke OK across perception/skills/integrations.

## v2.4 SysNav Integration — scope

8 code tasks across 6 waves:

| Wave | Task | Owner |
|------|------|-------|
| W0 | env probe (mj_ray + cube-face benchmarks) | dispatcher |
| W1 | T1 `MuJoCoLivox360` / T2 `GroundTruthOdom` / T3 G3 xmat fix | Alpha / Beta / Gamma serial |
| W2 | T4 `MuJoCoPano360` / T5 `LiveSysnavBridge` rclpy subscriber | Alpha / Beta serial |
| W3 | T6 `go2_vnav_bridge` wiring + integration tests / T7 `sysnav_sim_tool` CLI | Alpha / Beta serial |
| W4 | T8 `smoke_sysnav_sim.py` + `docs/sysnav_simulation.md` | Gamma |
| W5 | code-review + security-review (parallel subagents) | QA |
| W6 | CEO live-REPL smoke + tag | Yusen |

**Test budget**: ≥ 50 new unit tests + ≥ 5 integration tests, 90 %
coverage gate per wave. Each task's spec lists explicit RED test
names (write-fail-then-implement). Serial subagent dispatch per wave
to avoid OOM (per `feedback_no_parallel_agents.md`).

Estimated wall-clock: 3–4 days.

## Reference

- Spec: `.sdd/spec.md` (v2.4 SysNav Sim)
- Plan: `.sdd/plan.md`
- Tasks: `.sdd/task.md`
- Status: `.sdd/status.json` (phase=tasks, all approved)
- Bringup integration (real-robot side): `docs/sysnav_integration.md`
- Sim integration: `docs/sysnav_simulation.md` (T8 will write)
- Adapter (already landed): `vector_os_nano/integrations/sysnav_bridge/`
- SysNav repo (sibling lab): https://github.com/zwandering/SysNav

## Archive index

- `.sdd/archive-v2.4-perception-overhaul/` — YOLOE+SAM3 SDD (now redundant)
- `.sdd/archive-v2.3/` — Qwen perception cycle (impl deleted this session)
- `.sdd/archive-v2.2/` — loco manipulation infrastructure
- `.sdd/archive-v2.1-pick/` — Piper top-down grasp
- earlier archives unchanged.

## Session starter (next time)

```
cd ~/Desktop/vector_os_nano
cat agents/devlog/status.md          # this file
cat .sdd/spec.md                      # v2.4 SysNav Sim
.venv-nano/bin/python -m pytest tests/skills tests/integration/test_sysnav_bridge_mapping.py -q
git log --oneline 8dda396..HEAD       # this cycle's commits
```
