# Vector OS Nano SDK — Progress

**Last updated:** 2026-03-20 | **Version:** v0.1.0

## Status: Feature Complete — In Testing

All 4 development waves done. Full pipeline working end-to-end on hardware.

## Key Metrics

| Metric | Value |
|--------|-------|
| Tests passing | 696 |
| Source files | 54+ |
| Lines of code | 10,000+ |
| Test coverage | 85% |
| Tasks complete | 13/13 |

## Full Pipeline

```
NL command → LLM → scan → detect (VLM) → track (EdgeTAM) → 3D → calibrate → IK → pick → place
```

Background EdgeTAM tracking provides live bounding box overlay independent of pick pipeline.

LLM prompt is action-oriented: execute-first, no clarification questions by default.

Direct gripper commands (open/close/grip) bypass LLM entirely.

## Remaining Work

1. **Skill Manifest Protocol** — YAML-based alias mapping and LLM context enrichment (ADR-002, in design)
2. **Pick accuracy tuning** — calibration refinement on real hardware
3. **More testing** — extended hardware test cycles

## Architecture (Layers)

```
CLI (readline + optional Textual TUI)
Agent (planning + direct-mode dispatch)
Executor + World Model + Skills
Hardware Abstraction (Arm + Gripper)
--- optional ---
ROS2 Integration Layer
PyBullet Simulation Layer
```

## Modules

- **core**: types, config, protocols
- **hardware**: SO101Arm, SO101Gripper, SimulatedArm, SimulatedGripper
- **perception**: RealSenseCamera, VLMDetector, EdgeTAMTracker, pointcloud utils, calibration
- **llm**: Claude, OpenAI-compatible, Ollama providers
- **skills**: Pick, Place, Home, Scan, Detect + registry
- **cli**: readline shell, calibration wizard TUI
- **ros2**: 5 lifecycle nodes + launch file (optional)
- **utils**: logging, math helpers
