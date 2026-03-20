# SDD Changelog — Vector OS Nano SDK

## 2026-03-20

- **spec.md v0.1.0-draft**: Initial specification
  - Three-pillar architecture: pick reliability, developer terminal, multi-agent orchestration
  - SDK-first design: pure Python core, ROS2 optional
  - Cross-platform target: Linux v0.1, Windows/macOS v0.2
  - 6 core protocols: Arm, Gripper, Perception, LLM, Skill, WorldModel
  - Migration plan from vector_ws (9 ROS2 packages) to unified SDK
  - 20 unit test contracts, 9 integration test contracts, 8 system test contracts
  - 12 acceptance criteria for v0.1 release
  - Based on: 3 Opus deep analysis agents + 6 web research queries + CEO architecture discussion
