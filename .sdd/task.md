# Vector OS Nano SDK — Task Breakdown

**Date:** 2026-03-20
**Total Tasks:** 13
**Waves:** 4 (+ review wave)

---

## Execution Status
- Total tasks: 13
- Completed: 12
- In progress: 1 (T12)
- Pending: 0

---

## Wave 1: Foundation (all independent, fully parallel)

### Task 1: Package skeleton + core types
- **Status**: [x] complete (alpha, 2026-03-20 14:00Z) — 164 tests pass
- **Agent**: alpha
- **Depends**: none
- **Output**: `pyproject.toml`, `vector_os/__init__.py`, `vector_os/core/types.py`, `vector_os/core/config.py`, `config/default.yaml`
- **Branch**: feat/alpha-package-skeleton

### Task 2: Hardware abstraction + SO-101 driver
- **Status**: [x] complete (beta, 2026-03-20 15:00Z) — 113 tests pass
- **Agent**: beta
- **Depends**: none
- **Output**: `vector_os/hardware/arm.py`, `hardware/gripper.py`, `hardware/so101/arm.py`, `hardware/so101/gripper.py`, `hardware/so101/joint_config.py`, `hardware/so101/serial_bus.py`
- **Branch**: feat/beta-so101-hardware-driver

### Task 3: IK solver + world model + executor
- **Status**: [x] complete (gamma, 2026-03-20) — 89 tests pass
- **Agent**: gamma
- **Depends**: none
- **Output**: `vector_os/hardware/so101/ik_solver.py`, `vector_os/core/world_model.py`, `vector_os/core/executor.py`, `vector_os/core/skill.py`
- **Branch**: feat/gamma-ik-world-executor

---

## Wave 2: Intelligence + Perception (depends on Wave 1 types)

### Task 4: LLM providers
- **Status**: [x] complete (alpha, 2026-03-20) — 79 tests pass
- **Agent**: alpha
- **Depends**: Task 1 (types)
- **Output**: `vector_os/llm/base.py`, `llm/claude.py`, `llm/openai_compat.py`, `llm/local.py`, `llm/prompts.py`
- **Branch**: feat/alpha-llm-providers

### Task 5: Perception stack (camera + VLM + tracker + pointcloud)
- **Status**: [x] complete (beta, 2026-03-20) — 20 tests pass
- **Agent**: beta
- **Depends**: Task 1 (types)
- **Output**: `vector_os/perception/base.py`, `perception/realsense.py`, `perception/vlm.py`, `perception/tracker.py`, `perception/pointcloud.py`, `perception/pipeline.py`
- **Branch**: feat/beta-perception-stack

### Task 6: Built-in skills (pick, place, home, scan, detect)
- **Status**: [x] complete (gamma, 2026-03-20) — 72 tests pass
- **Agent**: gamma
- **Depends**: Task 1 (types), Task 2 (hardware), Task 3 (world model, skill protocol)
- **Output**: `vector_os/skills/pick.py`, `skills/place.py`, `skills/home.py`, `skills/scan.py`, `skills/detect.py`
- **Branch**: feat/gamma-builtin-skills

---

## Wave 3: Integration + CLI + ROS2 (depends on Wave 2)

### Task 7: Agent class (main entry point)
- **Status**: [x] complete (alpha, 2026-03-20) — 36 tests pass
- **Agent**: alpha
- **Depends**: Task 3 (executor, world_model), Task 4 (llm), Task 6 (skills)
- **Output**: `vector_os/core/agent.py`
- **Branch**: feat/alpha-agent-class

### Task 8: Simple CLI
- **Status**: [x] complete (beta, 2026-03-20) — 25 tests pass
- **Agent**: beta
- **Depends**: Task 7 (agent)
- **Output**: `vector_os/cli/simple.py`
- **Branch**: feat/beta-simple-cli

### Task 9: ROS2 integration layer
- **Status**: [x] complete (gamma, 2026-03-20) — 26 tests pass
- **Agent**: gamma
- **Depends**: Task 2 (hardware), Task 5 (perception), Task 6 (skills), Task 7 (agent)
- **Output**: `vector_os/ros2/` complete with nodes + launch
- **Branch**: feat/gamma-ros2-integration

---

## Wave 4: Polish + Sim + Dashboard (depends on Wave 3)

### Task 10: Calibration module + TUI wizard
- **Status**: [x] complete (alpha, 2026-03-20) — 28 tests pass
- **Agent**: alpha
- **Depends**: Task 2 (hardware), Task 5 (perception)
- **Output**: `vector_os/perception/calibration.py` (enhanced), `vector_os/cli/calibration_wizard.py`
- **Branch**: feat/alpha-calibration-wizard

### Task 11: PyBullet simulation
- **Status**: [x] complete (beta, 2026-03-20) — 75 tests pass
- **Agent**: beta
- **Depends**: Task 1 (types), Task 2 (hardware protocol)
- **Output**: `vector_os/hardware/sim/pybullet_arm.py`, `vector_os/perception/sim/pybullet_camera.py`
- **Branch**: feat/beta-pybullet-sim

### Task 12: Textual TUI dashboard
- **Status**: [~] in progress (gamma) — backlog
- **Agent**: gamma
- **Depends**: Task 7 (agent), Task 8 (cli)
- **Output**: `vector_os/cli/dashboard.py`
- **Details**: Textual App with tabs: Dashboard, Log, Params, Calibration

### Task 13: README + examples + pyproject.toml finalization
- **Status**: [x] complete (gamma, 2026-03-20) — no new tests
- **Agent**: gamma
- **Depends**: all previous
- **Output**: `README.md`, `examples/*.py`, final `pyproject.toml`, `LICENSE`
- **Branch**: feat/gamma-readme-examples-finalize

---

## Dependency Graph

```
T1 (types/config) ──────────┬──> T4 (llm)     ──> T7 (agent) ──> T8 (cli)
                             │                       │              │
T2 (hardware/SO101) ────────┼──> T6 (skills)  ──────┤              ├──> T13 (readme)
                             │                       │              │
T3 (ik/worldmodel/executor)─┘──> T5 (perception)──> T9 (ros2)     ├──> T10 (calibration)
                                                                    ├──> T11 (pybullet)
                                                                    └──> T12 (dashboard)
```

## Execution Waves

| Wave | Tasks | Agents | Gate | Status |
|------|-------|--------|------|--------|
| 1 | T1, T2, T3 | Alpha, Beta, Gamma | unit tests pass | COMPLETE (281 tests) |
| 2 | T4, T5, T6 | Alpha, Beta, Gamma | unit + integration tests pass | COMPLETE (452 tests) |
| 3 | T7, T8, T9 | Alpha, Beta, Gamma | agent integration test pass | COMPLETE (539 tests) |
| 4 | T10, T11, T12, T13 | Alpha, Beta, Gamma | all tests pass | PARTIAL (642 tests, T12 in progress) |
| review | — | code-reviewer, security-reviewer | approved | IN PROGRESS |

**Total: 642 tests passing, 12/13 tasks complete**
