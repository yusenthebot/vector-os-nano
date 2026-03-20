# Vector OS Nano SDK — Technical Plan

**Version:** 0.1.0
**Date:** 2026-03-20
**Author:** Lead Architect (Opus)
**Prereq:** spec.md approved by CEO with decisions M12-M17

---

## 1. Architecture Overview

The SDK is a single pip-installable Python package with 7 modules:

```
vector_os/
├── core/        # Agent engine: planner, executor, world model, skill protocol
├── llm/         # LLM providers: Claude, OpenAI, local (Ollama)
├── perception/  # Camera, VLM, tracker, pointcloud, calibration
├── hardware/    # Arm/gripper protocols + SO-101 implementation
├── skills/      # Built-in skills: pick, place, home, scan, detect
├── cli/         # Simple CLI + Textual TUI dashboard
└── ros2/        # Optional ROS2 integration layer
```

Dependencies flow downward only: `core` depends on nothing; `skills` depends on `core` + `hardware` + `perception`; `cli` depends on everything; `ros2` wraps everything.

---

## 2. Technical Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Package manager | pyproject.toml + hatchling | Modern Python, supports extras |
| IK solver | Pinocchio (pin) | Cross-platform, already proven in vector_ws |
| Camera | pyrealsense2 | Direct driver, no ROS2 needed |
| VLM | Moondream local | Free, runs on 8GB+ VRAM |
| Tracker | EdgeTAM (HuggingFace) | Best speed/quality for video segmentation |
| LLM default | Claude Sonnet via OpenRouter | Best reasoning for task planning |
| TUI framework | Textual | Async-native, solves Qt/ROS2 threading issue |
| Simulation | PyBullet | Lightweight, pip-installable, good enough for tabletop |
| Serialization | YAML (config), JSON (LLM comms) | Human-readable, standard |
| Testing | pytest | Standard, works without ROS2 |
| ROS2 integration | Conditional import (`try: import rclpy`) | Zero-cost when not used |

---

## 3. Module Design

### Module A: core/ — Agent Engine

**Responsibility:** Task planning, execution, world state management, skill protocol.
**Input:** Natural language instructions, skill registry, world state.
**Output:** Execution results, world state updates.
**Dependencies:** None (pure Python + standard library).

Key classes:
- `Agent` — main entry point, orchestrates planner + executor
- `TaskPlanner` — calls LLM to decompose goals into task plans
- `TaskExecutor` — runs plans deterministically, no LLM calls
- `WorldModel` — tracks objects, robot state, spatial relations
- `Skill` (Protocol) — abstract skill interface
- `SkillRegistry` — discovers and validates skills
- `SkillContext` — injected into skills during execution

Source: NEW code. WorldModel concepts from vector_ws `perception_skills.py` caching logic.

### Module B: llm/ — LLM Providers

**Responsibility:** Pluggable LLM backends for planning and queries.
**Input:** Goal text, world state JSON, skill schemas.
**Output:** TaskPlan or query response.
**Dependencies:** httpx.

Key classes:
- `LLMProvider` (Protocol)
- `ClaudeProvider` — Anthropic / OpenRouter
- `OpenAIProvider` — OpenAI-compatible APIs
- `LocalProvider` — Ollama local models

Source: Refactor from `vector_ws/src/so101_bringup/so101_bringup/llm_client.py`. Extract HTTP logic, add plan parsing, make provider-agnostic.

### Module C: perception/ — Perception Stack

**Responsibility:** Camera driver, VLM detection, object tracking, 3D pointcloud, calibration.
**Input:** Camera frames, natural language queries.
**Output:** Detections, tracked objects with 3D poses, calibration matrices.
**Dependencies:** pyrealsense2, torch, transformers, open3d, numpy.

Key classes:
- `PerceptionProtocol` — abstract interface
- `RealSenseCamera` — D405 driver (from `camera.launch.py` parameters)
- `VLMDetector` — Moondream/Qwen (from `vlm/models/moondream.py`)
- `EdgeTAMTracker` — tracking (from `track_anything/edge_tam.py`)
- `PointCloudProcessor` — RGBD→3D (from `vector_perception_utils/pointcloud_utils.py`)
- `Calibration` — workspace calibration (from `so101_skills/scripts/calibrate_workspace.py`)
- `PerceptionPipeline` — orchestrates detect→track→3D pipeline (from `track_anything/track_3d.py`)

Source: Extract from vector_ws. Remove all rclpy imports. Keep pure Python logic.

### Module D: hardware/ — Hardware Abstraction

**Responsibility:** Arm and gripper control, FK/IK, serial communication.
**Input:** Joint targets, gripper commands.
**Output:** Joint states, gripper state, FK results.
**Dependencies:** pyserial, pinocchio (pin), numpy.

Key classes:
- `ArmProtocol` — abstract arm interface
- `GripperProtocol` — abstract gripper interface
- `SO101Arm` — SO-101 implementation (from `so101_hardware/hardware_bridge.py`)
- `SO101Gripper` — SO-101 gripper (from `hardware_bridge.py` gripper section)
- `JointConfig` — encoder/radian mapping (from `so101_hardware/joint_config.py`)
- `IKSolver` — Pinocchio FK/IK (from `so101_skills/pinocchio_ik.py`)
- `SerialBus` — SCS protocol (from `hardware_bridge.py` serial section)

Source: Extract from vector_ws. Split hardware_bridge.py into serial_bus.py + arm.py + gripper.py.

### Module E: skills/ — Built-in Skills

**Responsibility:** Pick, place, home, scan, detect implementations.
**Input:** Skill parameters via SkillContext.
**Output:** SkillResult.
**Dependencies:** core/, hardware/, perception/.

Key classes:
- `PickSkill` — (from `skill_node_v2.py` `_execute_pick`)
- `PlaceSkill` — (from `skill_node_v2.py` `_execute_place`)
- `HomeSkill` — (from `skill_node_v2.py` home logic)
- `ScanSkill` — (from `skill_node_v2.py` scan logic)
- `DetectSkill` — (from `perception_skills.py` detect_all logic)

Source: Extract skill logic from `skill_node_v2.py`. Remove ROS2 service/action wrappers. Keep pure pick/place algorithms.

### Module F: cli/ — Command Line Interfaces

**Responsibility:** User interaction, command parsing, LLM chat interface, TUI dashboard.
**Input:** User text commands.
**Output:** Agent execution, status display.
**Dependencies:** core/, textual (for dashboard).

Key classes:
- `SimpleCLI` — readline-based (from `so101_bringup/cli.py`, stripped of ROS2)
- `Dashboard` — Textual TUI (NEW code, based on terminal GUI design analysis)

Source: Refactor `cli.py`. Replace ROS2 service calls with direct Agent API calls.

### Module G: ros2/ — ROS2 Integration Layer

**Responsibility:** Wrap all SDK components as ROS2 nodes. Provide action servers, TF2, lifecycle.
**Input:** ROS2 topics, services, actions.
**Output:** ROS2 topics, services, actions.
**Dependencies:** rclpy, sensor_msgs, geometry_msgs, control_msgs, vision_msgs, tf2_ros.

Key classes:
- `HardwareBridgeNode` — wraps SO101Arm as ROS2 node (from `hardware_bridge.py`)
- `PerceptionNode` — wraps PerceptionPipeline as ROS2 node (from `track_3d.py`)
- `SkillServerNode` — ExecuteSkill action server (NEW)
- `WorldModelNode` — world model ROS2 service (NEW)
- `AgentNode` — wraps Agent as ROS2 node (NEW)

Source: Thin wrappers around SDK classes. Import SDK class, add ROS2 pub/sub/service/action.

---

## 4. Data Flow

### Direct Mode (no ROS2)

```
User → Agent.execute("pick the red cup")
         │
         ├→ WorldModel.to_dict() → current state
         │
         ├→ TaskPlanner.plan(goal, state, skills) → LLM HTTP call → TaskPlan
         │
         ├→ TaskExecutor.execute(plan, context)
         │     │
         │     ├→ DetectSkill: PerceptionPipeline.detect("red cup") → Detection
         │     │                PerceptionPipeline.track(detection) → TrackedObject
         │     │                WorldModel.update(tracked_object)
         │     │
         │     ├→ PickSkill:   Calibration.camera_to_base(pose) → base_pose
         │     │                IKSolver.ik(target) → joint_angles
         │     │                SO101Arm.move_joints(angles) → serial write
         │     │                SO101Gripper.close() → serial write
         │     │                SO101Gripper.is_holding() → serial read
         │     │                WorldModel.apply_effects("pick", result)
         │     │
         │     └→ return ExecutionResult
         │
         └→ return to User
```

### ROS2 Mode

```
User → AgentNode → PlanTask.srv → PlannerNode → LLM HTTP → TaskPlan
                 → ExecuteSkill.action → SkillServerNode
                        │
                        ├→ /track_3d/detections_3d (subscribe)
                        ├→ /joint_states (subscribe)
                        ├→ /arm_controller/follow_joint_trajectory (action client)
                        ├→ /gripper_controller/gripper_command (action client)
                        └→ /world_model/query (service client)
```

---

## 5. Key Implementation Details

### 5.1 SO-101 Serial Communication (from vector_ws)

Port from `hardware_bridge.py`:
- SCS protocol via scservo_sdk at 1Mbps baud
- Read-before-write on connect (prevent startup jump)
- 12-bit encoder (4096 counts/rev), linear rad↔enc mapping
- Joint config: 6 joints (5 arm + 1 gripper) with per-joint limits
- Trajectory: linear interpolation in joint space, 50 waypoints over duration

### 5.2 Perception Pipeline (from vector_ws)

Port from `track_3d.py` + `edge_tam.py` + `pointcloud_utils.py`:
- Frame loop: color callback → cache latest depth → pair for RGBD
- VLM detect: Moondream `detect(image, query)` → bboxes
- EdgeTAM init: bboxes → `init_track(image, bboxes)` → masks
- EdgeTAM track: `process_image(image)` → updated masks (20fps)
- 3D: mask + depth → `rgbd_to_pointcloud_fast()` → `pointcloud_to_bbox3d_fast()` → centroid
- Optimization: frame stride (compute 3D every 5th frame, cache between)

### 5.3 Pick Algorithm (from vector_ws)

Port from `skill_node_v2.py`:
- Collect 20 position samples over 1s (50ms interval)
- Density clustering (1.5cm threshold) → stable position
- Camera-to-base via affine calibration matrix
- Gripper asymmetry compensation (left/center/right Y offsets)
- Pinocchio IK → joint angles
- Trajectory: home → pre-grasp (6cm above) → descent → close → lift
- Retry: up to 2 attempts, home between retries
- Gripper close: 3x send with 0.2s interval

### 5.4 Calibration (from vector_ws)

Port from `calibrate_workspace.py`:
- Affine 4x4 matrix via least-squares from paired (camera, base) points
- IMPROVEMENT: support RBF/TPS for nonlinear correction
- IMPROVEMENT: require Z variation in calibration points
- Save/load to YAML

### 5.5 Agent Planner

NEW code:
- System prompt: skill registry (JSON schema) + world state + constraints
- LLM returns JSON task plan: `[{skill, params, depends_on, pre/postconditions}]`
- Validation: check all skills exist, params match schema, dependencies acyclic
- Clarification: if LLM returns `requires_clarification`, pass question to user
- Retry: on failure, re-invoke LLM with execution trace + failure reason

### 5.6 PyBullet Simulation

NEW code:
- `SimulatedArm(ArmProtocol)` — PyBullet URDF loading, joint control
- `SimulatedCamera(PerceptionProtocol)` — PyBullet camera rendering
- Load SO-101 URDF into PyBullet
- Tabletop scene with randomized objects
- Used for testing without hardware

---

## 6. Test Strategy

### Unit Tests (pytest, no hardware, no GPU)

| Module | Tests | Focus |
|--------|-------|-------|
| core/world_model | 10+ | Object CRUD, predicates, spatial relations, serialization |
| core/executor | 8+ | Precondition, postcondition, dependency order, failure handling |
| core/skill | 5+ | Registry, schema validation, predicate evaluation |
| llm/prompts | 3+ | Prompt construction, plan parsing, error handling |
| hardware/joint_config | 5+ | Enc↔rad mapping, clamping, NaN safety |
| hardware/ik_solver | 4+ | FK roundtrip, unreachable detection, joint limits |

### Integration Tests (pytest, may need GPU, mock hardware)

| Test | Focus |
|------|-------|
| Agent simple pick | Full pipeline with mock arm |
| Agent multi-step | Plan decomposition + execution |
| Agent failure retry | Replanning after skill failure |
| Perception pipeline | VLM detect → EdgeTAM track → 3D |
| Calibration roundtrip | Calibrate → transform → verify error |
| Skill registration | Custom skill discoverable by planner |

### System Tests (real hardware or PyBullet)

| Test | Focus |
|------|-------|
| SO-101 connect/home | Serial lifecycle |
| Real pick center | Physical pick at workspace center |
| Real pick edge | Physical pick at workspace edge |
| Full pick-and-place | Complete cycle |
| PyBullet pick | Simulated pick pipeline |

### Coverage Target: 80%+ for core/, 60%+ for other modules.

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pinocchio wheel unavailable on some platforms | Breaks cross-platform | Fallback to analytical IK for SO-101 (5-DOF, solvable) |
| EdgeTAM model download fails | Blocks perception | Cache model locally, provide manual download instructions |
| LLM plan quality poor for complex tasks | User frustration | Validate plans before execution; cap at 5 steps initially |
| Serial port permissions on Linux | Blocks hardware | Document udev rules in README; provide setup script |
| PyBullet visual fidelity low | Tests don't catch real issues | Clearly label as "functional sim, not visual sim" |
| ROS2 conditional import breaks | Silent failures | Test both code paths in CI (with and without rclpy) |
