# Vector OS Nano — SDK Specification

**Version:** 0.1.0-draft
**Date:** 2026-03-20
**Author:** Lead Architect (Opus), approved by Yusen (CEO/CTO)
**Status:** DRAFT — pending CEO approval

---

## 1. Overview

Vector OS Nano is an open-source Python SDK that gives robot arms a natural language brain. Users connect hardware, provide an LLM API key, and control their robot with plain language commands. The SDK handles perception, task planning, skill execution, and world state management.

The SDK is designed as a **generalizable robot framework** — the SO-101 arm is the first reference implementation (POV), but the architecture must support future complex robot systems: multi-arm, mobile bases, multi-sensor fusion, and distributed multi-agent coordination.

```
pip install vector-os-nano
```

```python
from vector_os import Agent, SO101

arm = SO101(port="/dev/ttyACM0")
agent = Agent(arm=arm, llm_api_key="sk-...")
agent.execute("pick up the red cup and place it on the left")
```

---

## 2. Background & Motivation

### 2.1 Problem Statement

Building an intelligent robot arm today requires:
- ROS2 installation and workspace setup (hours)
- MoveIt2 configuration (days)
- Custom perception pipeline (weeks)
- LLM integration glue code (days)
- Coordinate transform debugging (days)

A hobbyist with a SO-101 arm and a camera should be able to go from unboxing to natural language control in under 30 minutes.

### 2.2 Why Now

- LLM function calling is mature enough for reliable task planning
- EdgeTAM enables real-time object tracking without per-object training
- VLMs (Moondream, Qwen) provide zero-shot object detection from natural language
- Low-cost arms (SO-100/101, LeRobot, Koch) have created a hobbyist market
- No existing framework combines LLM planning + deterministic execution + world model in a single pip-installable package

### 2.3 Existing Work

Vector OS Nano builds on a proven codebase: the `vector_ws` workspace (9 ROS2 packages) that currently achieves 95%+ pick success with natural language control on the SO-101 arm. This spec defines the refactoring of that codebase into a generalizable SDK.

### 2.4 Strategic Context

Vector OS Nano (single arm, tabletop) is the entry point of the Vector OS product line:

```
Vector OS Nano    — single arm, tabletop manipulation (THIS SPEC)
Vector OS Small   — arm + mobile base, indoor navigation
Vector OS Medium  — multi-arm, warehouse/logistics
Vector OS Large   — humanoid / quadruped, full-body control
```

The SDK architecture must support this progression. Interfaces defined now must not need breaking changes to support multi-arm or mobile base in the future.

---

## 3. Goals

### MUST (v0.1 release criteria)

- M1: `pip install vector-os-nano` works on Linux (x86_64, Python 3.10+)
- M2: SO-101 arm connects and moves via pure Python (no ROS2 required)
- M3: Agent accepts natural language, decomposes into skill sequence, executes deterministically
- M4: World model tracks objects, robot state, and spatial relations across a session
- M5: Pick skill achieves 90%+ success rate across full workspace (center + edges)
- M6: Built-in skills: pick, place, home, scan, detect
- M7: LLM provider is pluggable (Claude, OpenAI, local models)
- M8: Perception works with Intel RealSense D405 (color + depth)
- M9: ROS2 integration available via `vector-os-nano[ros2]` optional dependency
- M10: Hardware abstraction protocol allows future arm implementations without core changes
- M11: Open source under MIT license

- M12: Textual TUI developer dashboard (terminal-based, works over SSH)
- M13: Interactive TUI calibration wizard
- M14: Model auto-download on first run (EdgeTAM, Moondream)
- M15: Simulation support (PyBullet) for users without hardware
- M16: ROS2 integration layer fully implemented (optional to enable at runtime, code ships in v0.1)
- M17: README with clear instructions for both ROS2 and non-ROS2 modes

### SHOULD (v0.2 targets)

- S1: Windows and macOS support for core (no ROS2)
- S2: Grasp success detection via servo current/load feedback
- S3: Look-then-move visual correction at pre-grasp pose
- S4: Session persistence (world model saved/loaded across runs)
- S5: Skill composition (multi-step skills defined declaratively)
- S6: Support for SO-100 arm (Feetech SCS protocol variant)

### MAY (future roadmap)

- F1: Multi-arm coordination protocol
- F2: Mobile base integration (Nav2 compatible)
- F3: Behavior tree execution engine (upgrade from sequential executor)
- F4: Learning from experience (success/failure → offset corrections)
- F5: Voice input (microphone → ASR → agent)
- F6: Web-based remote monitoring dashboard
- F7: MCP server for LLM-to-robot bridge (Claude Desktop integration)

---

## 4. Non-Goals

- NG1: This is NOT a ROS2 replacement. ROS2 is an optional integration layer that provides additional capabilities.
- NG2: This is NOT a general-purpose robotics framework (like ROS2 itself). It is opinionated about the agent architecture.
- NG3: This is NOT a real-time control system. The LLM planner is inherently non-real-time. Execution uses soft-RT (servo-rate control via serial).
- NG4: No support for industrial robots or safety-critical applications in v0.1.
- NG5: No GUI (graphical). The developer terminal is TUI-only. Camera feed is a companion window.
- NG6: No cloud services or telemetry. The SDK runs fully offline (except LLM API calls).
- NG7: No custom hardware design. The SDK works with off-the-shelf arms and cameras.

---

## 5. User Scenarios

### Scenario 1: Hobbyist Quick Start (Windows/Linux, no ROS2)

- **Actor:** Hobbyist with SO-101 arm, D405 camera, Windows laptop with NVIDIA GPU
- **Trigger:** `pip install vector-os-nano` + `python quickstart.py`
- **Expected Behavior:**
  1. SDK auto-detects serial port and camera
  2. User types `agent.execute("pick up the red cup")`
  3. Camera detects cup via VLM, arm moves to pick position, grasps, lifts
  4. Agent reports success with world model update
- **Success Criteria:**
  - From pip install to first successful pick: < 30 minutes
  - No ROS2 installation required
  - No manual calibration required (auto-calibration or reasonable defaults)

### Scenario 2: Multi-Step Task (natural language decomposition)

- **Actor:** User with calibrated system
- **Trigger:** `agent.execute("sort the red blocks by size from left to right")`
- **Expected Behavior:**
  1. Planner queries world model for current objects
  2. Planner calls LLM to decompose: detect → identify sizes → plan placement order → pick/place sequence
  3. Executor runs plan step by step with pre/postcondition checks
  4. On failure (e.g., drop), executor pauses and planner generates recovery plan
  5. World model updated after each successful step
- **Success Criteria:**
  - Correct decomposition of 3+ step tasks
  - At least one retry on failure before giving up
  - World model reflects final state accurately

### Scenario 3: Custom Skill Registration

- **Actor:** Developer extending the SDK
- **Trigger:** User defines a new skill (e.g., "pour" for a cup)
- **Expected Behavior:**
  ```python
  from vector_os import Skill, SkillResult

  class PourSkill(Skill):
      name = "pour"
      description = "Tilt the held object to pour its contents"
      parameters = {"angle_deg": {"type": "float", "default": 90.0}}
      preconditions = ["gripper_holding_any"]
      postconditions = ["gripper_holding_any"]  # still holding after pour

      def execute(self, params, context):
          current_joints = context.arm.get_joint_positions()
          # Rotate wrist by angle
          target = current_joints.copy()
          target[4] += math.radians(params["angle_deg"])
          context.arm.move_joints(target, duration=2.0)
          return SkillResult(success=True)

  agent.register_skill(PourSkill())
  agent.execute("pour the cup")
  # LLM planner automatically discovers and uses the new skill
  ```
- **Success Criteria:**
  - Custom skill is discoverable by the LLM planner
  - Pre/postconditions are enforced by executor
  - No core SDK code modification required

### Scenario 4: ROS2 Power User

- **Actor:** Roboticist on Ubuntu with ROS2 Humble
- **Trigger:** `pip install vector-os-nano[ros2]` + launch file
- **Expected Behavior:**
  1. Full ROS2 node graph starts (hardware_bridge, perception, skills, world_model, planner)
  2. All nodes visible in `ros2 node list`, topics in `ros2 topic list`
  3. Skills available as both Python API and ROS2 action servers
  4. World model queryable via ROS2 service
  5. TF2 transforms published for all known objects
  6. Textual dashboard shows full system status
- **Success Criteria:**
  - All existing `vector_ws` functionality preserved
  - ROS2 mode adds: lifecycle management, TF2, action feedback, topic monitoring
  - Same Agent API regardless of mode

### Scenario 5: Future — Multi-Arm System

- **Actor:** Researcher with two SO-101 arms
- **Trigger:** (Future, not implemented in v0.1)
- **Expected Behavior:**
  ```python
  left_arm = SO101(port="/dev/ttyACM0", name="left")
  right_arm = SO101(port="/dev/ttyACM1", name="right")
  agent = Agent(arms=[left_arm, right_arm], ...)
  agent.execute("left arm picks the cup, right arm holds the plate")
  ```
- **Success Criteria:**
  - The `ArmProtocol` and `SkillContext` interfaces support this without breaking changes
  - v0.1 does NOT implement multi-arm, but the protocol does NOT prevent it

---

## 6. Technical Constraints

### 6.1 Runtime Environment

- **Python:** 3.10+ (ROS2 Humble compatibility)
- **OS (core):** Linux x86_64 (v0.1), Windows/macOS (v0.2)
- **OS (ROS2):** Ubuntu 22.04 LTS only
- **GPU:** NVIDIA with CUDA (EdgeTAM + VLM require GPU)
- **GPU VRAM:** 8GB minimum (16GB recommended for EdgeTAM + Moondream)

### 6.2 Dependency Budget

Core dependencies (must be pip-installable, no system packages):

| Dependency | Purpose | Cross-platform |
|-----------|---------|----------------|
| numpy | Math | Yes |
| pyserial | Servo communication | Yes |
| pyrealsense2 | Camera driver | Yes (wheels) |
| torch + torchvision | EdgeTAM, VLM inference | Yes (CUDA) |
| transformers | EdgeTAM model loading | Yes |
| open3d | Pointcloud processing | Yes |
| pinocchio (pin) | FK/IK solver | Yes (wheels) |
| httpx | LLM API calls | Yes |
| pyyaml | Configuration | Yes |
| textual | TUI dashboard (optional) | Yes |

ROS2 dependencies (optional, linux only):

| Dependency | Purpose |
|-----------|---------|
| rclpy | ROS2 Python client |
| sensor_msgs, geometry_msgs, vision_msgs | Standard messages |
| control_msgs, action_msgs | Action interfaces |
| tf2_ros | Transform management |
| moveit_msgs | MoveIt2 integration (optional) |

### 6.3 Performance Requirements

| Metric | Target |
|--------|--------|
| Agent cold start (no ROS2) | < 30 seconds |
| Agent cold start (ROS2 mode) | < 60 seconds |
| VLM detection latency | < 1 second |
| EdgeTAM tracking FPS | >= 15 fps |
| LLM planning latency | < 5 seconds per plan |
| Skill execution overhead | < 100ms (excluding arm motion) |
| Pick cycle time | < 15 seconds (detect + approach + grasp + lift) |
| World model query | < 10ms |

### 6.4 Compatibility Requirements

- SO-101 arm with Feetech STS3215 servos (SCS serial protocol)
- Intel RealSense D405 depth camera
- Compatible with LeRobot SO-100 firmware (SHOULD, v0.2)

---

## 7. Architecture Overview

### 7.1 Layered Design

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                      │
│    agent.execute("pick up the red cup")                  │
├─────────────────────────────────────────────────────────┤
│                    Agent Engine                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │ Planner  │  │ Executor │  │ World Model          │   │
│  │ (LLM)    │→ │ (determ) │→ │ (objects, robot,     │   │
│  │          │← │          │← │  relations, history) │   │
│  └──────────┘  └──────────┘  └──────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                    Skill Layer                            │
│  ┌──────┐ ┌───────┐ ┌──────┐ ┌──────┐ ┌────────┐       │
│  │ Pick │ │ Place │ │ Home │ │ Scan │ │ Custom │       │
│  └──┬───┘ └───┬───┘ └──┬───┘ └──┬───┘ └───┬────┘       │
│     └─────────┴────────┴────────┴─────────┘             │
│                    SkillProtocol                          │
├─────────────────────────────────────────────────────────┤
│               Hardware Abstraction Layer                  │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐  │
│  │ ArmProtocol   │  │ PerceptionProt│  │ GripperProt │  │
│  │  - move_joints│  │  - get_frame  │  │  - open     │  │
│  │  - get_joints │  │  - get_depth  │  │  - close    │  │
│  │  - fk / ik    │  │  - detect     │  │  - is_hold  │  │
│  └───────┬───────┘  └───────┬───────┘  └──────┬──────┘  │
│          │                  │                  │         │
├──────────┼──────────────────┼──────────────────┼─────────┤
│  Implementation Layer (swappable)                        │
│  ┌───────┴───────┐  ┌───────┴───────┐  ┌──────┴──────┐  │
│  │ SO101Arm      │  │ RealSenseCam  │  │ SO101Gripper│  │
│  │ (serial)      │  │ (pyrealsense2)│  │ (serial)    │  │
│  └───────────────┘  └───────────────┘  └─────────────┘  │
│                                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │ ROS2 Integration Layer (OPTIONAL)                 │   │
│  │  - ROS2 node wrappers for all protocols           │   │
│  │  - ExecuteSkill.action server                     │   │
│  │  - TF2 broadcasting                              │   │
│  │  - Lifecycle management                           │   │
│  │  - Topic-based perception bridge                  │   │
│  └───────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Core Principles

1. **Agent engine is hardware-agnostic.** The planner, executor, and world model never import hardware-specific code. They interact with hardware only through protocols.

2. **ROS2 is an integration layer, not a foundation.** The core SDK has zero ROS2 imports. ROS2 support is provided by adapter classes that implement the same protocols.

3. **Skills are the unit of robot capability.** Every robot action is a Skill with declared parameters, preconditions, postconditions, and effects. The LLM planner discovers skills from the registry.

4. **The LLM plans, the executor executes.** LLM is never in the control loop. Planning phase: LLM calls OK. Execution phase: zero LLM calls, deterministic Python only.

5. **World model is the single source of truth.** All agents (planner, executor, perception) read from and write to the world model. No hidden state.

6. **Protocols are designed for generalization.** `ArmProtocol` is not `SO101Protocol`. It supports any arm with joints and a gripper. Future arms (Koch, ALOHA, xArm) implement the same protocol.

---

## 8. Interface Definitions

### 8.1 Core Protocols

#### ArmProtocol

```python
class ArmProtocol(Protocol):
    """Abstract interface for any robot arm."""

    @property
    def name(self) -> str: ...

    @property
    def joint_names(self) -> list[str]: ...

    @property
    def dof(self) -> int: ...

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    def get_joint_positions(self) -> list[float]:
        """Current joint positions in radians."""
        ...

    def move_joints(self, positions: list[float], duration: float = 3.0) -> bool:
        """Move to target joint positions over duration seconds."""
        ...

    def move_cartesian(self, target_xyz: tuple[float, float, float],
                       duration: float = 3.0) -> bool:
        """Move end-effector to target position (IK solved internally)."""
        ...

    def fk(self, joint_positions: list[float]) -> tuple[list[float], list[list[float]]]:
        """Forward kinematics: joints → (position_xyz, rotation_3x3)."""
        ...

    def ik(self, target_xyz: tuple[float, float, float],
           current_joints: list[float] | None = None) -> list[float] | None:
        """Inverse kinematics: target position → joint positions or None."""
        ...

    def stop(self) -> None:
        """Emergency stop. Immediately halt all motion."""
        ...
```

#### GripperProtocol

```python
class GripperProtocol(Protocol):
    """Abstract interface for any gripper."""

    def open(self) -> bool: ...
    def close(self) -> bool: ...

    def is_holding(self) -> bool:
        """Returns True if gripper is closed on an object.
        Uses encoder position, current sensing, or force feedback
        depending on hardware capability."""
        ...

    def get_position(self) -> float:
        """Normalized position: 0.0 = fully closed, 1.0 = fully open."""
        ...

    def get_force(self) -> float | None:
        """Current grip force in Newtons, or None if not available."""
        ...
```

#### PerceptionProtocol

```python
class PerceptionProtocol(Protocol):
    """Abstract interface for robot perception."""

    def get_color_frame(self) -> np.ndarray:
        """Returns BGR image as (H, W, 3) uint8 array."""
        ...

    def get_depth_frame(self) -> np.ndarray:
        """Returns depth image as (H, W) uint16 array in millimeters."""
        ...

    def get_intrinsics(self) -> CameraIntrinsics:
        """Camera intrinsic parameters (fx, fy, cx, cy, width, height)."""
        ...

    def detect(self, query: str) -> list[Detection]:
        """Detect objects matching natural language query.
        Returns list of Detection(label, bbox, confidence)."""
        ...

    def track(self, detections: list[Detection]) -> list[TrackedObject]:
        """Initialize or continue tracking detected objects.
        Returns list of TrackedObject(track_id, mask, bbox_3d, pose)."""
        ...

    def get_point_cloud(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Returns (N, 3) point cloud in camera frame.
        If mask provided, returns only points within mask."""
        ...
```

#### LLMProvider Protocol

```python
class LLMProvider(Protocol):
    """Abstract interface for LLM backends."""

    def plan(self, goal: str, world_state: dict,
             skill_schemas: list[dict],
             history: list[dict] | None = None) -> TaskPlan:
        """Decompose a natural language goal into a task plan.

        Args:
            goal: Natural language instruction
            world_state: Current world model snapshot (JSON-serializable)
            skill_schemas: Available skills with parameter schemas
            history: Previous conversation/execution history

        Returns:
            TaskPlan with ordered steps, or a clarification request
        """
        ...

    def query(self, prompt: str, image: np.ndarray | None = None) -> str:
        """Free-form query (for scene description, error analysis, etc.)."""
        ...
```

#### Skill Protocol

```python
class Skill(Protocol):
    """Abstract interface for robot skills.
    Users implement this to add custom capabilities."""

    name: str
    description: str
    parameters: dict           # JSON Schema for skill parameters
    preconditions: list[str]   # Predicate expressions checked before execution
    postconditions: list[str]  # Predicate expressions checked after execution
    effects: dict              # World model mutations on success

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Execute the skill with given parameters.

        Args:
            params: Validated parameters matching self.parameters schema
            context: Access to arm, gripper, perception, world_model

        Returns:
            SkillResult(success, result_data, error_message)
        """
        ...
```

#### SkillContext

```python
@dataclass
class SkillContext:
    """Provided to skills during execution. Contains everything a skill needs."""

    arm: ArmProtocol
    gripper: GripperProtocol
    perception: PerceptionProtocol
    world_model: WorldModel
    calibration: Calibration
    logger: Logger

    # Future extensions (not implemented in v0.1, but present in interface)
    arms: dict[str, ArmProtocol] | None = None  # Multi-arm support
    base: Any | None = None                      # Mobile base support
```

### 8.2 Agent Engine

#### Agent (main entry point)

```python
class Agent:
    """The core orchestrator. Connects hardware, perception, and intelligence."""

    def __init__(
        self,
        arm: ArmProtocol,
        gripper: GripperProtocol | None = None,  # defaults to arm's built-in
        perception: PerceptionProtocol | None = None,
        llm: LLMProvider | None = None,
        llm_api_key: str | None = None,  # convenience: auto-creates ClaudeProvider
        skills: list[Skill] | None = None,  # additional custom skills
        config: dict | str | None = None,  # config dict or path to YAML
    ): ...

    def execute(self, instruction: str) -> ExecutionResult:
        """Execute a natural language instruction.

        Pipeline:
        1. Planner queries world model for current state
        2. Planner calls LLM to decompose instruction into task plan
        3. Executor validates plan (skills exist, params valid)
        4. Executor runs plan step-by-step:
           a. Check preconditions against world model
           b. Call skill.execute()
           c. Update world model with skill effects
           d. Check postconditions
        5. On failure: planner generates recovery plan, executor retries
        6. Returns final result with execution trace

        Returns:
            ExecutionResult with status, trace, world_model_diff
        """
        ...

    def register_skill(self, skill: Skill) -> None:
        """Register a custom skill. Immediately available to the planner."""
        ...

    @property
    def world(self) -> WorldModel:
        """Access the current world model state."""
        ...

    def calibrate(self) -> CalibrationResult:
        """Run interactive workspace calibration wizard."""
        ...

    def home(self) -> bool:
        """Convenience: move arm to home position."""
        ...

    def stop(self) -> None:
        """Emergency stop. Halt all motion immediately."""
        ...
```

#### WorldModel

```python
class WorldModel:
    """Persistent world state. Tracks objects, robot, and spatial relations."""

    @dataclass(frozen=True)
    class ObjectState:
        object_id: str
        label: str
        pose: Pose3D              # Position + orientation in base frame
        confidence: float         # Decays over time since last observation
        state: str                # "on_table", "grasped", "placed", "unknown"
        last_seen: float          # Timestamp (seconds since epoch)
        properties: dict          # {"color": "red", "size_cm": 3.2, ...}
        bbox_3d: BBox3D | None    # 3D bounding box if available

    @dataclass(frozen=True)
    class RobotState:
        joint_positions: list[float]
        gripper_state: str        # "open", "closed", "holding"
        held_object: str | None   # object_id of held object
        is_moving: bool
        ee_position: list[float]  # End-effector xyz from FK

    # Query methods
    def get_objects(self) -> list[ObjectState]: ...
    def get_object(self, object_id: str) -> ObjectState | None: ...
    def get_objects_by_label(self, label: str) -> list[ObjectState]: ...
    def get_robot(self) -> RobotState: ...
    def check_predicate(self, predicate: str) -> bool: ...
    def get_spatial_relations(self, object_id: str) -> dict: ...

    # Update methods (called by perception, executor, calibration)
    def update_from_detections(self, detections: list[TrackedObject]) -> None: ...
    def update_robot_state(self, joints: list[float], gripper: str) -> None: ...
    def apply_skill_effects(self, skill_name: str, params: dict, result: SkillResult) -> None: ...

    # Persistence
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

    # Serialization (for LLM planner context)
    def to_dict(self) -> dict: ...
```

#### TaskPlan and Executor

```python
@dataclass(frozen=True)
class TaskStep:
    step_id: str
    skill_name: str
    parameters: dict
    depends_on: list[str]       # step_ids that must complete first
    preconditions: list[str]    # checked before execution
    postconditions: list[str]   # checked after execution

@dataclass(frozen=True)
class TaskPlan:
    goal: str                   # Original natural language instruction
    steps: list[TaskStep]
    requires_clarification: bool = False
    clarification_question: str | None = None

@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    status: str                 # "completed", "failed", "partially_completed"
    steps_completed: int
    steps_total: int
    failed_step: TaskStep | None
    failure_reason: str | None
    trace: list[StepTrace]      # Detailed trace of each step
    world_model_diff: dict      # Changes to world model during execution

class TaskExecutor:
    """Deterministic task execution engine.
    NO LLM calls during execution. Pure Python logic."""

    def execute(self, plan: TaskPlan, context: SkillContext) -> ExecutionResult:
        """Execute a task plan step by step.

        For each step in topological order:
        1. Wait for dependencies to complete
        2. Check preconditions against world model
        3. Call skill.execute(params, context)
        4. Update world model with skill effects
        5. Check postconditions
        6. On any failure, return immediately with failure info
        """
        ...
```

### 8.3 ROS2 Integration Interfaces

These are only relevant when `vector-os-nano[ros2]` is installed.

#### ExecuteSkill.action

```
# vector_os_interfaces/action/ExecuteSkill.action

# Goal
string skill_name
string parameters_json

---

# Result
bool success
string result_json
string error_message

---

# Feedback
string phase          # "precondition", "executing", "postcondition"
float32 progress      # 0.0 to 1.0
string status_message
```

#### WorldModelQuery.srv

```
# vector_os_interfaces/srv/WorldModelQuery.srv

string query_json

---

bool success
string result_json
```

#### PlanTask.srv

```
# vector_os_interfaces/srv/PlanTask.srv

string goal_text
string context_json

---

bool success
string plan_json
string error_message
```

### 8.4 Configuration

```yaml
# config/default.yaml — shipped with SDK, user overrides

agent:
  max_planning_retries: 3
  max_execution_retries: 2
  planning_timeout_sec: 10.0

llm:
  provider: "claude"           # claude | openai | local
  model: "claude-sonnet-4-6"   # for planning
  api_base: "https://openrouter.ai/api/v1"
  temperature: 0.0
  max_tokens: 2048

arm:
  type: "so101"
  port: "/dev/ttyACM0"         # or "COM3" on Windows
  baudrate: 1000000

camera:
  type: "realsense"
  serial: ""                   # empty = auto-detect
  resolution: [640, 480]
  fps: 30

perception:
  vlm_provider: "moondream"
  vlm_model: "vikhyatk/moondream2"
  tracker: "edgetam"
  tracker_model: "yonigozlan/EdgeTAM-hf"

calibration:
  file: ""                     # empty = use defaults / auto-calibrate
  method: "affine"             # affine | rbf | tps
  num_points: 25

skills:
  pick:
    z_offset: 0.12
    pre_grasp_height: 0.06
    max_retries: 2
  place:
    default_height: 0.05
  home:
    joint_values: [-0.014, -1.238, 0.562, 0.858, 0.311]

ros2:
  enabled: false               # auto-detected if rclpy available
  namespace: ""
  enable_moveit: false
  enable_tf2: true
  enable_dashboard: false
```

---

## 9. Package Structure

```
vector-os-nano/
├── pyproject.toml                  # Build config, dependencies, extras
├── LICENSE                         # MIT
├── README.md                       # Quick start guide
│
├── vector_os/
│   ├── __init__.py                 # Public API: Agent, SO101, Skill, ...
│   ├── version.py                  # __version__
│   │
│   ├── core/                       # Agent engine (pure Python)
│   │   ├── __init__.py
│   │   ├── agent.py                # Agent class — main entry point
│   │   ├── planner.py              # LLM task planner
│   │   ├── executor.py             # Deterministic task executor
│   │   ├── world_model.py          # World state management
│   │   ├── skill.py                # Skill protocol, registry, built-in predicates
│   │   ├── types.py                # Shared types (Pose3D, BBox3D, Detection, etc.)
│   │   └── config.py               # Configuration loading and validation
│   │
│   ├── llm/                        # LLM providers (pure Python)
│   │   ├── __init__.py
│   │   ├── base.py                 # LLMProvider protocol
│   │   ├── claude.py               # Anthropic / OpenRouter
│   │   ├── openai_compat.py        # OpenAI-compatible APIs
│   │   ├── local.py                # Ollama / local model
│   │   └── prompts.py              # System prompts for planning
│   │
│   ├── perception/                 # Perception stack (pure Python + GPU)
│   │   ├── __init__.py
│   │   ├── base.py                 # PerceptionProtocol
│   │   ├── realsense.py            # RealSense camera driver
│   │   ├── vlm.py                  # VLM detection (Moondream/Qwen)
│   │   ├── tracker.py              # EdgeTAM tracking wrapper
│   │   ├── pointcloud.py           # RGBD → 3D pointcloud
│   │   └── calibration.py          # Camera-to-arm calibration
│   │
│   ├── hardware/                   # Hardware abstraction (pure Python)
│   │   ├── __init__.py
│   │   ├── arm.py                  # ArmProtocol
│   │   ├── gripper.py              # GripperProtocol
│   │   ├── so101/                  # SO-101 implementation
│   │   │   ├── __init__.py
│   │   │   ├── arm.py              # SO101Arm(ArmProtocol)
│   │   │   ├── gripper.py          # SO101Gripper(GripperProtocol)
│   │   │   ├── joint_config.py     # Encoder/radian mapping
│   │   │   ├── serial_bus.py       # SCS protocol communication
│   │   │   └── ik_solver.py        # Pinocchio IK for SO-101
│   │   └── urdf/                   # URDF models
│   │       └── so101.urdf
│   │
│   ├── skills/                     # Built-in skills (pure Python)
│   │   ├── __init__.py
│   │   ├── pick.py                 # PickSkill
│   │   ├── place.py                # PlaceSkill
│   │   ├── home.py                 # HomeSkill
│   │   ├── scan.py                 # ScanSkill
│   │   └── detect.py               # DetectSkill
│   │
│   ├── cli/                        # Command-line interfaces
│   │   ├── __init__.py
│   │   ├── simple.py               # readline CLI (entry point)
│   │   └── dashboard.py            # Textual TUI (SHOULD, v0.2)
│   │
│   └── ros2/                       # ROS2 integration (optional)
│       ├── __init__.py             # Conditional import guard
│       ├── adapter.py              # ROS2SkillAdapter, ROS2PerceptionAdapter
│       ├── nodes/
│       │   ├── agent_node.py       # ROS2 Agent node
│       │   ├── hardware_bridge.py  # ROS2 hardware bridge node
│       │   ├── perception_node.py  # ROS2 perception bridge
│       │   ├── world_model_node.py # ROS2 world model service
│       │   └── skill_server.py     # ExecuteSkill action server
│       ├── interfaces/             # ROS2 msg/srv/action (separate CMake pkg)
│       └── launch/
│           ├── nano.launch.py      # Full system launch
│           ├── arm.launch.py
│           └── perception.launch.py
│
├── config/
│   └── default.yaml                # Default configuration
│
├── examples/
│   ├── quickstart.py               # 10-line hello world
│   ├── custom_skill.py             # Define and register a skill
│   ├── multi_step_task.py          # Complex task decomposition
│   ├── calibration.py              # Run calibration wizard
│   └── ros2_launch.py              # ROS2 mode example
│
├── tests/
│   ├── unit/                       # Unit tests (no hardware)
│   │   ├── test_world_model.py
│   │   ├── test_executor.py
│   │   ├── test_planner.py
│   │   ├── test_skill_registry.py
│   │   └── test_ik_solver.py
│   ├── integration/                # Integration tests (mock hardware)
│   │   ├── test_agent_flow.py
│   │   ├── test_pick_skill.py
│   │   └── test_perception_pipeline.py
│   └── hardware/                   # Hardware tests (real arm required)
│       ├── test_so101_connection.py
│       └── test_real_pick.py
│
└── docs/
    ├── getting-started.md
    ├── architecture.md
    ├── custom-skills.md
    ├── ros2-integration.md
    └── api-reference.md
```

---

## 10. Test Contracts

### 10.1 Unit Tests (no hardware, no GPU, no network)

- [ ] `test_world_model_add_object`: Adding an object updates the objects dict
- [ ] `test_world_model_confidence_decay`: Object confidence decreases over time
- [ ] `test_world_model_predicate_gripper_empty`: Predicate evaluates correctly
- [ ] `test_world_model_spatial_relations`: left_of/right_of computed from poses
- [ ] `test_world_model_serialization`: to_dict() → from_dict() roundtrip
- [ ] `test_executor_precondition_fail`: Executor stops if precondition not met
- [ ] `test_executor_postcondition_fail`: Executor stops if postcondition not met
- [ ] `test_executor_dependency_order`: Steps execute in topological order
- [ ] `test_executor_success_trace`: Successful execution returns full trace
- [ ] `test_skill_registry_add`: Custom skill registered and discoverable
- [ ] `test_skill_registry_schema`: Skill schemas serialize to valid JSON
- [ ] `test_planner_prompt_construction`: System prompt includes world state + skills
- [ ] `test_planner_parse_plan`: JSON plan parsed into TaskPlan correctly
- [ ] `test_planner_clarification`: Ambiguous input returns clarification request
- [ ] `test_ik_solver_roundtrip`: FK(IK(target)) == target within 2mm
- [ ] `test_ik_solver_unreachable`: Returns None for unreachable positions
- [ ] `test_joint_config_enc_rad_roundtrip`: enc→rad→enc within ±1 count
- [ ] `test_joint_config_nan_safety`: NaN/Inf inputs handled gracefully
- [ ] `test_config_load_defaults`: Default config loads without error
- [ ] `test_config_override`: User config overrides defaults correctly

### 10.2 Integration Tests (mock hardware, may need GPU)

- [ ] `test_agent_simple_pick`: Agent.execute("pick the cup") with mock arm succeeds
- [ ] `test_agent_multi_step`: Agent decomposes and executes 3-step task
- [ ] `test_agent_failure_retry`: Agent retries on pick failure
- [ ] `test_agent_clarification`: Agent asks for clarification on ambiguous input
- [ ] `test_perception_detect_track`: VLM detect → EdgeTAM track pipeline works
- [ ] `test_perception_3d_position`: RGBD → 3D position within 5mm of ground truth
- [ ] `test_calibration_affine`: Affine calibration reduces error below 5mm
- [ ] `test_skill_registration`: Custom skill usable by agent after register
- [ ] `test_world_model_persistence`: Save → restart → load preserves state

### 10.3 System Tests (real hardware)

- [ ] `test_so101_connect_disconnect`: Serial connection lifecycle works
- [ ] `test_so101_home`: Arm moves to home position
- [ ] `test_so101_pick_center`: Pick succeeds for object at workspace center
- [ ] `test_so101_pick_edge`: Pick succeeds for object at workspace edge
- [ ] `test_so101_pick_and_place`: Full pick-and-place cycle completes
- [ ] `test_so101_grasp_detection`: is_holding() correctly reports grasp state
- [ ] `test_ros2_node_lifecycle`: All ROS2 nodes start and stop cleanly
- [ ] `test_ros2_skill_action`: ExecuteSkill action completes via ROS2

---

## 11. Acceptance Criteria

### v0.1 Release

- [ ] AC1: `pip install vector-os-nano` on clean Ubuntu 22.04 succeeds in < 5 minutes
- [ ] AC2: `examples/quickstart.py` runs to completion with SO-101 + D405
- [ ] AC3: `agent.execute("pick up the [object]")` succeeds 90%+ (20 trials, mixed positions)
- [ ] AC4: `agent.execute("pick X and place it at Y")` succeeds 80%+ (10 trials)
- [ ] AC5: Custom skill registration works per Scenario 3
- [ ] AC6: World model correctly tracks 3+ objects across 10+ agent actions
- [ ] AC7: Unit test coverage >= 80%
- [ ] AC8: All integration tests pass with mock hardware
- [ ] AC9: ROS2 mode (`pip install vector-os-nano[ros2]`) launches all nodes
- [ ] AC10: README quick start guide verified by non-developer (Yusen) in < 30 minutes
- [ ] AC11: No hardcoded secrets, paths, or API keys in source
- [ ] AC12: MIT license header on all source files

---

## 12. Migration Plan (vector_ws → vector-os-nano)

The existing `vector_ws` codebase (9 ROS2 packages, 5000+ lines) will be refactored:

| vector_ws source | vector-os-nano destination | Change |
|-----------------|---------------------------|--------|
| `so101_hardware/joint_config.py` | `hardware/so101/joint_config.py` | Direct move |
| `so101_hardware/hardware_bridge.py` | `hardware/so101/arm.py` + `ros2/nodes/hardware_bridge.py` | Split: serial logic → arm.py, ROS2 wrapper → ros2/ |
| `so101_skills/pinocchio_ik.py` | `hardware/so101/ik_solver.py` | Direct move |
| `so101_skills/skill_node_v2.py` | `skills/*.py` + `ros2/nodes/skill_server.py` | Split: skill logic → individual skills, ROS2 → ros2/ |
| `so101_skills/perception_skills.py` | `core/agent.py` (integrated into agent) | Absorbed into agent perception queries |
| `so101_skills/calibrate_workspace.py` | `perception/calibration.py` | Refactor to class-based API |
| `track_anything/track_3d.py` | `perception/tracker.py` + `ros2/nodes/perception_node.py` | Split: tracker logic → perception/, ROS2 → ros2/ |
| `track_anything/edge_tam.py` | `perception/tracker.py` | Integrated |
| `vlm/models/*.py` | `perception/vlm.py` + `llm/` | VLM detection → perception/, LLM → llm/ |
| `so101_bringup/cli.py` | `cli/simple.py` | Remove ROS2 dependency, use direct calls |
| `so101_bringup/llm_client.py` | `llm/claude.py` | Refactor to LLMProvider protocol |
| `so101_perception/camera.launch.py` | `perception/realsense.py` + `ros2/launch/` | Split: camera driver → perception/, launch → ros2/ |
| `vector_perception_utils/*.py` | `perception/pointcloud.py` | Consolidate utilities |
| `so101_description/urdf/` | `hardware/urdf/` | Direct move |
| `so101_moveit_config/` | `ros2/` (optional) | Only used in ROS2 mode |

**Key principle:** Every split follows the pattern "extract pure Python logic + optional ROS2 wrapper." The ROS2 wrapper imports the pure Python class and wraps it as a ROS2 node.

---

## 13. Open Questions

1. **Calibration UX:** Should the SDK include an interactive calibration wizard (GUI-like TUI), or is a script-based approach sufficient? The current `calibrate_workspace.py` requires measuring with a ruler.

2. **Model distribution:** EdgeTAM and Moondream models are 1-2GB each. Should they auto-download on first run (current approach), be bundled in the pip package (too large), or require manual download?

3. **SO-100 vs SO-101:** The SO-100 uses the same STS3215 servos but different URDF/joint limits. How much shared code vs separate implementations?

4. **Namespace for multi-arm:** When supporting multiple arms, should naming be positional (`left`, `right`) or indexed (`arm_0`, `arm_1`)?

5. **Simulation support priority:** Is PyBullet/MuJoCo simulation needed for v0.1, or can we defer to v0.2? Simulation would allow testing without hardware.

6. **CLI vs Dashboard priority:** Should v0.1 ship with the simple readline CLI only, or include the Textual dashboard? The dashboard is significantly more work but much better UX.

7. **Version compatibility:** Should the SDK support Python 3.10+ only (ROS2 Humble), or also 3.11/3.12 (breaking some ROS2 compatibility)?

---

## 14. Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-03-20 | 0.1.0-draft | Lead Architect (Opus) | Initial spec from CEO architecture discussion |
