# Vector OS Nano — Testing Guide

Step-by-step verification for developers and contributors. Run these tests in order after installation to confirm everything works.

## Prerequisites

See `docs/dependencies.md` for detailed version requirements and troubleshooting.

### Quick Setup

```bash
# Create isolated venv
python3.10 -m venv ~/vector_os_nano
source ~/vector_os_nano/bin/activate
pip install --upgrade pip setuptools wheel

# Install Vector OS Nano (core only, no GPU dependencies)
cd ~/Desktop/vector_os_nano
pip install -e "."

# For GPU support (RTX 5080 Blackwell — MUST use nightly)
pip install --upgrade torch torchvision -i https://download.pytorch.org/whl/nightly/cu128

# Install all optional features + dev tools
pip install -e ".[all,dev]"
```

### Per-Feature Installation

If you only need specific features (saves space/time):

```bash
# Perception only (camera + VLM)
pip install -e ".[perception]"

# IK solver only
pip install -e ".[ik]"
pip install pin>=3.9.0

# Dashboard TUI only
pip install -e ".[tui]"

# PyBullet simulation only
pip install -e ".[sim]"

# Testing tools
pip install -e ".[dev]"
```

### Critical GPU Setup for RTX 5080

**Do NOT use stable PyTorch.** RTX 5080 requires CUDA sm_120 support, only in nightly.

```bash
# Correct (nightly with cu128):
pip install --upgrade torch torchvision -i https://download.pytorch.org/whl/nightly/cu128

# Verify installation
python -c "import torch; print(torch.__version__)"
# Expected: 2.12.0.dev20260320+cu128 (or newer nightly)
```

### Hardware Support

For SO-101 arm control, scservo_sdk must be installed (NOT on PyPI):

```bash
# Copy from ROS2 workspace
cp ~/ros2_ws/src/so101_hardware/scservo_sdk \
   ~/vector_os_nano/lib/python3.10/site-packages/

# Verify
python -c "from scservo_sdk import ServoCommandBus; print('OK')"
```

---

## Test 1: Unit Tests (530+ tests)

Runs unit tests covering types, config, world model, executor, skills, joint config, IK, LLM prompts, calibration.

```bash
python -m pytest tests/unit/ -v --tb=short
```

Expected: all pass, 0 failures. Some tests may skip if optional dependencies are not installed:
- scipy, pybullet skipped if not installed
- transformers tests skipped if transformers<4.57.6

---

## Test 2: Core Imports

Verify all SDK modules load correctly.

```bash
python -c "
from vector_os_nano.core.agent import Agent
from vector_os_nano.core.types import Pose3D, Detection, TaskPlan, ExecutionResult
from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.core.executor import TaskExecutor
from vector_os_nano.core.config import load_config
from vector_os_nano.llm import ClaudeProvider, OpenAIProvider, LocalProvider
from vector_os_nano.hardware.so101.joint_config import JOINT_CONFIG, ARM_JOINT_NAMES
from vector_os_nano.ros2 import ROS2_AVAILABLE
print('All imports OK')
print(f'ROS2 available: {ROS2_AVAILABLE}')
"
```

Expected: `All imports OK`. ROS2 availability depends on your system.

---

## Test 3: Agent Without Hardware

Verify the agent initializes and handles missing hardware gracefully.

```bash
python examples/test_agent_no_hardware.py
```

Create this file if missing:

```python
# examples/test_agent_no_hardware.py
from vector_os_nano.core.agent import Agent

agent = Agent()
print(f"Skills: {agent.skills}")

r = agent.execute("home")
print(f"home: success={r.success}, reason={r.failure_reason}")
# Expected: success=False, reason=No arm connected

r = agent.execute("fly")
print(f"fly: success={r.success}, reason={r.failure_reason}")
# Expected: success=False, reason=Unknown command: 'fly'. No LLM configured.
```

---

## Test 4: World Model

Verify object tracking, predicates, spatial relations, and state transitions.

```bash
python examples/test_world_model.py
```

```python
# examples/test_world_model.py
import time
from vector_os_nano.core.agent import Agent
from vector_os_nano.core.world_model import ObjectState
from vector_os_nano.core.types import SkillResult

agent = Agent()
wm = agent.world

# Add objects
cup = ObjectState(object_id="cup_1", label="red cup", x=0.25, y=0.05, z=0.02,
                  confidence=0.95, state="on_table", last_seen=time.time())
battery = ObjectState(object_id="bat_1", label="battery", x=0.20, y=-0.08, z=0.01,
                      confidence=0.88, state="on_table", last_seen=time.time())
wm.add_object(cup)
wm.add_object(battery)
print(f"Objects: {[o.label for o in wm.get_objects()]}")
# Expected: ['red cup', 'battery']

# Predicates
print(f"Gripper empty? {wm.check_predicate('gripper_empty')}")       # True
print(f"Cup visible? {wm.check_predicate('object_visible(cup_1)')}")  # True

# Spatial relations
relations = wm.get_spatial_relations("cup_1")
print(f"Cup spatial: {relations}")
# Expected: battery in right_of and in_front_of

# State transitions
wm.apply_skill_effects("pick", {"object_id": "cup_1"}, SkillResult(success=True))
print(f"After pick: gripper={wm.get_robot().gripper_state}, held={wm.get_robot().held_object}")
# Expected: gripper=holding, held=cup_1

wm.apply_skill_effects("place", {}, SkillResult(success=True))
print(f"After place: gripper={wm.get_robot().gripper_state}, held={wm.get_robot().held_object}")
# Expected: gripper=open, held=None
```

---

## Test 5: IK Solver

Verify forward/inverse kinematics and workspace reachability. Requires `pip install pin`.

```bash
python examples/test_ik_solver.py
```

```python
# examples/test_ik_solver.py
import numpy as np
from vector_os_nano.hardware.so101.ik_solver import IKSolver

solver = IKSolver()
home = [-0.014, -1.238, 0.562, 0.858, 0.311]

# FK at home
pos, rot = solver.fk(home)
print(f"FK at home: [{pos[0]*100:.1f}, {pos[1]*100:.1f}, {pos[2]*100:.1f}] cm")
# Expected: approximately [13.4, 0.1, 25.7] cm

# IK to a target
target = (0.20, 0.0, 0.15)
solution, error = solver.ik_position(target, home)
print(f"IK to {target}: error={error*1000:.1f}mm")
# Expected: error < 5mm

# Workspace test
targets = [
    (0.15, 0.0, 0.10, "center near"),
    (0.25, 0.0, 0.10, "center far"),
    (0.20, 0.10, 0.10, "left"),
    (0.20, -0.10, 0.10, "right"),
    (0.50, 0.0, 0.10, "very far"),
]
for x, y, z, label in targets:
    sol, err = solver.ik_position((x, y, z), home)
    status = f"OK ({err*1000:.1f}mm)" if sol else "UNREACHABLE"
    print(f"  {label:15s} -> {status}")
# Expected: first 4 OK, last UNREACHABLE
```

---

## Test 6: LLM Planning (requires API key)

Verify LLM task decomposition. Requires an OpenRouter API key.

Setup: create `config/user.yaml` (DO NOT commit this file):
```yaml
llm:
  api_key: "your-openrouter-api-key"
  provider: "claude"
  model: "anthropic/claude-haiku-4-5"
  api_base: "https://openrouter.ai/api/v1"
```

```bash
python examples/test_llm_planning.py
```

```python
# examples/test_llm_planning.py
import time
from vector_os_nano.core.config import load_config
from vector_os_nano.core.world_model import WorldModel, ObjectState
from vector_os_nano.core.skill import SkillRegistry
from vector_os_nano.skills import get_default_skills
from vector_os_nano.llm.claude import ClaudeProvider

cfg = load_config("config/user.yaml")
llm = ClaudeProvider(api_key=cfg["llm"]["api_key"], model=cfg["llm"]["model"],
                     api_base=cfg["llm"]["api_base"])

registry = SkillRegistry()
for s in get_default_skills():
    registry.register(s)

wm = WorldModel()
wm.add_object(ObjectState(object_id="cup_1", label="red cup", x=0.25, y=0.05, z=0.02,
                           confidence=0.95, state="on_table", last_seen=time.time()))

# Test 1: Simple pick
plan = llm.plan("pick up the red cup", wm.to_dict(), registry.to_schemas())
print(f"Plan 'pick red cup': {len(plan.steps)} steps")
for s in plan.steps:
    print(f"  {s.step_id}: {s.skill_name}({s.parameters})")
# Expected: 1 step — pick with object reference

# Test 2: Ambiguous command
plan = llm.plan("put it there", wm.to_dict(), registry.to_schemas())
if plan.requires_clarification:
    print(f"Clarification: {plan.clarification_question}")
# Expected: asks for clarification

# Test 3: Free-form query
answer = llm.query("What is 2+2?")
print(f"Query: {answer}")
# Expected: "4" or similar
```

---

## Test 7: CLI Interactive Mode (requires API key)

Test the command-line interface with LLM support.

```bash
python examples/test_cli_interactive.py
```

```python
# examples/test_cli_interactive.py
import time
from vector_os_nano.core.agent import Agent
from vector_os_nano.core.config import load_config
from vector_os_nano.core.world_model import ObjectState
from vector_os_nano.cli.simple import SimpleCLI

cfg = load_config("config/user.yaml")
agent = Agent(llm_api_key=cfg["llm"]["api_key"], config="config/user.yaml")

# Pre-populate world model
agent.world.add_object(ObjectState(object_id="cup_1", label="red cup", x=0.25, y=0.05,
                                    z=0.02, confidence=0.95, state="on_table",
                                    last_seen=time.time()))

cli = SimpleCLI(agent=agent, verbose=True)
cli.run()
# Commands to try: help, status, skills, world, quit
```

---

## Test 8: Hardware Connection (requires SO-101 arm)

Verify serial communication with the SO-101 arm. **This will move the arm.**

Prerequisites:
- SO-101 arm connected via USB
- `scservo_sdk` installed in venv
- User in `dialout` group: `sudo adduser $USER dialout`

```bash
python examples/test_hardware_connection.py
```

```python
# examples/test_hardware_connection.py
"""Test SO-101 hardware — WILL MOVE THE ARM."""
import sys, os, time

port = "/dev/ttyACM0"
if not os.path.exists(port):
    print(f"ERROR: {port} not found. Is the SO-101 connected?")
    sys.exit(1)

from vector_os_nano.hardware.so101 import SO101Arm, SO101Gripper

arm = SO101Arm(port=port)
arm.connect()
print("CONNECTED!")

# Read joints
joints = arm.get_joint_positions()
print(f"Current joints (rad): {[round(j, 3) for j in joints]}")

# Move to home
input("Press Enter to move to HOME (arm will move)...")
arm.move_joints([-0.014, -1.238, 0.562, 0.858, 0.311], duration=3.0)
print("Home reached.")

# Gripper test
gripper = SO101Gripper(arm._bus)
input("Press Enter to OPEN gripper...")
gripper.open()
time.sleep(1.0)
print(f"Position: {gripper.get_position():.2f}")

input("Press Enter to CLOSE gripper...")
gripper.close()
time.sleep(1.0)
print(f"Position: {gripper.get_position():.2f}")
print(f"Is holding? {gripper.is_holding()}")

arm.disconnect()
print("PASSED!")
```

---

## Test 9: Full Pipeline (requires API key)

Test executor, world model updates, custom skills, and LLM discovery.

```bash
python examples/test_full_pipeline.py
```

```python
# examples/test_full_pipeline.py
import time
from vector_os_nano.core.agent import Agent
from vector_os_nano.core.config import load_config
from vector_os_nano.core.world_model import ObjectState
from vector_os_nano.core.types import SkillResult
from vector_os_nano.core.skill import SkillContext

cfg = load_config("config/user.yaml")
agent = Agent(llm_api_key=cfg["llm"]["api_key"], config="config/user.yaml")

# Add objects
agent.world.add_object(ObjectState(object_id="cup_1", label="red cup", x=0.25, y=0.05,
                                    z=0.02, confidence=0.95, state="on_table",
                                    last_seen=time.time()))

# Test: custom skill registered, LLM discovers it
class WaveSkill:
    name = "wave"
    description = "Wave the robot arm as a greeting"
    parameters = {"times": {"type": "int", "default": 3}}
    preconditions = []
    postconditions = []
    effects = {}
    def execute(self, params, context):
        print(f"  Waving {params.get('times', 3)} times! (simulated)")
        return SkillResult(success=True)

agent.register_skill(WaveSkill())
print(f"Skills: {agent.skills}")
# Expected: ['home', 'scan', 'detect', 'pick', 'place', 'wave']

# LLM should find and use wave skill
from vector_os_nano.llm.claude import ClaudeProvider
llm = ClaudeProvider(api_key=cfg["llm"]["api_key"], model=cfg["llm"]["model"],
                     api_base=cfg["llm"]["api_base"])
plan = llm.plan("wave 5 times", agent.world.to_dict(), agent._skill_registry.to_schemas())
print(f"LLM plan for 'wave 5 times': {[(s.skill_name, s.parameters) for s in plan.steps]}")
# Expected: [('wave', {'times': 5})]
```

---

## Test 10: Dashboard Testing

Test the Textual TUI dashboard with various hardware configurations.

### Test 10A: Dashboard Without Hardware (safest for testing)

Recommended for development and UI testing.

```bash
# Core + TUI
pip install -e ".[tui]"

# Launch dashboard without arm or camera
python -m vector_os_nano.cli.dashboard --no-arm --no-perception
```

Expected: Dashboard opens with 5 tabs (Dashboard, Log, Skills, World, Camera). All features available except hardware-dependent skills (pick, place, home). Use for:
- Testing UI/UX changes
- Iterating command input
- Verifying world model updates
- Camera tab displays grayscale test pattern (no live camera)

Keyboard shortcuts:
- `F1-F5`: Switch between tabs (Dashboard, Log, Skills, World, Camera)
- `F6`: Toggle fullscreen camera
- `/`: Focus command input
- `Ctrl+C`: Quit

### Test 10B: Dashboard With Perception Only (no arm)

For testing perception pipeline without arm control.

```bash
# Full install with perception + TUI
pip install -e ".[all]"

# Launch dashboard with camera + perception, no arm
python -m vector_os_nano.cli.dashboard --no-arm
```

Expected: Dashboard fully functional. Camera tab shows live D405 RGB/depth feed. Skills requiring arm motion (pick, place, home) fail gracefully with "No arm connected". Use for:
- Testing perception pipeline
- Verifying VLM detection in Camera tab
- Testing EdgeTAM tracking visualization
- LLM planning without hardware

### Test 10C: Full System Dashboard (all hardware)

End-to-end system test with all hardware connected.

```bash
# Connect SO-101 arm and RealSense D405 camera
python -m vector_os_nano.cli.dashboard
```

Expected: Full system operational. All skills available. Camera shows live tracking overlays. Monitor:
- Joint angles update in Dashboard tab (real-time progress bars)
- Camera tab shows object tracking (EdgeTAM masks)
- World model updates after detect/pick commands
- Skill execution logs in Log tab
- Status dots show connection state


## Test Summary

| # | Test | Requires | Validates | Time |
|---|------|----------|-----------|------|
| 1 | Unit tests | pytest | Types, config, world model, executor, skills | ~30s |
| 2 | Core imports | — | Module structure | ~2s |
| 3 | Agent no hardware | — | Graceful degradation | ~2s |
| 4 | World model | — | Objects, predicates, spatial, state | ~5s |
| 5 | IK solver | pin | FK/IK accuracy, reachability | ~10s |
| 6 | LLM planning | API key | Task decomposition | ~5s |
| 7 | CLI interactive | API key | Command interface | manual |
| 8 | Hardware | SO-101 + scservo_sdk | Serial, joints, gripper | manual |
| 9 | Full pipeline | API key | Custom skills, LLM discovery | ~10s |
| 10 | Dashboard | textual | TUI, tabs, keyboard shortcuts | manual |

**Total automated test time:** ~1 minute
