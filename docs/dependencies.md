# Vector OS Nano — Dependency Reference

Complete package versions and installation guide for Vector OS Nano SDK. Last updated: 2026-03-20.

This document covers:
1. **Core dependencies** — always installed
2. **Perception** — camera, VLM, object tracking
3. **IK** — inverse kinematics, motion planning
4. **TUI** — dashboard and terminal UI
5. **Simulation** — PyBullet physics
6. **Dev** — testing and development

---

## Installation Quick Start

```bash
# Create venv
python3.10 -m venv ~/vector_os_nano
source ~/vector_os_nano/bin/activate

# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Core SDK
cd ~/Desktop/vector_os
pip install -e "."

# Optional features (choose as needed)
pip install -e ".[perception]"    # Camera + VLM
pip install -e ".[ik]"            # IK solver
pip install -e ".[tui]"           # Dashboard
pip install -e ".[sim]"           # PyBullet
pip install -e ".[dev]"           # Testing tools

# All-in-one (CUDA 12.8, RTX 5080 ready)
pip install -e ".[all,dev]"

# ROS2 integration (requires separate apt install)
sudo apt install ros-humble-rclpy ros-humble-launch-ros
```

**Critical:** For GPU perception with RTX 5080 Blackwell (sm_120), use PyTorch nightly:

```bash
# This installs torch + torchvision nightly with cu128 support
pip install --upgrade torch torchvision -i https://download.pytorch.org/whl/nightly/cu128
```

---

## Core Dependencies

Always installed. No GPU required.

| Package | Version | Why Needed |
|---------|---------|-----------|
| **numpy** | >=1.24 | Numerical computations, joint angles, transformations |
| **pyserial** | >=3.5 | SO-101 arm serial communication (USB CDC) |
| **httpx** | >=0.25 | Async HTTP client for API calls (LLM providers) |
| **pyyaml** | >=6.0 | Configuration file parsing (config/*.yaml) |
| **filelock** | 3.25.2 | Concurrent file access in multiprocessing scenarios |
| **setuptools** | 78.1.0 | Package metadata and entry points |

**Note:** Core requires Python >=3.10. Tested on Python 3.10, 3.11, 3.12.

---

## Perception Dependencies

For camera input, vision language models, and object detection/tracking.

### Required for ALL perception features

| Package | Version | Why Needed |
|---------|---------|-----------|
| **torch** | 2.12.0.dev20260320+cu128 | Deep learning framework (CRITICAL: nightly build for Blackwell sm_120 support) |
| **torchvision** | 0.26.0.dev20260320+cu128 | Computer vision models (MUST match torch version, use nightly index) |
| **transformers** | **4.57.6** | HuggingFace models (PINNED: 4.44 lacks EdgeTam, 5.x has tied_weights bug) |
| **tokenizers** | 0.22.2 | Fast tokenization (comes with transformers, don't pin separately) |
| **safetensors** | 0.7.0 | Safe model serialization (used by transformers and timm) |
| **timm** | 1.0.25 | Vision backbone models (required by EdgeTAM, YOLO, etc.) |
| **huggingface_hub** | 0.36.2 | Model downloading and caching |
| **tqdm** | 4.67.3 | Progress bars for model downloads |

### Camera-specific

| Package | Version | Why Needed |
|---------|---------|-----------|
| **pyrealsense2** | 2.56.5.9235 | RealSense D455/D435 camera drivers |
| **opencv-python** | 4.13.0.92 | Image processing, calibration, color space conversions |

### Point cloud processing

| Package | Version | Why Needed |
|---------|---------|-----------|
| **open3d** | 0.19.0 | Point cloud filtering, visualization, voxelization |

### VLM models (via transformers 4.57.6)

Automatically available when transformers 4.57.6 is installed:
- **EdgeTAM** — open-vocabulary detection (requires timm 1.0.25)
- **Moondream** — lightweight visual QA
- **LLaVA** — vision language chat models
- **YOLO v11** — real-time object detection

### Advanced perception (optional)

| Package | Version | Why Needed |
|---------|---------|-----------|
| **scikit-learn** | 1.7.2 | Clustering, dimensionality reduction (optional) |
| **scipy** | 1.15.3 | Scientific computing (optional, required by some VLM postprocessing) |

**Installation:**
```bash
pip install -e ".[perception]"
# OR manually:
pip install pyrealsense2 torch torchvision transformers timm open3d opencv-python
```

---

## IK Dependencies

Inverse kinematics, motion planning, and workspace analysis for SO-101 arm.

| Package | Version | Why Needed |
|---------|---------|-----------|
| **pin** (Pinocchio) | 3.9.0 | Industrial-grade IK/FK, dynamics, joint limits validation |
| **eigenpy** | 3.12.0 | Python bindings for Eigen (compiled with pin) |
| **coal** | 3.0.2 | Collision detection for workspace planning |
| **urdfdom-py** | 1.2.1 | URDF parser (SO-101 arm model loading) |

**SO-101 URDF model** available in:
```
~/ros2_ws/src/so101_bringup/urdf/so101.urdf
```

**Installation:**
```bash
pip install -e ".[ik]"
# OR manually:
pip install pin urdfdom-py
```

**Verification:**
```bash
python examples/test_ik_solver.py
```

---

## TUI Dashboard Dependencies

Terminal UI with Textual framework.

| Package | Version | Why Needed |
|---------|---------|-----------|
| **textual** | 8.1.1 | TUI framework (tabs, panels, event loop) |
| **rich** | (via textual) | Colored output, tables, progress bars |

**Features:**
- Dashboard panel (live agent state, skills)
- Log panel (task execution trace)
- Skills panel (available commands)
- World panel (object tracking, spatial state)

**Installation:**
```bash
pip install -e ".[tui]"
# OR manually:
pip install textual
```

**Launch:**
```bash
vector-os-dashboard --no-arm
```

---

## Simulation Dependencies

PyBullet physics engine for behavior development without hardware.

| Package | Version | Why Needed |
|---------|---------|-----------|
| **pybullet** | 3.2.6 | Real-time physics simulation (gravity, collisions, joint limits) |
| **numpy** | >=1.24 | Gravity vectors, transformation matrices |

**Features:**
- SO-101 arm loaded from URDF
- Simulated gripper with contact forces
- Object interaction (pick/place physics)
- Workspace collision checking

**Installation:**
```bash
pip install -e ".[sim]"
# OR manually:
pip install pybullet
```

**Example:**
```python
from vector_os.simulation.pybullet_sim import PybulletSimulation
sim = PybulletSimulation(gui=True)
sim.load_so101_arm()
sim.step()
```

---

## ROS2 Integration

ROS2 Humble packages (installed via apt, not pip). Required for hardware nodes and system integration.

### Core ROS2 packages

| Package | Version | Via |
|---------|---------|-----|
| **rclpy** | 3.3.20 | apt: `ros-humble-rclpy` |
| **launch-ros** | 0.19.13 | apt: `ros-humble-launch-ros` |
| **rcl-interfaces** | 1.2.2 | apt: `ros-humble-rcl-interfaces` |

### Message/Service packages used by vector_os

| Package | Version | Via |
|---------|---------|-----|
| **std_msgs** | 4.9.1 | apt: `ros-humble-std-msgs` |
| **sensor_msgs** | 4.9.1 | apt: `ros-humble-sensor-msgs` |
| **geometry_msgs** | 4.9.1 | apt: `ros-humble-geometry-msgs` |
| **tf2-ros** | 0.25.19 | apt: `ros-humble-tf2-ros` |
| **tf2-geometry-msgs** | 0.25.19 | apt: `ros-humble-tf2-geometry-msgs` |
| **cv-bridge** | 3.2.1 | apt: `ros-humble-cv-bridge` |
| **image-transport** | 2.5.1 | apt: `ros-humble-image-transport` |

**Installation:**
```bash
# Full ROS2 Humble desktop
sudo apt install ros-humble-desktop

# Or minimal + navigation
sudo apt install \
  ros-humble-rclpy \
  ros-humble-launch-ros \
  ros-humble-sensor-msgs \
  ros-humble-geometry-msgs \
  ros-humble-tf2-ros \
  ros-humble-cv-bridge
```

---

## Development Dependencies

Testing, linting, and debugging tools.

| Package | Version | Why Needed |
|---------|---------|-----------|
| **pytest** | 7.0+ | Unit and integration test runner |
| **pytest-cov** | — | Coverage reporting (80%+ target) |
| **lark** | latest | Parser generation (ROS2 pytest plugin compatibility) |

**Installation:**
```bash
pip install -e ".[dev]"
# OR manually:
pip install pytest pytest-cov lark
```

**Disable ROS2 pytest plugins when running tests on non-ROS2 machine:**
```bash
python -m pytest tests/ -p no:launch_ros -p no:ament_copyright
```

---

## Hardware Requirement: scservo_sdk

**NOT available on PyPI.** Required for SO-101 arm control.

### Installation

1. **Copy from source repository:**
```bash
# In your venv
cp ~/ros2_ws/src/so101_hardware/scservo_sdk \
   ~/vector_os_nano/lib/python3.10/site-packages/scservo_sdk
```

2. **Or install from SO-101 source:**
```bash
cd ~/ros2_ws/src/so101_hardware
pip install -e .
```

3. **Or clone directly:**
```bash
git clone https://github.com/Vector-Robotics/scservo_sdk.git
cd scservo_sdk
pip install -e .
```

### Verification

```python
from scservo_sdk import *
bus = ServoCommandBus('/dev/ttyACM0')
# SUCCESS if no import error
```

---

## CRITICAL Version Constraints

### PyTorch for RTX 5080 Blackwell (sm_120)

**Do NOT use stable PyTorch.** RTX 5080 requires CUDA sm_120 support, only available in nightly builds.

```bash
# Correct (nightly with cu128):
pip install --upgrade torch torchvision -i https://download.pytorch.org/whl/nightly/cu128

# Wrong (will fail on RTX 5080):
pip install torch torchvision  # Uses stable with cu121
```

**Versions:**
- torch: 2.12.0.dev20260320+cu128 (or newer nightly)
- torchvision: 0.26.0.dev20260320+cu128 (MUST match torch version)

### Transformers 4.57.6

**Pinned to exact version:**
- 4.44: Lacks EdgeTAM model support
- 4.57.6: Has EdgeTAM, Moondream, LLaVA
- 5.x+: tied_weights bug with Moondream (BREAKS VLM)

```bash
pip install transformers==4.57.6
```

### Tokenizers

**Don't pin separately.** Installed automatically with transformers.

```bash
# Good:
pip install transformers==4.57.6
# -> includes tokenizers==0.22.2

# Bad:
pip install transformers==4.57.6 tokenizers==0.23.0
# -> will cause dependency conflict
```

### timm

**Required by EdgeTAM.** Use recent version (1.0.25+):

```bash
pip install timm>=1.0.25
```

### pin (Pinocchio)

**Version 3.9.0+** required for SO-101 URDF compatibility.

```bash
pip install pin>=3.9.0
```

### lark

**For ROS2 pytest plugins compatibility.** Install if you see:
```
pytest plugin launch_ros requires lark
```

```bash
pip install lark
```

---

## pyproject.toml Reference

Current exact pinning in `pyproject.toml`:

```toml
[project]
requires-python = ">=3.10"

dependencies = [
    "numpy>=1.24",
    "pyserial>=3.5",
    "httpx>=0.25",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
perception = [
    "pyrealsense2>=2.50",
    "torch>=2.0",
    "torchvision>=0.15",
    "transformers==4.57.6",  # PINNED
    "open3d>=0.17",
    "opencv-python>=4.8",
]
ik = [
    "pin>=3.9.0",
]
tui = [
    "textual>=0.40",
]
sim = [
    "pybullet>=3.2",
]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "lark",
]
```

---

## Build Verification

After installation, verify each component:

```bash
# Core imports
python -c "
from vector_os.core.agent import Agent
from vector_os.core.types import Pose3D, Detection
from vector_os.core.world_model import WorldModel
from vector_os.core.executor import TaskExecutor
print('✓ Core imports OK')
"

# Perception
python -c "
try:
    import torch; import transformers; import open3d
    print(f'✓ Perception OK: torch {torch.__version__}, transformers {transformers.__version__}')
except: print('✗ Perception not installed (optional)')
"

# IK
python -c "
try:
    import pinocchio as pin
    print(f'✓ IK OK: pinocchio {pin.__version__}')
except: print('✗ IK not installed (optional)')
"

# TUI
python -c "
try:
    from textual.app import App
    print('✓ TUI OK')
except: print('✗ TUI not installed (optional)')
"

# Simulation
python -c "
try:
    import pybullet
    print('✓ Simulation OK')
except: print('✗ Simulation not installed (optional)')
"

# Hardware
python -c "
try:
    from scservo_sdk import ServoCommandBus
    print('✓ Hardware SDK OK')
except: print('✗ Hardware SDK not installed (optional)')
"
```

Expected output:
```
✓ Core imports OK
✓ Perception OK: torch 2.12.0.dev20260320+cu128, transformers 4.57.6
✓ IK OK: pinocchio 3.9.0
✓ TUI OK
✓ Simulation OK
✓ Hardware SDK OK
```

---

## Testing Installation

See `docs/testing-guide.md` for step-by-step verification.

**Quick test:**
```bash
python -m pytest tests/unit -v --tb=short
```

Expected: 530+ tests pass, 0 failures.

---

## Troubleshooting

### `ImportError: cannot import name 'EdgeTam'`

**Cause:** transformers version mismatch
**Fix:**
```bash
pip install transformers==4.57.6 --force-reinstall
```

### `No CUDA visible devices / CUDA compute capability not supported`

**Cause:** PyTorch stable doesn't support RTX 5080
**Fix:**
```bash
pip install --upgrade torch torchvision -i https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
```

### `ModuleNotFoundError: No module named 'scservo_sdk'`

**Cause:** Hardware SDK not in venv
**Fix:**
```bash
cp ~/ros2_ws/src/so101_hardware/scservo_sdk ~/vector_os_nano/lib/python3.10/site-packages/
```

### `pin version 2.x detected; requires >=3.9.0`

**Cause:** Old pinocchio in system Python
**Fix:**
```bash
pip install pin>=3.9.0 --force-reinstall
```

### `tied_weights error with Moondream`

**Cause:** transformers 5.x+ has regression
**Fix:**
```bash
pip install transformers==4.57.6 --force-reinstall
```

---

## Last Verified

- **Date:** 2026-03-20
- **System:** Ubuntu 22.04 LTS
- **Python:** 3.10.16
- **ROS2:** Humble (apt installed)
- **CUDA:** 12.8 (torch nightly cu128)
- **GPU:** RTX 5080 Blackwell (sm_120)
- **All tests:** 530+ passing
