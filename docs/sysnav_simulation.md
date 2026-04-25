# SysNav Simulation Bringup

How to run [SysNav](https://github.com/zwandering/SysNav) on top of the
Vector OS Nano MuJoCo simulation. SysNav is the same CMU Robotics
Institute lab's project; we consume its scene-graph topics through the
Apache-2.0 adapter at
[`vector_os_nano/integrations/sysnav_bridge/`](../vector_os_nano/integrations/sysnav_bridge/).

> **License boundary**. SysNav is PolyForm-Noncommercial-1.0.0; Vector
> OS Nano is Apache 2.0. We never copy SysNav source files into this
> repo. SysNav runs as a sibling ROS2 workspace; only ROS2 messages
> cross the boundary.

For real-robot bringup see [`sysnav_integration.md`](sysnav_integration.md).

---

## 1. Architecture

```
┌──────────────────────────────────────────────────────┐
│ vector_os_nano (Apache 2.0)                          │
│                                                      │
│  vector-cli > start_sysnav_sim                       │
│   ├── MuJoCo subprocess (go2_room.xml + Go2 + Piper) │
│   │     emits via 3 NEW virtual sensors:             │
│   │       MuJoCoLivox360 → /registered_scan          │
│   │       MuJoCoPano360  → /camera/image             │
│   │       GroundTruthOdom → /state_estimation        │
│   │                                                  │
│   └── LiveSysnavBridge (rclpy subscriber)            │
│        listens for /object_nodes_list                │
│        → world_model.add_object(ObjectState)         │
└────────────────────────┬─────────────────────────────┘
                         │ ROS2 topics
                         ▼
┌──────────────────────────────────────────────────────┐
│ SysNav workspace (PolyForm-NC, sibling install)      │
│   ✓ semantic_mapping_node                            │
│   ✓ detection_node      (SAM2 + YOLO)                │
│   ✓ vlm_reasoning_node                               │
│   ✗ arise_slam_mid360   (sim provides ground truth)  │
│   ✗ livox_ros_driver2   (sim provides PointCloud2)   │
│   ✗ tare_planner        (use vector_navigation_stack)│
└──────────────────────────────────────────────────────┘
```

## 2. Prerequisites

Sourced once per terminal (both terminals — vector-cli and SysNav):

```bash
source /opt/ros/jazzy/setup.bash
```

SysNav workspace built and installed:

```bash
cd ~/Desktop
git clone --recurse-submodules --branch unitree_go2 \
    https://github.com/zwandering/SysNav.git
cd SysNav
# Follow §1-5 of SysNav's own README for SLAM dependencies.
colcon build --symlink-install
source install/setup.bash
```

Vector OS Nano environment (this repo):

```bash
cd ~/Desktop/vector_os_nano
source .venv-nano/bin/activate
pip install -e .[perception]
```

## 3. Smoke checks

Before a full bringup, run the smoke script:

```bash
# Mode 1 — dependency probe (no MuJoCo, no GPU)
python scripts/smoke_sysnav_sim.py --check-deps

# Mode 2 — virtual sensors only (no SysNav needed)
python scripts/smoke_sysnav_sim.py --no-sysnav

# Mode 3 — full bringup (SysNav workspace must be running)
python scripts/smoke_sysnav_sim.py
```

Mode 3 fails fast (exit 1) with an actionable message when SysNav is
absent — the bridge will not block agent startup.

## 4. Bringup sequence

Three terminals, in this order.

### Terminal 1 — SysNav

```bash
cd ~/Desktop/SysNav
source /opt/ros/jazzy/setup.bash
source install/setup.bash
./system_real_robot_with_exploration_planner_go2.sh
```

Wait for these log lines (printed by SysNav nodes):

```
[semantic_mapping_node] object_nodes_list publisher created
[detection_node]        ready
[vlm_reasoning_node]    ready
```

### Terminal 2 — Vector OS Nano

```bash
cd ~/Desktop/vector_os_nano
source /opt/ros/jazzy/setup.bash
source ~/Desktop/SysNav/install/setup.bash    # for tare_planner.msg
.venv-nano/bin/activate
vector-cli
> start_sysnav_sim
```

The `start_sysnav_sim` tool result reports whether the bridge is live:

```
sysnav-sim started: MuJoCo + bridge live, listening to /object_nodes_list.
```

If SysNav is not running, the tool still completes:

```
sysnav-sim started: MuJoCo up; bridge inactive (source SysNav workspace +
rebuild tare_planner.msg). World model will not populate from SysNav until
the bridge is restarted.
```

### Terminal 3 — RViz (optional)

```bash
ros2 run rviz2 rviz2 \
    -d ~/Desktop/SysNav/src/exploration_planner/tare_planner/rviz/tare_planner_ground.rviz
```

## 5. Topic contract

| Direction | Topic | Type | Producer | Notes |
|---|---|---|---|---|
| sim → SysNav | `/registered_scan`        | `sensor_msgs/PointCloud2` | `MuJoCoLivox360` | x/y/z/intensity float32, frame `map` |
| sim → SysNav | `/state_estimation`       | `nav_msgs/Odometry`       | `GroundTruthOdomPublisher` | 50 Hz, `map` → `sensor` |
| sim → SysNav | `/camera/image`           | `sensor_msgs/Image`       | `MuJoCoPano360` | rgb8, 1920×640 equirectangular, HFoV 360° VFoV 120° |
| SysNav → us  | `/object_nodes_list`      | `tare_planner/ObjectNodeList` | `semantic_mapping_node` | consumed by `LiveSysnavBridge` |
| SysNav → us  | `/object_type_query`      | `tare_planner/ObjectType` | `vlm_reasoning_node` | optional consumer |
| us → SysNav  | `/target_object_instruction` | `tare_planner/TargetObjectInstruction` | (future) | drives VLM reasoning |

## 6. Performance targets

Measured on RTX 5080 with the inline 4-wall MJCF:

| Operation | Target | Observed (probe) |
|---|---|---|
| `mj_ray` × 5760 rays | ≤ 50 ms | 13 ms |
| 6 cube-face renders @ 480² | ≤ 80 ms | 3.3 ms |
| End-to-end pano stitch (LUT remap) | ≤ 100 ms | < 80 ms |

Real go2_room.xml (full Go2 + Piper + 8 rooms) will be slower —
benchmark with `scripts/smoke_sysnav_sim.py --no-sysnav --timeout 30`
before declaring readiness.

## 7. Troubleshooting

### `start_sysnav_sim` reports "bridge inactive"

`tare_planner.msg` is not importable. The most common causes:

* SysNav workspace not built — run `colcon build` in `~/Desktop/SysNav`.
* Workspace not sourced — run `source ~/Desktop/SysNav/install/setup.bash`
  in the same shell where you run `vector-cli`.
* ROS2 distro mismatch — both shells must source the same Jazzy.

### `/object_nodes_list` never carries any node

SysNav runs but is not detecting anything:

* RViz: confirm `/registered_scan` shows points around the robot.
* `ros2 topic hz /registered_scan` should be ≥ 5 Hz.
* `ros2 topic hz /camera/image` should be ≥ 3 Hz.
* Detection requires the YOLO model weights — see SysNav's own
  `set_yolo_e.py` / `set_yolo_world.py`.

### MuJoCo GPU OOM under SysNav inference

SysNav's SAM2 + YOLO + spaCy load alongside our pano renders can push
VRAM. Mitigations:

* Lower `MuJoCoPano360` `out_w/out_h` (default 1920/640 — try 960/320).
* Drop pano `rate_hz` from 5 to 3.
* Run SysNav inference on a separate GPU if available.

### Topic frame mismatch / TF tree empty

The bridge publishes everything in `map` frame. SysNav expects the
same — if it complains about missing TF, ensure the SysNav launch is
NOT also trying to start `arise_slam_mid360`. Use the
`semantic_mapping_real.launch` variant or comment out the SLAM
include.

## 8. References

* Adapter package: [`vector_os_nano/integrations/sysnav_bridge/`](../vector_os_nano/integrations/sysnav_bridge/)
* Sensor package:  [`vector_os_nano/hardware/sim/sensors/`](../vector_os_nano/hardware/sim/sensors/)
* CLI tool:        [`vector_os_nano/vcli/tools/sysnav_sim_tool.py`](../vector_os_nano/vcli/tools/sysnav_sim_tool.py)
* Smoke script:    [`scripts/smoke_sysnav_sim.py`](../scripts/smoke_sysnav_sim.py)
* SysNav repo:     https://github.com/zwandering/SysNav
* SysNav paper:    arXiv [2603.06914](https://arxiv.org/abs/2603.06914)
