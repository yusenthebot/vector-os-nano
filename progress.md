# Vector OS Nano SDK — Progress

**Last updated:** 2026-04-02
**Version:** v1.0.0-dev
**Branch:** master

---

## Current: GroundingDINO + RGBD Object Detection & Mapping

### Sensor Configuration (sim-to-real)
```
Unitree Go2 quadruped
  ├── Livox MID-360 LiDAR  → /registered_scan (10Hz, 10k+ pts)
  │     30° tilt, -7/+52° FOV, sim: MuJoCo raycasting
  └── RealSense D435 RGBD  → mounted on Go2 head (MJCF: d435_rgb, d435_depth)
        fovy=42°, pos=(0.30, 0, 0.05) relative to base_link
        sim: MuJoCo named camera rendering
        real: rs2 aligned_depth_to_color
```

### Architecture
```
Subprocess (launch_vnav.sh)
  MuJoCoGo2 (convex MPC, 1kHz)
  Go2VNavBridge (200Hz odom, 10Hz scan, 5Hz RGBD)
  localPlanner + pathFollower + terrainAnalysis + FAR planner
  sensorScanGeneration → /state_estimation_at_scan → TARE

Agent Process (vector-cli via .venv-nano)
  Go2ROS2Proxy ←→ ROS2 topics ←→ Bridge
  │
  ├── RGB → GroundingDINO (grounding-dino-tiny, RTX 5080 CUDA)
  │         per-object bbox (u1,v1,u2,v2) + confidence
  │
  ├── RGB → VLM (GPT-4o-mini via OpenRouter)
  │         room identification + scene description
  │
  ├── Depth → D435 depth at each bbox center → metric distance
  │
  ├── Camera intrinsics (fovy=42°) + robot pose
  │         → depth_to_world() per object → world (x, y, z)
  │
  └── SceneGraph: 3-layer (rooms→viewpoints→objects)
        objects positioned via GroundingDINO bbox + depth projection
        persistent YAML (~/.vector_os_nano/scene_graph.yaml)

Skills (12): walk, turn, stand, sit, lie_down, navigate, explore,
             where_am_i, stop, look, describe_scene, patrol
```

### Object Detection Pipeline
```
D435 RGBD (mounted on Go2 head)
  │
  ├── RGB (320x240) → GroundingDINO → [Detection(label, bbox, confidence)]
  │                                        │
  │                    for each detection:  │
  │                    bbox center (cu, cv) │
  │                           ↓             │
  └── Depth (320x240) → depth[cv, cu]  → metric distance (m)
                              ↓
        depth_to_world(cu, cv, depth, intrinsics, robot_pose)
                              ↓
        Detection(label="bookshelf", world_x=17.0, world_y=5.3, depth=7.5m)
                              ↓
        SceneGraph.merge_object(category="bookshelf", x=17.0, y=5.3)
                              ↓
        RViz marker at (17.0, 5.3) — accurate world position

Measured accuracy (MuJoCo sim):
  bookshelf: 0.79m error, chair: 0.69m, stool: 0.84m
```

### Fallback chain (graceful degradation)
```
1. GroundingDINO + depth → per-object world coords (best)
2. VLM text + viewpoint heading cluster (when no torch/GPU)
3. Room center scatter (when no VLM either)
```

### Harness Results
| Suite | Result | Details |
|-------|--------|---------|
| Locomotion (L0-L4) | **26/26** | physics → navigation |
| Agent+Go2 | **5/5** | walk, turn, stand, sit, skills |
| VLM API (L0) | **4/4** | GPT-4o-mini reachable, JSON parse |
| Camera→VLM (L1) | **6/6** | MuJoCo frame → GPT-4o-mini → scene |
| Scene Skills (L2) | **17/17** | LookSkill, DescribeSceneSkill |
| Task Planning (L3) | **18/18** | fallback planner, JSON parse, Chinese |
| E2E Patrol (L4) | **4/4** | 2-room patrol, real API |
| Robustness (L6) | **32/32** | VLM errors, nav edge cases |
| SceneGraph (L7) | **55/55** | 3-layer graph, viewpoints, coverage, persist |
| RViz Markers (L8) | **38/38** | room fills, FOV cones, trajectory, nav goal |
| Proxy E2E (L9) | **26/26** | Go2ROS2Proxy camera → LookSkill → SceneGraph |
| Persistence (L9) | **28/28** | SceneGraph save/load lifecycle |
| Auto-Look (L10) | **8/8** | ExploreSkill + VLM auto-observe |
| Mobile Loop (L11) | **14/14** | LLM planning, fallback, execution |
| TARE Chain (L12) | **20/20** | wander interval, duty cycle, QoS |
| Depth Projection (L13) | **24/24** | D435 intrinsics, pixel→world |
| Detection (L14) | **12/12** | GroundingDINO inference, GPU, bbox |
| Detection Integration (L15) | **43/43** | detector wiring, LookSkill, SceneGraph, auto-look |
| **Total** | **310+** | 0 regressions |

### What's New (v1.0.0-dev)
- **GroundingDINO Object Detection**: Open-vocabulary detector (grounding-dino-tiny) on RTX 5080. Per-object bounding boxes for accurate positioning.
- **D435 Camera Mounted on Go2 Head**: MJCF named cameras (d435_rgb, d435_depth) fixed to base_link. No more free-camera approximation.
- **Per-Object Depth Projection**: Each detected object gets independent world (x,y,z) from bbox center depth + camera intrinsics + robot pose. ~0.8m accuracy.
- **VLM Timeout Fix**: Resize to 160px + quality 50, switch to gpt-4o-mini. 1-2s response (was timing out at 45s).
- **TARE Wander Fix**: 0.8s velocity interval keeps robot moving → TARE gets scan data → generates waypoints.
- **Anti-Flicker Markers**: 3s publish interval, 5s lifetime, hash-based change detection.
- **Auto-Look Non-Blocking**: VLM/detector calls in separate thread, don't block exploration loop.
- **`/clear_memory`**: Reset scene graph from CLI (works with or without running agent).

### What Works
- Go2 walks with unitree convex MPC (auto-detected, sinusoidal fallback)
- Livox MID360 + RealSense D435 simulation (LiDAR + RGBD, mounted cameras)
- Vector Nav Stack: localPlanner, pathFollower, terrainAnalysis, FAR planner
- TARE autonomous exploration with continuous wander velocity
- GroundingDINO object detection → D435 depth → world coordinate mapping
- VLM room identification + scene description (GPT-4o-mini)
- Multi-room patrol with spatial memory recording
- Agent SDK: natural language → Go2 skills (12 skills, Chinese + English)
- SceneGraph persists across sessions
- RViz: color-coded rooms, FOV cones, trajectory, objects at detected positions

### Known Issues
1. Some false detections in MuJoCo (low texture quality → floor detected as desk)
2. Objects at image edges may have inaccurate depth (D435 depth noise at boundaries)
3. FAR planner publishes /way_point but /global_path sometimes incomplete

### TODO
- [ ] Filter false detections: cross-reference GroundingDINO with VLM objects
- [ ] Multi-view object fusion: merge detections from different viewpoints
- [ ] Real D435 driver integration (rs2 → /camera/image + /camera/depth)
- [ ] Confidence-weighted object position averaging across observations
- [ ] Object tracking across frames (re-identification)

### CLI Commands
| Command | Purpose |
|---------|---------|
| `/clear_memory` | Reset scene graph, delete persist file |
| `/model <name>` | Switch LLM model |
| `/status` | Show hardware, tools, session info |
| `/help` | Show all commands |

### Scripts
| Script | Purpose |
|--------|---------|
| `./scripts/launch_vnav.sh` | Full nav stack + RViz |
| `.venv-nano/bin/vector-cli` | Agent CLI (must use .venv-nano for torch) |
| `.venv-nano/bin/python3 -m pytest tests/harness/ -v` | Full harness |
| `.venv-nano/bin/python3 tests/verify_compat.py` | Package compatibility check |
