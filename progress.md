# Vector OS Nano SDK — Progress

**Last updated:** 2026-04-01
**Version:** v0.6.0-dev
**Branch:** feat/vector-os-nano-python-sdk

---

## Current: Go2 VLM Perception + Multi-Step Task Planner

### Architecture
```
User
  ├── "巡逻全屋" ──→ MobileAgentLoop ──→ [navigate→look→navigate→look→...]
  ├── "去厨房看看" ──→ TaskPlanner ──→ navigate(kitchen) + look()
  ├── RViz teleop ──→ /joy ──→ bridge (direct velocity, 0.8 m/s)
  └── /goal_point ──→ FAR planner ──→ /way_point ──→ localPlanner

MuJoCoGo2 (convex MPC, 1kHz)
  ├── get_camera_frame() ──→ Go2VLMPerception (GPT-4o via OpenRouter)
  │     ├── describe_scene() → SceneDescription (summary, objects, room_type)
  │     ├── identify_room() → RoomIdentification (room, confidence)
  │     └── find_objects() → [DetectedObject]
  ├── publishes: /state_estimation (200Hz), /registered_scan (10Hz)
  ├── publishes: /camera/image (5Hz, 320x240), /speed (2Hz)
  └── SpatialMemory: room→objects mapping, visit tracking, persistence

Skills (12 total):
  walk, turn, stand, sit, lie_down, navigate, explore,
  where_am_i, stop, look, describe_scene, patrol
```

### Harness Results
| Suite | Result | Details |
|-------|--------|---------|
| Locomotion | **26/26** | L0 physics → L4 navigation |
| Agent+Go2 | **5/5** | walk, turn, stand, sit, skills |
| VLM API (L0) | **4/4** | GPT-4o reachable, JSON parse, latency, cost |
| Camera→VLM (L1) | **6/6** | MuJoCo frame → GPT-4o → scene description |
| Scene Skills (L2) | **17/17** | LookSkill, DescribeSceneSkill (mock VLM) |
| Task Planning (L3) | **18/18** | fallback planner, JSON parse, Chinese rooms |
| E2E Patrol (L4) | **4/4** | 2-room patrol, real API, spatial memory |
| VLM Accuracy (L5) | **1-2/8** | Diagnostic: MuJoCo rendering limits room ID |
| ToolAgent (L5) | **6/6** | 中文指令, navigate, look, multi-turn context |
| Robustness (L6) | **32/32** | VLM errors, nav edge cases, spatial memory |
| Nav2 integration | **11/11** | AMCL + MPPI, goal arrival |
| SLAM mapping | **3/3** | map grows during movement |
| **Total harness** | **117/118** | 1 xfail (VLM accuracy variance) |

### What's New (v0.6.0-dev)
- **VLM Perception**: GPT-4o via OpenRouter analyzes Go2 camera frames
  - Room identification, scene description, object detection
  - Cost tracking, retry logic, JSON parse with fallback
- **Multi-Step Task Planner**: MobileAgentLoop decomposes goals into skill chains
  - LLM planning with fallback heuristic (Chinese + English)
  - "巡逻全屋" → navigate+look for each room
  - Robot fall detection → auto-abort
- **New Skills**: look, describe_scene, patrol (3 new, 12 total)
- **Spatial Memory**: Persistent room→objects mapping with visit tracking
- **Test Collection**: Fixed 34→0 collection errors (installed openai, mcp, Pillow)
- **1660 tests collected** (was 782), 0 collection errors

### What Works
- Go2 walks with unitree convex MPC (auto-detected, sinusoidal fallback)
- Livox MID360 simulation: 30 deg tilt, -7/+52 deg FOV, 10k+ points/scan
- Vector Nav Stack: localPlanner, pathFollower, terrain_analysis, FAR planner
- TARE autonomous exploration (frontier-based TSP)
- Camera RGB from MuJoCo → GPT-4o scene understanding
- VLM room identification with confidence scores
- Multi-room patrol with spatial memory recording
- Agent SDK: natural language → Go2 skills (12 skills)
- SpatialMemory persists observations across patrol steps
- Dead-reckoning + Nav Stack navigation modes

### Known Issues
1. FAR planner publishes /way_point but not /global_path (graph_decoder issue)
2. Camera depth rendering intermittent (MuJoCo API)
3. MobileAgentLoop LLM planning requires API key (fallback planner works without)

### Scripts
| Script | Purpose |
|--------|---------|
| `./scripts/launch_explore.sh` | Autonomous exploration (TARE + VNav) |
| `./scripts/launch_vnav.sh` | Vector Nav Stack + RViz (manual/goal) |
| `./scripts/launch_nav2.sh --rviz` | Nav2 + AMCL alternative |
| `./scripts/launch_slam.sh` | SLAM real-time mapping |
| `.venv-nano/bin/python3 run.py --sim-go2` | Agent mode (NL + VLM) |
| `.venv-nano/bin/python3 -m pytest tests/harness/ -v` | Full harness (75 tests) |
