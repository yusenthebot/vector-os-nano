# Vector OS Nano SDK — Progress

**Last updated:** 2026-03-31
**Version:** v0.5.0-dev
**Branch:** feat/vector-os-nano-python-sdk

---

## Current: Go2 MuJoCo + Vector Navigation Stack

### Architecture
```
User
  ├── RViz teleop panel ──→ /joy ──→ bridge (direct velocity, 0.8 m/s)
  ├── "走到厨房" ──→ Agent ──→ NavigateSkill
  └── /goal_point ──→ FAR planner ──→ /way_point ──→ localPlanner
                                                        ↓
                                                      /path
                                                        ↓
MuJoCoGo2 (convex MPC, 1kHz)  ←── bridge ←── /navigation_cmd_vel
  ├── publishes: /state_estimation (200Hz), /registered_scan (10Hz, 10k+ pts)
  ├── publishes: /camera/image (5Hz, 320x240), /speed (2Hz)
  ├── TF: map→sensor, map→vehicle
  └── terrain_analysis → /terrain_map (~49k pts)
  └── TARE planner → /way_point (autonomous frontier exploration)
```

### Harness Results
| Suite | Result | Details |
|-------|--------|---------|
| Locomotion (pytest) | **26/26** | L0 physics → L4 navigation |
| Agent+Go2 | **5/5** | walk, turn, stand, sit, skills |
| Nav2 integration | **11/11** | AMCL + MPPI, goal arrival |
| SLAM mapping | **3/3** | map grows during movement |

### What Works
- Go2 walks with unitree convex MPC (auto-detected, sinusoidal fallback)
- Livox MID360 simulation: 30° tilt, -7° to +52° FOV, 10k+ points/scan
- Point cloud intensity = height above ground (terrain_analysis compatible)
- Vector Nav Stack: localPlanner, pathFollower, terrain_analysis, FAR planner
- TARE autonomous exploration (frontier-based TSP)
- Teleop from RViz panel (clears autonomous path on input)
- Camera RGB from MuJoCo renderer
- Nav2 + SLAM alternatives also available
- Agent SDK: natural language → Go2 skills
- Process cleanup on exit (no more zombie accumulation)

### Known Issues
1. FAR planner publishes /way_point but not /global_path (graph_decoder issue)
2. Camera depth rendering needs MuJoCo API fix

### Scripts
| Script | Purpose |
|--------|---------|
| `./scripts/launch_explore.sh` | Autonomous exploration (TARE + VNav) |
| `./scripts/launch_vnav.sh` | Vector Nav Stack + RViz (manual/goal) |
| `./scripts/launch_nav2.sh --rviz` | Nav2 + AMCL alternative |
| `./scripts/launch_slam.sh` | SLAM real-time mapping |
| `.venv-nano/bin/python3 scripts/go2_demo.py` | Visual locomotion demo |
| `.venv-nano/bin/python3 run.py --sim-go2` | Agent mode (NL control) |
| `.venv-nano/bin/python3 -m pytest tests/harness/ -v` | Locomotion harness |
| `./scripts/test_integration.sh` | Full integration harness |
