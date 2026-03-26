# Vector OS Nano SDK — Progress

**Last updated:** 2026-03-25
**Current version:** v0.5.0-dev
**Focus:** Hardware Abstraction Layer + Go2 Navigation Stack Integration

---

## v0.5.0-dev — IN PROGRESS

### Hardware Abstraction Layer (ADR-003) — DONE
- [x] BaseProtocol (hardware/base.py): formal Protocol for any mobile base
  - walk() blocking + set_velocity() streaming dual-mode
  - get_odometry(), get_lidar_scan(), get_position(), get_heading()
  - supports_holonomic, supports_lidar capability flags
- [x] Odometry + LaserScan pure-Python types (core/types.py)
- [x] SkillContext redesign: dict registries (arms, bases, grippers, perception_sources, services)
  - Backward-compatible properties: context.arm, context.base still work
  - Capability queries: has_arm(), has_base(), capabilities()
- [x] MuJoCoGo2 refactored: background 1kHz physics thread
  - set_velocity() non-blocking (for Nav2 cmd_vel)
  - walk() = set_velocity + sleep (backward compat for skills)
  - PD idle posture hold (no more z-collapse)
  - 3D lidar via mj_ray: 7 elevation rings x 360 azimuth = 2520 rays
  - Robot body excluded from ray detection
- [x] NavStackClient (core/nav_client.py): wraps vector_navigation_stack
  - /way_point (publish goal), /state_estimation (subscribe odom), /goal_reached
  - Lazy rclpy import — works without ROS2 installed
- [x] NavigateSkill: hardware-agnostic, uses NavStackClient when available, dead-reckoning fallback

### Navigation Stack Integration — WORKING
- [x] Unity simulation tested: robot navigates with FAR planner + local planner
- [x] cmd_vel_mux added to launch (pathFollower → /navigation_cmd_vel → mux → /cmd_vel)
- [x] Vector OS Nano brain controls nav stack: NavStackClient.navigate_to() → /way_point → robot moves
- [x] test_nav_brain.py: standalone demo proving brain→nav→robot loop
- [ ] MuJoCo bridge (go2_bridge.py): publishes /state_estimation + /registered_scan, terrain_map works, but pathFollower autonomy mode not resolved
- [ ] Semantic navigation: needs dynamic room discovery (not hardcoded coords)

### Go2 ROS2 Bridge (WIP)
- go2_bridge.py: MuJoCo → ROS2 topics bridge
  - Publishes /state_estimation (50Hz), /registered_scan (10Hz, XYZI PointCloud2), /tf
  - Subscribes /cmd_vel → set_velocity()
  - 3D lidar: floor + walls + ceiling detected (robot body excluded)
  - Fake joystick for pathFollower autonomy mode
  - Issue: pathFollower still outputs zero velocity in MuJoCo mode (Unity mode works fine)

---

## v0.4.0 — Go2 MuJoCo Milestone 1 — DONE

- Convex MPC locomotion (MIT Cheetah 3 paper, go2-convex-mpc library)
- MuJoCoGo2: walk/turn/stand/sit/lie_down via ToolAgent
- 6 Go2 skills: walk, turn, stand, sit, lie_down, navigate
- Indoor house scene: 20m x 14m, 7 rooms + central hallway, furniture
- `python run.py --sim-go2` launches Go2 MuJoCo with ToolAgent
- 48 unit + integration tests
- Dependencies: pinocchio 3.9 (pin), casadi 3.7, convex_mpc (editable)

---

## v0.2.0 — MCP + Memory + Router — DONE

- SessionMemory (50-entry cross-task memory)
- ModelRouter (complexity-driven model selection)
- MCP Server: 10 tools + 7 resources for Claude Desktop/Code
- Bug fixes: depth_scale, parameter passing, JSON schema

---

## v0.1.0 — Foundation — DONE

- Full NL pipeline: classify → plan → execute → summarize
- SkillFlow protocol, MuJoCo sim, web dashboard, CLI, ROS2

---

## Launcher Commands

```bash
# SO-101 arm
python run.py --sim              # MuJoCo arm with viewer
python run.py --sim-headless     # Headless

# Go2 quadruped
python run.py --sim-go2          # MuJoCo Go2 with ToolAgent + viewer

# Navigation stack (Unity)
cd ~/Desktop/vector_navigation_stack
export ROBOT_CONFIG_PATH="unitree/unitree_go2"
./src/.../environment/Model.x86_64 &
ros2 launch vehicle_simulator system_simulation_with_route_planner.launch.py &

# Vector OS Nano brain → nav stack
./scripts/run_nav_brain.sh
```
