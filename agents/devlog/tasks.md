# Task Board -- v0.3.0 Hardware Abstraction Layer

**Created:** 2026-03-25
**ADR:** docs/architecture-decisions/ADR-003-hardware-abstraction-layer.md
**Status:** Awaiting CEO approval of ADR-003 before execution begins

---

## Phase 1: Protocol + Types (non-breaking) [READY]

### T1: BaseProtocol [alpha]
- File: `vector_os_nano/hardware/base.py`
- Define `BaseProtocol` as `typing.Protocol` with `@runtime_checkable`
- Methods: connect, disconnect, stop, walk, set_velocity, get_position, get_heading, get_velocity, get_odometry, get_lidar_scan
- Properties: name, supports_holonomic, supports_lidar
- Tests: `tests/unit/test_base_protocol.py` -- verify MuJoCoGo2 satisfies protocol after Phase 3
- Acceptance: file exists, mypy clean, no imports from other VOS modules except core/types

### T2: Odometry + LaserScan Types [alpha]
- File: `vector_os_nano/core/types.py` (append to existing)
- Add frozen dataclasses: Odometry, LaserScan
- Both must have `to_dict()` and `from_dict()` methods
- Tests: `tests/unit/test_nav_types.py` -- construction, serialization roundtrip
- Acceptance: existing 852+ tests still pass, new types importable

### T3: PerceptionProtocol [beta]
- File: `vector_os_nano/hardware/perception.py`
- Formalize what skills assume: detect(query) -> list[Detection], track(detections) -> list[TrackedObject], update() -> list[TrackedObject], get_color_frame() -> ndarray | None
- Tests: verify MuJoCoPerception and PerceptionPipeline satisfy protocol
- Acceptance: protocol defined, runtime_checkable, no new dependencies

---

## Phase 2: SkillContext Redesign [READY after Phase 1]

### T4: SkillContext Rewrite [beta] -- COMPLETE
- File: `vector_os_nano/core/skill.py`
- Replace flat fields with dict registries: arms, grippers, bases, perception_sources, services
- Add property accessors: arm, gripper, base, perception (backward compat)
- Add capability queries: has_arm(), has_gripper(), has_base(), has_perception(), capabilities()
- Tests: tests/unit/test_skill_context.py (23 tests), all existing skill tests pass
- Result: 23 new tests + 0 regressions, base.setter added for test_go2_skills compatibility

### T5: Agent._build_context() Update [beta] -- COMPLETE (gamma)
- File: `vector_os_nano/core/agent.py`
- `_build_context()` already uses dict registries (bases=, arms=, grippers=, perception_sources=)
- `Agent.__init__` already accepts base= / arm= / gripper= / perception= flat kwargs
- `_sync_robot_state()` reads self._base correctly
- Tests: `tests/unit/test_agent_hal.py` (14 tests, all passing), 0 regressions
- Verified: test_world_model_base.py (17 tests), test_run_go2.py (6 tests), test_skill_context.py (23 tests) all pass

---

## Phase 3: MuJoCoGo2 Refactor [READY after Phase 1]

### T6: Background Physics Thread [gamma]
- File: `vector_os_nano/hardware/sim/mujoco_go2.py`
- Add `_physics_thread` that runs MPC loop at 1000 Hz
- Thread starts on `connect()`, stops on `disconnect()`
- Thread reads `_cmd_vel` (protected by Lock), runs MPC, calls `mj_step`, syncs viewer
- Add `set_velocity(vx, vy, vyaw)` -- non-blocking, sets `_cmd_vel`
- Refactor `walk()` to: set_velocity -> sleep(duration) -> set_velocity(0,0,0) -> check_upright
- Tests: `tests/unit/test_mujoco_go2_streaming.py` -- set_velocity works, walk() still works, thread starts/stops cleanly
- CRITICAL: All mj_* calls must happen on the physics thread. get_position/get_heading/get_velocity read from thread-safe snapshots.
- Acceptance: existing Go2 integration tests pass, new streaming tests pass

### T7: Odometry + Lidar Simulation [gamma]
- File: `vector_os_nano/hardware/sim/mujoco_go2.py`
- Implement `get_odometry()` -> Odometry (reads from physics thread snapshot)
- Implement `get_lidar_scan()` -> LaserScan (mj_ray, 360 rays, 10 Hz update in physics thread)
- Lidar is cached and updated every 100 physics steps
- Tests: `tests/unit/test_go2_sensors.py` -- odometry matches get_position/get_heading, lidar detects walls in go2_room.xml
- Acceptance: lidar scan has finite ranges near walls, inf in open space

---

## Phase 4: Hardware-Agnostic NavigateSkill [READY after Phase 2 + 3]

### T8: Unified NavigateSkill [alpha]
- File: `vector_os_nano/skills/navigate.py` (NEW top-level, not in go2/)
- Uses `context.base` (any BaseProtocol)
- Mode detection: if `context.services.get("nav2")` -> use Nav2 action client, else -> dead-reckoning
- Dead-reckoning logic moved from `skills/go2/navigate.py` (room map, waypoint graph)
- Room map is configurable (not hardcoded to go2_room.xml layout)
- Tests: `tests/unit/test_navigate_skill.py` -- mock base, dead-reckoning mode
- Acceptance: NavigateSkill works with any BaseProtocol, not imported from go2/

### T9: Deprecate go2/navigate.py [alpha]
- File: `vector_os_nano/skills/go2/navigate.py`
- Add deprecation warning, import and delegate to `skills/navigate.py`
- Update `skills/go2/__init__.py` -- get_go2_skills() returns new NavigateSkill
- Tests: no regressions
- Acceptance: `python run.py --sim-go2` uses new NavigateSkill

---

## Phase 5: ROS2 Go2 Bridge [BLOCKED on Phase 3]

### T10: Go2 ROS2 Bridge Nodes [any]
- File: `vector_os_nano/ros2/nodes/go2_bridge.py`
- Four nodes: Go2CmdVelBridge, Go2OdomBridge, Go2LidarBridge, Go2StateBridge
- CmdVelBridge: subscribes /cmd_vel, calls base.set_velocity()
- OdomBridge: publishes /odom + /tf at 50 Hz from base.get_odometry()
- LidarBridge: publishes /scan at 10 Hz from base.get_lidar_scan()
- StateBridge: publishes /joint_states at 10 Hz from base.get_joint_positions()
- Tests: unit tests with mock base
- Acceptance: nodes start without error, publish correct message types

### T11: Nav2 Launch File [any]
- File: `vector_os_nano/ros2/launch/go2_nav2.launch.py`
- Launches: Go2 bridge nodes + Nav2 stack (planner, controller, BT navigator)
- Parameters: map_file (static map from go2_room.xml), use_sim_time=true
- DEFERRED: Requires Nav2 parameter tuning, map generation from MuJoCo scene

---

## Dependency Graph

```
T1 (BaseProtocol) ─┐
T2 (Types)         ─┼─> T4 (SkillContext) ─> T5 (Agent) ─> T8 (NavigateSkill) ─> T9 (Deprecate)
T3 (PerceptionProto)┘                                             |
                                                                   v
T1 ─> T6 (Physics Thread) ─> T7 (Odom+Lidar) ─────────────> T10 (ROS2 Bridge) ─> T11 (Launch)
```

---

## Wave Execution Plan

**Wave 1** (parallel, no dependencies):
- [alpha] T1 + T2
- [beta] T3
- [gamma] T6

**Wave 2** (after Wave 1):
- [beta] T4 + T5
- [gamma] T7

**Wave 3** (after Wave 2):
- [alpha] T8 + T9

**Wave 4** (after Wave 3, optional / deferred):
- [any] T10 + T11
