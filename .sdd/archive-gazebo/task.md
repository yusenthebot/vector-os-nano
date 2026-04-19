# Gazebo Harmonic Integration — Task List

## Execution Status
- Total tasks: 12
- Completed: 0
- In progress: 0
- Pending: 12

---

## Wave 1: Core Infrastructure (parallel — Alpha, Beta, Gamma)

### Task 1: GazeboGo2Proxy
- **Status**: [ ] pending
- **Agent**: alpha
- **Depends**: none
- **Output**: `vector_os_nano/hardware/sim/gazebo_go2_proxy.py`, `vector_os_nano/hardware/sim/__init__.py` (modify)
- **TDD Deliverables**:
  - RED: `tests/unit/test_gazebo_proxy.py`
    - test_inherits_go2ros2proxy
    - test_name_returns_gazebo_go2
    - test_supports_lidar_true
    - test_node_name_is_gazebo_go2_proxy
    - test_is_gazebo_running_false_no_process (mock subprocess)
    - test_is_gazebo_running_true_clock_topic (mock subprocess with /clock)
    - test_is_gazebo_running_handles_timeout
    - test_connect_raises_when_not_running (mock is_gazebo_running=False)
    - test_connect_calls_super_when_running (mock is_gazebo_running=True, mock super)
    - test_disconnect_delegates_to_parent
    - test_init_py_exports_gazebo_proxy
  - GREEN: `gazebo_go2_proxy.py` (~40 lines) + `__init__.py` modification
  - REFACTOR: ensure docstrings match IsaacSimProxy style
- **Verify**: `pytest tests/unit/test_gazebo_proxy.py -v`

### Task 2: ros_gz_bridge Configuration + Validation Tests
- **Status**: [ ] pending
- **Agent**: beta
- **Depends**: none
- **Output**: `gazebo/config/bridge.yaml`
- **TDD Deliverables**:
  - RED: `tests/unit/test_gazebo_bridge.py`
    - test_bridge_yaml_exists
    - test_bridge_yaml_valid_syntax
    - test_bridge_has_state_estimation (topic, ros type, gz type, direction)
    - test_bridge_has_registered_scan
    - test_bridge_has_camera_image
    - test_bridge_has_camera_depth
    - test_bridge_has_cmd_vel_nav (direction = ROS_TO_GZ)
    - test_bridge_has_clock
    - test_bridge_has_imu
    - test_bridge_all_directions_valid (GZ_TO_ROS or ROS_TO_GZ only)
    - test_bridge_ros_types_importable (verify msg types exist in ROS2)
  - GREEN: `gazebo/config/bridge.yaml`
  - REFACTOR: add inline comments explaining each mapping
- **Verify**: `pytest tests/unit/test_gazebo_bridge.py -v`

### Task 3: Go2 SDF Model with Sensors
- **Status**: [ ] pending
- **Agent**: gamma
- **Depends**: none
- **Output**: `gazebo/models/go2/` directory
- **TDD Deliverables**:
  - RED: `tests/unit/test_gazebo_model.py`
    - test_go2_model_dir_exists
    - test_go2_has_sdf_or_urdf (model.sdf or model.urdf.xacro exists)
    - test_go2_has_12_joints (parse SDF/URDF, count revolute joints)
    - test_go2_has_gpu_lidar_sensor (sensor type="gpu_lidar" present)
    - test_go2_lidar_mount_position (x=0.3, z=0.2 relative to base_link)
    - test_go2_lidar_range_12m
    - test_go2_has_camera_sensor (type="camera" present)
    - test_go2_camera_mount_position (x=0.3, z=0.05)
    - test_go2_camera_resolution_640x480
    - test_go2_has_depth_camera_sensor
    - test_go2_has_imu_sensor
    - test_go2_has_ros2_control_plugin (gz_ros2_control system plugin)
  - GREEN: Copy go2_description from quadruped_ros2_control, add sensors.xacro, generate model.sdf
  - REFACTOR: clean unused meshes, verify collision geometry simplified
- **Pre-req**: Clone quadruped_ros2_control to ~/Desktop/ first
- **Verify**: `pytest tests/unit/test_gazebo_model.py -v && gz sdf -k gazebo/models/go2/model.sdf`

---

## Wave 2: Launch + Scripts (sequential after Wave 1 — Alpha)

### Task 4: ROS2 Launch File
- **Status**: [ ] pending
- **Agent**: alpha
- **Depends**: T1, T2, T3
- **Output**: `gazebo/launch/go2_sim.launch.py`
- **TDD Deliverables**:
  - RED: `tests/unit/test_gazebo_launch.py`
    - test_launch_file_exists
    - test_launch_is_valid_python (importable)
    - test_launch_declares_world_arg
    - test_launch_declares_gui_arg
    - test_launch_has_gz_sim_node
    - test_launch_has_spawn_entity
    - test_launch_has_bridge_node
    - test_launch_has_controller_spawner
  - GREEN: `gazebo/launch/go2_sim.launch.py` — full launch description
  - REFACTOR: add LaunchConfiguration defaults, comments
- **Verify**: `pytest tests/unit/test_gazebo_launch.py -v && ros2 launch gazebo/launch/go2_sim.launch.py --show-args`

### Task 5: Launch/Stop Scripts
- **Status**: [ ] pending
- **Agent**: beta
- **Depends**: T4
- **Output**: `scripts/launch_gazebo.sh`, `scripts/stop_gazebo.sh`
- **TDD Deliverables**:
  - RED: `tests/unit/test_gazebo_scripts.py`
    - test_launch_script_exists
    - test_launch_script_executable
    - test_launch_script_has_preflight_checks
    - test_launch_script_accepts_world_arg
    - test_launch_script_accepts_headless_arg
    - test_stop_script_exists
    - test_stop_script_executable
  - GREEN: Shell scripts with preflight, arg parsing, topic polling
  - REFACTOR: consistent error messages, usage help
- **Verify**: `pytest tests/unit/test_gazebo_scripts.py -v && bash scripts/launch_gazebo.sh --help`

### Task 6: CLI Backend Registration
- **Status**: [ ] pending
- **Agent**: gamma
- **Depends**: T1
- **Output**: `vector_os_nano/vcli/tools/sim_tool.py` (modify)
- **TDD Deliverables**:
  - RED: `tests/unit/test_gazebo_backend.py`
    - test_sim_tool_backend_enum_has_gazebo
    - test_start_gazebo_go2_method_exists
    - test_start_gazebo_go2_creates_gazebo_proxy (mock GazeboGo2Proxy)
    - test_start_gazebo_go2_registers_skills (mock Agent)
    - test_start_gazebo_go2_loads_scene_graph
    - test_start_gazebo_go2_attaches_vlm
    - test_execute_routes_to_gazebo (backend="gazebo" → _start_gazebo_go2)
    - test_gazebo_backend_default_not_changed (default still "isaac")
  - GREEN: Modify sim_tool.py — add enum value + _start_gazebo_go2() + routing
  - REFACTOR: extract shared Agent-building code if 3 backends share pattern
- **Verify**: `pytest tests/unit/test_gazebo_backend.py -v`

---

## Wave 3: Apartment World (parallel — Alpha, Beta)

### Task 7: Empty Room World (quick validation)
- **Status**: [ ] pending
- **Agent**: alpha
- **Depends**: T3
- **Output**: `gazebo/worlds/empty_room.sdf`
- **TDD Deliverables**:
  - RED: `tests/unit/test_gazebo_world.py::TestEmptyRoom`
    - test_empty_room_sdf_exists
    - test_empty_room_sdf_valid (gz sdf -k exit 0, or parse XML)
    - test_empty_room_has_ground_plane
    - test_empty_room_has_walls (4 walls enclosing space)
    - test_empty_room_has_light
    - test_empty_room_floor_size (>= 4m x 5m)
  - GREEN: Minimal SDF — ground, 4 walls, light, physics plugin
  - REFACTOR: add PBR floor texture
- **Verify**: `pytest tests/unit/test_gazebo_world.py::TestEmptyRoom -v`

### Task 8: Apartment World with Furniture
- **Status**: [ ] pending
- **Agent**: beta
- **Depends**: T7
- **Output**: `gazebo/worlds/apartment.sdf`, `gazebo/worlds/models/`
- **TDD Deliverables**:
  - RED: `tests/unit/test_gazebo_world.py::TestApartment`
    - test_apartment_sdf_exists
    - test_apartment_sdf_valid
    - test_apartment_room_count_gte_5 (parse walls → room detection)
    - test_apartment_has_door_openings (gaps in walls >= 0.8m)
    - test_apartment_has_furniture_gte_10 (model count)
    - test_apartment_has_graspable_objects_gte_8 (dynamic models with mass)
    - test_apartment_has_per_room_lighting
    - test_apartment_has_floor_materials (different material per room)
    - test_apartment_total_area_gte_50m2
    - test_graspable_objects_have_mass (each < 2kg)
    - test_graspable_objects_have_collision
    - test_graspable_objects_have_visual_mesh (not just box)
  - GREEN: Full apartment SDF with Fuel models + custom furniture + objects
  - REFACTOR: organize models into subdirectories, add material previews
- **Verify**: `pytest tests/unit/test_gazebo_world.py::TestApartment -v && gz sim apartment.sdf --headless-rendering -s -r --iterations 100`

---

## Wave 4: Controller Integration (sequential — Alpha)

### Task 9: Clone + Build quadruped_ros2_control
- **Status**: [ ] pending
- **Agent**: alpha
- **Depends**: none (can start early, but needed for T10)
- **Output**: Built packages in ~/Desktop/quadruped_ros2_control/install/
- **Steps**:
  1. `git clone https://github.com/legubiao/quadruped_ros2_control.git ~/Desktop/quadruped_ros2_control`
  2. `cd ~/Desktop/quadruped_ros2_control`
  3. `rosdep install --from-paths . --ignore-src -r -y`
  4. `colcon build --packages-select go2_description gz_quadruped_hardware unitree_guide_controller --symlink-install`
  5. Verify: `source install/setup.bash && ros2 pkg list | grep -E "go2_description|gz_quadruped|unitree_guide"`
- **No TDD** (external dependency build — verify via colcon)
- **Verify**: Build succeeds, packages visible in ros2 pkg list

### Task 10: unitree_guide_controller Integration
- **Status**: [ ] pending
- **Agent**: alpha
- **Depends**: T4, T9
- **Output**: Go2 standing + walking in Gazebo with guide controller
- **Steps**:
  1. Source quadruped_ros2_control install
  2. Launch: `ros2 launch gazebo/launch/go2_sim.launch.py controller:=guide`
  3. Verify Go2 spawns and stands (z ~ 0.32)
  4. Send velocity command: `ros2 topic pub /cmd_vel_nav geometry_msgs/msg/Twist ...`
  5. Verify Go2 moves
  6. Test with `vector sim start --backend gazebo` → walk command
- **TDD Deliverables** (integration-level):
  - `tests/unit/test_gazebo_controller.py`
    - test_guide_controller_config_exists (ros2_control.yaml has unitree_guide_controller)
    - test_guide_controller_joint_interfaces (12 joints x position+velocity+effort+kp+kd)
    - test_controller_config_has_imu_interface
- **Verify**: Manual: Go2 walks in Gazebo. Auto: `pytest tests/unit/test_gazebo_controller.py -v`

### Task 11: rl_quadruped_controller Integration (enhancement)
- **Status**: [ ] pending
- **Agent**: alpha
- **Depends**: T10
- **Output**: Go2 walking with RL policy in Gazebo
- **Steps**:
  1. Download libtorch 2.5.0 C++ (cu121, cxx11-abi)
  2. `colcon build --packages-select rl_quadruped_controller`
  3. Copy go2_jit.pt policy to controller config
  4. Launch with `controller:=rl`
  5. Verify smooth walking (RL policy should be smoother than guide)
- **TDD Deliverables**:
  - `tests/unit/test_gazebo_controller.py::TestRLController`
    - test_rl_controller_config_exists
    - test_rl_controller_has_model_path
    - test_rl_controller_observation_dims (48)
    - test_rl_controller_action_dims (12)
- **Verify**: Go2 walks smoothly with RL policy. Compare with guide controller.

---

## Wave 5: Integration Verification (Alpha)

### Task 12: End-to-End Pipeline Test
- **Status**: [ ] pending
- **Agent**: alpha
- **Depends**: T8, T10
- **Output**: Full pipeline verified
- **Steps**:
  1. `scripts/launch_gazebo.sh --world apartment`
  2. `vector sim start --backend gazebo`
  3. Verify all acceptance criteria from spec:
     - AC1: Go2 visible in apartment
     - AC2: /state_estimation >= 20 Hz
     - AC3: /registered_scan >= 1000 points
     - AC4: /camera/image 640x480
     - AC5: CLI connects
     - AC6: walk command moves Go2
     - AC7: existing tests pass (no regressions)
     - AC8: apartment has rooms + furniture + objects
     - AC9: VLM describes camera frame
  4. Run full test suite: `pytest tests/ -x`
  5. scripts/stop_gazebo.sh
- **Verify**: All 9 acceptance criteria pass

---

## Dependency Graph

```
T1 (Proxy) ──────────────┐
T2 (Bridge) ──────────────┤──> T4 (Launch) ──> T5 (Scripts)
T3 (Go2 Model) ──────────┘       │
     │                            │
     └──> T7 (Empty Room)        │
           └──> T8 (Apartment)   │
                    │             │
                    └─────────────┴──> T12 (E2E)

T9 (Clone+Build) ──> T10 (Guide Controller) ──> T11 (RL Controller)
                            │
                            └──> T12 (E2E)
```

## Execution Waves

| Wave | Tasks | Agents | Gate | Est. Tests |
|------|-------|--------|------|------------|
| 1 | T1, T2, T3 | Alpha, Beta, Gamma | Unit tests pass | ~33 |
| 2 | T4, T5, T6 | Alpha, Beta, Gamma | Launch + CLI tests pass | ~23 |
| 3 | T7, T8 | Alpha, Beta | World SDF validates | ~18 |
| 4 | T9, T10, T11 | Alpha | Go2 walks in Gazebo | ~7 |
| 5 | T12 | Alpha | All AC pass, no regressions | -- |
| -- | code-review + security-review | QA | Approve | -- |
