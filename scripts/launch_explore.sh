#!/bin/bash
# Go2 + Vector Nav Stack + TARE Autonomous Exploration — one-command launch
#
# Usage:
#   cd ~/Desktop/vector_os_nano
#   ./scripts/launch_explore.sh              # MuJoCo viewer + RViz
#   ./scripts/launch_explore.sh --no-gui     # headless MuJoCo
#
# Go2 will autonomously explore the environment using TARE planner.
# TARE finds frontiers, plans TSP tours, and publishes /way_point goals.
# FAR planner routes to each waypoint. localPlanner handles obstacles.

set -e
set -m

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
NAV_STACK="/home/yusen/Desktop/vector_navigation_stack"

NO_GUI=""
for arg in "$@"; do
    case $arg in --no-gui) NO_GUI="--no-gui" ;; esac
done

VENV_SP="$REPO_DIR/.venv-nano/lib/python3.12/site-packages"
CMEEL_SP="$VENV_SP/cmeel.prefix/lib/python3.12/site-packages"
CONVEX_SRC="/home/yusen/Desktop/go2-convex-mpc/src"
export PYTHONPATH="$VENV_SP:$CMEEL_SP:$CONVEX_SRC:$REPO_DIR:$PYTHONPATH"
export ROBOT_CONFIG_PATH="unitree/unitree_go2"

source /opt/ros/jazzy/setup.bash
source "$NAV_STACK/install/setup.bash"

PIDS=()
cleanup() {
    echo ""
    echo "Stopping all processes..."
    for p in "${PIDS[@]}"; do
        kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
    done
    sleep 1
    for proc in pathFollower terrainAnalysis terrainAnalysisExt sensorScanGeneration \
                localPlanner far_planner graphDecoder visualizationTools \
                odom_transformer vehicleTransPublisher sensorTransPublisher \
                tare_planner_node go2_vnav_bridge; do
        pkill -9 -f "$proc" 2>/dev/null || true
    done
    rm -f /dev/shm/fastrtps_* /tmp/vector_nav_active 2>/dev/null
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

RVIZ_CFG="$REPO_DIR/config/vnav.rviz"

echo "============================================"
echo "  Go2 + TARE Autonomous Exploration"
echo "============================================"
echo "  MuJoCo: Go2 MPC in house scene"
echo "  Local:  localPlanner + pathFollower"
echo "  Global: FAR Planner (visibility graph)"
echo "  Explore: TARE Planner (frontier TSP)"
echo "  Terrain: terrainAnalysis + ext"
echo "============================================"

# 1. Bridge (MuJoCoGo2 → ROS2 topics)
echo "[1/8] Starting bridge..."
python3 "$SCRIPT_DIR/go2_vnav_bridge.py" $NO_GUI &
PIDS+=($!)
sleep 7

# 2. Local planner stack
echo "[2/8] Starting local planner..."
ros2 launch local_planner local_planner.launch.py \
    robot_config:=unitree/unitree_go2 \
    autonomyMode:=true \
    joyToSpeedDelay:=0.0 \
    twoWayDrive:=true &
PIDS+=($!)
sleep 4

# 3. Sensor scan generation (produces /state_estimation_at_scan for TARE)
echo "[3/8] Starting sensor scan generation..."
ros2 run sensor_scan_generation sensorScanGeneration &
PIDS+=($!)
sleep 1

# 4. Terrain analysis
echo "[4/8] Starting terrain analysis..."
ros2 run terrain_analysis terrainAnalysis --ros-args \
  -p clearDyObs:=true \
  -p minDyObsDis:=0.14 \
  -p minOutOfFovPointNum:=20 \
  -p obstacleHeightThre:=0.15 \
  -p maxRelZ:=0.3 \
  -p limitGroundLift:=true \
  -p maxGroundLift:=0.05 \
  -p minDyObsVFOV:=-55.0 \
  -p maxDyObsVFOV:=10.0 &
PIDS+=($!)
ros2 run terrain_analysis_ext terrainAnalysisExt --ros-args \
  -p obstacleHeightThre:=0.15 \
  -p maxRelZ:=0.3 &
PIDS+=($!)
sleep 3

# 5. FAR Planner (routes to TARE waypoints)
echo "[5/8] Starting FAR planner..."
ros2 launch far_planner far_planner.launch config:=indoor &
PIDS+=($!)
sleep 3

# 6. TARE Planner (autonomous exploration — includes navigationBoundary)
echo "[6/7] Starting TARE exploration planner..."
ros2 launch tare_planner explore.launch scenario:=indoor_small &
PIDS+=($!)
sleep 2

# 7. Visualization + RViz
echo "[7/7] Starting visualization + RViz..."
ros2 run visualization_tools visualizationTools 2>/dev/null &
PIDS+=($!)
rviz2 -d "$RVIZ_CFG" 2>/dev/null &
PIDS+=($!)

# Nav flag NOT created here — dog stays still after seed.
# explore.py creates /tmp/vector_nav_active when user types "explore".

# Seed with initial movement so FAR + TARE can build initial graphs
echo ""
echo "Seeding planners with initial movement..."
sleep 2
for i in $(seq 1 4); do
    ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist "{linear: {x: 0.2}}" 2>/dev/null
    sleep 1
done
ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist "{}" 2>/dev/null
sleep 2

echo ""
echo "Ready! TARE is exploring autonomously."
echo "  Watch RViz for frontier markers and planned paths."
echo "  Teleop override: use RViz teleop panel (clears autonomous path)"
echo "  Stop exploration: Ctrl+C"
echo ""

wait ${PIDS[0]}
