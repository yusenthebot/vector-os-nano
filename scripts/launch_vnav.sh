#!/bin/bash
# Go2 + Vector Navigation Stack — one-command launch
#
# Usage:
#   cd ~/Desktop/vector_os_nano
#   ./scripts/launch_vnav.sh              # MuJoCo viewer + RViz
#   ./scripts/launch_vnav.sh --no-gui     # headless MuJoCo
#
# Send a goal:
#   ros2 topic pub --once /way_point geometry_msgs/msg/PointStamped \
#     "{header: {frame_id: 'map'}, point: {x: 5.0, y: 3.0, z: 0.0}}"

set -e

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

cleanup() {
    echo ""
    echo "Stopping all..."
    kill $BRIDGE_PID $LP_PID $SSG_PID $TA_PID $TAE_PID $VIS_PID $RVIZ_PID 2>/dev/null
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT

RVIZ_CFG="$REPO_DIR/config/nav2_go2.rviz"

echo "======================================"
echo "  Go2 + Vector Navigation Stack"
echo "======================================"
echo "  MuJoCo: Go2 MPC in house scene"
echo "  Nav: localPlanner + pathFollower + terrain analysis"
echo "  Config: unitree_go2 (autonomyMode=true)"
echo "======================================"

# 1. Bridge
echo "[1/5] Starting bridge..."
python3 "$SCRIPT_DIR/go2_vnav_bridge.py" $NO_GUI &
BRIDGE_PID=$!
sleep 7

# 2. Local planner stack (includes localPlanner, pathFollower, odomTransformer, static TFs)
echo "[2/5] Starting local planner (autonomyMode=true)..."
ros2 launch local_planner local_planner.launch.py \
    robot_config:=unitree/unitree_go2 \
    autonomyMode:=true \
    joyToSpeedDelay:=0.0 \
    twoWayDrive:=true &
LP_PID=$!
sleep 3

# 3. Sensor scan generation
echo "[3/5] Starting sensor scan generation..."
ros2 run sensor_scan_generation sensorScanGeneration &
SSG_PID=$!
sleep 1

# 4. Terrain analysis
echo "[4/5] Starting terrain analysis..."
ros2 run terrain_analysis terrainAnalysis &
TA_PID=$!
ros2 run terrain_analysis_ext terrainAnalysisExt &
TAE_PID=$!
sleep 2

# 5. Visualization + RViz
echo "[5/5] Starting visualization..."
ros2 run visualization_tools visualizationTools &
VIS_PID=$!
rviz2 -d "$RVIZ_CFG" &
RVIZ_PID=$!

echo ""
echo "Ready! Send a navigation goal:"
echo "  ros2 topic pub --once /way_point geometry_msgs/msg/PointStamped \\"
echo "    \"{header: {frame_id: 'map'}, point: {x: 5.0, y: 3.0, z: 0.0}}\""
echo ""
echo "Press Ctrl+C to stop."

wait $BRIDGE_PID
