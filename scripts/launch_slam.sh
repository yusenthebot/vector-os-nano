#!/bin/bash
# Go2 SLAM + Nav2 — real-time mapping and navigation
#
# Usage:
#   cd ~/Desktop/vector_os_nano
#   ./scripts/launch_slam.sh              # with MuJoCo viewer + RViz
#   ./scripts/launch_slam.sh --no-gui     # headless MuJoCo, RViz still opens
#
# This starts:
#   1. go2_nav_bridge.py (MuJoCoGo2 + ROS2 publishers)
#   2. slam_toolbox (online mapping — map builds as Go2 moves)
#   3. Nav2 navigation stack (planner + controller, NO AMCL)
#   4. RViz with point cloud + live map visualization
#
# After launch, drive Go2 to build the map:
#   ros2 topic pub /cmd_vel_nav geometry_msgs/msg/Twist \
#     "{linear: {x: 0.3}}" -r 10
#
# Or send a Nav2 goal (Nav2 plans on the SLAM map):
#   ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
#     "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 12.0, y: 3.0}, orientation: {w: 1.0}}}}" \
#     --feedback

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
NAV_WS="/home/yusen/Desktop/vector_go2_sim"

NO_GUI=""
for arg in "$@"; do
    case $arg in --no-gui) NO_GUI="--no-gui" ;; esac
done

# PYTHONPATH
VENV_SP="$REPO_DIR/.venv-nano/lib/python3.12/site-packages"
CMEEL_SP="$VENV_SP/cmeel.prefix/lib/python3.12/site-packages"
CONVEX_SRC="/home/yusen/Desktop/go2-convex-mpc/src"
export PYTHONPATH="$VENV_SP:$CMEEL_SP:$CONVEX_SRC:$REPO_DIR:$PYTHONPATH"

source /opt/ros/jazzy/setup.bash
if [ -f "$NAV_WS/install/setup.bash" ]; then
    source "$NAV_WS/install/setup.bash"
fi

set -m  # job control for process group cleanup

PIDS=()
cleanup() {
    echo ""
    echo "Stopping all processes..."
    for p in "${PIDS[@]}"; do
        kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
    done
    sleep 1
    for proc in go2_nav_bridge slam_toolbox nav2 controller_server planner_server bt_navigator; do
        pkill -9 -f "$proc" 2>/dev/null || true
    done
    rm -f /dev/shm/fastrtps_* 2>/dev/null
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

SLAM_PARAMS="$NAV_WS/src/vector_go2_navigation/config/slam_params.yaml"
NAV2_PARAMS="$NAV_WS/src/vector_go2_navigation/config/nav2_params.yaml"
RVIZ_CFG="$REPO_DIR/config/nav2_go2.rviz"

echo "======================================"
echo "  Go2 SLAM + Navigation"
echo "======================================"
echo "  Mode: Online SLAM (map builds in real-time)"
echo "  Bridge: MuJoCoGo2 MPC → /odom, /scan, /registered_scan"
echo "  SLAM: slam_toolbox async"
echo "  Nav2: MPPI + SmacPlanner2D (on live map)"
echo "======================================"

# 1. Bridge
echo "[1/4] Starting Go2 Nav Bridge..."
python3 "$SCRIPT_DIR/go2_nav_bridge.py" $NO_GUI &
PIDS+=($!)
sleep 6

# 2. SLAM Toolbox
echo "[2/4] Starting SLAM Toolbox..."
ros2 launch slam_toolbox online_async_launch.py \
    slam_params_file:="$SLAM_PARAMS" \
    use_sim_time:=false &
PIDS+=($!)
sleep 3

# 3. Nav2 (without map_server and AMCL — SLAM provides the map + TF)
echo "[3/4] Starting Nav2 (controller + planner + behaviors)..."
ros2 launch nav2_bringup navigation_launch.py \
    params_file:="$NAV2_PARAMS" \
    use_sim_time:=false \
    autostart:=true &
PIDS+=($!)
sleep 2

# 4. RViz
echo "[4/4] Starting RViz..."
rviz2 -d "$RVIZ_CFG" &
PIDS+=($!)

echo ""
echo "Ready! The map is empty — Go2 needs to move to build it."
echo ""
echo "Drive manually:    ros2 topic pub /cmd_vel_nav geometry_msgs/msg/Twist '{linear: {x: 0.3}}' -r 10"
echo "Nav2 goal:         ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \\"
echo "                     \"{pose: {header: {frame_id: 'map'}, pose: {position: {x: 12.0, y: 3.0}, orientation: {w: 1.0}}}}\" --feedback"
echo ""
echo "Press Ctrl+C to stop all."

wait ${PIDS[0]}
