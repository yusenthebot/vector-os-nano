#!/bin/bash
# SLAM verification — tests map building during movement
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
NAV_WS="/home/yusen/Desktop/vector_go2_sim"

VENV_SP="$REPO_DIR/.venv-nano/lib/python3.12/site-packages"
CMEEL_SP="$VENV_SP/cmeel.prefix/lib/python3.12/site-packages"
CONVEX_SRC="/home/yusen/Desktop/go2-convex-mpc/src"
export PYTHONPATH="$VENV_SP:$CMEEL_SP:$CONVEX_SRC:$REPO_DIR:$PYTHONPATH"

source /opt/ros/jazzy/setup.bash
if [ -f "$NAV_WS/install/setup.bash" ]; then
    source "$NAV_WS/install/setup.bash"
fi

SLAM_PARAMS="$NAV_WS/src/vector_go2_navigation/config/slam_params.yaml"

cleanup() {
    kill $BRIDGE_PID $SLAM_PID 2>/dev/null
    wait $BRIDGE_PID $SLAM_PID 2>/dev/null
}
trap cleanup EXIT

PASS=0; FAIL=0
report() {
    if [ "$2" = "PASS" ]; then echo "  [PASS] $1"; PASS=$((PASS+1))
    else echo "  [FAIL] $1 — $3"; FAIL=$((FAIL+1)); fi
}

echo "======================================"
echo "  SLAM Verification"
echo "======================================"

# 1. Bridge
echo "[1/4] Starting bridge (headless)..."
python3 "$SCRIPT_DIR/go2_nav_bridge.py" --no-gui &
BRIDGE_PID=$!
sleep 8

# 2. SLAM
echo "[2/4] Starting slam_toolbox..."
ros2 launch slam_toolbox online_async_launch.py \
    slam_params_file:="$SLAM_PARAMS" \
    use_sim_time:=false > /tmp/slam_test.log 2>&1 &
SLAM_PID=$!
sleep 5

# S0: /map topic exists
echo "[3/4] Checking SLAM topics..."
TOPICS=$(ros2 topic list 2>/dev/null)
echo "$TOPICS" | grep -q "/map" && report "S0: /map topic published" "PASS" || report "S0: /map topic" "FAIL" "not found"

# S1: map→odom TF
TF_CHECK=$(timeout 5 ros2 run tf2_ros tf2_echo map odom 2>&1 | head -5)
if echo "$TF_CHECK" | grep -q "Translation"; then
    report "S1: TF map→odom (SLAM localization)" "PASS"
else
    report "S1: TF map→odom" "FAIL" "not published"
fi

# S2: Map grows during movement
echo "[4/4] Driving Go2 forward to build map..."

# Get initial map size
MAP_INFO_1=$(timeout 3 ros2 topic echo /map --once 2>/dev/null | grep -E "width:|height:" | head -2)
W1=$(echo "$MAP_INFO_1" | grep "width:" | awk '{print $2}')
H1=$(echo "$MAP_INFO_1" | grep "height:" | awk '{print $2}')
echo "  Initial map: ${W1}x${H1}"

# Drive forward 5s
for i in $(seq 1 10); do
    ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist \
        "{linear: {x: 0.3}}" 2>/dev/null
    sleep 0.5
done
sleep 3

# Drive with turn to explore more
for i in $(seq 1 6); do
    ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist \
        "{angular: {z: 1.0}}" 2>/dev/null
    sleep 0.5
done
sleep 2

# Drive forward again
for i in $(seq 1 10); do
    ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist \
        "{linear: {x: 0.3}}" 2>/dev/null
    sleep 0.5
done
sleep 3

# Stop
ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist "{}" 2>/dev/null

# Get final map size
MAP_INFO_2=$(timeout 3 ros2 topic echo /map --once 2>/dev/null | grep -E "width:|height:" | head -2)
W2=$(echo "$MAP_INFO_2" | grep "width:" | awk '{print $2}')
H2=$(echo "$MAP_INFO_2" | grep "height:" | awk '{print $2}')
echo "  Final map: ${W2}x${H2}"

if [ -n "$W1" ] && [ -n "$W2" ]; then
    SIZE1=$((W1 * H1))
    SIZE2=$((W2 * H2))
    if [ "$SIZE2" -gt "$SIZE1" ]; then
        report "S2: Map grew during movement (${SIZE1} → ${SIZE2} cells)" "PASS"
    else
        report "S2: Map growth" "FAIL" "size unchanged (${SIZE1} → ${SIZE2})"
    fi
else
    # If map starts empty and then gets data, that's also growth
    if [ -n "$W2" ] && [ "$W2" -gt "0" ]; then
        report "S2: Map built from scratch (${W2}x${H2})" "PASS"
    else
        report "S2: Map growth" "FAIL" "could not read map size"
    fi
fi

echo ""
echo "======================================"
echo "  RESULTS: $PASS passed, $FAIL failed"
echo "======================================"
[ $FAIL -eq 0 ] && echo "  ALL TESTS PASSED" || echo "  SOME TESTS FAILED"
exit $FAIL
