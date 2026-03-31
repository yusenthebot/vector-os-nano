#!/bin/bash
# Vector Nav Stack end-to-end harness — checks every data flow layer
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
NAV_STACK="/home/yusen/Desktop/vector_navigation_stack"

VENV_SP="$REPO_DIR/.venv-nano/lib/python3.12/site-packages"
CMEEL_SP="$VENV_SP/cmeel.prefix/lib/python3.12/site-packages"
CONVEX_SRC="/home/yusen/Desktop/go2-convex-mpc/src"
export PYTHONPATH="$VENV_SP:$CMEEL_SP:$CONVEX_SRC:$REPO_DIR:$PYTHONPATH"
export ROBOT_CONFIG_PATH="unitree/unitree_go2"

source /opt/ros/jazzy/setup.bash
source "$NAV_STACK/install/setup.bash"

PIDS=()
cleanup() {
    echo "[cleanup] Stopping..."
    for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done
    wait 2>/dev/null
}
trap cleanup EXIT

PASS=0; FAIL=0
report() {
    if [ "$2" = "PASS" ]; then echo "  [PASS] $1"; PASS=$((PASS+1))
    else echo "  [FAIL] $1 — $3"; FAIL=$((FAIL+1)); fi
}

echo "======================================"
echo "  Vector Nav E2E Harness"
echo "======================================"

# --- Layer 1: Bridge ---
echo ""
echo "--- Layer 1: Bridge ---"
python3 "$SCRIPT_DIR/go2_vnav_bridge.py" --no-gui &
PIDS+=($!)
sleep 8

timeout 2 ros2 topic echo /state_estimation --once 2>/dev/null | grep -q "frame_id: map" && \
    report "L1: /state_estimation" "PASS" || report "L1: /state_estimation" "FAIL" "no data"
timeout 5 ros2 topic echo /registered_scan --once 2>/dev/null | grep -q "width:" && \
    report "L1: /registered_scan" "PASS" || \
    report "L1: /registered_scan" "FAIL" "no data"

# --- Layer 2: Sensor scan + Terrain ---
echo ""
echo "--- Layer 2: Terrain analysis ---"
ros2 run sensor_scan_generation sensorScanGeneration &
PIDS+=($!)
sleep 2

ros2 run terrain_analysis terrainAnalysis &
PIDS+=($!)
sleep 5

timeout 5 ros2 topic echo /terrain_map --once 2>/dev/null | grep -q "width:" && \
    report "L2: /terrain_map" "PASS" || report "L2: /terrain_map" "FAIL" "no data"

# --- Layer 3: Local planner ---
echo ""
echo "--- Layer 3: Local planner ---"
ros2 launch local_planner local_planner.launch.py \
    robot_config:=unitree/unitree_go2 \
    autonomyMode:=true \
    joyToSpeedDelay:=0.0 \
    twoWayDrive:=true > /tmp/lp_test.log 2>&1 &
PIDS+=($!)
sleep 5

# Check localPlanner is alive
ros2 topic info /way_point 2>/dev/null | grep -q "Subscription count: [1-9]" && \
    report "L3: localPlanner subscribes to /way_point" "PASS" || \
    report "L3: /way_point subscriber" "FAIL" "no subscriber"

# Check odom_base_link
timeout 2 ros2 topic echo /odom_base_link --once 2>/dev/null | grep -q "frame_id" && \
    report "L3: /odom_base_link" "PASS" || report "L3: /odom_base_link" "FAIL" "no data"

# --- Layer 4: Send goal and check response ---
echo ""
echo "--- Layer 4: Goal → velocity ---"

# Send waypoint (relative to Go2 at ~10,3 → send goal to 12,3)
echo "  Sending /way_point (12, 3)..."
ros2 topic pub --once /way_point geometry_msgs/msg/PointStamped \
    "{header: {frame_id: 'map'}, point: {x: 12.0, y: 3.0, z: 0.0}}" 2>/dev/null &
sleep 1

# Check if /path is published (localPlanner computed a plan)
echo "  Checking /path..."
PATH_DATA=$(timeout 5 ros2 topic echo /path --once 2>/dev/null | head -3)
if [ -n "$PATH_DATA" ]; then
    report "L4: /path published (localPlanner planned)" "PASS"
else
    report "L4: /path" "FAIL" "localPlanner did not generate a path"
fi

# Check /navigation_cmd_vel for non-zero
echo "  Checking /navigation_cmd_vel..."
NAV_CMD=$(timeout 5 ros2 topic echo /navigation_cmd_vel --once 2>/dev/null | grep -E "x:|z:" | head -4)
if echo "$NAV_CMD" | grep -q "x:"; then
    VX=$(echo "$NAV_CMD" | grep "x:" | head -1 | awk '{print $2}')
    report "L4: /navigation_cmd_vel (vx=$VX)" "PASS"
else
    report "L4: /navigation_cmd_vel" "FAIL" "no data from localPlanner"

    # Extra debug
    echo ""
    echo "  === DEBUG INFO ==="
    echo "  Nodes:"
    ros2 node list 2>/dev/null | sort
    echo "  /navigation_cmd_vel info:"
    ros2 topic info /navigation_cmd_vel 2>/dev/null
    echo "  Last 5 lines of local_planner log:"
    tail -5 /tmp/lp_test.log 2>/dev/null
    echo "  terrain_map info:"
    ros2 topic info /terrain_map 2>/dev/null
fi

# --- Layer 5: Robot actually moves ---
echo ""
echo "--- Layer 5: Robot movement ---"
# Use python to reliably parse position from /state_estimation
START_X=$(timeout 5 ros2 topic echo /state_estimation --once 2>/dev/null | python3 -c "
import sys
for line in sys.stdin:
    line = line.strip()
    if 'x:' in line and 'qx' not in line:
        val = line.split(':')[1].strip()
        try:
            f = float(val)
            if abs(f) > 1.0:  # position, not quaternion
                print(f); break
        except: pass
" 2>/dev/null)

# Keep sending goal for 15 seconds
for i in $(seq 1 5); do
    ros2 topic pub --once /way_point geometry_msgs/msg/PointStamped \
        "{header: {frame_id: 'map'}, point: {x: 12.0, y: 3.0, z: 0.0}}" 2>/dev/null
    sleep 3
done

END_X=$(timeout 5 ros2 topic echo /state_estimation --once 2>/dev/null | python3 -c "
import sys
for line in sys.stdin:
    line = line.strip()
    if 'x:' in line and 'qx' not in line:
        val = line.split(':')[1].strip()
        try:
            f = float(val)
            if abs(f) > 1.0:
                print(f); break
        except: pass
" 2>/dev/null)

if [ -n "$START_X" ] && [ -n "$END_X" ]; then
    DX=$(python3 -c "print(f'{abs(float(\"$END_X\") - float(\"$START_X\")):.3f}')" 2>/dev/null)
    if python3 -c "exit(0 if abs(float('$END_X') - float('$START_X')) > 0.3 else 1)" 2>/dev/null; then
        report "L5: Robot moved (dx=$DX)" "PASS"
    else
        report "L5: Robot movement" "FAIL" "dx=$DX (need >0.3)"
    fi
else
    report "L5: Robot movement" "FAIL" "could not read position"
fi

echo ""
echo "======================================"
echo "  RESULTS: $PASS passed, $FAIL failed"
echo "======================================"
[ $FAIL -eq 0 ] && echo "  ALL TESTS PASSED" || echo "  SOME TESTS FAILED"
exit $FAIL
