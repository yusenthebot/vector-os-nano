#!/bin/bash
# Visualization verification — tests PointCloud2 + topics
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

VENV_SP="$REPO_DIR/.venv-nano/lib/python3.12/site-packages"
CMEEL_SP="$VENV_SP/cmeel.prefix/lib/python3.12/site-packages"
CONVEX_SRC="/home/yusen/Desktop/go2-convex-mpc/src"
export PYTHONPATH="$VENV_SP:$CMEEL_SP:$CONVEX_SRC:$REPO_DIR:$PYTHONPATH"

source /opt/ros/jazzy/setup.bash

cleanup() {
    kill $BRIDGE_PID 2>/dev/null
    wait $BRIDGE_PID 2>/dev/null
}
trap cleanup EXIT

PASS=0; FAIL=0
report() {
    if [ "$2" = "PASS" ]; then echo "  [PASS] $1"; PASS=$((PASS+1))
    else echo "  [FAIL] $1 — $3"; FAIL=$((FAIL+1)); fi
}

echo "======================================"
echo "  Visualization Verification"
echo "======================================"

echo "[1/5] Starting bridge (headless)..."
python3 "$SCRIPT_DIR/go2_nav_bridge.py" --no-gui &
BRIDGE_PID=$!
sleep 8

echo "[2/5] Checking topics..."
TOPICS=$(ros2 topic list 2>/dev/null)
echo "$TOPICS" | grep -q "/registered_scan" && report "/registered_scan published" "PASS" || report "/registered_scan" "FAIL" "topic not found"
echo "$TOPICS" | grep -q "/scan" && report "/scan published" "PASS" || report "/scan" "FAIL" "not found"
echo "$TOPICS" | grep -q "/odom" && report "/odom published" "PASS" || report "/odom" "FAIL" "not found"

echo "[3/5] Checking PointCloud2 data..."
PC_DATA=$(timeout 3 ros2 topic echo /registered_scan --once 2>/dev/null | head -10)
if echo "$PC_DATA" | grep -q "width:"; then
    WIDTH=$(echo "$PC_DATA" | grep "width:" | awk '{print $2}')
    report "PointCloud2 has $WIDTH points" "PASS"
else
    report "PointCloud2 data" "FAIL" "no message received"
fi

echo "[4/5] Checking LaserScan data..."
SCAN_DATA=$(timeout 3 ros2 topic echo /scan --once 2>/dev/null | grep "ranges:" | head -1)
if [ -n "$SCAN_DATA" ]; then
    report "LaserScan has range data" "PASS"
else
    report "LaserScan data" "FAIL" "no ranges"
fi

echo "[5/5] Checking RViz config syntax..."
if [ -f "$REPO_DIR/config/nav2_go2.rviz" ]; then
    # Check that it only uses rviz_default_plugins (no nav2_rviz_plugins)
    if grep -q "nav2_rviz_plugins" "$REPO_DIR/config/nav2_go2.rviz"; then
        report "RViz config clean (no nav2 plugins)" "FAIL" "still references nav2_rviz_plugins"
    else
        report "RViz config clean (standard plugins only)" "PASS"
    fi
else
    report "RViz config exists" "FAIL" "file not found"
fi

echo ""
echo "======================================"
echo "  RESULTS: $PASS passed, $FAIL failed"
echo "======================================"
[ $FAIL -eq 0 ] && echo "  ALL TESTS PASSED" || echo "  SOME TESTS FAILED"
exit $FAIL
