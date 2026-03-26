#!/bin/bash
# Run Vector OS Nano agent with navigation stack integration
#
# Prerequisites: Unity + nav stack running in another terminal
#
# Usage:
#   conda deactivate
#   ./scripts/run_agent_nav.sh

set -e
export PATH=/usr/bin:/usr/local/bin:$PATH
export PYTHONPATH=""

source /opt/ros/humble/setup.bash
source ~/Desktop/vector_navigation_stack/install/setup.bash

NANO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$NANO_ROOT:$NANO_ROOT/.venv/lib/python3.10/site-packages:$HOME/Desktop/go2-convex-mpc/src:$PYTHONPATH"

python3 "$NANO_ROOT/scripts/agent_nav_demo.py"
