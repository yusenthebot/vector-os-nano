#!/usr/bin/env python3
"""Vector OS Nano Agent with Navigation Stack — Interactive Demo.

Connects ToolAgent (LLM brain) to the navigation stack via NavStackClient.
Supports:
  - "去厨房" → NavigateSkill → NavStackClient → /way_point → robot moves
  - "探索房子" → ExploreSkill → visits all rooms, builds spatial memory
  - "我在哪" → WhereAmISkill → reports current location from memory
  - "记住这里叫充电站" → RememberLocationSkill → saves custom waypoint
  - Natural conversation in Chinese/English

Usage:
    # In a conda-deactivated terminal with nav stack running:
    ./scripts/run_agent_nav.sh
"""
import os
import sys
import threading
import time

import rclpy
from rclpy.node import Node

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_os_nano.core.agent import Agent
from vector_os_nano.core.config import load_config
from vector_os_nano.core.nav_client import NavStackClient
from vector_os_nano.core.spatial_memory import SpatialMemory
from vector_os_nano.core.tool_agent import ToolAgent
from vector_os_nano.skills.go2 import get_go2_skills


def main():
    # --- ROS2 init ---
    rclpy.init()
    node = rclpy.create_node("vector_os_nano_agent")

    # Spin ROS2 in background
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # --- NavStackClient ---
    nav = NavStackClient(node=node)
    print(f"NavStackClient: {'connected' if nav.is_available else 'NOT AVAILABLE'}")

    if not nav.is_available:
        print("ERROR: Navigation stack not running. Start it first.")
        node.destroy_node()
        rclpy.shutdown()
        return

    # Wait for state estimation
    print("Waiting for state estimation...")
    for _ in range(20):
        time.sleep(0.5)
        odom = nav.get_state_estimation()
        if odom:
            print(f"Robot at: ({odom.x:.1f}, {odom.y:.1f})")
            break
    else:
        print("No state estimation. Is the simulation running?")
        node.destroy_node()
        rclpy.shutdown()
        return

    # --- Spatial Memory ---
    memory = SpatialMemory()
    # Pre-load house layout
    rooms = [
        ("hallway", 10.0, 5.0, "corridor"),
        ("living_room", 3.0, 2.5, "room"),
        ("dining_room", 3.0, 7.5, "room"),
        ("kitchen", 17.0, 2.5, "room"),
        ("study", 17.0, 7.5, "room"),
        ("master_bedroom", 3.5, 12.0, "room"),
        ("guest_bedroom", 16.0, 12.0, "room"),
        ("bathroom", 8.5, 12.0, "room"),
    ]
    for name, x, y, cat in rooms:
        memory.add_location(name, x, y, category=cat)

    # --- Agent ---
    cfg = load_config()
    api_key = cfg.get("llm", {}).get("api_key") or os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY")
        node.destroy_node()
        rclpy.shutdown()
        return

    # Create a lightweight "base" proxy that reads position from nav stack odom
    class NavBase:
        """Proxy base that reads position from NavStackClient instead of MuJoCo."""
        name = "nav_proxy"
        supports_holonomic = True
        supports_lidar = False

        def connect(self): pass
        def disconnect(self): pass
        def stop(self): pass
        def walk(self, vx=0, vy=0, vyaw=0, duration=1): return True
        def set_velocity(self, vx, vy, vyaw): pass

        def get_position(self):
            odom = nav.get_state_estimation()
            return [odom.x, odom.y, odom.z] if odom else [0, 0, 0]

        def get_heading(self):
            import math
            odom = nav.get_state_estimation()
            if not odom:
                return 0.0
            return math.atan2(2*(odom.qw*odom.qz + odom.qx*odom.qy),
                            1 - 2*(odom.qy**2 + odom.qz**2))

        def get_velocity(self):
            odom = nav.get_state_estimation()
            return [odom.vx, odom.vy, odom.vz] if odom else [0, 0, 0]

        def get_odometry(self): return nav.get_state_estimation()
        def get_lidar_scan(self): return None

    base = NavBase()

    agent = Agent(base=base, llm_api_key=api_key, config=cfg)

    # Register Go2 skills
    for skill in get_go2_skills():
        agent._skill_registry.register(skill)

    # Inject services into agent context
    agent._nav_client = nav
    agent._spatial_memory = memory

    # Monkey-patch _build_context to include nav + memory
    _orig_build = agent._build_context
    def _patched_build():
        ctx = _orig_build()
        ctx.services["nav"] = nav
        ctx.services["spatial_memory"] = memory
        return ctx
    agent._build_context = _patched_build

    # --- ToolAgent ---
    model = cfg.get("llm", {}).get("models", {}).get("agent", "openai/gpt-4o")
    api_base = cfg.get("llm", {}).get("api_base", "https://openrouter.ai/api/v1")

    tool_agent = ToolAgent(
        agent_ref=agent,
        api_key=api_key,
        model=model,
        api_base=api_base,
    )

    # --- Interactive loop ---
    print()
    print("=" * 60)
    print("  Vector OS Nano — Go2 Agent + Navigation Stack")
    print(f"  Model: {model}")
    print(f"  Skills: {', '.join(agent.skills)}")
    print(f"  Rooms: {', '.join(l.name for l in memory.get_all_locations())}")
    print("=" * 60)
    print()
    print("Commands: '去厨房', '探索房子', '我在哪', '记住这里叫X', 'q' to quit")
    print()

    while True:
        try:
            user_input = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            break

        def on_tool(name, params):
            detail = ", ".join(f"{k}={v}" for k, v in params.items()) if params else ""
            print(f"  TOOL {name}({detail})")

        try:
            response = tool_agent.chat(user_input, on_tool_call=on_tool)
        except Exception as exc:
            response = f"Error: {exc}"

        if response:
            print(f"\n  V: {response}\n")

    print("Shutting down...")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
