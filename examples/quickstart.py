"""Vector OS Nano -- Quick Start (10 lines).

Requirements:
    pip install vector-os-nano

Hardware:
    SO-101 arm connected via USB (/dev/ttyACM0 on Linux, COM3 on Windows)
    Intel RealSense D405 for object detection (optional)

Usage:
    python examples/quickstart.py
"""

from vector_os import Agent, SO101

# Connect to SO-101 arm
arm = SO101(port="/dev/ttyACM0")

# Create agent with LLM (Claude via OpenRouter by default)
agent = Agent(arm=arm, llm_api_key="your-key-here")

# Natural language control -- the LLM plans, the executor runs
result = agent.execute("pick up the red cup")
print(f"Success: {result.success}")

if not result.success:
    print(f"Error: {result.error}")
    if result.clarification:
        print(f"Agent asks: {result.clarification}")
