"""Direct commands without LLM.

When no llm_api_key is provided, Agent.execute() runs in direct mode.
Skill names are matched verbatim -- no planning call is made.

Requirements:
    pip install vector-os-nano

Usage:
    python examples/no_llm.py
"""

from vector_os import Agent, SO101

arm = SO101(port="/dev/ttyACM0")

# No LLM key -- direct skill execution only
agent = Agent(arm=arm)

with agent:
    # Each call maps directly to a named built-in skill
    result = agent.execute("home")
    print(f"home: {result.success}")

    result = agent.execute("scan")
    print(f"scan: {result.success}")

    # detect requires a connected camera (Intel RealSense D405)
    result = agent.execute("detect")
    print(f"detect: {result.success}")

    if result.success:
        # pick the first detected object
        result = agent.execute("pick")
        print(f"pick: {result.success}")

# arm torque disabled and serial port closed on __exit__
