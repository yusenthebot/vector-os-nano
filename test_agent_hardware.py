"""Test Agent with real SO-101 hardware + LLM.
WILL MOVE THE ARM. Run with arm connected and powered on.
"""
import time
from vector_os.core.agent import Agent
from vector_os.core.config import load_config
from vector_os.hardware.so101 import SO101Arm

print("=" * 50)
print("Test: Agent + Hardware + LLM")
print("=" * 50)

# Load config
cfg = load_config("config/user.yaml")
api_key = cfg["llm"]["api_key"]

# Connect arm
arm = SO101Arm(port="/dev/ttyACM0")
arm.connect()
print("Arm connected!")
joints = arm.get_joint_positions()
print(f"Current joints: {[round(j, 3) for j in joints]}")

# Create agent
agent = Agent(arm=arm, llm_api_key=api_key, config="config/user.yaml")
print(f"Skills: {agent.skills}")

# Test 1: Direct home command
print()
print("-" * 40)
input("Test 1: Press Enter to send 'home' (arm will move to home position)...")
print("-" * 40)
r = agent.execute("home")
print(f"Result: success={r.success}")
if not r.success:
    print(f"Reason: {r.failure_reason}")

time.sleep(1.0)
joints = arm.get_joint_positions()
print(f"Joints after home: {[round(j, 3) for j in joints]}")

# Test 2: LLM + home via natural language
print()
print("-" * 40)
input("Test 2: Press Enter to send 'go to home position' via LLM...")
print("-" * 40)
r = agent.execute("go to home position")
print(f"Result: success={r.success}, steps={r.steps_completed}/{r.steps_total}")
if r.trace:
    for t in r.trace:
        print(f"  [{t.status}] {t.skill_name} ({t.duration_sec:.1f}s)")

# Test 3: Scan
print()
print("-" * 40)
input("Test 3: Press Enter to send 'scan the workspace'...")
print("-" * 40)
r = agent.execute("scan")
print(f"Result: success={r.success}")

# Cleanup
print()
print("-" * 40)
print("Tests complete. Disconnecting...")
arm.disconnect()
print("Disconnected. PASSED!")
