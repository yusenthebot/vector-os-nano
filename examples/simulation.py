"""Run without hardware using PyBullet simulation.

SimulatedArm is planned for Task T10 (Wave 4). This example shows what
the interface will look like once it is available.

Requirements (future):
    pip install vector-os-nano[sim]

Note:
    This example will raise ImportError until vector_os_nano.hardware.sim is
    implemented. Tracked in agents/devlog/tasks.md as T10/T11.
"""

try:
    from vector_os_nano.hardware.sim import SimulatedArm
except ImportError:
    print(
        "SimulatedArm is not yet available.\n"
        "It is scheduled for Task T10 (Wave 4).\n"
        "Install PyBullet with: pip install vector-os-nano[sim]"
    )
    raise SystemExit(1)

from vector_os_nano import Agent

# Create simulated arm (gui=True opens a PyBullet window)
arm = SimulatedArm(gui=True)
arm.connect()

# Populate the scene with test objects
arm.add_object("red_cube", position=[0.25, 0.05, 0.03], color=[1.0, 0.0, 0.0, 1.0])
arm.add_object("blue_cube", position=[0.20, -0.05, 0.03], color=[0.0, 0.0, 1.0, 1.0])

# Run agent against the simulated arm (no real hardware required)
agent = Agent(arm=arm)
agent.execute("home")
agent.execute("detect")
agent.execute("pick the red cube")

arm.disconnect()
