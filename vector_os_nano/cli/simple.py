"""Vector OS Nano — Simple CLI.

readline-based interactive shell that wraps the Agent class.
No ROS2 — direct Python calls only.

Entry point: vector-os  (configured in pyproject.toml)
"""
from __future__ import annotations

import json
import logging
import readline  # noqa: F401 — side-effect: enables line editing and history
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vector_os_nano.core.types import ExecutionResult

logger = logging.getLogger(__name__)

from vector_os_nano.version import __version__ as _VERSION


class SimpleCLI:
    """Interactive command-line interface for Vector OS."""

    COMMANDS: dict[str, str] = {
        "pick":   "Pick an object: pick <object_name>",
        "place":  "Place held object: place [x y z]",
        "home":   "Move arm to home position",
        "scan":   "Move arm to scan position",
        "open":   "Open gripper",
        "close":  "Close gripper",
        "detect": "Detect objects: detect [query]",
        "stop":   "Stop all tasks, return home",
        "status": "Show system status",
        "skills": "List available skills",
        "world":  "Show world model state",
        "help":   "Show this help",
        "quit":   "Exit (also: exit, q, Ctrl-C)",
    }

    def __init__(self, agent: Any = None, verbose: bool = False) -> None:
        self._agent = agent
        self._verbose = verbose
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main readline input loop."""
        self._running = True
        self._print_banner()

        while self._running:
            try:
                user_input = input("vector> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                self._running = False
                break

            if not user_input:
                continue
            self._handle_input(user_input)

        print("Goodbye.")

    def _handle_input(self, text: str) -> None:
        """Route a line of input to the appropriate handler."""
        parts = text.split(None, 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command in ("quit", "exit", "q"):
            self._running = False
        elif command == "help":
            self._print_help()
        elif command == "status":
            self._handle_status()
        elif command == "skills":
            self._handle_skills()
        elif command == "world":
            self._handle_world()
        elif self._agent is not None:
            # All other input goes to the agent
            result = self._agent.execute(text)
            self._print_result(result)
        else:
            print(f"Unknown command: {command}. Type 'help' for commands.")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        TEAL = "\033[38;2;0;180;180m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        RESET = "\033[0m"

        logo = f"""
{TEAL}{BOLD}██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗ {RESET}
{TEAL}{BOLD}██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗{RESET}
{TEAL}{BOLD}██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝{RESET}
{TEAL}{BOLD}╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗{RESET}
{TEAL}{BOLD} ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║{RESET}
{TEAL}{BOLD}  ╚═══╝  ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝{RESET}
{DIM}         O S   N A N O  —  v{_VERSION}{RESET}
"""
        print(logo)
        print(f"  {TEAL}Natural language robot arm control.{RESET}")
        print(f"  Type {TEAL}'help'{RESET} for commands, or use natural language.\n")

    def _print_help(self) -> None:
        print("\nAvailable commands:")
        for cmd, desc in self.COMMANDS.items():
            print(f"  {cmd:12s} {desc}")
        print("\nOr type any natural language instruction (requires LLM).")
        print()

    def _handle_status(self) -> None:
        if self._agent is None:
            print("No agent configured.")
            return
        print(f"Skills: {', '.join(self._agent.skills)}")
        world = self._agent.world
        objects = world.get_objects()
        robot = world.get_robot()
        print(f"Objects: {len(objects)}")
        for obj in objects:
            print(
                f"  {obj.object_id}: {obj.label} "
                f"({obj.state}, conf={obj.confidence:.2f})"
            )
        print(f"Gripper: {robot.gripper_state}")
        if robot.held_object:
            print(f"Holding: {robot.held_object}")

    def _handle_skills(self) -> None:
        if self._agent is not None:
            print("Registered skills:", ", ".join(self._agent.skills))
        else:
            print("No agent configured.")

    def _handle_world(self) -> None:
        if self._agent is None:
            print("No agent configured.")
            return
        print(json.dumps(self._agent.world.to_dict(), indent=2))

    def _print_result(self, result: "ExecutionResult") -> None:
        if result.success:
            print(f"OK ({result.steps_completed}/{result.steps_total} steps)")
        elif result.status == "clarification_needed":
            print(f"Question: {result.clarification_question}")
        else:
            print(f"FAILED: {result.failure_reason}")

        if self._verbose and result.trace:
            for step in result.trace:
                status_label = "OK" if step.status == "success" else step.status
                print(
                    f"  [{status_label}] {step.skill_name} "
                    f"({step.duration_sec:.1f}s)"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the 'vector-os' console script."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Vector OS Nano CLI")
    parser.add_argument(
        "--port", default="/dev/ttyACM0", help="Serial port for SO-101 arm"
    )
    parser.add_argument(
        "--no-arm", action="store_true", help="Run without arm hardware"
    )
    parser.add_argument(
        "--no-perception", action="store_true", help="Run without camera"
    )
    parser.add_argument(
        "--llm-key", default=None, help="LLM API key (or set OPENROUTER_API_KEY)"
    )
    parser.add_argument(
        "--config", default=None, help="Path to config YAML"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--sim", action="store_true", help="Use PyBullet simulation"
    )

    args = parser.parse_args()

    # Lazy import of Agent to keep startup fast
    try:
        from vector_os_nano.core.agent import Agent
    except ImportError as exc:
        print(f"Error: cannot import Agent: {exc}", file=sys.stderr)
        sys.exit(1)

    arm = None
    gripper = None
    perception = None

    if not args.no_arm and not args.sim:
        try:
            from vector_os_nano.hardware.so101 import SO101Arm
            arm = SO101Arm(port=args.port)
            arm.connect()
            print(f"Connected to SO-101 on {args.port}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not connect to arm: {exc}")

    if args.sim:
        print("Simulation mode (PyBullet) — not yet implemented")

    api_key = args.llm_key
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")

    agent = Agent(
        arm=arm,
        gripper=gripper,
        perception=perception,
        llm_api_key=api_key,
        config=args.config,
    )

    cli = SimpleCLI(agent=agent, verbose=args.verbose)

    try:
        cli.run()
    finally:
        if arm is not None:
            arm.disconnect()
            print("Arm disconnected.")


if __name__ == "__main__":
    main()
