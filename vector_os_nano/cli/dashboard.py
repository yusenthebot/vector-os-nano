"""Vector OS Nano — Developer Dashboard (Textual TUI).

Rich terminal UI for power users. Shows live system status, joint states,
skill execution, world model, and a command input for natural language control.

Entry point: vector-os-dashboard  (configured in pyproject.toml)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        RichLog,
        Static,
        TabbedContent,
        TabPane,
    )

    TEXTUAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    TEXTUAL_AVAILABLE = False

logger = logging.getLogger(__name__)

_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Logging bridge: routes Python logging → RichLog widget
# ---------------------------------------------------------------------------

class _RichLogHandler(logging.Handler):
    """Logging handler that writes to a Textual RichLog widget."""

    def __init__(self, log_widget: Any) -> None:
        super().__init__()
        self._widget = log_widget

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            level = record.levelname
            if level == "ERROR" or level == "CRITICAL":
                self._widget.write(f"[red]{msg}[/red]")
            elif level == "WARNING":
                self._widget.write(f"[yellow]{msg}[/yellow]")
            elif level == "DEBUG":
                self._widget.write(f"[dim]{msg}[/dim]")
            else:
                self._widget.write(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main dashboard application
# ---------------------------------------------------------------------------

if TEXTUAL_AVAILABLE:

    class DashboardApp(App):  # type: ignore[misc]
        """Vector OS Nano developer dashboard.

        Provides a 4-tab Textual TUI:
        - Dashboard: live system status, joint states, skill execution, world summary
        - Log: Python logging output with level filtering
        - Skills: DataTable of all registered skills
        - World: JSON view of world model (auto-refreshing)

        A command input at the bottom is persistent across all tabs.

        Args:
            agent: An Agent instance, or None to show "not connected" status.
        """

        CSS = """
        /* Catppuccin Mocha inspired dark theme */
        Screen {
            background: #1e1e2e;
            color: #cdd6f4;
        }
        Header {
            background: #181825;
            color: #89b4fa;
        }
        Footer {
            background: #181825;
            color: #6c7086;
        }
        TabbedContent {
            height: 1fr;
        }
        TabPane {
            padding: 0;
        }
        #dashboard-layout {
            height: 1fr;
        }
        #left-col {
            width: 1fr;
            height: 100%;
        }
        #right-col {
            width: 1fr;
            height: 100%;
        }
        .panel-title {
            text-style: bold;
            color: #89b4fa;
            height: 1;
            padding: 0 1;
        }
        #status-panel {
            height: auto;
            min-height: 7;
            border: solid #585b70;
            padding: 0 1;
            margin-bottom: 1;
        }
        #skill-panel {
            height: auto;
            min-height: 4;
            border: solid #585b70;
            padding: 0 1;
            margin-bottom: 1;
        }
        #joint-panel {
            height: auto;
            min-height: 8;
            border: solid #585b70;
            padding: 0 1;
            margin-bottom: 1;
        }
        #world-panel {
            height: auto;
            min-height: 4;
            border: solid #585b70;
            padding: 0 1;
        }
        #action-buttons {
            height: 3;
            padding: 0 1;
        }
        #log-view {
            height: 1fr;
            border: solid #585b70;
        }
        #skills-table {
            height: 1fr;
        }
        #world-view {
            height: 1fr;
            border: solid #585b70;
        }
        #command-input {
            dock: bottom;
            height: 3;
            border: solid #45475a;
            background: #181825;
        }
        Button {
            margin: 0 1;
        }
        Button.primary {
            background: #89b4fa;
            color: #1e1e2e;
        }
        Button.error {
            background: #f38ba8;
            color: #1e1e2e;
        }
        """

        BINDINGS = [
            Binding("f1", "switch_tab('dashboard')", "Dashboard"),
            Binding("f2", "switch_tab('log')", "Log"),
            Binding("f3", "switch_tab('skills')", "Skills"),
            Binding("f4", "switch_tab('world')", "World"),
            Binding("ctrl+e", "estop", "E-STOP"),
            Binding("ctrl+c", "quit", "Quit"),
        ]

        def __init__(self, agent: Any = None) -> None:
            super().__init__()
            self._agent = agent
            self._log_handler: _RichLogHandler | None = None

        # ------------------------------------------------------------------
        # Compose
        # ------------------------------------------------------------------

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with TabbedContent():
                with TabPane("Dashboard", id="dashboard"):
                    with Horizontal(id="dashboard-layout"):
                        with Vertical(id="left-col"):
                            yield Label("SYSTEM STATUS", classes="panel-title")
                            yield Static(
                                self._render_status(),
                                id="status-panel",
                            )
                            yield Label("SKILL EXECUTION", classes="panel-title")
                            yield Static(
                                self._render_skill(),
                                id="skill-panel",
                            )
                            with Horizontal(id="action-buttons"):
                                yield Button("Home", id="btn-home", variant="primary")
                                yield Button("Scan", id="btn-scan")
                                yield Button("Detect", id="btn-detect")
                                yield Button("Stop", id="btn-stop", variant="error")
                        with Vertical(id="right-col"):
                            yield Label("JOINT STATES", classes="panel-title")
                            yield Static(
                                self._render_joints(),
                                id="joint-panel",
                            )
                            yield Label("WORLD MODEL", classes="panel-title")
                            yield Static(
                                self._render_world_summary(),
                                id="world-panel",
                            )
                with TabPane("Log", id="log"):
                    yield RichLog(id="log-view", highlight=True, markup=True)
                with TabPane("Skills", id="skills"):
                    yield DataTable(id="skills-table")
                with TabPane("World", id="world"):
                    yield RichLog(id="world-view", highlight=True, markup=True)
            yield Input(
                placeholder="vector> type command or natural language...",
                id="command-input",
            )
            yield Footer()

        # ------------------------------------------------------------------
        # Lifecycle
        # ------------------------------------------------------------------

        def on_mount(self) -> None:
            """Set up logging, populate skills table, start refresh timer."""
            self._setup_logging()
            self._populate_skills_table()
            self.set_interval(0.5, self._refresh_panels)
            self._log(f"[bold cyan]Vector OS Nano v{_VERSION} Dashboard started[/bold cyan]")
            if self._agent is None:
                self._log("[yellow]No agent configured — running in demo mode[/yellow]")

        # ------------------------------------------------------------------
        # Logging bridge
        # ------------------------------------------------------------------

        def _setup_logging(self) -> None:
            """Route Python logging to the Log tab RichLog widget."""
            try:
                log_view = self.query_one("#log-view", RichLog)
                handler = _RichLogHandler(log_view)
                handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
                logging.getLogger().addHandler(handler)
                self._log_handler = handler
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Skills table population
        # ------------------------------------------------------------------

        def _populate_skills_table(self) -> None:
            """Fill the skills DataTable with registered skills."""
            table = self.query_one("#skills-table", DataTable)
            table.add_columns("Skill", "Description", "Parameters", "Preconditions")
            if self._agent is None:
                return
            try:
                for name in self._agent.skills:
                    skill = self._agent._skill_registry.get(name)
                    if skill is None:
                        continue
                    params_dict = getattr(skill, "parameters", {}) or {}
                    pre_list = getattr(skill, "preconditions", []) or []
                    params_str = ", ".join(params_dict.keys()) if params_dict else "none"
                    pre_str = ", ".join(pre_list) if pre_list else "none"
                    table.add_row(
                        skill.name,
                        skill.description,
                        params_str,
                        pre_str,
                    )
            except Exception as exc:
                logger.debug("Error populating skills table: %s", exc)

        # ------------------------------------------------------------------
        # Panel refresh (called at 2Hz)
        # ------------------------------------------------------------------

        def _refresh_panels(self) -> None:
            """Update all live panels."""
            self._update_status_panel()
            self._update_joint_panel()
            self._update_skill_panel()
            self._update_world_panel()
            self._update_world_tab()

        def _update_status_panel(self) -> None:
            try:
                self.query_one("#status-panel", Static).update(self._render_status())
            except Exception:
                pass

        def _update_joint_panel(self) -> None:
            try:
                self.query_one("#joint-panel", Static).update(self._render_joints())
            except Exception:
                pass

        def _update_skill_panel(self) -> None:
            try:
                self.query_one("#skill-panel", Static).update(self._render_skill())
            except Exception:
                pass

        def _update_world_panel(self) -> None:
            try:
                self.query_one("#world-panel", Static).update(self._render_world_summary())
            except Exception:
                pass

        def _update_world_tab(self) -> None:
            """Refresh the World tab JSON view."""
            if self._agent is None:
                return
            try:
                world_view = self.query_one("#world-view", RichLog)
                world_view.clear()
                world_dict = self._agent.world.to_dict()
                world_view.write(json.dumps(world_dict, indent=2, default=str))
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Render helpers (return Rich markup strings)
        # ------------------------------------------------------------------

        def _render_status(self) -> str:
            """Render the system status panel content."""
            if self._agent is None:
                return (
                    "  Arm:      [red]not connected[/red]\n"
                    "  Camera:   [red]not connected[/red]\n"
                    "  LLM:      [yellow]none[/yellow]\n"
                    "  Tracking: [dim]idle[/dim]\n"
                    "  Objects:  [dim]0 detected[/dim]"
                )
            try:
                arm_status = (
                    "[green]connected[/green]"
                    if self._agent._arm is not None
                    else "[red]disconnected[/red]"
                )
                cam_status = (
                    "[green]connected[/green]"
                    if self._agent._perception is not None
                    else "[red]disconnected[/red]"
                )
                llm_status = (
                    "[green]configured[/green]"
                    if self._agent._llm is not None
                    else "[yellow]none[/yellow]"
                )
                objects = self._agent.world.get_objects()
                obj_count = len(objects)
                return (
                    f"  Arm:      {arm_status}\n"
                    f"  Camera:   {cam_status}\n"
                    f"  LLM:      {llm_status}\n"
                    f"  Tracking: [dim]idle[/dim]\n"
                    f"  Objects:  {obj_count} detected"
                )
            except Exception as exc:
                return f"  [red]Status error: {exc}[/red]"

        def _render_joints(self) -> str:
            """Render the joint states panel content."""
            joint_names = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ]
            if self._agent is None or self._agent._arm is None:
                lines = []
                for name in joint_names[:-1]:
                    lines.append(f"  {name:<16} [dim]N/A[/dim]")
                lines.append(f"  {'gripper':<16} [dim]N/A[/dim]")
                return "\n".join(lines)

            try:
                positions = self._agent._arm.get_joint_positions()
                if positions is None:
                    raise ValueError("No joint positions")
                lines = []
                for i, name in enumerate(joint_names[:-1]):
                    if i < len(positions):
                        val = positions[i]
                        lines.append(f"  {name:<16} [cyan]{val:>8.3f} rad[/cyan]")
                    else:
                        lines.append(f"  {name:<16} [dim]N/A[/dim]")
                # Gripper state
                if self._agent._gripper is not None:
                    g_pos = self._agent._gripper.get_position()
                    g_state = "OPEN" if g_pos > 0.5 else "CLOSED"
                    lines.append(f"  {'gripper':<16} [green]{g_state}[/green]")
                else:
                    lines.append(f"  {'gripper':<16} [dim]N/A[/dim]")
                return "\n".join(lines)
            except Exception as exc:
                return f"  [red]Joint read error: {exc}[/red]"

        def _render_skill(self) -> str:
            """Render the skill execution panel content."""
            if self._agent is None:
                return (
                    "  Current:  [dim]idle[/dim]\n"
                    "  Last:     [dim]none[/dim]\n"
                    "  Steps:    0/0"
                )
            return (
                "  Current:  [dim]idle[/dim]\n"
                "  Last:     [dim]none[/dim]\n"
                "  Steps:    0/0"
            )

        def _render_world_summary(self) -> str:
            """Render the world model summary panel content."""
            if self._agent is None:
                return (
                    "  Objects:  [dim]0[/dim]\n"
                    "  Robot:    gripper=[dim]unknown[/dim]"
                )
            try:
                objects = self._agent.world.get_objects()
                robot = self._agent.world.get_robot()
                lines = [f"  Objects ({len(objects)}):"]
                for obj in objects[:4]:  # Show up to 4 objects
                    lines.append(
                        f"    [cyan]{obj.label:<10}[/cyan] "
                        f"{obj.confidence:.2f}  {obj.state}"
                    )
                if len(objects) > 4:
                    lines.append(f"    ... +{len(objects) - 4} more")
                lines.append(f"  Robot:    gripper=[green]{robot.gripper_state}[/green]")
                if robot.held_object:
                    lines.append(f"  Holding:  [yellow]{robot.held_object}[/yellow]")
                return "\n".join(lines)
            except Exception as exc:
                return f"  [red]World error: {exc}[/red]"

        # ------------------------------------------------------------------
        # Command input handling
        # ------------------------------------------------------------------

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            """Handle command input — route to agent.execute()."""
            text = event.value.strip()
            if not text:
                return
            event.input.value = ""
            self._log(f"[bold]> {text}[/bold]")

            if self._agent is None:
                self._log("[red]No agent configured[/red]")
                return

            self.run_worker(self._execute_command(text), exclusive=False)

        async def _execute_command(self, text: str) -> None:
            """Execute a command via the agent (runs in a Textual worker)."""
            try:
                result = self._agent.execute(text)
                if result.success:
                    self._log(
                        f"[green]OK[/green] "
                        f"({result.steps_completed}/{result.steps_total} steps)"
                    )
                else:
                    status = result.status or "failed"
                    if status == "clarification_needed":
                        self._log(
                            f"[yellow]Question:[/yellow] {result.clarification_question}"
                        )
                    else:
                        self._log(f"[red]FAILED[/red]: {result.failure_reason}")
            except Exception as exc:
                self._log(f"[red]Error: {exc}[/red]")

        # ------------------------------------------------------------------
        # Button handlers
        # ------------------------------------------------------------------

        def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle quick action button presses."""
            btn_id = event.button.id
            if btn_id == "btn-home":
                self.run_worker(self._execute_command("home"), exclusive=False)
            elif btn_id == "btn-scan":
                self.run_worker(self._execute_command("scan"), exclusive=False)
            elif btn_id == "btn-detect":
                self.run_worker(
                    self._execute_command("detect all objects"), exclusive=False
                )
            elif btn_id == "btn-stop":
                self.action_estop()

        # ------------------------------------------------------------------
        # Actions (key bindings)
        # ------------------------------------------------------------------

        def action_estop(self) -> None:
            """Emergency stop — halts arm immediately."""
            if self._agent is not None:
                try:
                    self._agent.stop()
                except Exception as exc:
                    logger.warning("E-stop error: %s", exc)
            self._log("[bold red]E-STOP ACTIVATED[/bold red]")

        def action_switch_tab(self, tab_id: str) -> None:
            """Switch the active tab by ID."""
            try:
                self.query_one(TabbedContent).active = tab_id
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Log helper
        # ------------------------------------------------------------------

        def _log(self, message: str) -> None:
            """Write a message to the Log tab RichLog widget."""
            try:
                log_view = self.query_one("#log-view", RichLog)
                log_view.write(message)
            except Exception:
                pass

else:  # pragma: no cover
    # Stub when textual is not installed — prevents ImportError at module level
    class DashboardApp:  # type: ignore[no-redef]
        """Stub DashboardApp — textual not installed."""

        CSS = ""
        BINDINGS = []

        def __init__(self, agent: Any = None) -> None:
            self._agent = agent

        def run(self) -> None:
            print(
                "ERROR: textual is not installed. "
                "Install it with: pip install 'vector-os-nano[tui]'"
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for 'vector-os-dashboard' console script.

    Boots the full stack (arm + camera + VLM + tracker + calibration + LLM)
    then launches the Textual TUI. Same hardware init as run.py.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Vector OS Nano Dashboard")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--no-arm", action="store_true", help="No arm hardware")
    parser.add_argument("--no-perception", action="store_true", help="No camera/VLM")
    parser.add_argument("--llm-key", default=None, help="LLM API key")
    parser.add_argument("--config", default=None, help="Config YAML path")
    args = parser.parse_args()

    if not TEXTUAL_AVAILABLE:
        print("ERROR: textual not installed. pip install 'vector-os-nano[tui]'")
        sys.exit(1)

    from vector_os_nano.core.agent import Agent
    from vector_os_nano.core.config import load_config

    cfg = load_config(args.config or "config/user.yaml")

    # --- Arm ---
    arm = None
    gripper = None
    if not args.no_arm:
        try:
            from vector_os_nano.hardware.so101 import SO101Arm, SO101Gripper
            port = args.port or cfg.get("arm", {}).get("port", "/dev/ttyACM0")
            arm = SO101Arm(port=port)
            print(f"Connecting arm on {port}...")
            arm.connect()
            gripper = SO101Gripper(arm._bus)
            print(f"Arm connected. Joints: {[round(j,2) for j in arm.get_joint_positions()]}")
        except Exception as exc:
            print(f"Arm not available: {exc}")

    # --- Perception ---
    perception = None
    if not args.no_perception:
        try:
            from vector_os_nano.perception.realsense import RealSenseCamera
            from vector_os_nano.perception.vlm import VLMDetector
            from vector_os_nano.perception.tracker import EdgeTAMTracker
            from vector_os_nano.perception.pipeline import PerceptionPipeline

            print("Connecting camera...")
            camera = RealSenseCamera()
            camera.connect()
            print("Camera connected.")

            vlm_model = cfg.get("perception", {}).get("vlm_model") or os.environ.get("MOONDREAM_MODEL", "vikhyatk/moondream2")
            os.environ.setdefault("MOONDREAM_MODEL", vlm_model)
            print(f"Loading VLM ({vlm_model})...")
            vlm = VLMDetector()
            print("VLM loaded.")

            print("Loading tracker (EdgeTAM)...")
            tracker = EdgeTAMTracker()
            print("Tracker loaded.")

            perception = PerceptionPipeline(camera=camera, vlm=vlm, tracker=tracker)
            print("Perception ready.")
        except Exception as exc:
            print(f"Perception not available: {exc}")

    # --- Calibration ---
    calibration = None
    cal_file = cfg.get("calibration", {}).get("file", "config/workspace_calibration.yaml")
    if cal_file and os.path.exists(cal_file):
        try:
            import yaml
            import numpy as np
            from vector_os_nano.perception.calibration import Calibration
            with open(cal_file) as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict) and "transform_matrix" in data:
                cal = Calibration()
                cal._matrix = np.array(data["transform_matrix"], dtype=np.float64)
                pts_cam = data.get("points_camera")
                pts_base = data.get("points_base")
                if pts_cam and pts_base:
                    cal._cal_points_cam = np.array(pts_cam, dtype=np.float64)
                    cal._cal_points_base = np.array(pts_base, dtype=np.float64)
                calibration = cal
                print(f"Calibration loaded ({data.get('num_points', '?')} points)")
        except Exception as exc:
            print(f"Calibration failed: {exc}")

    # --- Agent ---
    api_key = args.llm_key or cfg.get("llm", {}).get("api_key") or os.environ.get("OPENROUTER_API_KEY")
    agent = Agent(arm=arm, gripper=gripper, perception=perception, llm_api_key=api_key, config=cfg)
    if calibration:
        agent._calibration = calibration

    print(f"\nSkills: {', '.join(agent.skills)}")
    print(f"LLM: {'configured' if api_key else 'none'}")
    print(f"Perception: {'ready' if perception else 'not available'}")
    print(f"Calibration: {'loaded' if calibration else 'not loaded'}\n")

    # --- Launch TUI ---
    app = DashboardApp(agent=agent)
    try:
        app.run()
    finally:
        if arm:
            arm.disconnect()
            print("Arm disconnected.")
        if perception and hasattr(perception, "stop_continuous_tracking"):
            perception.stop_continuous_tracking()


if __name__ == "__main__":  # pragma: no cover
    main()
