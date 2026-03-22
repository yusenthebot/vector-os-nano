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

from vector_os_nano.version import __version__ as _VERSION

# ---------------------------------------------------------------------------
# ASCII logo — Rich markup, teal color matching Catppuccin Mocha palette
# ---------------------------------------------------------------------------

_LOGO_RICH = """\
[bold #00b4b4]██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗ [/bold #00b4b4]
[bold #00b4b4]██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗[/bold #00b4b4]
[bold #00b4b4]██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝[/bold #00b4b4]
[bold #00b4b4]╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗[/bold #00b4b4]
[bold #00b4b4] ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║[/bold #00b4b4]
[bold #00b4b4]  ╚═══╝  ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝[/bold #00b4b4]
[dim]            O S    N A N O   —   v0.1.0[/dim]"""

# Bar width for joint angle visualization
_BAR_WIDTH = 16

# Joint rad ranges (from joint_config.py — duplicated to avoid hardware import)
_JOINT_RANGES: dict[str, tuple[float, float]] = {
    "shoulder_pan":  (-1.91986,  1.91986),
    "shoulder_lift": (-1.74533,  1.74533),
    "elbow_flex":    (-1.69,     1.69),
    "wrist_flex":    (-1.65806,  1.65806),
    "wrist_roll":    (-2.74385,  2.84121),
    "gripper":       (-1.0,      1.74533),
}


# ---------------------------------------------------------------------------
# Logging bridge: routes Python logging → RichLog widget
# ---------------------------------------------------------------------------

class _SelectableRichLog(RichLog):
    """RichLog that doesn't capture mouse — allows terminal native copy/paste."""

    def on_mouse_down(self, event: Any) -> None:
        """Don't capture mouse down — let terminal handle selection."""
        pass

    def on_mouse_move(self, event: Any) -> None:
        """Don't capture mouse move — let terminal handle selection."""
        pass

    def on_mouse_up(self, event: Any) -> None:
        """Don't capture mouse up — let terminal handle selection."""
        pass


class _RichLogHandler(logging.Handler):
    """Logging handler that writes to a Textual RichLog widget.

    Thread-safe: uses app.call_from_thread() when called from worker threads
    so that log messages appear in real-time during blocking operations.
    """

    def __init__(self, log_widget: Any) -> None:
        super().__init__()
        self._widget = log_widget

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            level = record.levelname
            if level == "ERROR" or level == "CRITICAL":
                formatted = f"[red]{msg}[/red]"
            elif level == "WARNING":
                formatted = f"[yellow]{msg}[/yellow]"
            elif level == "DEBUG":
                formatted = f"[dim]{msg}[/dim]"
            else:
                formatted = msg

            # Try thread-safe write via app.call_from_thread
            app = self._widget.app
            if app is not None and hasattr(app, 'call_from_thread'):
                try:
                    app.call_from_thread(self._widget.write, formatted)
                    return
                except Exception:
                    pass
            # Fallback: direct write (works if on main thread)
            self._widget.write(formatted)
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
        #logo-banner {
            text-align: center;
            height: auto;
            max-height: 9;
            padding: 0 1;
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
        #last-result {
            height: auto;
            min-height: 2;
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
            height: 3;
            border: solid #45475a;
            background: #181825;
        }
        #main-layout {
            height: 1fr;
        }
        #chat-log {
            height: 1fr;
            border: solid #585b70;
            background: #11111b;
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
            Binding("f1", "open_fullscreen_camera", "Camera Window"),
            Binding("ctrl+e", "estop", "E-STOP"),
            Binding("ctrl+c", "quit", "Quit"),
            Binding("slash", "focus_command", "/Command"),
            Binding("escape", "focus_command", ""),
        ]

        def __init__(self, agent: Any = None) -> None:
            super().__init__()
            self._agent = agent
            self._log_handler: _RichLogHandler | None = None
            self._current_skill: str = ""
            self._skill_progress: tuple[int, int] = (0, 0)
            self._last_result: str = ""
            self._stop_requested: bool = False
            self._exec_thread: Any = None

        # ------------------------------------------------------------------
        # Compose
        # ------------------------------------------------------------------

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static(_LOGO_RICH, id="logo-banner")
            with Horizontal(id="main-layout"):
                # Left panel: status + joints + skills
                with Vertical(id="left-col"):
                    yield Label("SYSTEM STATUS", classes="panel-title")
                    yield Static(self._render_status(), id="status-panel")
                    yield Label("JOINT STATES", classes="panel-title")
                    yield Static(self._render_joints(), id="joint-panel")
                    yield Label("SKILL EXECUTION", classes="panel-title")
                    yield Static(self._render_skill(), id="skill-panel")
                    with Horizontal(id="action-buttons"):
                        yield Button("Home", id="btn-home", variant="primary")
                        yield Button("Scan", id="btn-scan")
                        yield Button("Detect", id="btn-detect")
                        yield Button("Stop", id="btn-stop", variant="error")
                # Right panel: chat log + input together
                with Vertical(id="right-col"):
                    yield Label("CHAT", classes="panel-title")
                    yield _SelectableRichLog(id="chat-log", highlight=True, markup=True, wrap=True, auto_scroll=True)
                    yield Input(
                        placeholder="vector> type command or natural language...",
                        id="command-input",
                    )
            yield Footer()

        # ------------------------------------------------------------------
        # Lifecycle
        # ------------------------------------------------------------------

        def on_mount(self) -> None:
            """Set up logging, start refresh timer, auto-launch camera."""
            self._setup_logging()
            self.set_interval(1.0, self._refresh_panels)  # 1Hz (was 0.5s, too fast)
            self._log(f"[bold #00b4b4]Vector OS Nano v{_VERSION}[/bold #00b4b4]")
            self._log("Type commands below. F1=Camera | Ctrl+E=Stop | Shift+drag=Copy text\n")
            if self._agent is None:
                self._log("[yellow]No agent configured — running in demo mode[/yellow]")
            # Auto-launch camera window if perception available
            if self._agent and self._agent._perception is not None:
                self.call_after_refresh(self.action_open_fullscreen_camera)
            # Focus command input
            self.call_after_refresh(self.action_focus_command)

        # ------------------------------------------------------------------
        # Logging bridge
        # ------------------------------------------------------------------

        def _setup_logging(self) -> None:
            """Route ALL Python logging to the chat log."""
            try:
                chat_log = self.query_one("#chat-log", RichLog)
                handler = _RichLogHandler(chat_log)
                handler.setFormatter(logging.Formatter("[dim]%(name)s:[/dim] %(message)s"))
                handler.setLevel(logging.DEBUG)
                # Attach to root logger so ALL modules' logs appear
                root = logging.getLogger()
                root.addHandler(handler)
                root.setLevel(logging.INFO)
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
            """Update all live panels (2Hz)."""
            self._update_status_panel()
            self._update_joint_panel()
            self._update_skill_panel()

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

        def _update_dash_camera_preview(self) -> None:
            """Update the small camera preview on the Dashboard tab."""
            if self._agent is None or self._agent._perception is None:
                return
            try:
                tabs = self.query_one(TabbedContent)
                if tabs.active != "dashboard":
                    return
            except Exception:
                return
            try:
                from vector_os_nano.cli.frame_renderer import annotated_frame, frame_to_rich_text
                cam = self._agent._perception._camera
                color = cam.get_color_frame()
                if color is not None:
                    tracked = getattr(self._agent._perception, "_last_tracked", [])
                    if tracked:
                        preview = annotated_frame(color, tracked, width=40, height=10)
                    else:
                        preview = frame_to_rich_text(color, width=40, height=10)
                    self.query_one("#dash-camera-preview", Static).update(preview)
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

        def _update_camera_panel(self) -> None:
            """Update camera tab with half-block rendered frames.

            Only runs when the Camera tab is active to avoid wasting CPU/GPU
            converting frames that are not currently visible.
            """
            if self._agent is None or self._agent._perception is None:
                return

            # Only update when Camera tab is visible
            try:
                tabs = self.query_one(TabbedContent)
                if tabs.active != "camera":
                    return
            except Exception:
                return

            try:
                from vector_os_nano.cli.frame_renderer import (
                    annotated_frame,
                    depth_to_rich_text,
                    frame_to_rich_text,
                )

                cam = self._agent._perception._camera
                color = cam.get_color_frame()
                depth = cam.get_depth_frame()

                if color is not None:
                    tracked = getattr(self._agent._perception, "_last_tracked", [])
                    if tracked:
                        rgb_text = annotated_frame(color, tracked, width=55, height=25)
                    else:
                        rgb_text = frame_to_rich_text(color, width=55, height=25)
                    self.query_one("#camera-rgb", Static).update(rgb_text)

                if depth is not None:
                    depth_text = depth_to_rich_text(depth, width=55, height=25)
                    self.query_one("#camera-depth", Static).update(depth_text)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Render helpers (return Rich markup strings)
        # ------------------------------------------------------------------

        def _render_status(self) -> str:
            """Render the system status panel content with colored dots."""
            if self._agent is None:
                return (
                    "  [red]●[/red] Arm disconnected\n"
                    "  [red]●[/red] Camera disconnected\n"
                    "  [yellow]●[/yellow] LLM not configured\n"
                    "  [dim]●[/dim] Tracking: idle\n"
                    "  [dim]●[/dim] Objects: 0 detected"
                )
            try:
                arm_dot = "[green]●[/green]" if self._agent._arm is not None else "[red]●[/red]"
                arm_label = "Arm connected" if self._agent._arm is not None else "Arm disconnected"

                cam_dot = "[green]●[/green]" if self._agent._perception is not None else "[red]●[/red]"
                cam_label = "Camera connected" if self._agent._perception is not None else "Camera disconnected"

                llm_dot = "[green]●[/green]" if self._agent._llm is not None else "[yellow]●[/yellow]"
                llm_label = "LLM configured" if self._agent._llm is not None else "LLM not configured"

                objects = self._agent.world.get_objects()
                obj_count = len(objects)
                return (
                    f"  {arm_dot} {arm_label}\n"
                    f"  {cam_dot} {cam_label}\n"
                    f"  {llm_dot} {llm_label}\n"
                    f"  [dim]●[/dim] Tracking: idle\n"
                    f"  [dim]●[/dim] Objects: {obj_count} detected"
                )
            except Exception as exc:
                return f"  [red]Status error: {exc}[/red]"

        @staticmethod
        def _make_bar(value: float, rad_min: float, rad_max: float, width: int = _BAR_WIDTH) -> str:
            """Return a Rich markup horizontal bar for the given joint value."""
            span = rad_max - rad_min
            if span <= 0:
                ratio = 0.5
            else:
                ratio = max(0.0, min(1.0, (value - rad_min) / span))
            filled = int(round(ratio * width))
            empty = width - filled
            bar = f"[cyan]{'█' * filled}[/cyan][dim]{'░' * empty}[/dim]"
            return f"|{bar}|"

        def _render_joints(self) -> str:
            """Render the joint states panel with bar visualization."""
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
                    bar = self._make_bar(0.0, *_JOINT_RANGES.get(name, (-1.0, 1.0)))
                    lines.append(f"  {name:<16} {bar}  [dim]N/A[/dim]")
                bar = self._make_bar(0.0, *_JOINT_RANGES["gripper"])
                lines.append(f"  {'gripper':<16} {bar}  [dim]N/A[/dim]")
                return "\n".join(lines)

            try:
                positions = self._agent._arm.get_joint_positions()
                if positions is None:
                    raise ValueError("No joint positions")
                lines = []
                for i, name in enumerate(joint_names[:-1]):
                    if i < len(positions):
                        val = positions[i]
                        rad_min, rad_max = _JOINT_RANGES.get(name, (-3.14, 3.14))
                        bar = self._make_bar(val, rad_min, rad_max)
                        lines.append(f"  {name:<16} {bar}  [cyan]{val:>8.3f} rad[/cyan]")
                    else:
                        lines.append(f"  {name:<16} [dim]N/A[/dim]")
                # Gripper state
                if self._agent._gripper is not None:
                    g_pos = self._agent._gripper.get_position()
                    g_state = "OPEN" if g_pos > 0.5 else "CLOSED"
                    rad_min, rad_max = _JOINT_RANGES["gripper"]
                    bar = self._make_bar(g_pos, rad_min, rad_max)
                    lines.append(f"  {'gripper':<16} {bar}  [green]{g_state}[/green]")
                else:
                    bar = self._make_bar(0.0, *_JOINT_RANGES["gripper"])
                    lines.append(f"  {'gripper':<16} {bar}  [dim]N/A[/dim]")
                return "\n".join(lines)
            except Exception as exc:
                return f"  [red]Joint read error: {exc}[/red]"

        def _render_skill(self) -> str:
            """Render the skill execution panel with progress bar."""
            if not self._current_skill:
                return "  [dim]●[/dim] Idle"

            steps_done, steps_total = self._skill_progress
            if steps_total > 0:
                bar_width = 8
                filled = int(round(steps_done / steps_total * bar_width))
                empty = bar_width - filled - 1
                arrow = ">" if steps_done < steps_total else "="
                progress_bar = "[" + "=" * filled + arrow + " " * max(0, empty) + "]"
                return (
                    f"  [cyan]●[/cyan] {self._current_skill} — "
                    f"[cyan]{progress_bar}[/cyan] {steps_done}/{steps_total} steps"
                )
            return f"  [cyan]●[/cyan] {self._current_skill} — [dim]running...[/dim]"

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

        def on_input_submitted(self, event: Input.Submitted) -> None:
            """Handle Enter key in command input."""
            # Grab text before clearing
            inp = self.query_one("#command-input", Input)
            text = inp.value.strip()
            if not text:
                return

            # Clear and refocus immediately
            inp.value = ""
            inp.action_end()  # move cursor to end (clears selection)
            event.prevent_default()
            event.stop()

            # Show user message in chat
            self._log(f"\n[bold #00b4b4]You:[/bold #00b4b4] {text}")

            # Handle local commands first (same as SimpleCLI)
            cmd = text.lower().strip()
            if cmd in ("help", "?"):
                self._log("[bold]Commands:[/bold]")
                self._log("  pick <object>   — Pick an object")
                self._log("  place [x y z]   — Place held object")
                self._log("  home            — Move to home position")
                self._log("  scan            — Move to scan position")
                self._log("  open / close    — Gripper control")
                self._log("  detect [query]  — Detect objects")
                self._log("  status          — System status")
                self._log("  skills          — List skills")
                self._log("  world           — World model state")
                self._log("  Or type natural language")
                return
            if cmd == "status":
                if self._agent:
                    self._log(f"Skills: {', '.join(self._agent.skills)}")
                    robot = self._agent.world.get_robot()
                    objects = self._agent.world.get_objects()
                    self._log(f"Objects: {len(objects)}, Gripper: {robot.gripper_state}")
                else:
                    self._log("[yellow]No agent configured[/yellow]")
                return
            if cmd == "skills":
                if self._agent:
                    self._log(f"[cyan]{', '.join(self._agent.skills)}[/cyan]")
                return
            if cmd == "world":
                if self._agent:
                    import json as _json
                    self._log(_json.dumps(self._agent.world.to_dict(), indent=2, default=str))
                return
            if cmd in ("stop", "e-stop", "estop"):
                self._do_stop()
                return
            if cmd in ("quit", "exit", "q", "bye"):
                self._log("[dim]Goodbye.[/dim]")
                self.exit(return_code=0)
                return

            if self._agent is None:
                self._log("[red]No agent configured[/red]")
                return

            import threading
            self._stop_requested = False
            self._exec_thread = threading.Thread(
                target=self._execute_command_sync, args=(text,), daemon=True
            )
            self._exec_thread.start()

        def _execute_command_sync(self, text: str) -> None:
            """Execute a command in a THREAD. Aborts if _stop_requested."""
            self._current_skill = text
            self._skill_progress = (0, 0)
            self.call_from_thread(self._update_skill_panel)
            self.call_from_thread(self._log, "[dim]Executing...[/dim]")
            try:
                if self._stop_requested:
                    return
                # Force max_retries=1 so agent doesn't loop 3x if stop is hit mid-execution
                cfg = self._agent._config
                cfg.setdefault("agent", {})["max_planning_retries"] = 1
                result = self._agent.execute(text)
                cfg["agent"]["max_planning_retries"] = 3  # restore
                if self._stop_requested:
                    self.call_from_thread(self._log, "[yellow]Cancelled by STOP[/yellow]")
                    return
                if result.success:
                    steps_done = getattr(result, "steps_completed", 1)
                    steps_total = getattr(result, "steps_total", 1)
                    self._skill_progress = (steps_done, steps_total)
                    self.call_from_thread(self._update_skill_panel)
                    self.call_from_thread(
                        self._log,
                        f"[bold green]Robot:[/bold green] Done ({steps_done}/{steps_total} steps)",
                    )
                    if result.trace:
                        for t in result.trace:
                            s = "[green]OK[/green]" if t.status == "success" else f"[red]{t.status}[/red]"
                            self.call_from_thread(self._log, f"  {s} {t.skill_name} ({t.duration_sec:.1f}s)")
                else:
                    status = result.status or "failed"
                    if status == "clarification_needed":
                        self.call_from_thread(
                            self._log,
                            f"[bold yellow]Robot:[/bold yellow] {result.clarification_question}",
                        )
                    else:
                        self.call_from_thread(
                            self._log,
                            f"[bold red]Robot:[/bold red] {result.failure_reason}",
                        )
            except Exception as exc:
                self.call_from_thread(self._log, f"[bold red]Error:[/bold red] {exc}")
            finally:
                self._current_skill = ""
                self._skill_progress = (0, 0)
                self._update_skill_panel()

        def _update_last_result(self) -> None:
            """Update the #last-result widget on the Dashboard tab."""
            try:
                self.query_one("#last-result", Static).update(
                    self._last_result if self._last_result else "[dim]none[/dim]"
                )
            except Exception:
                pass

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
            """Emergency stop — halts arm immediately and goes home."""
            self._do_stop()

        def _do_stop(self) -> None:
            """Stop all tasks, halt arm, go home."""
            self._stop_requested = True
            self._log("[bold red]STOP — cancelling all tasks[/bold red]")

            if self._agent is None:
                return

            # Stop arm motion immediately
            try:
                self._agent.stop()
            except Exception as exc:
                logger.warning("Stop error: %s", exc)

            # Go home in background thread
            import threading
            def _go_home():
                try:
                    self._agent.execute("home")
                    self.call_from_thread(self._log, "[green]Returned to home position[/green]")
                except Exception as exc:
                    self.call_from_thread(self._log, f"[red]Home failed: {exc}[/red]")
                finally:
                    self._stop_requested = False
                    self._current_skill = ""
                    self._skill_progress = (0, 0)

            threading.Thread(target=_go_home, daemon=True).start()

        def action_switch_tab(self, tab_id: str) -> None:
            """Switch the active tab by ID."""
            try:
                self.query_one(TabbedContent).active = tab_id
            except Exception:
                pass

        def action_focus_command(self) -> None:
            """Focus the command input bar."""
            try:
                self.query_one("#command-input", Input).focus()
            except Exception:
                pass

        def action_open_fullscreen_camera(self) -> None:
            """Launch the OpenCV RGB+Depth side-by-side viewer in a background thread.

            Mirrors the camera viewer loop from run.py.  Only one viewer thread is
            started — calling again while a viewer is running is a no-op.
            """
            if self._agent is None or self._agent._perception is None:
                self._log("[yellow]No camera available for fullscreen view[/yellow]")
                return

            if getattr(self, "_cv_viewer_active", False):
                self._log("[dim]Fullscreen viewer already running[/dim]")
                return

            try:
                import threading
                import cv2
                import numpy as np

                perception = self._agent._perception
                self._cv_viewer_active = True

                # Load PIL font for Chinese text
                from PIL import Image as PILImage, ImageDraw, ImageFont
                _pil_font = None
                for fp in [
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                ]:
                    if os.path.exists(fp):
                        try:
                            _pil_font = ImageFont.truetype(fp, 16)
                            break
                        except Exception:
                            pass

                def _put_text(img, text, pos, color=(0, 255, 0)):
                    if _pil_font is None:
                        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        return img
                    pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    ImageDraw.Draw(pil).text(pos, text, font=_pil_font, fill=color)
                    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

                def _viewer_loop() -> None:
                    try:
                        cam = perception._camera
                        cv2.namedWindow("Vector OS", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Vector OS", 1280, 480)

                        while True:
                            try:
                                color = cam.get_color_frame()
                                depth = cam.get_depth_frame()
                                if color is None or depth is None:
                                    continue

                                # --- RGB + tracking overlay ---
                                rgb_display = color.copy()
                                last_tracked = getattr(perception, "_last_tracked", [])
                                for obj in last_tracked:
                                    if obj.bbox_2d:
                                        x1, y1, x2, y2 = [int(v) for v in obj.bbox_2d]
                                        cv2.rectangle(rgb_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        lbl = getattr(obj, "label", "object")
                                        if obj.pose:
                                            lbl += f" ({obj.pose.x:.2f},{obj.pose.y:.2f},{obj.pose.z:.2f})"
                                        rgb_display = _put_text(rgb_display, lbl, (x1, max(y1 - 20, 0)))

                                # --- Depth + mask + centroid ---
                                depth_f = np.clip(depth.astype(np.float32), 0, 500)
                                depth_u8 = (depth_f / 500.0 * 255).astype(np.uint8)
                                depth_colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

                                for obj in last_tracked:
                                    # Mask contour
                                    if obj.mask is not None and obj.mask.shape == depth_colored.shape[:2]:
                                        contours, _ = cv2.findContours(
                                            obj.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                                        )
                                        cv2.drawContours(depth_colored, contours, -1, (255, 255, 255), 2)
                                    # Centroid dot
                                    if obj.pose and obj.bbox_2d:
                                        cx = int((obj.bbox_2d[0] + obj.bbox_2d[2]) / 2)
                                        cy = int((obj.bbox_2d[1] + obj.bbox_2d[3]) / 2)
                                        cv2.circle(depth_colored, (cx, cy), 6, (0, 255, 0), -1)
                                        cv2.circle(depth_colored, (cx, cy), 8, (255, 255, 255), 2)
                                        info = f"{obj.pose.x:.3f},{obj.pose.y:.3f},{obj.pose.z:.3f}"
                                        cv2.putText(depth_colored, info, (cx + 10, cy),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                                combined = np.hstack([rgb_display, depth_colored])
                                cv2.imshow("Vector OS", combined)
                                key = cv2.waitKey(33)
                                if key == 27:
                                    break
                            except Exception:
                                pass
                    finally:
                        cv2.destroyAllWindows()
                        self._cv_viewer_active = False

                thread = threading.Thread(target=_viewer_loop, daemon=True)
                thread.start()
                self._log("[green]Fullscreen camera viewer started (ESC to close)[/green]")
            except Exception as exc:
                self._cv_viewer_active = False
                self._log(f"[red]Fullscreen viewer error: {exc}[/red]")

        # ------------------------------------------------------------------
        # Log helper
        # ------------------------------------------------------------------

        def _log(self, message: str) -> None:
            """Write a message to the chat log."""
            try:
                chat_log = self.query_one("#chat-log", RichLog)
                chat_log.write(message)
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

    Delegates to run.py with --dashboard to avoid duplicating hardware init.
    """
    import sys as _sys
    if "--dashboard" not in _sys.argv and "-d" not in _sys.argv:
        _sys.argv.append("--dashboard")
    try:
        import importlib.util as _ilu
        import os as _os
        # Resolve run.py: vector_os_nano/cli/dashboard.py -> three levels up
        _run_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(
                _os.path.abspath(__file__)
            ))),
            "run.py",
        )
        _spec = _ilu.spec_from_file_location("_vector_run", _run_path)
        if _spec is not None and _spec.loader is not None:
            _run_mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_run_mod)  # type: ignore[union-attr]
            _run_mod.main()
            return
    except Exception:
        pass
    # Fallback: run.py is on sys.path (installed entry-point scenario)
    from run import main as _run_main  # type: ignore[import]
    _run_main()


if __name__ == "__main__":  # pragma: no cover
    main()
