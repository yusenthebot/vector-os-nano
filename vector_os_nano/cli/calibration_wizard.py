"""Interactive calibration wizard for Vector OS Nano.

Guides the user through workspace calibration: collecting camera/base point
pairs, solving the transform (RBF or affine), displaying error stats, and
saving to YAML+NPY.

If Textual is installed a rich TUI is shown.
If Textual is not installed the wizard falls back to simple readline prompts.

Entry point: vector-os-calibrate  (configured in pyproject.toml)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Textual availability flag
# ---------------------------------------------------------------------------

try:
    from textual.app import App, ComposeResult
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Header,
        Input,
        Log,
        Static,
    )
    from textual.containers import Horizontal, Vertical

    TEXTUAL_AVAILABLE = True
except ImportError:  # pragma: no cover — textual may not be installed
    TEXTUAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# CalibrationWizard (core, no UI dependency)
# ---------------------------------------------------------------------------

class CalibrationWizard:
    """Workspace calibration wizard — collects point pairs and solves transform.

    Steps:
        1. Move arm to home position.
        2. Place object at known position (measure with ruler).
        3. Enter base-frame coordinates (X m, Y m, Z m).
        4. System reads camera-frame position from perception.
        5. Repeat for 25-30 points (include Z variation).
        6. Call solve() → returns Calibration + error stats.
        7. Call save(path) to persist.

    This class is UI-agnostic; the TUI/readline layers wrap it.
    """

    MIN_POINTS: int = 4
    RECOMMENDED_POINTS: int = 25

    def __init__(self) -> None:
        self._points_camera: list[np.ndarray] = []
        self._points_base: list[np.ndarray] = []
        self._calibration: Any | None = None  # vector_os_nano.perception.calibration.Calibration
        self.last_stats: dict | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_points(self) -> int:
        """Number of collected point pairs."""
        return len(self._points_camera)

    @property
    def points_camera(self) -> list[np.ndarray]:
        """Immutable view of collected camera-frame points."""
        return list(self._points_camera)

    @property
    def points_base(self) -> list[np.ndarray]:
        """Immutable view of collected base-frame points."""
        return list(self._points_base)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def add_point(
        self,
        camera: np.ndarray | list,
        base: np.ndarray | list,
    ) -> None:
        """Append a camera/base point pair.

        Args:
            camera: 3D point (3,) or list[3] in camera frame.
            base: 3D point (3,) or list[3] in arm base frame (metres).
        """
        cam = np.asarray(camera, dtype=np.float64).reshape(3)
        bas = np.asarray(base, dtype=np.float64).reshape(3)
        self._points_camera.append(cam)
        self._points_base.append(bas)
        logger.debug(
            "Point %d added: cam=%s base=%s",
            self.num_points,
            np.round(cam, 4),
            np.round(bas, 4),
        )

    def reset(self) -> None:
        """Clear all collected points and calibration state."""
        self._points_camera.clear()
        self._points_base.clear()
        self._calibration = None
        self.last_stats = None

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self) -> tuple[Any, dict]:
        """Solve the calibration from collected point pairs.

        Returns:
            (calibration, stats) tuple:
                calibration: Calibration instance (best available method)
                stats: dict with mean_m, max_m, per_point_m, num_points

        Raises:
            ValueError: If fewer than MIN_POINTS point pairs are available.
        """
        if self.num_points < self.MIN_POINTS:
            raise ValueError(
                f"Need at least {self.MIN_POINTS} points, got {self.num_points}"
            )

        pts_cam = np.array(self._points_camera, dtype=np.float64)
        pts_base = np.array(self._points_base, dtype=np.float64)

        # Warn on flat Z (delegates to calibration module)
        z_std = float(np.std(pts_cam[:, 2]))
        if z_std < 1e-6:
            logger.warning(
                "All collected camera Z values are identical (std=%.2e). "
                "Include Z variation for best accuracy.",
                z_std,
            )

        from vector_os_nano.perception.calibration import Calibration

        cal = Calibration()
        cal.solve_rbf(pts_cam, pts_base)

        # Compute error stats
        stats = cal.get_error_stats()
        stats["num_points"] = self.num_points

        self._calibration = cal
        self.last_stats = stats

        logger.info(
            "Calibration solved: %d points, mean_err=%.2f mm",
            self.num_points,
            (stats["mean_m"] or 0.0) * 1000,
        )
        return cal, stats

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save solved calibration to file.

        Args:
            path: Output file path (e.g., 'calibration.npy').

        Raises:
            RuntimeError: If solve() has not been called yet.
        """
        if self._calibration is None:
            raise RuntimeError(
                "No calibration to save — call solve() first"
            )
        self._calibration.save(path)
        logger.info("Calibration saved to %s", path)


# ---------------------------------------------------------------------------
# Readline fallback wizard
# ---------------------------------------------------------------------------

class _ReadlineWizard:
    """Simple readline-based calibration flow."""

    def __init__(self, wizard: CalibrationWizard, num_points: int = 25) -> None:
        self._wiz = wizard
        self._target = num_points

    def run(self) -> None:
        """Interactive readline calibration loop."""
        print("=" * 60)
        print("Vector OS — Workspace Calibration Wizard")
        print("=" * 60)
        print()
        print("Prerequisites:")
        print("  - Arm at HOME position")
        print("  - Camera pipeline running")
        print("  - Ruler or reference grid ready")
        print()
        print(f"Collect {self._target} point pairs (minimum {self._wiz.MIN_POINTS}).")
        print("Include points at different Z heights for best accuracy.")
        print("Type 'done' when finished, 'quit' to abort.")
        print()

        import sys

        step = 1
        while True:
            collected = self._wiz.num_points
            remaining = self._target - collected
            print(
                f"Step {step}/{self._target}  "
                f"(collected: {collected}, remaining: {remaining})"
            )
            try:
                cmd = input(
                    "Press Enter when object is in position "
                    "(or 'done'/'quit'): "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return

            if cmd == "quit":
                print("Aborted.")
                return
            if cmd == "done":
                if collected < self._wiz.MIN_POINTS:
                    print(
                        f"  Need at least {self._wiz.MIN_POINTS} points, "
                        f"have {collected}."
                    )
                    continue
                break

            # --- Prompt for camera frame ---
            print("  Enter camera-frame position (from perception pipeline):")
            cam = self._prompt_xyz("  Camera")
            if cam is None:
                continue

            # --- Prompt for base frame ---
            print(
                "  Measure base-frame position with ruler "
                "(X forward, Y left, Z up — in metres):"
            )
            bas = self._prompt_xyz("  Base")
            if bas is None:
                continue

            self._wiz.add_point(cam, bas)
            print(
                f"  Recorded: cam=({cam[0]:.4f}, {cam[1]:.4f}, {cam[2]:.4f})"
                f" -> base=({bas[0]:.4f}, {bas[1]:.4f}, {bas[2]:.4f})"
            )
            print(f"  Total points: {self._wiz.num_points}")
            print()
            step += 1

        # --- Solve ---
        print()
        print("Solving calibration...")
        try:
            cal, stats = self._wiz.solve()
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return

        self._print_stats(stats)

        # --- Save ---
        try:
            save_path = input(
                "Save to [calibration.npy]: "
            ).strip() or "calibration.npy"
        except (EOFError, KeyboardInterrupt):
            save_path = "calibration.npy"

        self._wiz.save(save_path)
        print(f"Saved to: {save_path}")

    def _prompt_xyz(self, label: str) -> np.ndarray | None:
        """Prompt for X Y Z (space-separated metres) and return array or None."""
        try:
            raw = input(f"  {label} X Y Z (m): ").strip()
            parts = raw.split()
            if len(parts) != 3:
                print("  Invalid: enter 3 numbers separated by spaces.")
                return None
            return np.array([float(p) for p in parts])
        except (EOFError, KeyboardInterrupt):
            return None
        except ValueError:
            print("  Invalid number.")
            return None

    def _print_stats(self, stats: dict) -> None:
        print()
        print("=" * 60)
        print("CALIBRATION RESULT")
        print("=" * 60)
        print(f"Points:     {stats['num_points']}")
        if stats["mean_m"] is not None:
            print(f"Mean error: {stats['mean_m'] * 1000:.2f} mm")
            print(f"Max error:  {stats['max_m'] * 1000:.2f} mm")  # type: ignore[operator]
            per_mm = [f"{e * 1000:.1f}" for e in (stats["per_point_m"] or [])]
            print(f"Per-point:  {', '.join(per_mm)} mm")
        print()


# ---------------------------------------------------------------------------
# Textual TUI app (only defined if textual is installed)
# ---------------------------------------------------------------------------

if TEXTUAL_AVAILABLE:  # pragma: no cover — only active if textual is installed

    class _CalibrationTUIApp(App):  # type: ignore[misc]
        """Textual TUI for the calibration wizard."""

        CSS = """
        Screen {
            layout: vertical;
        }
        #instructions {
            height: 6;
            border: round $primary;
            padding: 1 2;
        }
        #points_table {
            height: 1fr;
        }
        #input_row {
            height: 5;
            padding: 1 2;
        }
        #stats_panel {
            height: 5;
            border: round $success;
            padding: 1 2;
        }
        Button {
            margin: 0 1;
        }
        """

        BINDINGS = [
            ("q", "quit_app", "Quit"),
            ("s", "action_solve", "Solve"),
        ]

        def __init__(self, wizard: CalibrationWizard) -> None:
            super().__init__()
            self._wiz = wizard

        def compose(self) -> "ComposeResult":
            yield Header(show_clock=True)
            yield Static(
                "Vector OS Calibration Wizard\n"
                "Move arm to home. Place object at known positions.\n"
                "Enter camera coords, then base coords (X Y Z in metres).",
                id="instructions",
            )
            table = DataTable(id="points_table")
            table.add_columns(
                "#", "Cam X", "Cam Y", "Cam Z",
                "Base X", "Base Y", "Base Z", "Err(mm)"
            )
            yield table
            with Horizontal(id="input_row"):
                yield Input(placeholder="Cam X Y Z (m)", id="cam_input")
                yield Input(placeholder="Base X Y Z (m)", id="base_input")
                yield Button("Capture", id="btn_capture", variant="primary")
                yield Button("Solve", id="btn_solve", variant="success")
                yield Button("Save", id="btn_save", variant="warning")
                yield Button("Quit", id="btn_quit", variant="error")
            yield Static("", id="stats_panel")
            yield Footer()

        def on_button_pressed(self, event: "Button.Pressed") -> None:
            btn_id = event.button.id
            if btn_id == "btn_capture":
                self._capture_point()
            elif btn_id == "btn_solve":
                self._do_solve()
            elif btn_id == "btn_save":
                self._do_save()
            elif btn_id == "btn_quit":
                self.action_quit_app()

        def _capture_point(self) -> None:
            cam_in = self.query_one("#cam_input", Input)
            base_in = self.query_one("#base_input", Input)
            cam = self._parse_xyz(cam_in.value)
            base = self._parse_xyz(base_in.value)
            if cam is None or base is None:
                self.query_one("#stats_panel", Static).update(
                    "[red]Invalid input — enter 3 numbers for each field.[/]"
                )
                return
            self._wiz.add_point(cam, base)
            table = self.query_one("#points_table", DataTable)
            n = self._wiz.num_points
            table.add_row(
                str(n),
                f"{cam[0]:.4f}", f"{cam[1]:.4f}", f"{cam[2]:.4f}",
                f"{base[0]:.4f}", f"{base[1]:.4f}", f"{base[2]:.4f}",
                "—",
            )
            cam_in.value = ""
            base_in.value = ""

        def _do_solve(self) -> None:
            try:
                _, stats = self._wiz.solve()
            except ValueError as exc:
                self.query_one("#stats_panel", Static).update(
                    f"[red]Cannot solve: {exc}[/]"
                )
                return
            mean_mm = (stats["mean_m"] or 0.0) * 1000
            max_mm = (stats["max_m"] or 0.0) * 1000
            self.query_one("#stats_panel", Static).update(
                f"[green]Solved![/]  Points: {stats['num_points']}  "
                f"Mean: {mean_mm:.2f} mm  Max: {max_mm:.2f} mm"
            )
            # Update error column in table
            table = self.query_one("#points_table", DataTable)
            per_mm = stats["per_point_m"] or []
            for i, err in enumerate(per_mm):
                if i < table.row_count:
                    table.update_cell_at((i, 7), f"{err * 1000:.1f}")

        def _do_save(self) -> None:
            try:
                self._wiz.save("calibration.npy")
                self.query_one("#stats_panel", Static).update(
                    "[green]Saved to calibration.npy[/]"
                )
            except RuntimeError as exc:
                self.query_one("#stats_panel", Static).update(
                    f"[red]{exc}[/]"
                )

        def action_quit_app(self) -> None:
            self.exit()

        def action_solve(self) -> None:
            self._do_solve()

        @staticmethod
        def _parse_xyz(text: str) -> np.ndarray | None:
            parts = text.strip().split()
            if len(parts) != 3:
                return None
            try:
                return np.array([float(p) for p in parts])
            except ValueError:
                return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(num_points: int = 25, save_path: str = "calibration.npy") -> None:
    """Run the calibration wizard (TUI or readline fallback)."""
    wiz = CalibrationWizard()

    if TEXTUAL_AVAILABLE:  # pragma: no cover
        app = _CalibrationTUIApp(wiz)
        app.run()
    else:
        runner = _ReadlineWizard(wiz, num_points=num_points)
        runner.run()


if __name__ == "__main__":  # pragma: no cover
    main()
