"""Render camera frames as colored Unicode half-blocks for TUI display.

Each character cell represents 2 vertical pixels using U+2584 (lower half block):
- Background color = top pixel
- Foreground color = bottom pixel

This gives double vertical resolution compared to single-character approaches.
At 60 columns x 30 rows, the effective resolution is 60x60 pixels.
"""
from __future__ import annotations

import numpy as np

try:
    from rich.text import Text
    from rich.style import Style
    from rich.color import Color

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    RICH_AVAILABLE = False


HALF_BLOCK = "\u2584"  # ▄ lower half block


def frame_to_rich_text(
    frame: np.ndarray,
    width: int = 60,
    height: int = 30,
) -> "Text":
    """Convert a BGR/RGB numpy image to a Rich Text object using half-block characters.

    Each character cell renders 2 vertical pixels:
      - The character's background color = top pixel RGB
      - The character's foreground color = bottom pixel RGB

    Args:
        frame: (H, W, 3) uint8 image — BGR (OpenCV default) or RGB.
        width: Output width in character columns.
        height: Output height in character rows.  Actual pixel rows = height * 2.

    Returns:
        Rich Text object renderable in any Textual Static widget.
    """
    import cv2  # lazy import — may not be installed

    pixel_h = height * 2

    # Resize to target pixel dimensions
    resized = cv2.resize(frame, (width, pixel_h), interpolation=cv2.INTER_AREA)

    # Convert BGR -> RGB (OpenCV captures in BGR by default)
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    else:
        rgb = resized

    text = Text()

    for row in range(0, pixel_h, 2):
        for col in range(width):
            # Top pixel -> background color
            tr = int(rgb[row, col, 0])
            tg = int(rgb[row, col, 1])
            tb = int(rgb[row, col, 2])

            # Bottom pixel -> foreground color
            bottom_row = row + 1
            if bottom_row < pixel_h:
                br = int(rgb[bottom_row, col, 0])
                bg_val = int(rgb[bottom_row, col, 1])
                bb = int(rgb[bottom_row, col, 2])
            else:
                # Last row has no pair — use same color for both
                br, bg_val, bb = tr, tg, tb

            style = Style(
                color=Color.from_rgb(br, bg_val, bb),
                bgcolor=Color.from_rgb(tr, tg, tb),
            )
            text.append(HALF_BLOCK, style=style)

        # Append newline between rows (not after the last row)
        if row + 2 < pixel_h:
            text.append("\n")

    return text


def depth_to_rich_text(
    depth: np.ndarray,
    width: int = 60,
    height: int = 30,
) -> "Text":
    """Convert a uint16 depth image to a JET-colormap Rich Text.

    The depth range 0–500 mm is mapped to the full JET spectrum.  Values
    above 500 mm are clipped.

    Args:
        depth: (H, W) uint16 array of depth in millimeters.
        width: Output width in character columns.
        height: Output height in character rows.

    Returns:
        Rich Text object renderable in any Textual Static widget.
    """
    import cv2  # lazy import

    # Normalize 0-500 mm to 0-255 for JET colormap
    depth_f = np.clip(depth.astype(np.float32), 0, 500)
    depth_u8 = (depth_f / 500.0 * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

    return frame_to_rich_text(depth_colored, width, height)


def annotated_frame(
    frame: np.ndarray,
    tracked_objects: list,
    width: int = 60,
    height: int = 30,
) -> "Text":
    """Render frame with bounding box and label overlays for tracked objects.

    Draws green rectangles and label text for each object before converting
    to half-block Rich Text.  Falls back gracefully if an object lacks bbox_2d.

    Args:
        frame: (H, W, 3) uint8 BGR image from OpenCV.
        tracked_objects: List of TrackedObject (or any object with bbox_2d and label).
        width: Output width in character columns.
        height: Output height in character rows.

    Returns:
        Rich Text with bounding box overlays rendered as half-blocks.
    """
    import cv2  # lazy import

    display = frame.copy()

    for obj in tracked_objects:
        bbox = getattr(obj, "bbox_2d", None)
        if bbox is not None:
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = str(getattr(obj, "label", "object"))
                cv2.putText(
                    display,
                    label,
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            except (TypeError, ValueError):
                pass  # malformed bbox — skip overlay for this object

    return frame_to_rich_text(display, width, height)
