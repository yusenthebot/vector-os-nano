"""Unit tests for vector_os_nano.cli.frame_renderer.

All tests use synthetic numpy arrays — no real camera required.
cv2 and rich are imported lazily; tests skip if not available.
"""
from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if cv2 is not installed
cv2 = pytest.importorskip("cv2")
rich = pytest.importorskip("rich")

from rich.text import Text  # noqa: E402 — after importorskip

# Synthetic test frames
_DUMMY_COLOR = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
_DUMMY_DEPTH = np.random.randint(0, 500, (480, 640), dtype=np.uint16)


# ---------------------------------------------------------------------------
# T1 — frame_to_rich_text returns a Rich Text instance
# ---------------------------------------------------------------------------

def test_frame_to_rich_text_returns_text():
    from vector_os_nano.cli.frame_renderer import frame_to_rich_text

    result = frame_to_rich_text(_DUMMY_COLOR, width=20, height=10)
    assert isinstance(result, Text)


# ---------------------------------------------------------------------------
# T2 — output has expected number of newline-separated rows
# ---------------------------------------------------------------------------

def test_frame_to_rich_text_dimensions():
    from vector_os_nano.cli.frame_renderer import frame_to_rich_text

    width = 20
    height = 10  # 10 character rows -> 20 pixel rows
    result = frame_to_rich_text(_DUMMY_COLOR, width=width, height=height)

    plain = result.plain
    # height rows -> (height - 1) newlines separating them
    lines = plain.split("\n")
    assert len(lines) == height, f"Expected {height} lines, got {len(lines)}"
    # Each line must have exactly `width` characters (one half-block per col)
    for i, line in enumerate(lines):
        assert len(line) == width, (
            f"Line {i} has {len(line)} chars, expected {width}"
        )


# ---------------------------------------------------------------------------
# T3 — output contains the lower half-block character U+2584
# ---------------------------------------------------------------------------

def test_frame_to_rich_text_uses_half_blocks():
    from vector_os_nano.cli.frame_renderer import HALF_BLOCK, frame_to_rich_text

    result = frame_to_rich_text(_DUMMY_COLOR, width=10, height=5)
    assert HALF_BLOCK in result.plain


# ---------------------------------------------------------------------------
# T4 — depth_to_rich_text returns a Rich Text instance from uint16 depth
# ---------------------------------------------------------------------------

def test_depth_to_rich_text_returns_text():
    from vector_os_nano.cli.frame_renderer import depth_to_rich_text

    result = depth_to_rich_text(_DUMMY_DEPTH, width=20, height=10)
    assert isinstance(result, Text)


# ---------------------------------------------------------------------------
# T5 — depth_to_rich_text produces correct dimensions
# ---------------------------------------------------------------------------

def test_depth_to_rich_text_dimensions():
    from vector_os_nano.cli.frame_renderer import depth_to_rich_text

    width, height = 15, 8
    result = depth_to_rich_text(_DUMMY_DEPTH, width=width, height=height)
    lines = result.plain.split("\n")
    assert len(lines) == height
    for line in lines:
        assert len(line) == width


# ---------------------------------------------------------------------------
# T6 — annotated_frame with empty object list does not crash
# ---------------------------------------------------------------------------

def test_annotated_frame_with_empty_objects():
    from vector_os_nano.cli.frame_renderer import annotated_frame

    result = annotated_frame(_DUMMY_COLOR, tracked_objects=[], width=20, height=10)
    assert isinstance(result, Text)


# ---------------------------------------------------------------------------
# T7 — annotated_frame with a mock tracked object draws bbox
# ---------------------------------------------------------------------------

def test_annotated_frame_with_mock_object():
    from unittest.mock import MagicMock
    from vector_os_nano.cli.frame_renderer import annotated_frame

    obj = MagicMock()
    obj.bbox_2d = (100, 100, 200, 200)
    obj.label = "cup"

    # Should not raise; returns a Rich Text with the frame rendered
    result = annotated_frame(_DUMMY_COLOR, tracked_objects=[obj], width=20, height=10)
    assert isinstance(result, Text)
    assert len(result.plain) > 0


# ---------------------------------------------------------------------------
# T8 — annotated_frame silently skips objects with None bbox_2d
# ---------------------------------------------------------------------------

def test_annotated_frame_with_none_bbox():
    from unittest.mock import MagicMock
    from vector_os_nano.cli.frame_renderer import annotated_frame

    obj = MagicMock()
    obj.bbox_2d = None
    obj.label = "unknown"

    result = annotated_frame(_DUMMY_COLOR, tracked_objects=[obj], width=10, height=5)
    assert isinstance(result, Text)


# ---------------------------------------------------------------------------
# T9 — single-row frame (height=1) does not crash
# ---------------------------------------------------------------------------

def test_frame_to_rich_text_single_row():
    from vector_os_nano.cli.frame_renderer import frame_to_rich_text

    tiny = np.zeros((2, 4, 3), dtype=np.uint8)
    result = frame_to_rich_text(tiny, width=4, height=1)
    assert isinstance(result, Text)
    # height=1 -> one row, no trailing newline
    assert "\n" not in result.plain


# ---------------------------------------------------------------------------
# T10 — all-black frame produces text with expected style count
# ---------------------------------------------------------------------------

def test_frame_to_rich_text_black_frame():
    from vector_os_nano.cli.frame_renderer import frame_to_rich_text

    black = np.zeros((100, 100, 3), dtype=np.uint8)
    result = frame_to_rich_text(black, width=10, height=5)
    assert isinstance(result, Text)
    # All spans should be present
    assert len(result.plain.replace("\n", "")) == 10 * 5


# ---------------------------------------------------------------------------
# T11 — depth zero image renders without NaN/division errors
# ---------------------------------------------------------------------------

def test_depth_to_rich_text_zero_depth():
    from vector_os_nano.cli.frame_renderer import depth_to_rich_text

    zero_depth = np.zeros((480, 640), dtype=np.uint16)
    result = depth_to_rich_text(zero_depth, width=10, height=5)
    assert isinstance(result, Text)
