# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCoPano360 unit tests (v2.4 T4).

Most tests exercise the pure LUT helper without allocating a GL
context. Tests that need rendered cube faces use the inline tiny MJCF
fixture and check shape / aspect-ratio invariants only — actual image
content correctness is covered in
``tests/integration/test_pano360_against_world.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from vector_os_nano.hardware.sim.sensors import MuJoCoPano360, PanoSample
from vector_os_nano.hardware.sim.sensors.pano360 import (
    _build_equirec_lut,
    _stitch_equirec,
)


# ---------------------------------------------------------------------------
# LUT — pure math (no GL, no MuJoCo data)
# ---------------------------------------------------------------------------


def test_lut_shape_matches_output_resolution() -> None:
    face_idx, fx, fy = _build_equirec_lut(1920, 640, 480, math.radians(120))
    assert face_idx.shape == (640, 1920)
    assert fx.shape == (640, 1920)
    assert fy.shape == (640, 1920)


def test_lut_face_indices_in_range_zero_to_five() -> None:
    face_idx, _, _ = _build_equirec_lut(384, 128, 96, math.radians(120))
    assert face_idx.min() >= 0
    assert face_idx.max() <= 5


def test_lut_central_pixel_maps_to_front_face_centre() -> None:
    """The output centre column row (azimuth 0, elevation 0) must hit
    the forward face's central pixel."""
    out_w, out_h, face_size = 384, 128, 96
    face_idx, fx, fy = _build_equirec_lut(
        out_w, out_h, face_size, math.radians(120),
    )
    cx, cy = out_w // 2, out_h // 2
    assert face_idx[cy, cx] == 0       # _FACE_FORWARD
    half = (face_size - 1) / 2.0
    assert fx[cy, cx] == pytest.approx(half, abs=1.0)
    assert fy[cy, cx] == pytest.approx(half, abs=1.0)


def test_lut_left_quarter_maps_to_left_face() -> None:
    """Output column at u = 3*W/4 — i.e. azimuth ~ +π/2 (camera left,
    world +Y) — must hit the left-face."""
    out_w, out_h, face_size = 384, 128, 96
    face_idx, _, _ = _build_equirec_lut(
        out_w, out_h, face_size, math.radians(120),
    )
    u_left = (3 * out_w) // 4
    cy = out_h // 2
    assert face_idx[cy, u_left] == 3   # _FACE_LEFT


def test_lut_right_quarter_maps_to_right_face() -> None:
    out_w, out_h, face_size = 384, 128, 96
    face_idx, _, _ = _build_equirec_lut(
        out_w, out_h, face_size, math.radians(120),
    )
    u_right = out_w // 4
    cy = out_h // 2
    assert face_idx[cy, u_right] == 1  # _FACE_RIGHT


def test_lut_back_columns_at_left_and_right_edges() -> None:
    """Both u=0 and u=out_w-1 are near azimuth ±π → back face."""
    out_w, out_h, face_size = 384, 128, 96
    face_idx, _, _ = _build_equirec_lut(
        out_w, out_h, face_size, math.radians(120),
    )
    cy = out_h // 2
    assert face_idx[cy, 0] == 2        # _FACE_BACK
    assert face_idx[cy, out_w - 1] == 2


def test_lut_clipping_keeps_face_pixels_in_bounds() -> None:
    face_idx, fx, fy = _build_equirec_lut(384, 128, 96, math.radians(120))
    assert fx.min() >= 0.0
    assert fx.max() <= 95.0
    assert fy.min() >= 0.0
    assert fy.max() <= 95.0


def test_lut_top_row_does_not_hit_down_face() -> None:
    """The top row corresponds to elevation +vfov/2; for vfov=120° that
    is +60° which still falls in the forward/back faces, not down."""
    face_idx, _, _ = _build_equirec_lut(384, 128, 96, math.radians(120))
    assert 5 not in set(face_idx[0])    # _FACE_DOWN absent on top row


# ---------------------------------------------------------------------------
# Stitcher — pure helper
# ---------------------------------------------------------------------------


def test_stitch_red_forward_returns_red_centre() -> None:
    """Synthetic uniform-red forward face → output centre pixel red."""
    out_w, out_h, face_size = 384, 128, 96
    face_idx, fx, fy = _build_equirec_lut(
        out_w, out_h, face_size, math.radians(120),
    )
    cube = np.zeros((6, face_size, face_size, 3), dtype=np.uint8)
    cube[0, :, :, 0] = 255              # red on forward face
    pano = _stitch_equirec(cube, face_idx, fx, fy)
    assert pano[out_h // 2, out_w // 2, 0] == 255
    assert pano[out_h // 2, out_w // 2, 1] == 0
    assert pano[out_h // 2, out_w // 2, 2] == 0


def test_stitch_handles_unused_face_with_zero_pixels() -> None:
    """Faces never sampled (e.g. up/down with vfov=120) leave pano black."""
    out_w, out_h, face_size = 384, 128, 96
    face_idx, fx, fy = _build_equirec_lut(
        out_w, out_h, face_size, math.radians(120),
    )
    cube = np.zeros((6, face_size, face_size, 3), dtype=np.uint8)
    cube[0, :, :] = (255, 255, 255)     # only forward populated
    pano = _stitch_equirec(cube, face_idx, fx, fy)
    assert pano.shape == (out_h, out_w, 3)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_unknown_body_name_raises(tiny_model_data) -> None:
    with pytest.raises(ValueError, match="not found"):
        MuJoCoPano360(*tiny_model_data, body_name="nonexistent")


def test_zero_dimensions_raise(tiny_model_data) -> None:
    with pytest.raises(ValueError):
        MuJoCoPano360(*tiny_model_data, out_w=0)
    with pytest.raises(ValueError):
        MuJoCoPano360(*tiny_model_data, out_h=0)
    with pytest.raises(ValueError):
        MuJoCoPano360(*tiny_model_data, face_size=0)


def test_invalid_vfov_raises(tiny_model_data) -> None:
    with pytest.raises(ValueError):
        MuJoCoPano360(*tiny_model_data, vfov_deg=0.0)
    with pytest.raises(ValueError):
        MuJoCoPano360(*tiny_model_data, vfov_deg=200.0)


# ---------------------------------------------------------------------------
# Properties + LUT exposure
# ---------------------------------------------------------------------------


def test_out_dimension_properties(tiny_model_data) -> None:
    pano = MuJoCoPano360(
        *tiny_model_data, out_w=640, out_h=320, face_size=160,
    )
    assert pano.out_w == 640
    assert pano.out_h == 320
    assert pano.face_size == 160


def test_lut_property_returns_copies(tiny_model_data) -> None:
    pano = MuJoCoPano360(
        *tiny_model_data, out_w=384, out_h=128, face_size=96,
    )
    fi, fx, fy = pano.lut
    fi[0, 0] = 99                       # mutate
    fi2, _, _ = pano.lut
    assert fi2[0, 0] != 99


def test_image_aspect_ratio_default_3_to_1(tiny_model_data) -> None:
    pano = MuJoCoPano360(*tiny_model_data)
    assert pano.out_w / pano.out_h == 1920 / 640


# ---------------------------------------------------------------------------
# Renderer-backed step() — uses GL; flagged so CI without EGL can skip
# ---------------------------------------------------------------------------


@pytest.fixture
def small_pano(tiny_model_data):
    """Smaller-than-default pano for fast GL renders in tests."""
    return MuJoCoPano360(
        *tiny_model_data,
        out_w=192, out_h=64, face_size=64, vfov_deg=120.0, rate_hz=10.0,
    )


def test_step_returns_rgb_uint8_with_correct_shape(small_pano) -> None:
    sample = small_pano.step(now=0.0)
    assert isinstance(sample, PanoSample)
    assert sample.image.shape == (64, 192, 3)
    assert sample.image.dtype == np.uint8


def test_sample_encoding_is_rgb8(small_pano) -> None:
    sample = small_pano.step(now=0.0)
    assert sample.encoding == "rgb8"


def test_sample_height_width_match_image(small_pano) -> None:
    sample = small_pano.step(now=0.0)
    assert sample.height == 64
    assert sample.width == 192


def test_sample_frame_id_default_map(small_pano) -> None:
    sample = small_pano.step(now=0.0)
    assert sample.frame_id == "map"


def test_rate_limit_returns_cached_when_called_too_fast(small_pano) -> None:
    first = small_pano.step(now=0.0)
    second = small_pano.step(now=0.05)   # 50 ms < 100 ms (10 Hz)
    assert second is first


def test_rate_limit_yields_after_interval(small_pano) -> None:
    first = small_pano.step(now=0.0)
    second = small_pano.step(now=0.20)
    assert second is not first


def test_close_releases_renderer_idempotently(small_pano) -> None:
    small_pano.step(now=0.0)
    small_pano.close()
    small_pano.close()      # second call must be a no-op


def test_pano_sample_is_frozen(small_pano) -> None:
    sample = small_pano.step(now=0.0)
    with pytest.raises(Exception):
        sample.frame_id = "nope"        # type: ignore[misc]
