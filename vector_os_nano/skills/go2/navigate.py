"""Backward-compatible re-export. Use vector_os_nano.skills.navigate instead.

This module is kept for import compatibility only. All logic now lives in
vector_os_nano/skills/navigate.py (hardware-agnostic).
"""
import warnings as _warnings

_warnings.warn(
    "vector_os_nano.skills.go2.navigate is deprecated. "
    "Import from vector_os_nano.skills.navigate instead.",
    DeprecationWarning,
    stacklevel=2,
)

from vector_os_nano.skills.navigate import NavigateSkill  # noqa: F401, E402

__all__ = ["NavigateSkill"]
