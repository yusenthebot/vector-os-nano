"""Tests for ros_gz_bridge YAML configuration.

Validates that gazebo/config/bridge.yaml exists, is valid YAML,
and maps all required topics between Gazebo Harmonic and ROS2.

Level: Unit — pure file-parsing, no ROS2 or Gazebo runtime needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_BRIDGE_YAML = _REPO_ROOT / "gazebo" / "config" / "bridge.yaml"

# Required topic names from the nav stack mapping table
_REQUIRED_TOPICS = {
    "/state_estimation",
    "/registered_scan",
    "/camera/image",
    "/camera/depth",
    "/cmd_vel_nav",
    "/clock",
    "/imu/data",
}

_VALID_DIRECTIONS = {"GZ_TO_ROS", "ROS_TO_GZ"}
_REQUIRED_FIELDS = {"topic_name", "ros_type_name", "gz_type_name", "direction"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_bridge() -> list[dict]:
    """Load and return the bridge YAML as a list of mappings."""
    return yaml.safe_load(_BRIDGE_YAML.read_text(encoding="utf-8"))


def _find_entry(entries: list[dict], topic: str) -> dict | None:
    """Return the first entry whose topic_name matches, or None."""
    for entry in entries:
        if entry.get("topic_name") == topic:
            return entry
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bridge_yaml_exists() -> None:
    assert _BRIDGE_YAML.exists(), f"bridge.yaml not found at {_BRIDGE_YAML}"


def test_bridge_yaml_valid_syntax() -> None:
    content = _BRIDGE_YAML.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content)
    assert parsed is not None, "bridge.yaml must not be empty"


def test_bridge_is_list() -> None:
    entries = _load_bridge()
    assert isinstance(entries, list), (
        f"bridge.yaml top-level must be a list, got {type(entries).__name__}"
    )


def test_bridge_has_state_estimation() -> None:
    entries = _load_bridge()
    entry = _find_entry(entries, "/state_estimation")
    assert entry is not None, "bridge.yaml must have an entry for /state_estimation"


def test_bridge_has_registered_scan() -> None:
    entries = _load_bridge()
    entry = _find_entry(entries, "/registered_scan")
    assert entry is not None, "bridge.yaml must have an entry for /registered_scan"


def test_bridge_has_camera_image() -> None:
    entries = _load_bridge()
    entry = _find_entry(entries, "/camera/image")
    assert entry is not None, "bridge.yaml must have an entry for /camera/image"


def test_bridge_has_camera_depth() -> None:
    entries = _load_bridge()
    entry = _find_entry(entries, "/camera/depth")
    assert entry is not None, "bridge.yaml must have an entry for /camera/depth"


def test_bridge_has_cmd_vel_nav() -> None:
    """cmd_vel_nav must exist AND be ROS_TO_GZ (commands flow from ROS to Gazebo)."""
    entries = _load_bridge()
    entry = _find_entry(entries, "/cmd_vel_nav")
    assert entry is not None, "bridge.yaml must have an entry for /cmd_vel_nav"
    assert entry.get("direction") == "ROS_TO_GZ", (
        f"/cmd_vel_nav direction must be ROS_TO_GZ, got {entry.get('direction')!r}"
    )


def test_bridge_has_clock() -> None:
    entries = _load_bridge()
    entry = _find_entry(entries, "/clock")
    assert entry is not None, "bridge.yaml must have an entry for /clock"


def test_bridge_has_imu() -> None:
    entries = _load_bridge()
    entry = _find_entry(entries, "/imu/data")
    assert entry is not None, "bridge.yaml must have an entry for /imu/data"


def test_bridge_all_directions_valid() -> None:
    entries = _load_bridge()
    for entry in entries:
        direction = entry.get("direction")
        assert direction in _VALID_DIRECTIONS, (
            f"Entry {entry.get('topic_name')!r} has invalid direction {direction!r}; "
            f"must be one of {_VALID_DIRECTIONS}"
        )


def test_bridge_all_entries_have_required_fields() -> None:
    entries = _load_bridge()
    for entry in entries:
        missing = _REQUIRED_FIELDS - set(entry.keys())
        assert not missing, (
            f"Entry {entry.get('topic_name', '<unknown>')!r} is missing "
            f"required fields: {missing}"
        )
