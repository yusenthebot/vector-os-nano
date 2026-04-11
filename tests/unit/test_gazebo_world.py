"""Tests for Gazebo Harmonic world files: empty_room.sdf and apartment.sdf.

Validates that world SDF files exist, are valid XML, and contain all required
structural elements (plugins, lights, ground plane, walls, furniture, objects).

Level: Unit — pure file-parsing, no Gazebo runtime required.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_WORLD_FILE = _REPO_ROOT / "gazebo" / "worlds" / "empty_room.sdf"
_APARTMENT_FILE = _REPO_ROOT / "gazebo" / "worlds" / "apartment.sdf"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sdf_tree() -> ET.ElementTree:
    """Parse empty_room.sdf once for the whole module."""
    return ET.parse(str(_WORLD_FILE))


@pytest.fixture(scope="module")
def sdf_root(sdf_tree: ET.ElementTree) -> ET.Element:
    """Return the root <sdf> element."""
    return sdf_tree.getroot()


@pytest.fixture(scope="module")
def world_elem(sdf_root: ET.Element) -> ET.Element:
    """Return the <world> element."""
    elem = sdf_root.find("world")
    assert elem is not None, "<world> element not found in SDF"
    return elem


# ---------------------------------------------------------------------------
# Existence and parse tests
# ---------------------------------------------------------------------------


class TestEmptyRoom:
    def test_empty_room_sdf_exists(self) -> None:
        """gazebo/worlds/empty_room.sdf must exist."""
        assert _WORLD_FILE.is_file(), f"File not found: {_WORLD_FILE}"

    def test_empty_room_sdf_valid_xml(self) -> None:
        """empty_room.sdf must be parseable as valid XML."""
        tree = ET.parse(str(_WORLD_FILE))
        assert tree.getroot() is not None

    # -----------------------------------------------------------------------
    # Structure
    # -----------------------------------------------------------------------

    def test_empty_room_has_world_element(self, sdf_root: ET.Element) -> None:
        """Root <sdf> element must contain a <world> child."""
        world = sdf_root.find("world")
        assert world is not None, "<world> element not found under <sdf>"
        assert world.get("name") == "empty_room", (
            f"Expected world name='empty_room', got '{world.get('name')}'"
        )

    # -----------------------------------------------------------------------
    # Plugins
    # -----------------------------------------------------------------------

    def test_empty_room_has_physics_plugin(self, sdf_root: ET.Element) -> None:
        """World must declare the Physics plugin."""
        plugin_names = {
            p.get("name", "") for p in sdf_root.findall(".//plugin")
        }
        assert "gz::sim::systems::Physics" in plugin_names, (
            f"gz::sim::systems::Physics not found. Plugins: {plugin_names}"
        )

    def test_empty_room_has_sensors_plugin(self, sdf_root: ET.Element) -> None:
        """World must declare the Sensors plugin."""
        plugin_names = {
            p.get("name", "") for p in sdf_root.findall(".//plugin")
        }
        assert "gz::sim::systems::Sensors" in plugin_names, (
            f"gz::sim::systems::Sensors not found. Plugins: {plugin_names}"
        )

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------

    def test_empty_room_has_ground_plane(self, sdf_root: ET.Element) -> None:
        """World must contain a model named 'ground_plane'."""
        model_names = {
            m.get("name", "") for m in sdf_root.findall(".//model")
        }
        assert "ground_plane" in model_names, (
            f"'ground_plane' model not found. Models: {model_names}"
        )

    def test_empty_room_has_walls(self, sdf_root: ET.Element) -> None:
        """World must contain at least 4 models with 'wall' in the name."""
        wall_models = [
            m for m in sdf_root.findall(".//model")
            if "wall" in m.get("name", "").lower()
        ]
        assert len(wall_models) >= 4, (
            f"Expected >= 4 wall models, found {len(wall_models)}: "
            f"{[m.get('name') for m in wall_models]}"
        )

    # -----------------------------------------------------------------------
    # Lights
    # -----------------------------------------------------------------------

    def test_empty_room_has_directional_light(self, sdf_root: ET.Element) -> None:
        """World must contain at least one directional light."""
        dir_lights = sdf_root.findall(".//light[@type='directional']")
        assert dir_lights, "No <light type='directional'> found in SDF"

    def test_empty_room_has_point_light(self, sdf_root: ET.Element) -> None:
        """World must contain at least one point light (ceiling lamp)."""
        point_lights = sdf_root.findall(".//light[@type='point']")
        assert point_lights, "No <light type='point'> found in SDF"

    # -----------------------------------------------------------------------
    # Dimensions
    # -----------------------------------------------------------------------

    def test_empty_room_walls_height_gte_2m(self, sdf_root: ET.Element) -> None:
        """All wall box geometries must have z-size >= 2.0 m."""
        wall_models = [
            m for m in sdf_root.findall(".//model")
            if "wall" in m.get("name", "").lower()
        ]
        assert wall_models, "No wall models found"
        for wall in wall_models:
            size_elem = wall.find(".//geometry/box/size")
            assert size_elem is not None, (
                f"No <geometry><box><size> in wall '{wall.get('name')}'"
            )
            z_size = float(size_elem.text.split()[2])
            assert z_size >= 2.0, (
                f"Wall '{wall.get('name')}' height {z_size} < 2.0 m"
            )

    def test_empty_room_floor_area_gte_20m2(self, sdf_root: ET.Element) -> None:
        """Ground plane size must be >= 4m x 5m (area >= 20 m²)."""
        ground = sdf_root.find(".//model[@name='ground_plane']")
        assert ground is not None, "'ground_plane' model not found"
        # plane geometry uses <size> with 2 values: width depth
        size_elem = ground.find(".//geometry/plane/size")
        assert size_elem is not None, (
            "No <geometry><plane><size> in ground_plane"
        )
        parts = size_elem.text.split()
        assert len(parts) >= 2, f"Unexpected plane size format: '{size_elem.text}'"
        width, depth = float(parts[0]), float(parts[1])
        area = width * depth
        assert area >= 20.0, (
            f"Ground plane area {area} m² < 20 m² (need >= 4x5)"
        )
        assert width >= 4.0, f"Ground plane width {width} < 4.0 m"
        assert depth >= 5.0 or width >= 5.0, (
            f"Neither dimension >= 5.0 m (got {width} x {depth})"
        )
