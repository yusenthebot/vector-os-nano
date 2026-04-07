"""L35b: Proxy door-chain uses SceneGraph instead of hardcoded."""
import inspect

import pytest


class TestProxyDoorChainNoHardcode:
    """Verify proxy._navigate_via_doors uses SceneGraph."""

    def test_no_import_of_room_doors(self) -> None:
        """go2_ros2_proxy.py should not import _ROOM_DOORS."""
        from vector_os_nano.hardware.sim import go2_ros2_proxy

        source = inspect.getsource(go2_ros2_proxy)
        assert "_ROOM_DOORS" not in source, "_ROOM_DOORS still referenced in proxy"

    def test_no_import_of_room_centers(self) -> None:
        """go2_ros2_proxy.py should not import _ROOM_CENTERS."""
        from vector_os_nano.hardware.sim import go2_ros2_proxy

        source = inspect.getsource(go2_ros2_proxy)
        assert "_ROOM_CENTERS" not in source, "_ROOM_CENTERS still referenced in proxy"

    def test_no_import_of_detect_current_room(self) -> None:
        """go2_ros2_proxy.py should not import _detect_current_room."""
        from vector_os_nano.hardware.sim import go2_ros2_proxy

        source = inspect.getsource(go2_ros2_proxy)
        assert "_detect_current_room" not in source

    def test_navigate_via_doors_uses_scene_graph(self) -> None:
        """_navigate_via_doors should reference self._scene_graph."""
        from vector_os_nano.hardware.sim import go2_ros2_proxy

        source = inspect.getsource(go2_ros2_proxy.Go2ROS2Proxy._navigate_via_doors)
        assert "_scene_graph" in source or "scene_graph" in source

    def test_navigate_via_doors_returns_false_without_sg(self) -> None:
        """Should return False when _scene_graph is None."""
        from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy

        proxy = Go2ROS2Proxy()
        proxy._scene_graph = None
        result = proxy._navigate_via_doors(10.0, 5.0, 30.0)
        assert result is False

    def test_navigate_via_doors_returns_false_empty_sg(self) -> None:
        """Should return False when SceneGraph has no rooms."""
        from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy
        from vector_os_nano.core.scene_graph import SceneGraph

        proxy = Go2ROS2Proxy()
        proxy._scene_graph = SceneGraph()
        result = proxy._navigate_via_doors(10.0, 5.0, 30.0)
        assert result is False
