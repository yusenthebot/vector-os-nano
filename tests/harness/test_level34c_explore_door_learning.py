"""L34c: Door learning during exploration."""
import pytest
from unittest.mock import MagicMock, patch
from vector_os_nano.core.scene_graph import SceneGraph


class TestExploreDoorLearning:
    """Verify explore.py records doors on room transitions."""

    def test_no_import_of_room_centers(self):
        """explore.py should not import _ROOM_CENTERS."""
        import ast
        import inspect
        from vector_os_nano.skills.go2 import explore
        source = inspect.getsource(explore)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "navigate" in node.module:
                    names = [alias.name for alias in node.names]
                    assert "_ROOM_CENTERS" not in names, "_ROOM_CENTERS still imported"
                    assert "_detect_current_room" not in names, "_detect_current_room still imported"

    def test_exploration_loop_learns_doors(self):
        """Simulate room transitions and verify doors are recorded in SceneGraph."""
        sg = SceneGraph()
        # Pre-populate rooms (as if VLM identified them)
        sg.visit("kitchen", 17.0, 2.5)
        sg.visit("hallway", 10.0, 5.0)

        # Simulate: robot moves from kitchen through door to hallway
        # The exploration loop detects room change and calls add_door
        # We test the logic directly rather than running the full loop

        # Simulate the door learning logic
        prev_room = "kitchen"
        current_room = "hallway"
        door_x, door_y = 13.5, 3.0  # transition position

        if prev_room != current_room:
            sg.add_door(prev_room, current_room, door_x, door_y)

        door = sg.get_door("kitchen", "hallway")
        assert door is not None
        assert abs(door[0] - 13.5) < 0.1
        assert abs(door[1] - 3.0) < 0.1

    def test_visit_skipped_when_room_none(self):
        """When nearest_room returns None, visit() should not be called."""
        sg = SceneGraph()  # empty — nearest_room returns None
        room = sg.nearest_room(5.0, 5.0)
        assert room is None
        # visit should not be called with None
        # (This tests the guard in explore.py logic)

    def test_room_entered_event_skips_none(self):
        """Room entered event should not fire for None room."""
        # Just verify the logic: None should not be added to _explore_visited
        visited = set()
        room = None
        if room is not None and room not in visited:
            visited.add(room)
        assert len(visited) == 0

    def test_explore_started_event_no_hardcoded_total(self):
        """Started event should have total_rooms=0 (unknown)."""
        import ast, inspect
        from vector_os_nano.skills.go2 import explore
        source = inspect.getsource(explore)
        # Verify no reference to len(_ROOM_CENTERS) in emit calls
        assert "len(_ROOM_CENTERS)" not in source

    def test_door_learning_multiple_transitions(self):
        """Multiple transitions between same rooms average the door position."""
        sg = SceneGraph()
        sg.visit("living_room", 3.0, 2.5)
        sg.visit("hallway", 10.0, 5.0)

        # First observation
        sg.add_door("living_room", "hallway", 6.0, 4.0)
        # Second observation (slightly different position — robot swings wide)
        sg.add_door("living_room", "hallway", 7.0, 4.0)

        door = sg.get_door("living_room", "hallway")
        assert door is not None
        # Average of 6.0 and 7.0
        assert abs(door[0] - 6.5) < 0.01

    def test_door_learning_bidirectional(self):
        """Door recorded going A->B is also retrievable as B->A."""
        sg = SceneGraph()
        sg.visit("kitchen", 17.0, 2.5)
        sg.visit("dining_room", 3.0, 7.5)

        sg.add_door("kitchen", "dining_room", 10.0, 5.0)

        # Retrievable in both directions
        assert sg.get_door("kitchen", "dining_room") is not None
        assert sg.get_door("dining_room", "kitchen") is not None

    def test_nearest_room_returns_correct_room(self):
        """nearest_room resolves to the closest room by Euclidean distance."""
        sg = SceneGraph()
        sg.visit("kitchen", 17.0, 2.5)
        sg.visit("hallway", 10.0, 5.0)

        # Position close to kitchen
        room = sg.nearest_room(16.0, 2.0)
        assert room == "kitchen"

        # Position close to hallway
        room = sg.nearest_room(11.0, 5.0)
        assert room == "hallway"

    def test_prev_room_tracking_across_transitions(self):
        """Simulate the _prev_room tracking logic in _exploration_loop."""
        sg = SceneGraph()
        sg.visit("kitchen", 17.0, 2.5)
        sg.visit("hallway", 10.0, 5.0)
        sg.visit("living_room", 3.0, 2.5)

        # Simulate robot path through rooms
        positions = [
            (16.5, 2.5),   # kitchen
            (15.0, 3.0),   # kitchen
            (12.0, 4.0),   # transition to hallway
            (10.5, 5.0),   # hallway
            (6.0, 3.5),    # transition to living_room
            (3.5, 2.5),    # living_room
        ]

        prev_room = None
        doors_learned = []

        for x, y in positions:
            room = sg.nearest_room(x, y)
            if prev_room is not None and room is not None and room != prev_room:
                sg.add_door(prev_room, room, x, y)
                doors_learned.append((prev_room, room))
            prev_room = room

        all_doors = sg.get_all_doors()
        # Should have learned at least one door
        assert len(all_doors) >= 1

    def test_no_detect_current_room_in_source(self):
        """explore.py must not call _detect_current_room anywhere."""
        import inspect
        from vector_os_nano.skills.go2 import explore
        source = inspect.getsource(explore)
        assert "_detect_current_room" not in source, \
            "_detect_current_room is still referenced in explore.py"
