"""Spatial Memory — persistent map of visited locations, landmarks, and objects.

The agent builds a semantic map as it explores. Each visit records:
  - Where: coordinates + named location
  - What: objects/features observed
  - When: timestamp

This enables:
  - "What rooms have I visited?" → recall from memory
  - "Where did I see the cup?" → lookup object location
  - "Go back to where I found the book" → navigate to remembered coords
  - "What haven't I explored yet?" → frontier detection

No ROS2 dependency. Pure Python.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LocationMemory:
    """A remembered location with semantic labels and observations."""
    name: str                          # "kitchen", "hallway_north", "waypoint_3"
    x: float
    y: float
    category: str = "unknown"          # room, corridor, landmark, object_location
    visits: int = 0
    first_visited: float = 0.0         # timestamp
    last_visited: float = 0.0          # timestamp
    objects_seen: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "x": self.x, "y": self.y,
            "category": self.category, "visits": self.visits,
            "first_visited": self.first_visited,
            "last_visited": self.last_visited,
            "objects_seen": list(self.objects_seen),
            "notes": list(self.notes),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LocationMemory":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SpatialMemory:
    """Persistent spatial memory for an autonomous agent.

    Stores named locations, objects, and exploration state.
    Can be saved/loaded to JSON for persistence across sessions.
    """

    def __init__(self) -> None:
        self._locations: dict[str, LocationMemory] = {}
        self._object_locations: dict[str, list[tuple[float, float, str]]] = {}
        # object_name → [(x, y, location_name), ...]
        self._visit_history: list[dict] = []  # chronological visit log
        self._exploration_waypoints: list[tuple[float, float, str]] = []

    # ------------------------------------------------------------------
    # Location management
    # ------------------------------------------------------------------

    def add_location(
        self,
        name: str,
        x: float,
        y: float,
        category: str = "unknown",
        tags: list[str] | None = None,
    ) -> LocationMemory:
        """Register a named location. Updates if already exists."""
        if name in self._locations:
            loc = self._locations[name]
            loc.x = x
            loc.y = y
            if category != "unknown":
                loc.category = category
            if tags:
                loc.tags = list(set(loc.tags + tags))
            return loc

        loc = LocationMemory(
            name=name, x=x, y=y, category=category,
            tags=tags or [],
        )
        self._locations[name] = loc
        logger.info("SpatialMemory: added location '%s' at (%.1f, %.1f)", name, x, y)
        return loc

    def visit(self, name: str, x: float, y: float, objects_seen: list[str] | None = None) -> None:
        """Record a visit to a location."""
        now = time.time()
        if name not in self._locations:
            self.add_location(name, x, y)

        loc = self._locations[name]
        loc.visits += 1
        loc.last_visited = now
        if loc.first_visited == 0:
            loc.first_visited = now
        if objects_seen:
            for obj in objects_seen:
                if obj not in loc.objects_seen:
                    loc.objects_seen.append(obj)
                # Track object → location mapping
                if obj not in self._object_locations:
                    self._object_locations[obj] = []
                entry = (x, y, name)
                if entry not in self._object_locations[obj]:
                    self._object_locations[obj].append(entry)

        self._visit_history.append({
            "time": now, "location": name, "x": x, "y": y,
            "objects": objects_seen or [],
        })

    def get_location(self, name: str) -> LocationMemory | None:
        """Look up a location by name."""
        return self._locations.get(name)

    def get_all_locations(self) -> list[LocationMemory]:
        """Return all known locations."""
        return list(self._locations.values())

    def get_visited_locations(self) -> list[LocationMemory]:
        """Return only locations that have been visited."""
        return [loc for loc in self._locations.values() if loc.visits > 0]

    def get_unvisited_locations(self) -> list[LocationMemory]:
        """Return locations added but never visited."""
        return [loc for loc in self._locations.values() if loc.visits == 0]

    # ------------------------------------------------------------------
    # Object queries
    # ------------------------------------------------------------------

    def find_object(self, name: str) -> list[tuple[float, float, str]]:
        """Find where an object was last seen. Returns [(x, y, location_name), ...]."""
        return self._object_locations.get(name, [])

    def objects_at(self, location_name: str) -> list[str]:
        """List objects seen at a given location."""
        loc = self._locations.get(location_name)
        return list(loc.objects_seen) if loc else []

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def nearest_location(self, x: float, y: float) -> LocationMemory | None:
        """Find the nearest named location to (x, y)."""
        best = None
        best_dist = float("inf")
        for loc in self._locations.values():
            d = math.sqrt((loc.x - x) ** 2 + (loc.y - y) ** 2)
            if d < best_dist:
                best_dist = d
                best = loc
        return best

    def current_location_name(self, x: float, y: float, radius: float = 2.0) -> str | None:
        """Determine which named location the robot is currently in."""
        loc = self.nearest_location(x, y)
        if loc is None:
            return None
        d = math.sqrt((loc.x - x) ** 2 + (loc.y - y) ** 2)
        return loc.name if d < radius else None

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def set_exploration_waypoints(self, waypoints: list[tuple[float, float, str]]) -> None:
        """Set waypoints for systematic exploration. Each: (x, y, name)."""
        self._exploration_waypoints = list(waypoints)

    def get_next_exploration_target(self) -> tuple[float, float, str] | None:
        """Return the next unvisited exploration waypoint, or None if all visited."""
        for x, y, name in self._exploration_waypoints:
            if name not in self._locations or self._locations[name].visits == 0:
                return (x, y, name)
        return None

    # ------------------------------------------------------------------
    # Summary for LLM context
    # ------------------------------------------------------------------

    def summary_for_llm(self) -> str:
        """Generate a concise summary of spatial memory for the LLM planner."""
        parts = []
        visited = self.get_visited_locations()
        if visited:
            room_list = ", ".join(f"{l.name}({l.visits}x)" for l in visited)
            parts.append(f"Visited: {room_list}")

        unvisited = self.get_unvisited_locations()
        if unvisited:
            parts.append(f"Unvisited: {', '.join(l.name for l in unvisited)}")

        if self._object_locations:
            obj_list = ", ".join(
                f"{obj}@{locs[-1][2]}" for obj, locs in self._object_locations.items()
            )
            parts.append(f"Objects: {obj_list}")

        return " | ".join(parts) if parts else "No spatial memory yet."

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save spatial memory to JSON."""
        data = {
            "locations": {k: v.to_dict() for k, v in self._locations.items()},
            "object_locations": self._object_locations,
            "visit_history": self._visit_history[-100:],  # keep last 100
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Load spatial memory from JSON."""
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text())
        for k, v in data.get("locations", {}).items():
            self._locations[k] = LocationMemory.from_dict(v)
        self._object_locations = data.get("object_locations", {})
        self._visit_history = data.get("visit_history", [])
