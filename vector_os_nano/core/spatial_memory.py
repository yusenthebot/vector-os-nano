"""Persistent spatial memory for Vector OS Nano SDK.

Tracks rooms visited, objects observed per room, and a bounded navigation
event log.  Provides structured context for the agent's spatial reasoning
and LLM prompts.

Persistence: YAML via PyYAML (yaml.safe_dump / yaml.safe_load).
Thread-safety: all mutations are guarded by a threading.Lock.
No ROS2 dependencies — pure Python stdlib + PyYAML.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PERSIST_PATH = "/tmp/vector_spatial_memory.yaml"
_MAX_EVENTS = 200

# ---------------------------------------------------------------------------
# Record types (frozen — immutable after construction)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocationRecord:
    """Snapshot of everything the robot knows about one named location.

    Attributes:
        name:         Room or custom location name.
        x:            Position X in the robot's world frame.
        y:            Position Y in the robot's world frame.
        visit_count:  Number of times ``visit()`` was called for this room.
        last_visited: Unix timestamp of the most recent visit.
        objects_seen: Tuple of object/feature names observed in this room.
        description:  Free-text VLM scene description (optional).
    """

    name: str
    x: float
    y: float
    visit_count: int = 0
    last_visited: float = 0.0
    objects_seen: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for yaml.safe_dump."""
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "visit_count": self.visit_count,
            "last_visited": self.last_visited,
            "objects_seen": list(self.objects_seen),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LocationRecord:
        """Deserialise from a plain dict loaded by yaml.safe_load."""
        return cls(
            name=str(d["name"]),
            x=float(d["x"]),
            y=float(d["y"]),
            visit_count=int(d.get("visit_count", 0)),
            last_visited=float(d.get("last_visited", 0.0)),
            objects_seen=tuple(str(o) for o in d.get("objects_seen", [])),
            description=str(d.get("description", "")),
        )


@dataclass(frozen=True)
class SpatialEvent:
    """Single entry in the spatial event log.

    Attributes:
        timestamp:  Unix timestamp when the event occurred.
        location:   Room or location name associated with the event.
        event_type: One of "visit", "observe", "navigate", or a custom string.
        details:    Human-readable detail string (optional).
    """

    timestamp: float
    location: str
    event_type: str
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for yaml.safe_dump."""
        return {
            "timestamp": self.timestamp,
            "location": self.location,
            "event_type": self.event_type,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpatialEvent:
        """Deserialise from a plain dict loaded by yaml.safe_load."""
        return cls(
            timestamp=float(d["timestamp"]),
            location=str(d["location"]),
            event_type=str(d["event_type"]),
            details=str(d.get("details", "")),
        )


# ---------------------------------------------------------------------------
# SpatialMemory
# ---------------------------------------------------------------------------


class SpatialMemory:
    """Persistent spatial memory for mobile robot.

    Tracks rooms visited, objects seen, and navigation history.
    Provides context for the agent's spatial reasoning.

    All public mutating methods are thread-safe.  Read-only accessors that
    return plain Python values (str, list) are also guarded so callers do
    not observe partial state.

    Args:
        persist_path: Path to a YAML file used for persistence.  Defaults to
                      ``/tmp/vector_spatial_memory.yaml``.  Pass ``None`` to
                      disable persistence entirely.
    """

    def __init__(self, persist_path: str | None = _DEFAULT_PERSIST_PATH) -> None:
        self._persist_path: str | None = persist_path
        self._locations: dict[str, LocationRecord] = {}
        self._events: list[SpatialEvent] = []
        self._lock: threading.Lock = threading.Lock()

        if self._persist_path is not None:
            self.load()

    # ------------------------------------------------------------------
    # Mutating methods
    # ------------------------------------------------------------------

    def visit(self, room: str, x: float, y: float) -> None:
        """Record visiting a room at position (x, y).

        Increments visit_count and updates last_visited timestamp.  Creates a
        new LocationRecord if this room has not been seen before.

        Args:
            room: Room name string (e.g. "kitchen").
            x:    Robot world-frame X position at time of visit.
            y:    Robot world-frame Y position at time of visit.
        """
        with self._lock:
            existing = self._locations.get(room)
            if existing is None:
                updated = LocationRecord(
                    name=room,
                    x=x,
                    y=y,
                    visit_count=1,
                    last_visited=time.time(),
                )
            else:
                updated = LocationRecord(
                    name=existing.name,
                    x=x,
                    y=y,
                    visit_count=existing.visit_count + 1,
                    last_visited=time.time(),
                    objects_seen=existing.objects_seen,
                    description=existing.description,
                )
            self._locations[room] = updated
            self._append_event(
                SpatialEvent(
                    timestamp=updated.last_visited,
                    location=room,
                    event_type="visit",
                    details=f"x={x:.2f}, y={y:.2f}",
                )
            )

    def observe(
        self, room: str, objects: list[str], description: str = ""
    ) -> None:
        """Record objects or features observed in a room.

        Merges new object names into the existing set for this room.  Creates
        a minimal LocationRecord (x=0, y=0) if the room is not yet known.

        Args:
            room:        Room name string.
            objects:     List of object/feature names observed.
            description: Optional VLM scene description to store.
        """
        with self._lock:
            existing = self._locations.get(room)
            if existing is None:
                existing = LocationRecord(name=room, x=0.0, y=0.0)

            merged_objects = _merge_objects(existing.objects_seen, objects)
            new_desc = description if description else existing.description

            self._locations[room] = LocationRecord(
                name=existing.name,
                x=existing.x,
                y=existing.y,
                visit_count=existing.visit_count,
                last_visited=existing.last_visited,
                objects_seen=merged_objects,
                description=new_desc,
            )
            detail_parts: list[str] = [f"objects={objects}"]
            if description:
                detail_parts.append(f"desc={description[:60]}")
            self._append_event(
                SpatialEvent(
                    timestamp=time.time(),
                    location=room,
                    event_type="observe",
                    details=", ".join(detail_parts),
                )
            )

    def remember_location(self, name: str, x: float, y: float) -> None:
        """Save a custom named location (user-defined bookmark).

        Does not increment visit_count.  If a record already exists, updates
        x/y while preserving all other fields.

        Args:
            name: Bookmark label (e.g. "charging_dock").
            x:    World-frame X coordinate.
            y:    World-frame Y coordinate.
        """
        with self._lock:
            existing = self._locations.get(name)
            if existing is None:
                self._locations[name] = LocationRecord(name=name, x=x, y=y)
            else:
                self._locations[name] = LocationRecord(
                    name=existing.name,
                    x=x,
                    y=y,
                    visit_count=existing.visit_count,
                    last_visited=existing.last_visited,
                    objects_seen=existing.objects_seen,
                    description=existing.description,
                )
            self._append_event(
                SpatialEvent(
                    timestamp=time.time(),
                    location=name,
                    event_type="navigate",
                    details=f"bookmark x={x:.2f}, y={y:.2f}",
                )
            )

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def get_location(self, name: str) -> LocationRecord | None:
        """Return the LocationRecord for a room by name, or None if unknown.

        Args:
            name: Room or bookmark name.

        Returns:
            LocationRecord if found, else None.
        """
        with self._lock:
            return self._locations.get(name)

    def get_all_locations(self) -> list[LocationRecord]:
        """Return all known LocationRecords as a list.

        Returns:
            List of LocationRecord, unordered.
        """
        with self._lock:
            return list(self._locations.values())

    def get_unvisited_rooms(self, all_rooms: list[str]) -> list[str]:
        """Return rooms from all_rooms that have not yet been visited.

        A room is considered visited when its visit_count > 0.

        Args:
            all_rooms: Reference list of all expected room names.

        Returns:
            Subset of all_rooms with visit_count == 0.
        """
        with self._lock:
            return [
                r
                for r in all_rooms
                if self._locations.get(r) is None
                or self._locations[r].visit_count == 0
            ]

    def get_visited_rooms(self) -> list[str]:
        """Return list of room names that have been visited at least once.

        Returns:
            Room name strings in insertion order.
        """
        with self._lock:
            return [
                name
                for name, rec in self._locations.items()
                if rec.visit_count > 0
            ]

    def get_room_summary(self) -> str:
        """Return human-readable summary of all spatial knowledge for LLM context.

        Format example::

            Rooms visited (3/8): kitchen (2 visits, saw: fridge, counter, island),
            living_room (1 visit, saw: sofa, TV), hallway (1 visit).
            Unvisited: dining_room, study, master_bedroom, guest_bedroom, bathroom.

        If no rooms are known, returns a brief "No locations known yet." string.

        Returns:
            Multi-line summary string.
        """
        with self._lock:
            if not self._locations:
                return "No locations known yet."

            visited: list[LocationRecord] = [
                rec for rec in self._locations.values() if rec.visit_count > 0
            ]
            unvisited_names: list[str] = [
                name
                for name, rec in self._locations.items()
                if rec.visit_count == 0
            ]

            visited_total = len(visited)
            total = len(self._locations)

            visited_parts: list[str] = []
            for rec in visited:
                visit_word = "visit" if rec.visit_count == 1 else "visits"
                if rec.objects_seen:
                    obj_str = ", ".join(rec.objects_seen)
                    visited_parts.append(
                        f"{rec.name} ({rec.visit_count} {visit_word}, saw: {obj_str})"
                    )
                else:
                    visited_parts.append(
                        f"{rec.name} ({rec.visit_count} {visit_word})"
                    )

            lines: list[str] = []
            if visited_parts:
                lines.append(
                    f"Rooms visited ({visited_total}/{total}): "
                    + ", ".join(visited_parts)
                    + "."
                )
            else:
                lines.append(f"Rooms visited (0/{total}).")

            if unvisited_names:
                lines.append("Unvisited: " + ", ".join(unvisited_names) + ".")

            return "\n".join(lines)

    def get_events(self, limit: int = 20) -> list[SpatialEvent]:
        """Return the most recent spatial events.

        Args:
            limit: Maximum number of events to return (default 20).

        Returns:
            List of SpatialEvent, most-recent last, bounded by limit.
        """
        with self._lock:
            return list(self._events[-limit:])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist locations and events to YAML file.

        No-op if persist_path was set to None.  Raises IOError / OSError on
        write failure — callers should handle if persistence is best-effort.
        """
        if self._persist_path is None:
            return

        with self._lock:
            payload: dict[str, Any] = {
                "locations": [rec.to_dict() for rec in self._locations.values()],
                "events": [ev.to_dict() for ev in self._events],
            }

        with open(self._persist_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, default_flow_style=False, allow_unicode=True)

    def load(self) -> None:
        """Load locations and events from YAML file.

        No-op if persist_path is None or file does not exist.  On parse
        errors the method logs a warning and returns without raising, leaving
        the in-memory state empty.
        """
        if self._persist_path is None:
            return

        try:
            with open(self._persist_path, "r", encoding="utf-8") as fh:
                payload = yaml.safe_load(fh)
        except FileNotFoundError:
            return
        except (OSError, yaml.YAMLError) as exc:
            import warnings

            warnings.warn(
                f"SpatialMemory: could not load {self._persist_path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        if not isinstance(payload, dict):
            return

        with self._lock:
            self._locations = {}
            for raw in payload.get("locations", []):
                try:
                    rec = LocationRecord.from_dict(raw)
                    self._locations[rec.name] = rec
                except (KeyError, ValueError, TypeError):
                    continue

            self._events = []
            for raw in payload.get("events", []):
                try:
                    self._events.append(SpatialEvent.from_dict(raw))
                except (KeyError, ValueError, TypeError):
                    continue
            # Enforce cap after load
            if len(self._events) > _MAX_EVENTS:
                self._events = self._events[-_MAX_EVENTS:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_event(self, event: SpatialEvent) -> None:
        """Append an event; trim to _MAX_EVENTS.  Caller must hold _lock."""
        self._events.append(event)
        if len(self._events) > _MAX_EVENTS:
            del self._events[: len(self._events) - _MAX_EVENTS]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _merge_objects(
    existing: tuple[str, ...], new_objects: list[str]
) -> tuple[str, ...]:
    """Return a tuple that is the union of existing and new_objects.

    Preserves insertion order: existing objects come first, then any new
    objects not already in the set.  Comparison is case-sensitive.

    Args:
        existing:    Current objects_seen tuple.
        new_objects: List of newly observed object names.

    Returns:
        New tuple with merged unique object names.
    """
    seen: set[str] = set(existing)
    additions: list[str] = [o for o in new_objects if o not in seen]
    return existing + tuple(additions)
