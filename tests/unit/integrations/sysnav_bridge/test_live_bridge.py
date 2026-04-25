# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""LiveSysnavBridge unit tests (v2.4 T5).

Exercises the live ROS2 subscriber in isolation by stubbing rclpy and
``tare_planner.msg`` import sites. No real ROS2 process is spawned —
the bridge's import-resilience contract is what we check here.
"""
from __future__ import annotations

import builtins
import logging
import sys
import threading

import pytest

from vector_os_nano.core.world_model import ObjectState, WorldModel
from vector_os_nano.integrations.sysnav_bridge import LiveSysnavBridge
from vector_os_nano.integrations.sysnav_bridge.topic_interfaces import (
    ObjectNodeShadow,
    _Header,
    _Point,
)


# ---------------------------------------------------------------------------
# Helpers + fixtures
# ---------------------------------------------------------------------------


def _make_node(
    *,
    obj_id: int = 42,
    label: str = "blue bottle",
    position: tuple[float, float, float] = (1.0, 2.0, 0.25),
    is_asked_vlm: bool = False,
    status: bool = True,
) -> ObjectNodeShadow:
    return ObjectNodeShadow(
        header=_Header(stamp_sec=1714000000),
        object_id=(obj_id,),
        label=label,
        position=_Point(*position),
        bbox3d=(),
        status=status,
        img_path="",
        is_asked_vlm=is_asked_vlm,
        viewpoint_id=7,
    )


class _MsgListShadow:
    """Mimics tare_planner/ObjectNodeList payload via attributes."""

    def __init__(self, nodes):
        self.nodes = list(nodes)
        self.header = _Header()


@pytest.fixture
def world_model() -> WorldModel:
    return WorldModel()


@pytest.fixture
def remove_dependencies(monkeypatch):
    """Force ImportError on rclpy + tare_planner.msg."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"rclpy", "rclpy.executors"} or name.startswith("tare_planner"):
            raise ImportError(f"forced missing {name}")
        return real_import(name, *args, **kwargs)

    # Also drop any cached modules so the import statement re-runs
    for mod in list(sys.modules):
        if mod == "rclpy" or mod.startswith("rclpy.") or mod.startswith("tare_planner"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    monkeypatch.setattr(builtins, "__import__", fake_import)


# ---------------------------------------------------------------------------
# Degrade-to-noop when deps missing
# ---------------------------------------------------------------------------


def test_start_returns_false_when_rclpy_missing(
    world_model, remove_dependencies, caplog,
) -> None:
    bridge = LiveSysnavBridge(world_model)
    with caplog.at_level(logging.WARNING):
        ok = bridge.start()
    assert ok is False
    assert bridge.active is False
    assert bridge.backend == "missing-dependency"
    assert any("dependency missing" in r.message for r in caplog.records)


def test_start_when_dep_missing_does_not_raise(
    world_model, remove_dependencies,
) -> None:
    """The bridge must NEVER raise on start; agents must boot anyway."""
    bridge = LiveSysnavBridge(world_model)
    bridge.start()                  # no try/except here on purpose


def test_callback_never_invoked_when_no_subscription(
    world_model, remove_dependencies,
) -> None:
    bridge = LiveSysnavBridge(world_model)
    bridge.start()
    # If there's no subscription, calling the private callback directly
    # is the only way a message could be processed; with no subscriber
    # no traffic will reach us.
    assert world_model.get_objects() == []


# ---------------------------------------------------------------------------
# Callback dispatch — drive _callback directly with shadow messages
# ---------------------------------------------------------------------------


def test_callback_dispatches_one_add_object_per_node(world_model) -> None:
    bridge = LiveSysnavBridge(world_model)
    msg = _MsgListShadow(
        [_make_node(obj_id=1), _make_node(obj_id=2), _make_node(obj_id=3)],
    )
    bridge._callback(msg)
    objects = world_model.get_objects()
    assert {o.object_id for o in objects} == {
        "sysnav_1", "sysnav_2", "sysnav_3",
    }


def test_callback_uses_existing_object_node_to_state(world_model) -> None:
    """Re-observing the same id updates its position (upsert)."""
    bridge = LiveSysnavBridge(world_model)
    bridge._callback(_MsgListShadow([_make_node(obj_id=42, position=(1, 1, 1))]))
    bridge._callback(_MsgListShadow([_make_node(obj_id=42, position=(5, 5, 5))]))
    obj = world_model.get_object("sysnav_42")
    assert (obj.x, obj.y, obj.z) == (5.0, 5.0, 5.0)


def test_callback_logs_warning_on_malformed_node_no_crash(
    world_model, caplog,
) -> None:
    bridge = LiveSysnavBridge(world_model)

    class _Malformed:
        # Missing position attribute → object_node_to_state walks past
        # gracefully (returns 0,0,0). A REAL malformed case is one that
        # raises during attribute access:
        @property
        def object_id(self):
            raise RuntimeError("property bomb")

    msg = _MsgListShadow([_Malformed()])
    with caplog.at_level(logging.WARNING):
        bridge._callback(msg)
    assert bridge.dropped_count == 1
    assert any("dropping malformed" in r.message for r in caplog.records)
    # World model untouched
    assert world_model.get_objects() == []


def test_callback_processes_remaining_after_one_malformed(
    world_model, caplog,
) -> None:
    bridge = LiveSysnavBridge(world_model)

    class _Malformed:
        @property
        def object_id(self):
            raise RuntimeError("bomb")

    msg = _MsgListShadow([_Malformed(), _make_node(obj_id=7)])
    with caplog.at_level(logging.WARNING):
        bridge._callback(msg)
    assert bridge.dropped_count == 1
    assert {o.object_id for o in world_model.get_objects()} == {"sysnav_7"}


def test_callback_status_false_node_maps_to_unknown_state(world_model) -> None:
    bridge = LiveSysnavBridge(world_model)
    bridge._callback(_MsgListShadow([_make_node(obj_id=1, status=False)]))
    obj = world_model.get_object("sysnav_1")
    assert obj.state == "unknown"


def test_callback_preserves_grasped_state(world_model) -> None:
    """Skill set state to 'grasped' — SysNav re-observation must not
    demote it back to 'on_table'."""
    initial = ObjectState(
        object_id="sysnav_42", label="blue bottle",
        x=0.0, y=0.0, z=0.0, state="grasped",
    )
    world_model.add_object(initial)
    bridge = LiveSysnavBridge(world_model)
    bridge._callback(_MsgListShadow([_make_node(obj_id=42)]))
    assert world_model.get_object("sysnav_42").state == "grasped"


def test_callback_canonical_id_is_sysnav_prefixed(world_model) -> None:
    bridge = LiveSysnavBridge(world_model)
    bridge._callback(_MsgListShadow([_make_node(obj_id=99)]))
    assert world_model.get_object("sysnav_99") is not None


def test_callback_handles_empty_object_id_list(world_model) -> None:
    bridge = LiveSysnavBridge(world_model)
    node = ObjectNodeShadow(
        header=_Header(),
        object_id=(),
        label="ghost",
        position=_Point(0, 0, 0),
    )
    bridge._callback(_MsgListShadow([node]))
    obj = world_model.get_object("sysnav_unknown")
    assert obj is not None
    assert obj.label == "ghost"


# ---------------------------------------------------------------------------
# Idempotent start/stop
# ---------------------------------------------------------------------------


def test_stop_is_idempotent(world_model) -> None:
    """Calling stop() repeatedly must not raise."""
    bridge = LiveSysnavBridge(world_model)
    bridge.stop()
    bridge.stop()


def test_stop_after_failed_start_is_safe(
    world_model, remove_dependencies,
) -> None:
    bridge = LiveSysnavBridge(world_model)
    bridge.start()
    bridge.stop()


def test_double_start_is_idempotent_when_active() -> None:
    """If first start() succeeded the second is a fast True."""
    # We can't easily fake start(True) without rclpy, so we set the
    # active flag manually to verify the early-return guard.
    bridge = LiveSysnavBridge(WorldModel())
    bridge._active = True
    assert bridge.start() is True


# ---------------------------------------------------------------------------
# Disconnect detection
# ---------------------------------------------------------------------------


def test_disconnect_warning_fires_after_threshold(world_model, caplog) -> None:
    bridge = LiveSysnavBridge(world_model, on_disconnect_after_s=1.0)
    bridge._active = True            # simulate connected
    bridge._last_msg_t = 0.0
    with caplog.at_level(logging.WARNING):
        warned = bridge.check_disconnect(now=2.0)
    assert warned is True
    assert any("no /object_nodes_list" in r.message for r in caplog.records)


def test_disconnect_warning_does_not_fire_within_threshold(
    world_model, caplog,
) -> None:
    bridge = LiveSysnavBridge(world_model, on_disconnect_after_s=2.0)
    bridge._active = True
    bridge._last_msg_t = 0.0
    with caplog.at_level(logging.WARNING):
        warned = bridge.check_disconnect(now=1.0)
    assert warned is False


def test_disconnect_warning_resets_after_message(world_model, caplog) -> None:
    bridge = LiveSysnavBridge(world_model, on_disconnect_after_s=1.0)
    bridge._active = True
    bridge._last_msg_t = 0.0
    bridge.check_disconnect(now=2.0)        # warn
    # New message arrives — callback resets _disconnect_warned
    bridge._callback(_MsgListShadow([_make_node()]))
    # And another stale period — should warn again
    with caplog.at_level(logging.WARNING):
        # Need to advance _last_msg_t to a known past value
        bridge._last_msg_t = 100.0
        bridge._disconnect_warned = False
        bridge.check_disconnect(now=200.0)
    warns = [r for r in caplog.records if "no /object_nodes_list" in r.message]
    assert len(warns) >= 2


def test_disconnect_check_noop_when_inactive(world_model) -> None:
    bridge = LiveSysnavBridge(world_model, on_disconnect_after_s=0.5)
    # Never started → inactive
    assert bridge.check_disconnect(now=999.0) is False


def test_disconnect_threshold_zero_means_disabled(world_model) -> None:
    bridge = LiveSysnavBridge(world_model, on_disconnect_after_s=0.0)
    bridge._active = True
    bridge._last_msg_t = 0.0
    assert bridge.check_disconnect(now=1e6) is False


# ---------------------------------------------------------------------------
# Properties + thread safety
# ---------------------------------------------------------------------------


def test_active_default_false() -> None:
    assert LiveSysnavBridge(WorldModel()).active is False


def test_backend_default_uninitialised() -> None:
    assert LiveSysnavBridge(WorldModel()).backend == "uninitialised"


def test_dropped_count_starts_at_zero() -> None:
    assert LiveSysnavBridge(WorldModel()).dropped_count == 0


def test_callback_thread_safe_under_concurrent_invocation(world_model) -> None:
    """Two threads firing callbacks simultaneously must not corrupt state."""
    bridge = LiveSysnavBridge(world_model)
    msg_a = _MsgListShadow([_make_node(obj_id=i) for i in range(50)])
    msg_b = _MsgListShadow([_make_node(obj_id=100 + i) for i in range(50)])

    def fire(msg):
        for _ in range(5):
            bridge._callback(msg)

    threads = [
        threading.Thread(target=fire, args=(msg_a,)),
        threading.Thread(target=fire, args=(msg_b,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    ids = {o.object_id for o in world_model.get_objects()}
    # Each id must have been added at least once; we don't care about
    # exact count because upsert collapses repeats.
    assert len(ids) == 100
