# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Integration tests for Ros2Runtime with REAL rclpy (no mocks).

Regression guard for live-REPL Bug 1: "Executor is already spinning".
Skipped entirely if rclpy is not available in the current environment.
"""
from __future__ import annotations

import time

import pytest

rclpy = pytest.importorskip("rclpy")
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import String  # noqa: E402

# ---------------------------------------------------------------------------
# All tests in this file require a live DDS context
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.ros2


# ---------------------------------------------------------------------------
# Isolation fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_runtime():
    """Reset Ros2Runtime singleton + shut down rclpy cleanly around each test.

    rclpy must be initialised *before* any Node() constructor is called.
    We call rclpy.init() here so the test body can construct Nodes freely;
    Ros2Runtime.add_node() will see rclpy.ok()==True and skip its own init.
    """
    from vector_os_nano.hardware.ros2 import runtime as rt_mod

    rt_mod._runtime = None

    # Ensure rclpy context is live before the test body constructs any Node.
    if not rclpy.ok():
        rclpy.init()

    yield

    # Teardown: shut down runtime if it still exists, then ensure rclpy is off.
    if rt_mod._runtime is not None:
        try:
            rt_mod._runtime.shutdown()
        except Exception:  # noqa: BLE001
            pass
        rt_mod._runtime = None
    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Test 1 — three nodes coexist without "Executor is already spinning"
# ---------------------------------------------------------------------------


def test_three_stub_nodes_spin_concurrently_no_already_spinning():
    """Add 3 nodes with pub/sub pairs; each subscriber must receive a message.

    Failure mode = rclpy raising 'Executor is already spinning' (live-REPL
    Bug 1 symptom) or one/more subscribers never firing within the deadline.
    """
    from vector_os_nano.hardware.ros2.runtime import get_ros2_runtime

    runtime = get_ros2_runtime()

    received: dict[str, list[str]] = {"a": [], "b": [], "c": []}

    def make_node(name: str) -> tuple[Node, object]:
        node = Node(f"coexist_{name}")
        pub = node.create_publisher(String, f"/coexist_{name}", 10)

        def _cb(msg: String, key: str = name) -> None:
            received[key].append(msg.data)

        node.create_subscription(String, f"/coexist_{name}", _cb, 10)
        return node, pub

    node_a, pub_a = make_node("a")
    node_b, pub_b = make_node("b")
    node_c, pub_c = make_node("c")

    # add_node must not raise; the shared executor absorbs all three.
    runtime.add_node(node_a)
    runtime.add_node(node_b)
    runtime.add_node(node_c)

    # Give the spin thread time to pick up the newly registered nodes.
    time.sleep(0.2)

    msg = String()
    msg.data = "hello"
    pub_a.publish(msg)
    pub_b.publish(msg)
    pub_c.publish(msg)

    # Wait up to 2 s for all three subscribers to fire.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if all(received[k] for k in "abc"):
            break
        time.sleep(0.05)

    assert received["a"], "Node A subscriber did not receive its message"
    assert received["b"], "Node B subscriber did not receive its message"
    assert received["c"], "Node C subscriber did not receive its message"

    # Clean up nodes before fixture teardown destroys the executor.
    for n in (node_a, node_b, node_c):
        runtime.remove_node(n)
        n.destroy_node()


# ---------------------------------------------------------------------------
# Test 2 — shutdown tears down cleanly
# ---------------------------------------------------------------------------


def test_runtime_shutdown_tears_down_cleanly():
    """shutdown() must stop the spin thread and release executor resources.

    Note: the _isolate_runtime fixture calls rclpy.init() before this test
    runs, so Ros2Runtime sees rclpy.ok()==True and sets _we_inited_rclpy=False.
    As a result, runtime.shutdown() correctly skips rclpy.shutdown() (it did
    not own the rclpy context).  We verify is_running instead, which is the
    primary contract.
    """
    from vector_os_nano.hardware.ros2.runtime import get_ros2_runtime

    runtime = get_ros2_runtime()

    n = Node("shutdown_test_node")
    runtime.add_node(n)
    assert runtime.is_running, "Runtime should be running after add_node"

    runtime.shutdown()

    assert not runtime.is_running, "Runtime should not be running after shutdown"

    # destroy_node after shutdown may raise; tolerate best-effort.
    try:
        n.destroy_node()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Test 3 — add_node after shutdown: re-init or clean error
# ---------------------------------------------------------------------------


def test_add_node_after_shutdown_raises_or_reinits():
    """After runtime.shutdown(), a new singleton must work without crashing.

    The _isolate_runtime fixture owns the rclpy context for the whole test,
    so rclpy.ok() remains True even after runtime.shutdown() (which correctly
    skips rclpy.shutdown() since it did not init rclpy).  Creating a fresh
    Ros2Runtime therefore falls into path (a): re-use the live context.

    This guards against any regression where the second runtime fails to pick
    up the existing rclpy context and crashes on add_node().
    """
    from vector_os_nano.hardware.ros2 import runtime as rt_mod
    from vector_os_nano.hardware.ros2.runtime import get_ros2_runtime

    # First lifecycle: add_node then shutdown (does not kill rclpy context).
    runtime = get_ros2_runtime()
    n1 = Node("first")
    runtime.add_node(n1)
    n1.destroy_node()
    runtime.shutdown()

    # Reset the singleton so get_ros2_runtime() creates a brand-new instance.
    rt_mod._runtime = None
    runtime2 = get_ros2_runtime()

    try:
        n2 = Node("second")
        runtime2.add_node(n2)
        # Path (a): re-use of live rclpy context succeeded.
        assert rclpy.ok(), "rclpy context must still be valid"
        assert runtime2.is_running, "Second runtime should be running"
        runtime2.remove_node(n2)
        n2.destroy_node()
        runtime2.shutdown()
    except RuntimeError as exc:
        # Path (b): clean error path — must reference ROS2 state.
        msg = str(exc).lower()
        assert "ros" in msg or "rclpy" in msg or "shutdown" in msg or "context" in msg, (
            f"RuntimeError does not mention ROS2 context/state: {exc}"
        )
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"Unexpected exception type after re-init attempt: {type(exc).__name__}: {exc}"
        )
