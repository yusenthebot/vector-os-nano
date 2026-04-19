# Task T4 — Wire Ros2Runtime into 3 ROS2 proxies

## Files Modified

### 1. `vector_os_nano/hardware/sim/go2_ros2_proxy.py`
- `__init__()` (line 80): added `self._shared_runtime_used: bool = False`
- `connect()` lines 151-154 (original spin block) replaced with 11-line shared/legacy branch (VECTOR_SHARED_EXECUTOR gate)
- `disconnect()` lines 168-176 replaced with 15-line version that calls `get_ros2_runtime().remove_node()` before `destroy_node()`

### 2. `vector_os_nano/hardware/sim/piper_ros2_proxy.py` — PiperROS2Proxy
- `__init__()` (line 120): added `self._shared_runtime_used: bool = False`
- `connect()` lines 185-188 (original spin block) replaced with 12-line shared/legacy branch
- `disconnect()` lines 204-212 replaced with 17-line version including `remove_node()` call

### 3. `vector_os_nano/hardware/sim/piper_ros2_proxy.py` — PiperGripperROS2Proxy
- `__init__()` (line 479): added `self._shared_runtime_used: bool = False`
- `connect()` lines 496-499 (original spin block) replaced with 13-line shared/legacy branch
- `disconnect()` lines 504-511 replaced with 17-line version including `remove_node()` call

## Test File Created

`tests/hardware/sim/test_ros2_proxies_runtime_wiring.py` — 5 tests, ~210 LoC

| Test | What it checks |
|---|---|
| `test_go2_proxy_connect_uses_shared_runtime_when_env_on` | env=1: add_node called with proxy._node |
| `test_piper_proxy_connect_uses_shared_runtime_when_env_on` | env=1: add_node called |
| `test_piper_gripper_proxy_connect_uses_shared_runtime_when_env_on` | env=1: add_node called |
| `test_go2_proxy_connect_uses_legacy_spin_when_env_zero` | env=0: add_node NOT called, Thread.start captured |
| `test_go2_proxy_disconnect_calls_remove_node` | env=1: remove_node called with correct node on disconnect |

## Test Output

```
5 passed (T4 wiring tests)
5 passed (Wave 1 runtime tests — no regression)
Total: 10/10
```

## Ruff

T4-introduced code: clean.

Pre-existing warnings (not in scope — in `open()`/`close()` methods, untouched):
- `vector_os_nano/hardware/sim/piper_ros2_proxy.py:544` E702 (semicolon)
- `vector_os_nano/hardware/sim/piper_ros2_proxy.py:553` E702 (semicolon)

## DONE
