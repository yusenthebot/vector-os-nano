# Plan — v2.2 Loco Manipulation Readiness

**Classification**: Non-architectural (no new ROS2 interfaces, no cross-package
data flow changes, no new core nodes, no new external dependencies). CEO review
async.

**Reads**: `.sdd/spec.md` (approved 2026-04-19)
**Status**: draft

---

## 1. Reuse Scan Summary

What already exists that we must keep using:

| Existing | Location | Use |
|---|---|---|
| `Go2ROS2Proxy.connect()` `rclpy.spin(node)` thread | `hardware/sim/go2_ros2_proxy.py:150-154` | Replace with `Ros2Runtime.add_node` |
| `PiperROS2Proxy.connect()` + `PiperGripperROS2Proxy.connect()` spins | `hardware/sim/piper_ros2_proxy.py:185-188, ~480` | Same replacement |
| `rclpy.init()` / `rclpy.ok()` guard pattern | `go2_ros2_proxy.py:94-95` | Move into Ros2Runtime |
| `WorldModel.get_objects_by_label` (substring match) | `core/world_model.py:150-166` | Keep as fallback; normaliser runs in skill before calling this |
| `ObjectState` dataclass (frozen) | `core/world_model.py:34-46` | Read-only use by skills |
| `PickTopDownSkill._resolve_target` | `skills/pick_top_down.py:261-286` | Modify in place — inject normaliser + single-candidate fallback |
| `NavigateSkill._navigate_with_proxy` flow | `skills/navigate.py:448-499` | Copy pattern for mobile_pick's nav step |
| `Go2ROS2Proxy.navigate_to(x, y, timeout, on_progress)` | `hardware/sim/go2_ros2_proxy.py` | Exactly the nav call mobile_pick uses |
| `Go2ROS2Proxy.get_position()/.get_heading()` | same | Needed by compute_approach_pose |
| `PiperROS2Proxy.ik_top_down / move_joints` | `hardware/sim/piper_ros2_proxy.py` | Already implements ArmProtocol surface used by place_top_down |
| `PiperGripperROS2Proxy.open/close/is_holding` | same | Place uses `open()`; pick uses all three |
| `SimStartTool._populate_pickables_from_mjcf` | `vcli/tools/sim_tool.py:234-~280` | Unchanged |
| `SimStartTool._start_go2` registration block | `vcli/tools/sim_tool.py:382-389` | Extend: register 3 new skills when `with_arm=True` |
| Skill registration + alias pattern via `@skill(...)` | `core/skill.py` | Reuse unchanged |
| `GoalDecomposer` system prompt builder | `vcli/cognitive/goal_decomposer.py:296-400` | Hook point for source-aware prompt |
| pytest mocks from `tests/skills/test_pick_top_down.py` | tests/ | Copy patterns for new skill tests |
| `scripts/verify_pick_top_down.py` | scripts/ | Template for new `verify_loco_pick_place.py` |

What exists nearby but we are NOT reusing:

| Not reused | Reason |
|---|---|
| `skills/place.py` (the SO-101 place) | Uses 5-DoF, location-map-based API. New `place_top_down.py` targets Piper 6-DoF with explicit xyz. Different surface area. |
| `skills/pick.py` (SO-101 pick) | Same reason |
| VGG-driven multi-step decomposition for pick | Spec §11 Q5: direct skill wins for demo-quality determinism |

No ROS2 community package matches `Ros2Runtime` (the need is process-local
rclpy multiplexing; each app writes its own). No PyPI library wraps
MultiThreadedExecutor in a singleton. Writing it ourselves is ~40 lines.

---

## 2. Architecture Overview

```
 REPL                      VectorEngine                      Hardware (ROS2 subproc)
 ─────                     ────────────                      ─────────────────────────
                           ┌──────────┐
  "抓前面绿色"       ─────▶│  VGG +   │
                           │  tool_use│
                           └────┬─────┘
                                │ pick_top_down(...)
                                ▼
                           ┌─────────────────┐     /piper/joint_cmd         ┌──────────────┐
                           │ PickTopDownSkill│────────────────────────────▶ │              │
                           └─────────────────┘                               │              │
                                                                             │  go2_vnav_   │
                                                                             │  bridge.py   │
  "去拿蓝瓶"         ─────▶┌─────────────────┐                                │  (subproc)   │
                           │ MobilePickSkill │                                │              │
                           │   1 approach    │                                │  MuJoCoGo2   │
                           │   2 navigate    │──── /way_point ───────────────▶│   + Piper    │
                           │   3 wait_stable │                                │   (nq=27)    │
                           │   4 pick_top    │                                │              │
                           └─────────────────┘                                │              │
                                                                             │              │
  "放到 (x, y, z)"   ─────▶┌─────────────────┐     /piper/joint_cmd          │              │
                           │ PlaceTopDownSk. │────────────────────────────▶  │              │
                           └─────────────────┘                               │              │
                                                                             └──────────────┘
                                   ▲
                                   │ context.arm / gripper / base / world_model
                           ┌───────┴───────────┐
                           │ Ros2Runtime       │       single MultiThreadedExecutor
                           │  (process global) │──────▶ shared spin thread
                           └───────────────────┘
                                   ▲
                  ┌────────────────┼────────────────┐
                  │                │                │
         Go2ROS2Proxy     PiperROS2Proxy     PiperGripperROS2Proxy
         (base nodes)    (arm node)         (gripper node)
```

Request path stays identical to v2.1 — the only runtime change is that
three `rclpy.spin(node)` threads collapse into one
`MultiThreadedExecutor.spin()` thread owned by Ros2Runtime.

---

## 3. Module Designs

### 3.1 `hardware/ros2/runtime.py` (new, ~80 LoC)

**Responsibility**: process singleton that owns one rclpy MultiThreadedExecutor
and the spin thread. Idempotent init, thread-safe add/remove.

```python
# public surface
class Ros2Runtime:
    def add_node(self, node: rclpy.node.Node) -> None: ...
    def remove_node(self, node: rclpy.node.Node) -> None: ...
    def shutdown(self) -> None: ...
    @property
    def is_running(self) -> bool: ...

def get_ros2_runtime() -> Ros2Runtime: ...
```

**Internals**:
- Module-level `_runtime: Ros2Runtime | None = None` + `_lock = threading.Lock()`
- `get_ros2_runtime()` constructs singleton under lock on first call
- `add_node()`:
  - Lock, check rclpy init (`if not rclpy.ok(): rclpy.init()`)
  - If no executor yet: create `MultiThreadedExecutor(num_threads=4)`, start daemon
    thread `self._spin_thread = Thread(target=self._executor.spin, daemon=True)`
  - `self._executor.add_node(node)`, track in `self._nodes: set`
- `remove_node()`: `self._executor.remove_node(node)`, drop from set
- `shutdown()`: `executor.shutdown()`, `thread.join(timeout=2.0)`, `rclpy.shutdown()` (iff we inited)
  - Record `_we_inited_rclpy` on first init so we only shutdown rclpy that we own
- Registers `atexit.register(self.shutdown)` on first init

**Imports**: deferred (inside methods) so module import doesn't require rclpy
to be installed — matches existing pattern across the codebase.

**Failure modes**: `add_node` raises `RuntimeError("ROS2 runtime not available")`
if rclpy import fails.

### 3.2 `skills/utils/approach_pose.py` (new, ~60 LoC)

**Responsibility**: pure function mapping (object_xyz, dog_pose, clearance) →
(approach_x, approach_y, approach_yaw).

```python
def compute_approach_pose(
    object_xyz: tuple[float, float, float],
    dog_pose: tuple[float, float, float],         # x, y, yaw
    clearance: float = 0.55,
    approach_direction: str | None = None,
) -> tuple[float, float, float]:
```

**Algorithm** (approach_direction in {None, "from_dog"}):
1. `dx = dog_x - obj_x; dy = dog_y - obj_y`
2. `dist = sqrt(dx² + dy²)`; if `dist < 1e-6` → raise `ValueError` (caller must
   handle by re-positioning dog first).
3. `ux, uy = dx/dist, dy/dist`  (unit vector from object to dog)
4. `approach_x = obj_x + clearance * ux`, `approach_y = obj_y + clearance * uy`
5. `approach_yaw = atan2(obj_y - approach_y, obj_x - approach_x)` (dog faces object)
6. Return `(approach_x, approach_y, approach_yaw)`

`approach_direction="from_normal"` → `raise NotImplementedError("reserved for v2.3")`.

No side effects, no logging in hot path, easy to unit-test.

### 3.3 `skills/pick_top_down.py` — modify in place

Change 1: add module-level color map + normaliser.

```python
_CN_COLOR_MAP: dict[str, str] = {
    "红": "red",   "红色": "red",
    "绿": "green", "绿色": "green",
    "蓝": "blue",  "蓝色": "blue",
    "黄": "yellow","黄色": "yellow",
    "白": "white", "白色": "white",
    "黑": "black", "黑色": "black",
}

def _normalise_color_keyword(label: str) -> str | None:
    """Return normalised label with Chinese colors replaced by English, or
    None if no known Chinese color token was found."""
```

Algorithm: iterate `_CN_COLOR_MAP` keys; if present in `label`, replace that
token with the English word, return the new string. Return `None` if no
token matched.

Change 2: `_resolve_target` gains a third pass + single-candidate fallback.

```python
# After existing obj_id + object_label passes:
if label:
    normalised = _normalise_color_keyword(label)
    if normalised is not None:
        matches = wm.get_objects_by_label(normalised)
        if matches:
            obj = matches[0]
            return (obj.object_id, _xyz_of(obj))

# Last resort: if exactly one pickable exists, assume it.
all_pickables = [o for o in wm.get_objects() if o.object_id.startswith("pickable_")]
if len(all_pickables) == 1 and (label or obj_id):
    obj = all_pickables[0]
    logger.info("[PICK-TD] single-candidate fallback to %s", obj.object_id)
    return (obj.object_id, _xyz_of(obj))

return None
```

Diff discipline: keep existing branches unchanged; new code is additive, so
AC-1 (no regression on v2.1 tests) holds trivially.

### 3.4 `skills/place_top_down.py` (new, ~180 LoC)

Mirror `pick_top_down.py` structure closely. Key differences:

- Preconditions: `["gripper_holding_any"]`
- Postconditions + effects: release
- Parameters:
  - `target_xyz` (optional, explicit)
  - `receptacle_id` (optional, world_model lookup)
  - `drop_height` (default 0.05)
- Algorithm:
  1. Resolve target xyz (explicit OR receptacle_obj.xyz + drop_height)
  2. `pre_place = (x, y, z + _DEFAULT_PRE_PLACE_HEIGHT)`
  3. `place = (x, y, z)`
  4. IK → move → descend → open → lift (no return-home)
- Failure modes: `no_arm`, `no_gripper`, `ik_unreachable`, `move_failed`,
  `receptacle_not_found`, `missing_target` (neither xyz nor receptacle_id)
- Copy `_xyz_of` from pick_top_down OR import it (decision: copy for module
  independence; it's 2 lines)

Use the **exact same aliases** registration pattern as pick to get alias-
based routing: `@skill(aliases=["put", "drop", "放", "放下", "放到", ...])`.

### 3.5 `skills/mobile_pick.py` (new, ~150 LoC)

Composition skill. Does not duplicate pick logic; calls the instance.

```python
@skill(aliases=["去拿", "去抓", "拿来", "取来", "fetch", "go get"], direct=False)
class MobilePickSkill:
    name = "mobile_pick"

    def __init__(self):
        self._pick = PickTopDownSkill()  # single shared instance

    def execute(self, params, context) -> SkillResult:
        # 1 target resolve
        target = self._pick._resolve_target(params, context.world_model)
        if target is None:
            return SkillResult(success=False,
                               error_message="Cannot locate target object",
                               result_data={"diagnosis": "object_not_found"})
        obj_id, obj_xyz = target

        # 2 approach pose
        if context.base is None:
            return SkillResult(success=False,
                               error_message="No base connected",
                               result_data={"diagnosis": "no_base"})
        dog_x, dog_y, _ = context.base.get_position()
        dog_yaw = context.base.get_heading()
        clearance = cfg.get("clearance", 0.55)

        # 3 reachability fast-path
        from ..skills.utils.approach_pose import compute_approach_pose
        ax, ay, ayaw = compute_approach_pose(obj_xyz, (dog_x, dog_y, dog_yaw), clearance)

        already = (_dist_xy(dog_x, dog_y, ax, ay) < 0.10 and
                   abs(_ang_diff(dog_yaw, ayaw)) < math.radians(20))
        if not already and not params.get("skip_navigate"):
            ok = context.base.navigate_to(ax, ay, timeout=20.0)
            if not ok:
                return SkillResult(success=False,
                                   error_message="Navigation to approach pose failed",
                                   result_data={"diagnosis": "nav_failed",
                                                "approach": [ax, ay, ayaw]})

        # 4 wait stable (inline)
        if not _wait_stable(context.base, max_speed=0.05,
                            settle_duration=1.0, timeout=5.0):
            return SkillResult(success=False,
                               error_message="Dog did not settle before pick",
                               result_data={"diagnosis": "wait_stable_timeout"})

        # 5 delegate to pick_top_down
        pick_params = {**params, "object_id": obj_id}
        result = self._pick.execute(pick_params, context)
        if result.success:
            result.result_data["mobile_pick"] = {
                "approach": [round(ax, 2), round(ay, 2), round(ayaw, 2)],
                "nav_distance": round(_dist_xy(dog_x, dog_y, ax, ay), 2),
            }
        return result
```

Inline `_wait_stable(base, max_speed, settle_duration, timeout)`:
- Poll `base.get_position()` at ~5 Hz
- Track consecutive time where `dist_delta / dt < max_speed`
- If sustained for `settle_duration` → return True
- If `time > timeout` → return False
- No velocity API needed (we don't have `get_velocity` on Go2ROS2Proxy); use
  position deltas.

**Parameter metadata**: copy pick_top_down's `object_id` + `object_label`
entries verbatim, add `skip_navigate: {type: bool, default: false, source: static}`.

**Failure modes**: `no_base`, `no_world_model`, `object_not_found`,
`nav_failed`, `wait_stable_timeout` + pick_top_down's (via delegation).

### 3.6 `skills/mobile_place.py` (new, ~140 LoC)

Same structure as mobile_pick:

```python
@skill(aliases=["去放", "放到", "拿去放", "送到", "deliver"], direct=False)
class MobilePlaceSkill:
    name = "mobile_place"

    def __init__(self):
        self._place = PlaceTopDownSkill()

    def execute(self, params, context):
        # Resolve target xyz: explicit OR receptacle_id lookup
        # compute_approach_pose → navigate → wait_stable → place
```

**Failure modes**: `no_base`, `gripper_empty`, `receptacle_not_found`,
`nav_failed`, `wait_stable_timeout` + place_top_down's set.

### 3.7 `hardware/sim/go2_ros2_proxy.py` — modify

Replace spin thread block with runtime registration:

```python
# OLD
self._spin_thread = threading.Thread(
    target=lambda: rclpy.spin(self._node), daemon=True
)
self._spin_thread.start()

# NEW
if os.environ.get("VECTOR_SHARED_EXECUTOR", "1") == "1":
    from vector_os_nano.hardware.ros2.runtime import get_ros2_runtime
    get_ros2_runtime().add_node(self._node)
else:
    # Legacy per-proxy spin (rollback flag)
    self._spin_thread = threading.Thread(
        target=lambda: rclpy.spin(self._node), daemon=True
    )
    self._spin_thread.start()
```

`disconnect()` mirrored: `get_ros2_runtime().remove_node(self._node)` under flag.

The env flag is on by default so nothing else needs to change. Flag flip to
`0` restores pre-v2.2 behaviour for emergency rollback.

### 3.8 `hardware/sim/piper_ros2_proxy.py` — modify

Same replacement in `PiperROS2Proxy.connect()` (line ~185) and
`PiperGripperROS2Proxy.connect()` (line ~480). Two call sites, identical
patch.

### 3.9 `vcli/cognitive/goal_decomposer.py` — modify (small)

The decomposer feeds skill descriptions into its LLM prompt to let the LLM
construct GoalTrees. We need it to pass a new clue: "this skill does NOT
need a detect step — it reads from world_model".

Change plan:
- Where `_build_skill_catalog()` (or equivalent) stringifies each skill's
  description, append `" (source: world_model)"` when ALL required params
  have `source` starting with `world_model.` or `static`.
- Add a prompt line: "If a skill's description includes `(source: world_model)`,
  do NOT precede it with detect_* steps — the target is already known."

Minimal diff — no new data classes. The LLM's adherence is imperfect; the
single-candidate fallback in 3.3 is the robust back-stop. We accept imperfect
LLM adherence because the skill side fixes it anyway.

### 3.10 `vcli/tools/sim_tool.py` — modify

In `_start_go2` after `PickTopDownSkill` registration:

```python
if piper_arm is not None:
    from vector_os_nano.skills.pick_top_down import PickTopDownSkill
    from vector_os_nano.skills.place_top_down import PlaceTopDownSkill
    from vector_os_nano.skills.mobile_pick import MobilePickSkill
    from vector_os_nano.skills.mobile_place import MobilePlaceSkill
    agent._skill_registry.register(PickTopDownSkill())
    agent._skill_registry.register(PlaceTopDownSkill())
    agent._skill_registry.register(MobilePickSkill())
    agent._skill_registry.register(MobilePlaceSkill())
```

Everything else (world_model populate, Piper proxy setup) unchanged.

---

## 4. Data Flow (unchanged topology)

No new topics, services, or actions. Existing flow:

```
/piper/joint_cmd       main proc → bridge      Float64MultiArray (6)
/piper/gripper_cmd     main proc → bridge      Float64 (0..1 normalised)
/piper/joint_state     bridge → main proc      JointState (6 arm + 1 gripper) @ 20 Hz
/state_estimation      bridge → main proc      Odometry @ 20 Hz
/cmd_vel_nav           main proc → bridge      Twist (velocity cmds)
/way_point             main ↔ FAR              PointStamped (global waypoints)
/goal_point            main → FAR              PointStamped (nav goal)
```

mobile_pick uses `base.navigate_to()` which publishes `/goal_point` and waits
for `/way_point` from FAR (existing behaviour). No new messages.

---

## 5. Test Strategy

### 5.1 Unit tests (fast, no rclpy required)

Isolated with monkeypatches / mocks; no simulator, no ROS2 bring-up.

| File | Tests | Notes |
|---|---|---|
| `tests/hardware/ros2/test_runtime.py` | 5 | Mock `rclpy` via monkeypatch; verify init / add / remove / shutdown semantics; test singleton behaviour; test 3-node coexistence without "already spinning" — the key Bug 1 regression test |
| `tests/skills/utils/test_approach_pose.py` | 6 | Pure math; cover cardinal directions, yaw orientation, clearance distance, degenerate dog==object |
| `tests/skills/test_pick_top_down.py` (extend existing) | +5 | `_normalise_color_keyword` unit + 4 `_resolve_target` cases (CN color, EN label, single-candidate, missing) — existing 13 tests untouched |
| `tests/skills/test_place_top_down.py` | 8 | Mock arm + gripper + world_model; cover happy path, IK unreachable, move_failed, gripper empty (mock allows), explicit xyz vs receptacle, home-skip |
| `tests/skills/test_mobile_pick.py` | 8 | Mock base + arm + gripper; cover already-reachable fast-path, nav → wait → pick ordering, nav_failed, stable_timeout, pick failure propagation, skip_navigate flag |
| `tests/skills/test_mobile_place.py` | 7 | Same structure as mobile_pick |
| `tests/vcli/cognitive/test_decomposer_source.py` | 3 | Monkeypatch LLM backend; verify prompt substrings + post-processing behaviour |

**New test count**: ≈42

### 5.2 Integration tests (marked @pytest.mark.ros2, require rclpy)

| File | Tests | Notes |
|---|---|---|
| `tests/hardware/sim/test_ros2_proxies_coexist.py` | 3 | Build 3 stub nodes on same Ros2Runtime, verify all receive messages, no "already spinning" raised. Uses real `rclpy` but no MuJoCo |

### 5.3 E2E harness (subprocess per attempt)

`scripts/verify_loco_pick_place.py` — a subprocess-per-attempt driver with:
```
--repeat N
--mode [pick_only | pick_and_place]
--objects blue,green,red       # which pickables to cycle
```

Spawns a bridge subprocess with `VECTOR_SIM_WITH_ARM=1`, connects proxies in
main, runs the requested skill sequence, asserts result_data, teardown.
Template: `scripts/verify_pick_top_down.py` (exists).

### 5.4 Live REPL (human)

`docs/v2.2_live_repl_checklist.md` — 4-step manual checklist for Yusen to
execute in `vector-cli`.

### 5.5 Regression

All existing tests must still pass:
- `pytest tests/` (all of current 3200+ collected) — no new red
- Particularly `tests/skills/test_pick_top_down.py` (13), `tests/hardware/sim/test_mujoco_piper.py` (17), `tests/skills/test_navigate.py`

---

## 6. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Shared executor breaks existing nav stack timing | med | high | `VECTOR_SHARED_EXECUTOR=1` flag (default on); AC-3 integration test catches coexistence; rollback = set flag to 0 |
| R2 | MultiThreadedExecutor callback reentry on arm callbacks causes races in PiperROS2Proxy state cache | low | med | `_state_lock` already guards joint_state cache; add same lock to `_last_gripper_pos` setter in PiperGripperROS2Proxy |
| R3 | compute_approach_pose places dog inside table geometry (table is 40x50 cm at (11, 3), clearance 0.55 puts dog at ~(10.45, 3) which is inside room but possibly too close to table legs) | med | high | Unit test rejects `dist < 1e-6`; nav stack safety radius 0.35 blocks collision; if nav fails, mobile_pick returns nav_failed (no crash). Future: bisect clearance up if first nav fails |
| R4 | wait_stable 5 s timeout expires on slow GPU / bridge pause | low | med | Inline helper with config override `config.skills.mobile_pick.wait_stable_timeout`; E2E has 3 retries via subprocess |
| R5 | VGG LLM ignores new "(source: world_model)" prompt hint | med | low | Skill-side single-candidate fallback catches this; Bug 2 validated without needing LLM compliance |
| R6 | Race between `rclpy.init()` in Ros2Runtime and elsewhere | low | high | `Ros2Runtime._lock` guards init; `rclpy.ok()` checked before init |
| R7 | `atexit.register(shutdown)` hangs if executor has a stuck callback | low | med | `executor.shutdown()` has timeout; thread join with timeout=2.0 — if hang, process still exits (daemon thread) |
| R8 | MobilePickSkill's `_pick = PickTopDownSkill()` constructor has side effects (skill decorator registers) | low | low | `@skill(direct=False)` does NOT register — confirmed; decorator just adds metadata |
| R9 | Test flakiness from `time.sleep` in mobile_pick tests | med | low | All mocks skip real sleeps via `monkeypatch.setattr("time.sleep", lambda x: None)` pattern |

---

## 7. Wave Grouping (for Phase 3 task decomposition)

Three waves; within each wave tasks run in parallel.

### Wave 1 — Foundations (no dependencies)

- **T1.1** `Ros2Runtime` module + unit tests (Alpha)
- **T1.2** `compute_approach_pose` util + unit tests (Beta)
- **T1.3** `_normalise_color_keyword` helper + 5 new tests in
  test_pick_top_down.py (Gamma)

### Wave 2 — Integration into existing proxies + skills

Depends on Wave 1 (for 2.1, 2.4).

- **T2.1** Wire Ros2Runtime into Go2ROS2Proxy + PiperROS2Proxy +
  PiperGripperROS2Proxy (Alpha — knows proxies)
- **T2.2** `PlaceTopDownSkill` + 8 unit tests (Beta)
- **T2.3** VGG source metadata hook in goal_decomposer + 3 tests (Gamma)
- **T2.4** Update `_resolve_target` in pick_top_down to include normaliser
  + single-candidate fallback (Gamma — continues from T1.3)

### Wave 3 — Composition + E2E

Depends on Wave 2.

- **T3.1** `MobilePickSkill` + 8 unit tests (Alpha)
- **T3.2** `MobilePlaceSkill` + 7 unit tests (Beta)
- **T3.3** sim_tool registration wiring + 1 smoke test (Gamma)
- **T3.4** `scripts/verify_loco_pick_place.py` E2E harness (Gamma)
- **T3.5** `tests/hardware/sim/test_ros2_proxies_coexist.py` integration
  (Alpha)

### Wave 4 — QA + delivery

- Full test suite: `pytest tests/` green
- Coverage report on new modules ≥ 80%
- Code-review + security-review (parallel)
- Live REPL checklist doc
- Yusen live validation
- Commit + merge prep

---

## 8. File Manifest

### New (10)

```
vector_os_nano/hardware/ros2/__init__.py        # empty
vector_os_nano/hardware/ros2/runtime.py
vector_os_nano/skills/utils/__init__.py         # empty (if missing)
vector_os_nano/skills/utils/approach_pose.py
vector_os_nano/skills/place_top_down.py
vector_os_nano/skills/mobile_pick.py
vector_os_nano/skills/mobile_place.py
tests/hardware/ros2/__init__.py                  # empty
tests/hardware/ros2/test_runtime.py
tests/hardware/sim/test_ros2_proxies_coexist.py
tests/skills/utils/__init__.py                   # empty
tests/skills/utils/test_approach_pose.py
tests/skills/test_place_top_down.py
tests/skills/test_mobile_pick.py
tests/skills/test_mobile_place.py
tests/vcli/cognitive/test_decomposer_source.py
scripts/verify_loco_pick_place.py
docs/v2.2_live_repl_checklist.md
```

### Modified (5)

```
vector_os_nano/hardware/sim/go2_ros2_proxy.py        (connect+disconnect)
vector_os_nano/hardware/sim/piper_ros2_proxy.py      (2 classes, connect+disconnect)
vector_os_nano/skills/pick_top_down.py               (_resolve_target, +normaliser)
vector_os_nano/vcli/cognitive/goal_decomposer.py     (skill catalog + prompt)
vector_os_nano/vcli/tools/sim_tool.py                (register 3 new skills)
```

### Documentation (2)

```
progress.md                                          (v2.2 section)
agents/devlog/status.md                              (current state)
```

Not touched: `scripts/go2_vnav_bridge.py` (bridge stays), `mjcf/*` (scene
stays), world_model (API unchanged), all other skills.

---

## 9. Build / Test Commands

```bash
# Unit (fast, no rclpy)
.venv-nano/bin/python -m pytest \
    tests/hardware/ros2/test_runtime.py \
    tests/skills/utils/test_approach_pose.py \
    tests/skills/test_place_top_down.py \
    tests/skills/test_mobile_pick.py \
    tests/skills/test_mobile_place.py \
    tests/vcli/cognitive/test_decomposer_source.py \
    tests/skills/test_pick_top_down.py -v

# Integration (rclpy required)
.venv-nano/bin/python -m pytest \
    tests/hardware/sim/test_ros2_proxies_coexist.py -v -m ros2

# E2E (full sim)
.venv-nano/bin/python scripts/verify_loco_pick_place.py \
    --repeat 3 --mode pick_only
.venv-nano/bin/python scripts/verify_loco_pick_place.py \
    --repeat 3 --mode pick_and_place

# Coverage
.venv-nano/bin/python -m pytest tests/ \
    --cov=vector_os_nano.skills.mobile_pick \
    --cov=vector_os_nano.skills.mobile_place \
    --cov=vector_os_nano.skills.place_top_down \
    --cov=vector_os_nano.skills.utils.approach_pose \
    --cov=vector_os_nano.hardware.ros2.runtime \
    --cov-report=term-missing
```

---

## 10. Estimated Effort

| Wave | Dev time (parallel) | Risk buffer |
|---|---|---|
| Wave 1 | 2 hrs (3 devs) | +30 min |
| Wave 2 | 2.5 hrs | +45 min |
| Wave 3 | 3 hrs | +1 hr |
| Wave 4 | 1.5 hrs | +30 min |
| **Total** | **~9 hrs of work in parallel, ~16-18 hrs single-threaded** | |

With 3 parallel Sonnet agents (Alpha/Beta/Gamma), wall-clock ≈ 4 hrs.

---

## 11. Success Definition

Plan is successful when:
- All 10 spec acceptance criteria pass
- `progress.md` + `status.md` updated
- Single PR/merge prepared with v2.1 + v2.2 commits
- Yusen can demo "去桌子拿蓝瓶 → 送到另一桌" in one REPL session

Not in scope for this plan (future v2.3):
- perception-driven grasp
- real Piper hardware
- base-arm coordinated motion
- approach_direction="from_normal"
