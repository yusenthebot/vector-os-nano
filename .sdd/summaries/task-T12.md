# Task T12 — E2E verify_loco_pick_place.py

## File

- `scripts/verify_loco_pick_place.py` — 513 LoC

## Dry-run output

```
[dry-run] verify_loco_pick_place — import validation
  mode:    pick_only
  objects: blue bottle, green bottle, red can
  repeat:  3

  [OK] MobilePickSkill from vector_os_nano.skills.mobile_pick
  [OK] MobilePlaceSkill from vector_os_nano.skills.mobile_place
  [OK] Go2ROS2Proxy from vector_os_nano.hardware.sim.go2_ros2_proxy
  [OK] PiperROS2Proxy from vector_os_nano.hardware.sim.piper_ros2_proxy
  [OK] PiperGripperROS2Proxy from vector_os_nano.hardware.sim.piper_ros2_proxy
  [OK] MobilePickSkill() instantiated (name='mobile_pick')
  [OK] MobilePlaceSkill() instantiated (name='mobile_place')

[dry-run] PASS — all imports OK
Exit: 0
```

## Help output

```
usage: verify_loco_pick_place.py [-h] [--repeat N]
                                 [--mode {pick_only,pick_and_place}]
                                 [--objects LIST] [--dry-run]

options:
  -h, --help            show this help message and exit
  --repeat N            Number of attempts per object (default: 3)
  --mode {pick_only,pick_and_place}
                        What flow to run (default: pick_only)
  --objects LIST        Comma-separated labels to cycle through
  --dry-run             Exit 0 without spawning subprocess; CI smoke
```

## AST / ruff

- AST: clean
- ruff: All checks passed (1 F541 fixed inline)

## Deviations from verify_pick_top_down.py pattern

| Pattern | verify_pick_top_down.py | verify_loco_pick_place.py |
|---------|------------------------|--------------------------|
| Subprocess approach | Inline Python string sent via `-c` (no ROS2) | `launch_explore.sh` bash subprocess (full ROS2 stack) |
| Hardware path | Direct in-process MuJoCo (MuJoCoGo2 + MuJoCoPiper) | ROS2 proxies (Go2ROS2Proxy + PiperROS2Proxy) |
| Readiness check | Static `time.sleep(0.8)` | Poll `/state_estimation` + `/piper/joint_state` topics (up to 30s) |
| Skill under test | PickTopDownSkill | MobilePickSkill (+ MobilePlaceSkill in pick_and_place mode) |
| subprocess.run vs Popen | `subprocess.run(..., capture_output=True)` | `subprocess.Popen` with `preexec_fn=os.setsid` + `killpg` |
| Log capture | `capture_output=True` (in-memory) | File redirect to `/tmp/verify_loco_pick_place_<PID>_<seq>.log` |
| LoC | 183 | 513 (more complex: ROS2 bridge + proxy lifecycle + place mode) |

The larger LoC is inherent: spawning a full nav+bridge stack over ROS2 requires
bridge readiness polling, proxy connect/disconnect lifecycle, and a two-skill
flow (pick + place), none of which exist in the simpler direct-MuJoCo harness.

## Verdict

DONE
