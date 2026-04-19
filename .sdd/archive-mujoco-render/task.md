# Tasks: MuJoCo Rendering Enhancement

## Wave 1: Scene Visuals (go2_room.xml)

### Task 1: Shadow + offscreen quality
- File: `go2_room.xml`
- Change `shadowsize="4096"` to `"8192"`
- Add `<global offwidth="1280" offheight="960"/>` inside `<visual>`
- Test: XML loads without error, existing tests pass

### Task 2: Fill lights
- File: `go2_room.xml`
- Add bounce fill lights (upward, `castshadow="false"`) for hall, living, dining, kitchen
- Increase sun diffuse from 0.25 to 0.35
- Test: scene loads, no light count overflow

### Task 3: Wall texture
- File: `go2_room.xml`
- Add `wall_tex` texture (builtin="flat", subtle rgb variation)
- Update `wall_mat` to use texture
- Test: walls render with texture, not flat color

### Task 4: Texture refinement
- File: `go2_room.xml`
- Tighten `wood_tex` rgb1/rgb2 for subtler grain
- Increase `wood_mat` texrepeat from 6x6 to 10x10
- Test: floor looks less checker-like

### Task 5: Baseboards
- File: `go2_room.xml`
- Add thin box geoms along exterior walls at floor level
- `contype="0" conaffinity="0"`, material `dark_wood`
- Height ~0.08m, depth ~0.01m
- Test: no collision interference, visually present

### Task 6: Door frames
- File: `go2_room.xml`
- Add frame geoms (2 vertical + 1 horizontal) at each doorway
- `contype="0" conaffinity="0"`, material `dark_wood`
- Frame width ~0.06m
- Test: frames don't block doorways (no collision), visually present

## Wave 2: Camera + Renderer (mujoco_go2.py)

### Task 7: Camera resolution
- File: `mujoco_go2.py`
- Change defaults: 320x240 -> 640x480 in `get_camera_frame`, `get_depth_frame`, `get_rgbd_frame`
- Test: `pytest tests/ -k camera` — check for hardcoded 320/240 assertions

### Task 8: Renderer flags
- File: `mujoco_go2.py`
- After creating `_cam_renderer`, enable `mjRND_SHADOW` and `mjRND_REFLECTION`
- Same for `_depth_renderer`
- Test: renderer creates without error

### Task 9: Viewer tracking camera
- File: `mujoco_go2.py`
- Add `viewer_track: bool = True` to `__init__`
- In physics loops (sinusoidal + mpc), update `viewer.cam.lookat` to robot XY position
- Change room scene defaults: distance=6, elevation=-35
- Test: existing tests pass (no GUI in tests)

## Wave 3: Docs

### Task 10: Update progress.md
- Remove any stale Gazebo TODOs
- Add MuJoCo rendering enhancement section
- Update date
