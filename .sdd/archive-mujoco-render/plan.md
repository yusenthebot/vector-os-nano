# Plan: MuJoCo Rendering Enhancement

**Status**: APPROVED (non-architectural, via ultraplan 2026-04-11)

## Files to Modify

| File | Changes |
|------|---------|
| `vector_os_nano/hardware/sim/go2_room.xml` | Shadow quality, fill lights, wall texture, baseboards, door frames |
| `vector_os_nano/hardware/sim/mujoco_go2.py` | Camera 640x480, renderer flags, viewer tracking cam |
| `progress.md` | Update date, reflect rendering enhancement |

## Implementation

### Wave 1: Scene Visuals (go2_room.xml)

1. **Shadow + offscreen quality**: shadowsize 4096->8192, add `<global offwidth="1280" offheight="960"/>`
2. **Fill lights**: Add upward-facing bounce lights per large room (`castshadow="false"`), increase sun intensity
3. **Wall texture**: Add subtle plaster texture for walls (replaces flat rgba)
4. **Texture refinement**: Tighten wood floor rgb1/rgb2 gap, increase texrepeat for subtler grain
5. **Baseboards**: Thin geoms at floor-wall junctions (`contype="0" conaffinity="0"`, `dark_wood`)
6. **Door frames**: Two vertical + one horizontal geom per doorway

### Wave 2: Camera + Renderer (mujoco_go2.py)

7. **Resolution**: Default 320x240 -> 640x480 for `get_camera_frame`, `get_depth_frame`, `get_rgbd_frame`
8. **Renderer flags**: Enable `mjRND_SHADOW` and `mjRND_REFLECTION` on Renderer creation
9. **Viewer tracking**: Add `viewer_track` param, `_sync_viewer_camera()` in physics loop
10. **Viewer defaults**: distance 22->6, elevation -65->-35 for room scene

### Wave 3: Docs + Verification

11. **progress.md**: Update rendering section
12. **Test verification**: Run full test suite, visual check

## Test Strategy

- Existing unit tests must pass unchanged (camera resolution is parameterized)
- No new test files needed — changes are visual/config, verified by existing tests + manual GUI check
- Camera default change may require updating test assertions if any hardcode 320x240
