"""Post-install compatibility verification.

Run after installing torch + transformers to ensure ALL existing
packages still work correctly. Fails fast on any import error.

Usage:
    .venv-nano/bin/python3 tests/verify_compat.py
"""
import sys
import traceback

CRITICAL_IMPORTS = [
    # Core simulation
    ("mujoco", "MuJoCo physics engine"),
    ("numpy", "NumPy array operations"),
    ("scipy", "SciPy scientific computing"),

    # MPC / control
    ("casadi", "CasADi optimization (convex MPC)"),

    # Perception
    ("PIL", "Pillow image processing"),
    ("httpx", "HTTP client (VLM API)"),

    # ROS2 (may fail if not sourced — OK)
    # ("rclpy", "ROS2 Python client"),

    # CLI
    ("rich", "Rich terminal UI"),
    ("prompt_toolkit", "Interactive prompt"),
    ("yaml", "YAML config parser"),

    # New packages
    ("torch", "PyTorch deep learning"),
    ("transformers", "HuggingFace transformers"),

    # Project modules
    ("vector_os_nano.core.types", "Core types"),
    ("vector_os_nano.core.scene_graph", "SceneGraph"),
    ("vector_os_nano.core.skill", "Skill framework"),
    ("vector_os_nano.perception.vlm_go2", "VLM perception"),
    ("vector_os_nano.perception.depth_projection", "Depth projection"),
]

def main():
    print("=" * 60)
    print("  Compatibility Verification")
    print("=" * 60)

    failed = []
    passed = []

    for module, desc in CRITICAL_IMPORTS:
        try:
            __import__(module)
            ver = ""
            mod = sys.modules[module]
            for attr in ("__version__", "VERSION", "version"):
                if hasattr(mod, attr):
                    ver = f" ({getattr(mod, attr)})"
                    break
            print(f"  OK  {module}{ver} — {desc}")
            passed.append(module)
        except Exception as exc:
            print(f"  FAIL {module} — {desc}")
            print(f"       {exc}")
            failed.append((module, str(exc)))

    # Verify torch CUDA
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            gpu = torch.cuda.get_device_name(0)
            print(f"\n  GPU: {gpu}")
            print(f"  CUDA: {torch.version.cuda}")
        else:
            print(f"\n  GPU: Not available (CPU only)")
    except Exception:
        pass

    # Verify numpy compatibility (scipy warning check)
    try:
        import numpy as np
        import scipy
        print(f"\n  numpy {np.__version__} + scipy {scipy.__version__}: ", end="")
        # This triggers the version check
        from scipy import linalg
        linalg.inv(np.eye(3))
        print("OK")
    except Exception as exc:
        print(f"WARN: {exc}")

    # Verify MuJoCo still works
    try:
        import mujoco
        print(f"  mujoco {mujoco.__version__}: ", end="")
        # Quick sanity check
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><geom size='1'/></body></worldbody></mujoco>")
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        print("OK (physics step works)")
    except Exception as exc:
        print(f"FAIL: {exc}")
        failed.append(("mujoco_physics", str(exc)))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Passed: {len(passed)}/{len(CRITICAL_IMPORTS)}")
    if failed:
        print(f"  FAILED: {len(failed)}")
        for mod, err in failed:
            print(f"    - {mod}: {err}")
        sys.exit(1)
    else:
        print("  All compatible.")
        sys.exit(0)

if __name__ == "__main__":
    main()
