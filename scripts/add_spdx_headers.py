#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Insert Apache-2.0 SPDX headers into Python source files.

Idempotent: skips files whose first 5 lines already contain
``SPDX-License-Identifier``. Preserves shebangs and PEP 263 encoding
declarations by inserting the header after them.

Run from the repo root:

    python3 scripts/add_spdx_headers.py

Use ``--check`` to fail (exit 1) if any file is missing a header — useful
in CI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HEADER_LINES = (
    "# SPDX-License-Identifier: Apache-2.0\n",
    "# Copyright (c) 2024-2026 Vector Robotics\n",
)
HEADER_TEXT = "".join(HEADER_LINES)

SCAN_ROOTS = ("vector_os_nano", "tests", "scripts")
EXCLUDE_DIR_PARTS = {"__pycache__", ".venv-nano", ".venv", ".sdd", ".git"}
EXCLUDE_FILES = {"add_spdx_headers.py"}  # this script already has its header


def has_header(text: str) -> bool:
    """Return True if SPDX-License-Identifier appears in the first 5 lines."""
    head = text.splitlines()[:5]
    return any("SPDX-License-Identifier" in line for line in head)


def insert_position(lines: list[str]) -> int:
    """Return line index where SPDX header should be inserted.

    Inserts after a shebang and/or PEP 263 encoding declaration so those
    keep their required first-/second-line position.
    """
    pos = 0
    if pos < len(lines) and lines[pos].startswith("#!"):
        pos += 1
    if pos < len(lines) and (
        "coding:" in lines[pos] or "coding=" in lines[pos]
    ) and lines[pos].lstrip().startswith("#"):
        pos += 1
    return pos


def process_file(path: Path, *, check: bool) -> tuple[bool, bool]:
    """Process one file. Returns (already_had_header, was_modified)."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return (False, False)

    if has_header(text):
        return (True, False)

    if check:
        return (False, False)

    lines = text.splitlines(keepends=True)
    pos = insert_position(lines)

    # Add a blank line after the header unless the next line is already
    # blank or starts a docstring on its own.
    suffix = ""
    if pos < len(lines) and lines[pos].strip() != "":
        suffix = "\n"

    new_lines = lines[:pos] + list(HEADER_LINES) + [suffix] if suffix else (
        lines[:pos] + list(HEADER_LINES)
    )
    new_lines += lines[pos:]
    path.write_text("".join(new_lines), encoding="utf-8")
    return (False, True)


def iter_python_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for src_root in SCAN_ROOTS:
        base = root / src_root
        if not base.is_dir():
            continue
        for p in base.rglob("*.py"):
            if any(part in EXCLUDE_DIR_PARTS for part in p.parts):
                continue
            if p.name in EXCLUDE_FILES:
                continue
            out.append(p)
    return sorted(out)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail (exit 1) if any file is missing the SPDX header.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    files = iter_python_files(repo_root)

    had = 0
    changed = 0
    missing: list[Path] = []
    for path in files:
        already, modified = process_file(path, check=args.check)
        if already:
            had += 1
        elif modified:
            changed += 1
        else:
            missing.append(path)

    if args.check:
        if missing:
            print(f"FAIL: {len(missing)} file(s) missing SPDX header:")
            for p in missing[:20]:
                print(f"  {p.relative_to(repo_root)}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
            return 1
        print(f"OK: {had} files have headers, 0 missing.")
        return 0

    print(
        f"Scanned {len(files)} files: {had} already had headers, "
        f"{changed} updated."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
