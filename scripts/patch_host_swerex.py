#!/usr/bin/env python3
"""Apply LocalAgent's SWE-ReX Background Operations patch to the host venv.

Run with the Python interpreter used by LocalAgent:
    .venv/bin/python scripts/patch_host_swerex.py
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = PROJECT_ROOT / "container" / "swerex-1.4.0-background-operations.patch"
EXPECTED_VERSION = "1.4.0"


def _run_patch(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["patch", "--batch", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    try:
        version = importlib.metadata.version("swe-rex")
    except importlib.metadata.PackageNotFoundError:
        print("SWE-ReX is not installed in this Python environment.", file=sys.stderr)
        return 2
    if version != EXPECTED_VERSION:
        print(
            f"Expected swe-rex {EXPECTED_VERSION}, found {version}. "
            "Do not apply this patch to another version.",
            file=sys.stderr,
        )
        return 2

    spec = importlib.util.find_spec("swerex")
    if not spec or not spec.submodule_search_locations:
        print("Cannot locate the installed swerex package.", file=sys.stderr)
        return 2
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    abstract_file = package_dir / "runtime" / "abstract.py"
    if not abstract_file.is_file() or not PATCH_FILE.is_file():
        print("SWE-ReX source or LocalAgent patch file is missing.", file=sys.stderr)
        return 2

    common = ["-p2", "-d", str(package_dir), "-i", str(PATCH_FILE)]
    check = _run_patch("--dry-run", "--forward", *common)
    if check.returncode == 0:
        apply = _run_patch("--forward", *common)
        if apply.returncode:
            print(f"Patch application failed; no successful completion reported:\n{apply.stdout}", file=sys.stderr)
            return 1
        print(apply.stdout)
        print(f"SWE-ReX {version}: Background Operations patch applied to {package_dir}.")
        return 0

    reverse = _run_patch("--dry-run", "--reverse", *common)
    if reverse.returncode == 0:
        print(f"SWE-ReX {version}: Background Operations patch is already applied.")
        return 0

    print(
        "The SWE-ReX patch cannot be applied cleanly. The installed package may differ "
        "from the expected unpatched 1.4.0 source. No files were changed by the dry run.\n"
        f"Forward check:\n{check.stdout}\nReverse check:\n{reverse.stdout}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
