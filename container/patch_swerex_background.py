#!/usr/bin/env python3

import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path


PATCH_FILE = Path("/tmp/swerex-1.4.0-background-operations.patch")
EXPECTED_VERSION = "1.4.0"


def run_patch(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["patch", "--batch", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    version = importlib.metadata.version("swe-rex")
    if version != EXPECTED_VERSION:
        sys.exit(f"Expected swe-rex {EXPECTED_VERSION}, found {version}")

    spec = importlib.util.find_spec("swerex")
    if not spec or not spec.submodule_search_locations:
        sys.exit("Cannot locate installed swerex package")

    package_dir = Path(next(iter(spec.submodule_search_locations)))
    if not PATCH_FILE.is_file():
        sys.exit(f"Patch file not found: {PATCH_FILE}")

    common = ["-p2", "-d", str(package_dir), "-i", str(PATCH_FILE)]

    check = run_patch("--dry-run", "--forward", *common)
    if check.returncode == 0:
        print(check.stdout)
        apply = run_patch("--forward", *common)
        if apply.returncode:
            sys.exit(f"Patch application failed:\n{apply.stdout}")
        print(apply.stdout)
        print("SWE-ReX background operations patch applied.")
        return

    reverse = run_patch("--dry-run", "--reverse", *common)
    if reverse.returncode == 0:
        print("SWE-ReX background operations patch is already applied.")
        return

    sys.exit(
        "Patch cannot be applied cleanly; installation may be modified.\n"
        f"Forward check:\n{check.stdout}\n"
        f"Reverse check:\n{reverse.stdout}"
    )


if __name__ == "__main__":
    main()
