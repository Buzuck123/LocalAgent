#!/usr/bin/env python3
"""Verify the container capabilities declared in capabilities.json."""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

MANIFEST = Path("/usr/local/share/localagent/capabilities.json")
REPORT = Path("/etc/localagent-capabilities.json")


def import_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def main():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    capabilities = []
    all_ok = True
    for item in data.get("capabilities", []):
        checks = item.get("checks", {})
        missing_commands = [name for name in checks.get("commands", []) if shutil.which(name) is None]
        missing_imports = [name for name in checks.get("python_imports", []) if not import_available(name)]
        missing_paths = [name for name in checks.get("paths", []) if not Path(name).exists()]
        browser_name = checks.get("playwright_browser")
        if browser_name:
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as playwright:
                    executable = getattr(playwright, str(browser_name)).executable_path
                if not Path(executable).is_file():
                    missing_paths.append(executable)
            except Exception:
                missing_paths.append(f"playwright:{browser_name}")
        ok = not (missing_commands or missing_imports or missing_paths)
        all_ok = all_ok and ok
        capabilities.append({
            "name": item.get("name", "unknown"),
            "available": ok,
            "missing_commands": missing_commands,
            "missing_python_imports": missing_imports,
            "missing_paths": missing_paths,
        })
    # Keep a machine-readable inventory for diagnostics and automatic drift
    # inspection. This full package list is not injected into the model prompt.
    system_inventory = []
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f=${binary:Package}\\t${Version}\\n"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        if result.returncode == 0:
            system_inventory = [
                {"name": line.split("\t", 1)[0], "version": line.split("\t", 1)[1]}
                for line in result.stdout.splitlines() if "\t" in line
            ]
    except (OSError, subprocess.TimeoutExpired):
        pass
    python_inventory = []
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode == 0:
            python_inventory = json.loads(result.stdout or "[]")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass

    report = {
        "manifest_version": data.get("version"),
        "checked_at_startup": True,
        "all_available": all_ok,
        "capabilities": capabilities,
        "installed_packages": {
            "system": system_inventory,
            "python": python_inventory,
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_available": all_ok, "capabilities_checked": len(capabilities), "report": str(REPORT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
