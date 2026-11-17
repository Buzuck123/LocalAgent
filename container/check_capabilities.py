#!/usr/bin/env python3
"""Verify the container capabilities declared in capabilities.json."""
import importlib.util
import json
import shutil
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
        ok = not (missing_commands or missing_imports or missing_paths)
        all_ok = all_ok and ok
        capabilities.append({
            "name": item.get("name", "unknown"),
            "available": ok,
            "missing_commands": missing_commands,
            "missing_python_imports": missing_imports,
            "missing_paths": missing_paths,
        })
    report = {
        "manifest_version": data.get("version"),
        "checked_at_startup": True,
        "all_available": all_ok,
        "capabilities": capabilities,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_available": all_ok, "capabilities_checked": len(capabilities), "report": str(REPORT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
