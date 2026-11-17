"""Compact model-facing capability catalog backed by the container manifest."""
from __future__ import annotations

import json
from pathlib import Path

_MANIFEST = Path(__file__).resolve().parent.parent / "container" / "capabilities.json"


def capability_catalog_prompt() -> str:
    try:
        data = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return (
            "COMPUTER CAPABILITIES\n"
            "- Use the persistent Linux terminal for shell commands, Python, files and network work.\n"
            "- Web search: use web-search \"query\"; browser automation: Playwright with Chromium."
        )
    lines = [
        "COMPUTER CAPABILITIES",
        "- terminal is a persistent Linux Bash environment. Run ordinary shell commands, Python scripts, file operations and programs there.",
        "- Web search: use web-search \"query\" (add --json or --limit when useful).",
        "- Browser: use Playwright with Chromium for JavaScript-rendered pages, DOM inspection, clicks, forms and downloads.",
        "- HTTP/API: use requests, httpx or aiohttp for ordinary requests.",
    ]
    for item in data.get("capabilities", []):
        name = str(item.get("name") or "").strip()
        example = str(item.get("example") or "").strip()
        if name and example and name not in {"Веб-поиск", "Браузер", "HTTP и API"}:
            lines.append(f"- {name}: {example}")
    lines.append(
        "- Capabilities are checked when the container starts. If a listed capability is unavailable, inspect /etc/localagent-capabilities.json; do not assume a failed check is usable."
    )
    return "\n".join(lines)
