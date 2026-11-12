"""Smoke checks for the active LocalAgent tool catalog."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from . import db
from .config import settings
from .tools import TOOL_DEFINITIONS, dispatch_tool


def _tool_names() -> list[str]:
    return [item.get("function", {}).get("name", "") for item in TOOL_DEFINITIONS
            if item.get("function", {}).get("name")]


def run_tool_self_test() -> dict:
    """Check the terminal lifecycle and retained web/internal capabilities."""
    from .sandbox_manager import sandbox_manager
    run_id = uuid.uuid4().hex[:10]
    conversation = db.create_conversation(f"LocalAgent self-test {run_id}")
    cid = conversation["id"]
    results: dict[str, dict[str, Any]] = {}
    source = Path(settings.database_path).parent / "attachments" / f".selftest-{run_id}.txt"
    published_file = Path(settings.workspace_path) / "downloads" / f"selftest-{run_id}.txt"

    def record(name: str, status: str, detail: str, result=None):
        results[name] = {"status": status, "detail": detail}
        if isinstance(result, dict):
            results[name]["result"] = result

    try:
        terminal = dispatch_tool("terminal", {"command": "printf 'localagent-terminal-ok\n'; pwd"}, conversation_id=cid)
        if terminal.get("exit_code") == 0 and "localagent-terminal-ok" in terminal.get("stdout", ""):
            record("terminal", "passed", "Persistent Bash session returned stdout and exit_code=0", terminal)
        else:
            record("terminal", "failed", "Terminal smoke command failed", terminal)

        state = dispatch_tool("terminal", {"command": "export LOCALAGENT_SELFTEST_VALUE=kept; cd /tmp; printf ready"}, conversation_id=cid)
        verify = dispatch_tool("terminal", {"command": 'test "$LOCALAGENT_SELFTEST_VALUE" = kept && test "$PWD" = /tmp'}, conversation_id=cid)
        if state.get("exit_code") == 0 and verify.get("exit_code") == 0:
            record("terminal_persistence", "passed", "Shell variable and current directory survived across calls", verify)
        else:
            record("terminal_persistence", "failed", "Shell state did not persist", verify)

        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("publish-smoke-ok", encoding="utf-8")
        # Upload a host file to the container, then publish a container file back.
        uploaded = sandbox_manager.upload_file(cid, source, source.name)
        copied = dispatch_tool("terminal", {"command": f"cp -- {json.dumps(uploaded)} /tmp/publish-smoke.txt"}, conversation_id=cid)
        published = dispatch_tool("publish_file", {"source_path": "/tmp/publish-smoke.txt",
            "destination_path": f"selftest-{run_id}.txt"}, conversation_id=cid)
        if copied.get("exit_code") == 0 and published.get("published"):
            record("publish_file", "passed", "Container file was transferred to the download area", published)
        else:
            record("publish_file", "failed", "File publishing failed", published)
        # Retained web tools are network-dependent; failures are reported honestly.
        for name, args in (("web_search", {"query": "example domain", "max_results": 1}),
                           ("open_web_page", {"url": "https://example.com"}),
                           ("open_browser_page", {"url": "https://example.com"})):
            try:
                value = dispatch_tool(name, args)
                record(name, "passed" if value and not value.get("error") else "not_tested",
                       "Request completed" if value and not value.get("error") else "Network or browser dependency unavailable", value)
            except Exception as exc:
                record(name, "not_tested", f"{type(exc).__name__}: {str(exc)[:250]}")

        try:
            from .core import _run_internal_tool, _store_tool_result
            result_id, archive = _store_tool_result({}, {"content": "retrieval-smoke-ok"})
            value = _run_internal_tool("get_tool_result", {"result_id": result_id}, archive)
            record("get_tool_result", "passed" if "retrieval-smoke-ok" in json.dumps(value) else "failed",
                   "Internal result retrieval path checked", value)
        except Exception as exc:
            record("get_tool_result", "failed", f"{type(exc).__name__}: {str(exc)[:250]}")
    except Exception as exc:
        record("terminal", "failed", f"{type(exc).__name__}: {str(exc)[:400]}")
    finally:
        source.unlink(missing_ok=True)
        published_file.unlink(missing_ok=True)
        try:
            sandbox_manager.delete(cid)
        except Exception as exc:
            record("sandbox_cleanup", "failed", f"{type(exc).__name__}: {str(exc)[:250]}")
        db.delete_conversation(cid)

    for name in _tool_names():
        if name not in results and name != "self_test_tools":
            record(name, "not_tested", "No safe independent smoke check was run for this tool.")
    results["self_test_tools"] = {"status": "passed", "detail": "Self-test report generated."}
    statuses = [item["status"] for key, item in results.items() if key != "self_test_tools"]
    return {"ok": "failed" not in statuses, "run_id": run_id,
            "tested": sum(x in {"passed", "failed"} for x in statuses),
            "passed": statuses.count("passed"), "failed": statuses.count("failed"),
            "not_tested": statuses.count("not_tested"), "tools": results,
            "note": "not_tested means the capability was not confirmed; network-dependent checks may be unavailable."}
