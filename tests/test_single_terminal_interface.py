from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from agent import sandbox_manager as sandbox_module
from agent.tools import TOOL_DEFINITIONS


def _manager(monkeypatch, *, idle=0.02, maximum=0.2):
    manager = sandbox_module.SandboxManager.__new__(sandbox_module.SandboxManager)
    manager._guard = threading.RLock()
    manager._active_operations = {}
    manager._new_sessions = set()
    manager.get_runtime = lambda cid: SimpleNamespace(ensure_session=lambda session="main": False)
    monkeypatch.setattr(
        sandbox_module, "settings",
        SimpleNamespace(
            terminal_idle_timeout_seconds=idle,
            terminal_max_wait_seconds=maximum,
        ),
    )
    return manager


def test_model_sees_one_terminal_tool_only():
    names = [item["function"]["name"] for item in TOOL_DEFINITIONS]
    assert names.count("terminal") == 1
    assert not ({"terminal_start", "terminal_poll", "terminal_input",
                 "terminal_interrupt", "terminal_resize"} & set(names))


def test_terminal_returns_complete_transcript_without_exit_code(monkeypatch):
    manager = _manager(monkeypatch)
    calls = iter([
        {"state": "running", "stdout": "first line\\r\\n", "next_cursor": 11},
        {"state": "completed", "stdout": "second line\\r\\n", "next_cursor": 23,
         "exit_code": 7, "duration_seconds": 0.1},
    ])
    manager.start_operation = lambda *a, **k: {"operation_id": "op-1", "state": "running"}

    def get_operation(*a, **k):
        return next(calls)

    manager.get_operation = get_operation
    result = manager.terminal_command("conversation-1", "printf test")
    assert result["_terminal_text"] == "first line\\r\\nsecond line\\r\\n"
    assert result["state"] == "completed"
    assert result["exit_code"] == 7  # internal only; core strips it from the model transcript


def test_terminal_returns_control_after_idle_and_waits_again(monkeypatch):
    manager = _manager(monkeypatch, idle=0.1, maximum=0.2)
    manager.start_operation = lambda *a, **k: {"operation_id": "op-2", "state": "running"}
    outputs = iter([
        {"state": "running", "stdout": "progress\\n", "next_cursor": 8},
        {"state": "running", "stdout": "", "next_cursor": 8},
        {"state": "completed", "stdout": "done\\n", "next_cursor": 13, "exit_code": 0},
    ])
    def get_operation(*a, **k):
        time.sleep(0.11)
        try:
            return next(outputs)
        except StopIteration:
            return {"state": "completed", "stdout": "", "next_cursor": 13, "exit_code": 0}
    manager.get_operation = get_operation

    first = manager.terminal_command("conversation-2", "long-running-command")
    assert first["state"] == "running"
    assert "progress" in first["_terminal_text"]
    assert "ещё выполняется" in first["_terminal_text"]

    second = manager.terminal_command("conversation-2", "")
    assert second["state"] == "completed"
    assert "progress\\n" in second["_terminal_text"]
    assert "done\\n" in second["_terminal_text"]


def test_active_terminal_routes_text_as_input_and_ctrl_c_as_interrupt(monkeypatch):
    manager = _manager(monkeypatch, idle=0.1, maximum=0.2)
    manager._active_operations["conversation-3"] = {
        "operation_id": "op-3", "session": "main", "cursor": 0,
        "output": "", "started_monotonic": time.monotonic(),
    }
    sent = []
    interrupted = []
    manager.send_operation_input = lambda cid, oid, value, sendline=True: sent.append((cid, oid, value, sendline)) or {}
    manager.interrupt_operation = lambda cid, oid: interrupted.append((cid, oid)) or {"state": "interrupted"}
    def poll(*a, **k):
        time.sleep(0.11)
        return {
            "state": "interrupted" if interrupted else "running", "stdout": "", "next_cursor": 0,
            "exit_code": 130 if interrupted else None,
        }
    manager.get_operation = poll

    manager.terminal_command("conversation-3", "yes")
    assert sent == [("conversation-3", "op-3", "yes", True)]

    # Recreate an active operation to test Ctrl+C routing.
    manager._active_operations["conversation-3"] = {
        "operation_id": "op-4", "session": "main", "cursor": 0,
        "output": "", "started_monotonic": time.monotonic(),
    }
    manager.get_operation = lambda *a, **k: {
        "state": "interrupted", "stdout": "", "next_cursor": 0, "exit_code": 130,
    }
    manager.terminal_command("conversation-3", "\x03")
    assert interrupted[-1] == ("conversation-3", "op-4")


def test_terminal_recovers_persisted_operation_and_accumulated_output(monkeypatch):
    manager = _manager(monkeypatch)
    manager._find_persisted_active_operation_details = lambda cid: {
        "operation_id": "persisted-op",
        "state": "running",
        "cursor": 17,
        "_terminal_output": "before restart\\n",
        "_terminal_command": "long-command",
    }
    seen = []
    def poll(cid, oid, **kwargs):
        seen.append((cid, oid, kwargs["cursor"]))
        return {
            "state": "completed",
            "stdout": "after restart\\n",
            "next_cursor": 31,
            "exit_code": 0,
        }
    manager.get_operation = poll
    result = manager.terminal_command("conversation-restart", "")
    assert result["_terminal_text"] == "before restart\\nafter restart\\n"
    assert result["_terminal_command"] == "long-command"
    assert seen == [("conversation-restart", "persisted-op", 17)]


def test_terminal_reports_when_bash_session_had_to_be_recreated(monkeypatch):
    manager = _manager(monkeypatch)
    manager.start_operation = lambda *a, **k: {"operation_id": "op-session", "state": "running"}
    manager.get_operation = lambda *a, **k: {
        "state": "completed", "stdout": "done\\n", "next_cursor": 5, "exit_code": 0,
    }
    manager.get_runtime = lambda cid: SimpleNamespace(ensure_session=lambda session="main": True)
    result = manager.terminal_command("conversation-session", "exit")
    assert result["shell_state"] == "recreated"
    assert "Bash-сессия" in result["_terminal_text"]



def test_terminal_rejects_actual_nul_byte(monkeypatch):
    manager = _manager(monkeypatch)
    try:
        manager.terminal_command("conversation-nul", "echo safe\x00unsafe")
    except ValueError as exc:
        assert "NUL" in str(exc)
    else:
        raise AssertionError("NUL byte was accepted")
