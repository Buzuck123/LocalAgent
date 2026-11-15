import threading

import pytest

from agent import core
from agent.sandbox_manager import SandboxBusyError, SandboxManager


def _manager_with_active_operation(operation_id="remote-op-1"):
    manager = object.__new__(SandboxManager)
    manager._guard = threading.RLock()
    manager._locks = {}
    manager._active_operations = {
        "conversation-1": {
            "operation_id": operation_id,
            "session": "main",
        }
    }
    return manager


def test_controller_returns_explicit_blocker_for_non_terminal_tools(monkeypatch):
    class Manager:
        def get_active_operation(self, conversation_id):
            assert conversation_id == "conversation-1"
            return {"operation_id": "remote-op-1", "state": "running"}

    monkeypatch.setattr("agent.sandbox_manager.sandbox_manager", Manager())

    blocker = core._active_terminal_operation_blocker("conversation-1")

    assert blocker["ok"] is False
    assert blocker["blocked_by_active_terminal_operation"] is True
    assert blocker["active_operation"] == {"state": "running"}
    assert "remote-op-1" not in blocker["error"]
    assert blocker["required_tool"] == "terminal"
    assert "пустой command" in blocker["error"]
    assert "CTRL+C" in blocker["error"]


def test_controller_allows_only_terminal_control_tools_during_active_operation():
    assert "terminal" in core._TERMINAL_CONTROL_TOOLS
    assert "terminal_poll" in core._TERMINAL_CONTROL_TOOLS
    assert "ask_model" not in core._TERMINAL_CONTROL_TOOLS
    assert "load_file_for_model" not in core._TERMINAL_CONTROL_TOOLS
    assert "publish_file" not in core._TERMINAL_CONTROL_TOOLS


def test_sandbox_manager_detects_in_memory_active_operation_and_blocks_session_use():
    manager = _manager_with_active_operation()

    active = manager.get_active_operation("conversation-1")

    assert active["operation_id"] == "remote-op-1"
    assert active["state"] == "running"
    with pytest.raises(SandboxBusyError, match="Bash-сессия занята"):
        manager.ensure_session_idle("conversation-1")


def test_sandbox_manager_detects_persisted_active_operation_after_restart(monkeypatch):
    manager = object.__new__(SandboxManager)
    manager._guard = threading.RLock()
    manager._locks = {}
    manager._active_operations = {}
    manager._find_persisted_active_operation_details = lambda cid: {
        "operation_id": "persisted-op",
        "state": "waiting_input",
    }

    active = manager.get_active_operation("conversation-1")

    assert active["operation_id"] == "persisted-op"
    assert active["state"] == "waiting_input"
    with pytest.raises(SandboxBusyError, match="Bash-сессия занята"):
        manager.ensure_session_idle("conversation-1")


def test_download_file_checks_busy_guard_before_using_runtime(tmp_path, monkeypatch):
    manager = _manager_with_active_operation()
    monkeypatch.setattr(manager, "get_runtime", lambda cid: (_ for _ in ()).throw(
        AssertionError("runtime must not be touched while a command is active")
    ))
    target = tmp_path / "download.bin"

    with pytest.raises(SandboxBusyError, match="Bash-сессия занята"):
        manager.download_file(
            "conversation-1", "/workspace/source.bin", target
        )

    assert not target.exists()



def test_controller_allows_poll_and_interrupt_but_blocks_new_shell_command(monkeypatch):
    class Manager:
        def get_active_operation(self, conversation_id):
            return {"operation_id": "remote-op", "state": "running"}

    monkeypatch.setattr("agent.sandbox_manager.sandbox_manager", Manager())

    assert core._active_terminal_operation_blocker(
        "conversation-1", terminal_command=""
    ) is None
    assert core._active_terminal_operation_blocker(
        "conversation-1", terminal_command="CTRL+C"
    ) is None

    blocker = core._active_terminal_operation_blocker(
        "conversation-1", terminal_command="ls -la"
    )
    assert blocker["blocked_by_active_terminal_operation"] is True
    assert "не запущена" in blocker["error"]
    assert "stdin" in blocker["error"]


def test_controller_allows_interactive_input_only_when_terminal_waits_for_input(monkeypatch):
    class Manager:
        def get_active_operation(self, conversation_id):
            return {"operation_id": "remote-op", "state": "waiting_input"}

    monkeypatch.setattr("agent.sandbox_manager.sandbox_manager", Manager())

    assert core._active_terminal_operation_blocker(
        "conversation-1", terminal_command="yes"
    ) is None



def test_attachment_upload_reuses_deterministic_path_without_touching_busy_pty(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from agent import sandbox_manager as sandbox_module

    attachment_root = tmp_path / "attachments"
    attachment_root.mkdir()
    source = attachment_root / "clip.wav"
    source.write_bytes(b"audio")
    monkeypatch.setattr(
        sandbox_module,
        "settings",
        SimpleNamespace(
            database_path=tmp_path / "agent.sqlite3",
            workspace_path=tmp_path / "workspace",
        ),
    )
    manager = _manager_with_active_operation()
    monkeypatch.setattr(manager, "get_runtime", lambda cid: (_ for _ in ()).throw(
        AssertionError("busy PTY must not be used for attachment upload")
    ))

    destination = manager.upload_file("conversation-1", source, "clip.wav")

    assert destination.startswith("/workspace/uploads/")
