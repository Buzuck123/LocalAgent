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



@pytest.mark.parametrize(
    "invoke",
    [
        lambda manager: manager.start_operation("conversation-1", "sleep 1"),
        lambda manager: manager.execute("conversation-1", "printf ok"),
    ],
)
def test_shell_execution_entry_points_refuse_a_busy_session(invoke):
    manager = _manager_with_active_operation()

    with pytest.raises(SandboxBusyError, match="Bash-сессия занята"):
        invoke(manager)



def test_empty_terminal_poll_bypasses_exact_repeat_guard():
    assert core._is_safe_terminal_recovery_call(
        "terminal", {"command": ""}, "conversation-1"
    )


def test_terminal_interrupt_bypasses_repeat_guard_only_for_active_operation(monkeypatch):
    class Manager:
        def __init__(self, active):
            self.active = active

        def get_active_operation(self, conversation_id):
            return self.active

    monkeypatch.setattr(
        "agent.sandbox_manager.sandbox_manager",
        Manager({"operation_id": "op-1", "state": "running"}),
    )
    assert core._is_safe_terminal_recovery_call(
        "terminal", {"command": "CTRL+C"}, "conversation-1"
    )

    monkeypatch.setattr("agent.sandbox_manager.sandbox_manager", Manager(None))
    assert not core._is_safe_terminal_recovery_call(
        "terminal", {"command": "CTRL+C"}, "conversation-1"
    )


def test_nonempty_shell_commands_remain_subject_to_repeat_guard():
    assert not core._is_safe_terminal_recovery_call(
        "terminal", {"command": "ffmpeg -i input.mp4 output.wav"}, "conversation-1"
    )
    assert not core._is_safe_terminal_recovery_call(
        "ask_model", {"file": "/workspace/audio.wav"}, "conversation-1"
    )


def test_reset_terminal_session_recreates_only_shell_after_unknown_operation():
    class Runtime:
        def __init__(self):
            self.closed = []
            self.created = []
            self.commands = []

        def interrupt_operation(self, operation_id, *, session="main"):
            raise ValueError(f"Unknown operation {operation_id}")

        def close_session(self, session="main"):
            self.closed.append(session)

        def ensure_session(self, session="main"):
            self.created.append(session)
            return True

        def run(self, command, *, session="main", timeout=10):
            self.commands.append((command, session, timeout))
            return {
                "stdout": "__LOCALAGENT_RESET_READY__",
                "stderr": "",
                "exit_code": 0,
                "timed_out": False,
            }

    manager = _manager_with_active_operation("stale-operation")
    manager._new_sessions = set()
    manager._runtimes = {}
    manager._find_persisted_active_operation_details = lambda conversation_id: None
    runtime = Runtime()
    manager.get_runtime = lambda conversation_id, **kwargs: runtime

    result = manager.reset_terminal_session(
        "conversation-1",
        confirm_reset=True,
        reason="Polling and CTRL+C return Unknown operation",
    )

    assert result["state"] == "reset"
    assert result["shell_state"] == "recreated"
    assert result["container_recreated"] is False
    assert result["workspace_preserved"] is True
    assert result["verified"] is True
    assert result["interrupt_error"] == "ValueError"
    assert runtime.closed == ["main"]
    assert runtime.created == ["main"]
    assert "unset SWEREX_AUTH_TOKEN" in runtime.commands[0][0]
    assert manager.get_active_operation("conversation-1") is None


def test_reset_terminal_session_requires_explicit_confirmation():
    manager = _manager_with_active_operation()
    manager._new_sessions = set()
    manager._runtimes = {}

    with pytest.raises(ValueError, match="confirm_reset must be true"):
        manager.reset_terminal_session(
            "conversation-1", confirm_reset=False, reason="test"
        )


def test_reset_terminal_session_does_not_clear_active_state_if_new_shell_fails():
    class Runtime:
        def interrupt_operation(self, operation_id, *, session="main"):
            return {"state": "unknown"}

        def close_session(self, session="main"):
            pass

        def ensure_session(self, session="main"):
            return False

    manager = _manager_with_active_operation("stale-operation")
    manager._new_sessions = set()
    manager._runtimes = {}
    manager.get_runtime = lambda conversation_id, **kwargs: Runtime()

    with pytest.raises(RuntimeError, match="не подтвердил создание новой Bash-сессии"):
        manager.reset_terminal_session(
            "conversation-1", confirm_reset=True, reason="runtime is inconsistent"
        )

    assert manager.get_active_operation("conversation-1")["operation_id"] == "stale-operation"


def test_controller_allows_explicit_terminal_reset_during_active_operation():
    assert "reset_terminal_session" in core._TERMINAL_CONTROL_TOOLS


def test_persisted_reset_receipt_clears_old_running_operation(monkeypatch):
    import json

    class Rows:
        def fetchall(self):
            return [(json.dumps({"state": "reset", "shell_state": "recreated"}),)]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, query, params):
            assert "reset_terminal_session" in query
            assert params == ("conversation-1",)
            return Rows()

    monkeypatch.setattr("agent.sandbox_manager.db.connect", lambda: Connection())
    manager = object.__new__(SandboxManager)

    assert manager._find_persisted_active_operation_details("conversation-1") is None
