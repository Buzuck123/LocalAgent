import threading

from agent import core
from agent.sandbox_manager import SandboxManager


def test_terminal_waits_through_silence_until_internal_checkpoint(monkeypatch):
    import agent.sandbox_manager as sandbox_module
    from types import SimpleNamespace

    manager = object.__new__(SandboxManager)
    manager._guard = threading.RLock()
    manager._active_operations = {}
    manager._new_sessions = set()
    manager._runtimes = {}
    manager.start_operation = lambda *args, **kwargs: {
        "operation_id": "op-1", "state": "running"
    }
    polls = []

    def get_operation(*args, **kwargs):
        polls.append(kwargs)
        return {"state": "running", "stdout": "", "next_cursor": 0}

    manager.get_operation = get_operation
    monkeypatch.setattr(
        manager, "get_runtime",
        lambda *args, **kwargs: SimpleNamespace(ensure_session=lambda session: False),
    )
    monkeypatch.setattr(
        sandbox_module, "settings",
        SimpleNamespace(
            terminal_idle_timeout_seconds=1,
            terminal_max_wait_seconds=300,
            terminal_poll_interval_seconds=1,
        ),
    )
    tick = {"value": 0}

    def monotonic():
        tick["value"] += 100
        return float(tick["value"])

    monkeypatch.setattr(sandbox_module.time, "monotonic", monotonic)

    result = manager.terminal_command("conversation-1", "pip install package")

    assert len(polls) >= 2
    assert result["_diagnostic_due"] is True
    assert result["_monitor_state"] == "running_silent"
    assert result["state"] == "running"


def test_terminal_diagnostic_isolated_from_user_task_and_uses_only_diagnostic_tools(monkeypatch):
    from agent import db
    from agent import sandbox_manager as sandbox_module

    class Manager:
        def collect_terminal_diagnostics(self, conversation_id, operation_id, **kwargs):
            assert conversation_id == "conversation-1"
            assert operation_id == "terminal-op"
            return {"runtime_responsive": True, "container_stats": "available"}

    monkeypatch.setattr(sandbox_module, "sandbox_manager", Manager())
    monkeypatch.setattr(db, "create_terminal_diagnostic", lambda *args, **kwargs: {})
    monkeypatch.setattr(db, "update_terminal_diagnostic", lambda *args, **kwargs: {})
    captured = {}

    def fake_chat(messages, tools=None, reasoning=True, model=None):
        captured["messages"] = messages
        captured["tools"] = tools
        return {"message": {"content": '{"action":"continue_waiting","reason":"Процесс активен."}'}}

    monkeypatch.setattr(core, "_client_chat", fake_chat)
    action, reason, diagnostic_id, messages, actions, recovery = core._run_terminal_diagnostic(
        "task-1",
        "conversation-1",
        {
            "operation_id": "terminal-op",
            "_terminal_command": "pip install package",
            "_terminal_output": "",
            "_elapsed_seconds": 300,
            "_idle_seconds": 300,
        },
    )

    assert action == "continue_waiting"
    assert recovery is None
    assert diagnostic_id
    assert "НЕ получаешь её исходный запрос" in messages[0]["content"]
    assert all("private user goal" not in str(item) for item in messages)
    assert {item["function"]["name"] for item in captured["tools"]} == {
        "get_additional_terminal_diagnostics",
        "diagnostic_terminal",
        "propose_terminal_recovery",
    }
    assert "pip install package" in str(captured["messages"])
    assert actions[0]["type"] == "evidence"
    assert actions[0]["evidence"]


def test_diagnostic_agent_cannot_reset_bash_directly(monkeypatch):
    from agent import db
    from agent import sandbox_manager as sandbox_module

    class Manager:
        reset_called = False

        def collect_terminal_diagnostics(self, conversation_id, operation_id, **kwargs):
            return {"runtime_responsive": True, "operation_state": "running"}

        def reset_terminal_session(self, *args, **kwargs):
            self.reset_called = True
            raise AssertionError("LLM must not invoke recovery directly")

    manager = Manager()
    monkeypatch.setattr(sandbox_module, "sandbox_manager", manager)
    monkeypatch.setattr(db, "create_terminal_diagnostic", lambda *args, **kwargs: {})
    monkeypatch.setattr(db, "update_terminal_diagnostic", lambda *args, **kwargs: {})

    def fake_chat(messages, tools=None, reasoning=True, model=None):
        return {"message": {"tool_calls": [{
            "id": "call-1",
            "function": {"name": "reset_bash_session", "arguments": {"reason": "stale session"}},
        }]}}
    monkeypatch.setattr(core, "_client_chat", fake_chat)

    action, reason, diagnostic_id, messages, actions, recovery = core._run_terminal_diagnostic(
        "task-1", "conversation-1",
        {"operation_id": "terminal-op", "_terminal_command": "sleep 999",
         "_terminal_output": "", "_elapsed_seconds": 600, "_idle_seconds": 600},
    )
    assert action == "continue_waiting"
    assert recovery is None
    assert manager.reset_called is False


def test_ollama_message_sanitizer_uses_diagnostic_tool_catalog():
    diagnostic_call = {
        "role": "assistant",
        "tool_calls": [{
            "id": "diag-call",
            "function": {
                "name": "get_additional_terminal_diagnostics",
                "arguments": {"reason": "Need a second process sample."},
            },
        }],
    }
    tool_result = {
        "role": "tool",
        "tool_name": "get_additional_terminal_diagnostics",
        "content": '{"ok":true,"diagnostics":{"runtime_responsive":true}}',
    }

    sanitized = core._sanitize_messages_for_ollama(
        [diagnostic_call, tool_result],
        tool_definitions=core.TERMINAL_DIAGNOSTIC_TOOLS,
    )

    assert sanitized[0]["tool_calls"][0]["function"]["name"] == "get_additional_terminal_diagnostics"
    assert sanitized[1]["role"] == "tool"
    assert sanitized[1]["tool_name"] == "get_additional_terminal_diagnostics"



def test_normal_long_terminal_result_exposes_only_terminal_output():
    result = {
        "_terminal_text": "raw terminal output\\n[internal monitor notice]",
        "_terminal_output": "raw terminal output",
        "state": "failed",
        "exit_code": 1,
        "_diagnostic_due": False,
    }
    assert core._model_visible_tool_result("terminal", result) == {
        "output": "raw terminal output"
    }


def test_terminal_recovery_report_is_separate_from_normal_output():
    report = core._terminal_recovery_report(
        "reset_bash_session",
        "Bash session was not responsive.",
        {"recovery": {"state": "reset", "verified": True}},
    )
    assert "Bash-сессия пересоздана" in report
    assert "команда не считается выполненной" in report

def test_controller_rejects_recovery_without_real_evidence(monkeypatch):
    from agent import db
    from agent import sandbox_manager as sandbox_module

    class Manager:
        interrupt_called = False

        def collect_terminal_diagnostics(self, conversation_id, operation_id, **kwargs):
            return {"runtime_responsive": False, "operation_state": "unknown"}

        def interrupt_operation(self, conversation_id, operation_id):
            self.interrupt_called = True
            raise AssertionError("controller must reject missing evidence")

    manager = Manager()
    monkeypatch.setattr(sandbox_module, "sandbox_manager", manager)
    monkeypatch.setattr(db, "create_terminal_diagnostic", lambda *args, **kwargs: {})
    monkeypatch.setattr(db, "update_terminal_diagnostic", lambda *args, **kwargs: {})

    def fake_chat(messages, tools=None, reasoning=True, model=None):
        return {"message": {"tool_calls": [{
            "id": "call-interrupt",
            "function": {
                "name": "propose_terminal_recovery",
                "arguments": {
                    "action": "interrupt_terminal_operation",
                    "reason": "The operation appears stuck.",
                    "evidence_ids": ["ev_model_invented"],
                },
            },
        }]}}
    monkeypatch.setattr(core, "_client_chat", fake_chat)
    action, reason, diagnostic_id, messages, actions, recovery = core._run_terminal_diagnostic(
        "task-1", "conversation-1",
        {"operation_id": "terminal-op", "_terminal_command": "sleep 999",
         "_terminal_output": "", "_elapsed_seconds": 600, "_idle_seconds": 600},
    )

    assert action == "continue_waiting"
    assert recovery is None
    assert manager.interrupt_called is False
    assert "отсутствующие доказательства" in str(messages)


def test_terminal_diagnostic_keeps_primary_agent_blocked_until_recovery(monkeypatch):
    from agent import db
    from agent import sandbox_manager as sandbox_module

    class Manager:
        def verify_terminal_session(self, conversation_id):
            return {"verified": True}

    monkeypatch.setattr(sandbox_module, "sandbox_manager", Manager())
    monkeypatch.setattr(
        db, "get_task",
        lambda task_id: {"id": task_id, "conversation_id": "conversation-1"},
    )
    monkeypatch.setattr(core, "_record_tool_debug_trace", lambda *args, **kwargs: None)
    monkeypatch.setattr(db, "update_terminal_diagnostic", lambda *args, **kwargs: {})
    calls = []

    def fake_dispatch(name, arguments, **kwargs):
        calls.append((name, arguments))
        return {
            "_terminal_text": "still running",
            "_terminal_output": "still running",
            "state": "running",
            "operation_id": "terminal-op",
            "_diagnostic_due": True,
        }

    monkeypatch.setattr(core, "dispatch_tool", fake_dispatch)
    reviews = []

    def fake_diagnostic(*args, **kwargs):
        reviews.append(True)
        if len(reviews) < 5:
            return "continue_waiting", "Still active.", "diag-1", [], [], None
        return "interrupt_operation", "Operation is stuck.", "diag-1", [], [], {
            "state": "interrupted", "interrupted": True,
        }

    monkeypatch.setattr(core, "_run_terminal_diagnostic", fake_diagnostic)
    result = core._dispatch_tool_for_task(
        "terminal", {"command": "sleep 999"}, "task-1"
    )

    assert len(reviews) == 5
    assert len(calls) == 5  # initial command plus four polls of the same active operation
    assert result["state"] == "interrupted"
    assert "Диагностика приостановлена после 3 циклов" not in result["_terminal_output"]


def test_terminal_diagnostic_llm_failure_is_preserved_for_task_pause(monkeypatch):
    from agent import db
    from agent import sandbox_manager as sandbox_module

    updates = []

    class Manager:
        def collect_terminal_diagnostics(self, conversation_id, operation_id, **kwargs):
            return {"runtime_responsive": True}

    monkeypatch.setattr(sandbox_module, "sandbox_manager", Manager())
    monkeypatch.setattr(db, "create_terminal_diagnostic", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        db, "update_terminal_diagnostic",
        lambda diagnostic_id, **kwargs: updates.append((diagnostic_id, kwargs)) or {},
    )

    def unavailable(*args, **kwargs):
        raise ConnectionError("LLM server is offline")

    monkeypatch.setattr(core, "_client_chat", unavailable)
    terminal_result = {
        "operation_id": "terminal-op",
        "_terminal_command": "pip install package",
        "_terminal_output": "Downloading package...",
        "_elapsed_seconds": 300,
        "_idle_seconds": 300,
    }

    try:
        core._run_terminal_diagnostic("task-1", "conversation-1", terminal_result)
        assert False, "expected diagnostic LLM failure"
    except core.TerminalDiagnosticLLMUnavailable as exc:
        assert exc.terminal_result["operation_id"] == "terminal-op"
        assert exc.messages[0]["role"] == "system"
        assert exc.diagnostic_id
    assert updates[-1][1]["status"] == "paused"


def test_resume_pending_terminal_diagnostic_polls_original_operation_and_verifies_bash(monkeypatch):
    from agent import sandbox_manager as sandbox_module

    class Manager:
        def get_operation(self, conversation_id, operation_id, *, cursor, max_output_chars, wait_timeout):
            assert operation_id == "terminal-op"
            return {
                "state": "completed", "stdout": "install finished",
                "stderr": "", "exit_code": 0, "cursor": 12,
                "timed_out": False, "interrupted": False, "duration_seconds": 360,
            }

        def verify_terminal_session(self, conversation_id):
            return {"verified": True}

    monkeypatch.setattr(sandbox_module, "sandbox_manager", Manager())
    monkeypatch.setattr(
        core, "_run_terminal_diagnostic",
        lambda *args, **kwargs: (
            "continue_waiting", "Runtime and operation state captured.",
            "diag-1", [{"role": "system", "content": "diagnostic"}], [], None,
        ),
    )
    result = core._resume_pending_terminal_diagnostic(
        "task-1", "conversation-1",
        {
            "terminal_result": {
                "operation_id": "terminal-op", "_terminal_command": "pip install package",
                "_terminal_output": "", "_terminal_text": "", "cursor": 4,
                "state": "running", "_elapsed_seconds": 300,
            },
            "diagnostic_id": "diag-1",
            "messages": [{"role": "system", "content": "diagnostic"}],
            "actions": [],
        },
    )

    assert result["state"] == "completed"
    assert result["shell_state"] == "available"
    assert "Bash-сессия прошла проверочную команду" in result["_terminal_output"]
    assert "Проверь установленные/обновлённые пакеты" in result["_terminal_output"]

def test_interrupt_race_polls_original_operation_instead_of_starting_new_command(monkeypatch):
    from agent import db
    from agent import sandbox_manager as sandbox_module

    monkeypatch.setattr(
        db, "get_task",
        lambda task_id: {"id": task_id, "conversation_id": "conversation-1"},
    )
    monkeypatch.setattr(core, "_record_tool_debug_trace", lambda *args, **kwargs: None)
    dispatch_calls = []

    def fake_dispatch(name, arguments, **kwargs):
        dispatch_calls.append((name, arguments))
        return {
            "_terminal_text": "partial output",
            "_terminal_output": "partial output",
            "state": "running",
            "operation_id": "terminal-op",
            "cursor": 7,
            "_diagnostic_due": True,
        }

    class Manager:
        def get_operation(self, conversation_id, operation_id, *, cursor, wait_timeout):
            assert conversation_id == "conversation-1"
            assert operation_id == "terminal-op"
            assert cursor == 7
            return {
                "state": "completed",
                "stdout": "final output",
                "exit_code": 0,
                "timed_out": False,
                "interrupted": False,
                "cursor": 12,
            }

        def verify_terminal_session(self, conversation_id):
            return {"verified": True}

    monkeypatch.setattr(core, "dispatch_tool", fake_dispatch)
    monkeypatch.setattr(sandbox_module, "sandbox_manager", Manager())
    monkeypatch.setattr(
        core, "_run_terminal_diagnostic",
        lambda *args, **kwargs: (
            "interrupt_operation", "The command finished during interrupt.",
            "diag-1", [], [], {"state": "completed"},
        ),
    )
    monkeypatch.setattr(db, "update_terminal_diagnostic", lambda *args, **kwargs: {})

    result = core._dispatch_tool_for_task(
        "terminal", {"command": "long command"}, "task-1"
    )

    assert len(dispatch_calls) == 1
    assert result["state"] == "completed"
    assert "final output" in result["_terminal_output"]




def test_diagnostic_report_is_compact_and_explains_metric_limits():
    from agent.terminal_diagnostics import build_diagnostic_context

    snapshot = {
        "sampled_at": "2026-11-19T12:00:00+00:00",
        "operation_id": "op-1",
        "command": "python3 worker.py --input data.csv",
        "elapsed_seconds": 1000,
        "idle_seconds": 1000,
        "operation_state": "running",
        "runtime_responsive": True,
        "container_top": {"returncode": 0, "output": "PID COMMAND\n1 server"},
        "container_stats": {"returncode": 0, "output": "0.00%|50MiB / 4GiB|1.2%|4"},
    }
    report, evidence = build_diagnostic_context(snapshot)

    assert report["operation"]["command"] == snapshot["command"]
    assert "не доказывает зависание" in " ".join(report["important_limits"])
    assert any("всего контейнера" in item["meaning"] for item in evidence)
    assert len(str(report)) < 12000


def test_repeated_silence_requires_spaced_samples_and_read_only_inspection():
    from agent.terminal_diagnostics import build_diagnostic_context, authorize_recovery

    first = {
        "sampled_at": "2026-11-19T12:00:00+00:00", "operation_id": "op-1",
        "command": "python3 worker.py", "elapsed_seconds": 1000, "idle_seconds": 1000,
        "operation_state": "running", "runtime_responsive": True,
    }
    _, evidence1 = build_diagnostic_context(first)
    second = dict(first, sampled_at="2026-11-19T12:00:20+00:00",
                  elapsed_seconds=1020, idle_seconds=1020)
    _, evidence2 = build_diagnostic_context(second, evidence1)
    diagnostic = {
        "id": "ev_readonly_1", "kind": "diagnostic_terminal",
        "operation_id": "op-1", "timestamp": "2026-11-19T12:00:21+00:00",
        "command": "ps -eo pid,stat,args", "success": True,
    }
    evidence = evidence1 + evidence2 + [diagnostic]
    selected = [
        next(item["id"] for item in evidence if item["kind"] == "operation_state"),
        next(item["id"] for item in evidence if item["kind"] == "no_terminal_output"),
        next(item["id"] for item in evidence if item["kind"] == "repeated_no_progress"),
        diagnostic["id"],
    ]
    allowed, reason = authorize_recovery(
        "interrupt_terminal_operation",
        {"evidence_ids": selected}, evidence, second,
    )
    assert allowed is True

    invented, _ = authorize_recovery(
        "interrupt_terminal_operation",
        {"evidence_ids": ["ev_not_real"]}, evidence, second,
    )
    assert invented is False


def test_diagnostic_terminal_only_allows_read_commands_and_uses_separate_session():
    import pytest

    manager = object.__new__(SandboxManager)

    class Runtime:
        def __init__(self):
            self.sessions = []
            self.calls = []

        def is_alive(self):
            return True

        def ensure_session(self, session):
            self.sessions.append(session)
            return True

        def run(self, command, *, session, timeout):
            self.calls.append((command, session, timeout))
            return {"stdout": "PID COMMAND", "stderr": "", "exit_code": 0, "timed_out": False}

    runtime = Runtime()
    manager._runtimes = {"conversation-1": runtime}
    result = manager.run_diagnostic_terminal("conversation-1", "ps -eo pid,ppid,stat,args")

    assert result["ok"] is True
    assert runtime.sessions == ["diagnostic"]
    command, session, timeout = runtime.calls[0]
    assert session == "diagnostic"
    assert "localagent-diagnostic" in command
    assert "--no-new-privs" in command
    assert timeout == 10

    with pytest.raises(ValueError):
        manager.run_diagnostic_terminal("conversation-1", "rm -rf /host-workspace")
    with pytest.raises(ValueError):
        manager.run_diagnostic_terminal("conversation-1", "bash -c 'touch /host-workspace/x'")
