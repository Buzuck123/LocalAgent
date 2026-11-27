import threading

import pytest

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


def test_unclosed_quote_is_sent_to_interactive_bash_for_ps2_detection(monkeypatch):
    from types import SimpleNamespace

    manager = object.__new__(SandboxManager)
    manager._guard = threading.RLock()
    manager._active_operations = {}
    manager._lock_for = lambda conversation_id: threading.RLock()
    manager.ensure_session_idle = lambda conversation_id: None
    manager._ensure_network_proxy = lambda runtime, command: None
    started = {}

    class Runtime:
        def start_operation(self, command, *, session, timeout, idempotency_key):
            started["command"] = command
            return {"operation_id": "op-incomplete", "state": "running"}

    manager.get_runtime = lambda conversation_id: Runtime()
    monkeypatch.setattr(
        "agent.sandbox_manager.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2, stdout="", stderr="bash: -c: unexpected EOF while looking for matching quote",
        ),
    )

    result = manager.start_operation("conversation-1", "echo 'unfinished")

    assert result["operation_id"] == "op-incomplete"
    assert started["command"].startswith("export PS2='__LOCALAGENT_BASH_CONTINUATION__'\n")
    assert "echo 'unfinished" in started["command"]


def test_ps2_marker_waits_for_no_progress_then_hands_off_to_diagnostics(monkeypatch):
    import agent.sandbox_manager as sandbox_module

    manager = object.__new__(SandboxManager)
    manager._guard = threading.RLock()
    manager._active_operations = {
        "conversation-1": {
            "operation_id": "op-heredoc", "session": "main",
            "cursor": 0, "output": "", "command": "cat <<EOF",
            "command_timeout": None, "waiting_for_continuation": True,
            "continuation_prompt_seen_at": None,
            "started_monotonic": 0.0, "last_output_monotonic": 0.0,
            "started_at_wall": 1000, "last_output_at_wall": 1000,
        }
    }
    manager._new_sessions = set()
    manager._runtimes = {}
    polls = []

    def get_operation(*args, **kwargs):
        polls.append(kwargs)
        return {
            "state": "running",
            "stdout": "__LOCALAGENT_BASH_CONTINUATION__" if len(polls) == 1 else "",
            "next_cursor": 1,
        }

    manager.get_operation = get_operation
    ticks = {"value": 0.0}

    def monotonic():
        ticks["value"] += 0.6
        return ticks["value"]

    monkeypatch.setattr(sandbox_module.time, "monotonic", monotonic)
    monkeypatch.setattr(
        sandbox_module, "settings",
        type("Settings", (), {
            "terminal_idle_timeout_seconds": 1,
            "terminal_max_wait_seconds": 30,
            "terminal_poll_interval_seconds": 0.25,
        })(),
    )

    result = manager.terminal_command("conversation-1", "")

    assert len(polls) >= 2
    assert result["state"] == "running"
    assert result["_diagnostic_due"] is True
    assert result["_diagnostic_reason"] == "bash_continuation_stalled"
    assert result["operation_id"] == "op-heredoc"
    assert "LOCALAGENT_BASH_CONTINUATION" not in result["_terminal_text"]


def test_valid_multiline_command_is_not_instrumented_with_ps2(monkeypatch):
    from types import SimpleNamespace

    manager = object.__new__(SandboxManager)
    manager._guard = threading.RLock()
    manager._active_operations = {}
    manager._lock_for = lambda conversation_id: threading.RLock()
    manager.ensure_session_idle = lambda conversation_id: None
    manager._ensure_network_proxy = lambda runtime, command: None
    started = {}

    class Runtime:
        def start_operation(self, command, *, session, timeout, idempotency_key):
            started["command"] = command
            started["timeout"] = timeout
            return {"operation_id": "op-valid", "state": "running"}

    manager.get_runtime = lambda conversation_id: Runtime()
    monkeypatch.setattr(
        "agent.sandbox_manager.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = manager.start_operation("conversation-1", "cat <<'EOF'\nhello\nEOF", timeout=77)

    assert started["command"] == "cat <<'EOF'\nhello\nEOF"
    assert started["timeout"] == 77
    assert result["expects_continuation"] is False
    assert result["timeout"] == 77


def test_diagnostic_recovery_tools_are_exposed_in_order():
    initial = {item["function"]["name"] for item in core._terminal_diagnostic_tools([])}
    after_interrupt = {item["function"]["name"] for item in core._terminal_diagnostic_tools([
        {"action": "interrupt_operation", "result": {"state": "failed"}}
    ])}
    after_reset = {item["function"]["name"] for item in core._terminal_diagnostic_tools([
        {"action": "interrupt_operation"},
        {"action": "reset_bash_session"},
    ])}
    after_restart = {item["function"]["name"] for item in core._terminal_diagnostic_tools([
        {"action": "interrupt_operation"},
        {"action": "reset_bash_session"},
        {"action": "restart_container"},
    ])}

    assert "interrupt_terminal_operation" in initial
    assert "reset_bash_session" not in initial
    assert "reset_bash_session" in after_interrupt
    assert "restart_sandbox_container" not in after_interrupt
    assert "restart_sandbox_container" in after_reset
    assert "recreate_sandbox_container" not in after_reset
    assert "recreate_sandbox_container" not in after_restart


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
    assert "не получаешь пользовательский запрос" in messages[0]["content"]
    assert all("private user goal" not in str(item) for item in messages)
    assert {item["function"]["name"] for item in captured["tools"]} == {
        "get_additional_terminal_diagnostics",
        "diagnostic_terminal",
        "interrupt_terminal_operation",
    }
    assert "pip install package" in str(captured["messages"])
    assert actions[0]["type"] == "snapshot"
    assert actions[0]["report"]["operation"]["state"] == "unknown"


def test_failed_main_probe_after_interrupt_exposes_session_reset_next(monkeypatch):
    from agent import db
    from agent import sandbox_manager as sandbox_module

    class Manager:
        snapshots = 0

        def collect_terminal_diagnostics(self, conversation_id, operation_id, **kwargs):
            self.snapshots += 1
            state = "running" if self.snapshots == 1 else "interrupted"
            return {"runtime_responsive": True, "operation_state": state}

        def interrupt_operation(self, conversation_id, operation_id):
            return {"state": "interrupted", "interrupted": True}

        def verify_terminal_session(self, conversation_id):
            return {"verified": False, "error": "main PTY did not answer"}

    manager = Manager()
    monkeypatch.setattr(sandbox_module, "sandbox_manager", manager)
    monkeypatch.setattr(db, "create_terminal_diagnostic", lambda *args, **kwargs: {})
    monkeypatch.setattr(db, "update_terminal_diagnostic", lambda *args, **kwargs: {})
    calls = []

    def fake_chat(messages, tools=None, reasoning=True, model=None):
        calls.append(tools)
        if len(calls) == 1:
            return {"message": {"tool_calls": [{
                "id": "interrupt",
                "function": {
                    "name": "interrupt_terminal_operation",
                    "arguments": {"reason": "No terminal progress"},
                },
            }]}}
        return {"message": {"content": '{"action":"continue_waiting","reason":"Need the next recovery stage."}'}}

    monkeypatch.setattr(core, "_client_chat", fake_chat)
    result = core._run_terminal_diagnostic(
        "task-1", "conversation-1",
        {"operation_id": "terminal-op", "_terminal_command": "cat > file",
         "_terminal_output": "", "_elapsed_seconds": 900, "_idle_seconds": 900},
    )
    assert result[0] == "recovery_failed"

    core._run_terminal_diagnostic(
        "task-1", "conversation-1",
        {"operation_id": "terminal-op", "_terminal_command": "cat > file",
         "_terminal_output": "", "_elapsed_seconds": 901, "_idle_seconds": 901},
        diagnostic_id=result[2], diagnostic_messages=result[3], diagnostic_actions=result[4],
    )
    next_tools = {item["function"]["name"] for item in calls[1]}
    assert "reset_bash_session" in next_tools
    assert "restart_sandbox_container" not in next_tools


def test_container_restart_automatically_recreates_when_restart_fails(monkeypatch):
    from agent import db
    from agent import sandbox_manager as sandbox_module

    class Manager:
        recreated = False

        def collect_terminal_diagnostics(self, conversation_id, operation_id, **kwargs):
            return {"runtime_responsive": False, "operation_state": "interrupted"}

        def restart_container(self, conversation_id, *, reason):
            raise RuntimeError("container could not restart")

        def recreate_container(self, conversation_id, *, reason):
            self.recreated = True
            return {"state": "recreated", "verified": True, "workspace_preserved": True}

    manager = Manager()
    monkeypatch.setattr(sandbox_module, "sandbox_manager", manager)
    monkeypatch.setattr(db, "update_terminal_diagnostic", lambda *args, **kwargs: {})
    calls = []

    def fake_chat(messages, tools=None, reasoning=True, model=None):
        calls.append(tools)
        return {"message": {"tool_calls": [{
            "id": "restart",
            "function": {
                "name": "restart_sandbox_container",
                "arguments": {"reason": "SWE-ReX is not responding"},
            },
        }]}}

    monkeypatch.setattr(core, "_client_chat", fake_chat)
    action, reason, diagnostic_id, messages, actions, recovery = core._run_terminal_diagnostic(
        "task-1", "conversation-1",
        {"operation_id": "terminal-op", "_terminal_command": "sleep 999",
         "_terminal_output": "", "_elapsed_seconds": 900, "_idle_seconds": 900},
        diagnostic_id="diag-1",
        diagnostic_messages=[{"role": "system", "content": core.TERMINAL_DIAGNOSTIC_SYSTEM_PROMPT}],
        diagnostic_actions=[
            {"action": "interrupt_operation", "result": {"state": "interrupted", "verified": False}},
            {"action": "reset_bash_session", "result": {"state": "reset", "verified": False}},
        ],
    )

    assert action == "restart_container"
    assert recovery["state"] == "recreated"
    assert recovery["verified"] is True
    assert manager.recreated is True


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
    assert core._model_visible_tool_result("terminal", result) == "raw terminal output"


def test_terminal_recovery_report_is_separate_from_normal_output():
    report = core._terminal_recovery_report(
        "reset_bash_session",
        "Bash session was not responsive.",
        {"recovery": {"state": "reset", "verified": True}},
    )
    assert "Bash-сессия пересоздана" in report
    assert "команда не считается выполненной" in report

def test_legacy_evidence_proposal_is_not_exposed_or_executed(monkeypatch):
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
    assert "propose_terminal_recovery" not in {item["function"]["name"] for item in core.TERMINAL_DIAGNOSTIC_TOOLS}


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
            "state": "interrupted", "interrupted": True, "verified": True,
        }

    monkeypatch.setattr(core, "_run_terminal_diagnostic", fake_diagnostic)
    result = core._dispatch_tool_for_task(
        "terminal", {"command": "sleep 999"}, "task-1"
    )

    assert len(reviews) == 5
    assert len(calls) == 5  # initial command plus four polls of the same active operation
    assert result["state"] == "interrupted"
    assert "Диагностика приостановлена после 3 циклов" not in result["_terminal_output"]


def test_internal_terminal_poll_error_stays_inside_controller(monkeypatch):
    from agent import db
    from agent import sandbox_manager as sandbox_module

    class Manager:
        def is_runtime_responsive(self, conversation_id):
            return True

        def get_operation(self, conversation_id, operation_id, **kwargs):
            return {
                "state": "completed",
                "stdout": "final command output",
                "stderr": "",
                "cursor": 42,
                "exit_code": 0,
                "timed_out": False,
                "interrupted": False,
            }

    monkeypatch.setattr(sandbox_module, "sandbox_manager", Manager())
    monkeypatch.setattr(core, "_verify_or_recover_terminal_session", lambda *args, **kwargs: {"verified": True})
    monkeypatch.setattr(
        db, "get_task",
        lambda task_id: {"id": task_id, "conversation_id": "conversation-1"},
    )
    monkeypatch.setattr(core, "_record_tool_debug_trace", lambda *args, **kwargs: None)
    monkeypatch.setattr(db, "update_terminal_diagnostic", lambda *args, **kwargs: {})
    calls = []

    def fake_dispatch(name, arguments, **kwargs):
        calls.append((name, arguments))
        if len(calls) == 1:
            return {
                "_terminal_text": "",
                "_terminal_output": "",
                "_terminal_command": "sleep 30",
                "state": "running",
                "operation_id": "terminal-op",
                "cursor": 0,
                "_diagnostic_due": True,
            }
        raise RuntimeError("temporary proxy JSONDecodeError")

    monkeypatch.setattr(core, "dispatch_tool", fake_dispatch)
    monkeypatch.setattr(
        core, "_run_terminal_diagnostic",
        lambda *args, **kwargs: ("continue_waiting", "Still active.", "diag-1", [], [], None),
    )

    result = core._dispatch_tool_for_task(
        "terminal", {"command": "sleep 30"}, "task-1"
    )

    assert len(calls) == 2  # initial command plus one internal poll
    assert result["state"] == "completed"
    assert result["_terminal_text"] == "final command output"
    assert "JSONDecodeError" not in result["_terminal_text"]
    assert "_runtime_error" not in result
    assert "_diagnostic_due" not in result
    assert "operation_id" not in result


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


def test_resume_pending_diagnostic_returns_natural_completion_silently(monkeypatch):
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
    assert result["_terminal_output"] == "install finished"
    assert "Диагностика была приостановлена" not in result["_terminal_output"]
    assert "Проверь установленные/обновлённые пакеты" not in result["_terminal_output"]

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




def test_diagnostic_report_is_compact_without_evidence_gate():
    from agent.terminal_diagnostics import build_diagnostic_context

    snapshot = {
        "operation_id": "op-1",
        "command": "python3 worker.py --input data.csv",
        "elapsed_seconds": 1000,
        "idle_seconds": 1000,
        "operation_state": "running",
        "runtime_responsive": True,
        "container_stats": {"returncode": 0, "output": "0.00%|50MiB / 4GiB|1.2%|4"},
    }
    report, evidence = build_diagnostic_context(snapshot)

    assert report["command"] == snapshot["command"]
    assert report["operation"]["state"] == "running"
    assert report["operation"]["silent_seconds"] == 1000
    assert report["runtime"]["swerex_responsive"] is True
    assert evidence == []
    assert "не доказывает зависание" in " ".join(report["important_notes"])
    assert "evidence_ids" not in str(report)
    assert len(str(report)) < 12000



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



def test_diagnostic_terminal_allows_package_presence_checks_but_not_mutations():
    import pytest

    manager = object.__new__(SandboxManager)

    class Runtime:
        def is_alive(self):
            return True

        def ensure_session(self, session):
            return True

        def run(self, command, *, session, timeout):
            assert session == "diagnostic"
            assert "pip show requests" in command
            return {"stdout": "Name: requests", "stderr": "", "exit_code": 0, "timed_out": False}

    manager._runtimes = {"conversation-1": Runtime()}
    result = manager.run_diagnostic_terminal("conversation-1", "pip show requests")
    assert result["ok"] is True
    assert "Name: requests" in result["stdout"]

    with pytest.raises(ValueError):
        manager.run_diagnostic_terminal("conversation-1", "pip install requests")
    with pytest.raises(ValueError):
        manager.run_diagnostic_terminal("conversation-1", "git reset --hard")


def test_diagnostic_terminal_does_not_create_session_when_swerex_is_unresponsive():
    import pytest
    from agent.sandbox_manager import SandboxError

    manager = object.__new__(SandboxManager)

    class Runtime:
        created = False

        def is_alive(self):
            return False

        def ensure_session(self, session):
            self.created = True
            raise AssertionError("session must not be created before health check")

    runtime = Runtime()
    manager._runtimes = {"conversation-1": runtime}

    with pytest.raises(SandboxError, match="SWE-ReX не отвечает"):
        manager.run_diagnostic_terminal("conversation-1", "ps -ef")
    assert runtime.created is False


def test_diagnostic_summary_does_not_require_process_evidence():
    from agent.terminal_diagnostics import build_diagnostic_context

    snapshot = {
        "operation_id": "op-2",
        "command": "cat > result.json",
        "elapsed_seconds": 1200,
        "idle_seconds": 1200,
        "operation_state": "running",
        "runtime_responsive": True,
        "diagnostic_reason": "bash_continuation_stalled",
        "bash_continuation_pending": True,
        "process_snapshot": {"ok": False, "error": "diagnostic session unavailable"},
    }
    report, observations = build_diagnostic_context(snapshot)
    assert report["operation"]["state"] == "running"
    assert report["operation"]["monitor_reason"] == "bash_continuation_stalled"
    assert report["operation"]["bash_continuation_pending"] is True
    assert "processes" not in report["runtime"]
    assert observations == []

@pytest.mark.parametrize(("command", "expected"), [
    ("pip list", False),
    ("pip show vite", False),
    ("pip install vite", True),
    ("python -m pip install vite", True),
    ("npm run build", False),
    ("npm install", True),
    ("pnpm test", False),
    ("pnpm add vite", True),
    ("git status", False),
    ("git submodule status", False),
    ("git submodule update", True),
    ("git clone https://example.invalid/repo.git", True),
    ("uv tool list", False),
    ("uv tool install ruff", True),
    ("cat > package.json <<'EOF'\\n{\\n  \"script\": \"npm run build\"\\n}\\nEOF", False),
    ("echo 'curl https://example.invalid' > notes.txt", False),
    ("curl https://example.invalid", True),
])
def test_network_requirement_uses_executable_commands_not_arbitrary_text(command, expected):
    assert SandboxManager._command_requires_network(command) is expected


def test_executor_cannot_interrupt_or_recreate_terminal_environment():
    names = {
        item["function"]["name"]
        for item in core.TOOL_DEFINITIONS
    }
    assert "reset_terminal_session" not in names
    assert "terminal_interrupt" not in names
    assert "reset_terminal_session" not in core.SYSTEM_PROMPT
    from agent import tools as tools_module
    assert "terminal_interrupt" not in tools_module._LEGACY_CHECKPOINT_TOOL_NAMES
    assert core._is_safe_terminal_recovery_call(
        "terminal", {"command": "CTRL+C"}, "conversation-1"
    ) is False
    assert core._is_safe_terminal_recovery_call(
        "terminal", {"command": ""}, "conversation-1"
    ) is False
    assert core._is_safe_terminal_recovery_call(
        "reset_terminal_session", {"confirm_reset": True}, "conversation-1"
    ) is False


def test_terminal_timeout_is_retained_as_operation_metadata(monkeypatch):
    import agent.sandbox_manager as sandbox_module
    from types import SimpleNamespace

    manager = object.__new__(SandboxManager)
    manager._guard = threading.RLock()
    manager._active_operations = {}
    manager._new_sessions = set()
    manager._runtimes = {}
    manager._lock_for = lambda conversation_id: threading.RLock()
    captured = {}

    def start_operation(conversation_id, command, timeout=None, **kwargs):
        captured["timeout"] = timeout
        return {"operation_id": "op-timeout", "state": "running",
                "expects_continuation": False, "timeout": timeout}

    manager.start_operation = start_operation
    manager.get_operation = lambda *args, **kwargs: {
        "state": "completed", "stdout": "done", "next_cursor": 1, "exit_code": 0,
    }
    manager.get_runtime = lambda *args, **kwargs: SimpleNamespace(
        run=lambda *a, **k: {"exit_code": 0, "stdout": "__LOCALAGENT_TERMINAL_READY__"}
    )
    monkeypatch.setattr(
        sandbox_module, "settings",
        SimpleNamespace(
            terminal_idle_timeout_seconds=1,
            terminal_poll_interval_seconds=0.25,
        ),
    )

    result = manager.terminal_command("conversation-1", "echo done", timeout=73)

    assert captured["timeout"] == 73
    assert result["_terminal_hard_timeout"] == 73

def test_progressing_command_is_not_diagnosed_by_legacy_max_wait(monkeypatch):
    import agent.sandbox_manager as sandbox_module
    from types import SimpleNamespace

    manager = object.__new__(SandboxManager)
    manager._guard = threading.RLock()
    manager._active_operations = {}
    manager._new_sessions = set()
    manager._runtimes = {}
    manager._lock_for = lambda conversation_id: threading.RLock()
    manager.start_operation = lambda *args, **kwargs: {
        "operation_id": "op-progress", "state": "running", "expects_continuation": False,
    }
    polls = []

    def get_operation(*args, **kwargs):
        polls.append(kwargs)
        return {
            "state": "completed" if len(polls) == 4 else "running",
            "stdout": f"progress-{len(polls)}\\n",
            "next_cursor": len(polls),
            "exit_code": 0 if len(polls) == 4 else None,
        }

    manager.get_operation = get_operation
    manager.get_runtime = lambda *args, **kwargs: SimpleNamespace(
        run=lambda *a, **k: {"exit_code": 0, "stdout": "__LOCALAGENT_TERMINAL_READY__"}
    )
    monkeypatch.setattr(
        sandbox_module, "settings",
        SimpleNamespace(
            terminal_idle_timeout_seconds=10,
            terminal_max_wait_seconds=0.01,
            terminal_poll_interval_seconds=0.25,
        ),
    )

    result = manager.terminal_command("conversation-1", "long-running-command")

    assert len(polls) == 4
    assert result["state"] == "completed"
    assert result["_diagnostic_due"] is False
