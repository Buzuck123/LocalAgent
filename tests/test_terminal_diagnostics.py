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
        "interrupt_terminal_operation",
        "reset_bash_session",
        "restart_sandbox_container",
    }


def test_terminal_diagnostic_can_reset_bash_without_main_agent_context(monkeypatch):
    from agent import db
    from agent import sandbox_manager as sandbox_module

    class Manager:
        def collect_terminal_diagnostics(self, conversation_id, operation_id, **kwargs):
            return {"runtime_responsive": True}

        def reset_terminal_session(self, conversation_id, *, confirm_reset, reason):
            assert confirm_reset is True
            return {"state": "reset", "workspace_preserved": True}

    monkeypatch.setattr(sandbox_module, "sandbox_manager", Manager())
    monkeypatch.setattr(db, "create_terminal_diagnostic", lambda *args, **kwargs: {})
    monkeypatch.setattr(db, "update_terminal_diagnostic", lambda *args, **kwargs: {})

    def fake_chat(messages, tools=None, reasoning=True, model=None):
        return {"message": {
            "tool_calls": [{
                "id": "call-1",
                "function": {
                    "name": "reset_bash_session",
                    "arguments": {"reason": "Runtime operation ID is stale."},
                },
            }],
        }}

    monkeypatch.setattr(core, "_client_chat", fake_chat)
    action, reason, diagnostic_id, messages, actions, recovery = core._run_terminal_diagnostic(
        "task-1", "conversation-1",
        {"operation_id": "terminal-op", "_terminal_command": "sleep 999",
         "_terminal_output": "", "_elapsed_seconds": 600, "_idle_seconds": 600},
    )

    assert action == "reset_bash_session"
    assert recovery["state"] == "reset"
    assert any(item.get("role") == "tool" for item in messages)



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
