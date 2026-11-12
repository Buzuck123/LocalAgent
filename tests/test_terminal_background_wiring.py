from types import SimpleNamespace

from agent import tools


def test_default_terminal_uses_single_facade_and_hides_operation_id(monkeypatch):
    calls = []

    class Manager:
        def terminal_command(self, conversation_id, command, *, timeout=None, idempotency_key=None):
            calls.append((conversation_id, command, timeout, idempotency_key))
            return {
                "_terminal_text": "still running\\n[Команда ещё выполняется.]",
                "operation_id": "remote-op-1", "state": "running", "exit_code": None,
            }

    monkeypatch.setattr("agent.sandbox_manager.sandbox_manager", Manager())
    result = tools.dispatch_tool(
        "terminal", {"command": "sleep 60", "timeout": None},
        conversation_id="conversation-1", idempotency_key="tool-operation-1",
    )
    assert result["operation_id"] == "remote-op-1"
    assert result["state"] == "running"
    assert result["_terminal_text"].startswith("still running")
    assert calls == [("conversation-1", "sleep 60", None, "tool-operation-1")]


def test_terminal_tool_catalog_exposes_no_control_subtools():
    names = {item["function"]["name"] for item in tools.TOOL_DEFINITIONS}
    assert "terminal" in names
    assert not ({"terminal_start", "terminal_poll", "terminal_input",
                 "terminal_interrupt", "terminal_resize"} & names)


def test_agent_dispatch_routes_single_terminal_to_conversation(monkeypatch):
    from agent import core

    calls = []
    monkeypatch.setattr(core.db, "get_task", lambda tid: {"id": tid, "conversation_id": "cid"})
    monkeypatch.setattr(core.db, "record_debug_trace", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "dispatch_tool", lambda name, args, **kwargs: calls.append((name, kwargs)) or {"ok": True})
    result = core._dispatch_tool_for_task("terminal", {"command": "pwd"}, "task", "journal-id")
    assert result == {"ok": True}
    assert calls == [("terminal", {"conversation_id": "cid", "idempotency_key": "journal-id"})]


def test_background_operation_normalization_preserves_state_and_cursor():
    from agent.sandbox_manager import SandboxManager

    class Runtime:
        def get_operation(self, operation_id, **kwargs):
            return {
                "operation_id": operation_id, "state": "timed_out", "output": "partial output",
                "next_cursor": 14, "oldest_cursor": 0, "output_truncated": True,
                "exit_code": 124, "failure_reason": "time limit reached",
                "started_at": 10.0, "finished_at": 15.5,
            }

    manager = SandboxManager(runtime_factory=lambda **kwargs: Runtime(), command_runner=lambda *a, **k: None)
    manager.get_runtime = lambda cid: Runtime()
    result = manager.get_operation("conversation-3", "op-3", cursor=7)
    assert result["state"] == "timed_out"
    assert result["stdout"] == "partial output"
    assert result["stderr"] == "time limit reached"
    assert result["output_stream"] == "combined"
    assert result["timed_out"] is True
    assert result["duration_seconds"] == 5.5
    assert result["stdout_truncated"] is True
    assert result["cursor"] == 14


def test_agent_dispatch_routes_all_terminal_control_tools_to_conversation(monkeypatch):
    from agent import core

    calls = []
    monkeypatch.setattr(core.db, "get_task", lambda tid: {"id": tid, "conversation_id": "cid"})
    monkeypatch.setattr(core.db, "record_debug_trace", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "dispatch_tool", lambda name, args, **kwargs: calls.append((name, kwargs)) or {"ok": True})
    result = core._dispatch_tool_for_task("terminal_poll", {"operation_id": "op"}, "task", "journal-id")
    assert result == {"ok": True}
    assert calls == [("terminal_poll", {"conversation_id": "cid", "idempotency_key": "journal-id"})]


def test_persisted_running_operation_can_be_found_after_manager_restart(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from agent import db
    from agent.sandbox_manager import SandboxManager

    monkeypatch.setattr(db, "settings", SimpleNamespace(database_path=tmp_path / "ops.sqlite3"))
    db.init_db()
    conversation = db.create_conversation("recover op")
    task = db.create_task(conversation["id"], "long command")
    db.create_tool_operation("journal-1", task["id"], "terminal", {"command": "sleep 30"})
    db.complete_tool_operation("journal-1", {
        "operation_id": "remote-op", "state": "running", "background": True,
    })
    manager = SandboxManager(command_runner=lambda *a, **k: None)
    assert manager._find_persisted_active_operation(conversation["id"]) == "remote-op"


def test_single_terminal_does_not_force_controller_polling():
    from agent.core import _update_pending_terminal_operations, _tool_result_is_successful

    state = {}
    result = {
        "_terminal_text": "partial transcript\\n[Команда ещё выполняется.]",
        "operation_id": "op-live", "state": "running", "cursor": 12,
    }
    _update_pending_terminal_operations(state, "terminal", result)
    assert "pending_terminal_operations" not in state
    assert _tool_result_is_successful(result) is True
    # Legacy checkpoint/control compatibility remains available internally.
    _update_pending_terminal_operations(state, "terminal_poll", {
        "operation_id": "legacy-op", "state": "running", "next_cursor": 28,
    })
    assert state["pending_terminal_operations"]["legacy-op"]["cursor"] == 28
    assert _tool_result_is_successful({"state": "failed", "failure_reason": "spawn error"}) is False



def test_mid_operation_proxy_failure_triggers_health_recheck_not_command_replay(monkeypatch):
    from agent.sandbox_manager import SandboxManager

    class Runtime:
        def __init__(self):
            self.polls = 0
        def get_operation(self, operation_id, **kwargs):
            self.polls += 1
            return {"operation_id": operation_id, "state": "completed", "output": "E: 503 Forwarding failure",
                    "exit_code": 100, "started_at": 1.0, "finished_at": 2.0}

    runtime = Runtime()
    manager = SandboxManager(command_runner=lambda *a, **k: None)
    manager._uses_default_runner = True
    manager.get_runtime = lambda cid: runtime
    manager._proxy_manager.ensure = lambda network, force=False: {"state": "ready"}
    manager._probe_proxy_from_sandbox = lambda runtime: {"state": "online", "external_internet": True}
    result = manager.get_operation("conversation", "operation")
    assert result["exit_code"] == 100
    assert result["proxy_recovery"]["recovered"] is True
    assert result["proxy_recovery"]["command_retried"] is False
    assert runtime.polls == 1


def test_non_network_command_failure_does_not_trigger_proxy_recovery():
    from agent.sandbox_manager import SandboxManager

    class Runtime:
        def get_operation(self, operation_id, **kwargs):
            return {"operation_id": operation_id, "state": "completed", "output": "syntax error",
                    "exit_code": 2}

    manager = SandboxManager(command_runner=lambda *a, **k: None)
    manager._uses_default_runner = True
    manager.get_runtime = lambda cid: Runtime()
    manager._proxy_manager.ensure = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected recovery"))
    result = manager.get_operation("conversation", "operation")
    assert "proxy_recovery" not in result


def test_terminal_internal_metadata_is_hidden_from_model_but_kept_for_acceptance():
    import json
    from agent.core import _build_tool_messages
    payload = {
        "ok": True,
        "result": {
            "_terminal_text": "created file\\n",
            "_terminal_output": "created file\\n",
            "_terminal_command": "touch result.txt",
            "operation_id": "internal-op",
            "state": "completed",
            "exit_code": 0,
        },
    }
    message = _build_tool_messages("terminal", json.dumps(payload))[0]
    assert message["content"] == "created file\\n"
    assert "operation_id" not in message["content"]
    assert message["_localagent_internal"]["operation_id"] == "internal-op"
    assert message["_localagent_internal"]["exit_code"] == 0
