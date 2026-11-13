import json
import threading
from types import SimpleNamespace

import agent.core as core


def make_agent():
    agent = core.Agent.__new__(core.Agent)
    agent._lock = threading.RLock()
    agent.controls = {}
    agent.threads = {}
    return agent


def test_ambiguous_tool_is_not_repeated(monkeypatch):
    checkpoint = {
        "version": 1,
        "round": 0,
        "phase": "tool_running",
        "messages": [{"role": "user", "content": "Создай файл"}],
        "pending_calls": [{"name": "write_file", "arguments": {
            "path": "result.txt", "content": "test"
        }}],
        "next_tool_index": 0,
        "current_tool": {
            "index": 0,
            "name": "write_file",
            "arguments": {"path": "result.txt", "content": "test"},
        },
    }
    task = {
        "id": "task-ambiguous",
        "conversation_id": "conversation-1",
        "user_message": "Создай файл",
        "checkpoint_json": json.dumps(checkpoint),
    }
    updates = []
    dispatched = []

    monkeypatch.setattr(core.db, "get_task", lambda task_id: task)
    monkeypatch.setattr(
        core.db,
        "update_task",
        lambda task_id, **fields: updates.append(fields),
    )
    monkeypatch.setattr(
        core,
        "dispatch_tool",
        lambda name, args: dispatched.append((name, args)),
    )

    agent = make_agent()
    agent._run_task(task["id"], core.TaskControl())

    assert dispatched == []
    assert any(u.get("status") == "interrupted" for u in updates)
    assert any(
        "Повторный запуск запрещён" in u.get("error", "")
        for u in updates
    )


def test_pending_tool_resumes_and_continues(monkeypatch):
    checkpoint = {
        "version": 1,
        "round": 0,
        "phase": "tools_pending",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Создай файл"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "terminal",
                        "arguments": {
                            "command": "printf test > result.txt",
                        },
                    }
                }],
            },
        ],
        "pending_calls": [{
            "name": "terminal",
            "arguments": {"command": "printf test > result.txt"},
        }],
        "next_tool_index": 0,
    }
    task = {
        "id": "task-pending",
        "conversation_id": "conversation-1",
        "user_message": "Создай файл",
        "checkpoint_json": json.dumps(checkpoint),
    }
    history = [{"role": "user", "content": "Создай файл"}]
    updates = []
    dispatched = []
    chat_calls = []

    monkeypatch.setattr(core.db, "get_task", lambda task_id: task)
    monkeypatch.setattr(
        core.db,
        "update_task",
        lambda task_id, **fields: updates.append(fields),
    )

    monkeypatch.setattr(
        core.db,
        "create_tool_operation",
        lambda operation_id, task_id, tool_name, arguments=None: {
            "operation_id": operation_id,
            "task_id": task_id,
            "tool_name": tool_name,
            "status": "running",
            "result_json": None,
        },
    )
    monkeypatch.setattr(
        core.db,
        "complete_tool_operation",
        lambda operation_id, result: {
            "operation_id": operation_id,
            "status": "completed",
            "result_json": json.dumps(result, ensure_ascii=False),
        },
    )

    def add_message(cid, role, content):
        history.append({"role": role, "content": content})

    monkeypatch.setattr(core.db, "add_message", add_message)
    monkeypatch.setattr(
        core.db,
        "get_messages",
        lambda cid, limit=100: history[-limit:],
    )

    def fake_dispatch(name, args, **kwargs):
        # Sandbox dispatch receives conversation_id and idempotency_key
        # from the current terminal runtime adapter.
        dispatched.append((name, args))
        return {"command": args["command"], "exit_code": 0, "timed_out": False}

    monkeypatch.setattr(core, "dispatch_tool", fake_dispatch)

    class FakeClient:
        def chat(self, messages, tools):
            chat_calls.append(messages)
            return {
                "message": {
                    "role": "assistant",
                    "content": "Файл создан.",
                }
            }

    monkeypatch.setattr(core, "client", FakeClient())
    monkeypatch.setattr(core, "settings", SimpleNamespace(max_tool_rounds=3, auto_resume_tasks=False))

    agent = make_agent()
    agent._run_task(task["id"], core.TaskControl())

    assert len(dispatched) == 1
    assert dispatched[0][0] == "terminal"
    assert dispatched[0][1] == {"command": "printf test > result.txt"}
    assert len(chat_calls) == 1
    assert any(u.get("status") == "completed" for u in updates)
    assert any(
        u.get("result") == "Файл создан."
        for u in updates
    )
    assert any(
        m["role"] == "tool"
        and json.loads(m["content"])["ok"] is True
        for m in chat_calls[0]
    )


def test_legacy_checkpoint_with_tool_call_is_not_replayed(monkeypatch):
    checkpoint = {
        "round": 0,
        "messages": [
            {"role": "user", "content": "Создай файл"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "write_file",
                        "arguments": {
                            "path": "result.txt",
                            "content": "test",
                        },
                    }
                }],
            },
        ],
    }
    task = {
        "id": "task-legacy",
        "conversation_id": "conversation-1",
        "user_message": "Создай файл",
        "checkpoint_json": json.dumps(checkpoint),
    }
    updates = []
    dispatched = []
    chat_calls = []

    monkeypatch.setattr(core.db, "get_task", lambda task_id: task)
    monkeypatch.setattr(
        core.db, "update_task",
        lambda task_id, **fields: updates.append(fields),
    )
    monkeypatch.setattr(
        core, "dispatch_tool",
        lambda name, args: dispatched.append((name, args)),
    )

    class FakeClient:
        def chat(self, messages, tools):
            chat_calls.append(messages)
            return {"message": {"role": "assistant", "content": "Продолжено"}}

    monkeypatch.setattr(core, "client", FakeClient())

    agent = make_agent()
    agent._run_task(task["id"], core.TaskControl())

    assert dispatched == []
    assert chat_calls == []
    assert any(update.get("status") == "interrupted" for update in updates)
    assert any(
        "Старая контрольная точка" in update.get("error", "")
        for update in updates
    )


def test_legacy_checkpoint_without_tool_calls_resumes(monkeypatch):
    checkpoint = {
        "round": 1,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Продолжи"},
        ],
    }
    task = {
        "id": "task-legacy-safe",
        "conversation_id": "conversation-1",
        "user_message": "Продолжи",
        "checkpoint_json": json.dumps(checkpoint),
    }
    updates = []
    history = [{"role": "user", "content": "Продолжи"}]

    monkeypatch.setattr(core.db, "get_task", lambda task_id: task)
    monkeypatch.setattr(
        core.db, "update_task",
        lambda task_id, **fields: updates.append(fields),
    )
    monkeypatch.setattr(
        core.db, "get_messages",
        lambda cid, limit=100: history[-limit:],
    )
    monkeypatch.setattr(
        core.db, "add_message",
        lambda cid, role, content: history.append(
            {"role": role, "content": content}
        ),
    )

    class FakeClient:
        def chat(self, messages, tools):
            return {
                "message": {
                    "role": "assistant",
                    "content": "Продолжено без повторного запроса.",
                }
            }

    monkeypatch.setattr(core, "client", FakeClient())
    monkeypatch.setattr(
        core, "settings",
        SimpleNamespace(max_tool_rounds=3, auto_resume_tasks=False),
    )

    agent = make_agent()
    agent._run_task(task["id"], core.TaskControl())

    assert any(u.get("status") == "completed" for u in updates)
    assert sum(m["role"] == "user" for m in history) == 1
    assert any(
        u.get("result") == "Продолжено без повторного запроса."
        for u in updates
    )


def test_checkpoint_resume_does_not_duplicate_user_message(monkeypatch):
    checkpoint = {
        "version": 1,
        "round": 0,
        "phase": "loop",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Продолжи"},
        ],
        "pending_calls": [],
        "next_tool_index": 0,
    }
    task = {
        "id": "task-no-duplicate",
        "conversation_id": "conversation-1",
        "user_message": "Продолжи",
        "checkpoint_json": json.dumps(checkpoint),
    }
    updates = []
    history = [{"role": "user", "content": "Продолжи"}]

    monkeypatch.setattr(core.db, "get_task", lambda task_id: task)
    monkeypatch.setattr(
        core.db, "update_task",
        lambda task_id, **fields: updates.append(fields),
    )
    monkeypatch.setattr(
        core.db, "get_messages",
        lambda cid, limit=100: history[-limit:],
    )
    monkeypatch.setattr(
        core.db, "add_message",
        lambda cid, role, content: history.append(
            {"role": role, "content": content}
        ),
    )

    class FakeClient:
        def chat(self, messages, tools):
            return {
                "message": {
                    "role": "assistant",
                    "content": "Готово.",
                }
            }

    monkeypatch.setattr(core, "client", FakeClient())
    monkeypatch.setattr(
        core, "settings",
        SimpleNamespace(max_tool_rounds=3, auto_resume_tasks=False),
    )

    agent = make_agent()
    agent._run_task(task["id"], core.TaskControl())

    assert sum(m["role"] == "user" for m in history) == 1
    assert any(u.get("status") == "completed" for u in updates)


def test_journal_running_operation_is_not_repeated(monkeypatch):
    checkpoint = {
        "version": 1,
        "round": 0,
        "phase": "tool_running",
        "messages": [{"role": "user", "content": "Создай файл"}],
        "pending_calls": [{
            "name": "write_file",
            "arguments": {"path": "result.txt", "content": "test"},
        }],
        "next_tool_index": 0,
        "current_tool": {
            "index": 0,
            "name": "write_file",
            "arguments": {"path": "result.txt", "content": "test"},
            "operation_id": "op-running-test",
        },
    }
    task = {
        "id": "task-journal-running",
        "conversation_id": "conversation-1",
        "user_message": "Создай файл",
        "checkpoint_json": json.dumps(checkpoint),
    }
    updates = []
    dispatched = []

    monkeypatch.setattr(core.db, "get_task", lambda task_id: task)
    monkeypatch.setattr(
        core.db,
        "update_task",
        lambda task_id, **fields: updates.append(fields),
    )
    monkeypatch.setattr(
        core.db,
        "get_tool_operation",
        lambda operation_id: {
            "operation_id": operation_id,
            "task_id": task["id"],
            "tool_name": "write_file",
            "status": "running",
            "result_json": None,
        },
    )
    monkeypatch.setattr(
        core,
        "dispatch_tool",
        lambda name, args: dispatched.append((name, args)),
    )

    make_agent()._run_task(task["id"], core.TaskControl())

    assert dispatched == []
    assert any(u.get("status") == "interrupted" for u in updates)


def test_journal_running_operation_is_not_repeated(monkeypatch):
    checkpoint = {
        "version": 1,
        "round": 0,
        "phase": "tool_running",
        "messages": [{"role": "user", "content": "Создай файл"}],
        "pending_calls": [{
            "name": "write_file",
            "arguments": {"path": "result.txt", "content": "test"},
        }],
        "next_tool_index": 0,
        "current_tool": {
            "index": 0,
            "name": "write_file",
            "arguments": {"path": "result.txt", "content": "test"},
            "operation_id": "op-running-test",
        },
    }
    task = {
        "id": "task-journal-running",
        "conversation_id": "conversation-1",
        "user_message": "Создай файл",
        "checkpoint_json": json.dumps(checkpoint),
    }
    updates = []
    dispatched = []

    monkeypatch.setattr(core.db, "get_task", lambda task_id: task)
    monkeypatch.setattr(
        core.db,
        "update_task",
        lambda task_id, **fields: updates.append(fields),
    )
    monkeypatch.setattr(
        core.db,
        "get_tool_operation",
        lambda operation_id: {
            "operation_id": operation_id,
            "task_id": task["id"],
            "tool_name": "write_file",
            "status": "running",
            "result_json": None,
        },
    )
    monkeypatch.setattr(
        core,
        "dispatch_tool",
        lambda name, args: dispatched.append((name, args)),
    )

    make_agent()._run_task(task["id"], core.TaskControl())

    assert dispatched == []
    assert any(u.get("status") == "interrupted" for u in updates)


def test_completed_journal_operation_resumes_without_dispatch(monkeypatch):
    expected_result = {
        "path": "result.txt",
        "bytes_written": 4,
        "overwritten": False,
    }
    checkpoint = {
        "version": 1,
        "round": 0,
        "phase": "tool_running",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Создай файл"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "write_file",
                        "arguments": {
                            "path": "result.txt",
                            "content": "test",
                        },
                    }
                }],
            },
        ],
        "pending_calls": [{
            "name": "write_file",
            "arguments": {"path": "result.txt", "content": "test"},
        }],
        "next_tool_index": 0,
        "current_tool": {
            "index": 0,
            "name": "write_file",
            "arguments": {"path": "result.txt", "content": "test"},
            "operation_id": "op-completed-test",
        },
    }
    task = {
        "id": "task-journal-completed",
        "conversation_id": "conversation-1",
        "user_message": "Создай файл",
        "checkpoint_json": json.dumps(checkpoint),
    }
    updates = []
    dispatched = []
    chat_messages = []
    history = [{"role": "user", "content": "Создай файл"}]

    monkeypatch.setattr(
        core.db,
        "add_message",
        lambda cid, role, content: history.append({
            "role": role,
            "content": content,
        }),
    )
    monkeypatch.setattr(
        core.db,
        "get_messages",
        lambda cid, limit=100: history[-limit:],
    )

    monkeypatch.setattr(core.db, "get_task", lambda task_id: task)
    monkeypatch.setattr(
        core.db,
        "update_task",
        lambda task_id, **fields: updates.append(fields),
    )
    monkeypatch.setattr(
        core.db,
        "get_tool_operation",
        lambda operation_id: {
            "operation_id": operation_id,
            "task_id": task["id"],
            "tool_name": "write_file",
            "status": "completed",
            "result_json": json.dumps(expected_result),
        },
    )
    monkeypatch.setattr(
        core,
        "dispatch_tool",
        lambda name, args: dispatched.append((name, args)),
    )

    class FakeClient:
        def chat(self, messages, tools):
            chat_messages.append(messages)
            return {
                "message": {
                    "role": "assistant",
                    "content": "Файл создан.",
                }
            }

    monkeypatch.setattr(core, "client", FakeClient())
    monkeypatch.setattr(
        core,
        "settings",
        SimpleNamespace(max_tool_rounds=3, auto_resume_tasks=False),
    )

    make_agent()._run_task(task["id"], core.TaskControl())

    assert dispatched == []
    assert chat_messages, "Агент должен продолжить работу после восстановления"
    tool_messages = [
        message for message in chat_messages[0]
        if message.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    restored_content = json.loads(tool_messages[0]["content"])
    assert restored_content == {"ok": True, "result": expected_result}
    assert not any(
        update.get("status") == "interrupted"
        for update in updates
    )


def test_receipt_failure_does_not_advance_checkpoint_or_repeat_tool(monkeypatch):
    checkpoint = {
        "version": 1,
        "round": 0,
        "phase": "tools_pending",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Создай файл"},
        ],
        "pending_calls": [{
            "name": "terminal",
            "arguments": {"command": "printf test > result.txt"},
        }],
        "next_tool_index": 0,
    }
    task = {
        "id": "task-receipt-failure",
        "conversation_id": "conversation-1",
        "user_message": "Создай файл",
        "checkpoint_json": json.dumps(checkpoint),
    }
    updates = []
    dispatched = []
    messages = []

    monkeypatch.setattr(core.db, "get_task", lambda task_id: task)
    monkeypatch.setattr(
        core.db,
        "update_task",
        lambda task_id, **fields: updates.append(fields),
    )
    monkeypatch.setattr(
        core.db,
        "create_tool_operation",
        lambda **kwargs: {
            "operation_id": kwargs["operation_id"],
            "task_id": kwargs["task_id"],
            "tool_name": kwargs["tool_name"],
            "status": "running",
            "result_json": None,
        },
    )

    def dispatch(name, args, **kwargs):
        # Match the current sandbox dispatch contract.
        dispatched.append((name, args))
        return {
            "path": "result.txt",
            "bytes_written": 4,
            "overwritten": False,
        }

    monkeypatch.setattr(core, "dispatch_tool", dispatch)

    def fail_receipt(operation_id, result):
        raise OSError("simulated receipt write failure")

    monkeypatch.setattr(core.db, "complete_tool_operation", fail_receipt)
    monkeypatch.setattr(
        core.db,
        "add_message",
        lambda cid, role, content: messages.append({
            "role": role,
            "content": content,
        }),
    )

    make_agent()._run_task(task["id"], core.TaskControl())

    assert len(dispatched) == 1
    assert dispatched[0][0] == "terminal"
    assert dispatched[0][1] == {"command": "printf test > result.txt"}
    assert any(
        update.get("status") == "interrupted"
        and "simulated receipt write failure" in update.get("error", "")
        for update in updates
    )

    saved_checkpoints = [
        json.loads(update["checkpoint_json"])
        for update in updates
        if "checkpoint_json" in update
    ]
    assert saved_checkpoints
    last_checkpoint = saved_checkpoints[-1]
    assert last_checkpoint["phase"] == "tool_running"
    assert last_checkpoint["current_tool"]["name"] == "terminal"
    assert not any(
        message.get("role") == "tool"
        and message.get("tool_name") == "terminal"
        for message in last_checkpoint["messages"]
    )
