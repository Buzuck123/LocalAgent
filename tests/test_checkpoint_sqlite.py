import json
from pathlib import Path
from types import SimpleNamespace

import agent.core as core


def test_sqlite_checkpoint_survives_agent_restart_without_replaying_tool(
    tmp_path, monkeypatch
):
    # Изолируем базу от рабочих данных приложения.
    database_path = tmp_path / "integration.sqlite3"

    monkeypatch.setattr(
        core.db,
        "settings",
        SimpleNamespace(database_path=database_path),
    )
    monkeypatch.setattr(
        core,
        "settings",
        SimpleNamespace(max_tool_rounds=3, auto_resume_tasks=False),
    )

    # Настоящая SQLite: схема, соединения и CRUD-функции проекта.
    core.db.init_db()
    conversation = core.db.create_conversation("Integration test")
    task = core.db.create_task(
        conversation["id"],
        "Создай тестовый файл",
    )

    # Имитируем сохранённое состояние процесса, прерванного во время инструмента.
    checkpoint = {
        "version": 1,
        "round": 0,
        "phase": "tool_running",
        "messages": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "Создай тестовый файл"},
        ],
        "pending_calls": [
            {
                "name": "write_file",
                "arguments": {
                    "path": "result.txt",
                    "content": "test",
                },
            }
        ],
        "next_tool_index": 0,
        "current_tool": {
            "index": 0,
            "name": "write_file",
            "arguments": {
                "path": "result.txt",
                "content": "test",
            },
        },
    }

    core.db.update_task(
        task["id"],
        checkpoint_json=json.dumps(checkpoint, ensure_ascii=False),
        status="running",
    )

    # Проверяем сохранность после повторного чтения из SQLite.
    saved_task = core.db.get_task(task["id"])
    saved_checkpoint = json.loads(saved_task["checkpoint_json"])

    assert saved_checkpoint["phase"] == "tool_running"
    assert saved_checkpoint["current_tool"]["name"] == "write_file"

    # Новый экземпляр агента имитирует восстановление после перезапуска.
    dispatched = []
    monkeypatch.setattr(
        core,
        "dispatch_tool",
        lambda name, args: dispatched.append((name, args)),
    )

    agent = core.Agent()
    agent._run_task(task["id"], core.TaskControl())

    recovered_task = core.db.get_task(task["id"])

    assert dispatched == []
    assert recovered_task["status"] == "interrupted"
    assert "Повторный запуск запрещён" in recovered_task["error"]

    # Проверяем, что исходная контрольная точка осталась в базе.
    final_checkpoint = json.loads(recovered_task["checkpoint_json"])
    assert final_checkpoint["phase"] == "tool_running"
    assert final_checkpoint["current_tool"]["name"] == "write_file"


def test_sqlite_resume_uses_saved_tool_result_without_reexecution(
    tmp_path, monkeypatch
):
    import threading

    database_path = tmp_path / "resume.sqlite3"

    monkeypatch.setattr(
        core.db,
        "settings",
        SimpleNamespace(database_path=database_path),
    )
    monkeypatch.setattr(
        core,
        "settings",
        SimpleNamespace(max_tool_rounds=3, auto_resume_tasks=False),
    )

    core.db.init_db()
    conversation = core.db.create_conversation("Resume integration test")
    task = core.db.create_task(
        conversation["id"],
        "Создай тестовый файл",
    )

    # Моделируем инструмент, который уже выполнился.
    # Его результат и сообщение tool записаны в контрольную точку.
    tool_result = {
        "ok": True,
        "result": {"path": "result.txt", "bytes_written": 4},
    }
    checkpoint = {
        "version": 1,
        "round": 0,
        "phase": "tools_pending",
        "messages": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "Создай тестовый файл"},
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
            {
                "role": "tool",
                "tool_name": "write_file",
                "content": json.dumps(tool_result, ensure_ascii=False),
            },
        ],
        "pending_calls": [
            {
                "name": "write_file",
                "arguments": {
                    "path": "result.txt",
                    "content": "test",
                },
            }
        ],
        "next_tool_index": 1,
    }

    core.db.update_task(
        task["id"],
        checkpoint_json=json.dumps(checkpoint, ensure_ascii=False),
        status="interrupted",
    )

    # Если инструмент будет вызван повторно, тест это обнаружит.
    dispatched = []

    def forbidden_dispatch(name, args):
        dispatched.append((name, args))
        raise AssertionError("Инструмент нельзя выполнять повторно")

    monkeypatch.setattr(core, "dispatch_tool", forbidden_dispatch)

    # Модель получает сохранённый результат и возвращает финальный ответ.
    chat_inputs = []

    class FakeClient:
        def chat(self, messages, tools):
            chat_inputs.append(messages)
            assert any(
                message.get("role") == "tool"
                and json.loads(message["content"]) == tool_result
                for message in messages
            )
            return {
                "message": {
                    "role": "assistant",
                    "content": "Файл создан ранее.",
                }
            }

    monkeypatch.setattr(core, "client", FakeClient())

    agent = core.Agent.__new__(core.Agent)
    agent.controls = {}
    agent.threads = {}
    agent._lock = threading.RLock()

    agent._run_task(task["id"], core.TaskControl())

    recovered_task = core.db.get_task(task["id"])

    assert dispatched == []
    assert len(chat_inputs) == 1
    assert recovered_task["status"] == "completed"
    assert recovered_task["result"] == "Файл создан ранее."
    assert json.loads(recovered_task["checkpoint_json"])["phase"] == "completed"


def test_checkpoint_survives_separate_python_processes(tmp_path):
    import os
    import subprocess
    import sys

    database_path = tmp_path / "process_restart.sqlite3"
    workspace_path = tmp_path / "workspace"

    env = os.environ.copy()
    env["DATABASE_PATH"] = str(database_path)
    env["WORKSPACE_PATH"] = str(workspace_path)
    env["AUTO_RESUME_TASKS"] = "false"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.environ.get("PYTHONPATH", ""), str(Path(__file__).resolve().parents[1])]))

    # Первый процесс создаёт задачу и сохраняет состояние перед инструментом.
    create_script = r'''
import json
from agent import db

db.init_db()
conversation = db.create_conversation("Process restart test")
task = db.create_task(conversation["id"], "Создай тестовый файл")

checkpoint = {
    "version": 1,
    "round": 0,
    "phase": "tool_running",
    "messages": [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "Создай тестовый файл"}
    ],
    "pending_calls": [{
        "name": "write_file",
        "arguments": {"path": "result.txt", "content": "test"}
    }],
    "next_tool_index": 0,
    "current_tool": {
        "index": 0,
        "name": "write_file",
        "arguments": {"path": "result.txt", "content": "test"}
    }
}

db.update_task(
    task["id"],
    status="running",
    checkpoint_json=json.dumps(checkpoint, ensure_ascii=False),
)
print(task["id"])
'''

    first = subprocess.run(
        [sys.executable, "-c", create_script],
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert first.returncode == 0, (
        f"Первый процесс завершился с ошибкой:\n{first.stdout}\n{first.stderr}"
    )

    task_id = first.stdout.strip().splitlines()[-1]
    assert task_id

    # Второй, независимый процесс открывает ту же базу и восстанавливает задачу.
    recover_script = r'''
import json
import sys
from agent import db
import agent.core as core

task_id = sys.argv[1]
db.init_db()

task = db.get_task(task_id)
assert task is not None, "Задача не найдена после перезапуска процесса"

checkpoint = json.loads(task["checkpoint_json"])
assert checkpoint["phase"] == "tool_running"

calls = []
core.dispatch_tool = lambda name, args: calls.append((name, args))

agent = core.Agent()
agent._run_task(task_id, core.TaskControl())

recovered = db.get_task(task_id)

assert calls == [], f"Инструмент был повторно вызван: {calls}"
assert recovered["status"] == "interrupted", recovered
assert "Повторный запуск запрещён" in recovered["error"], recovered
assert json.loads(recovered["checkpoint_json"])["phase"] == "tool_running"

print("PROCESS_RESTART_RECOVERY_OK")
'''

    second = subprocess.run(
        [sys.executable, "-c", recover_script, task_id],
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert second.returncode == 0, (
        f"Второй процесс завершился с ошибкой:\n{second.stdout}\n{second.stderr}"
    )
    assert "PROCESS_RESTART_RECOVERY_OK" in second.stdout


def test_tool_running_recovers_completed_write_from_journal(
    tmp_path, monkeypatch
):
    """Восстановление после записи в журнал, но до обновления checkpoint."""
    import json
    import threading
    import uuid
    from types import SimpleNamespace

    from agent import tools as agent_tools

    database_path = tmp_path / "crash_boundary.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setattr(
        core.db,
        "settings",
        SimpleNamespace(database_path=database_path),
    )
    monkeypatch.setattr(
        core,
        "settings",
        SimpleNamespace(
            max_tool_rounds=3,
            auto_resume_tasks=False,
            workspace_path=workspace.resolve(),
        ),
    )
    monkeypatch.setattr(
        agent_tools,
        "settings",
        SimpleNamespace(workspace_path=workspace.resolve()),
    )

    core.db.init_db()
    conversation = core.db.create_conversation("Crash boundary test")
    task = core.db.create_task(conversation["id"], "Создай result.txt")

    operation_id = str(uuid.uuid4())
    arguments = {
        "path": "result.txt",
        "content": "durable result",
        "overwrite": False,
    }

    # До вызова инструмента журнал фиксирует намерение.
    core.db.create_tool_operation(
        operation_id,
        task["id"],
        "write_file",
    )

    # Имитируем успешный вызов инструмента до аварии.
    actual_result = agent_tools.dispatch_tool("write_file", arguments)

    # Результат уже долговечно сохранён в журнале.
    core.db.complete_tool_operation(operation_id, actual_result)

    target = workspace / "result.txt"
    assert target.read_text(encoding="utf-8") == "durable result"
    assert core.db.get_tool_operation(operation_id)["status"] == "completed"

    # А контрольная точка осталась в состоянии до получения результата.
    checkpoint = {
        "version": 1,
        "round": 0,
        "phase": "tool_running",
        "messages": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "Создай result.txt"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "write_file",
                        "arguments": arguments,
                    }
                }],
            },
        ],
        "pending_calls": [{
            "name": "write_file",
            "arguments": arguments,
        }],
        "next_tool_index": 0,
        "current_tool": {
            "name": "write_file",
            "index": 0,
            "operation_id": operation_id,
        },
    }
    core.db.update_task(
        task["id"],
        checkpoint_json=json.dumps(checkpoint, ensure_ascii=False),
        status="interrupted",
    )

    # Повторный вызов инструмента запрещён: восстановление должно
    # использовать только подтверждённый результат из журнала.
    dispatched = []

    def forbidden_dispatch(name, args):
        dispatched.append((name, args))
        raise AssertionError("write_file нельзя выполнять повторно")

    monkeypatch.setattr(core, "dispatch_tool", forbidden_dispatch)

    chat_inputs = []

    class FakeClient:
        def chat(self, messages, tools):
            chat_inputs.append(messages)

            recovered_messages = [
                message
                for message in messages
                if message.get("role") == "tool"
                and message.get("tool_name") == "write_file"
            ]
            assert len(recovered_messages) == 1

            content = json.loads(recovered_messages[0]["content"])
            assert content == {"ok": True, "result": actual_result}

            return {
                "message": {
                    "role": "assistant",
                    "content": "Файл создан, результат восстановлен.",
                }
            }

    monkeypatch.setattr(core, "client", FakeClient())

    agent = core.Agent.__new__(core.Agent)
    agent.controls = {}
    agent.threads = {}
    agent._lock = threading.RLock()

    agent._run_task(task["id"], core.TaskControl())

    recovered_task = core.db.get_task(task["id"])
    final_checkpoint = json.loads(recovered_task["checkpoint_json"])

    assert dispatched == []
    assert len(chat_inputs) == 1
    assert target.read_text(encoding="utf-8") == "durable result"
    assert recovered_task["status"] == "completed"
    assert recovered_task["result"] == "Файл создан, результат восстановлен."
    assert final_checkpoint["phase"] == "completed"
    assert "current_tool" not in final_checkpoint
