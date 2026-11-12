import json
import subprocess
from types import SimpleNamespace

import agent.core as core
from agent.planning import create_plan, set_step_status


def test_resume_continues_from_saved_in_progress_step(tmp_path, monkeypatch):
    database_path = tmp_path / "recovery.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Изолируем тест от рабочей базы и настоящей директории файлов.
    monkeypatch.setattr(
        core.db,
        "settings",
        SimpleNamespace(database_path=database_path),
    )
    monkeypatch.setattr(
        core,
        "settings",
        SimpleNamespace(
            max_tool_rounds=10,
            auto_resume_tasks=False,
            workspace_path=workspace,
        ),
    )

    core.db.init_db()
    conversation = core.db.create_conversation("Plan recovery test")
    task = core.db.create_task(
        conversation["id"],
        "Создай два файла последовательно",
    )

    # Первый этап уже выполнен и подтверждён.
    plan = create_plan(
        "Создай два файла последовательно",
        [
            {
                "id": "step_1",
                "description": "Создать первый файл",
                "expected_result": "first.txt содержит FIRST",
                "acceptance_check": {
                    "type": "file_contains",
                    "path": "first.txt",
                    "text": "FIRST",
                },
            },
            {
                "id": "step_2",
                "description": "Создать второй файл",
                "expected_result": "second.txt содержит SECOND",
                "acceptance_check": {
                    "type": "file_contains",
                    "path": "second.txt",
                    "text": "SECOND",
                },
            },
        ],
    )

    first_file = workspace / "first.txt"
    first_file.write_text("FIRST", encoding="utf-8")

    set_step_status(
        plan,
        "step_1",
        "completed",
        verification="Проверено тестом: first.txt содержит FIRST",
    )

    assert plan["steps"][0]["status"] == "completed"
    assert plan["steps"][1]["status"] == "in_progress"

    # Имитируем прерывание: checkpoint сохранён после первого этапа,
    # до запроса модели о следующем инструменте.
    checkpoint = {
        "version": 1,
        "round": 0,
        "phase": "loop",
        "messages": [
            {
                "role": "system",
                "content": core.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": task["user_message"],
            },
        ],
        "pending_calls": [],
        "next_tool_index": 0,
        "plan": plan,
    }

    core.db.update_task(
        task["id"],
        status="interrupted",
        checkpoint_json=json.dumps(checkpoint, ensure_ascii=False),
    )

    # Убеждаемся, что состояние действительно прочитано из SQLite.
    saved_task = core.db.get_task(task["id"])
    saved_checkpoint = json.loads(saved_task["checkpoint_json"])
    saved_plan = saved_checkpoint["plan"]

    assert saved_plan["steps"][0]["status"] == "completed"
    assert saved_plan["steps"][1]["status"] == "in_progress"

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1

            # При восстановлении планирование не должно запускаться заново.
            if self.calls == 1:
                names = {item["function"]["name"] for item in (tools or [])}
                assert "terminal" in names
                assert "web_search" in names
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "terminal",
                                    "arguments": {"command": "printf '%s' 'SECOND' > second.txt"},
                                }
                            }
                        ],
                    }
                }

            if self.calls == 2:
                return {"message": {"role": "assistant", "content": "", "tool_calls": [{
                    "function": {"name": "terminal", "arguments": {"command": "grep -F -- SECOND second.txt"}}
                }]}}
            assert self.calls == 3
            return {"message": {"role": "assistant", "content": "Второй файл создан и проверен."}}

    fake_client = FakeClient()
    monkeypatch.setattr(core, "client", fake_client)

    dispatched = []

    def fake_dispatch(name, args, **kwargs):
        dispatched.append((name, dict(args)))
        assert name == "terminal"
        result = subprocess.run(["bash", "-lc", args["command"]], cwd=workspace,
                                capture_output=True, text=True, check=False)
        return {"command": args["command"], "stdout": result.stdout, "stderr": result.stderr,
                "exit_code": result.returncode, "timed_out": False}

    monkeypatch.setattr(core, "dispatch_tool", fake_dispatch)

    agent = core.Agent()
    agent._run_task(task["id"], core.TaskControl())

    recovered_task = core.db.get_task(task["id"])
    recovered_checkpoint = json.loads(recovered_task["checkpoint_json"])
    recovered_plan = recovered_checkpoint["plan"]

    # Повторно выполнился только второй этап.
    assert [args["command"] for _, args in dispatched] == [
        "printf '%s' 'SECOND' > second.txt", "grep -F -- SECOND second.txt"
    ]

    # Содержимое обоих файлов соответствует ожидаемому результату.
    assert first_file.read_text(encoding="utf-8") == "FIRST"
    assert (workspace / "second.txt").read_text(encoding="utf-8") == "SECOND"

    # План и задача завершены, подтверждения сохранены.
    assert recovered_task["status"] == "completed"
    assert recovered_task["result"] == "Второй файл создан и проверен."
    assert recovered_checkpoint["phase"] == "completed"
    assert recovered_plan["status"] == "completed"
    assert [step["status"] for step in recovered_plan["steps"]] == [
        "completed",
        "completed",
    ]
    assert all(
        isinstance(step.get("verification"), str)
        and step["verification"].strip()
        for step in recovered_plan["steps"]
    )

    # Продолжение, терминальная проверка и финальный ответ; план заново не создавался.
    assert fake_client.calls == 3
