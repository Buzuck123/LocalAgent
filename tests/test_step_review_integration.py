import json
from types import SimpleNamespace

import agent.core as core


def test_model_reviews_step_result_before_task_completes(tmp_path, monkeypatch):
    database_path = tmp_path / "review.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setattr(
        core.db, "settings", SimpleNamespace(database_path=database_path)
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
    conversation = core.db.create_conversation("Step review test")
    task = core.db.create_task(conversation["id"], "Подготовь краткий анализ проекта в несколько этапов")

    plan_response = {"steps": [{
        "id": "step_1",
        "description": "Подготовить анализ",
        "expected_result": "Получен краткий анализ",
        "completion_mode": "response",
        "success_criteria": ["Анализ содержит вывод по запросу"],
    }, {
        "id": "step_2",
        "description": "Сформулировать итоговый вывод",
        "expected_result": "Итоговый вывод сформулирован",
        "completion_mode": "response",
        "success_criteria": ["Итоговый вывод дан пользователю"],
    }]}
    review_response = {
        "decision": "complete_step",
        "reason": "Результат соответствует критерию",
        "verification": "В ответе сформулирован вывод по запросу",
        "final_answer": "Анализ подготовлен.",
    }

    class FakeClient:
        def __init__(self):
            self.calls = []

        def chat(self, messages, tools=None):
            self.calls.append((messages, tools))
            if len(self.calls) == 1:
                assert tools is None
                return {"message": {"content": json.dumps(plan_response, ensure_ascii=False)}}
            if tools is None:
                return {"message": {"content": json.dumps(review_response, ensure_ascii=False)}}
            return {"message": {"role": "assistant", "content": "Вывод: задача решается таким способом."}}

    fake_client = FakeClient()
    monkeypatch.setattr(core, "client", fake_client)

    agent = core.Agent()
    agent._run_task(task["id"], core.TaskControl())

    saved_task = core.db.get_task(task["id"])
    checkpoint = json.loads(saved_task["checkpoint_json"])
    assert saved_task["status"] == "completed", saved_task.get("error") or saved_task.get("result") or saved_task
    assert saved_task["result"] == "Анализ подготовлен."
    assert checkpoint["plan"]["status"] == "completed"
    assert checkpoint["plan"]["steps"][0]["verification"] == review_response["verification"]
    assert checkpoint["last_step_decision"]["decision"] == "complete_step"
    review_calls = [
        index for index, (messages, _tools) in enumerate(fake_client.calls)
        if any(
            message.get("role") == "system"
            and "Review the current LocalAgent plan step" in str(message.get("content", ""))
            for message in messages
        )
    ]
    assert review_calls, "The model must review the step before task completion"


def test_plan_revision_preserves_completed_progress(tmp_path, monkeypatch):
    database_path = tmp_path / "revision.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        core.db, "settings", SimpleNamespace(database_path=database_path)
    )
    monkeypatch.setattr(
        core,
        "settings",
        SimpleNamespace(
            max_tool_rounds=15,
            auto_resume_tasks=False,
            workspace_path=workspace,
        ),
    )
    core.db.init_db()
    conversation = core.db.create_conversation("Plan revision test")
    task = core.db.create_task(conversation["id"], "Подготовь результат адаптивно")

    initial_plan = {"steps": [
        {
            "id": "step_1",
            "description": "Изучить исходные данные",
            "expected_result": "Сформулированы исходные выводы",
            "completion_mode": "response",
            "success_criteria": ["Есть исходные выводы"],
        },
        {
            "id": "step_2",
            "description": "Использовать первоначальный подход",
            "expected_result": "Получен результат первоначальным способом",
            "success_criteria": ["Результат получен"],
        },
    ]}

    class FakeClient:
        def __init__(self):
            self.review_index = 0
            self.execution_index = 0

        def chat(self, messages, tools=None):
            system = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict) and m.get("role") == "system")
            if "Create a concise execution plan" in system:
                return {"message": {"content": json.dumps(initial_plan, ensure_ascii=False)}}
            if "Prepare a concrete execution instruction" in system:
                return {"message": {"content": json.dumps({
                    "task_prompt": "Выполни текущий этап и оцени результат."
                }, ensure_ascii=False)}}
            if "Review the current LocalAgent plan step" in system:
                self.review_index += 1
                if self.review_index == 1:
                    decision = {
                        "decision": "complete_step",
                        "reason": "Исходные выводы получены",
                        "verification": "В результате записаны исходные выводы",
                    }
                elif self.review_index == 2:
                    decision = {
                        "decision": "revise_plan",
                        "reason": "Первоначальный подход не подходит",
                        "plan_update": {"steps": [{
                            "id": "step_3",
                            "description": "Применить альтернативный подход",
                            "expected_result": "Получен результат альтернативным способом",
                            "completion_mode": "response",
                            "success_criteria": ["Альтернативный результат проверен"],
                            "dependencies": ["step_1"],
                        }]},
                    }
                else:
                    decision = {
                        "decision": "complete_step",
                        "reason": "Альтернативный результат подтверждён",
                        "verification": "Получен и проверен альтернативный результат",
                        "final_answer": "Задача завершена альтернативным способом.",
                    }
                return {"message": {"content": json.dumps(decision, ensure_ascii=False)}}
            self.execution_index += 1
            text = (
                "Исходные выводы сформулированы."
                if self.execution_index == 1
                else "Первоначальный подход не сработал."
                if self.execution_index == 2
                else "Альтернативный результат получен."
            )
            return {"message": {"role": "assistant", "content": text}}

    monkeypatch.setattr(core, "client", FakeClient())
    core.Agent()._run_task(task["id"], core.TaskControl())

    saved_task = core.db.get_task(task["id"])
    checkpoint = json.loads(saved_task["checkpoint_json"])
    plan = checkpoint["plan"]
    assert saved_task["status"] == "completed", saved_task.get("error") or saved_task.get("result") or saved_task
    assert [step["id"] for step in plan["steps"]] == ["step_1", "adaptive_1_step_1"]
    assert plan["steps"][0]["status"] == "completed"
    assert "completion_mode" not in plan["steps"][0]
    assert plan["steps"][1]["status"] == "completed"
    assert plan["steps"][1]["dependencies"] == ["step_1"]
    assert "completion_mode" not in plan["steps"][1]
    assert plan["status"] == "completed"
    assert checkpoint["last_plan_revision_reason"] == "Первоначальный подход не подходит"
