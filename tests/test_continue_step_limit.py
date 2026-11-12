import json
from types import SimpleNamespace

import agent.core as core


def test_repeated_continue_step_requests_replan_then_pauses_if_reviewer_refuses(tmp_path, monkeypatch):
    database_path = tmp_path / "continue_limit.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setattr(core.db, "settings", SimpleNamespace(database_path=database_path))
    monkeypatch.setattr(
        core,
        "settings",
        SimpleNamespace(
            max_tool_rounds=20,
            auto_resume_tasks=False,
            workspace_path=workspace,
        ),
    )
    core.db.init_db()
    conversation = core.db.create_conversation("Continue-step limit test")
    task = core.db.create_task(conversation["id"], "Подготовь короткий результат")

    plan_response = {"steps": [{
        "id": "step_1",
        "description": "Подготовить короткий результат",
        "expected_result": "Короткий результат подготовлен",
        "success_criteria": ["Есть текстовый результат"],
        "completion_mode": "response",
    }]}
    continue_response = {
        "decision": "continue_step",
        "reason": "Нужно продолжить работу",
        "task_prompt": "Попробуй ещё раз подготовить результат.",
    }

    class FakeClient:
        def __init__(self):
            self.review_calls = 0

        def chat(self, messages, tools=None):
            system = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict) and m.get("role") == "system")
            if "Create a concise execution plan" in system:
                return {"message": {"content": json.dumps(plan_response, ensure_ascii=False)}}
            if "Prepare a concrete execution instruction" in system:
                return {"message": {"content": json.dumps({
                    "task_prompt": "Подготовь короткий результат текстом."
                }, ensure_ascii=False)}}
            if "Review the current LocalAgent plan step" in system:
                self.review_calls += 1
                return {"message": {"content": json.dumps(continue_response, ensure_ascii=False)}}
            return {"message": {"role": "assistant", "content": "Продолжаю."}}

    fake = FakeClient()
    monkeypatch.setattr(core, "client", fake)

    core.Agent()._run_task(task["id"], core.TaskControl())

    saved = core.db.get_task(task["id"])
    checkpoint = json.loads(saved["checkpoint_json"])
    assert saved["status"] == "paused", saved
    assert "безрезультатных попыток/продолжений" in saved["result"]
    assert "не удалось получить альтернативный план" in saved["result"]
    assert "резервный планировщик" in saved["result"]
    assert checkpoint["step_attempts"]["step_1"] == 4
    assert checkpoint["plan"]["steps"][0]["status"] == "question"
    assert fake.review_calls == 7
