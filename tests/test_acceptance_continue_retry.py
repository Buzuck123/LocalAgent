import json
import subprocess
from types import SimpleNamespace

import agent.core as core


def test_failed_acceptance_check_forces_retry_even_if_reviewer_says_continue(tmp_path, monkeypatch):
    database_path = tmp_path / "acceptance_continue.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(core.db, "settings", SimpleNamespace(database_path=database_path))
    monkeypatch.setattr(core, "settings", SimpleNamespace(
        max_tool_rounds=15, auto_resume_tasks=False, workspace_path=workspace,
    ))
    core.db.init_db()
    conversation = core.db.create_conversation("Acceptance continue test")
    task = core.db.create_task(conversation["id"], "Разработай проект и создай файл с точным текстом")

    expected = "Проект: Planning Test\\nСтатус: выполнен"
    plan_response = {"steps": [{
        "id": "step_1",
        "description": "Создать task.txt с точным содержимым",
        "expected_result": "Файл содержит точный текст",
        "success_criteria": ["Файл содержит требуемый текст"],
        "acceptance_check": {"type": "file_contains", "path": "task.txt", "text": expected},
    }, {
        "id": "step_2",
        "description": "Подготовить итоговое сообщение",
        "expected_result": "Результат работы кратко описан",
        "completion_mode": "response",
        "success_criteria": ["Пользователь получил краткий итог"],
    }]}
    continue_response = {
        "decision": "continue_step", "reason": "Нужно продолжить",
        "task_prompt": "Продолжи работу.",
    }
    complete_response = {
        "decision": "complete_step", "reason": "Файл подтверждён",
        "verification": "Файл прочитан и подтверждён",
        "final_answer": "Файл создан и проверен.",
    }

    class FakeClient:
        def __init__(self):
            self.execution_calls = 0
            self.review_calls = 0

        def chat(self, messages, tools=None):
            system = messages[0].get("content", "") if messages else ""
            if "Create a concise execution plan" in system:
                return {"message": {"content": json.dumps(plan_response, ensure_ascii=False)}}
            if "Prepare a concrete execution instruction" in system:
                return {"message": {"content": json.dumps({"task_prompt": "Создай task.txt"}, ensure_ascii=False)}}
            if "Review the current LocalAgent plan step" in system:
                self.review_calls += 1
                decision = continue_response if self.review_calls == 1 else complete_response
                return {"message": {"content": json.dumps(decision, ensure_ascii=False)}}
            self.execution_calls += 1
            if self.execution_calls == 1:
                return {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {
                    "name": "terminal", "arguments": {"command": "printf '%s' 'wrong content' > task.txt"}
                }}]}}
            if self.execution_calls == 2:
                return {"message": {"role": "assistant", "content": "Файл почти готов."}}
            if self.execution_calls == 3:
                return {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {
                    "name": "terminal", "arguments": {"command": "printf '%s' '" + expected + "' > task.txt"}
                }}]}}
            if self.execution_calls == 4:
                return {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {
                    "name": "terminal", "arguments": {"command": "grep -F -- '" + expected + "' task.txt"}
                }}]}}
            return {"message": {"role": "assistant", "content": "Файл создан и проверен."}}

    fake = FakeClient()
    monkeypatch.setattr(core, "client", fake)

    def fake_dispatch(name, args, **kwargs):
        assert name == "terminal"
        result = subprocess.run(["bash", "-lc", args["command"]], cwd=workspace,
                                capture_output=True, text=True, check=False)
        return {"command": args["command"], "stdout": result.stdout, "stderr": result.stderr,
                "exit_code": result.returncode, "timed_out": False}

    monkeypatch.setattr(core, "dispatch_tool", fake_dispatch)
    core.Agent()._run_task(task["id"], core.TaskControl())

    saved = core.db.get_task(task["id"])
    checkpoint = json.loads(saved["checkpoint_json"])
    assert saved["status"] == "completed", saved
    assert (workspace / "task.txt").read_text(encoding="utf-8") == expected
    assert checkpoint["plan"]["steps"][0]["status"] == "completed"
    assert checkpoint["step_attempts"]["step_1"] == 1
