import json
import subprocess
from types import SimpleNamespace

import agent.core as core
from agent.planning import verify_acceptance_check_state


def test_file_acceptance_check_uses_actual_workspace_state(tmp_path):
    check = {"type": "file_exists", "path": "reports/research.md"}
    assert verify_acceptance_check_state(check, tmp_path, []) is None

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "research.md").write_text("Architecture report", encoding="utf-8")
    proof = verify_acceptance_check_state(check, tmp_path, [])
    assert proof and "файл существует" in proof


def test_file_contains_acceptance_check_checks_actual_contents(tmp_path):
    target = tmp_path / "research.md"
    target.write_text("Architecture overview", encoding="utf-8")
    check = {"type": "file_contains", "path": "research.md", "text": "Architecture"}
    assert verify_acceptance_check_state(check, tmp_path, [])
    check["text"] = "does not exist"
    assert verify_acceptance_check_state(check, tmp_path, []) is None


def test_file_acceptance_check_rejects_path_escape(tmp_path):
    check = {"type": "file_exists", "path": "../outside.txt"}
    (tmp_path.parent / "outside.txt").write_text("outside", encoding="utf-8")
    assert verify_acceptance_check_state(check, tmp_path, []) is None


def test_python_acceptance_check_requires_successful_tool_receipt(tmp_path):
    check = {"type": "python_exit_code", "path": "experiment/check.py"}
    assert verify_acceptance_check_state(check, tmp_path, []) is None
    messages = [{
        "role": "tool",
        "tool_name": "run_python",
        "content": json.dumps({"ok": True, "result": {
            "script": "experiment/check.py", "exit_code": 0, "timed_out": False
        }}),
    }]
    assert verify_acceptance_check_state(check, tmp_path, messages)


def test_model_cannot_complete_file_step_without_creating_file(tmp_path, monkeypatch):
    database_path = tmp_path / "gate.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "reports").mkdir()

    monkeypatch.setattr(core.db, "settings", SimpleNamespace(database_path=database_path))
    monkeypatch.setattr(core, "settings", SimpleNamespace(
        max_tool_rounds=12,
        auto_resume_tasks=False,
        workspace_path=workspace,
    ))
    core.db.init_db()
    conversation = core.db.create_conversation("Acceptance gate test")
    task = core.db.create_task(conversation["id"], "Создай отчёт research.md")

    plan_response = {"steps": [{
        "id": "step_1",
        "description": "Создать отчёт",
        "expected_result": "Файл research.md существует",
        "success_criteria": ["Файл создан"],
        "acceptance_check": {"type": "file_exists", "path": "reports/research.md"},
    }]}
    complete_response = {
        "decision": "complete_step",
        "reason": "Отчёт готов",
        "verification": "Файл будет создан после ответа модели",
        "final_answer": "Отчёт создан.",
    }

    class FakeClient:
        def __init__(self):
            self.execution_calls = 0
            self.review_calls = 0

        def chat(self, messages, tools=None):
            system = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict) and m.get("role") == "system")
            if "Create a concise execution plan" in system:
                return {"message": {"content": json.dumps(plan_response, ensure_ascii=False)}}
            if "Prepare a concrete execution instruction" in system:
                return {"message": {"content": json.dumps({
                    "task_prompt": "Создай отчёт командами Linux через terminal"
                }, ensure_ascii=False)}}
            if "Review the current LocalAgent plan step" in system:
                self.review_calls += 1
                return {"message": {"content": json.dumps(complete_response, ensure_ascii=False)}}

            self.execution_calls += 1
            if self.execution_calls == 1:
                # The model promises to create the file but does not call a tool.
                return {"message": {"role": "assistant", "content": "Файл будет создан после этого ответа."}}
            return {"message": {"role": "assistant", "content": "Отчёт создан."}}

    fake = FakeClient()
    monkeypatch.setattr(core, "client", fake)

    def fake_dispatch(name, args, **kwargs):
        assert name == "terminal"
        result = subprocess.run(["bash", "-lc", args["command"]], cwd=workspace,
                                capture_output=True, text=True, check=False)
        return {"command": args["command"], "stdout": result.stdout, "stderr": result.stderr,
                "exit_code": result.returncode, "timed_out": False}

    monkeypatch.setattr(core, "dispatch_tool", fake_dispatch)
    original_chat = fake.chat

    def chat_with_tool(messages, tools=None):
        system = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict) and m.get("role") == "system")
        # First execution response is deliberately a promise. After the gate
        # forces a retry, create the file and then verify it in a separate call.
        if tools and fake.execution_calls >= 1 and not (workspace / "reports/research.md").exists():
            fake.execution_calls += 1
            return {"message": {"role": "assistant", "content": "", "tool_calls": [{
                "function": {"name": "terminal", "arguments": {
                    "command": "mkdir -p reports && printf '%s' 'Architecture report' > reports/research.md"
                }}
            }]}}
        if tools and (workspace / "reports/research.md").exists() and not getattr(fake, "verified", False):
            fake.verified = True
            return {"message": {"role": "assistant", "content": "", "tool_calls": [{
                "function": {"name": "terminal", "arguments": {"command": "test -f reports/research.md"}}
            }]}}
        return original_chat(messages, tools)

    monkeypatch.setattr(fake, "chat", chat_with_tool)
    core.Agent()._run_task(task["id"], core.TaskControl())

    saved = core.db.get_task(task["id"])
    checkpoint = json.loads(saved["checkpoint_json"])
    assert saved["status"] == "completed", saved
    assert (workspace / "reports/research.md").is_file()
    step = checkpoint["plan"]["steps"][0]
    assert step["status"] == "completed"
    assert "Фактически проверено" in step["verification"]
    assert checkpoint["step_attempts"]["step_1"] == 1
    assert checkpoint["last_step_decision"]["decision"] == "complete_step"


def test_model_task_prompt_cannot_override_canonical_acceptance_path():
    step = {
        "acceptance_check": {"type": "file_exists", "path": "planning_audit/experiment/research.md"}
    }
    prompt = core._bind_task_prompt_to_acceptance_check(
        step,
        "Создай файл в планировкиAudit/experiment/research.md",
    )
    assert "planning_audit/experiment/research.md" in prompt
    assert "Если ниже или в предыдущем тексте упоминается другой путь" in prompt


def test_resume_reopens_unverified_completed_step_and_dependents(tmp_path):
    from agent.planning import create_plan

    plan = create_plan("Создать отчёт", [
        {
            "id": "step_1",
            "description": "Создать файл отчёта",
            "expected_result": "Файл существует",
            "acceptance_check": {"type": "file_exists", "path": "reports/research.md"},
        },
        {
            "id": "step_2",
            "description": "Проверить отчёт",
            "expected_result": "Отчёт проверен",
            "dependencies": ["step_1"],
        },
    ])
    plan["steps"][0]["status"] = "completed"
    plan["steps"][0]["verification"] = "Файл будет создан позже"
    plan["steps"][1]["status"] = "blocked"
    plan["status"] = "revision_needed"
    attempts = {"step_1": 2, "step_2": 4}

    affected = core._revalidate_completed_steps(plan, [], tmp_path, attempts)

    assert affected == {"step_1", "step_2"}
    assert plan["steps"][0]["status"] == "in_progress"
    assert plan["steps"][0]["verification"] is None
    assert plan["steps"][1]["status"] == "pending"
    assert plan["current_step_id"] == "step_1"
    assert plan["status"] == "in_progress"
    assert attempts == {}
