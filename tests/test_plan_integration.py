import json
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import agent.core as core


def test_plan_steps_are_verified_and_checkpointed(tmp_path, monkeypatch):
    database_path = tmp_path / "integration.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setattr(
        core.db, "settings", SimpleNamespace(database_path=database_path)
    )
    monkeypatch.setattr(
        core, "settings",
        SimpleNamespace(
            max_tool_rounds=10,
            auto_resume_tasks=False,
            workspace_path=workspace,
        ),
    )
    # Injected clients use the deterministic route hook; this test explicitly
    # exercises Plan rather than the unrelated Direct-vs-Plan model selector.
    monkeypatch.setattr(core, "classify_execution_mode", lambda *args, **kwargs: "planned")

    core.db.init_db()
    conversation = core.db.create_conversation("Plan integration test")
    task = core.db.create_task(
        conversation["id"], "Создай два тестовых файла и проверь результат"
    )

    planned_steps = {
        "steps": [
            {
                "id": "step_1",
                "description": "Создать первый файл",
                "expected_result": "first.txt содержит VERIFIED",
                "acceptance_check": {
                    "type": "file_contains", "path": "first.txt", "text": "VERIFIED"
                },
            },
            {
                "id": "step_2",
                "description": "Создать второй файл",
                "expected_result": "second.txt существует",
                "acceptance_check": {"type": "file_exists", "path": "second.txt"},
            },
        ]
    }
    # The controller may make additional LLM requests to prepare a stage
    # prompt. Drive this fake by the purpose of each request and by commands
    # actually executed, not by brittle absolute call numbers.
    tool_commands = [
        "printf '%s' 'VERIFIED' > first.txt",
        "grep -F -- VERIFIED first.txt",
        "printf '%s' 'SECOND FILE' > second.txt",
        "test -f second.txt",
    ]

    class FakeClient:
        def __init__(self):
            self.calls = 0
            self.plan_requests = 0
            self.step_prompt_requests = 0
            self.review_requests = 0

        @staticmethod
        def _message(content, tool_calls=None):
            message = {"role": "assistant", "content": content}
            if tool_calls is not None:
                message["tool_calls"] = tool_calls
            return {"message": message}

        def chat(self, messages, tools=None):
            self.calls += 1
            system_text = "\\n".join(
                str(item.get("content", ""))
                for item in messages
                if isinstance(item, dict) and item.get("role") == "system"
            )

            if "Create a concise execution plan" in system_text:
                self.plan_requests += 1
                return self._message(json.dumps(planned_steps, ensure_ascii=False))

            if "Prepare a concise working instruction" in system_text:
                self.step_prompt_requests += 1
                return self._message(json.dumps({
                    "task_prompt": "Создай требуемый файл и выполни указанную проверку."
                }, ensure_ascii=False))

            if "Review the current LocalAgent plan step" in system_text:
                self.review_requests += 1
                raise AssertionError(
                    "Детерминированный acceptance_check должен завершить этап "
                    "без дополнительного LLM-review"
                )

            # FakeRuntime is the source of truth for which actions have run.
            executed = set(runtime.commands)
            if tool_commands[0] not in executed:
                command = tool_commands[0]
            elif tool_commands[1] not in executed:
                command = tool_commands[1]
            elif tool_commands[2] not in executed:
                return self._message("Первый файл создан и проверен.")
            elif tool_commands[3] not in executed:
                command = tool_commands[3]
            else:
                return self._message("Оба файла созданы и проверены.")

            assert isinstance(tools, list)
            assert any(
                item.get("function", {}).get("name") == "terminal"
                for item in tools
            )
            return self._message("", [{
                "function": {
                    "name": "terminal", "arguments": {"command": command}
                }
            }])

    fake_client = FakeClient()
    monkeypatch.setattr(core, "client", fake_client)

    # Exercise the actual Sandbox Manager -> SandboxRuntime boundary. Only the
    # remote runtime is replaced; terminal orchestration and tool dispatch remain real.
    from agent.sandbox_manager import sandbox_manager

    class FakeRuntime:
        def __init__(self):
            self.operations = {}
            self.commands = []

        def start_operation(self, command, *, session="main", timeout=None,
                            idempotency_key=None):
            operation_id = str(uuid.uuid4())
            self.commands.append(command)
            result = subprocess.run(
                ["bash", "-c", command], cwd=workspace,
                capture_output=True, text=True, check=False,
            )
            self.operations[operation_id] = {
                "state": "completed",
                "output": result.stdout + result.stderr,
                "exit_code": result.returncode,
                "next_cursor": len(result.stdout + result.stderr),
            }
            return {"operation_id": operation_id, "state": "running"}

        def get_operation(self, operation_id, *, session="main", cursor=0,
                          max_output_chars=65536, wait_timeout=0.0):
            return dict(self.operations[operation_id])

        def ensure_session(self, session="main"):
            return False

    runtime = FakeRuntime()
    monkeypatch.setattr(sandbox_manager, "get_runtime", lambda conversation_id: runtime)
    monkeypatch.setattr(sandbox_manager, "_active_operations", {})
    monkeypatch.setattr(sandbox_manager, "_new_sessions", set())
    monkeypatch.setattr(
        sandbox_manager, "_find_persisted_active_operation", lambda conversation_id: None
    )

    agent = core.Agent()
    agent._run_task(task["id"], core.TaskControl())

    saved_task = core.db.get_task(task["id"])
    assert saved_task["status"] == "completed", saved_task
    assert saved_task["result"] == "Оба файла созданы и проверены."
    assert runtime.commands == tool_commands

    assert (workspace / "first.txt").read_text(encoding="utf-8") == "VERIFIED"
    assert (workspace / "second.txt").read_text(encoding="utf-8") == "SECOND FILE"

    checkpoint = json.loads(saved_task["checkpoint_json"])
    plan = checkpoint["plan"]
    assert checkpoint["phase"] == "completed"
    assert checkpoint["execution_mode"] == "planned"
    assert plan["status"] == "completed"
    assert [step["status"] for step in plan["steps"]] == ["completed", "completed"]
    assert all(
        isinstance(step.get("verification"), str) and step["verification"].strip()
        for step in plan["steps"]
    )
    assert fake_client.plan_requests == 1
    assert fake_client.review_requests == 0


def test_review_reconciliation_marks_scope_leaked_future_stage_complete(tmp_path):
    from agent.planning import create_plan, set_step_status, reconcile_plan_with_review_evidence

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plan = create_plan("Создать отчёт", [
        {"id": "step_1", "description": "Подготовить исходные данные", "expected_result": "Данные подготовлены"},
        {"id": "step_2", "description": "Создать Excel", "expected_result": "Книга Excel создана"},
        {"id": "step_3", "description": "Сформировать JSON", "expected_result": "JSON сформирован", "dependencies": ["step_2"]},
    ])
    set_step_status(plan, "step_1", "completed", "Исходные данные подготовлены")

    reconciled = reconcile_plan_with_review_evidence(
        plan,
        [
            {"step_id": "step_2", "verification": "report.xlsx создан и прочитан; в книге четыре листа"},
            {"step_id": "step_3", "verification": "summary.json создан и сверка значений прошла"},
        ],
        [{"tool_name": "terminal", "result_summary": '{"exit_code": 0}'}],
        workspace,
        messages=[],
        after_step_id="step_1",
    )

    assert [item["step_id"] for item in reconciled] == ["step_2", "step_3"]
    assert plan["steps"][1]["status"] == "completed"
    assert plan["steps"][2]["status"] == "completed"
    assert plan["status"] == "completed"


def test_review_reconciliation_requires_tool_evidence_and_respects_acceptance_checks(tmp_path):
    from agent.planning import create_plan, set_step_status, reconcile_plan_with_review_evidence

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plan = create_plan("Создать файл", [
        {"id": "step_1", "description": "Подготовить папку", "expected_result": "Папка подготовлена"},
        {
            "id": "step_2", "description": "Создать файл", "expected_result": "report.txt существует",
            "acceptance_check": {"type": "file_exists", "path": "report.txt"},
        },
    ])
    set_step_status(plan, "step_1", "completed", "Папка подготовлена")
    reviewed = [{"step_id": "step_2", "verification": "Модель утверждает, что файл создан"}]

    assert reconcile_plan_with_review_evidence(
        plan, reviewed, [], workspace, messages=[], after_step_id="step_1"
    ) == []
    assert plan["steps"][1]["status"] == "in_progress"

    assert reconcile_plan_with_review_evidence(
        plan, reviewed, [{"tool_name": "terminal"}], workspace, messages=[], after_step_id="step_1"
    ) == []
    assert plan["steps"][1]["status"] == "in_progress"
