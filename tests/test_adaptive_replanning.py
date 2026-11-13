import json

from agent import core


def test_review_requires_plan_revision_after_retry_budget(monkeypatch):
    calls = []
    responses = [
        {
            "decision": "continue_step",
            "reason": "Нужно попробовать ещё раз",
            "task_prompt": "Повтори действие",
        },
        {
            "decision": "revise_plan",
            "reason": "Исходный подход не работает; использую другой способ",
            "plan_update": {
                "steps": [
                    {
                        "id": "alternative_step",
                        "description": "Использовать альтернативный способ",
                        "expected_result": "Результат получен",
                        "success_criteria": ["Результат подтверждён инструментом"],
                        "dependencies": [],
                        "completion_mode": "tool",
                    }
                ]
            },
        },
    ]

    def fake_client_chat(messages, tools=None, reasoning=True):
        calls.append(messages)
        return {"message": {"content": json.dumps(responses.pop(0), ensure_ascii=False)}}

    monkeypatch.setattr(core, "_client_chat", fake_client_chat)
    plan = {
        "goal": "Выполнить задачу",
        "steps": [
            {
                "id": "step_1",
                "description": "Исходный подход",
                "expected_result": "Результат получен",
                "success_criteria": ["Результат получен"],
                "completion_mode": "tool",
            }
        ],
    }
    step = plan["steps"][0]

    decision = core._review_step(
        plan,
        step,
        "Действие не сработало",
        [{"role": "assistant", "content": "Не получилось"}],
        revision_required="Этап не выполнен за 3 попытки; нужен другой подход",
    )

    assert decision["decision"] == "revise_plan"
    assert len(calls) == 2
    first_payload = json.loads(calls[0][1]["content"])
    assert "нужен другой подход" in first_payload["revision_required"]
    second_payload = json.loads(calls[1][1]["content"])
    assert "Предыдущее решение не прошло проверку" in second_payload["correction"]


def test_review_can_choose_replan_without_forced_revision(monkeypatch):
    response = {
        "decision": "revise_plan",
        "reason": "Зависимость недоступна, требуется другой порядок",
        "plan_update": {
            "steps": [
                {
                    "id": "step_alt",
                    "description": "Альтернативный этап",
                    "expected_result": "Получить результат",
                    "completion_mode": "tool",
                    "dependencies": [],
                }
            ]
        },
    }
    monkeypatch.setattr(
        core,
        "_client_chat",
        lambda *args, **kwargs: {"message": {"content": json.dumps(response, ensure_ascii=False)}},
    )
    decision = core._review_step(
        {"goal": "Задача", "steps": []},
        {"id": "step_1", "expected_result": "Получить результат"},
        "Инструмент сообщил об ошибке",
        [],
    )
    assert decision["decision"] == "revise_plan"


def test_agent_replans_after_four_unsuccessful_step_reviews(tmp_path, monkeypatch):
    from types import SimpleNamespace

    database_path = tmp_path / "adaptive.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(core.db, "settings", SimpleNamespace(database_path=database_path))
    monkeypatch.setattr(
        core,
        "settings",
        SimpleNamespace(max_tool_rounds=30, auto_resume_tasks=False, workspace_path=workspace),
    )
    core.db.init_db()
    conversation = core.db.create_conversation("Adaptive replanning test")
    task = core.db.create_task(conversation["id"], "Получи результат другим способом при неудаче")

    initial_plan = {"steps": [
        {
            "id": "step_1",
            "description": "Первоначальный подход",
            "expected_result": "Результат получен",
            "success_criteria": ["Результат действительно получен"],
            "completion_mode": "response",
        },
        {
            "id": "step_2",
            "description": "Проверить полученный результат",
            "expected_result": "Результат проверен",
            "success_criteria": ["Результат действительно проверен"],
            "dependencies": ["step_1"],
            "completion_mode": "response",
        },
    ]}
    revised_plan = {
        "decision": "revise_plan",
        "reason": "Первоначальный подход четыре раза не дал результата; выбран альтернативный способ",
        "plan_update": {"steps": [
            {
                "id": "step_alt_1",
                "description": "Выполнить альтернативный подход",
                "expected_result": "Результат получен альтернативным способом",
                "success_criteria": ["Альтернативный результат получен"],
                "dependencies": [],
                "completion_mode": "response",
            },
            {
                "id": "step_alt_2",
                "description": "Проверить альтернативный результат",
                "expected_result": "Альтернативный результат проверен",
                "success_criteria": ["Проверка пройдена"],
                "dependencies": ["step_alt_1"],
                "completion_mode": "response",
            },
        ]},
    }

    class FakeClient:
        def __init__(self):
            self.reviews = 0
            self.executions = 0
            self.saw_forced_revision = False

        def chat(self, messages, tools=None):
            system = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict) and m.get("role") == "system")
            if "Create a concise execution plan" in system:
                return {"message": {"content": json.dumps(initial_plan, ensure_ascii=False)}}
            if "Prepare a concrete execution instruction" in system:
                return {"message": {"content": json.dumps({
                    "task_prompt": "Попробуй выполнить текущий подход и сообщи фактический результат."
                }, ensure_ascii=False)}}
            if "Review the current LocalAgent plan step" in system:
                self.reviews += 1
                payload = json.loads(messages[1]["content"])
                if payload.get("revision_required"):
                    self.saw_forced_revision = True
                    return {"message": {"content": json.dumps(revised_plan, ensure_ascii=False)}}
                if str(payload.get("current_step", {}).get("id", "")).startswith("adaptive_1_step_"):
                    return {"message": {"content": json.dumps({
                        "decision": "complete_step",
                        "reason": "Альтернативный результат получен",
                        "verification": "Альтернативный результат подтверждён",
                        "final_answer": "Задача завершена после перепланирования.",
                    }, ensure_ascii=False)}}
                return {"message": {"content": json.dumps({
                    "decision": "continue_step",
                    "reason": "Первоначальный подход не дал результата",
                    "task_prompt": "Не повторяй утверждение об успехе; попробуй выполнить этап ещё раз.",
                }, ensure_ascii=False)}}
            self.executions += 1
            return {"message": {"role": "assistant", "content": "Первоначальный подход не дал результата."}}

    fake = FakeClient()
    monkeypatch.setattr(core, "client", fake)
    core.Agent()._run_task(task["id"], core.TaskControl())

    saved = core.db.get_task(task["id"])
    checkpoint = json.loads(saved["checkpoint_json"])
    assert saved["status"] == "completed", saved
    assert fake.saw_forced_revision is True
    assert checkpoint["plan_revision_count"] == 1
    assert [step["id"] for step in checkpoint["plan"]["steps"]] == [
        "adaptive_1_step_1", "adaptive_1_step_2"
    ]
    assert all(step["status"] == "completed" for step in checkpoint["plan"]["steps"])
    assert checkpoint["last_plan_revision_reason"] == revised_plan["reason"]


def test_fallback_planner_builds_unique_remaining_step_ids(monkeypatch):
    def fake_create_plan(prompt, reasoning=True, validate=True):
        assert "only for the remaining work" in prompt
        assert "Completed steps" in prompt
        return {
            "goal": "remaining",
            "steps": [
                {
                    "id": "step_1",
                    "description": "Новый подход",
                    "expected_result": "Результат получен",
                    "success_criteria": ["Проверено"],
                    "dependencies": [],
                    "completion_mode": "tool",
                },
                {
                    "id": "step_2",
                    "description": "Проверить результат",
                    "expected_result": "Проверка пройдена",
                    "success_criteria": ["Тесты пройдены"],
                    "dependencies": ["step_1"],
                    "completion_mode": "tool",
                },
            ],
        }

    monkeypatch.setattr(core, "_create_plan_with_retries", fake_create_plan)
    steps = core._create_alternative_remaining_steps(
        {
            "goal": "Создать рабочий проект",
            "steps": [
                {"id": "step_0", "status": "completed", "description": "Создать каталог", "verification": "Каталог существует"},
                {"id": "step_1", "status": "in_progress", "description": "Старый неудачный способ"},
            ],
        },
        {
            "id": "step_1",
            "description": "Старый неудачный способ",
            "expected_result": "Результат получен",
            "success_criteria": ["Результат получен"],
            "completion_mode": "tool",
        },
        "Подход не работает",
        "Ошибка инструмента",
        [],
        [],
        revision_number=1,
    )

    assert [step["id"] for step in steps] == ["adaptive_1_step_1", "adaptive_1_step_2"]
    assert steps[1]["dependencies"] == ["adaptive_1_step_1"]
    assert all(step["id"] != "step_0" for step in steps)


def test_review_rejects_invalid_acceptance_check_in_revised_plan(monkeypatch):
    calls = []
    responses = [
        {
            "decision": "revise_plan",
            "reason": "Нужен другой способ",
            "plan_update": {"steps": [{
                "id": "bad_step",
                "description": "Создать результат",
                "expected_result": "Файл создан",
                "completion_mode": "tool",
                "acceptance_check": {"type": "file_exists", "path": "result.txt", "extra": True},
            }]},
        },
        {
            "decision": "revise_plan",
            "reason": "Исправленный план",
            "plan_update": {"steps": [{
                "id": "good_step",
                "description": "Создать результат",
                "expected_result": "Файл создан",
                "completion_mode": "tool",
                "acceptance_check": {"type": "file_exists", "path": "result.txt"},
            }]},
        },
    ]

    def fake_client_chat(messages, tools=None, reasoning=True):
        calls.append(messages)
        return {"message": {"content": json.dumps(responses.pop(0), ensure_ascii=False)}}

    monkeypatch.setattr(core, "_client_chat", fake_client_chat)
    decision = core._review_step(
        {"goal": "Создать проверяемый файл", "steps": []},
        {"id": "step_1", "description": "Исходный способ", "expected_result": "Файл создан"},
        "Первый способ не сработал",
        [],
    )

    assert decision["decision"] == "revise_plan"
    # Step IDs are controller-owned and normalized by the parser.
    assert decision["plan_update"]["steps"][0]["id"] == "revised_step_1"
    assert len(calls) == 2
    correction_payload = json.loads(calls[1][1]["content"])
    assert "Предыдущее решение не прошло проверку" in correction_payload["correction"]
    assert "acceptance_check" in correction_payload["correction"]


def test_revised_plan_can_depend_on_preserved_completed_step():
    decision = {
        "decision": "revise_plan",
        "reason": "Первоначальный подход не подходит",
        "plan_update": {
            "steps": [{
                "id": "step_3",
                "description": "Применить альтернативный подход",
                "expected_result": "Получен альтернативный результат",
                "completion_mode": "response",
                "success_criteria": ["Альтернативный результат проверен"],
                # step_1 is intentionally absent from plan_update: it is retained
                # from the original plan because it was already completed.
                "dependencies": ["step_1"],
            }]
        },
    }

    parsed = core._parse_step_decision(json.dumps(decision, ensure_ascii=False))

    assert parsed["decision"] == "revise_plan"
    assert parsed["plan_update"]["steps"][0]["dependencies"] == ["step_1"]


def test_agent_replans_after_three_tool_errors(tmp_path, monkeypatch):
    from types import SimpleNamespace

    database_path = tmp_path / "tool_errors.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setattr(
        core.db, "settings",
        SimpleNamespace(database_path=database_path),
    )
    monkeypatch.setattr(
        core, "settings",
        SimpleNamespace(
            max_tool_rounds=30,
            auto_resume_tasks=False,
            workspace_path=workspace,
        ),
    )

    core.db.init_db()
    conversation = core.db.create_conversation("Three tool errors")
    task = core.db.create_task(
        conversation["id"],
        "Получи результат и смени подход при ошибках",
    )

    initial_plan = {"steps": [
        {
            "id": "step_1",
            "description": "Первоначальный подход",
            "expected_result": "Получен результат",
            "success_criteria": ["Результат получен"],
            "completion_mode": "response",
        },
        {
            "id": "step_2",
            "description": "Проверить результат",
            "expected_result": "Результат проверен",
            "success_criteria": ["Результат проверен"],
            "dependencies": ["step_1"],
            "completion_mode": "response",
        },
    ]}

    revised_plan = {
        "decision": "revise_plan",
        "reason": "Три ошибки инструмента требуют другого подхода",
        "plan_update": {"steps": [
            {
                "id": "step_alt_1",
                "description": "Альтернативный подход",
                "expected_result": "Получен альтернативный результат",
                "success_criteria": ["Альтернативный результат получен"],
                "dependencies": [],
                "completion_mode": "response",
            },
            {
                "id": "step_alt_2",
                "description": "Проверить альтернативный результат",
                "expected_result": "Альтернативный результат проверен",
                "success_criteria": ["Проверка пройдена"],
                "dependencies": ["step_alt_1"],
                "completion_mode": "response",
            },
        ]},
    }

    class FakeClient:
        def __init__(self):
            self.execution_calls = 0
            self.tool_errors_replanned = False

        def chat(self, messages, tools=None):
            system = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict) and m.get("role") == "system")

            if "Create a concise execution plan" in system:
                return {"message": {
                    "content": json.dumps(initial_plan, ensure_ascii=False)
                }}

            if "Prepare a concrete execution instruction" in system:
                return {"message": {
                    "content": json.dumps({
                        "task_prompt": "Выполни текущий подход и проверь результат."
                    }, ensure_ascii=False)
                }}

            if "Review the current LocalAgent plan step" in system:
                payload = json.loads(messages[1]["content"])

                if payload.get("revision_required"):
                    self.tool_errors_replanned = True
                    return {"message": {
                        "content": json.dumps(revised_plan, ensure_ascii=False)
                    }}

                if str(payload.get("current_step", {}).get("id", "")).startswith("adaptive_1_step_"):
                    return {"message": {
                        "content": json.dumps({
                            "decision": "complete_step",
                            "reason": "Альтернативный результат получен",
                            "verification": "Результат альтернативного подхода подтверждён",
                            "final_answer": "Задача завершена после смены подхода.",
                        }, ensure_ascii=False)
                    }}

                return {"message": {
                    "content": json.dumps({
                        "decision": "continue_step",
                        "reason": "Первоначальный подход не дал результата",
                        "task_prompt": "Повтори действие и сообщи результат.",
                    }, ensure_ascii=False)
                }}

            self.execution_calls += 1

            if self.execution_calls <= 3:
                return {"message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "terminal",
                            "arguments": {"command": f"false # attempt-{self.execution_calls}"},
                        }
                    }],
                }}

            return {"message": {
                "role": "assistant",
                "content": (
                    "Альтернативный результат получен."
                    if self.tool_errors_replanned
                    else "Первоначальный подход не дал результата."
                ),
            }}

    fake = FakeClient()
    monkeypatch.setattr(core, "client", fake)

    dispatched = []

    def fake_dispatch(name, args, **kwargs):
        dispatched.append((name, dict(args)))
        assert name == "terminal"
        raise RuntimeError("Искусственная ошибка вызова инструмента для теста")

    monkeypatch.setattr(core, "dispatch_tool", fake_dispatch)

    core.Agent()._run_task(task["id"], core.TaskControl())

    saved = core.db.get_task(task["id"])
    checkpoint = json.loads(saved["checkpoint_json"])

    assert len(dispatched) == 3, dispatched
    assert fake.tool_errors_replanned is True
    assert checkpoint["plan_revision_count"] == 1
    assert [step["id"] for step in checkpoint["plan"]["steps"]] == [
        "adaptive_1_step_1", "adaptive_1_step_2"
    ]
    assert all(step["status"] == "completed" for step in checkpoint["plan"]["steps"])
    assert saved["status"] == "completed", saved


def test_fallback_replanning_normalizes_dependencies(monkeypatch):
    import json
    from agent import core

    raw_plan = {
        "steps": [
            {
                "id": "adaptive_step_1",
                "description": "Создать основу альтернативного решения",
                "expected_result": "Основа создана",
                "success_criteria": ["Основа создана"],
                # Ссылка на будущий этап должна быть отброшена.
                "dependencies": ["step_3"],
                "completion_mode": "tool",
            },
            {
                "id": "adaptive_step_2",
                "description": "Продолжить альтернативное решение",
                "expected_result": "Решение продолжено",
                "success_criteria": ["Решение продолжено"],
                "dependencies": ["adaptive_step_1"],
                "completion_mode": "tool",
            },
            {
                "id": "adaptive_step_3",
                "description": "Завершить альтернативное решение",
                "expected_result": "Решение завершено",
                "success_criteria": ["Решение завершено"],
                "dependencies": ["adaptive_step_2", "step_3"],
                "completion_mode": "response",
            },
        ]
    }

    def fake_client_chat(messages, tools=None, reasoning=True):
        return {
            "message": {
                "content": json.dumps(raw_plan, ensure_ascii=False)
            }
        }

    monkeypatch.setattr(core, "_client_chat", fake_client_chat)

    original_plan = {
        "goal": "Исходная цель пользователя",
        "steps": [
            {
                "id": "step_3",
                "description": "Первоначальный неудачный подход",
                "expected_result": "Получить результат",
                "task_prompt": "Выполни первоначальный подход",
                "status": "in_progress",
            }
        ],
    }

    result = core._create_alternative_remaining_steps(
        plan=original_plan,
        failed_step=original_plan["steps"][0],
        failure_reason="Три ошибки инструмента",
        model_result="Первоначальный подход не сработал",
        messages=[],
        step_tool_evidence=[],
        revision_number=1,
        reasoning=False,
    )

    assert [step["id"] for step in result] == [
        "adaptive_1_step_1",
        "adaptive_1_step_2",
        "adaptive_1_step_3",
    ]

    assert result[0]["dependencies"] == []
    assert result[1]["dependencies"] == ["adaptive_1_step_1"]
    assert result[2]["dependencies"] == ["adaptive_1_step_2"]

    # Обычная строгая проверка планов не должна измениться.
    from agent.planning import create_plan
    import pytest

    with pytest.raises(ValueError, match="зависимости должны ссылаться"):
        create_plan(
            "Обычный план",
            [
                {
                    "id": "first",
                    "description": "Первый этап",
                    "expected_result": "Готово",
                    "dependencies": ["future"],
                },
                {
                    "id": "future",
                    "description": "Второй этап",
                    "expected_result": "Готово",
                },
            ],
        )


def test_step_reviewer_parser_accepts_optional_reconciled_steps():
    decision = {
        "decision": "complete_step",
        "reason": "Текущий этап и дополнительный результат проверены",
        "verification": "Текущий этап завершён",
        "reconciled_steps": [{"step_id": "step_3", "verification": "Файл отчёта создан и проверен"}],
    }
    parsed = core._parse_step_decision(json.dumps(decision, ensure_ascii=False))
    assert parsed["reconciled_steps"] == decision["reconciled_steps"]


def test_step_reviewer_parser_defaults_reconciled_steps_for_legacy_responses():
    decision = {
        "decision": "complete_step",
        "reason": "Готово",
        "verification": "Подтверждено",
    }
    parsed = core._parse_step_decision(json.dumps(decision, ensure_ascii=False))
    assert parsed["reconciled_steps"] == []
