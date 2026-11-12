import json
from types import SimpleNamespace

import pytest

import agent.core as core
from agent.planning import create_plan, validate_plan


def test_plan_normalizes_success_criteria_and_task_prompt():
    plan = create_plan(
        "Подготовить отчёт",
        [{
            "id": "step_1",
            "description": "Изучить данные",
            "expected_result": "Подготовлено резюме данных",
            "success_criteria": ["Резюме содержит основные выводы"],
            "task_prompt": "Изучи предоставленные данные и подготовь резюме",
            "dependencies": [],
        }],
    )

    step = plan["steps"][0]
    assert step["success_criteria"] == ["Резюме содержит основные выводы"]
    assert step["task_prompt"] == "Изучи предоставленные данные и подготовь резюме"
    assert step["dependencies"] == []
    assert validate_plan(plan)


def test_plan_defaults_success_criteria_to_expected_result():
    plan = create_plan(
        "Тест",
        [{"description": "Сделать работу", "expected_result": "Получен результат"}],
    )
    assert plan["steps"][0]["success_criteria"] == ["Получен результат"]
    assert plan["steps"][0]["task_prompt"] is None


def test_plan_rejects_empty_success_criteria():
    with pytest.raises(ValueError, match="success_criteria"):
        create_plan(
            "Тест",
            [{
                "description": "Сделать работу",
                "expected_result": "Получен результат",
                "success_criteria": [],
            }],
        )


def test_planner_retries_after_invalid_plan(monkeypatch):
    responses = [
        {"message": {"content": json.dumps({"steps": [{
            "id": "step_1",
            "description": "Без ожидаемого результата",
        }]})}},
        {"message": {"content": json.dumps({"steps": [
            {
                "id": "step_1",
                "description": "Изучить исходные требования",
                "expected_result": "Требования и ограничения зафиксированы",
                "success_criteria": ["Определены основные требования"],
            },
            {
                "id": "step_2",
                "description": "Подготовить план работ",
                "expected_result": "План содержит конкретные этапы",
                "success_criteria": ["Все этапы имеют ожидаемый результат"],
            },
        ]})}},
    ]
    captured = []

    class FakeClient:
        def chat(self, messages, tools=None):
            captured.append(messages)
            return responses.pop(0)

    monkeypatch.setattr(core, "client", FakeClient())
    plan = core._create_plan_with_retries("Составь план")

    assert len(captured) == 2
    assert "не прошёл проверку" in captured[1][-1]["content"]
    assert validate_plan(plan)


def test_step_decision_requires_verification_for_completion():
    with pytest.raises(ValueError, match="verification"):
        core._parse_step_decision(json.dumps({
            "decision": "complete_step",
            "reason": "Похоже, готово",
        }))


def test_step_decision_accepts_complete_with_verification():
    decision = core._parse_step_decision(json.dumps({
        "decision": "complete_step",
        "reason": "Критерий подтверждён",
        "verification": "В результате есть требуемое резюме",
    }))
    assert decision["decision"] == "complete_step"


def test_plan_rejects_dependency_on_later_or_unknown_step():
    with pytest.raises(ValueError, match="зависимости должны ссылаться"):
        create_plan(
            "Тест",
            [
                {
                    "id": "step_1",
                    "description": "Первый этап",
                    "expected_result": "Результат 1",
                    "dependencies": ["step_2"],
                },
                {
                    "id": "step_2",
                    "description": "Второй этап",
                    "expected_result": "Результат 2",
                },
            ],
        )


def test_plain_text_plan_is_accepted_without_json_schema():
    parsed = core._parse_plan_json("""
1. Изучить существующий проект и требования
2. Реализовать недостающие функции
3. Проверить результат и подготовить инструкцию
""")
    assert [step["description"] for step in parsed["steps"]] == [
        "Изучить существующий проект и требования",
        "Реализовать недостающие функции",
        "Проверить результат и подготовить инструкцию",
    ]
    plan = core.create_plan("Создать систему", parsed["steps"])
    assert core.validate_plan(plan)
    assert plan["steps"][1]["success_criteria"] == ["Реализовать недостающие функции"]


def test_bulleted_plan_accepts_wrapped_lines():
    parsed = core._parse_plan_json("""
- Изучить данные и определить структуру
  включая существующие таблицы и ограничения
- Создать основную логику
""")
    assert len(parsed["steps"]) == 2
    assert "включая существующие таблицы" in parsed["steps"][0]["description"]


def test_planning_context_snapshot_contains_all_stages_and_revision_history():
    plan = core.create_plan("Сделать задачу", [
        {"id": "a", "description": "Первый этап", "expected_result": "Готово"},
        {"id": "b", "description": "Второй этап", "expected_result": "Готово"},
    ])
    plan["steps"][0]["status"] = "completed"
    plan["steps"][0]["verification"] = "Проверено"
    state = {
        "step_attempts": {"b": 2},
        "plan_revision_count": 1,
        "plan_revision_history": [{"revision": 1, "reason": "Изменил подход"}],
    }
    snapshot = core._planning_context_snapshot(
        plan, state, [
            {"role": "user", "content": "Сохрани исходный формат данных"},
            {"role": "assistant", "content": "Старое предположение: сделать все этапы сейчас"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "data.csv"}}}]},
        ]
    )
    assert len(snapshot["steps"]) == 2
    assert snapshot["steps"][0]["status"] == "completed"
    assert snapshot["steps"][1]["attempts"] == 2
    assert snapshot["revision_history"][0]["reason"] == "Изменил подход"
    assert snapshot["recent_dialogue"][0]["content"] == "Сохрани исходный формат данных"
    assert snapshot["recent_dialogue"][1]["role"] == "assistant_tool_calls"
    assert "Старое предположение" not in json.dumps(snapshot["recent_dialogue"], ensure_ascii=False)


def test_plan_context_keeps_current_focus_without_rigid_stage_lock():
    plan = create_plan("Подготовить проект", [
        {"id": "stage_1", "description": "Создать структуру", "expected_result": "Каталоги созданы"},
        {"id": "stage_2", "description": "Подготовить данные", "expected_result": "CSV готов"},
    ])
    messages = [{"role": "system", "content": "base"}]
    core._sync_plan_context(messages, plan)
    context = next(m["content"] for m in messages if core.PLAN_CONTEXT_MARKER in m["content"])
    assert "Current focus: stage_1" in context
    assert "not a strict allowlist" in context
    assert "adjacent work" in context
    assert "Report completed work and its observable evidence" in context
    assert "Do not begin, implement, or pre-empt any later step" not in context
    assert "Plan overview (orientation, not a rigid script)" in context


def test_workspace_reconciliation_marks_only_future_steps_with_verified_checks(tmp_path):
    from agent.planning import reconcile_plan_with_workspace

    (tmp_path / "already.csv").write_text("sku,price\nA1,10\n", encoding="utf-8")
    plan = create_plan("Создать проект", [
        {"id": "stage_1", "description": "Создать структуру", "expected_result": "Каталоги созданы"},
        {"id": "stage_2", "description": "Подготовить CSV", "expected_result": "CSV существует",
         "acceptance_check": {"type": "file_exists", "path": "already.csv"}},
        {"id": "stage_3", "description": "Написать обработчик", "expected_result": "Скрипт готов"},
    ])

    from agent.planning import set_step_status
    set_step_status(plan, "stage_1", "completed", verification="Каталоги созданы")
    reconciled = reconcile_plan_with_workspace(
        plan, tmp_path, after_step_id="stage_1"
    )

    assert [item["step_id"] for item in reconciled] == ["stage_2"]
    assert plan["steps"][1]["status"] == "completed"
    assert plan["steps"][2]["status"] == "in_progress"
    assert plan["current_step_id"] == "stage_3"


def test_workspace_reconciliation_does_not_infer_steps_without_checks(tmp_path):
    from agent.planning import reconcile_plan_with_workspace

    (tmp_path / "output.txt").write_text("done", encoding="utf-8")
    plan = create_plan("Создать проект", [
        {"id": "stage_1", "description": "Структура", "expected_result": "Каталоги созданы"},
        {"id": "stage_2", "description": "Создать файл", "expected_result": "Файл готов"},
    ])
    from agent.planning import set_step_status
    set_step_status(plan, "stage_1", "completed", verification="Каталоги созданы")
    reconciled = reconcile_plan_with_workspace(plan, tmp_path, after_step_id="stage_1")
    assert reconciled == []
    assert plan["steps"][1]["status"] == "in_progress"


def test_plan_context_is_inserted_after_compacted_history_summary():
    plan = create_plan("Подготовить проект", [
        {"id": "stage_1", "description": "Создать структуру", "expected_result": "Каталоги созданы"},
    ])
    messages = [
        {"role": "system", "content": "base"},
        {"role": "system", "content": core.CONTEXT_SUMMARY_MARKER + "\nOld factual observations"},
        {"role": "user", "content": "continue"},
    ]
    core._sync_plan_context(messages, plan)
    summary_index = next(i for i, m in enumerate(messages) if core.CONTEXT_SUMMARY_MARKER in m.get("content", ""))
    plan_index = next(i for i, m in enumerate(messages) if core.PLAN_CONTEXT_MARKER in m.get("content", ""))
    assert plan_index == summary_index + 1


def test_one_stage_planner_response_falls_back_instead_of_launching_a_plan(monkeypatch):
    response = {"message": {"content": json.dumps({"steps": [{
        "id": "step_1",
        "description": "Complete the entire request",
        "expected_result": "The request is complete",
    }]})}}
    calls = []
    def fake_chat(messages, tools=None, reasoning=True):
        calls.append(messages)
        return response
    monkeypatch.setattr(core, "_client_chat", fake_chat)
    plan = core._create_plan_with_retries("Create a bounded output")
    assert len(calls) == 3
    assert len(plan["steps"]) == 1
    assert plan.get("planner_fallback", {}).get("used") is True
