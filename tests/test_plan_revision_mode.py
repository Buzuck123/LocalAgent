import json

from agent.core import _parse_step_decision


def test_plan_revision_accepts_compact_string_steps():
    decision = {
        "decision": "revise_plan",
        "reason": "Нужно изменить оставшийся маршрут",
        "plan_update": {"steps": [
            "Сначала изучить существующую структуру",
            "Затем реализовать недостающую часть",
        ]},
    }
    parsed = _parse_step_decision(json.dumps(decision, ensure_ascii=False))
    steps = parsed["plan_update"]["steps"]
    assert len(steps) == 2
    assert steps[0]["description"] == "Сначала изучить существующую структуру"
    assert "completion_mode" not in steps[0]
    assert steps[1]["id"] == "revised_step_2"


def test_plan_revision_normalizes_legacy_structured_steps_to_controller_fields():
    decision = {
        "decision": "revise_plan",
        "reason": "Нужен альтернативный подход",
        "plan_update": {"steps": [{
            "id": "step_2",
            "description": "Подготовить альтернативный текстовый результат",
            "expected_result": "Получен результат",
            "success_criteria": ["Есть вывод"],
            "completion_mode": "response",
            "status": "completed",
            "verification": "untrusted model claim",
            "dependencies": ["step_1"],
        }]},
    }
    parsed = _parse_step_decision(json.dumps(decision, ensure_ascii=False))
    step = parsed["plan_update"]["steps"][0]
    assert step["id"] == "revised_step_1"
    assert step["description"] == "Подготовить альтернативный текстовый результат"
    assert step["expected_result"] == "Получен результат"
    # Dependencies are retained for the controller to remap, including
    # references to previously completed stages.
    assert step["dependencies"] == ["step_1"]
    assert "completion_mode" not in step
    assert "status" not in step
    assert "verification" not in step


def test_question_decision_requires_a_user_question():
    from agent.core import _parse_step_decision

    decision = {
        "decision": "question",
        "reason": "Проверены разумные варианты, но без данных пользователя продолжать нельзя.",
        "question": "Какой из двух вариантов выбрать?",
    }
    parsed = _parse_step_decision(json.dumps(decision, ensure_ascii=False))
    assert parsed["decision"] == "question"
    assert parsed["question"] == "Какой из двух вариантов выбрать?"


def test_blocked_is_not_a_valid_step_decision_anymore():
    from agent.core import _parse_step_decision
    decision = {"decision": "blocked", "reason": "Не получилось"}
    try:
        _parse_step_decision(json.dumps(decision, ensure_ascii=False))
    except ValueError:
        pass
    else:
        raise AssertionError("blocked must not be a stage decision")


def test_controller_reassigns_revision_ids_and_preserves_terminal_stages():
    from agent.core import (
        _controller_revise_plan_specs,
        _restore_revised_plan_terminal_states,
    )
    from agent.planning import create_plan

    previous_terminal = [
        {
            "id": "step_1", "description": "Already done",
            "expected_result": "Done", "success_criteria": ["Done"],
            "status": "completed", "verification": "Observed output",
        },
        {
            "id": "step_2", "description": "Intentionally skipped",
            "expected_result": "Not needed", "success_criteria": ["Not needed"],
            "status": "skipped", "verification": "No longer required",
        },
    ]
    remaining = [
        {
            "id": "revised_step_1", "description": "New first stage",
            "expected_result": "First result", "success_criteria": ["First result"],
            "dependencies": ["step_1"],
        },
        {
            "id": "revised_step_2", "description": "New second stage",
            "expected_result": "Second result", "success_criteria": ["Second result"],
            "dependencies": ["revised_step_1"], "status": "completed",
            "verification": "Model supplied, must be ignored",
        },
    ]
    specs = _controller_revise_plan_specs(previous_terminal, remaining, revision_number=3)
    assert [step["id"] for step in specs] == [
        "step_1", "step_2", "adaptive_3_step_1", "adaptive_3_step_2"
    ]
    assert specs[2]["dependencies"] == ["step_1"]
    assert specs[3]["dependencies"] == ["adaptive_3_step_1"]
    assert "status" not in specs[3]
    assert "verification" not in specs[3]

    plan = create_plan("Goal", specs)
    next_step = _restore_revised_plan_terminal_states(plan, previous_terminal)
    assert plan["steps"][0]["status"] == "completed"
    assert plan["steps"][0]["verification"] == "Observed output"
    assert plan["steps"][1]["status"] == "skipped"
    assert plan["steps"][1]["verification"] == "No longer required"
    assert next_step["id"] == "adaptive_3_step_1"


def test_revision_cannot_replace_untouched_plan_with_one_stage():
    from agent.core import _validate_revision_size_for_plan

    untouched = {
        "steps": [
            {"id": "step_1", "status": "in_progress"},
            {"id": "step_2", "status": "pending"},
        ]
    }
    one_stage = {"decision": "revise_plan", "plan_update": {"steps": ["Do one thing"]}}
    try:
        _validate_revision_size_for_plan(untouched, one_stage)
    except ValueError as exc:
        assert "one-stage plan" in str(exc)
    else:
        raise AssertionError("A one-stage revised plan must not be launched")

    single_stage_original = {
        "steps": [{"id": "step_1", "status": "in_progress"}]
    }
    _validate_revision_size_for_plan(single_stage_original, one_stage)

    progressed = {
        "steps": [
            {"id": "step_1", "status": "completed"},
            {"id": "step_2", "status": "in_progress"},
        ]
    }
    _validate_revision_size_for_plan(progressed, one_stage)
