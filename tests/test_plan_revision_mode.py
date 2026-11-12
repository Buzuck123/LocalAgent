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


def test_plan_revision_keeps_legacy_structured_steps():
    decision = {
        "decision": "revise_plan",
        "reason": "Нужен альтернативный подход",
        "plan_update": {"steps": [{
            "id": "step_2",
            "description": "Подготовить альтернативный текстовый результат",
            "expected_result": "Получен результат",
            "success_criteria": ["Есть вывод"],
            "completion_mode": "response",
        }]},
    }
    parsed = _parse_step_decision(json.dumps(decision, ensure_ascii=False))
    assert parsed["plan_update"]["steps"][0]["completion_mode"] == "response"  # legacy field accepted, not enforced


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
