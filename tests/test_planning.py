import pytest

from agent.planning import (
    create_plan,
    validate_plan,
    current_step,
    set_step_status,
    pause_plan,
)


def make_plan():
    return create_plan(
        "Подготовить отчёт",
        [
            {
                "id": "step_1",
                "description": "Собрать данные",
                "expected_result": "Получен набор исходных данных",
            },
            {
                "id": "step_2",
                "description": "Подготовить отчёт",
                "expected_result": "Отчёт сохранён и проверен",
            },
        ],
    )


def test_new_plan_starts_with_first_step():
    plan = make_plan()

    assert validate_plan(plan)
    assert plan["status"] == "in_progress"
    assert current_step(plan)["id"] == "step_1"
    assert plan["steps"][0]["status"] == "in_progress"
    assert plan["steps"][1]["status"] == "pending"


def test_step_cannot_complete_without_verification():
    plan = make_plan()

    with pytest.raises(ValueError, match="подтверждение"):
        set_step_status(plan, "step_1", "completed")


def test_verified_step_advances_to_next_step():
    plan = make_plan()

    set_step_status(
        plan,
        "step_1",
        "completed",
        verification="Набор данных существует и прочитан",
    )

    assert plan["steps"][0]["status"] == "completed"
    assert current_step(plan)["id"] == "step_2"
    assert plan["steps"][1]["status"] == "in_progress"


def test_plan_completes_only_after_all_steps_are_verified():
    plan = make_plan()

    set_step_status(
        plan, "step_1", "completed",
        verification="Исходные данные проверены",
    )
    set_step_status(
        plan, "step_2", "completed",
        verification="Отчёт проверен",
    )

    assert plan["status"] == "completed"
    assert plan["current_step_id"] is None
    assert current_step(plan) is None


def test_pause_preserves_incomplete_steps():
    plan = pause_plan(make_plan())

    assert plan["status"] == "paused"
    assert current_step(plan)["id"] == "step_1"


def test_duplicate_step_ids_are_rejected():
    with pytest.raises(ValueError, match="Повторяющийся id"):
        create_plan(
            "Тест",
            [
                {
                    "id": "same",
                    "description": "Первый этап",
                    "expected_result": "Результат 1",
                },
                {
                    "id": "same",
                    "description": "Второй этап",
                    "expected_result": "Результат 2",
                },
            ],
        )
