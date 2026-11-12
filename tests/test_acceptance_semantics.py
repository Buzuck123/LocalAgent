from pathlib import Path

from agent.planning import (
    _valid_acceptance_check,
    verify_acceptance_check,
    verify_acceptance_check_state,
    create_plan,
)
from agent.core import _bind_task_prompt_to_acceptance_check, PLAN_SYSTEM_PROMPT


def test_directory_exists_acceptance_check_validates_real_directory(tmp_path):
    target = tmp_path / "planning_test"
    target.mkdir()
    check = {"type": "directory_exists", "path": "planning_test"}

    assert _valid_acceptance_check(check)
    proof = verify_acceptance_check_state(check, tmp_path, [])
    assert proof and "каталог существует" in proof


def test_directory_exists_does_not_pass_for_missing_directory(tmp_path):
    check = {"type": "directory_exists", "path": "planning_test"}
    assert verify_acceptance_check_state(check, tmp_path, []) is None


def test_directory_check_prompt_does_not_request_marker_file():
    step = {
        "acceptance_check": {
            "type": "directory_exists",
            "path": "planning_test",
        }
    }
    prompt = _bind_task_prompt_to_acceptance_check(step, "Создай каталог.")
    assert "create_directory" in prompt
    assert "Не создавай файл-маркер" in prompt
    assert ".test_planning_test_dir.txt" not in prompt


def test_file_contains_check_remains_exact_and_safe(tmp_path):
    (tmp_path / "task.txt").write_text("Проект: Planning Test\nСтатус: выполнен", encoding="utf-8")
    check = {
        "type": "file_contains",
        "path": "task.txt",
        "text": "Проект: Planning Test\nСтатус: выполнен",
    }
    assert verify_acceptance_check_state(check, tmp_path, [])


def test_directory_acceptance_check_can_be_created_as_plan():
    plan = create_plan(
        "Создай каталог planning_test",
        [{
            "id": "step_1",
            "description": "Создать каталог",
            "expected_result": "Каталог создан",
            "success_criteria": ["Каталог существует"],
            "completion_mode": "tool",
            "acceptance_check": {
                "type": "directory_exists",
                "path": "planning_test",
            },
        }],
    )
    assert plan["steps"][0]["acceptance_check"]["type"] == "directory_exists"


def test_directory_tool_receipt_requires_matching_path(tmp_path):
    (tmp_path / "planning_test").mkdir()
    proof = verify_acceptance_check(
        {"type": "directory_exists", "path": "planning_test"},
        "create_directory",
        {"path": "planning_test"},
        {"ok": True, "result": {"created": "planning_test", "type": "directory"}},
        tmp_path,
    )
    assert proof
    assert verify_acceptance_check(
        {"type": "directory_exists", "path": "planning_test"},
        "create_directory",
        {"path": "other"},
        {"ok": True, "result": {"created": "other", "type": "directory"}},
        tmp_path,
    ) is None



def test_planner_prompt_forbids_fake_marker_and_arbitrary_exact_summary():
    assert "Never replace a directory existence check with a fabricated marker file" in PLAN_SYSTEM_PROMPT
    assert "Do not turn your own examples" in PLAN_SYSTEM_PROMPT
