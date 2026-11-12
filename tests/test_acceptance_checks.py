from pathlib import Path

import pytest

from agent.planning import (
    create_plan,
    validate_plan,
    verify_acceptance_check,
)


def test_plan_accepts_file_exists_check():
    plan = create_plan(
        "Создать файл",
        [{
            "id": "step_1",
            "description": "Создать файл",
            "expected_result": "Файл создан",
            "acceptance_check": {
                "type": "file_exists",
                "path": "result.txt",
            },
        }],
    )

    assert validate_plan(plan)
    assert plan["steps"][0]["acceptance_check"]["type"] == "file_exists"


def test_legacy_plan_without_acceptance_check_remains_valid():
    plan = create_plan(
        "Старая задача",
        [{
            "id": "step_1",
            "description": "Выполнить задачу",
            "expected_result": "Получен результат",
        }],
    )

    assert validate_plan(plan)
    assert plan["steps"][0]["acceptance_check"] is None


def test_plan_rejects_unknown_acceptance_check():
    with pytest.raises(ValueError, match="acceptance_check"):
        create_plan(
            "Задача",
            [{
                "id": "step_1",
                "description": "Этап",
                "expected_result": "Результат",
                "acceptance_check": {
                    "type": "model_says_success",
                },
            }],
        )


def test_file_exists_check_requires_matching_successful_tool(tmp_path):
    (tmp_path / "result.txt").write_text("done", encoding="utf-8")
    check = {"type": "file_exists", "path": "result.txt"}

    proof = verify_acceptance_check(
        check,
        "write_file",
        {"path": "result.txt", "content": "done"},
        {"path": "result.txt", "bytes_written": 4, "overwritten": False},
        tmp_path,
    )

    assert proof is not None
    assert "result.txt" in proof


def test_file_check_rejects_unrelated_path(tmp_path):
    (tmp_path / "result.txt").write_text("done", encoding="utf-8")

    proof = verify_acceptance_check(
        {"type": "file_exists", "path": "result.txt"},
        "write_file",
        {"path": "other.txt", "content": "done"},
        {"path": "other.txt", "bytes_written": 4, "overwritten": False},
        tmp_path,
    )

    assert proof is None


def test_file_contains_check_reads_actual_file(tmp_path):
    (tmp_path / "report.txt").write_text(
        "Summary: verified",
        encoding="utf-8",
    )

    check = {
        "type": "file_contains",
        "path": "report.txt",
        "text": "verified",
    }
    arguments = {"path": "report.txt", "content": "Summary: verified"}
    result = {"path": "report.txt", "bytes_written": 18, "overwritten": False}

    assert verify_acceptance_check(
        check, "write_file", arguments, result, tmp_path
    ) is not None

    (tmp_path / "report.txt").write_text("Summary: pending", encoding="utf-8")
    assert verify_acceptance_check(
        check, "write_file", arguments, result, tmp_path
    ) is None


def test_file_check_rejects_path_outside_workspace(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside.txt"
    outside.write_text("secret", encoding="utf-8")

    try:
        proof = verify_acceptance_check(
            {"type": "file_exists", "path": f"../{outside.name}"},
            "write_file",
            {
                "path": f"../{outside.name}",
                "content": "secret",
            },
            {
                "path": f"../{outside.name}",
                "bytes_written": 6,
                "overwritten": False,
            },
            tmp_path,
        )
        assert proof is None
    finally:
        outside.unlink(missing_ok=True)


def test_python_check_requires_zero_exit_and_no_timeout():
    check = {"type": "python_exit_code", "path": "check.py"}
    args = {"path": "check.py"}

    assert verify_acceptance_check(
        check,
        "run_python",
        args,
        {
            "script": "check.py",
            "exit_code": 0,
            "timed_out": False,
        },
        Path("."),
    ) is not None

    assert verify_acceptance_check(
        check,
        "run_python",
        args,
        {
            "script": "check.py",
            "exit_code": 1,
            "timed_out": False,
        },
        Path("."),
    ) is None

    assert verify_acceptance_check(
        check,
        "run_python",
        args,
        {
            "script": "check.py",
            "exit_code": None,
            "timed_out": True,
        },
        Path("."),
    ) is None



def test_python_tests_passed_requires_assertions_and_successful_run(tmp_path):
    check = {"type": "python_tests_passed", "path": "check.py"}
    args = {"path": "check.py"}
    script = tmp_path / "check.py"
    script.write_text("assert sum([1, 2, 3]) == 6\\n", encoding="utf-8")
    result = {"script": "check.py", "exit_code": 0, "timed_out": False}
    assert verify_acceptance_check(check, "run_python", args, result, tmp_path) is not None

    script.write_text("print(sum([1, 2, 3]))\\n", encoding="utf-8")
    assert verify_acceptance_check(check, "run_python", args, result, tmp_path) is None
