import json
from pathlib import Path

import pytest

from agent.planning import verify_acceptance_check, verify_acceptance_check_state
from agent.tools import _task_work_directory
from agent.core import _validate_plan_acceptance_paths


def test_plan_acceptance_allows_safe_work_path():
    _validate_plan_acceptance_paths([
        {"acceptance_check": {"type": "python_exit_code", "path": "/work/script.py"}}
    ])


def test_plan_acceptance_accepts_paths_in_the_container_filesystem():
    _validate_plan_acceptance_paths([
        {"acceptance_check": {"type": "file_exists", "path": "/tmp/outside.txt"}},
        {"acceptance_check": {"type": "file_exists", "path": "../relative-output.txt"}},
    ])


def test_work_file_acceptance_check_verifies_actual_file(tmp_path):
    task_id = "task-test-acceptance"
    work_dir, _ = _task_work_directory(tmp_path, task_id)
    output = work_dir / "result.csv"
    output.write_text("name,value\nA,1\n", encoding="utf-8")
    check = {"type": "file_contains", "path": "/work/result.csv", "text": "A,1"}
    args = {"path": "/work/result.csv", "content": "name,value\nA,1\n"}
    result = {"path": "/work/result.csv", "bytes_written": output.stat().st_size}

    proof = verify_acceptance_check(check, "write_file", args, result, tmp_path, task_id)
    assert proof is not None
    assert verify_acceptance_check_state(check, tmp_path, [], task_id) is not None


def test_work_python_script_acceptance_uses_tool_receipt(tmp_path):
    task_id = "task-test-python-check"
    work_dir, _ = _task_work_directory(tmp_path, task_id)
    script = work_dir / "script.py"
    script.write_text("assert 1 + 1 == 2\n", encoding="utf-8")
    check = {"type": "python_tests_passed", "path": "/work/script.py"}
    args = {"path": "/work/script.py"}
    result = {"script": "/work/script.py", "exit_code": 0, "timed_out": False}
    assert verify_acceptance_check(check, "run_python", args, result, tmp_path, task_id) is not None

    messages = [{
        "role": "tool",
        "tool_name": "run_python",
        "content": json.dumps({"ok": True, "result": result}),
    }]
    assert verify_acceptance_check_state(check, tmp_path, messages, task_id) is not None


def test_work_acceptance_rejects_symlink_escape(tmp_path):
    task_id = "task-test-symlink"
    work_dir, _ = _task_work_directory(tmp_path, task_id)
    outside = tmp_path.parent / f"{tmp_path.name}_outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = work_dir / "link.txt"
    link.symlink_to(outside)
    try:
        check = {"type": "file_exists", "path": "/work/link.txt"}
        assert verify_acceptance_check_state(check, tmp_path, [], task_id) is None
    finally:
        outside.unlink(missing_ok=True)
