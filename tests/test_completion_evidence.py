from agent.core import _completion_evidence_problem, _tool_result_is_successful
from agent.planning import create_plan, validate_plan


def test_ordinary_step_does_not_require_a_new_tool_call():
    step = {"acceptance_check": None}
    assert _completion_evidence_problem(step, [], None) is None


def test_ordinary_step_can_be_completed_from_existing_facts():
    step = {"acceptance_check": None}
    evidence = [{"tool_name": "read_file", "result": {"content": "found"}}]
    assert _completion_evidence_problem(step, evidence, None) is None


def test_acceptance_check_requires_independent_proof():
    step = {
        "completion_mode": "verified",
        "acceptance_check": {"type": "file_exists", "path": "out.txt"},
    }
    assert _completion_evidence_problem(step, [{"tool_name": "write_file"}], None)
    assert _completion_evidence_problem(step, [], "File actually exists") is None


def test_tool_result_error_is_not_evidence():
    assert _tool_result_is_successful({"error": "permission denied"}) is False
    assert _tool_result_is_successful({"ok": False, "error": "failed"}) is False
    assert _tool_result_is_successful({"exit_code": 2}) is False
    assert _tool_result_is_successful({"exit_code": 0, "timed_out": False}) is True


def test_new_plan_does_not_assign_a_completion_mode():
    plan = create_plan("Do work", [
        {"id": "step_1", "description": "Work", "expected_result": "Done"},
        {"id": "step_2", "description": "Answer", "expected_result": "Text", "completion_mode": "response"},
    ])
    assert validate_plan(plan)
    assert all("completion_mode" not in step for step in plan["steps"])


def test_acceptance_check_is_the_only_machine_completion_gate():
    plan = create_plan("Create file", [{
        "id": "step_1", "description": "Create file", "expected_result": "File exists",
        "completion_mode": "response",
        "acceptance_check": {"type": "file_exists", "path": "out.txt"},
    }])
    assert plan["steps"][0]["acceptance_check"] == {"type": "file_exists", "path": "out.txt"}
    assert "completion_mode" not in plan["steps"][0]


def test_skipped_step_is_a_valid_terminal_plan_state():
    from agent.planning import set_step_status, current_step

    plan = create_plan("Do work", [
        {"id": "step_1", "description": "Optional output", "expected_result": "Optional file"},
        {"id": "step_2", "description": "Answer user", "expected_result": "Answer"},
    ])
    set_step_status(plan, "step_1", "skipped", verification="Пользователю этот результат не требуется.")
    assert plan["steps"][0]["status"] == "skipped"
    assert plan["current_step_id"] == "step_2"
    assert current_step(plan)["id"] == "step_2"
    assert plan["status"] == "in_progress"


def test_terminal_acceptance_uses_private_receipt_metadata_not_model_output():
    import json
    from agent.planning import verify_acceptance_check_state

    message = {
        "role": "tool",
        "tool_name": "terminal",
        "content": "file exists\\n",
        "_localagent_internal": {
            "_terminal_command": "test -f result.txt",
            "state": "completed",
            "exit_code": 0,
            "timed_out": False,
        },
    }
    proof = verify_acceptance_check_state(
        {"type": "file_exists", "path": "result.txt"},
        ".",
        [message],
    )
    assert proof and "result.txt" in proof
