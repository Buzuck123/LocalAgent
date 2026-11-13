from agent.orchestration import (
    ToolContractError,
    append_execution_event,
    classify_failure,
    update_task_state,
    validate_tool_call,
)


DEFINITIONS = [
    {"type": "function", "function": {
        "name": "sample", "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["path"], "additionalProperties": False,
        },
    }},
]


def test_tool_contract_accepts_valid_arguments():
    result = validate_tool_call("sample", {"path": "data.txt", "limit": 5}, DEFINITIONS)
    assert result["schema_validated"] is True


def test_tool_contract_rejects_missing_required_argument():
    try:
        validate_tool_call("sample", {"limit": 5}, DEFINITIONS)
    except ToolContractError as exc:
        assert "missing required" in str(exc)
    else:
        raise AssertionError("missing required argument was accepted")


def test_tool_contract_rejects_unknown_field_and_bad_type():
    for arguments, expected in [({"path": "x", "extra": True}, "unexpected field"),
                                ({"path": 12}, "expected string"),
                                ({"path": "x", "limit": 11}, "exceeds maximum")]:
        try:
            validate_tool_call("sample", arguments, DEFINITIONS)
        except ToolContractError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"invalid arguments accepted: {arguments}")


def test_unknown_tool_is_rejected():
    try:
        validate_tool_call("not_registered", {}, DEFINITIONS)
    except ToolContractError as exc:
        assert "Unknown tool" in str(exc)
    else:
        raise AssertionError("unknown tool was accepted")


def test_failure_classifier_distinguishes_mutating_path_error():
    info = classify_failure("terminal", {"command": "echo x > /work/x.py"}, "ValueError: Path escapes the workspace directory")
    assert info["category"] == "access_boundary"
    assert info["tool_effect"] == "mutating"
    assert info["retry_safe_without_state_check"] is False


def test_failure_classifier_marks_read_only_service_retry_policy():
    info = classify_failure("web_search", {"query": "test"}, "ConnectionError: service temporarily unavailable")
    assert info["category"] == "transient_service"
    assert info["tool_effect"] == "read_only"
    assert info["retry_safe_without_state_check"] is True


def test_execution_events_are_bounded_and_task_state_tracks_outcomes():
    checkpoint = {"execution_events": [{"type": "old"}] * 100}
    append_execution_event(checkpoint, "tool_result", tool_name="read_file", detail="read ok", status="succeeded")
    assert len(checkpoint["execution_events"]) == 100
    update_task_state(checkpoint, tool_name="read_file", arguments={"path": "x"}, success=True)
    update_task_state(checkpoint, tool_name="write_file", arguments={"path": "../x"}, success=False,
                      failure={"category": "access_boundary", "error_excerpt": "outside", "recovery_action": "replan"})
    state = checkpoint["task_state"]
    assert state["actions_total"] == 2
    assert state["actions_succeeded"] == 1
    assert state["actions_failed"] == 1
    assert state["failed_categories"]["access_boundary"] == 1
    assert state["last_error"]["recovery_action"] == "replan"


def test_mutating_tool_is_not_marked_safe_to_retry_after_transient_error():
    info = classify_failure("terminal", {"command": "echo update"},
                            "ConnectionError: temporarily unavailable")
    assert info["category"] == "transient_service"
    assert info["tool_effect"] == "mutating"
    assert info["retry_safe_without_state_check"] is False


def test_matching_successful_terminal_action_resolves_prior_missing_file_error():
    checkpoint = {}
    update_task_state(
        checkpoint,
        tool_name="load_file_for_model",
        arguments={"path": "/work/inventory.csv"},
        success=False,
        failure={
            "category": "missing_resource",
            "error_excerpt": "FileNotFoundError: [Errno 2] No such file or directory (ENOENT): '/work/inventory.csv'",
            "recovery_action": "discover_actual_resources",
        },
    )
    unresolved = checkpoint["task_state"]["unresolved_errors"]
    assert len(unresolved) == 1
    assert unresolved[0]["arguments"] == {"path": "/work/inventory.csv"}

    update_task_state(
        checkpoint,
        tool_name="terminal",
        arguments={"command": "printf sku,qty > /work/inventory.csv"},
        success=True,
    )
    assert checkpoint["task_state"]["unresolved_errors"] == []


def test_unrelated_success_does_not_resolve_missing_file_error():
    checkpoint = {}
    update_task_state(
        checkpoint,
        tool_name="read_file",
        arguments={"path": "inventory.csv"},
        success=False,
        failure={"category": "missing_resource", "error_excerpt": "FileNotFoundError: not found"},
    )
    update_task_state(
        checkpoint,
        tool_name="write_file",
        arguments={"path": "README.md"},
        success=True,
    )
    assert len(checkpoint["task_state"]["unresolved_errors"]) == 1


def test_terminal_is_conservatively_mutating_and_package_install_repairs_import_failure():
    info = classify_failure("terminal", {"command": "python script.py"},
                            "ModuleNotFoundError: No module named 'openpyxl'")
    assert info["tool_effect"] == "mutating"
    assert info["retry_safe_without_state_check"] is False

    checkpoint = {}
    update_task_state(checkpoint, tool_name="terminal", arguments={"command": "python script.py"},
                      success=False, failure={
                          "category": "environment_dependency",
                          "error_excerpt": "ModuleNotFoundError: No module named 'openpyxl'",
                          "recovery_action": "check_available_runtime",
                      })
    update_task_state(checkpoint, tool_name="terminal", arguments={"command": "pip install openpyxl"},
                      success=True)
    assert checkpoint["task_state"]["unresolved_errors"] == []


def test_failure_classifier_does_not_mislabel_timed_out_false_as_timeout():
    info = classify_failure(
        "terminal", {"command": "python check.py"},
        '{"exit_code": 1, "timed_out": false, "stderr": "ValueError: bad data"}',
    )
    assert info["category"] == "verification_failure"
    assert info["recovery_action"] == "inspect_evidence_and_fix_cause"


def test_failure_classifier_recognizes_actual_structured_timeout():
    info = classify_failure(
        "terminal", {"command": "python slow.py"},
        '{"exit_code": 124, "timed_out": true, "stderr": "command timed out"}',
    )
    assert info["category"] == "timeout"


def test_simple_action_evidence_is_migrated_to_first_planned_step_idempotently():
    from agent.orchestration import migrate_simple_action_evidence_to_plan

    plan = {
        "steps": [
            {"id": "step_1", "status": "in_progress"},
            {"id": "step_2", "status": "pending"},
        ]
    }
    evidence = {
        "simple_action": [
            {"tool_name": "terminal", "arguments_summary": '{"command":"pwd"}', "result_summary": '{"exit_code":0}'},
            {"tool_name": "terminal", "arguments_summary": '{"command":"ls"}', "result_summary": '{"exit_code":0}'},
        ]
    }

    assert migrate_simple_action_evidence_to_plan(plan, evidence) == 2
    assert len(evidence["step_1"]) == 2
    assert all(item["evidence_scope"] == "task_level_before_planning" for item in evidence["step_1"])
    assert all(item["migrated_from_step"] == "simple_action" for item in evidence["step_1"])
    assert migrate_simple_action_evidence_to_plan(plan, evidence) == 0
    assert len(evidence["step_1"]) == 2
