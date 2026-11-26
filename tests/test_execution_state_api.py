import json

from fastapi import HTTPException

from agent import web


def test_execution_endpoint_returns_bounded_state_and_plan(monkeypatch):
    checkpoint = {
        "phase": "loop",
        "activity": "Читаю файл",
        "task_state": {
            "actions_total": 4,
            "actions_succeeded": 3,
            "actions_failed": 1,
            "successful_tools": ["read_file", "write_file"],
            "failed_categories": {"missing_resource": 1},
            "last_error": {"tool": "read_file", "category": "missing_resource"},
            "unresolved_errors": [{"tool": "read_file", "category": "missing_resource"}],
        },
        "execution_events": [{"at": "2026-10-28T10:00:00+00:00", "type": "tool_result",
                              "tool": "read_file", "status": "failed", "detail": "файл не найден"}],
        "plan": {"goal": "Сделать результат", "status": "in_progress", "steps": [
            {"id": "s1", "description": "Изучить входные данные", "status": "completed"},
            {"id": "s2", "description": "Создать результат", "status": "pending"},
        ]},
    }
    monkeypatch.setattr(web.db, "get_task", lambda tid: {"id": tid, "checkpoint_json": json.dumps(checkpoint)})

    result = web.task_execution("task-1")

    assert result["task_id"] == "task-1"
    assert result["task_state"]["actions_total"] == 4
    assert result["events"][0]["detail"] == "файл не найден"
    assert result["plan"]["completed_steps"] == 1
    assert result["plan"]["remaining_steps"] == ["Создать результат"]


def test_execution_endpoint_returns_404_for_unknown_task(monkeypatch):
    monkeypatch.setattr(web.db, "get_task", lambda tid: None)
    try:
        web.task_execution("missing")
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("unknown task should return 404")


def test_operation_detail_is_task_scoped_and_redacts_secrets(monkeypatch):
    operation = {
        "operation_id": "op-1",
        "task_id": "task-1",
        "tool_name": "terminal",
        "status": "completed",
        "created_at": "2026-11-26T10:00:00+00:00",
        "updated_at": "2026-11-26T10:00:01+00:00",
        "arguments_json": json.dumps({"command": "echo ok", "api_key": "do-not-show"}),
        "result_json": json.dumps({"exit_code": 0, "stdout": "ok", "password": "hidden"}),
    }
    monkeypatch.setattr(web.db, "get_task", lambda tid: {"id": tid})
    monkeypatch.setattr(web.db, "get_tool_operation", lambda operation_id: operation)

    result = web.task_operation_detail("task-1", "op-1")

    assert result["arguments"]["command"] == "echo ok"
    assert result["arguments"]["api_key"] == "[REDACTED]"
    assert result["result"]["stdout"] == "ok"
    assert result["result"]["password"] == "[REDACTED]"


def test_operation_detail_does_not_cross_task_boundary(monkeypatch):
    operation = {"operation_id": "op-1", "task_id": "other-task"}
    monkeypatch.setattr(web.db, "get_task", lambda tid: {"id": tid})
    monkeypatch.setattr(web.db, "get_tool_operation", lambda operation_id: operation)

    try:
        web.task_operation_detail("task-1", "op-1")
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("operation from another task must not be exposed")
