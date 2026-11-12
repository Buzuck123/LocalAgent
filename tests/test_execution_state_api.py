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
