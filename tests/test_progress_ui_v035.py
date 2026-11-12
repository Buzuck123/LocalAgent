import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from agent import db
from agent import web


def setup_db(monkeypatch, tmp_path):
    database_path = tmp_path / "agent.sqlite3"
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    monkeypatch.setattr(db, "settings", SimpleNamespace(database_path=database_path, workspace_path=workspace_path))
    db.init_db()
    conversation = db.create_conversation("Progress test")
    task = db.create_task(conversation["id"], "Check progress")
    return task


def test_tool_operation_endpoint_returns_summarized_history(monkeypatch, tmp_path):
    task = setup_db(monkeypatch, tmp_path)
    operation_id = "op-success"
    db.create_tool_operation(operation_id, task["id"], "sqlite_query", {"database_path": "agent", "action": "list_tables"})
    db.complete_tool_operation(operation_id, {"table_count": 4, "tables": []})

    response = web.task_operations(task["id"], limit=10, offset=0)
    assert response["total"] == 1
    assert response["operations"][0]["tool_name"] == "sqlite_query"
    assert response["operations"][0]["status"] == "completed"
    assert "database_path=agent" in response["operations"][0]["activity"]


def test_pause_report_separates_success_and_failure(monkeypatch, tmp_path):
    task = setup_db(monkeypatch, tmp_path)
    db.create_tool_operation("op-ok", task["id"], "write_file", {"path": "output.csv"})
    db.complete_tool_operation("op-ok", {"path": "output.csv", "bytes_written": 12})
    db.create_tool_operation("op-bad", task["id"], "sqlite_query", {"database_path": "agent", "action": "sql"})
    db.complete_tool_operation("op-bad", {"__localagent_tool_error__": {"error": "not authorized", "exception_type": "DatabaseError"}})

    checkpoint = {"pause_reason": "tool_call_limit", "plan": {"steps": [
        {"id": "s1", "description": "Создать таблицу", "status": "completed"},
        {"id": "s2", "description": "Проверить результат", "status": "pending"},
    ]}}
    report = web._pause_report(task, checkpoint)
    assert "Достигнут лимит вызовов инструментов" in report
    assert "Успешные операции" in report
    assert "Ошибки" in report
    assert "Создать таблицу" in report
    assert "Проверить результат" in report


def test_activity_panel_is_below_dialog_and_has_operation_history():
    html = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")
    assert html.index('<div id="messages">') < html.index('<div id="activity-panel"')
    assert 'id="activity-history"' in html
    assert "'/api/tasks/'+encodeURIComponent(task.id)+'/operations" in html
    assert html.count('id="activity-panel"') == 1
