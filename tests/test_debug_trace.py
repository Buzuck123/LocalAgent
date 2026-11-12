import json
from types import SimpleNamespace

from agent import db


def _database(monkeypatch, tmp_path, *, enabled=True, thinking=False):
    path = tmp_path / "trace.sqlite3"
    monkeypatch.setattr(db, "settings", SimpleNamespace(
        database_path=path, debug_llm_trace=enabled, debug_llm_thinking=thinking,
    ))
    db.init_db()
    conversation = db.create_conversation("Trace test")
    task = db.create_task(conversation["id"], "Test trace")
    return conversation, task


def test_trace_is_disabled_by_default(monkeypatch, tmp_path):
    _, task = _database(monkeypatch, tmp_path, enabled=False)
    assert db.record_debug_trace(task["id"], "llm_request", {"prompt": "secret"}) is None
    assert db.list_debug_trace(task["id"]) == []


def test_trace_records_unique_events_and_redacts_secrets(monkeypatch, tmp_path):
    _, task = _database(monkeypatch, tmp_path, enabled=True)
    one = db.record_debug_trace(task["id"], "llm_request", {
        "prompt": "Authorization: Bearer abc.def",
        "auth_token": "sensitive-value",
        "command": "curl https://example.invalid?token=abc123",
    })
    two = db.record_debug_trace(task["id"], "tool_result", {"ok": True})
    events = db.list_debug_trace(task["id"])
    assert one and two and one != two
    assert [item["event_id"] for item in events] == [one, two]
    rendered = json.dumps(events, ensure_ascii=False)
    assert "sensitive-value" not in rendered
    assert "abc.def" not in rendered
    assert "abc123" not in rendered


def test_thinking_requires_both_trace_and_thinking_flags(monkeypatch, tmp_path):
    _, task = _database(monkeypatch, tmp_path, enabled=True, thinking=False)
    assert db.record_debug_trace(task["id"], "llm_thinking", {"thinking": "not stored"}, thinking=True) is None
    assert db.list_debug_trace(task["id"]) == []


def test_trace_rows_cascade_when_conversation_is_deleted(monkeypatch, tmp_path):
    conversation, task = _database(monkeypatch, tmp_path, enabled=True)
    db.record_debug_trace(task["id"], "tool_call", {"name": "terminal"})
    assert db.list_debug_trace(task["id"])
    assert db.delete_conversation(conversation["id"])
    with db.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM debug_trace").fetchone()[0] == 0
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []


def test_task_status_transitions_are_trace_events_when_enabled(monkeypatch, tmp_path):
    _, task = _database(monkeypatch, tmp_path, enabled=True)
    db.update_task(task["id"], status="running")
    events = db.list_debug_trace(task["id"])
    assert any(item["event_type"] == "task_status" and item["payload"]["status"] == "running" for item in events)


def test_debug_trace_settings_api_persists_ui_override(monkeypatch, tmp_path):
    from agent import web
    from types import SimpleNamespace
    _database(monkeypatch, tmp_path, enabled=False)
    monkeypatch.setattr(web, "settings", SimpleNamespace(debug_llm_trace=False, debug_llm_thinking=False))
    assert web.get_debug_trace_settings()["enabled"] is False
    saved = web.update_debug_trace_settings(web.DebugTraceSettings(enabled=True, thinking=True))
    assert saved["enabled"] is True and saved["thinking"] is True
    assert db._debug_setting_enabled("debug_llm_trace", False) is True
    assert db._debug_setting_enabled("debug_llm_thinking", False) is True


def test_trace_export_api_returns_task_scoped_events(monkeypatch, tmp_path):
    from agent import web
    _, task = _database(monkeypatch, tmp_path, enabled=True)
    db.record_debug_trace(task["id"], "controller_checkpoint", {"phase": "loop"})
    exported = web.export_task_trace(task["id"])
    assert exported["task_id"] == task["id"]
    assert exported["events"][0]["event_type"] == "controller_checkpoint"
