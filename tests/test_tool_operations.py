import json

import pytest

from agent import db


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Каждый тест использует отдельную временную SQLite-базу."""
    from dataclasses import replace

    test_settings = replace(
        db.settings,
        database_path=tmp_path / "test_agent.sqlite3",
    )
    monkeypatch.setattr(db, "settings", test_settings)
    db.init_db()
    # Journal rows are task-scoped and must reference real tasks now that the
    # production schema enforces ON DELETE CASCADE.
    cid = db.create_conversation("fixture") ["id"]
    ts = db.now()
    with db.connect() as con:
        for task_id in ("task-001", "task-002", "task-args"):
            con.execute("INSERT INTO tasks(id,conversation_id,user_message,status,created_at,updated_at) VALUES(?,?,?,'queued',?,?)",
                        (task_id, cid, "fixture task", ts, ts))


def test_register_operation_creates_running_record(isolated_db):
    operation_id = "op-001"

    db.create_tool_operation(
        operation_id=operation_id,
        task_id="task-001",
        tool_name="write_file",
    )

    operation = db.get_tool_operation(operation_id)

    assert operation is not None
    assert operation["operation_id"] == operation_id
    assert operation["task_id"] == "task-001"
    assert operation["tool_name"] == "write_file"
    assert operation["status"] == "running"
    assert operation["result_json"] is None


def test_operation_id_cannot_be_registered_twice(isolated_db):
    db.create_tool_operation(
        operation_id="op-001",
        task_id="task-001",
        tool_name="write_file",
    )

    with pytest.raises(Exception):
        db.create_tool_operation(
            operation_id="op-001",
            task_id="task-002",
            tool_name="write_file",
        )


def test_completed_operation_preserves_result_after_reconnect(isolated_db):
    operation_id = "op-002"
    expected_result = {
        "path": "result.txt",
        "bytes_written": 8,
        "overwritten": False,
    }

    db.create_tool_operation(
        operation_id=operation_id,
        task_id="task-001",
        tool_name="write_file",
    )

    db.complete_tool_operation(operation_id, expected_result)

    # Повторное обращение к БД имитирует чтение после перезапуска.
    operation = db.get_tool_operation(operation_id)

    assert operation["status"] == "completed"
    assert json.loads(operation["result_json"]) == expected_result


def test_unknown_operation_cannot_be_completed(isolated_db):
    with pytest.raises(Exception):
        db.complete_tool_operation(
            "unknown-operation",
            {"path": "result.txt"},
        )


def test_running_operation_is_not_treated_as_completed(isolated_db):
    operation_id = "op-003"

    db.create_tool_operation(
        operation_id=operation_id,
        task_id="task-001",
        tool_name="write_file",
    )

    operation = db.get_tool_operation(operation_id)

    assert operation["status"] == "running"
    assert operation["result_json"] is None
    assert operation["status"] != "completed"


def test_completed_operation_cannot_be_completed_again(isolated_db):
    operation_id = "op-004"

    db.create_tool_operation(
        operation_id=operation_id,
        task_id="task-001",
        tool_name="write_file",
    )
    db.complete_tool_operation(operation_id, {"path": "first.txt"})

    with pytest.raises(ValueError):
        db.complete_tool_operation(
            operation_id,
            {"path": "second.txt"},
        )

    operation = db.get_tool_operation(operation_id)
    assert operation["status"] == "completed"
    assert json.loads(operation["result_json"]) == {"path": "first.txt"}


def test_running_operation_remains_ambiguous_after_db_reopen(isolated_db):
    operation_id = "op-005"

    db.create_tool_operation(
        operation_id=operation_id,
        task_id="task-001",
        tool_name="write_file",
    )

    operation = db.get_tool_operation(operation_id)

    assert operation["status"] == "running"
    assert operation["result_json"] is None


def test_operation_journal_stores_safe_argument_summary(isolated_db):
    operation = db.create_tool_operation(
        operation_id="op-args",
        task_id="task-args",
        tool_name="write_file",
        arguments={"path": "result.txt", "overwrite": False, "content": "private document contents"},
    )

    arguments = json.loads(operation["arguments_json"])
    assert arguments["path"] == "result.txt"
    assert arguments["overwrite"] is False
    assert arguments["content"]["omitted_from_journal"] is True
    assert arguments["content"]["sha256"]
    assert "private document contents" not in operation["arguments_json"]


def test_init_db_migrates_existing_tool_operations_table(tmp_path, monkeypatch):
    import sqlite3
    from dataclasses import replace

    test_settings = replace(
        db.settings,
        database_path=tmp_path / "legacy.sqlite3",
    )
    monkeypatch.setattr(db, "settings", test_settings)
    with sqlite3.connect(test_settings.database_path) as con:
        con.execute("""CREATE TABLE tool_operations (
            operation_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('running', 'completed')),
            result_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
    db.init_db()
    with db.connect() as con:
        columns = {row[1] for row in con.execute("PRAGMA table_info(tool_operations)")}
    assert "arguments_json" in columns
