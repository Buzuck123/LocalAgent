
import pytest

from agent import db, tools, core


def _setup(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    from types import SimpleNamespace
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=workspace, database_path=tmp_path / "agent.sqlite3"))
    monkeypatch.setattr(db, "settings", SimpleNamespace(database_path=tmp_path / "agent.sqlite3"))
    db.init_db()
    cid = db.create_conversation("usage fixture")["id"]
    ts = db.now()
    with db.connect() as con:
        con.execute("INSERT INTO tasks(id,conversation_id,user_message,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                    ("t1", cid, "usage fixture", "queued", ts, ts))
    return workspace


def test_virtual_workspace_and_work_paths_are_consistent(tmp_path, monkeypatch):
    workspace = _setup(tmp_path, monkeypatch)
    task_id = "task-test-048"
    tools.dispatch_tool("create_directory", {"path": "/work/data"}, task_id=task_id)
    tools.dispatch_tool("write_file", {"path": "/work/data/a.txt", "content": "one"}, task_id=task_id)
    tools.dispatch_tool("write_file", {"path": "/work/data/a.txt", "content": "two"}, task_id=task_id)
    work_file = workspace / ".agent_runs" / ("task-" + task_id) / "work" / "data" / "a.txt"
    assert work_file.read_text() == "two"
    assert tools.dispatch_tool("read_file", {"path": "/work/data/a.txt"}, task_id=task_id)["content"] == "two"


def test_workspace_absolute_virtual_path(tmp_path, monkeypatch):
    workspace = _setup(tmp_path, monkeypatch)
    (workspace / "a.txt").write_text("ok")
    assert tools.dispatch_tool("read_file", {"path": "/workspace/a.txt"}, task_id="x")["content"] == "ok"


def test_workspace_write_still_protects_existing_file(tmp_path, monkeypatch):
    workspace = _setup(tmp_path, monkeypatch)
    (workspace / "a.txt").write_text("old")
    with pytest.raises(FileExistsError):
        tools.dispatch_tool("write_file", {"path": "/workspace/a.txt", "content": "new"}, task_id="x")
    assert (workspace / "a.txt").read_text() == "old"


def test_path_traversal_rejected(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        tools.dispatch_tool("write_file", {"path": "/work/../escape.txt", "content": "x"}, task_id="x")
    with pytest.raises(ValueError):
        tools.dispatch_tool("read_file", {"path": "/etc/passwd"}, task_id="x")


def test_model_usage_persisted_and_aggregated(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    db.record_model_request("t1", "qwen3.5:9b", {
        "prompt_eval_count": 1200, "eval_count": 100,
        "prompt_eval_duration": 1_000_000, "eval_duration": 2_000_000,
    })
    db.record_model_request("t1", "qwen3.5:9b", {
        "prompt_eval_count": 1800, "eval_count": 150,
    })
    usage = db.get_model_usage("t1")
    assert usage["requests"] == 2
    assert usage["prompt_tokens_total"] == 3000
    assert usage["generated_tokens_total"] == 250
    assert usage["current_prompt_tokens"] == 1800
    assert usage["max_prompt_tokens"] == 1800


def test_tool_scope_only_contains_registered_tools():
    definitions = core.tools_for_task("Создай Python-скрипт в /work и скопируй результат в workspace")
    names = {item["function"]["name"] for item in definitions}
    registry = {item["function"]["name"] for item in tools.TOOL_DEFINITIONS}
    assert names <= registry
    assert "terminal" in names and "publish_file" in names
    assert "copy_file" not in names and "publish_workspace_file" not in names


def test_planner_path_binding_does_not_force_workspace_namespace():
    result = core._bind_task_prompt_to_acceptance_check(
        {"acceptance_check": {"type": "file_exists", "path": "/work/result.csv"}},
        "Создай результат",
    )
    assert "/work/result.csv" in result
    assert "внутри workspace" not in result


def test_tool_introspection_receives_entire_registry():
    definitions = core.tools_for_task("Подробно расскажи про все твои инструменты, как их вызывать и как они работают. Сами инструменты не вызывай, ответь текстом.")
    assert {x["function"]["name"] for x in definitions} == {x["function"]["name"] for x in tools.TOOL_DEFINITIONS}


def test_acceptance_resolves_workspace_virtual_absolute_path(tmp_path):
    from agent.planning import _acceptance_path_candidate
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "report.txt").write_text("ok")
    assert _acceptance_path_candidate("/workspace/report.txt", workspace) == workspace / "report.txt"


def test_copy_between_virtual_namespaces_and_protect_reserved_area(tmp_path, monkeypatch):
    workspace = _setup(tmp_path, monkeypatch)
    task_id = "copy-test"
    tools.dispatch_tool("write_file", {"path": "/work/source.txt", "content": "payload"}, task_id=task_id)
    result = tools.dispatch_tool("copy_file", {"source": "/work/source.txt", "destination": "/workspace/copied.txt"}, task_id=task_id)
    assert result["verified"] is True
    assert (workspace / "copied.txt").read_text() == "payload"
    with pytest.raises(ValueError, match="reserved"):
        tools.dispatch_tool("copy_file", {"source": "/work/source.txt", "destination": "/workspace/.agent_runs/x.txt"}, task_id=task_id)


def test_existing_database_migration_keeps_messages_and_adds_usage_table(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    cid = db.create_conversation("migration")['id']
    db.add_message(cid, "user", "keep this")
    db.init_db()
    assert db.get_messages(cid)[0]["content"] == "keep this"
    with db.connect() as con:
        columns = {row[1] for row in con.execute("PRAGMA table_info(model_requests)")}
    assert {"task_id", "prompt_eval_count", "eval_count", "prompt_eval_duration_ns", "eval_duration_ns"} <= columns


