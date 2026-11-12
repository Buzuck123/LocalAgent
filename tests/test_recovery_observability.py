import json
from dataclasses import replace

from agent import core, db


def _isolated_agent(tmp_path, monkeypatch, max_tool_rounds=10):
    db_settings = replace(db.settings, database_path=tmp_path / "agent.sqlite3")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    core_settings = replace(
        core.settings,
        database_path=tmp_path / "agent.sqlite3",
        workspace_path=workspace,
        max_tool_rounds=max_tool_rounds,
        loop_warning_threshold=3,
        loop_pause_threshold=6,
    )
    monkeypatch.setattr(db, "settings", db_settings)
    monkeypatch.setattr(core, "settings", core_settings)
    db.init_db()
    conversation = db.create_conversation("recovery test")
    return conversation, core.Agent()


def test_tool_activity_label_shows_safe_path_but_not_file_contents():
    label = core._tool_activity_label(
        "write_file",
        {"path": "reports/result.xlsx", "content": "PRIVATE CONTENT MUST NOT APPEAR"},
    )
    assert "write_file" in label
    assert "reports/result.xlsx" in label
    assert "PRIVATE CONTENT MUST NOT APPEAR" not in label


def test_execution_alias_uses_unified_runner(monkeypatch):
    agent = core.Agent()
    calls = []
    monkeypatch.setattr(agent, "_run_unified_task", lambda task_id, control: calls.append(task_id))
    agent._run_task("unified-task", core.TaskControl())
    agent._run_autonomous_task("compat-task", core.TaskControl())
    assert calls == ["unified-task", "compat-task"]
