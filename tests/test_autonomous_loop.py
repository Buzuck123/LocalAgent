import json
from dataclasses import replace

from agent import core, db


def test_submit_always_persists_unified_execution_mode(tmp_path, monkeypatch):
    db_settings = replace(db.settings, database_path=tmp_path / "agent.sqlite3")
    core_settings = replace(
        core.settings,
        database_path=tmp_path / "agent.sqlite3",
        workspace_path=tmp_path / "workspace",
    )
    core_settings.workspace_path.mkdir()
    monkeypatch.setattr(db, "settings", db_settings)
    monkeypatch.setattr(core, "settings", core_settings)
    db.init_db()
    conversation = db.create_conversation("unified mode test")
    agent = core.Agent()
    monkeypatch.setattr(agent, "_run_task", lambda *args: None)
    task = agent.submit(conversation["id"], "Что такое рекурсия?")
    agent.threads[task["id"]].join(timeout=1)
    checkpoint = json.loads(db.get_task(task["id"])["checkpoint_json"])
    assert checkpoint["mode"] == "unified"
    assert checkpoint["version"] == 4


def test_worker_entry_points_delegate_to_same_runner(monkeypatch):
    agent = core.Agent()
    calls = []
    monkeypatch.setattr(agent, "_run_unified_task", lambda task_id, control: calls.append((task_id, control)))
    control = core.TaskControl()
    agent._run_task("task-a", control)
    agent._run_autonomous_task("task-b", control)
    assert calls == [("task-a", control), ("task-b", control)]
