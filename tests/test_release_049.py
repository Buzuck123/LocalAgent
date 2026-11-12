from pathlib import Path
from types import SimpleNamespace

from agent import core, tools


def test_dispatcher_passes_task_context_for_work_shorthand(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        tools,
        "settings",
        SimpleNamespace(workspace_path=workspace, database_path=tmp_path / "agent.sqlite3"),
    )
    task_id = "release-049-work-alias"

    result = core._dispatch_tool_for_task(
        "write_file",
        {"path": "work/script.py", "content": "print('ok')"},
        task_id,
    )

    assert result["namespace"] == "work"
    assert result["bytes_written"] > 0
    expected = workspace / ".agent_runs" / f"task-{task_id}" / "work" / "script.py"
    assert expected.read_text(encoding="utf-8") == "print('ok')"


def test_dispatcher_keeps_workspace_relative_paths_on_legacy_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(core, "dispatch_tool", lambda name, args, **kwargs: calls.append(kwargs) or {"ok": True})

    core._dispatch_tool_for_task("write_file", {"path": "script.py", "content": "x"}, "task-1")

    assert calls == [{}]



def test_dispatcher_passes_task_context_for_work_database_path(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        tools,
        "settings",
        SimpleNamespace(workspace_path=workspace, database_path=tmp_path / "agent.sqlite3"),
    )
    task_id = "release-049-db-alias"

    result = core._dispatch_tool_for_task(
        "sqlite_create_database",
        {"database_path": "work/task.sqlite3"},
        task_id,
    )

    assert result["created"] is True
    expected = workspace / ".agent_runs" / f"task-{task_id}" / "work" / "task.sqlite3"
    assert expected.is_file()

def test_ollama_chat_does_not_override_model_tag_parameters(monkeypatch):
    expected_settings = SimpleNamespace(
        model="ornith-1.5:9b",
        ollama_url="http://127.0.0.1:11434",
        keep_alive="10m",
        num_ctx=65536,
        num_gpu=99,
        num_predict=32768,
        temperature=0.7,
    )
    monkeypatch.setattr(core, "settings", expected_settings)
    monkeypatch.setattr(core.db, "get_app_setting", lambda key, default=None: default)
    client = core.OllamaClient()
    captured = {}

    def fake_post(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"message": {"role": "assistant", "content": "ok"}}

    monkeypatch.setattr(client, "_post", fake_post)
    result = client.chat([{"role": "user", "content": "hello"}], reasoning=False)

    assert result["message"]["content"] == "ok"
    assert captured["path"] == "/api/chat"
    assert captured["payload"]["model"] == "ornith-1.5:9b"
    assert captured["payload"]["think"] is False
    assert "options" not in captured["payload"]
