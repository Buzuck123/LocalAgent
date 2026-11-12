from pathlib import Path
from types import SimpleNamespace

from agent import core, tools


def _tool_names(definitions):
    return {item["function"]["name"] for item in definitions}


def test_python_workflow_uses_terminal_and_explicit_file_publishing():
    names = _tool_names(core.tools_for_task("Создай Python-скрипт, запусти его и проверь результат"))
    assert "terminal" in names and "publish_file" in names
    assert not {"run_python", "copy_file", "publish_workspace_file"} & names


def test_adjacent_assistant_text_is_merged_and_followed_by_user_turn():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "do work"},
        {"role": "assistant", "content": "Первый фрагмент"},
        {"role": "assistant", "content": "Второй фрагмент"},
    ]
    prepared = core._prepare_messages_for_next_model_turn(messages, "Исправь оставшуюся проблему")
    assert [m["role"] for m in prepared][-2:] == ["assistant", "user"]
    assert prepared[-2]["content"] == "Первый фрагмент\n\nВторой фрагмент"
    assert prepared[-1]["content"] == "Исправь оставшуюся проблему"


def test_tool_call_messages_are_not_silently_merged():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file"}}]},
        {"role": "assistant", "content": "unexpected"},
    ]
    try:
        core._prepare_messages_for_next_model_turn(messages)
    except ValueError as exc:
        assert "два сообщения assistant" in str(exc)
    else:
        raise AssertionError("Malformed tool-call history must not be silently rewritten")


def test_planner_cannot_turn_its_own_filename_into_a_required_acceptance_path():
    steps = [{"acceptance_check": {"type": "file_exists", "path": "/work/create_excel.py"}}]
    core._remove_unrequested_acceptance_paths(
        "Создай Python-скрипт в /work и запусти его", steps
    )
    assert steps[0]["acceptance_check"]["path"] == "/work/create_excel.py"


def test_explicit_user_filename_keeps_acceptance_check():
    steps = [{"acceptance_check": {"type": "file_exists", "path": "/work/report.py"}}]
    core._remove_unrequested_acceptance_paths(
        "Создай /work/report.py и запусти его", steps
    )
    assert steps[0]["acceptance_check"]["path"] == "/work/report.py"


def test_copy_file_overwrite_keeps_backup_of_workspace_file(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        tools, "settings",
        SimpleNamespace(workspace_path=workspace, database_path=tmp_path / "agent.sqlite3"),
    )
    task_id = "release-0410-copy"
    work_dir, _ = tools._task_work_directory(workspace, task_id)
    source = work_dir / "result.txt"
    source.write_text("new content", encoding="utf-8")
    destination = workspace / "result.txt"
    destination.write_text("old content", encoding="utf-8")

    result = tools.copy_workspace_file("/work/result.txt", "result.txt", True, task_id=task_id)

    assert result["verified"] is True
    assert result["overwritten"] is True
    assert destination.read_text(encoding="utf-8") == "new content"
    backup = workspace / result["backup_path"]
    assert backup.read_text(encoding="utf-8") == "old content"


def test_ollama_model_selection_requires_installed_model(monkeypatch):
    client = core.OllamaClient()
    monkeypatch.setattr(client, "models", lambda: {"models": [{"name": "ornith:9b"}], "selected_model": client.model})
    result = client.select_model("ornith:9b")
    assert result == {"selected_model": "ornith:9b"}
    try:
        client.select_model("not-installed:latest")
    except ValueError as exc:
        assert "не установлена" in str(exc)
    else:
        raise AssertionError("An unavailable Ollama model must be rejected")
