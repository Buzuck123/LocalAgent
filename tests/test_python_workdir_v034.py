from types import SimpleNamespace

import pytest

from agent import tools


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=root.resolve()))
    return root


def test_task_workdir_is_persistent_and_isolated(workspace):
    work_a, rel_a = tools._task_work_directory(workspace, "task-a")
    (work_a / "created.xlsx").write_bytes(b"book")

    work_a_again, rel_a_again = tools._task_work_directory(workspace, "task-a")
    work_b, _ = tools._task_work_directory(workspace, "task-b")

    assert work_a_again == work_a
    assert rel_a_again == rel_a
    assert (work_a_again / "created.xlsx").read_bytes() == b"book"
    assert work_b != work_a
    assert not (work_b / "created.xlsx").exists()


def test_publish_file_from_task_work_to_workspace(workspace):
    work, _ = tools._task_work_directory(workspace, "task-a")
    (work / "result.txt").write_text("verified result\n", encoding="utf-8")

    (workspace / "reports").mkdir()
    result = tools.dispatch_tool(
        "publish_workspace_file",
        {"source_path": "result.txt", "destination_path": "reports/result.txt"},
        task_id="task-a",
    )

    assert result["verified"] is True
    assert result["destination_path"] == "reports/result.txt"
    assert result["bytes_written"] == len(b"verified result\n")
    assert (workspace / "reports/result.txt").read_text(encoding="utf-8") == "verified result\n"


def test_publish_does_not_overwrite_without_explicit_permission(workspace):
    work, _ = tools._task_work_directory(workspace, "task-a")
    (work / "result.txt").write_text("new", encoding="utf-8")
    (workspace / "result.txt").write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError):
        tools.dispatch_tool(
            "publish_workspace_file",
            {"source_path": "result.txt", "destination_path": "result.txt"},
            task_id="task-a",
        )

    assert (workspace / "result.txt").read_text(encoding="utf-8") == "original"


def test_publish_overwrite_creates_backup(workspace):
    work, _ = tools._task_work_directory(workspace, "task-a")
    (work / "result.txt").write_text("new", encoding="utf-8")
    target = workspace / "result.txt"
    target.write_text("original", encoding="utf-8")

    result = tools.dispatch_tool(
        "publish_workspace_file",
        {"source_path": "result.txt", "destination_path": "result.txt", "overwrite": True},
        task_id="task-a",
    )

    assert target.read_text(encoding="utf-8") == "new"
    assert result["overwritten"] is True
    backup = workspace / result["backup_path"]
    assert backup.read_text(encoding="utf-8") == "original"


def test_publish_rejects_reserved_agent_runs_destination(workspace):
    work, _ = tools._task_work_directory(workspace, "task-a")
    (work / "result.txt").write_text("new", encoding="utf-8")

    with pytest.raises(ValueError, match="reserved"):
        tools.dispatch_tool(
            "publish_workspace_file",
            {"source_path": "result.txt", "destination_path": ".agent_runs/evil.txt"},
            task_id="task-a",
        )


def test_publish_requires_existing_task_context(workspace):
    with pytest.raises(ValueError, match="current task context"):
        tools.dispatch_tool(
            "publish_workspace_file",
            {"source_path": "result.txt", "destination_path": "result.txt"},
        )


def test_terminal_schema_and_publish_tool_replace_old_python_and_copy_tools():
    definitions = {item["function"]["name"]: item["function"] for item in tools.TOOL_DEFINITIONS}
    terminal = definitions["terminal"]["parameters"]
    timeout = terminal["properties"]["timeout"]
    assert timeout["minimum"] == 1 and timeout["maximum"] == 600
    assert "command" in terminal["properties"] and "required" not in terminal
    assert "publish_file" in definitions
    assert "run_python" not in definitions and "copy_file" not in definitions


def test_tool_activity_shows_publish_destination():
    from agent import core

    summary = core._tool_activity_summary(
        "publish_workspace_file",
        {"source_path": "report.xlsx", "destination_path": "report.xlsx"},
        {"ok": True, "result": {
            "destination_path": "report.xlsx",
            "bytes_written": 1024,
            "verified": True,
            "backup_path": None,
        }},
    )
    assert "publish_workspace_file" in summary
    assert "report.xlsx" in summary
    assert "1024" in summary
