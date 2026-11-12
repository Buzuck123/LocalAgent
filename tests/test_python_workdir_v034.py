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


def test_tool_activity_shows_python_script_and_result_preview():
    from agent import core

    summary = core._tool_activity_summary(
        "run_python",
        {"path": "process_data.py", "timeout_seconds": 60},
        {"ok": True, "result": {
            "script": "process_data.py",
            "exit_code": 0,
            "timed_out": False,
            "stdout": "Rows processed: 2411\nWorkbook verified\n",
            "stderr": "",
            "work_dir": ".agent_runs/task-abc/work",
        }},
    )
    assert "run_python" in summary
    assert "process_data.py" in summary
    assert "exit_code=0" in summary
    assert "Rows processed: 2411" in summary


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


def test_run_python_binds_persistent_workdir_and_accepts_120_seconds(workspace, tmp_path, monkeypatch):
    from types import SimpleNamespace

    project = tmp_path / "project"
    agent_dir = project / "agent"
    venv = project / ".venv"
    (venv / "bin").mkdir(parents=True)
    agent_dir.mkdir()
    (venv / "bin" / "python").write_text("placeholder", encoding="utf-8")
    (venv / "pyvenv.cfg").write_text("home = /usr", encoding="utf-8")
    (workspace / "probe.py").write_text("print('probe')", encoding="utf-8")
    monkeypatch.setattr(tools, "__file__", str(agent_dir / "tools.py"))
    monkeypatch.setattr(tools.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write(b"SCRIPT_RAN\n")
        kwargs["stderr"].write(b"")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tools.subprocess, "run", fake_run)
    first = tools.run_python_sandbox("probe.py", timeout_seconds=120, task_id="task-a")
    work_dir = workspace / first["work_dir"]
    (work_dir / "persist.txt").write_text("persistent", encoding="utf-8")
    second = tools.run_python_sandbox("probe.py", timeout_seconds=120, task_id="task-a")

    command = calls[0][0]
    assert calls[0][1]["timeout"] == 124
    assert "--bind" in command and str(workspace.resolve()) in command
    assert "--tmpfs" in command and "/workspace/.agent_runs" in command
    bind_index = command.index("--bind", command.index("/opt/localagent-venv") + 1)
    assert command[bind_index + 1] == str(work_dir)
    assert command[bind_index + 2] == "/work"
    assert command[command.index("--chdir") + 1] == "/work"
    assert first["exit_code"] == 0 and first["stdout"].strip() == "SCRIPT_RAN"
    assert second["work_dir"] == first["work_dir"]
    assert (workspace / second["work_dir"] / "persist.txt").read_text(encoding="utf-8") == "persistent"
    assert first["output_dir"] != second["output_dir"]
