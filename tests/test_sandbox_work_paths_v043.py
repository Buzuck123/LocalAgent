from types import SimpleNamespace

import pytest

from agent import tools


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=root.resolve()))
    return root


def test_work_file_write_read_list_and_edit_are_task_scoped(workspace):
    created = tools.dispatch_tool(
        "create_directory", {"path": "/work/scripts"}, task_id="task-a"
    )
    assert created["created"] == "/work/scripts"

    result = tools.dispatch_tool(
        "write_file",
        {"path": "/work/scripts/read_excel.py", "content": "print('first')\n"},
        task_id="task-a",
    )
    assert result["path"] == "/work/scripts/read_excel.py"

    read = tools.dispatch_tool(
        "read_file", {"path": "/work/scripts/read_excel.py"}, task_id="task-a"
    )
    assert read["content"] == "print('first')\n"

    listing = tools.dispatch_tool("list_files", {"path": "/work/scripts"}, task_id="task-a")
    assert [item["name"] for item in listing["entries"]] == ["read_excel.py"]

    edited = tools.dispatch_tool(
        "edit_file",
        {"path": "/work/scripts/read_excel.py", "old_text": "first", "new_text": "second"},
        task_id="task-a",
    )
    assert edited["replacements"] == 1
    assert (workspace / ".agent_runs/task-task-a/work/scripts/read_excel.py").read_text() == "print('second')\n"


def test_agent_dispatch_passes_current_task_id_for_work_paths(workspace):
    from agent import core

    result = core._dispatch_tool_for_task(
        "write_file",
        {"path": "/work/created_by_agent.py", "content": "print('ok')"},
        "task-a",
    )
    assert result["path"] == "/work/created_by_agent.py"
    assert (workspace / ".agent_runs/task-task-a/work/created_by_agent.py").is_file()


def test_move_and_delete_files_inside_work_are_supported(workspace):
    tools.dispatch_tool(
        "write_file", {"path": "/work/old.txt", "content": "content"}, task_id="task-a"
    )
    moved = tools.dispatch_tool(
        "move_file", {"source": "/work/old.txt", "destination": "/work/new.txt"}, task_id="task-a"
    )
    assert moved["source"] == "/work/old.txt"
    assert moved["destination"] == "/work/new.txt"
    deleted = tools.dispatch_tool("delete_file", {"path": "/work/new.txt"}, task_id="task-a")
    assert deleted["deleted"] == "/work/new.txt"
    assert not (workspace / ".agent_runs/task-task-a/work/new.txt").exists()


def test_move_cannot_cross_workspace_and_work_boundary(workspace):
    (workspace / "persistent.txt").write_text("keep", encoding="utf-8")
    tools.dispatch_tool(
        "write_file", {"path": "/work/temp.txt", "content": "temporary"}, task_id="task-a"
    )
    with pytest.raises(ValueError, match="both be workspace paths"):
        tools.dispatch_tool(
            "move_file", {"source": "/work/temp.txt", "destination": "persistent.txt"},
            task_id="task-a",
        )
    assert (workspace / "persistent.txt").read_text(encoding="utf-8") == "keep"


def test_work_path_is_isolated_by_task_and_requires_task_context(workspace):
    with pytest.raises(ValueError, match="current task context"):
        tools.dispatch_tool("write_file", {"path": "/work/script.py", "content": "print(1)"})

    tools.dispatch_tool(
        "write_file", {"path": "/work/script.py", "content": "print(1)"}, task_id="task-a"
    )
    with pytest.raises(FileNotFoundError):
        tools.dispatch_tool("read_file", {"path": "/work/script.py"}, task_id="task-b")


def test_work_path_rejects_traversal_and_symlinks(workspace, tmp_path):
    with pytest.raises(ValueError):
        tools.dispatch_tool(
            "write_file", {"path": "/work/../escape.txt", "content": "bad"}, task_id="task-a"
        )

    work, _ = tools._task_work_directory(workspace, "task-a")
    outside = tmp_path / "outside"
    outside.mkdir()
    (work / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symbolic links"):
        tools.dispatch_tool(
            "write_file", {"path": "/work/linked/escape.txt", "content": "bad"}, task_id="task-a"
        )
    assert not (outside / "escape.txt").exists()


def test_run_python_executes_script_created_in_work(workspace, tmp_path, monkeypatch):
    project = tmp_path / "project"
    agent_dir = project / "agent"
    venv = project / ".venv"
    (venv / "bin").mkdir(parents=True)
    agent_dir.mkdir()
    (venv / "bin" / "python").write_text("placeholder", encoding="utf-8")
    (venv / "pyvenv.cfg").write_text("home = /usr", encoding="utf-8")
    monkeypatch.setattr(tools, "__file__", str(agent_dir / "tools.py"))
    monkeypatch.setattr(tools.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write(b"SCRIPT_RAN\n")
        kwargs["stderr"].write(b"")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tools.subprocess, "run", fake_run)
    tools.dispatch_tool(
        "write_file",
        {"path": "/work/script_read_excel.py", "content": "print('SCRIPT_RAN')\n"},
        task_id="task-a",
    )
    result = tools.dispatch_tool(
        "run_python", {"path": "/work/script_read_excel.py"}, task_id="task-a"
    )

    assert result["exit_code"] == 0
    assert result["script"] == "/work/script_read_excel.py"
    assert result["stdout"].strip() == "SCRIPT_RAN"
    assert "/work/script_read_excel.py" in calls[0][-1]
    assert "--bind" in calls[0] and str(workspace / ".agent_runs/task-task-a/work") in calls[0]
