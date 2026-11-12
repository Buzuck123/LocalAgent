from types import SimpleNamespace

from agent import core, tools


def _names(definitions):
    return {item["function"]["name"] for item in definitions}


def test_bounded_python_script_task_skips_planner():
    goal = (
        "Создай Python-скрипт в /work, который создаст Excel-файл с двумя листами, "
        "заполнит данными, сохранит общую таблицу в CSV и проверит результат. "
        "Если что-то не получится, попробуй исправить."
    )
    assert core._is_bounded_python_script_workflow(goal) is True
    assert core.classify_execution_mode(goal) == "simple"


def test_python_script_task_keeps_full_registry_available():
    goal = (
        "Создай Python-скрипт в /work, который создаст Excel-файл и CSV, "
        "запусти и проверь результат."
    )
    names = _names(core.tools_for_task(goal))
    assert {"terminal", "publish_file", "web_search"} <= names
    assert not {"write_file", "run_python", "read_file", "list_files", "read_output_file",
                "read_excel", "write_excel", "excel_aggregate", "sqlite_query"} & names


def test_complex_python_project_still_uses_planner():
    goal = "Разработай полноценное Python-приложение с веб-интерфейсом и базой данных"
    assert core._is_bounded_python_script_workflow(goal) is False
    assert core.classify_execution_mode(goal) == "planned"


def test_agent_dispatch_write_then_run_uses_same_task_workdir(tmp_path, monkeypatch):
    monkeypatch.setattr(core.db, "record_debug_trace", lambda *args, **kwargs: None)
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=root.resolve()))
    write = core._dispatch_tool_for_task(
        "write_file", {"path": "/work/check.py", "content": "print('OK')\n"}, "task-regression"
    )
    assert write["path"] == "/work/check.py"
    assert write["work_dir"].endswith("task-task-regression/work")
    resolved = tools._resolve_task_work_path("/work/check.py", "task-regression")[0]
    assert resolved.is_file() and resolved.read_text() == "print('OK')\n"


def test_missing_work_script_error_identifies_task_directory(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=root.resolve()))
    try:
        tools.run_python_sandbox(path="/work/missing.py", task_id="task-diag")
    except RuntimeError as exc:
        # Sandbox availability may be checked before path existence on some hosts.
        assert "Sandbox unavailable" in str(exc)
    except FileNotFoundError as exc:
        assert "task_id='task-diag'" in str(exc)
        assert "resolved_work_dir=" in str(exc)
    else:
        raise AssertionError("missing script unexpectedly executed")


def test_agent_dispatch_write_then_run_executes_same_work_file(tmp_path, monkeypatch):
    monkeypatch.setattr(core.db, "record_debug_trace", lambda *args, **kwargs: None)
    root = tmp_path / "workspace"
    root.mkdir()
    project = tmp_path / "project"
    agent_dir = project / "agent"
    venv = project / ".venv"
    (venv / "bin").mkdir(parents=True)
    agent_dir.mkdir()
    (venv / "bin" / "python").write_text("placeholder", encoding="utf-8")
    (venv / "pyvenv.cfg").write_text("home = /usr", encoding="utf-8")
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=root.resolve()))
    monkeypatch.setattr(tools, "__file__", str(agent_dir / "tools.py"))
    monkeypatch.setattr(tools.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        kwargs["stdout"].write(b"SCRIPT_RAN\n")
        kwargs["stderr"].write(b"")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tools.subprocess, "run", fake_run)
    written = core._dispatch_tool_for_task(
        "write_file", {"path": "/work/check.py", "content": "print('SCRIPT_RAN')\n"}, "task-e2e"
    )
    executed = core._dispatch_tool_for_task(
        "run_python", {"path": "/work/check.py", "timeout_seconds": 30}, "task-e2e"
    )
    assert written["work_dir"] == executed["work_dir"]
    assert executed["exit_code"] == 0
    assert executed["stdout"].strip() == "SCRIPT_RAN"
    assert executed["script"] == "/work/check.py"
