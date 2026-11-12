import pytest
from types import SimpleNamespace
from agent import tools
from agent.tools import safe_calculate, dispatch_tool

def test_calculate():
    assert safe_calculate("(2 + 3) * 4")["result"] == 20
    assert safe_calculate("2 ** 3")["result"] == 8

def test_calculator_rejects_code():
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('id')")

def test_workspace_read_and_list(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=tmp_path.resolve()))
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    assert dispatch_tool("read_file", {"path": "hello.txt"})["content"] == "hello"
    assert dispatch_tool("list_files", {})["entries"][0]["name"] == "hello.txt"

def test_workspace_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=tmp_path.resolve()))
    with pytest.raises(ValueError):
        dispatch_tool("read_file", {"path": "../outside.txt"})


def test_read_output_file_reads_only_bounded_run_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=tmp_path.resolve()))
    run_dir = tmp_path / ".agent_runs" / "run-demo123"
    run_dir.mkdir(parents=True)
    (run_dir / "stdout.log").write_text("0123456789" * 1000, encoding="utf-8")
    result = dispatch_tool("read_output_file", {
        "path": ".agent_runs/run-demo123/stdout.log", "start": 10, "length": 9000
    })
    assert result["start"] == 10
    assert len(result["content"]) == 6000
    assert result["has_more"] is True
    assert result["total_bytes"] == 10000


def test_read_output_file_rejects_other_workspace_files(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=tmp_path.resolve()))
    (tmp_path / "secret.txt").write_text("no", encoding="utf-8")
    with pytest.raises(ValueError):
        dispatch_tool("read_output_file", {"path": "secret.txt"})



def test_host_workspace_mount_alias_reads_shared_work_directory(tmp_path, monkeypatch):
    workspace = tmp_path.resolve()
    (workspace / "work").mkdir()
    (workspace / "work" / "vocals.wav").write_bytes(b"audio")
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=workspace))

    result = dispatch_tool("list_files", {"path": "/host-workspace"})
    assert any(entry["name"] == "vocals.wav" for entry in result["entries"])

    resolved, namespace = tools._virtual_path("/host-workspace/vocals.wav")
    assert resolved == workspace / "work" / "vocals.wav"
    assert namespace == "workspace"


def test_host_workspace_mount_alias_is_supported_by_host_file_tools(tmp_path, monkeypatch):
    workspace = tmp_path.resolve()
    (workspace / "work").mkdir()
    (workspace / "work" / "notes.txt").write_text("from container mount", encoding="utf-8")
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=workspace))

    assert dispatch_tool("read_file", {"path": "/host-workspace/notes.txt"})["content"] == "from container mount"
    written = dispatch_tool("write_file", {
        "path": "/host-workspace/result.txt", "content": "written via alias"
    })
    assert written
    assert (workspace / "work" / "result.txt").read_text(encoding="utf-8") == "written via alias"


def test_host_workspace_mount_alias_rejects_traversal_and_symlinks(tmp_path, monkeypatch):
    workspace = tmp_path.resolve()
    mounted = workspace / "work"
    mounted.mkdir()
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (mounted / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=workspace))

    with pytest.raises(ValueError):
        tools._virtual_path("/host-workspace/../outside/secret.txt")
    with pytest.raises(ValueError):
        tools._virtual_path("/host-workspace/escape/secret.txt")
