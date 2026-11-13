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

def test_workspace_virtual_path_resolution(tmp_path, monkeypatch):
    workspace = tmp_path.resolve()
    (workspace / "hello.txt").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=workspace))
    target, namespace = tools._virtual_path("hello.txt")
    assert target == workspace / "hello.txt"
    assert namespace == "workspace"

def test_workspace_path_traversal_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=tmp_path.resolve()))
    with pytest.raises(ValueError):
        tools._virtual_path("../outside.txt")


def test_archived_tool_result_retrieval_has_no_hidden_6000_character_cap():
    from agent.context import retrieve_tool_result_fragment

    raw = '{"value":"' + ("x" * 12000) + '"}'
    result = retrieve_tool_result_fragment({"tr_large": raw}, "tr_large", start=100, length=9000)
    assert result["ok"] is True
    assert len(result["content"]) == 9000
    assert result["start"] == 100
    assert result["end"] == 9100
    assert result["has_more"] is True
    assert result["next_start"] == 9100


def test_load_file_for_model_accepts_readable_absolute_container_paths(tmp_path, monkeypatch):
    from agent import attachments, sandbox_manager as sandbox_manager_module

    monkeypatch.setattr(attachments, "ATTACHMENT_ROOT", tmp_path / "attachments")
    def fake_download(conversation_id, source_path, target_path, *, max_bytes=None):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("container text", encoding="utf-8")
        return target_path
    monkeypatch.setattr(
        sandbox_manager_module, "sandbox_manager",
        SimpleNamespace(download_file=fake_download),
    )
    def reject_workspace_path(*args, **kwargs):
        raise ValueError("not a workspace namespace")
    monkeypatch.setattr(tools, "_resolve_workspace_or_work_file", reject_workspace_path)

    result = tools.load_file_for_model("/workspace/example.txt", conversation_id="conversation-1")
    assert result["ok"] is True
    assert result["namespace"] == "container"
    assert result["path"] == "/workspace/example.txt"
    assert result["extracted_text"] == "container text"


def test_host_workspace_mount_alias_reads_shared_work_directory(tmp_path, monkeypatch):
    workspace = tmp_path.resolve()
    (workspace / "work").mkdir()
    (workspace / "work" / "vocals.wav").write_bytes(b"audio")
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=workspace))

    result = tools.list_workspace_files("/host-workspace")
    assert any(entry["name"] == "vocals.wav" for entry in result["entries"])

    resolved, namespace = tools._virtual_path("/host-workspace/vocals.wav")
    assert resolved == workspace / "work" / "vocals.wav"
    assert namespace == "workspace"


def test_host_workspace_mount_alias_is_supported_by_host_file_tools(tmp_path, monkeypatch):
    workspace = tmp_path.resolve()
    (workspace / "work").mkdir()
    (workspace / "work" / "notes.txt").write_text("from container mount", encoding="utf-8")
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=workspace))

    assert tools.read_workspace_file("/host-workspace/notes.txt")["content"] == "from container mount"
    written = tools.write_workspace_file("/host-workspace/result.txt", "written via alias")
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


def test_tool_result_archive_does_not_evict_older_results():
    from agent.core import _store_tool_result

    archive = {}
    ids = []
    for index in range(205):
        result_id, archive = _store_tool_result(archive, {"index": index})
        ids.append(result_id)
    assert len(archive) == 205
    assert ids[0] in archive


def test_load_file_for_model_delivers_sequential_paragraph_chunks(tmp_path, monkeypatch):
    from agent import attachments, sandbox_manager as sandbox_manager_module

    text = ("A" * 20000) + "\n\n" + ("B" * 20000)
    monkeypatch.setattr(attachments, "ATTACHMENT_ROOT", tmp_path / "attachments")
    def fake_download(conversation_id, source_path, target_path, *, max_bytes=None):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(text, encoding="utf-8")
        return target_path
    def reject_workspace_path(*args, **kwargs):
        raise ValueError("not a workspace path")
    monkeypatch.setattr(sandbox_manager_module, "sandbox_manager", SimpleNamespace(download_file=fake_download))
    monkeypatch.setattr(tools, "_resolve_workspace_or_work_file", reject_workspace_path)

    first = tools.load_file_for_model("/etc/large.txt", start=0, length=19000, conversation_id="conversation-2")
    assert first["has_more"] is True
    assert first["end_char"] > 19000
    assert first["total_chars"] > 30000
    assert first["extracted_text"].endswith("\n\n")
    second = tools.load_file_for_model(
        "/etc/large.txt", start=first["next_start"], length=8000,
        conversation_id="conversation-2",
    )
    assert second["start_char"] == first["next_start"]
    assert second["extracted_text"].startswith("B")



def test_host_side_web_tools_are_not_exposed_or_executed():
    names = {
        item["function"]["name"]
        for item in tools.TOOL_DEFINITIONS
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    assert "web_search" not in names
    assert "open_browser_page" not in names

    for retired_name in ("web_search", "open_browser_page"):
        result = dispatch_tool(retired_name, {"query": "must not run"})
        assert result["ok"] is False
        assert "retired" in result["error"].lower()


def test_system_prompt_describes_container_web_capabilities_and_permissions():
    from agent.core import SYSTEM_PROMPT

    assert "`web-search`" in SYSTEM_PROMPT
    assert "Playwright" in SYSTEM_PROMPT
    assert "localagent_http" in SYSTEM_PROMPT
    assert "root INSIDE the container" in SYSTEM_PROMPT
    assert "/host-workspace" in SYSTEM_PROMPT
    assert "do not spend calls discovering or configuring the proxy" in SYSTEM_PROMPT


def test_saved_checkpoint_refreshes_the_operational_system_prompt():
    from agent.core import SYSTEM_PROMPT, _refresh_saved_system_prompt

    messages = [
        {"role": "system", "content": "obsolete host web-search instructions"},
        {"role": "user", "content": "continue"},
    ]
    refreshed = _refresh_saved_system_prompt(messages)

    assert refreshed[0]["content"] == SYSTEM_PROMPT
    assert refreshed[1]["content"] == "continue"
