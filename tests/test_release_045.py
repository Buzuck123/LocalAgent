import base64
import json
from types import SimpleNamespace

import pytest

from agent import core, tools


def _names(definitions):
    return {item["function"]["name"] for item in definitions}


def test_attachment_questions_skip_planner_but_workflows_do_not():
    image = [{"kind": "image", "name": "photo.jpg"}]
    text_doc = [{"kind": "document", "name": "notes.pdf"}]
    workbook = [{"kind": "document", "name": "book.xlsx"}]
    assert core.classify_execution_mode("Расскажи, что на фото", image) == "simple"
    assert core.classify_execution_mode("Опиши изображение", image) == "simple"
    assert core.classify_execution_mode("Что написано в документе?", text_doc) == "simple"
    assert core.classify_execution_mode("Создай описание товара по фото", image) == "simple"
    assert core.classify_execution_mode("Обработай фото и расправь рукава", image) == "planned"
    assert core.classify_execution_mode("Создай Excel по данным фото", image) == "planned"
    assert core.classify_execution_mode("Обработай книгу и создай отчёт", workbook) == "planned"
    assert core.classify_execution_mode("Сохрани результат в новый файл", image) == "planned"


def test_tool_scope_matches_only_the_tools_sent_to_ollama():
    definitions = [
        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "copy_file", "parameters": {"type": "object"}}},
    ]
    messages = core._messages_with_tool_scope(
        [{"role": "system", "content": core.SYSTEM_PROMPT}, {"role": "user", "content": "copy"}],
        definitions,
    )
    scope = messages[1]["content"]
    assert "read_file, copy_file" in scope
    assert "run_python" not in scope


def test_empty_tool_scope_explicitly_disables_tools():
    messages = core._messages_with_tool_scope(
        [{"role": "system", "content": core.SYSTEM_PROMPT}], []
    )
    assert "No tools are provided" in messages[1]["content"]


def test_empty_tool_registry_rejects_every_model_tool_call():
    with pytest.raises(core.ToolContractError, match="Unknown tool"):
        core._run_internal_tool("write_file", {"path": "x.txt", "content": "x"}, {}, tool_definitions=[])


def test_copy_file_from_work_to_workspace_and_back(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=root.resolve()))
    (root / "out").mkdir()
    work, _ = tools._task_work_directory(root, "task-a")
    (work / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    published = tools.dispatch_tool(
        "copy_file", {"source": "/work/data.csv", "destination": "out/data.csv"}, task_id="task-a"
    )
    assert published["ok"] is True
    assert published["verified"] is True
    assert (root / "out/data.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"

    copied_back = tools.dispatch_tool(
        "copy_file", {"source": "out/data.csv", "destination": "/work/copy.csv"}, task_id="task-a"
    )
    assert copied_back["ok"] is True
    assert (work / "copy.csv").read_bytes() == (root / "out/data.csv").read_bytes()


def test_copy_file_protects_existing_destination(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=root.resolve()))
    (root / "source.txt").write_text("new", encoding="utf-8")
    (root / "target.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        tools.dispatch_tool("copy_file", {"source": "source.txt", "destination": "target.txt"})
    assert (root / "target.txt").read_text(encoding="utf-8") == "old"


def test_copy_file_rejects_symlinks_and_path_traversal(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=root.resolve()))
    with pytest.raises(ValueError):
        tools.dispatch_tool("copy_file", {"source": "link.txt", "destination": "copy.txt"})
    with pytest.raises(ValueError):
        tools.dispatch_tool("copy_file", {"source": "../outside.txt", "destination": "copy.txt"})


def test_load_file_for_model_returns_native_image_attachment(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=root.resolve()))
    image_path = root / "photo.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"test-png")
    result = tools.dispatch_tool("load_file_for_model", {"path": "photo.png", "question": "Describe it"})
    assert result["ok"] is True
    assert result["delivery"] == "ollama_native_images"
    assert result["_model_attachment"]["path"] == str(image_path)
    assert result["_model_attachment"]["source"] == "workspace_file"


def test_load_file_for_model_extracts_text_file(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=root.resolve()))
    (root / "notes.md").write_text("Important note\nSecond line", encoding="utf-8")
    result = tools.dispatch_tool("load_file_for_model", {"path": "notes.md"})
    assert result["ok"] is True
    assert result["kind"] == "text"
    assert "Important note" in result["extracted_text"]


def test_load_file_for_model_extracts_xlsx_preview(tmp_path, monkeypatch):
    from openpyxl import Workbook

    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=root.resolve()))
    book = Workbook()
    book.active.append(["Product", "Quantity"])
    book.active.append(["Widget", 12])
    book.save(root / "sample.xlsx")
    result = tools.dispatch_tool("load_file_for_model", {"path": "sample.xlsx"})
    assert result["kind"] == "text"
    assert "Widget" in result["extracted_text"]
    assert "read_excel" in result["note"]


def test_model_tool_message_promotes_image_reference_to_attachment():
    payload = {"ok": True, "result": {
        "name": "photo.png", "_model_attachment": {
            "kind": "image", "name": "photo.png", "path": "/tmp/photo.png", "source": "workspace_file"
        }
    }}
    messages = core._build_tool_messages("load_file_for_model", json.dumps(payload))
    assert messages[0]["role"] == "tool"
    assert "_model_attachment" not in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["attachments"][0]["name"] == "photo.png"


def test_tools_are_registered_and_selected_for_file_copy_and_model_loading():
    definitions = {item["function"]["name"]: item["function"] for item in tools.TOOL_DEFINITIONS}
    assert "terminal" in definitions and "publish_file" in definitions
    assert "load_file_for_model" in definitions
    assert not {"copy_file", "read_file", "write_file"} & set(definitions)
    for prompt in ("Скопируй файл", "Проанализируй файл", "Расскажи, что внутри photo.jpg", "Передай файл в LLM для анализа"):
        names = _names(core.tools_for_task(prompt))
        assert "terminal" in names and "load_file_for_model" in names


def test_finalize_refuses_unresolved_tool_errors():
    state = {"unresolved_errors": [{"tool": "run_python", "error": "Unknown tool"}]}
    assert "не может быть отмечена выполненной" in core._unresolved_tool_errors_problem(state)
    assert core._unresolved_tool_errors_problem({"unresolved_errors": []}) is None


def test_ollama_client_sends_workspace_image_from_model_tool(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    image_path = root / "photo.png"
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"tool-image"
    image_path.write_bytes(image_bytes)
    monkeypatch.setattr(core, "settings", SimpleNamespace(
        ollama_url="http://127.0.0.1:11434", model="qwen3.5:9b", workspace_path=root.resolve(),
        keep_alive="10m", num_ctx=32768, request_timeout=300,
    ))
    client = core.OllamaClient()
    captured = {}

    def fake_post(endpoint, payload, timeout=None):
        captured.update(payload)
        return {"message": {"content": "описание"}}

    monkeypatch.setattr(client, "_post", fake_post)
    tool_messages = core._build_tool_messages("load_file_for_model", json.dumps({
        "ok": True,
        "result": {"name": "photo.png", "_model_attachment": {
            "kind": "image", "name": "photo.png", "path": str(image_path),
            "source": "workspace_file",
        }},
    }))
    client.chat(tool_messages)
    assert captured["messages"][1]["role"] == "user"
    assert captured["messages"][1]["images"] == [base64.b64encode(image_bytes).decode("ascii")]


def test_attached_image_question_uses_direct_model_turn_without_planner_and_keeps_tools_visible(tmp_path, monkeypatch):
    from dataclasses import replace
    from agent import db

    database_path = tmp_path / "agent.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(db, "settings", replace(db.settings, database_path=database_path))
    monkeypatch.setattr(core, "settings", replace(
        core.settings, database_path=database_path, workspace_path=workspace,
        auto_resume_tasks=False,
    ))
    db.init_db()
    conversation = db.create_conversation("direct image question")
    attachment = {
        "kind": "image", "name": "photo.jpg", "mime_type": "image/jpeg",
        "path": str(tmp_path / "managed-photo.jpg"), "size_bytes": 123,
    }
    task = db.create_task(conversation["id"], "Расскажи, что на фото", [attachment])

    def planner_must_not_run(*args, **kwargs):
        raise AssertionError("planner must not be called for a simple attached-image question")

    monkeypatch.setattr(core, "_create_plan_with_retries", planner_must_not_run)
    calls = []

    class FakeClient:
        def chat(self, messages, tools=None, reasoning=True):
            calls.append((messages, tools))
            return {"message": {"role": "assistant", "content": "На фото изображена кофта."}}

    monkeypatch.setattr(core, "client", FakeClient())
    core.Agent()._run_task(task["id"], core.TaskControl())

    saved = db.get_task(task["id"])
    assert saved["status"] == "completed"
    assert len(calls) == 1
    assert calls[0][1] is not None
    assert "terminal" in {item["function"]["name"] for item in calls[0][1]}
    user_message = next(item for item in calls[0][0] if item.get("role") == "user")
    assert user_message["attachments"] == [attachment]
    assert "LOCAL_AGENT_TOOL_SCOPE" in calls[0][0][1]["content"]


def test_load_file_for_model_reads_current_task_work_directory(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=root.resolve()))
    work, _ = tools._task_work_directory(root, "task-model-file")
    (work / "input.txt").write_text("Task-scoped source", encoding="utf-8")
    result = tools.dispatch_tool(
        "load_file_for_model", {"path": "/work/input.txt"}, task_id="task-model-file"
    )
    assert result["namespace"] == "work"
    assert result["extracted_text"] == "Task-scoped source"


def test_release_version_is_synchronized():
    from pathlib import Path
    from agent.version import VERSION

    root = Path(__file__).resolve().parents[1]
    assert VERSION == "0.7.2"
    assert (root / "VERSION").read_text(encoding="utf-8").strip() == VERSION
    assert f"Версия: {VERSION}" in (root / "README.md").read_text(encoding="utf-8")


def test_unknown_tool_call_cannot_mark_direct_attachment_task_completed(tmp_path, monkeypatch):
    from dataclasses import replace
    from agent import db

    database_path = tmp_path / "agent.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(db, "settings", replace(db.settings, database_path=database_path))
    monkeypatch.setattr(core, "settings", replace(
        core.settings, database_path=database_path, workspace_path=workspace,
        auto_resume_tasks=False,
    ))
    db.init_db()
    conversation = db.create_conversation("unknown tool guard")
    task = db.create_task(conversation["id"], "Расскажи, что на фото", [{
        "kind": "image", "name": "photo.jpg", "mime_type": "image/jpeg",
        "path": str(tmp_path / "photo.jpg"), "size_bytes": 20,
    }])

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, reasoning=True):
            self.calls += 1
            if self.calls == 1:
                return {"message": {"role": "assistant", "content": "", "tool_calls": [{
                    "function": {"name": "tool_that_does_not_exist", "arguments": {
                        "path": "fake.txt", "content": "fake",
                    }}
                }]}}
            return {"message": {"role": "assistant", "content": "Выполнено."}}

    monkeypatch.setattr(core, "client", FakeClient())
    core.Agent()._run_task(task["id"], core.TaskControl())
    saved = db.get_task(task["id"])
    checkpoint = json.loads(saved["checkpoint_json"])
    assert saved["status"] == "paused"
    assert "не может быть отмечена выполненной" in saved["result"]
    assert checkpoint["task_state"]["actions_failed"] == 1
    assert checkpoint["task_state"]["unresolved_errors"]
