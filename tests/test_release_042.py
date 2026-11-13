import json
import base64


from agent import core, tools, attachments


def test_execution_mode_routes_simple_and_complex_tasks():
    assert core.classify_execution_mode("Создай папку test") == "simple"
    assert core.classify_execution_mode("Обработай книгу Excel и создай два отчёта") == "planned"
    assert core.classify_execution_mode("Посмотри это", attachments=[{"kind": "image"}]) == "simple"


def test_all_tools_remain_visible_and_domains_are_grouped_by_scope():
    excel_names = {x["function"]["name"] for x in core.tools_for_task("Сгруппируй данные Excel по товару")}
    assert "terminal" in excel_names
    assert "publish_file" in excel_names
    assert "open_browser_page" in excel_names
    assert not {"excel_aggregate", "read_excel", "write_excel"} & excel_names
    web_names = {x["function"]["name"] for x in core.tools_for_task("Найди в интернете документацию")}
    assert "web_search" in web_names
    assert "terminal" in web_names


def test_ollama_client_sends_image_as_native_images_field(tmp_path, monkeypatch):
    monkeypatch.setattr(attachments, "ATTACHMENT_ROOT", tmp_path)
    monkeypatch.setattr(core, "ATTACHMENT_ROOT", tmp_path)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes")
    client = core.OllamaClient()
    captured = {}
    def fake_post(endpoint, payload, timeout=None):
        captured.update(payload)
        return {"message": {"content": "ok"}}
    monkeypatch.setattr(client, "_post", fake_post)
    client.chat([{"role": "user", "content": "Что на изображении?", "attachments": [{"kind": "image", "name": "image.png", "path": str(image_path)}]}])
    message = captured["messages"][0]
    assert message["images"] == [base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes").decode("ascii")]
    assert "attachments" not in message


def test_document_attachment_text_is_added_to_model_message():
    result = attachments.model_message_content("Суммируй файл", [{"kind": "document", "name": "note.txt", "extracted_text": "Важная строка"}])
    assert "Важная строка" in result
    assert "note.txt" in result


def test_document_attachment_is_saved_in_managed_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(attachments, "ATTACHMENT_ROOT", tmp_path / "data" / "attachments")
    monkeypatch.setattr(attachments, "settings", SimpleNamespace(workspace_path=tmp_path / "workspace", database_path=tmp_path / "data" / "agent.sqlite3"))
    import base64
    item = {"name": "notes.txt", "mime_type": "text/plain", "data_base64": base64.b64encode(b"hello document").decode("ascii")}
    saved = attachments.save_attachments([item])[0]
    assert saved["kind"] == "document"
    assert saved["extracted_text"] == "hello document"
    assert saved["workspace_path"].startswith(".attachments/")
    assert (tmp_path / "workspace" / saved["workspace_path"]).read_bytes() == b"hello document"
    model_content = attachments.model_message_content("Summarize", [saved])
    assert saved["workspace_path"] not in model_content
    assert "hello document" in model_content


def test_plan_acceptance_paths_allow_the_full_container_filesystem():
    for path in ("/work/result.txt", "/tmp/result.txt", "/var/tmp/output.csv", "../relative.txt"):
        core._validate_plan_acceptance_paths([{"acceptance_check": {"type": "file_exists", "path": path}}])


def test_inline_python_acceptance_check_requires_real_success():
    from agent.planning import verify_acceptance_check
    check = {"type": "python_code_exit_code"}
    args = {"code": "print('ok')"}
    assert verify_acceptance_check(check, "run_python", args, {"inline_code": True, "exit_code": 0, "timed_out": False}, ".")
    assert verify_acceptance_check(check, "run_python", args, {"inline_code": True, "exit_code": 1, "timed_out": False}, ".") is None


def test_simple_inspection_and_unknown_imperatives_stay_direct():
    assert core.classify_execution_mode("Проверь, открывается ли файл report.xlsx") == "simple"
    assert core.classify_execution_mode("Найди файл с отчётом") == "simple"
    assert core.classify_execution_mode("Сделай одну простую проверку") == "simple"


def test_repeated_direct_errors_ask_llm_instead_of_forcing_plan(monkeypatch):
    from agent import core

    class FakeClient:
        def __init__(self, mode):
            self.mode = mode
            self.calls = 0
        def chat(self, messages, tools=None, reasoning=True):
            self.calls += 1
            assert tools is None
            return {"message": {"content": json.dumps({"mode": self.mode, "reason": "model choice"})}}

    direct = FakeClient("direct")
    monkeypatch.setattr(core, "client", direct)
    assert core.reconsider_execution_mode_after_errors("Проверь файл", [{"error": "bad call"}]) == "direct"
    assert direct.calls == 1

    plan = FakeClient("plan")
    monkeypatch.setattr(core, "client", plan)
    assert core.reconsider_execution_mode_after_errors("Создай приложение", [{"error": "bad call"}]) == "plan"
    assert plan.calls == 1
