import io
import json
import zipfile
from pathlib import Path

from PIL import Image

from agent import attachments, core


def test_docx_embedded_images_are_extracted_as_model_attachments(tmp_path, monkeypatch):
    monkeypatch.setattr(attachments, "ATTACHMENT_ROOT", tmp_path / "attachments")
    image_bytes = io.BytesIO()
    Image.new("RGB", (640, 480), "white").save(image_bytes, format="PNG")
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("word/media/image1.png", image_bytes.getvalue())
    visuals, warning = attachments._extract_document_visuals("report.docx", archive_bytes.getvalue())
    assert warning is None
    assert len(visuals) == 1
    assert visuals[0]["kind"] == "image"
    assert Path(visuals[0]["path"]).is_file()


def test_tool_receipt_delivers_multiple_document_visuals():
    payload = {
        "result": {
            "ok": True,
            "kind": "text",
            "name": "report.pdf",
            "extracted_text": "Extracted report text",
            "_model_attachments": [
                {"kind": "image", "name": "page 1", "path": "/tmp/page1.jpg", "source": "managed_attachment"},
                {"kind": "image", "name": "page 2", "path": "/tmp/page2.jpg", "source": "managed_attachment"},
            ],
        }
    }
    messages = core._build_tool_messages("load_file_for_model", json.dumps(payload))
    assert len(messages) == 2
    assert "Extracted report text" in messages[0]["content"]
    assert len(messages[1]["attachments"]) == 2


def test_ask_model_includes_text_and_document_visuals(monkeypatch):
    captured = {}

    def fake_load(path, *, conversation_id):
        return {
            "ok": True,
            "kind": "text",
            "name": "report.pdf",
            "extracted_text": "PDF body",
            "size_bytes": 100,
            "_model_attachments": [
                {"kind": "image", "name": "page 1", "path": "/tmp/page1.jpg", "source": "managed_attachment"},
            ],
        }

    def fake_chat(messages, tools=None, reasoning=True, model=None):
        captured["messages"] = messages
        return {"message": {"content": "reviewed"}}

    monkeypatch.setattr(core, "_client_chat", fake_chat)
    monkeypatch.setattr(attachments, "_extract_document_visuals", lambda *args, **kwargs: ([], None))
    from agent import tools
    monkeypatch.setattr(tools, "load_file_for_model", fake_load)

    result = core._ask_model_tool(
        {"prompt": "Review this PDF", "file": "/host-workspace/report.pdf"},
        conversation_id="conversation-visuals",
        context_messages=[],
        main_model="qwen3.5:9b",
        context_limit=65536,
    )
    assert result["ok"] is True
    assert "PDF body" in captured["messages"][1]["content"]
    assert len(captured["messages"][1]["attachments"]) == 1


def test_web_upload_sends_prompt_once_and_directs_agent_to_loader(monkeypatch):
    from agent import web

    monkeypatch.setattr(web.db, "get_conversation", lambda cid: {"id": cid})
    monkeypatch.setattr(
        web, "save_uploads_to_host_workspace",
        lambda items: [{
            "name": "brief.pdf",
            "size_bytes": 1234,
            "container_path": "/host-workspace/abc_brief.pdf",
        }],
    )
    captured = {}

    def fake_submit(cid, message, reasoning=True, attachments=None):
        captured.update({"cid": cid, "message": message, "reasoning": reasoning, "attachments": attachments})
        return {"id": "task-1"}

    monkeypatch.setattr(web.agent, "submit", fake_submit)
    payload = web.MessageCreate(
        content="Summarize the brief",
        reasoning=False,
        attachments=[web.AttachmentInput(name="brief.pdf", mime_type="application/pdf", data_base64="eA==")],
    )
    result = web.send_message("conversation-web", payload)
    assert result["id"] == "task-1"
    assert captured["message"].count("Summarize the brief") == 1
    assert "/host-workspace/abc_brief.pdf" in captured["message"]
    assert "load_file_for_model" in captured["message"]
    assert captured["attachments"] is None


def test_pdf_pages_with_vector_graphics_are_rendered(tmp_path, monkeypatch):
    import fitz

    monkeypatch.setattr(attachments, "ATTACHMENT_ROOT", tmp_path / "attachments")
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.draw_rect(fitz.Rect(50, 50, 300, 300), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
    raw = document.tobytes()
    document.close()

    visuals, warning = attachments._extract_document_visuals("diagram.pdf", raw)
    assert warning is None
    assert len(visuals) == 1
    assert visuals[0]["name"].endswith("page 1")
    assert Path(visuals[0]["path"]).is_file()
