import json

from agent.llama_cpp import LlamaCppClient
from agent.config import settings


def test_model_profile_uses_configured_server_context():
    client = LlamaCppClient()
    profile = client.model_profile("bonsai")
    assert profile["model"] == "bonsai"
    assert profile["num_ctx"] == settings.llama_cpp_context
    assert profile["source"] == "llama_cpp_config_fallback_unconfirmed"


def test_openai_tool_receipt_is_paired_with_call_id():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {"name": "terminal", "arguments": {"command": "pwd"}},
            }],
        },
        {"role": "tool", "tool_name": "terminal", "content": "working directory"},
    ]
    result = LlamaCppClient._safe_messages(messages, "bonsai")
    assert result[0]["tool_calls"][0]["function"]["arguments"] == '{"command": "pwd"}'
    assert result[1] == {
        "role": "tool", "content": "working directory", "tool_call_id": "call_123"
    }


def test_image_attachment_uses_openai_multimodal_content_parts(tmp_path):
    from PIL import Image
    image_path = tmp_path / "diagram.png"
    Image.new("RGB", (16, 12), "white").save(image_path)
    messages = [{
        "role": "user", "content": "Inspect this image",
        "attachments": [{"kind": "image", "name": "diagram.png", "path": str(image_path)}],
    }]
    result = LlamaCppClient._safe_messages(messages, "gemma4-12b")
    parts = result[0]["content"]
    assert parts[0] == {"type": "text", "text": "Inspect this image"}
    assert parts[1]["type"] == "text"
    assert parts[2]["type"] == "image_url"
    assert parts[2]["image_url"]["url"].startswith("data:image/png;base64,")


def test_trace_redacts_multimodal_payloads():
    result = LlamaCppClient._trace_safe_messages([{
        "role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}},
        ],
    }])
    assert result[0]["content"][0]["text"] == "look"
    assert result[0]["content"][1] == {"type": "image_url", "payload": "[omitted]"}


def test_streaming_openai_tool_call_is_normalized(monkeypatch):
    class FakeResponse:
        def __init__(self, lines):
            self.lines = lines
            self.closed = False

        def __iter__(self):
            return iter(self.lines)

        def close(self):
            self.closed = True

    class FakeOpener:
        def __init__(self, response):
            self.response = response
            self.request = None

        def open(self, request, timeout=None):
            self.request = request
            return self.response

    chunks = [
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_abc", "type": "function",
            "function": {"name": "terminal", "arguments": '{"command": '},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "function": {"arguments": '"pwd"}'},
        }]}}]},
        {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 120, "completion_tokens": 8}},
    ]
    response = FakeResponse([("data: " + json.dumps(item) + "\n\n").encode() for item in chunks]
                            + [b"data: [DONE]\n\n"])
    opener = FakeOpener(response)
    client = LlamaCppClient()
    client.opener = opener
    monkeypatch.setattr(client, "_current_task_id", lambda: None, raising=False)

    result = client.chat([{"role": "user", "content": "show cwd"}], tools=[{
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run terminal",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }], model="bonsai")

    assert result["message"]["tool_calls"][0]["id"] == "call_abc"
    assert result["message"]["tool_calls"][0]["function"]["arguments"] == {"command": "pwd"}
    assert result["eval_count"] == 8
    assert response.closed is True
    assert opener.request.full_url.endswith("/chat/completions")
    sent_payload = json.loads(opener.request.data.decode("utf-8"))
    assert sent_payload["stream_options"] == {"include_usage": True}
    assert client.last_generation_stats["current_prompt_tokens"] == 120



def test_document_visuals_are_marked_for_multimodal_delivery(tmp_path):
    receipt = {
        "ok": True,
        "result": {
            "kind": "text",
            "extracted_text": "Readable document text",
            "note": "Document has rendered visual pages.",
        },
    }
    from PIL import Image
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_load_file",
                "type": "function",
                "function": {"name": "load_file_for_model", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_name": "load_file_for_model",
            "tool_call_id": "call_load_file",
            "content": json.dumps(receipt),
        },
        {
            "role": "user",
            "content": "Media extracted from report.pdf has been supplied alongside the tool result.",
            "attachments": [{"kind": "image", "name": "report-page-1.png", "path": str(image_path)}],
        },
    ]
    result = LlamaCppClient._safe_messages(messages, "bonsai")
    prior = json.loads(result[1]["content"])
    assert prior["result"]["visual_media_delivery"] == "attached_to_llama_cpp_multimodal_request"
    assert "мультимодальный API llama.cpp" in prior["result"]["note"]
    assert result[2]["content"][0]["text"].startswith("Media extracted")
    assert result[2]["content"][-1]["type"] == "image_url"


def test_multiple_system_messages_are_merged_at_the_beginning():
    messages = [
        {"role": "user", "content": "First user turn"},
        {"role": "system", "content": "Primary agent instructions"},
        {"role": "assistant", "content": "Previous response"},
        {"role": "system", "content": "Current plan instructions"},
        {"role": "user", "content": "Continue"},
    ]

    result = LlamaCppClient._safe_messages(messages, "bonsai")

    system_messages = [item for item in result if item["role"] == "system"]
    assert len(system_messages) == 1
    assert result[0] is system_messages[0]
    assert "Primary agent instructions" in result[0]["content"]
    assert "Current plan instructions" in result[0]["content"]
    assert [item["content"] for item in result if item["role"] == "user"] == [
        "First user turn", "Continue"
    ]
    assert [item["role"] for item in result[1:]] == [
        "user", "assistant", "user"
    ]


def test_router_model_list_and_selection_use_server_ids(monkeypatch):
    client = LlamaCppClient()
    monkeypatch.setattr(client, "_json_request", lambda path: {
        "data": [
            {"id": "gemma4-12b", "context_length": 65536},
            {"id": "qwen35-9b", "context_length": 65536},
        ]
    })
    models = client.models()
    assert [item["name"] for item in models["models"]] == ["gemma4-12b", "qwen35-9b"]
    assert models["selected_model"] in {"gemma4-12b", "qwen35-9b"}
    profile = client.model_profile("gemma4-12b")
    assert profile["num_ctx"] == 65536
    assert client.select_model("qwen35-9b")["selected_model"] == "qwen35-9b"


def test_audio_attachment_uses_openai_input_audio_shape(tmp_path):
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"RIFF" + b"\x00" * 40)
    messages = [{
        "role": "user", "content": "Transcribe this",
        "attachments": [{"kind": "audio", "name": "speech.wav", "path": str(audio_path)}],
    }]
    result = LlamaCppClient._safe_messages(messages, "gemma4-12b")
    audio_part = next(part for part in result[0]["content"] if part["type"] == "input_audio")
    assert audio_part["input_audio"]["format"] == "wav"
    assert audio_part["input_audio"]["data"]
