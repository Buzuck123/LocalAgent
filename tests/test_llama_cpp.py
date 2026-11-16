import json

from agent.llama_cpp import LlamaCppClient
from agent.config import settings


def test_model_profile_uses_configured_server_context():
    client = LlamaCppClient()
    profile = client.model_profile("bonsai")
    assert profile["model"] == "bonsai"
    assert profile["num_ctx"] == settings.llama_cpp_context
    assert profile["source"] == "llama_cpp_server_config"


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


def test_text_only_media_is_never_silently_claimed_as_attached():
    messages = [{
        "role": "user",
        "content": "Inspect this image",
        "attachments": [{"kind": "image", "name": "diagram.png"}],
    }]
    result = LlamaCppClient._safe_messages(messages, "bonsai")
    assert "Медиа не переданы модели" in result[0]["content"]
    assert "diagram.png" in result[0]["content"]
    assert "Inspect this image" in result[0]["content"]
    assert "images" not in result[0]


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
            "function": {"name": "terminal", "arguments": "{\\"command\\""},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": ": \\"pwd\\"}"},
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



def test_document_visuals_are_not_claimed_as_delivered():
    receipt = {
        "ok": True,
        "result": {
            "kind": "text",
            "extracted_text": "Readable document text",
            "note": "Document has rendered visual pages.",
        },
    }
    messages = [
        {
            "role": "tool",
            "tool_name": "load_file_for_model",
            "content": json.dumps(receipt),
        },
        {
            "role": "user",
            "content": "Media extracted from report.pdf has been supplied alongside the tool result.",
            "attachments": [{"kind": "image", "name": "report-page-1.png"}],
        },
    ]
    result = LlamaCppClient._safe_messages(messages, "bonsai")
    prior = json.loads(result[0]["content"])
    assert prior["result"]["visual_media_delivery"] == "not_attached_llama_cpp_text_mode"
    assert "только извлечённый текст" in prior["result"]["note"]
    assert "Медиа не переданы модели" in result[1]["content"]
    assert "has been supplied" not in result[1]["content"]
