from pathlib import Path
from types import SimpleNamespace

from agent import core, tools


def test_ollama_chat_does_not_override_model_tag_parameters(monkeypatch):
    expected_settings = SimpleNamespace(
        model="ornith-1.5:9b",
        ollama_url="http://127.0.0.1:11434",
        keep_alive="10m",
        num_ctx=65536,
        num_gpu=99,
        num_predict=32768,
        temperature=0.7,
    )
    monkeypatch.setattr(core, "settings", expected_settings)
    monkeypatch.setattr(core.db, "get_app_setting", lambda key, default=None: default)
    client = core.OllamaClient()
    captured = []

    def fake_post(path, payload, timeout=None):
        captured.append((path, payload))
        if path == "/api/show":
            return {"parameters": ""}
        return {"message": {"role": "assistant", "content": "ok"}}

    monkeypatch.setattr(client, "_post", fake_post)
    monkeypatch.setattr(client, "_post_stream", fake_post)
    result = client.chat([{"role": "user", "content": "hello"}], reasoning=False)

    assert result["message"]["content"] == "ok"
    chat_requests = [payload for path, payload in captured if path == "/api/chat"]
    assert len(chat_requests) == 1
    chat_payload = chat_requests[0]
    assert chat_payload["model"] == "ornith-1.5:9b"
    assert chat_payload["think"] is False
    assert "options" not in chat_payload
