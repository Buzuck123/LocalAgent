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
    captured = {}

    def fake_post(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"message": {"role": "assistant", "content": "ok"}}

    monkeypatch.setattr(client, "_post", fake_post)
    result = client.chat([{"role": "user", "content": "hello"}], reasoning=False)

    assert result["message"]["content"] == "ok"
    assert captured["path"] == "/api/chat"
    assert captured["payload"]["model"] == "ornith-1.5:9b"
    assert captured["payload"]["think"] is False
    assert "options" not in captured["payload"]
