import json

from agent import core, web


def test_ollama_chat_passes_think_flag(monkeypatch):
    captured = {}
    def fake_post(endpoint, payload, timeout=None):
        captured.update(payload)
        return {"message": {"role": "assistant", "content": "ok"}}
    monkeypatch.setattr(core.client, "_post", fake_post)
    core.client.chat([{"role": "user", "content": "hi"}], reasoning=False)
    assert captured["think"] is False
    core.client.chat([{"role": "user", "content": "hi"}], reasoning=True)
    assert captured["think"] is True


def test_message_create_defaults_to_reasoning_on():
    payload = web.MessageCreate(content="test")
    assert payload.reasoning is True


def test_tasks_endpoint_exposes_checkpoint_activity(monkeypatch):
    task = {"id": "t1", "status": "running", "checkpoint_json": json.dumps({"activity": "Читаю файл", "reasoning": False})}
    monkeypatch.setattr(web.db, "list_tasks", lambda conversation_id=None: [task])
    result = web.tasks("c1")
    assert result[0]["activity"] == "Читаю файл"
    assert result[0]["reasoning"] is False
