import json

import pytest

from agent import core, web


def test_provider_switch_is_blocked_by_queued_or_running_tasks(monkeypatch):
    monkeypatch.setattr(
        web.db,
        "list_all_tasks",
        lambda: [{"id": "live", "status": "running", "checkpoint_json": "{}"}],
    )
    with pytest.raises(web.HTTPException) as exc:
        web.select_llm_provider(web.ProviderSelect(provider="llama_cpp"))
    assert exc.value.status_code == 409


def test_provider_switch_pins_legacy_paused_tasks_before_switch(monkeypatch):
    tasks = [
        {"id": "paused-qwen", "status": "paused", "checkpoint_json": json.dumps({"model": "qwen3.5:9b"})},
        {"id": "interrupted-old", "status": "interrupted", "checkpoint_json": "{}"},
        {"id": "done", "status": "completed", "checkpoint_json": "{}"},
    ]
    updates = []
    switched = []
    monkeypatch.setattr(web.db, "list_all_tasks", lambda: tasks)
    monkeypatch.setattr(
        web.db,
        "update_task",
        lambda task_id, **kwargs: updates.append((task_id, kwargs)),
    )
    monkeypatch.setattr(core, "active_llm_provider", "ollama")
    monkeypatch.setattr(
        core,
        "switch_llm_provider",
        lambda provider: switched.append(provider) or {"provider": provider},
    )
    monkeypatch.setattr(web, "get_llm_provider", lambda: {"status": {"available": True}})

    result = web.select_llm_provider(web.ProviderSelect(provider="llama_cpp"))

    assert result["provider"] == "llama_cpp"
    assert switched == ["llama_cpp"]
    pinned = {task_id: json.loads(kwargs["checkpoint_json"]) for task_id, kwargs in updates}
    assert pinned["paused-qwen"]["provider"] == "ollama"
    assert pinned["paused-qwen"]["model"] == "qwen3.5:9b"
    assert pinned["interrupted-old"]["provider"] == "ollama"
    assert "done" not in pinned


def test_task_client_context_overrides_active_provider_client(monkeypatch):
    original = object()
    pinned = object()
    monkeypatch.setattr(core, "client", original)
    token = core._CURRENT_TASK_CLIENT.set(pinned)
    try:
        assert core._task_client() is pinned
    finally:
        core._CURRENT_TASK_CLIENT.reset(token)
    assert core._task_client() is original
