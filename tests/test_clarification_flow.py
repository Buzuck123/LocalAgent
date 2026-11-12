import json
from dataclasses import replace

import pytest
from agent import core, db


def _isolated_db(tmp_path, monkeypatch):
    db_settings = replace(db.settings, database_path=tmp_path / "agent.sqlite3")
    monkeypatch.setattr(db, "settings", db_settings)
    db.init_db()
    return db.create_conversation("clarification test")


def test_preflight_invalid_response_defaults_to_proceed(monkeypatch):
    class FakeClient:
        def chat(self, messages, tools=None, reasoning=True):
            return {"message": {"content": "not json"}}

    monkeypatch.setattr(core, "client", FakeClient())
    result = core._request_autonomous_preflight("Задача")
    assert result["decision"] == "proceed"


def test_preflight_detects_clear_conflict(monkeypatch):
    class FakeClient:
        def chat(self, messages, tools=None, reasoning=True):
            return {"message": {"content": json.dumps({
                "decision": "clarify", "reason": "Числа несовместимы", "question": "Какое значение верное?"
            })}}

    monkeypatch.setattr(core, "client", FakeClient())
    result = core._request_autonomous_preflight("Сумма должна быть одновременно 10 и 20")
    assert result["decision"] == "clarify"
    assert result["question"] == "Какое значение верное?"


def test_normal_message_is_saved_and_routes_to_waiting_task(tmp_path, monkeypatch):
    conversation = _isolated_db(tmp_path, monkeypatch)
    task = db.create_task(conversation["id"], "Исходная задача")
    checkpoint = {
        "mode": "autonomous", "phase": "paused", "pause_reason": "needs_user_input",
        "preflight_checked": True, "preflight_pending": False,
        "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "Исходная задача"}],
        "clarification": {"question": "Какой вариант выбрать?", "answer": None, "resolved": False},
        "plan": {
            "status": "paused", "current_step_id": "step_1",
            "steps": [{"id": "step_1", "description": "Выбрать вариант", "status": "question"}],
        },
    }
    db.update_task(task["id"], status="paused", checkpoint_json=json.dumps(checkpoint))

    started = []

    class FakeThread:
        def __init__(self, target, args, daemon):
            self.target, self.args = target, args
        def start(self):
            started.append((self.target, self.args))
        def is_alive(self):
            return False

    monkeypatch.setattr(core.threading, "Thread", FakeThread)
    agent = core.Agent()
    result = agent.submit(conversation["id"], "Используй фактические данные")

    assert result["id"] == task["id"]
    assert len(started) == 1
    assert db.get_task(task["id"])["user_message"] == "Исходная задача"
    saved = json.loads(db.get_task(task["id"])["checkpoint_json"])
    assert saved["clarification"]["resolved"] is True
    assert saved["clarification"]["answer"] == "Используй фактические данные"
    assert saved["plan"]["status"] == "in_progress"
    assert saved["plan"]["steps"][0]["status"] == "in_progress"
    assert saved["plan"]["current_step_id"] == "step_1"
    assert saved["messages"][-1] == {"role": "user", "content": "Используй фактические данные"}
    assert db.get_messages(conversation["id"])[-1]["content"] == "Используй фактические данные"


def test_resume_cannot_bypass_required_clarification(tmp_path, monkeypatch):
    conversation = _isolated_db(tmp_path, monkeypatch)
    task = db.create_task(conversation["id"], "Исходная задача")
    db.update_task(task["id"], status="paused", checkpoint_json=json.dumps({
        "pause_reason": "needs_user_input", "clarification": {"question": "Уточните", "resolved": False}
    }))
    with pytest.raises(ValueError, match="ожидает ответа"):
        core.Agent().resume(task["id"])
