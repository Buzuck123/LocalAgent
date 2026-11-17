import json
from dataclasses import replace

import pytest
from agent import core, db


def _isolated_db(tmp_path, monkeypatch):
    db_settings = replace(db.settings, database_path=tmp_path / "agent.sqlite3")
    monkeypatch.setattr(db, "settings", db_settings)
    db.init_db()
    return db.create_conversation("clarification test")


def test_create_task_atomically_persists_original_user_event(tmp_path, monkeypatch):
    conversation = _isolated_db(tmp_path, monkeypatch)
    task = db.create_task(conversation["id"], "Найди и сравни варианты")
    messages = db.get_messages(conversation["id"])
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == task["user_message"]
    assert messages[0]["_event_id"] == task["initial_message_event_id"]


def test_normal_message_is_saved_and_routes_to_waiting_task(tmp_path, monkeypatch):
    conversation = _isolated_db(tmp_path, monkeypatch)
    task = db.create_task(conversation["id"], "Исходная задача")
    checkpoint = {
        "mode": "unified", "phase": "paused", "pause_reason": "needs_user_input",
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
    saved_answer = saved["messages"][-1]
    persisted_answer = db.get_messages(conversation["id"])[-1]
    assert saved_answer["role"] == "user"
    assert saved_answer["content"] == "Используй фактические данные"
    assert saved_answer["_event_id"]
    assert persisted_answer["content"] == "Используй фактические данные"
    assert persisted_answer["_event_id"] == saved_answer["_event_id"]


@pytest.mark.parametrize("status", ["paused", "interrupted"])
def test_normal_message_routes_to_clarification_task_after_restart(tmp_path, monkeypatch, status):
    conversation = _isolated_db(tmp_path, monkeypatch)
    task = db.create_task(conversation["id"], "Исходная задача")
    checkpoint = {
        "pause_reason": "needs_user_input",
        "clarification": {"question": "Уточните", "answer": None, "resolved": False},
        "messages": [{"role": "user", "content": "Исходная задача"}],
    }
    db.update_task(task["id"], status=status, checkpoint_json=json.dumps(checkpoint))
    # A newer unrelated task must not steal the clarification reply.
    other = db.create_task(conversation["id"], "Другая самостоятельная задача")
    db.update_task(other["id"], status="running", checkpoint_json=json.dumps({
        "messages": [{"role": "user", "content": "Другая самостоятельная задача"}]
    }))

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
    answer = "Продолжай с вариантом Б"
    result = agent.submit(conversation["id"], answer)
    assert result["id"] == task["id"]
    assert result["status"] == "queued"
    assert len(started) == 1
    target, args = started[0]
    assert target.__self__ is agent
    assert target.__func__ is core.Agent._run_task
    assert args[0] == task["id"]
    assert args[0] != other["id"]
    saved_task = db.get_task(task["id"])
    assert saved_task["user_message"] == "Исходная задача"
    saved = json.loads(saved_task["checkpoint_json"])
    assert saved["clarification"]["resolved"] is True
    assert saved["clarification"]["answer"] == answer
    assert saved["phase"] == "loop"
    assert "pause_reason" not in saved
    assert db.get_messages(conversation["id"])[-1]["content"] == answer


def test_resume_cannot_bypass_required_clarification(tmp_path, monkeypatch):
    conversation = _isolated_db(tmp_path, monkeypatch)
    task = db.create_task(conversation["id"], "Исходная задача")
    db.update_task(task["id"], status="paused", checkpoint_json=json.dumps({
        "pause_reason": "needs_user_input", "clarification": {"question": "Уточните", "resolved": False}
    }))
    with pytest.raises(ValueError, match="ожидает ответа"):
        core.Agent().resume(task["id"])
