import json
from dataclasses import replace

from agent import core, db


def test_completion_reviewer_retries_invalid_json_once(monkeypatch):
    responses = [
        {"message": {"content": "Решение: complete"}},
        {"message": {"content": json.dumps({
            "decision": "complete",
            "requires_action": False,
            "evidence_tools": [],
            "reason": "Ответ соответствует информационному запросу.",
        }, ensure_ascii=False)}},
    ]
    calls = []

    class FakeClient:
        def chat(self, messages, tools=None, reasoning=True):
            calls.append(messages)
            return responses.pop(0)

    monkeypatch.setattr(core, "client", FakeClient())
    verdict, error = core._request_autonomous_completion_verdict(
        "Что такое рекурсия?", "Рекурсия — это...", [], reasoning=False
    )

    assert error is None
    assert verdict["decision"] == "complete"
    assert len(calls) == 2
    assert "The previous reviewer response" in calls[1][0]["content"]


def test_completion_reviewer_stops_after_two_invalid_responses(monkeypatch):
    calls = []

    class FakeClient:
        def chat(self, messages, tools=None, reasoning=True):
            calls.append(messages)
            return {"message": {"content": "не JSON"}}

    monkeypatch.setattr(core, "client", FakeClient())
    verdict, error = core._request_autonomous_completion_verdict(
        "Создай отчёт", "Отчёт готов", [], reasoning=True
    )

    assert verdict is None
    assert error and "дважды" in error
    assert len(calls) == 2
