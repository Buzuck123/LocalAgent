from agent import core


def test_client_without_reasoning_parameter_remains_supported(monkeypatch):
    calls = []

    class LegacyClient:
        def chat(self, messages, tools=None):
            calls.append((messages, tools))
            return {"message": {"content": "ok"}}

    monkeypatch.setattr(core, "client", LegacyClient())
    result = core._client_chat([{"role": "user", "content": "test"}], None, reasoning=False)

    assert result["message"]["content"] == "ok"
    assert len(calls) == 1
    assert calls[0][0][0]["content"] == "test"


def test_client_with_reasoning_parameter_receives_selected_mode(monkeypatch):
    calls = []

    class ReasoningClient:
        def chat(self, messages, tools=None, reasoning=True):
            calls.append(reasoning)
            return {"message": {"content": "ok"}}

    monkeypatch.setattr(core, "client", ReasoningClient())
    core._client_chat([{"role": "user", "content": "test"}], None, reasoning=False)

    assert calls == [False]
