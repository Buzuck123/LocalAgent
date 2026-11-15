import json
import threading
import time
from types import SimpleNamespace

import pytest


class FakeSocket:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, value):
        self.timeouts.append(value)


class FakeResponse:
    status = 200

    def __init__(self, lines):
        self.lines = list(lines)
        self.closed = False

    def __iter__(self):
        for line in self.lines:
            if isinstance(line, Exception):
                raise line
            yield line

    def read(self, size=-1):
        return b""

    def close(self):
        self.closed = True


class FakeConnection:
    instances = []

    def __init__(self, host, port, timeout, **kwargs):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.kwargs = kwargs
        self.sock = FakeSocket()
        self.response = None
        self.request_args = None
        self.closed = False
        self.__class__.instances.append(self)

    def connect(self):
        return None

    def request(self, method, target, body=None, headers=None):
        self.request_args = (method, target, body, headers)

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


def _chunk(message, done=False, **extra):
    return (json.dumps({"message": message, "done": done, **extra}) + "\n").encode()


def test_stream_uses_phase_timeouts_without_total_generation_deadline(monkeypatch):
    from agent import core

    FakeConnection.instances.clear()
    response = FakeResponse([
        _chunk({"content": "hello"}),
        _chunk({"content": ""}, done=True, eval_count=1, eval_duration=1_000_000_000),
    ])
    FakeConnection.getresponse = lambda self: response
    monkeypatch.setattr(core.http.client, "HTTPConnection", FakeConnection)
    events = []
    monkeypatch.setattr(core.db, "record_ollama_request_event",
                        lambda *args, **kwargs: events.append((args, kwargs)))
    monkeypatch.setattr(core, "settings", SimpleNamespace(
        ollama_url="http://127.0.0.1:11434", model="test-model",
        ollama_connect_timeout=10, ollama_first_token_timeout=600,
        ollama_stream_idle_timeout=300, debug_llm_trace=False, debug_llm_thinking=False,
    ))

    client = core.OllamaClient()
    result = client._post_stream("/api/chat", {"model": "test-model"})

    assert result["message"]["content"] == "hello"
    connection = FakeConnection.instances[-1]
    assert connection.timeout == 10
    assert connection.sock.timeouts == [600, 300]
    assert connection.closed
    event_names = [args[3] for args, _ in events]
    assert event_names == [
        "request_started", "connected", "response_headers", "first_chunk", "request_completed"
    ]


@pytest.mark.parametrize(
    ("connected", "first_chunk", "expected_phase", "expected_timeout"),
    [
        (False, False, "connect", 10),
        (True, False, "first_token", 600),
        (True, True, "stream_idle", 300),
    ],
)
def test_stream_timeout_is_classified_by_last_response_phase(
    monkeypatch, connected, first_chunk, expected_phase, expected_timeout
):
    from agent import core

    FakeConnection.instances.clear()

    class TimeoutConnection(FakeConnection):
        def connect(self):
            if not connected:
                raise TimeoutError("connect stalled")

        def getresponse(self):
            if connected and not first_chunk:
                raise TimeoutError("first response stalled")
            if first_chunk:
                return FakeResponse([_chunk({"content": "first"}), TimeoutError("stream stalled")])

    monkeypatch.setattr(core.http.client, "HTTPConnection", TimeoutConnection)
    monkeypatch.setattr(core.db, "record_ollama_request_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "settings", SimpleNamespace(
        ollama_url="http://127.0.0.1:11434", model="test-model",
        ollama_connect_timeout=10, ollama_first_token_timeout=600,
        ollama_stream_idle_timeout=300, debug_llm_trace=False, debug_llm_thinking=False,
    ))

    client = core.OllamaClient()
    with pytest.raises(core.OllamaRequestTimeout) as caught:
        client._post_stream("/api/chat", {"model": "test-model"})
    assert caught.value.phase == expected_phase
    assert caught.value.timeout_seconds == expected_timeout


def test_chat_calls_are_serialized(monkeypatch):
    from agent.core import OllamaClient

    client = OllamaClient()
    active = 0
    maximum = 0
    guard = threading.Lock()

    def fake_chat(*args, **kwargs):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.04)
        with guard:
            active -= 1
        return {"ok": True}

    monkeypatch.setattr(client, "_chat_unlocked", fake_chat)
    threads = [threading.Thread(target=client.chat, args=([{"role": "user", "content": str(i)}],))
               for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert maximum == 1


def test_ollama_request_events_are_persisted_independently_of_debug_trace(tmp_path, monkeypatch):
    from agent import db

    monkeypatch.setattr(db, "settings", SimpleNamespace(database_path=tmp_path / "events.sqlite3"))
    db.init_db()
    db.record_ollama_request_event(
        "request-1", "task-1", "gemma4:12b", "request_timeout",
        phase="first_token", elapsed_ms=600000,
        error_type="OllamaRequestTimeout", error="No response",
        details={"timeout_seconds": 600},
    )
    rows = db.list_ollama_request_events("task-1")
    assert len(rows) == 1
    assert rows[0]["request_id"] == "request-1"
    assert rows[0]["event_type"] == "request_timeout"
    assert rows[0]["phase"] == "first_token"
    assert rows[0]["details"] == {"timeout_seconds": 600}


def test_file_tool_descriptions_document_limits_and_optional_overlap():
    from agent.tools import TOOL_DEFINITIONS

    descriptions = {
        item["function"]["name"]: item["function"]["description"]
        for item in TOOL_DEFINITIONS
    }
    for name in ("ask_model", "load_file_for_model"):
        text = descriptions[name].lower()
        assert "12 mb" in text
        assert "30 seconds" in text
        assert "100 mb" in text
        assert "60 seconds" in text
        assert "50 mb" in text
        assert "overlap" in text
        assert "timestamps" in text


def test_next_chat_checks_ollama_after_timeout_before_starting_generation(monkeypatch):
    from agent.core import OllamaClient

    client = OllamaClient()
    order = []
    client._recovery_state = {
        "request_id": "previous-request", "task_id": "task-1",
        "model": "test-model", "phase": "first_token",
    }
    monkeypatch.setattr(client, "_status_unlocked", lambda: order.append("health_check") or {})
    monkeypatch.setattr(client, "_chat_unlocked", lambda *args, **kwargs: order.append("generation") or {"ok": True})

    result = client.chat([{"role": "user", "content": "continue"}])

    assert result == {"ok": True}
    assert order == ["health_check", "generation"]
    assert client._recovery_state is None


def test_failed_ollama_recovery_check_blocks_the_next_generation(monkeypatch):
    from agent.core import OllamaClient

    client = OllamaClient()
    client._recovery_state = {
        "request_id": "previous-request", "task_id": "task-1",
        "model": "test-model", "phase": "stream_idle",
    }
    generation_calls = []
    monkeypatch.setattr(client, "_status_unlocked", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(client, "_chat_unlocked", lambda *args, **kwargs: generation_calls.append(True))

    with pytest.raises(RuntimeError, match="new generation was not started"):
        client.chat([{"role": "user", "content": "retry"}])
    assert generation_calls == []
    assert client._recovery_state is not None
