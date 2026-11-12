import threading
import time

from agent.core import TaskCancelled, TaskControl, OllamaClient


def test_task_control_waits_for_independent_remote_cancel_before_boundary_returns(monkeypatch):
    monkeypatch.setattr("agent.core.db.update_task", lambda *args, **kwargs: None)
    control = TaskControl()
    assert control.begin_cancel() is True
    assert control.begin_cancel() is False
    finished = threading.Event()

    def boundary():
        try:
            control.boundary("task-1", {})
        except TaskCancelled:
            finished.set()

    worker = threading.Thread(target=boundary)
    worker.start()
    time.sleep(0.05)
    assert not finished.is_set()
    control.finish_cancel_control({"state": "interrupted", "interrupted": True})
    worker.join(1)
    assert finished.is_set()
    assert control.cancellation_confirmed()


def test_unconfirmed_remote_cancel_is_not_reported_as_confirmed():
    control = TaskControl()
    control.begin_cancel()
    control.finish_cancel_control({"state": "unknown", "interrupted": False})
    assert not control.cancellation_confirmed()


def test_ollama_cancel_closes_only_the_registered_task_stream():
    class Response:
        def __init__(self):
            self.closed = False
        def close(self):
            self.closed = True

    client = OllamaClient()
    one, two = Response(), Response()
    with client._active_response_lock:
        client._active_responses["task-1"] = one
        client._active_responses["task-2"] = two
    assert client.cancel_generation("task-1") is True
    assert one.closed is True
    assert two.closed is False


def test_cancellation_confirmation_distinguishes_unknown_from_terminal_state():
    control = TaskControl()
    control.begin_cancel()
    control.finish_cancel_control({"state": "running", "interrupted": False})
    assert not control.cancellation_confirmed()
    control.finish_cancel_control({"state": "completed", "exit_code": 0})
    assert control.cancellation_confirmed()


def test_agent_persists_cancelling_before_remote_interrupt_thread(monkeypatch):
    from agent import core

    state = {"id": "task-cancel", "conversation_id": "conversation-cancel", "status": "running"}
    monkeypatch.setattr(core.db, "get_task", lambda tid: dict(state))
    monkeypatch.setattr(core.db, "update_task", lambda tid, **fields: state.update(fields))
    monkeypatch.setattr(core.db, "record_debug_trace", lambda *args, **kwargs: None)
    monkeypatch.setattr(core.client, "cancel_generation", lambda tid: True)

    from agent.sandbox_manager import sandbox_manager
    monkeypatch.setattr(sandbox_manager, "interrupt_operation", lambda cid: {"state": "interrupted", "interrupted": True})

    class ImmediateThread:
        def __init__(self, target, **kwargs):
            self.target = target
        def start(self):
            assert state["status"] == "cancelling"
            self.target()

    monkeypatch.setattr(core.threading, "Thread", ImmediateThread)
    agent = core.Agent()
    control = TaskControl()
    agent.controls["task-cancel"] = control
    result = agent.cancel("task-cancel")
    assert result["status"] == "cancelling"  # the worker owns the final terminal state
    assert control.cancel_control_done.is_set()
    assert control.cancellation_confirmed()
