import asyncio
import sys
import types
from types import SimpleNamespace

from agent.sandbox_runtime import SwerexRemoteRuntime


def test_swerex_background_api_adapter_preserves_idempotency_timeout_and_cursor(monkeypatch):
    calls = {}

    class Request:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class BashAction(Request):
        pass

    class FakeRemote:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
        async def start_operation(self, request):
            calls["start"] = request
            return {"operation_id": "remote-1", "session": "main", "state": "running", "created_at": 1.0}
        async def get_operation(self, request):
            calls["poll"] = request
            return {"operation_id": "remote-1", "session": "main", "state": "completed",
                    "output": "done", "next_cursor": 9, "exit_code": 0}
        async def interrupt_operation(self, request):
            calls["interrupt"] = request
            return {"operation_id": "remote-1", "state": "interrupted"}
        async def send_input(self, request):
            calls["input"] = request
            return {"operation_id": "remote-1", "state": "running"}
        async def resize_terminal(self, request):
            calls["resize"] = request
            return None

    swerex = types.ModuleType("swerex")
    runtime_mod = types.ModuleType("swerex.runtime")
    remote_mod = types.ModuleType("swerex.runtime.remote")
    abstract_mod = types.ModuleType("swerex.runtime.abstract")
    remote_mod.RemoteRuntime = FakeRemote
    for name, cls in {
        "BashAction": BashAction,
        "StartOperationRequest": Request,
        "GetOperationRequest": Request,
        "InterruptOperationRequest": Request,
        "SendInputRequest": Request,
        "ResizeTerminalRequest": Request,
    }.items():
        setattr(abstract_mod, name, cls)
    monkeypatch.setitem(sys.modules, "swerex", swerex)
    monkeypatch.setitem(sys.modules, "swerex.runtime", runtime_mod)
    monkeypatch.setitem(sys.modules, "swerex.runtime.remote", remote_mod)
    monkeypatch.setitem(sys.modules, "swerex.runtime.abstract", abstract_mod)

    adapter = SwerexRemoteRuntime(host="http://127.0.0.1", port=1234, auth_token="secret",
                                  loop_runner=lambda awaitable, timeout=10: asyncio.run(awaitable))
    started = adapter.start_operation("sleep 60", timeout=None, idempotency_key="journal-uuid")
    assert started["operation_id"] == "remote-1"
    assert calls["start"].kwargs["idempotency_key"] == "journal-uuid"
    assert calls["start"].kwargs["action"].kwargs["timeout"] is None

    polled = adapter.get_operation("remote-1", cursor=7, max_output_chars=100, wait_timeout=2.5)
    assert polled["next_cursor"] == 9
    assert calls["poll"].kwargs["cursor"] == 7
    assert calls["poll"].kwargs["wait_timeout"] == 2.5

    assert adapter.interrupt_operation("remote-1")["state"] == "interrupted"
    assert adapter.send_input("remote-1", "y", sendline=True)["state"] == "running"
    adapter.resize_terminal(rows=40, cols=120)
    assert calls["resize"].kwargs == {"session": "main", "rows": 40, "cols": 120}
