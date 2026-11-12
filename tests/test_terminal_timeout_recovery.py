import asyncio
import json
import sys
import types
from types import SimpleNamespace

from agent import core
from agent.orchestration import classify_failure
from agent.sandbox_manager import SandboxManager
from agent.sandbox_runtime import SwerexRemoteRuntime


def test_unbalanced_quote_is_rejected_before_runtime_is_created():
    manager = SandboxManager(runtime_factory=lambda **kwargs: (_ for _ in ()).throw(AssertionError("runtime must not start")))
    result = manager.execute("conversation", "touch docs/README.md && echo 'unfinished", timeout=120)
    assert result["syntax_error"] is True
    assert result["timed_out"] is False
    assert result["exit_code"] != 0
    # Bash localizes parser diagnostics according to the host locale.  The
    # contract here is that a useful syntax diagnostic is returned, not its
    # English wording.
    assert result["stderr"].strip()


def test_swerex_timeout_resets_only_shell_and_probes_new_session(monkeypatch):
    calls = {"run": 0, "close": [], "create": []}

    class BashAction:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class CreateBashSessionRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class CloseSessionRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class UploadRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRemote:
        def __init__(self, **kwargs):
            pass

        async def is_alive(self, timeout=None):
            return SimpleNamespace(is_alive=True)

        async def create_session(self, request):
            calls["create"].append(request.kwargs)
            return SimpleNamespace()

        async def close_session(self, request):
            calls["close"].append(request.kwargs)
            return SimpleNamespace()

        async def run_in_session(self, action):
            calls["run"] += 1
            if calls["run"] == 1:
                raise TimeoutError("SWE-ReX command timed out")
            return SimpleNamespace(output="__LOCALAGENT_SHELL_READY__\n", exit_code=0,
                                   failure_reason="", timed_out=False)

    swerex = types.ModuleType("swerex")
    runtime_mod = types.ModuleType("swerex.runtime")
    local_mod = types.ModuleType("swerex.runtime.local")
    remote_mod = types.ModuleType("swerex.runtime.remote")
    local_mod.BashAction = BashAction
    local_mod.CreateBashSessionRequest = CreateBashSessionRequest
    local_mod.CloseSessionRequest = CloseSessionRequest
    local_mod.UploadRequest = UploadRequest
    remote_mod.RemoteRuntime = FakeRemote
    monkeypatch.setitem(sys.modules, "swerex", swerex)
    monkeypatch.setitem(sys.modules, "swerex.runtime", runtime_mod)
    monkeypatch.setitem(sys.modules, "swerex.runtime.local", local_mod)
    monkeypatch.setitem(sys.modules, "swerex.runtime.remote", remote_mod)

    adapter = SwerexRemoteRuntime(
        host="http://127.0.0.1", port=1234, auth_token="secret",
        loop_runner=lambda awaitable, timeout=10: asyncio.run(awaitable),
    )
    result = adapter.run("sleep 999", session="main", timeout=1)
    assert result["timed_out"] is True
    assert result["shell_recovered"] is True
    assert result["shell_state_reset"] is True
    assert calls["close"] == [{"session": "main"}]
    assert calls["create"] == [{"session": "main"}]
    assert calls["run"] == 2


def test_tool_arguments_are_strictly_normalized_without_guessing():
    parsed, error = core._parse_tool_arguments({"command": "pwd"})
    assert parsed == {"command": "pwd"}
    assert error is None

    parsed, error = core._parse_tool_arguments('{"command": "pwd"}')
    assert parsed == {"command": "pwd"}
    assert error is None

    parsed, error = core._parse_tool_arguments('{"command": "pwd"')
    assert parsed == {}
    assert "not valid JSON" in error

    parsed, error = core._parse_tool_arguments('["pwd"]')
    assert parsed == {}
    assert "must decode to an object" in error


def test_terminal_timeout_classification_requires_shell_recovery_before_retry():
    recovered = classify_failure(
        "terminal", {"command": "sleep 5"},
        json.dumps({"timed_out": True, "shell_recovered": True, "exit_code": 124}),
    )
    assert recovered["category"] == "timeout"
    assert recovered["recovery_action"] == "inspect_state_after_timeout"
    assert recovered["retry_safe_without_state_check"] is False

    unrecovered = classify_failure(
        "terminal", {"command": "sleep 5"},
        json.dumps({"timed_out": True, "shell_recovered": False, "exit_code": 124}),
    )
    assert unrecovered["recovery_action"] == "restore_terminal_before_retry"
