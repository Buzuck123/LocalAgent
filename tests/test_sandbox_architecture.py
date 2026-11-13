import asyncio
import json
import sys
import os
import subprocess
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import db
from agent.config import settings
from agent.sandbox_manager import SandboxManager, SandboxError
from agent import sandbox_manager as sandbox_manager_module
from agent.sandbox_runtime import SwerexRemoteRuntime
from agent.tools import TOOL_DEFINITIONS


class FakeRuntime:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.alive = True
        self.sessions = []
        self.commands = []
        self.closed = []

    def is_alive(self):
        return self.alive

    def ensure_session(self, session="main"):
        self.sessions.append(session)

    def run(self, command, *, session="main", timeout=120):
        self.commands.append((command, session, timeout))
        return {"stdout": "ok\n", "stderr": "", "exit_code": 0, "timed_out": False}

    def close_session(self, session="main"):
        self.closed.append(session)

    def detach(self):
        pass


class FakePodman:
    def __init__(self, *, existing=None, running=True):
        self.existing = existing
        self.running = running
        self.calls = []
        self.started = False

    def __call__(self, args, *, env=None, check=True):
        self.calls.append((list(args), env, check))
        cmd = args[1:]
        if cmd[:2] == ["info", "--format"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"host": {"security": {"rootless": True}}}), stderr="")
        if cmd and cmd[0] == "ps":
            container_id = self.existing
            if container_id and "--no-trunc" not in cmd:
                container_id = container_id[:12]
            return SimpleNamespace(returncode=0, stdout=(container_id + "\n") if container_id else "", stderr="")
        if cmd and cmd[0] == "run":
            self.existing = "container-new"
            return SimpleNamespace(returncode=0, stdout="container-new\n", stderr="")
        if cmd and cmd[0] == "inspect":
            running = self.running or self.started
            payload = [{"Id": cmd[1], "State": {"Running": running},
                        "NetworkSettings": {"Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "32777"}]}}}]
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
        if cmd and cmd[0] == "start":
            self.started = True
            return SimpleNamespace(returncode=0, stdout=cmd[1], stderr="")
        if cmd and cmd[0] == "rm":
            self.existing = None
            return SimpleNamespace(returncode=0, stdout="container removed", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def _fake_db(monkeypatch, record=None):
    state = {"record": record, "status": None, "deleted": False}
    monkeypatch.setattr(db, "get_sandbox", lambda cid: state["record"])
    monkeypatch.setattr(db, "upsert_sandbox", lambda cid, container_id, name, port, token, session, status: state.update(record={
        "conversation_id": cid, "container_id": container_id, "container_name": name,
        "host_port": port, "auth_token": token, "session_name": session, "status": status,
    }))
    monkeypatch.setattr(db, "update_sandbox_status", lambda cid, status: state.update(status=status))
    monkeypatch.setattr(db, "delete_sandbox", lambda cid: state.update(record=None, deleted=True) or True)
    monkeypatch.setattr(db, "list_sandboxes", lambda: [state["record"]] if state["record"] else [])
    return state


def test_tool_catalog_uses_terminal_and_keeps_only_non_duplicate_capabilities():
    names = {item["function"]["name"] for item in TOOL_DEFINITIONS}
    assert names == {"get_tool_result", "terminal", "publish_file",
                     "load_file_for_model"}
    assert not names.intersection({"read_file", "write_file", "run_python", "read_excel",
                                   "write_excel", "sqlite_query", "copy_file"})


def test_new_sandbox_is_created_once_with_rootless_limits_and_dynamic_loopback_port(monkeypatch, tmp_path):
    state = _fake_db(monkeypatch)
    host_workspace = tmp_path / "workspace" / "work"
    test_settings = SimpleNamespace(
        **{**settings.__dict__, "workspace_path": tmp_path / "workspace"}
    )
    monkeypatch.setattr(sandbox_manager_module, "settings", test_settings)
    podman = FakePodman()
    runtimes = []

    def factory(**kwargs):
        runtime = FakeRuntime(**kwargs)
        runtimes.append(runtime)
        return runtime

    manager = SandboxManager(runtime_factory=factory, command_runner=podman)
    runtime = manager.get_runtime("conversation-1")
    assert runtime is runtimes[0]
    assert state["record"]["container_id"] == "container-new"
    assert state["record"]["host_port"] == 32777
    assert state["record"]["session_name"] == "main"
    run_call = next(args for args, _env, _check in podman.calls if args[1:2] == ["run"])
    assert "--cpus" in run_call and "2.0" in run_call
    assert "--memory" in run_call and "4g" in run_call
    assert "--network" in run_call and "localagent-sandbox-internal-test" in run_call
    assert "127.0.0.1::8000" in run_call
    assert "-v" in run_call
    assert f"{host_workspace.resolve()}:/host-workspace:rw" in run_call
    assert host_workspace.is_dir()
    assert not any("podman" in str(x).lower() for x in runtime.commands)
    assert runtime.sessions == ["main"]

    assert manager.get_runtime("conversation-1") is runtime
    assert len([1 for args, _, _ in podman.calls if args[1:2] == ["run"]]) == 1


def test_stopped_container_is_started_and_session_is_recreated(monkeypatch):
    record = {"conversation_id": "conversation-2", "container_id": "container-2",
              "container_name": "old-name", "host_port": 12345, "auth_token": "secret",
              "session_name": "main", "status": "stopped"}
    _fake_db(monkeypatch, record)
    podman = FakePodman(existing="container-2", running=False)
    runtimes = []
    manager = SandboxManager(runtime_factory=lambda **kw: runtimes.append(FakeRuntime(**kw)) or runtimes[-1],
                             command_runner=podman)
    manager.get_runtime("conversation-2")
    assert any(args[1:2] == ["start"] for args, _, _ in podman.calls)
    assert runtimes[0].sessions == ["main"]


def test_missing_container_is_not_silently_recreated(monkeypatch):
    record = {"conversation_id": "conversation-3", "container_id": "missing",
              "container_name": "missing", "host_port": 12345, "auth_token": "secret",
              "session_name": "main", "status": "running"}
    _fake_db(monkeypatch, record)
    podman = FakePodman(existing=None)
    manager = SandboxManager(runtime_factory=lambda **kw: FakeRuntime(**kw), command_runner=podman)
    with pytest.raises(SandboxError, match="отсутствует"):
        manager.get_runtime("conversation-3")
    assert not any(args[1:2] == ["run"] for args, _, _ in podman.calls)


def test_delete_closes_only_main_session_and_removes_one_container(monkeypatch):
    record = {"conversation_id": "conversation-4", "container_id": "container-4",
              "container_name": "old-name", "host_port": 12345, "auth_token": "secret",
              "session_name": "main", "status": "running"}
    state = _fake_db(monkeypatch, record)
    podman = FakePodman(existing="container-4")
    runtime = FakeRuntime()
    manager = SandboxManager(runtime_factory=lambda **kw: runtime, command_runner=podman)
    manager._runtimes["conversation-4"] = runtime
    manager.delete("conversation-4")
    assert runtime.closed == ["main"]
    assert state["deleted"] is True
    assert any(args[1:2] == ["rm"] and args[-1] == "container-4" for args, _, _ in podman.calls)



def test_delete_uses_full_podman_container_id(monkeypatch):
    full_id = "39874f507170abcdef0123456789abcdef0123456789abcdef0123456789abcd"
    record = {"conversation_id": "conversation-long-id", "container_id": full_id,
              "container_name": "sandbox", "host_port": 12345, "auth_token": "secret",
              "session_name": "main", "status": "running"}
    _fake_db(monkeypatch, record)
    podman = FakePodman(existing=full_id)
    manager = SandboxManager(runtime_factory=lambda **kw: FakeRuntime(**kw), command_runner=podman)

    manager.delete("conversation-long-id")

    ps_call = next(args for args, _, _ in podman.calls if args[1:2] == ["ps"])
    assert "--no-trunc" in ps_call
    assert any(args[1:2] == ["rm"] and args[-1] == full_id for args, _, _ in podman.calls)



def test_swerex_adapter_uses_main_session_and_preserves_nonzero_exit(monkeypatch):
    calls = {}

    class BashAction:
        def __init__(self, **kwargs):
            calls["action"] = kwargs

    class CreateBashSessionRequest:
        def __init__(self, **kwargs): calls["create"] = kwargs

    class CloseSessionRequest:
        def __init__(self, **kwargs): calls["close"] = kwargs

    class UploadRequest:
        def __init__(self, **kwargs): calls["upload"] = kwargs

    class FakeRemote:
        def __init__(self, **kwargs): calls["remote_init"] = kwargs
        async def is_alive(self, timeout=None): return SimpleNamespace(is_alive=True)
        async def create_session(self, request): return SimpleNamespace()
        async def run_in_session(self, action): return SimpleNamespace(output="bad", exit_code=7, failure_reason="", expect_string="prompt")
        async def close_session(self, request): return SimpleNamespace()
        async def upload(self, request): return SimpleNamespace()

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

    adapter = SwerexRemoteRuntime(host="http://127.0.0.1", port=1234, auth_token="secret",
                                  loop_runner=lambda awaitable, timeout=10: asyncio.run(awaitable))
    assert adapter.is_alive() is True
    adapter.ensure_session("main")
    result = adapter.run("false", session="main", timeout=30)
    assert calls["action"] == {"command": "false", "session": "main", "timeout": 30, "check": "silent"}
    assert result["exit_code"] == 7 and result["stdout"] == "bad"
    adapter.close_session("main")
    adapter.upload_file("/tmp/source", "/workspace/source")
    assert calls["create"] == {"session": "main"}
    assert calls["close"] == {"session": "main"}
    assert calls["upload"] == {"source_path": "/tmp/source", "target_path": "/workspace/source"}


def test_project_version_and_swerex_pin_are_consistent():
    root = Path(__file__).resolve().parents[1]
    assert (root / "VERSION").read_text().strip() == "0.7.2"
    assert 'VERSION = "0.7.2"' in (root / "agent" / "version.py").read_text()
    assert "swe-rex==1.4.0" in (root / "requirements.txt").read_text()
    assert "swe-rex==1.4.0" in (root / "container" / "requirements.txt").read_text()


def test_container_launcher_removes_token_from_environment_and_protects_server_process():
    root = Path(__file__).resolve().parents[1]
    script = root / "container" / "swerex_server.py"
    code = (
        "import os, runpy; "
        f"ns=runpy.run_path({str(script)!r}); "
        "token=ns['take_and_scrub_auth_token'](); "
        "print(token == 'unit-test-secret', 'SWEREX_AUTH_TOKEN' not in os.environ)"
    )
    env = os.environ.copy()
    env["SWEREX_AUTH_TOKEN"] = "unit-test-secret"
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True True"
    assert "unit-test-secret" not in result.stderr


def test_sandbox_database_migration_is_additive_and_cascades(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "settings", SimpleNamespace(database_path=tmp_path / "agent.sqlite3"))
    db.init_db()
    conversation = db.create_conversation("sandbox migration")
    cid = conversation["id"]
    db.upsert_sandbox(cid, "container-id", "container-name", 32123, "auth-secret", "main", "running")
    record = db.get_sandbox(cid)
    assert record["container_id"] == "container-id"
    assert record["auth_token"] == "auth-secret"
    assert db.list_sandboxes()[0]["conversation_id"] == cid
    with db.connect() as con:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"conversations", "messages", "tasks", "conversation_sandboxes"} <= tables
    db.delete_conversation(cid)
    assert db.get_sandbox(cid) is None


def test_rootless_requirement_is_enforced_before_container_creation(monkeypatch):
    _fake_db(monkeypatch)
    podman = FakePodman()

    def runner(args, *, env=None, check=True):
        if args[1:3] == ["info", "--format"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"host": {"security": {"rootless": False}}}), stderr="")
        return podman(args, env=env, check=check)

    manager = SandboxManager(runtime_factory=lambda **kw: FakeRuntime(**kw), command_runner=runner)
    with pytest.raises(SandboxError, match="rootless"):
        manager.get_runtime("not-rootless")
    assert not any(args[1:2] == ["run"] for args, _, _ in podman.calls)


def test_delete_refuses_to_remove_container_when_db_and_label_disagree(monkeypatch):
    record = {"conversation_id": "conversation-mismatch", "container_id": "expected-id",
              "container_name": "expected", "host_port": 12345, "auth_token": "secret",
              "session_name": "main", "status": "running"}
    _fake_db(monkeypatch, record)
    podman = FakePodman(existing="different-id")
    manager = SandboxManager(runtime_factory=lambda **kw: FakeRuntime(**kw), command_runner=podman)
    with pytest.raises(SandboxError, match="не совпадает"):
        manager.delete("conversation-mismatch")
    assert not any(args[1:2] == ["rm"] for args, _, _ in podman.calls)


def test_attachment_upload_uses_swerex_upload_without_host_mounts(tmp_path, monkeypatch):
    attachments = tmp_path / "attachments"
    attachments.mkdir()
    source = attachments / "report.pdf"
    source.write_bytes(b"pdf-bytes")
    monkeypatch.setattr(sandbox_manager_module, "settings", SimpleNamespace(
        database_path=tmp_path / "agent.sqlite3", workspace_path=tmp_path / "workspace",
        sandbox_enabled=True, podman_path="podman", sandbox_image="localhost/localagent-sandbox:ubuntu24.04",
        sandbox_network="localagent-sandbox-internal-test", sandbox_cpus=2.0,
        sandbox_memory="4g", sandbox_container_port=8000, swerex_timeout=10.0,
    ))
    record = {"conversation_id": "conversation-upload", "container_id": "container-upload",
              "container_name": "upload", "host_port": 12345, "auth_token": "secret",
              "session_name": "main", "status": "running"}
    _fake_db(monkeypatch, record)
    podman = FakePodman(existing="container-upload")

    class UploadRuntime(FakeRuntime):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.uploads = []

        def run(self, command, *, session="main", timeout=120):
            self.commands.append((command, session, timeout))
            if command.startswith("test -f"):
                return {"stdout": "", "stderr": "", "exit_code": 1, "timed_out": False}
            return {"stdout": "", "stderr": "", "exit_code": 0, "timed_out": False}

        def upload_file(self, source_path, target_path):
            self.uploads.append((source_path, target_path))

    runtime = UploadRuntime()
    manager = SandboxManager(runtime_factory=lambda **kw: runtime, command_runner=podman)
    target = manager.upload_file("conversation-upload", source, "report.pdf")
    assert target.startswith("/workspace/uploads/") and target.endswith("_report.pdf")
    assert runtime.uploads == [(str(source.resolve()), target)]
    assert not any("-v" in args or "--volume" in args for args, _, _ in podman.calls)


def test_swerex_lost_session_is_recreated_once(monkeypatch):
    calls = {"runs": 0, "creates": 0}

    class BashAction:
        def __init__(self, **kwargs): pass

    class CreateBashSessionRequest:
        def __init__(self, **kwargs): pass

    class CloseSessionRequest:
        def __init__(self, **kwargs): pass

    class SessionDoesNotExistError(Exception): pass

    class FakeRemote:
        def __init__(self, **kwargs): pass
        async def is_alive(self, timeout=None): return SimpleNamespace(is_alive=True)
        async def create_session(self, request): calls["creates"] += 1
        async def run_in_session(self, action):
            calls["runs"] += 1
            if calls["runs"] == 1:
                raise SessionDoesNotExistError("lost")
            return SimpleNamespace(output="recovered", exit_code=0, failure_reason="", expect_string="prompt")
        async def close_session(self, request): pass

    swerex = types.ModuleType("swerex")
    runtime_mod = types.ModuleType("swerex.runtime")
    local_mod = types.ModuleType("swerex.runtime.local")
    remote_mod = types.ModuleType("swerex.runtime.remote")
    local_mod.BashAction = BashAction
    local_mod.CreateBashSessionRequest = CreateBashSessionRequest
    local_mod.CloseSessionRequest = CloseSessionRequest
    remote_mod.RemoteRuntime = FakeRemote
    monkeypatch.setitem(sys.modules, "swerex", swerex)
    monkeypatch.setitem(sys.modules, "swerex.runtime", runtime_mod)
    monkeypatch.setitem(sys.modules, "swerex.runtime.local", local_mod)
    monkeypatch.setitem(sys.modules, "swerex.runtime.remote", remote_mod)
    adapter = SwerexRemoteRuntime(host="http://127.0.0.1", port=1, auth_token="secret",
                                  loop_runner=lambda awaitable, timeout=10: asyncio.run(awaitable))
    result = adapter.run("pwd", session="main")
    assert result["stdout"] == "recovered" and result["exit_code"] == 0
    assert calls == {"runs": 2, "creates": 1}


def test_conversation_attachment_cleanup_only_removes_managed_files(tmp_path, monkeypatch):
    from agent import attachments

    database = tmp_path / "agent.sqlite3"
    workspace = tmp_path / "workspace"
    attachment_root = tmp_path / "attachments"
    managed_file = attachment_root / "image.jpg"
    legacy_file = workspace / ".attachments" / "report.pdf"
    outside_file = tmp_path / "keep.txt"
    attachment_root.mkdir()
    (workspace / ".attachments").mkdir(parents=True)
    managed_file.write_bytes(b"image")
    legacy_file.write_bytes(b"pdf")
    outside_file.write_text("keep")

    config = SimpleNamespace(database_path=database, workspace_path=workspace)
    monkeypatch.setattr(db, "settings", config)
    monkeypatch.setattr(attachments, "settings", config)
    db.init_db()
    conversation = db.create_conversation("attachment cleanup")
    cid = conversation["id"]
    db.add_message(cid, "user", "files", attachments=[{"path": str(managed_file)}])
    db.create_task(cid, "process files", [{"workspace_path": ".attachments/report.pdf"}])

    result = attachments.cleanup_conversation_attachments(cid)
    assert result["removed"] == 2
    assert not managed_file.exists() and not legacy_file.exists()
    assert outside_file.exists()


def test_restore_all_reconnects_existing_containers_without_creating_new_ones(monkeypatch):
    record = {"conversation_id": "conversation-restore", "container_id": "container-restore",
              "container_name": "restore", "host_port": 12345, "auth_token": "secret",
              "session_name": "main", "status": "running"}
    _fake_db(monkeypatch, record)
    podman = FakePodman(existing="container-restore")
    runtime = FakeRuntime()
    manager = SandboxManager(runtime_factory=lambda **kw: runtime, command_runner=podman)
    assert manager.restore_all() == {"conversation-restore": "restored"}
    assert runtime.sessions == ["main"]
    assert not any(args[1:2] == ["run"] for args, _, _ in podman.calls)


def test_detach_never_calls_remote_runtime_close():
    class DetachOnlyRuntime(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.detached = False
        def detach(self):
            self.detached = True
        def close(self):
            raise AssertionError("RemoteRuntime.close must not be called on ordinary detach")

    runtime = DetachOnlyRuntime()
    manager = SandboxManager(runtime_factory=lambda **kw: runtime, command_runner=FakePodman())
    manager._runtimes["conversation-detach"] = runtime
    manager.detach_all()
    assert runtime.detached is True


def test_download_route_rejects_traversal_and_symlinks(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from agent import web

    monkeypatch.setattr(web, "settings", SimpleNamespace(workspace_path=tmp_path))
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    safe = downloads / "safe.txt"
    safe.write_text("safe")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    link = downloads / "link.txt"
    link.symlink_to(outside)

    response = web.download_published_file("safe.txt")
    assert Path(response.path) == safe
    with pytest.raises(HTTPException):
        web.download_published_file("../outside.txt")
    with pytest.raises(HTTPException):
        web.download_published_file("link.txt")


def test_publish_file_chooses_unique_non_overwriting_download_names(tmp_path, monkeypatch):
    import base64
    from agent import tools
    from agent import sandbox_manager as sandbox_manager_module

    def fake_execute(cid, command, timeout=120, output_limit=60000):
        if command.startswith("test -f"):
            return {"stdout": "5", "stderr": "", "exit_code": 0, "timed_out": False}
        return {"stdout": base64.b64encode(b"hello").decode(), "stderr": "", "exit_code": 0, "timed_out": False}

    monkeypatch.setattr(sandbox_manager_module, "sandbox_manager", SimpleNamespace(execute=fake_execute))
    config_module = sys.modules["agent.config"]
    monkeypatch.setattr(config_module, "settings", SimpleNamespace(workspace_path=tmp_path))
    first = tools.publish_file_from_sandbox("conversation-publish", "/tmp/hello.txt")
    second = tools.publish_file_from_sandbox("conversation-publish", "/tmp/hello.txt")
    first_target = tmp_path / first["path"]
    second_target = tmp_path / second["path"]
    assert first_target.read_bytes() == b"hello"
    assert second_target.read_bytes() == b"hello"
    assert first_target != second_target
    assert first["filename"] == "hello.txt"
    assert second["filename"] == "hello (1).txt"
    assert first["download_url"] == "/api/files/hello.txt"


def test_container_file_download_uses_bounded_chunks_and_checksum(tmp_path, monkeypatch):
    import base64
    import hashlib
    import re
    from contextlib import nullcontext
    from agent import sandbox_manager as sandbox_manager_module

    manager = sandbox_manager_module.sandbox_manager
    data = bytes(range(256)) * 8192
    commands = []

    class FakeRuntime:
        def run(self, command, *, session="main", timeout=120):
            commands.append(command)
            if "stat -c" in command:
                return {"stdout": str(len(data)), "stderr": "", "exit_code": 0}
            if command.startswith("dd if="):
                match = re.search(r"skip=(\d+) count=(\d+)", command)
                start, count = int(match.group(1)), int(match.group(2))
                return {"stdout": base64.b64encode(data[start:start + count]).decode(), "stderr": "", "exit_code": 0}
            if command.startswith("sha256sum"):
                return {"stdout": hashlib.sha256(data).hexdigest() + "  /tmp/source.bin", "stderr": "", "exit_code": 0}
            raise AssertionError(command)

    monkeypatch.setattr(manager, "_lock_for", lambda conversation_id: nullcontext())
    monkeypatch.setattr(manager, "get_runtime", lambda conversation_id: FakeRuntime())
    target = tmp_path / "downloaded.bin"
    result = manager.download_file("conversation-1", "/tmp/source.bin", target, max_bytes=3 * 1024 * 1024)
    assert result == target
    assert target.read_bytes() == data
    assert sum(command.startswith("dd if=") for command in commands) == 2

    with pytest.raises(ValueError, match="allowed size limit"):
        manager.download_file("conversation-1", "/tmp/source.bin", tmp_path / "too-large.bin", max_bytes=10)
