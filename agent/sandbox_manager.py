"""Conversation-scoped rootless Podman lifecycle manager."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import logging
import subprocess
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
import threading
import base64
import shlex
from pathlib import Path
from typing import Any

from . import db
from .config import settings
from .sandbox_runtime import SwerexRemoteRuntime, SandboxRuntime
from .proxy import XrayProxyManager, ProxyUnavailable

logger = logging.getLogger(__name__)


class _LoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._serve, name="localagent-swerex-loop", daemon=True)
        self.thread.start()

    def _serve(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def __call__(self, awaitable, timeout: float = 660):
        future = asyncio.run_coroutine_threadsafe(awaitable, self.loop)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            raise


class SandboxError(RuntimeError):
    pass


class SandboxBusyError(SandboxError):
    """Raised when a non-terminal operation tries to use a busy Bash session."""


class SandboxManager:
    def __init__(self, *, runtime_factory=None, command_runner=None):
        self._runtime_factory = runtime_factory or self._make_runtime
        self._uses_default_runner = command_runner is None
        self._command_runner = command_runner or self._run_command
        self._proxy_manager = XrayProxyManager(
            self._command_runner, settings.podman_path,
            getattr(settings, "xray_proxy_container", "localagent-xray-proxy"),
        )
        self._loop = _LoopThread()
        self._runtimes: dict[str, SandboxRuntime] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.RLock()
        self._active_operations: dict[str, dict[str, Any]] = {}
        self._diagnostic_operation_samples: dict[tuple[str, str], dict[str, Any]] = {}
        self._new_sessions: set[str] = set()
        self._recreated_containers: set[str] = set()

    def _lock_for(self, conversation_id: str):
        with self._guard:
            return self._locks.setdefault(conversation_id, threading.RLock())

    def get_active_operation(self, conversation_id: str) -> dict[str, Any] | None:
        """Return the active terminal operation, including one recovered from SQLite."""
        if not conversation_id:
            return None
        with self._guard:
            active = self._active_operations.get(conversation_id)
            if active:
                return {
                    "operation_id": str(active.get("operation_id") or ""),
                    "state": str(active.get("state") or "running"),
                    "session": str(active.get("session") or "main"),
                }
        persisted = self._find_persisted_active_operation_details(conversation_id)
        if not isinstance(persisted, dict) or not persisted.get("operation_id"):
            return None
        return {
            "operation_id": str(persisted["operation_id"]),
            "state": str(persisted.get("state") or "running"),
            "session": "main",
        }

    def ensure_session_idle(self, conversation_id: str) -> None:
        """Fail before any ordinary runtime.run call can collide with a live command."""
        active = self.get_active_operation(conversation_id)
        if active:
            raise SandboxBusyError(
                "Bash-сессия занята фоновой терминальной операцией "
                f"(состояние: {active['state']}). "
                "Сначала вызови terminal с пустой command для ожидания, "
                "передай интерактивный ввод через terminal или отправь CTRL+C. "
                "Обычные команды и операции с файлами пока не выполняются."
            )

    @staticmethod
    def _run_command(args: list[str], *, env=None, check=True):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=90, env=env, check=False)
        except FileNotFoundError as exc:
            raise SandboxError("Podman не найден. Установите rootless Podman и проверьте PATH.") from exc
        except subprocess.TimeoutExpired as exc:
            raise SandboxError("Истёк тайм-аут выполнения команды Podman.") from exc
        if check and result.returncode != 0:
            # Podman output can contain infrastructure details; do not echo env/token.
            detail = (result.stderr or result.stdout or "").strip()[:500]
            raise SandboxError(f"Podman завершился с кодом {result.returncode}: {detail}")
        return result

    def _podman(self, *args, env=None, check=True):
        return self._command_runner([settings.podman_path, *args], env=env, check=check)

    def _make_runtime(self, *, port: int, token: str) -> SandboxRuntime:
        return SwerexRemoteRuntime(
            host="http://127.0.0.1", port=port, auth_token=token,
            timeout=settings.swerex_timeout, loop_runner=self._loop,
        )

    @staticmethod
    def _container_name(conversation_id: str) -> str:
        digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:20]
        return f"localagent-sandbox-{digest}"

    def _inspect(self, container_id: str) -> dict[str, Any]:
        result = self._podman("inspect", container_id)
        try:
            payload = json.loads(result.stdout)
            return payload[0]
        except (ValueError, IndexError, TypeError) as exc:
            raise SandboxError("Podman inspect вернул некорректные данные контейнера.") from exc

    def _find_container(self, conversation_id: str) -> str | None:
        result = self._podman(
            "ps", "-a", "--no-trunc",
            "--filter", f"label=localagent.task_id={conversation_id}",
            "--format", "{{.ID}}", check=False,
        )
        if result.returncode != 0:
            raise SandboxError("Не удалось найти контейнер диалога через Podman.")
        ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(ids) > 1:
            raise SandboxError("Обнаружено несколько контейнеров для одного диалога; автоматический выбор небезопасен.")
        return ids[0] if ids else None

    @staticmethod
    def _host_port(inspect_data: dict[str, Any]) -> int:
        ports = (inspect_data.get("NetworkSettings") or {}).get("Ports") or {}
        bindings = ports.get(f"{settings.sandbox_container_port}/tcp") or []
        for item in bindings:
            if item.get("HostIp") == "127.0.0.1" and item.get("HostPort"):
                try:
                    return int(item["HostPort"])
                except (TypeError, ValueError):
                    break
        raise SandboxError("Не удалось получить опубликованный loopback host port из Podman inspect.")

    def _create_container(self, conversation_id: str, token: str) -> tuple[str, int]:
        info = self._podman("info", "--format", "json")
        try:
            rootless_value = json.loads(info.stdout).get("host", {}).get("security", {}).get("rootless")
            rootless = rootless_value is True or str(rootless_value).strip().lower() == "true"
        except (ValueError, AttributeError, TypeError):
            rootless = False
        if not rootless:
            raise SandboxError("Podman не подтверждён как rootless; создание контейнера остановлено.")
        host_workspace = settings.workspace_path / "work"
        if host_workspace.is_symlink():
            raise SandboxError(
                f"Каталог общей рабочей папки не должен быть символической ссылкой: {host_workspace}"
            )
        try:
            host_workspace.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SandboxError(
                f"Не удалось создать общую рабочую папку: {host_workspace}: {exc}"
            ) from exc

        if not host_workspace.is_dir():
            raise SandboxError(
                f"Путь общей рабочей папки не является каталогом: {host_workspace}"
            )

        host_workspace = host_workspace.resolve()

        env = os.environ.copy()
        env["SWEREX_AUTH_TOKEN"] = token
        name = self._container_name(conversation_id)
        result = self._podman(
            "run", "-d", "--name", name,
            "--label", f"localagent.task_id={conversation_id}",
            "--network", settings.sandbox_network,
            "--cpus", str(settings.sandbox_cpus),
            "--memory", settings.sandbox_memory,
            "-p", f"127.0.0.1::{settings.sandbox_container_port}",
            "-v", f"{host_workspace}:/host-workspace:rw",
            "-e", "SWEREX_AUTH_TOKEN",
            settings.sandbox_image,
            env=env,
        )
        lines = result.stdout.strip().splitlines()
        if not lines:
            raise SandboxError("Podman не вернул ID созданного контейнера.")
        container_id = lines[-1]
        try:
            inspect_data = self._inspect(container_id)
            if not (inspect_data.get("State") or {}).get("Running"):
                raise SandboxError("Созданный sandbox-контейнер не перешёл в состояние Running.")
            return container_id, self._host_port(inspect_data)
        except Exception:
            self._podman("rm", "-f", container_id, check=False)
            raise

    def get_runtime(self, conversation_id: str, *, check_proxy: bool = True) -> SandboxRuntime:
        if not conversation_id:
            raise ValueError("conversation_id is required")
        if not settings.sandbox_enabled:
            raise SandboxError("Новая sandbox отключена настройкой SANDBOX_ENABLED.")
        # Only the real host-backed manager enforces the shared proxy lifecycle;
        # injected command runners remain deterministic in unit tests.
        if check_proxy and getattr(settings, "sandbox_proxy_required", True) and self._uses_default_runner:
            try:
                self._proxy_manager.ensure(settings.sandbox_network)
            except ProxyUnavailable as exc:
                raise SandboxError(f"Прокси для интернет-доступа песочницы недоступен: {exc}") from exc
        with self._lock_for(conversation_id):
            cached = self._runtimes.get(conversation_id)
            if cached and cached.is_alive():
                return cached

            record = db.get_sandbox(conversation_id)
            found_id = self._find_container(conversation_id)
            # A running, previously persisted container may already own a live
            # Bash session. Only create a session when the container is new or
            # was stopped and has just been started.
            session_may_be_missing = not bool(record)
            if record:
                container_id = record["container_id"]
                if found_id and found_id != container_id:
                    raise SandboxError("ID контейнера в БД не совпадает с контейнером по label.")
                if not found_id:
                    # The workspace is a host bind mount, so it survives container
                    # replacement. Recreate the missing container with its persisted
                    # runtime token and report the new shell to the caller.
                    container_id, port = self._create_container(conversation_id, record["auth_token"])
                    db.upsert_sandbox(
                        conversation_id, container_id, self._container_name(conversation_id),
                        port, record["auth_token"], "main", "running",
                    )
                    found_id = container_id
                    session_may_be_missing = True
                    with self._guard:
                        self._recreated_containers.add(conversation_id)
                inspect_data = self._inspect(found_id)
                if not (inspect_data.get("State") or {}).get("Running"):
                    self._podman("start", found_id)
                    session_may_be_missing = True
                    inspect_data = self._inspect(found_id)
                    if not (inspect_data.get("State") or {}).get("Running"):
                        raise SandboxError("Существующий контейнер не удалось запустить; данные не удалялись.")
                port = self._host_port(inspect_data)
                token = record["auth_token"]
                db.upsert_sandbox(conversation_id, found_id, record["container_name"], port, token, "main", "running")
            else:
                if found_id:
                    raise SandboxError("Найден контейнер с label диалога, но запись БД отсутствует; токен восстановления недоступен.")
                token = secrets.token_urlsafe(32)
                container_id, port = self._create_container(conversation_id, token)
                try:
                    db.upsert_sandbox(conversation_id, container_id, self._container_name(conversation_id), port, token, "main", "running")
                except Exception:
                    self._podman("rm", "-f", container_id, check=False)
                    raise

            runtime = self._runtime_factory(port=port, token=token)
            alive = False
            for attempt in range(10):
                if runtime.is_alive():
                    alive = True
                    break
                if attempt < 9:
                    time.sleep(1)
            if not alive:
                raise SandboxError("Контейнер работает, но SWE-ReX Remote Server не отвечает или отклонил авторизацию.")
            session_created = runtime.ensure_session("main") if session_may_be_missing else False
            if session_created:
                with self._guard:
                    self._new_sessions.add(conversation_id)
            persisted_active_operation = self._find_persisted_active_operation_details(conversation_id)
            if persisted_active_operation:
                # The persistent PTY may still be executing a command after a
                # LocalAgent restart. Do not send auth cleanup or proxy probes
                # through that occupied shell; get_operation must be able to poll it.
                self._runtimes[conversation_id] = runtime
                return runtime
            # The auth token is needed by the SWE-ReX server only. Remove it
            # from the persistent interactive shell before the model can use it.
            cleared = runtime.run("unset SWEREX_AUTH_TOKEN", session="main", timeout=10)
            if cleared.get("shell_recovered") is True:
                # run() had to create a missing PTY while reconnecting. Preserve
                # that fact so the terminal can report loss of shell-local state.
                with self._guard:
                    self._new_sessions.add(conversation_id)
            if cleared.get("timed_out") and cleared.get("shell_recovered") is True:
                # A stale interactive shell may have survived a server/client
                # reconnect. The adapter reset it; retry this idempotent cleanup.
                cleared = runtime.run("unset SWEREX_AUTH_TOKEN", session="main", timeout=10)
            if cleared.get("exit_code") != 0 or cleared.get("timed_out"):
                raise SandboxError("Не удалось очистить переменную авторизации или проверить Bash-сессию.")
            if check_proxy and getattr(settings, "sandbox_proxy_required", True) and self._uses_default_runner:
                self._probe_proxy_from_sandbox(runtime)
            self._runtimes[conversation_id] = runtime
            return runtime

    def _probe_proxy_from_sandbox(self, runtime: SandboxRuntime) -> dict[str, Any]:
        """Probe DNS, SOCKS5 and Privoxy egress from the actual sandbox network."""
        script = r'''import json, socket, urllib.request
host = "localagent-xray-proxy.dns.podman"
result = {"dns_resolved": False, "socks_reachable": False, "privoxy_egress": False, "external_internet": False, "failed_stage": "dns"}
try:
    addresses = socket.getaddrinfo(host, 10808, type=socket.SOCK_STREAM)
    result["dns_resolved"] = bool(addresses)
    result["failed_stage"] = "socks5_tcp"
    with socket.create_connection((host, 10808), timeout=3) as sock:
        result["socks_reachable"] = True
        sock.settimeout(3)
        sock.sendall(b"\x05\x01\x00")
        response = sock.recv(2)
        result["socks_handshake"] = len(response) == 2 and response[0] == 5 and response[1] in (0, 2)
        result["socks_reachable"] = result["socks_reachable"] and result["socks_handshake"]
        if not result["socks_reachable"]:
            result["failed_stage"] = "socks5_handshake"
    if result["socks_reachable"]:
        result["failed_stage"] = "privoxy_egress"
except Exception as exc:
    result["error"] = "DNS/SOCKS: " + type(exc).__name__
    if not result["dns_resolved"]:
        result["failed_stage"] = "dns"
    elif not result.get("socks_handshake"):
        result["failed_stage"] = "socks5_handshake"
    else:
        result["failed_stage"] = "socks5_tcp"
try:
    proxy = urllib.request.ProxyHandler({"http": "http://127.0.0.1:8118", "https": "http://127.0.0.1:8118"})
    opener = urllib.request.build_opener(proxy)
    errors = []
    for url in ("https://www.cloudflare.com/cdn-cgi/trace", "https://api.ipify.org"):
        try:
            with opener.open(url, timeout=8) as response:
                body = response.read(256)
                if response.status == 200 and body:
                    result["privoxy_egress"] = True
                    result["external_internet"] = True
                    break
        except Exception as exc:
            errors.append(type(exc).__name__)
    if not result["external_internet"]:
        result["error"] = "Privoxy/egress: " + ",".join(errors[:2])
        result["failed_stage"] = "privoxy_egress"
except Exception as exc:
    result["error"] = "Privoxy: " + type(exc).__name__
    result["failed_stage"] = "privoxy_egress"
print(json.dumps(result))
'''
        try:
            command = "python3 -c " + shlex.quote(script)
            result = runtime.run(command, session="main", timeout=25)
            payload = json.loads(result.get("stdout") or "{}") if result.get("exit_code") == 0 else {
                "error": str(result.get("stderr") or "proxy probe failed")[:400]
            }
            if not isinstance(payload, dict):
                payload = {"error": "invalid proxy probe response"}
        except Exception as exc:
            payload = {"error": type(exc).__name__}
        return self._proxy_manager.record_runtime_probe(payload)

    @staticmethod
    def _command_requires_network(command: str) -> bool:
        return bool(re.search(
            r"(?i)(?:\bapt(?:-get)?\b|\bpip3?\b|\buv\s+pip\b|\bcurl\b|\bwget\b|"
            r"\bgit\s+clone\b|\bnpm\b|\bpnpm\b|\byarn\b|\bcargo\s+(?:install|fetch)\b|"
            r"\bgo\s+get\b|\bplaywright\s+install\b|\bchromium\s+install\b|"
            r"\brequests\b|\burllib\b|\bhttpx\b|https?://)",
            str(command or ""),
        ))

    def _ensure_network_proxy(self, runtime: SandboxRuntime, command: str) -> dict[str, Any] | None:
        if (not getattr(settings, "sandbox_proxy_required", True)
                or not self._command_requires_network(command) or not self._uses_default_runner):
            return None
        try:
            status = self._probe_proxy_from_sandbox(runtime)
            if status.get("state") == "online":
                return status
            # One controlled lifecycle recheck/start and one bounded health retry.
            self._proxy_manager.ensure(settings.sandbox_network, force=True)
            status = self._probe_proxy_from_sandbox(runtime)
            if status.get("state") == "online":
                return status
            raise SandboxError(
                "Интернет песочницы недоступен через общий Xray/Privoxy: "
                + str(status.get("probe_error") or status.get("state") or "health check failed")[:400]
            )
        except ProxyUnavailable as exc:
            raise SandboxError(f"Общий Xray-прокси недоступен: {exc}") from exc

    def proxy_status(self) -> dict[str, Any]:
        return self._proxy_manager.status()

    def start_operation(self, conversation_id: str, command: str, timeout: int | None = None,
                        idempotency_key: str | None = None) -> dict[str, Any]:
        """Start a remote terminal operation and return immediately with its ID."""
        if not isinstance(command, str) or not command.strip() or "\x00" in command:
            raise ValueError("command must be a non-empty string without NUL bytes")
        if len(command) > 100_000:
            raise ValueError("command exceeds 100000 characters")
        if timeout is not None and (not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 600):
            raise ValueError("timeout must be None or between 1 and 600 seconds")
        syntax = subprocess.run(["bash", "-n", "-c", command], capture_output=True, text=True, timeout=5, check=False)
        if syntax.returncode:
            syntax_error = str(syntax.stderr or "")
            incomplete_input = any(marker in syntax_error for marker in (
                "unexpected EOF while looking for matching",
                "syntax error: unexpected end of file",
            ))
            if not incomplete_input:
                return {"state": "failed", "exit_code": 2, "stdout": syntax.stdout or "",
                        "stderr": (syntax_error or "Invalid Bash syntax")[:4000], "syntax_error": True}
        with self._lock_for(conversation_id):
            self.ensure_session_idle(conversation_id)
            runtime = self.get_runtime(conversation_id)
            self._ensure_network_proxy(runtime, command)
            starter = getattr(runtime, "start_operation", None)
            if not callable(starter):
                raise SandboxError("Подключённый SandboxRuntime не поддерживает фоновые операции SWE-ReX.")
            # Give Bash a recognizable secondary prompt. If a heredoc, quote,
            # compound statement or line continuation is left open, Bash emits
            # this marker instead of looking like a silently running command.
            shell_command = "export PS2='__LOCALAGENT_BASH_CONTINUATION__'\n" + command
            result = starter(shell_command, session="main", timeout=timeout, idempotency_key=idempotency_key)
            operation_id = str(result.get("operation_id") or "")
            if not operation_id:
                raise SandboxError("SWE-ReX не вернул operation_id для фоновой операции.")
            with self._guard:
                self._active_operations[conversation_id] = {"operation_id": operation_id, "session": "main"}
        return {"operation_id": operation_id, "session": "main", "state": result.get("state", "running"),
                "created_at": result.get("created_at"), "command": command, "background": True}

    @staticmethod
    def _terminal_control_text(output: str, state: str, elapsed: float, idle_seconds: float,
                               *, waiting_input: bool = False,
                               continuation_input: bool = False) -> str:
        text = output or ""
        if continuation_input:
            notice = "[Bash ожидает продолжения незавершённой команды (например, закрывающий EOF heredoc или кавычку). Передай недостающий текст через terminal.]"
        elif waiting_input:
            notice = "[Терминал ожидает ввода. Передай ответ следующей командой terminal.]"
        elif state in {"running", "waiting_input"}:
            if idle_seconds >= 0:
                notice = (
                    f"[Команда ещё выполняется ({elapsed:.1f} с); последний вывод был {idle_seconds:.1f} с назад. "
                    "Что делать: подождать — вызвать terminal с пустой командой; "
                    "передать ввод — отправить текст; прервать — Ctrl+C.]"
                )
            else:
                notice = f"[Команда ещё выполняется ({elapsed:.1f} с).]"
        else:
            notice = ""
        if notice:
            return (text.rstrip() + "\n" + notice).lstrip("\n")
        return text

    def terminal_command(self, conversation_id: str, command: str = "", *,
                         timeout: int | None = None,
                         idempotency_key: str | None = None) -> dict[str, Any]:
        """Single-tool terminal facade: start, wait, interactive input, or Ctrl+C.

        operation IDs and output cursors stay inside LocalAgent. The model only
        sees the terminal transcript and a short status notice when work continues.
        """
        if not isinstance(command, str) or "\x00" in command:
            raise ValueError("terminal command/input must be a string without NUL bytes")
        if len(command) > 100_000:
            raise ValueError("terminal command/input exceeds 100000 characters")

        with self._guard:
            active = self._active_operations.get(conversation_id)
        recovered_operation = False
        if active is None:
            recovered_id = self._find_persisted_active_operation(conversation_id)
            if recovered_id:
                recovered_operation = True
                recovered_now = time.monotonic()
                recovered_record = self._find_persisted_active_operation_details(conversation_id) or {}
                active = {
                    "operation_id": recovered_id, "session": "main",
                    "cursor": int(recovered_record.get("cursor", 0) or 0),
                    "output": str(recovered_record.get("_terminal_output", "") or ""),
                    "command": recovered_record.get("_terminal_command"),
                    "command_timeout": recovered_record.get("_terminal_hard_timeout"),
                    "runtime_errors": list(recovered_record.get("_terminal_runtime_errors") or [])[-10:],
                    "started_monotonic": recovered_now - max(0.0, time.time() - float(recovered_record.get("_terminal_started_at") or time.time())),
                    "last_output_monotonic": recovered_now - max(0.0, time.time() - float(recovered_record.get("_terminal_last_output_at") or time.time())),
                    "started_at_wall": float(recovered_record.get("_terminal_started_at") or time.time()),
                    "last_output_at_wall": float(recovered_record.get("_terminal_last_output_at") or time.time()),
                }
                with self._guard:
                    self._active_operations[conversation_id] = active

        # A live operation owns the one persistent terminal. A blank call waits;
        # Ctrl+C interrupts; any other text is sent as interactive input.
        if active is None:
            if not command.strip():
                return {"_terminal_text": "Терминал свободен.", "state": "idle"}
            with self._guard:
                self._new_sessions.discard(conversation_id)
            started = self.start_operation(
                conversation_id, command, timeout, idempotency_key=idempotency_key,
            )
            if started.get("state") == "failed":
                output = str(started.get("stdout") or started.get("stderr") or "")
                return {"_terminal_text": output or "Не удалось запустить команду.",
                        "state": "failed", "exit_code": started.get("exit_code")}
            operation_id = str(started["operation_id"])
            started_now = time.monotonic()
            active = {
                "operation_id": operation_id, "session": "main", "cursor": 0,
                "output": "", "command": command, "command_timeout": timeout,
                "runtime_errors": [], "started_monotonic": started_now,
                "last_output_monotonic": started_now,
                "started_at_wall": time.time(), "last_output_at_wall": time.time(),
            }
            with self._guard:
                self._active_operations[conversation_id] = active
        else:
            operation_id = str(active["operation_id"])
            if command in {"\x03", "CTRL+C", "ctrl+c", "Ctrl+C"}:
                interrupted = self.interrupt_operation(conversation_id, operation_id)
                active["interrupt_requested"] = True
                active["interrupt_result"] = interrupted
            elif command:
                # A previous PS2 marker is historical once more input is sent;
                # only a newly emitted marker can indicate that Bash still waits.
                active["output"] = str(active.get("output") or "").replace(
                    "__LOCALAGENT_BASH_CONTINUATION__", ""
                )
                active["waiting_for_continuation"] = False
                self.send_operation_input(conversation_id, operation_id, command, sendline=True)

        command_started_at = float(active.get("started_monotonic", time.monotonic()))
        call_started_at = time.monotonic()
        idle_limit = max(0.1, float(getattr(settings, "terminal_idle_timeout_seconds", 300)))
        max_wait = max(1.0, float(getattr(settings, "terminal_max_wait_seconds", 300)))
        last_output_at = float(active.get("last_output_monotonic", call_started_at))
        state = "running"
        latest: dict[str, Any] = {}
        runtime_errors = 0
        runtime_error_text = None
        diagnostic_reason = None
        poll_count = 0
        poll_interval = max(0.25, float(getattr(settings, "terminal_poll_interval_seconds", 2)))
        while True:
            now = time.monotonic()
            elapsed = now - command_started_at
            remaining = max_wait - (now - call_started_at)
            if remaining <= 0:
                diagnostic_reason = "max_wait"
                break
            try:
                latest = self.get_operation(
                    conversation_id, operation_id,
                    cursor=int(active.get("cursor", 0)),
                    max_output_chars=1_048_576,
                    wait_timeout=min(poll_interval, max(0.1, remaining)),
                )
                poll_count += 1
                runtime_errors = 0
                runtime_error_text = None
            except Exception as exc:
                runtime_errors += 1
                runtime_error_text = f"{type(exc).__name__}: {str(exc)[:500]}"
                active["last_runtime_error"] = runtime_error_text
                active.setdefault("runtime_errors", []).append({
                    "elapsed_seconds": round(max(0.0, time.monotonic() - command_started_at), 3),
                    "error": runtime_error_text,
                })
                active["runtime_errors"] = active["runtime_errors"][-10:]
                # A single failed poll is not proof of runtime failure.
                if runtime_errors >= 3:
                    state = "runtime_unresponsive"
                    diagnostic_reason = "runtime_unresponsive"
                    latest = {"state": state, "runtime_error": runtime_error_text}
                    break
                time.sleep(min(1.0, max(0.1, remaining)))
                continue
            chunk = str(latest.get("stdout") or latest.get("output") or "")
            if chunk:
                active["output"] = str(active.get("output") or "") + chunk
                last_output_at = time.monotonic()
                active["last_output_monotonic"] = last_output_at
                active["last_output_at_wall"] = time.time()
            if "__LOCALAGENT_BASH_CONTINUATION__" in str(active.get("output") or ""):
                # This is Bash's PS2 prompt: the submitted command is syntactically
                # incomplete and Bash is waiting for more command text (e.g. heredoc EOF).
                state = "waiting_input"
                active["waiting_for_continuation"] = True
                latest["state"] = "waiting_input"
                latest["waiting_input"] = True
                break
            active["output_truncated"] = bool(active.get("output_truncated")) or bool(
                latest.get("stdout_truncated") or latest.get("output_truncated")
            )
            active["cursor"] = int(latest.get("next_cursor", latest.get("cursor", active.get("cursor", 0))) or 0)
            state = str(latest.get("state") or "failed")
            if state in {"completed", "interrupted", "timed_out", "failed"}:
                break
            if state == "waiting_input":
                break
            # One quiet poll is not enough to distinguish a silent command from
            # a transient gap. Require a second observation before handing off
            # to diagnostics; waiting_input and terminal states were handled above.
            if poll_count >= 2 and time.monotonic() - last_output_at >= idle_limit:
                diagnostic_reason = "idle_timeout"
                break

        elapsed = max(0.0, time.monotonic() - command_started_at)
        output = str(active.get("output") or "")
        output = output.replace("__LOCALAGENT_BASH_CONTINUATION__", "")
        waiting = state == "waiting_input"
        still_running = state not in {"completed", "interrupted", "timed_out", "failed"}
        if state == "failed" and (latest.get("stderr") or latest.get("failure_reason")):
            diagnostic = str(latest.get("stderr") or latest.get("failure_reason"))
            if diagnostic and diagnostic not in output:
                output = (output.rstrip() + "\n" + diagnostic).lstrip("\n")
        if state == "timed_out":
            diagnostic = str(latest.get("stderr") or "Команда превысила установленный жёсткий timeout.")
            if diagnostic and diagnostic not in output:
                output = (output.rstrip() + "\n" + diagnostic).lstrip("\n")

        shell_state = "busy" if still_running else "unknown"
        if not still_running:
            try:
                # The operation has finished, so the PTY is no longer busy.
                # Verify it with a harmless command; never call create_session()
                # as a health check (SWE-ReX logs SessionExistsError server-side).
                marker = "__LOCALAGENT_TERMINAL_READY__"
                runtime = self.get_runtime(conversation_id)
                runner = getattr(runtime, "run", None)
                if callable(runner):
                    probe = runner(f"printf '{marker}'", session="main", timeout=10)
                    verified = (
                        probe.get("exit_code") == 0
                        and not probe.get("timed_out")
                        and marker in str(probe.get("stdout") or "")
                    )
                    session_was_recreated = bool(probe.get("shell_recovered"))
                else:
                    # Compatibility for lightweight runtime adapters that expose
                    # only session creation; production SandboxRuntime implements run().
                    session_was_recreated = bool(runtime.ensure_session("main"))
                    verified = session_was_recreated
                    probe = {"stderr": "Runtime has no run() health-check method"}
                with self._guard:
                    created_during_recovery = conversation_id in self._new_sessions
                    self._new_sessions.discard(conversation_id)
                if verified:
                    shell_state = (
                        "recreated" if session_was_recreated or
                        (recovered_operation and created_during_recovery) else "available"
                    )
                    if shell_state == "recreated":
                        output = (output.rstrip() + "\n" +
                                  "[Bash-сессия была создана заново; переменные и функции предыдущей сессии не сохранены.]").lstrip("\n")
                else:
                    shell_state = "unavailable"
                    detail = str(probe.get("stderr") or probe.get("failure_reason") or "Bash probe returned no confirmation")
                    output = (output.rstrip() + "\n" +
                              "[Не удалось подтвердить работоспособность Bash-сессии: " + detail[:300] + "]").lstrip("\n")
            except Exception as session_exc:
                shell_state = "unavailable"
                output = (output.rstrip() + "\n" +
                          "[Не удалось подтвердить/восстановить Bash-сессию: " +
                          f"{type(session_exc).__name__}: {str(session_exc)[:300]}]").lstrip("\n")
        text = self._terminal_control_text(
            output, "running" if still_running else state, elapsed,
            max(0.0, time.monotonic() - last_output_at),
            waiting_input=waiting,
            continuation_input=bool(active.get("waiting_for_continuation")),
        )
        if active.get("output_truncated"):
            text = (text.rstrip() + "\n" +
                    "[Внимание: SWE-ReX сообщил об ограничении объёма одного фрагмента вывода; "
                    "экранный вывод может быть неполным.]").lstrip("\n")
        if latest.get("container_recreated"):
            text = (text.rstrip() + "\n" +
                    "[Контейнер отсутствовал и был автоматически пересоздан. Общая рабочая папка сохранена; "
                    "результат прежней терминальной операции недоступен.]").lstrip("\n")
        if still_running:
            active["state"] = "waiting_input" if waiting else state
        idle_seconds = max(0.0, time.monotonic() - last_output_at)
        return {
            "_terminal_text": text,
            "_terminal_output": output,
            "_terminal_command": active.get("command"),
            "_terminal_hard_timeout": active.get("command_timeout"),
            "_terminal_started_at": active.get("started_at_wall"),
            "_terminal_last_output_at": active.get("last_output_at_wall"),
            "_terminal_runtime_errors": list(active.get("runtime_errors") or [])[-10:],
            "state": state if not still_running else ("waiting_input" if waiting else state),
            "operation_id": operation_id if still_running else None,
            "cursor": active.get("cursor", 0),
            "container_recreated": bool(latest.get("container_recreated")),
            "shell_recreated": bool(latest.get("shell_recreated")),
            "exit_code": latest.get("exit_code"),
            "timed_out": latest.get("timed_out", state == "timed_out"),
            "interrupted": latest.get("interrupted", state == "interrupted"),
            "duration_seconds": latest.get("duration_seconds", elapsed),
            "output_truncated": bool(active.get("output_truncated")),
            "shell_state": shell_state,
            "input_wait_reason": (
                "bash_continuation" if waiting and active.get("waiting_for_continuation")
                else "interactive_program" if waiting else None
            ),
            "_diagnostic_due": bool(still_running and not waiting and diagnostic_reason),
            "_diagnostic_reason": diagnostic_reason,
            "_monitor_state": (
                "running_silent" if still_running and idle_seconds >= idle_limit
                else ("running" if still_running else state)
            ),
            "_elapsed_seconds": round(elapsed, 3),
            "_idle_seconds": round(idle_seconds, 3),
            "_runtime_error": runtime_error_text,
        }

    def collect_terminal_diagnostics(
        self, conversation_id: str, operation_id: str, *,
        command: str = "", output: str = "", elapsed_seconds: float = 0,
        idle_seconds: float = 0, runtime_error: str | None = None,
        started_at: str | None = None, last_output_at: str | None = None,
        cursor: int = 0,
    ) -> dict[str, Any]:
        """Collect bounded diagnostics without touching the occupied main PTY."""
        from datetime import datetime, timezone

        def age_from(value, fallback):
            if value is None:
                return max(0.0, float(fallback or 0))
            try:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return max(0.0, time.time() - float(value))
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                return max(0.0, float(fallback or 0))

        record = db.get_sandbox(conversation_id) or {}
        container_id = str(record.get("container_id") or "")
        elapsed_now = age_from(started_at, elapsed_seconds)
        sample_key = (str(conversation_id), str(operation_id))
        samples = getattr(self, "_diagnostic_operation_samples", None)
        if samples is None:
            samples = {}
            self._diagnostic_operation_samples = samples
        sample_state = samples.setdefault(sample_key, {
            "cursor": max(0, int(cursor or 0)),
            "started_at": started_at,
            "last_output_at": last_output_at,
        })
        sample_state["cursor"] = max(int(sample_state.get("cursor") or 0), max(0, int(cursor or 0)))
        if started_at is not None:
            sample_state["started_at"] = started_at
        if last_output_at is not None and sample_state.get("last_output_at") is None:
            sample_state["last_output_at"] = last_output_at
        elapsed_now = age_from(sample_state.get("started_at"), elapsed_seconds)
        idle_now = age_from(sample_state.get("last_output_at"), idle_seconds)
        snapshot: dict[str, Any] = {
            "sampled_at": db.now(), "operation_id": str(operation_id),
            "command": str(command or "")[:100000],
            "terminal_output_tail": str(output or "")[-3000:],
            "elapsed_seconds": round(elapsed_now, 1),
            "idle_seconds": round(idle_now, 1),
            "runtime_error": str(runtime_error or "")[:500],
            "container_id": container_id or None,
            "operation_state": "unknown",
        }
        runtime = self._runtimes.get(conversation_id)
        if runtime is None and record:
            try:
                runtime = self._runtime_factory(port=record["host_port"], token=record["auth_token"])
            except Exception as exc:
                snapshot["runtime_client_error"] = type(exc).__name__
        try:
            snapshot["runtime_responsive"] = bool(runtime.is_alive()) if runtime is not None else False
        except Exception as exc:
            snapshot["runtime_responsive"] = False
            snapshot["runtime_health_error"] = type(exc).__name__
        if snapshot["runtime_responsive"] and runtime is not None:
            getter = getattr(runtime, "get_operation", None)
            if callable(getter):
                try:
                    operation = getter(
                        operation_id, session="main", cursor=int(sample_state.get("cursor") or 0),
                        max_output_chars=2048, wait_timeout=0,
                    )
                    snapshot["operation_state"] = str(operation.get("state") or "unknown")
                    snapshot["operation_exit_code"] = operation.get("exit_code")
                    next_cursor = operation.get("next_cursor")
                    if isinstance(next_cursor, int):
                        sample_state["cursor"] = max(int(sample_state.get("cursor") or 0), next_cursor)
                    delta = str(operation.get("output") or "")
                    if delta:
                        snapshot["new_output_since_cursor"] = delta[-1500:]
                        snapshot["terminal_output_tail"] = delta[-3000:]
                        sample_state["last_output_at"] = time.time()
                        snapshot["idle_seconds"] = 0.0
                        # Keep the normal terminal monitor's cursor and idle clock
                        # synchronized with this diagnostic-side observation.
                        with self._guard:
                            active = self._active_operations.get(conversation_id)
                            if active and active.get("operation_id") == operation_id:
                                active["output"] = str(active.get("output") or "") + delta
                                active["cursor"] = max(
                                    int(active.get("cursor") or 0),
                                    int(next_cursor or sample_state.get("cursor") or 0),
                                )
                                active["last_output_monotonic"] = time.monotonic()
                                active["last_output_at_wall"] = time.time()
                    else:
                        snapshot["idle_seconds"] = round(
                            age_from(sample_state.get("last_output_at"), idle_seconds), 1
                        )
                    snapshot["diagnostic_cursor"] = sample_state.get("cursor")
                except Exception as exc:
                    snapshot["operation_poll_error"] = type(exc).__name__
        if container_id:
            try:
                proc = subprocess.run(
                    [settings.podman_path, "inspect", container_id],
                    capture_output=True, text=True, timeout=8, check=False,
                )
                output_text = str(proc.stdout or proc.stderr or "")
                secret = str(record.get("auth_token") or "")
                if secret:
                    output_text = output_text.replace(secret, "[REDACTED]")
                snapshot["container_inspect"] = {
                    "returncode": int(proc.returncode), "output": output_text[:3000],
                }
            except Exception as exc:
                snapshot["container_inspect"] = {"error": type(exc).__name__}
        return snapshot

    def run_diagnostic_terminal(self, conversation_id: str, command: str, *, timeout: int = 10) -> dict[str, Any]:
        """Run one allow-listed read-only command in an isolated, unprivileged SWE-ReX session."""
        if not isinstance(command, str) or not command.strip() or len(command) > 2000 or "\x00" in command:
            raise ValueError("Диагностическая команда должна быть непустой и не длиннее 2000 символов.")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 20:
            raise ValueError("Тайм-аут диагностической команды должен быть от 1 до 20 секунд.")
        argv = shlex.split(command, posix=True)
        allowed = {
            "ps", "ls", "stat", "file", "head", "tail", "cat", "df", "free",
            "uptime", "ss", "lsof", "du", "wc", "readlink", "realpath",
            "id", "whoami", "uname", "date", "mount", "namei",
            "pip", "pip3", "dpkg-query", "dpkg", "npm", "git", "test",
        }
        executable = os.path.basename(argv[0]) if argv else ""
        if not argv or executable not in allowed:
            raise ValueError("Разрешены read-only проверки: ps, ls, stat, file, head, tail, cat, df, free, uptime, ss, lsof, du, wc, readlink, realpath, id, whoami, uname, date, mount, namei, pip show, dpkg-query -W, dpkg -s, npm list, git status/diff/log/show, test -e/-f/-d.")
        read_only_subcommands = {
            "pip": {"show"}, "pip3": {"show"},
            "dpkg-query": {"-W", "--show"}, "dpkg": {"-s"},
            "npm": {"list", "ls"},
            "git": {"status", "diff", "log", "show", "rev-parse"},
            "test": {"-e", "-f", "-d", "-s"},
        }
        if executable in read_only_subcommands and (
            len(argv) < 2 or argv[1] not in read_only_subcommands[executable]
        ):
            raise ValueError(f"Для {executable} разрешены только read-only подкоманды: "
                             + ", ".join(sorted(read_only_subcommands[executable])))
        forbidden = {"-exec", "-execdir", "-delete", "-fprint", "-fls", "--pre", "--post", "--output",
                     "--force", "--hard", "install", "uninstall", "remove", "upgrade", "update",
                     "checkout", "reset", "clean", "commit", "merge", "rebase", "push", "pull"}
        if any(arg in forbidden or arg.startswith("--pre=") or arg.startswith("--post=") for arg in argv[1:]):
            raise ValueError("Команда содержит аргумент, несовместимый с режимом только чтения.")
        runtime = self._runtimes.get(conversation_id)
        if runtime is None:
            record = db.get_sandbox(conversation_id)
            if not record:
                raise SandboxError("Sandbox не найден; диагностическая Bash-сессия не запускается.")
            runtime = self._runtime_factory(port=record["host_port"], token=record["auth_token"])
            self._runtimes[conversation_id] = runtime
        if not runtime.is_alive():
            raise SandboxError("SWE-ReX не отвечает; диагностическая Bash-сессия не запускается.")
        runtime.ensure_session("diagnostic")
        # Use only the allow-listed basename, but let the container PATH locate
        # tools such as pip3/npm which may live outside /usr/bin.
        argv[0] = executable
        command_line = "exec /usr/bin/setpriv --no-new-privs --bounding-set=-all --inh-caps=-all --ambient-caps=-all --reset-env --reuid=localagent-diagnostic --regid=localagent-diagnostic --clear-groups -- " + shlex.join(argv)
        result = runtime.run(command_line, session="diagnostic", timeout=timeout)
        output = str(result.get("stdout") or "")
        return {
            "ok": result.get("exit_code") == 0 and not result.get("timed_out"),
            "stdout": output[-8000:],
            "stderr": str(result.get("stderr") or "")[-1500:],
            "exit_code": result.get("exit_code"),
            "timed_out": bool(result.get("timed_out")),
            "session": "diagnostic",
            "execution_user": "localagent-diagnostic",
            "workspace_access": "unprivileged; subject to container filesystem permissions",
        }

    def is_runtime_responsive(self, conversation_id: str) -> bool:
        """Check SWE-ReX availability without touching the persistent PTY."""
        runtime = self._runtimes.get(conversation_id)
        if runtime is None:
            record = db.get_sandbox(conversation_id)
            if not record:
                return False
            try:
                runtime = self._runtime_factory(port=record["host_port"], token=record["auth_token"])
                self._runtimes[conversation_id] = runtime
            except Exception:
                return False
        try:
            return bool(runtime.is_alive())
        except Exception:
            return False

    def verify_terminal_session(self, conversation_id: str) -> dict[str, Any]:
        """Prove the persistent Bash session can execute a harmless command."""
        try:
            # Never send a health probe into a PTY owned by a live operation.
            active = self.get_active_operation(conversation_id)
            if active:
                return {
                    "verified": False,
                    "error": f"Bash-сессия занята операцией {active['operation_id']} (state={active['state']}).",
                }
            runtime = self.get_runtime(conversation_id, check_proxy=False)
            marker = "__LOCALAGENT_RESUME_BASH_READY__"
            probe = runtime.run(
                f"unset SWEREX_AUTH_TOKEN; printf '{marker}'",
                session="main", timeout=10,
            )
            verified = (
                probe.get("exit_code") == 0
                and not probe.get("timed_out")
                and marker in str(probe.get("stdout") or "")
            )
            return {
                "verified": bool(verified),
                "exit_code": probe.get("exit_code"),
                "timed_out": bool(probe.get("timed_out")),
                "shell_recreated": bool(probe.get("shell_recovered") or probe.get("shell_state_reset")),
                "error": None if verified else str(
                    probe.get("stderr") or probe.get("failure_reason") or "Bash probe returned no confirmation"
                )[:500],
            }
        except Exception as exc:
            return {
                "verified": False,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }

    def restart_container(self, conversation_id: str, *, reason: str) -> dict[str, Any]:
        """Restart the existing container without removing it or its writable layer."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A non-empty restart reason is required")
        with self._lock_for(conversation_id):
            record = db.get_sandbox(conversation_id)
            if not record or not record.get("container_id"):
                raise SandboxError("Не найден существующий контейнер для перезапуска.")
            container_id = str(record["container_id"])
            self._podman("restart", container_id)
            stale = self._runtimes.pop(conversation_id, None)
            detach = getattr(stale, "detach", None)
            if callable(detach):
                detach()
            runtime = self._runtime_factory(port=record["host_port"], token=record["auth_token"])
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if runtime.is_alive():
                    break
                time.sleep(1)
            if not runtime.is_alive():
                raise SandboxError("Контейнер перезапущен, но SWE-ReX не подтвердил доступность.")
            runtime.ensure_session("main")
            probe = runtime.run("unset SWEREX_AUTH_TOKEN; printf '__LOCALAGENT_CONTAINER_RESTART_READY__'",
                                session="main", timeout=10)
            if probe.get("exit_code") != 0 or probe.get("timed_out") or "__LOCALAGENT_CONTAINER_RESTART_READY__" not in str(probe.get("stdout") or ""):
                raise SandboxError("Контейнер запущен, но проверка новой Bash-сессии не прошла.")
            with self._guard:
                self._active_operations.pop(conversation_id, None)
                self._new_sessions.discard(conversation_id)
                self._runtimes[conversation_id] = runtime
            db.update_sandbox_status(conversation_id, "running")
            return {"state": "restarted", "container_recreated": False, "container_id": container_id,
                    "workspace_preserved": True, "verified": True, "reason": reason.strip()}

    def recreate_container(self, conversation_id: str, *, reason: str) -> dict[str, Any]:
        """Replace an unusable sandbox container while preserving the host workspace."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A non-empty recreation reason is required")
        with self._lock_for(conversation_id):
            record = db.get_sandbox(conversation_id)
            if not record:
                raise SandboxError("Не найдена запись sandbox; токен для восстановления отсутствует.")
            old_id = str(record.get("container_id") or "")
            if old_id:
                self._podman("rm", "-f", old_id, check=False)
            stale = self._runtimes.pop(conversation_id, None)
            detach = getattr(stale, "detach", None)
            if callable(detach):
                detach()
            container_id, port = self._create_container(conversation_id, str(record["auth_token"]))
            db.upsert_sandbox(
                conversation_id, container_id, self._container_name(conversation_id),
                port, str(record["auth_token"]), "main", "running",
            )
            runtime = self._runtime_factory(port=port, token=str(record["auth_token"]))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not runtime.is_alive():
                time.sleep(1)
            if not runtime.is_alive():
                raise SandboxError("Новый контейнер создан, но SWE-ReX не отвечает.")
            runtime.ensure_session("main")
            probe = runtime.run(
                "unset SWEREX_AUTH_TOKEN; printf '__LOCALAGENT_CONTAINER_RECREATED_READY__'",
                session="main", timeout=10,
            )
            if (probe.get("exit_code") != 0 or probe.get("timed_out")
                    or "__LOCALAGENT_CONTAINER_RECREATED_READY__" not in str(probe.get("stdout") or "")):
                raise SandboxError("Новый контейнер запущен, но проверка Bash-сессии не прошла.")
            with self._guard:
                self._active_operations.pop(conversation_id, None)
                self._new_sessions.discard(conversation_id)
                self._runtimes[conversation_id] = runtime
            return {
                "state": "recreated", "container_recreated": True,
                "previous_container_id": old_id or None, "container_id": container_id,
                "workspace_preserved": True, "verified": True, "reason": reason.strip(),
                "message": "Контейнер пересоздан; общая рабочая папка сохранена, внутреннее состояние старого контейнера утрачено.",
            }

    def _recover_proxy_after_network_error(self, runtime: SandboxRuntime, result: dict[str, Any]) -> dict[str, Any] | None:
        if not getattr(settings, "sandbox_proxy_required", True) or not self._uses_default_runner:
            return None
        combined = (str(result.get("output") or "") + "\n" + str(result.get("failure_reason") or "")).lower()
        indicators = (
            "503 forwarding failure", "proxy error", "proxy connection", "socks5",
            "temporary failure in name resolution", "could not resolve", "name or service not known",
            "network is unreachable", "no route to host",
        )
        if not any(indicator in combined for indicator in indicators):
            return None
        try:
            # Recover infrastructure only; never replay the arbitrary user command.
            self._proxy_manager.ensure(settings.sandbox_network, force=True)
            status = self._probe_proxy_from_sandbox(runtime)
            return {
                "detected": True, "proxy_state": status.get("state"),
                "recovered": status.get("state") == "online",
                "failed_stage": status.get("failed_stage"),
                "error": status.get("probe_error"), "command_retried": False,
                "message": "Proxy health was rechecked; the failed command was not re-executed.",
            }
        except Exception as exc:
            return {
                "detected": True, "proxy_state": "offline", "recovered": False,
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "command_retried": False,
                "message": "Proxy recovery failed; the command was not re-executed.",
            }

    def get_operation(self, conversation_id: str, operation_id: str, *, cursor: int = 0,
                      max_output_chars: int = 65536, wait_timeout: float = 0.0) -> dict[str, Any]:
        runtime = self.get_runtime(conversation_id)
        with self._guard:
            recreated_containers = getattr(self, "_recreated_containers", set())
            container_recreated = conversation_id in recreated_containers
            if container_recreated:
                recreated_containers.discard(conversation_id)
        if container_recreated:
            result = {
                "state": "failed", "output": "",
                "failure_reason": "Контейнер отсутствовал и был автоматически пересоздан; результат прежней операции недоступен.",
                "exit_code": 1, "container_recreated": True,
            }
        else:
            getter = getattr(runtime, "get_operation", None)
            if not callable(getter):
                raise SandboxError("Подключённый SandboxRuntime не поддерживает polling фоновых операций.")
            try:
                result = getter(operation_id, session="main", cursor=cursor,
                                max_output_chars=max_output_chars, wait_timeout=wait_timeout)
            except Exception as exc:
                checker = getattr(runtime, "_has_exception_named", None)
                def has_error(*names):
                    if type(exc).__name__ in names:
                        return True
                    return bool(checker(exc, *names)) if callable(checker) else False

                missing_session = has_error("SessionDoesNotExistError")
                missing_operation = has_error("OperationDoesNotExistError", "OperationNotFoundError")
                if missing_session:
                    creator = getattr(runtime, "ensure_session", None)
                    created = bool(creator("main")) if callable(creator) else False
                    if not created:
                        raise SandboxError("Основная Bash-сессия отсутствует и не была пересоздана.") from exc
                    probe = runtime.run(
                        "unset SWEREX_AUTH_TOKEN; printf '__LOCALAGENT_MISSING_SESSION_RECOVERED__'",
                        session="main", timeout=10,
                    )
                    if (probe.get("exit_code") != 0 or probe.get("timed_out")
                            or "__LOCALAGENT_MISSING_SESSION_RECOVERED__" not in str(probe.get("stdout") or "")):
                        raise SandboxError("Bash-сессия пересоздана, но проверочная команда не прошла.") from exc
                    with self._guard:
                        self._new_sessions.add(conversation_id)
                    result = {
                        "state": "failed", "output": "",
                        "failure_reason": "Основная Bash-сессия отсутствовала и была автоматически пересоздана; результат прежней операции недоступен.",
                        "exit_code": 1, "shell_recreated": True,
                    }
                elif missing_operation:
                    with self._guard:
                        shell_was_recreated = conversation_id in self._new_sessions
                    if shell_was_recreated:
                        result = {
                            "state": "failed", "output": "",
                            "failure_reason": "После автоматического запуска контейнера или пересоздания Bash-сессии исходная операция больше недоступна; её результат не был восстановлен.",
                            "exit_code": 1, "shell_recreated": True,
                        }
                    else:
                        raise
                else:
                    raise
        result = dict(result)
        state = str(result.get("state") or "failed")
        output = str(result.get("output") or "")
        failure_reason = str(result.get("failure_reason") or "")
        started_at = result.get("started_at")
        finished_at = result.get("finished_at")
        duration = None
        if isinstance(started_at, (int, float)) and isinstance(finished_at, (int, float)):
            duration = max(0.0, round(finished_at - started_at, 3))
        result.update({
            "stdout": output,
            "stderr": failure_reason,
            "output_stream": "combined",
            "exit_code": result.get("exit_code"),
            "duration_seconds": duration,
            "timed_out": state == "timed_out",
            "interrupted": state == "interrupted",
            "waiting_input": state == "waiting_input",
            "stdout_truncated": bool(result.get("output_truncated", False)),
            "cursor": result.get("next_cursor", cursor),
        })
        if state in {"completed", "failed"} and result.get("exit_code") not in (None, 0):
            proxy_recovery = self._recover_proxy_after_network_error(runtime, result)
            if proxy_recovery is not None:
                result["proxy_recovery"] = proxy_recovery
        if state in {"completed", "interrupted", "timed_out", "failed"}:
            with self._guard:
                active = self._active_operations.get(conversation_id)
                if active and active.get("operation_id") == operation_id:
                    self._active_operations.pop(conversation_id, None)
        return result

    def _find_persisted_active_operation_details(self, conversation_id: str) -> dict[str, Any] | None:
        """Recover the latest live operation receipt, including its private cursor/transcript."""
        try:
            with db.connect() as con:
                rows = con.execute(
                    "SELECT o.result_json FROM tool_operations o JOIN tasks t ON t.id=o.task_id "
                    "WHERE t.conversation_id=? AND o.status='completed' "
                    "AND o.tool_name IN ('terminal','terminal_start','terminal_poll','reset_terminal_session') "
                    "ORDER BY o.rowid DESC LIMIT 100", (conversation_id,),
                ).fetchall()
            for row in rows:
                try:
                    payload = json.loads(row[0] or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("state") in {"running", "waiting_input"} and payload.get("operation_id"):
                    return payload
                if payload.get("state") in {"completed", "interrupted", "timed_out", "failed", "reset", "restarted", "idle"}:
                    return None
        except Exception as exc:
            logger.warning("Could not recover persisted terminal operation (%s)", type(exc).__name__)
        return None

    def _find_persisted_active_operation(self, conversation_id: str) -> str | None:
        """Backward-compatible ID-only accessor used by cancellation paths."""
        record = self._find_persisted_active_operation_details(conversation_id)
        return str(record["operation_id"]) if record and record.get("operation_id") else None

    def interrupt_operation(self, conversation_id: str, operation_id: str | None = None) -> dict[str, Any]:
        # Deliberately does not acquire the per-conversation lifecycle lock: Cancel
        # must be able to interrupt while the normal execution path is polling.
        with self._guard:
            active = self._active_operations.get(conversation_id)
            target = operation_id or (active or {}).get("operation_id")
        if not target:
            target = self._find_persisted_active_operation(conversation_id)
        if not target:
            return {"state": "idle", "interrupted": False, "message": "Нет активной терминальной операции."}
        runtime = self._runtimes.get(conversation_id)
        if runtime is None:
            record = db.get_sandbox(conversation_id)
            if not record:
                return {"state": "unknown", "interrupted": False, "message": "Sandbox не найден; удалённая операция не подтверждена как остановленная."}
            runtime = self._runtime_factory(port=record["host_port"], token=record["auth_token"])
        interrupter = getattr(runtime, "interrupt_operation", None)
        if not callable(interrupter):
            return {"state": "unsupported", "interrupted": False, "message": "Runtime не поддерживает удалённый interrupt."}
        result = interrupter(target, session="main")
        if result.get("state") in {"interrupted", "completed", "timed_out", "failed"}:
            with self._guard:
                active = self._active_operations.get(conversation_id)
                if active and active.get("operation_id") == target:
                    self._active_operations.pop(conversation_id, None)
        return result

    def reset_terminal_session(
        self,
        conversation_id: str,
        *,
        confirm_reset: bool,
        reason: str,
    ) -> dict[str, Any]:
        """Recreate only the persistent Bash session, keeping its container and files intact."""
        if confirm_reset is not True:
            raise ValueError("confirm_reset must be true to recreate the Bash session")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A non-empty reason is required for an audited terminal reset")
        if len(reason) > 1000:
            raise ValueError("reason exceeds 1000 characters")

        with self._lock_for(conversation_id):
            active = self.get_active_operation(conversation_id)
            # Resetting a local PTY must remain possible even when the container's
            # outbound proxy is unhealthy; no network access is needed here.
            runtime = self.get_runtime(conversation_id, check_proxy=False)

            interrupt_result = None
            interrupt_error = None
            if active:
                interrupter = getattr(runtime, "interrupt_operation", None)
                if callable(interrupter):
                    try:
                        interrupt_result = interrupter(
                            str(active.get("operation_id") or ""), session="main"
                        )
                    except Exception as exc:
                        # A stale SWE-ReX operation ID is exactly one reason to
                        # recreate the PTY. Closing the session is the fallback.
                        interrupt_error = type(exc).__name__

            closer = getattr(runtime, "close_session", None)
            creator = getattr(runtime, "ensure_session", None)
            if not callable(closer) or not callable(creator):
                raise SandboxError("Runtime не поддерживает безопасное пересоздание Bash-сессии.")

            closer("main")
            created = creator("main")
            if created is not True:
                raise SandboxError(
                    "SWE-ReX не подтвердил создание новой Bash-сессии после закрытия старой; "
                    "активное состояние сохранено."
                )

            probe = runtime.run(
                "unset SWEREX_AUTH_TOKEN; printf '__LOCALAGENT_RESET_READY__'",
                session="main", timeout=10,
            )
            if (probe.get("exit_code") != 0 or probe.get("timed_out")
                    or "__LOCALAGENT_RESET_READY__" not in str(probe.get("stdout") or "")):
                raise SandboxError(
                    "Новая Bash-сессия создана, но проверочная команда не прошла; "
                    "состояние операции не сброшено."
                )

            with self._guard:
                self._active_operations.pop(conversation_id, None)
                self._new_sessions.discard(conversation_id)
                self._runtimes[conversation_id] = runtime

            return {
                "state": "reset" if active else "idle",
                "shell_state": "recreated",
                "previous_operation_state": str(active.get("state") or "running") if active else None,
                "interrupt_result": interrupt_result,
                "interrupt_error": interrupt_error,
                "reason": reason.strip(),
                "container_recreated": False,
                "workspace_preserved": True,
                "verified": True,
                "message": (
                    "Bash-сессия пересоздана в прежнем контейнере. Файлы и workspace сохранены; "
                    "переменные, функции и процессы старой shell-сессии не сохраняются."
                ),
            }

    def send_operation_input(self, conversation_id: str, operation_id: str, value: str, sendline: bool = True) -> dict[str, Any]:
        runtime = self.get_runtime(conversation_id)
        sender = getattr(runtime, "send_input", None)
        if not callable(sender):
            raise SandboxError("Runtime не поддерживает интерактивный ввод.")
        return sender(operation_id, value, session="main", sendline=sendline)

    def resize_terminal(self, conversation_id: str, rows: int = 24, cols: int = 80) -> dict[str, Any]:
        if not 1 <= int(rows) <= 1000 or not 1 <= int(cols) <= 1000:
            raise ValueError("rows and cols must be between 1 and 1000")
        runtime = self.get_runtime(conversation_id)
        resize = getattr(runtime, "resize_terminal", None)
        if not callable(resize):
            raise SandboxError("Runtime не поддерживает изменение размера терминала.")
        resize(session="main", rows=int(rows), cols=int(cols))
        return {"resized": True, "rows": int(rows), "cols": int(cols)}

    def execute(self, conversation_id: str, command: str, timeout: int = 120,
                output_limit: int = 60_000) -> dict[str, Any]:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        if len(command) > 100_000:
            raise ValueError("command exceeds 100000 characters")
        if "\x00" in command:
            raise ValueError("command must not contain NUL bytes")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 600:
            raise ValueError("timeout must be between 1 and 600 seconds")

        # Parse only; never execute on the host. This catches unterminated quotes,
        # malformed heredocs and other shell syntax that could leave the persistent
        # interactive Bash process waiting for more input indefinitely.
        try:
            syntax = subprocess.run(
                ["bash", "-n", "-c", command], capture_output=True, text=True,
                timeout=5, check=False,
            )
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Bash syntax validation timed out", "exit_code": 124,
                    "timed_out": False, "syntax_error": True, "command": command}
        if syntax.returncode != 0:
            return {"stdout": syntax.stdout or "", "stderr": (syntax.stderr or "Invalid Bash syntax")[:4000],
                    "exit_code": 2, "timed_out": False, "syntax_error": True, "command": command}

        with self._lock_for(conversation_id):
            self.ensure_session_idle(conversation_id)
            runtime = self.get_runtime(conversation_id)
            self._ensure_network_proxy(runtime, command)
            try:
                result = runtime.run(command, session="main", timeout=timeout)
            except Exception as exc:
                db.update_sandbox_status(conversation_id, "error")
                detail = str(exc)[:400]
                record = db.get_sandbox(conversation_id) or {}
                secret = record.get("auth_token")
                if secret:
                    detail = detail.replace(secret, "[REDACTED]")
                raise SandboxError(f"Ошибка выполнения в Linux-среде: {type(exc).__name__}: {detail}") from exc
            if result.get("timed_out") and result.get("shell_recovered") is not True:
                db.update_sandbox_status(conversation_id, "error")
                stale = self._runtimes.pop(conversation_id, None)
                detach = getattr(stale, "detach", None)
                if callable(detach):
                    detach()
            else:
                db.update_sandbox_status(conversation_id, "running")
            # Never allow the SWE-ReX auth secret to appear in model-visible output.
            record = db.get_sandbox(conversation_id) or {}
            secret = record.get("auth_token")
            if secret:
                for key in ("stdout", "stderr"):
                    result[key] = str(result.get(key, "")).replace(secret, "[REDACTED]")
            for key in ("stdout", "stderr"):
                value = str(result.get(key, ""))
                if len(value) > output_limit:
                    result[key] = value[:output_limit] + "\n[output truncated]"
                    result[key + "_truncated"] = True
            result["command"] = command
            return result

    def download_file(self, conversation_id: str, source_path: str, target_path: str | Path, *, max_bytes: int | None = None) -> Path:
        """Copy a readable regular container file through bounded base64 chunks."""
        import base64

        if not isinstance(source_path, str) or not source_path.startswith("/") or "\x00" in source_path:
            raise ValueError("source_path must be an absolute container path")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        quoted = shlex.quote(source_path)
        chunk_bytes = 1024 * 1024
        with self._lock_for(conversation_id):
            self.ensure_session_idle(conversation_id)
            runtime = self.get_runtime(conversation_id)
            probe = runtime.run(
                "test -f " + quoted + " && test -r " + quoted + " && stat -c '%s' -- " + quoted,
                session="main", timeout=20,
            )
            if probe.get("exit_code") != 0:
                raise FileNotFoundError("Container file is missing, unreadable, or not a regular file")
            try:
                expected_size = int(str(probe.get("stdout", "")).strip().splitlines()[-1])
            except (ValueError, IndexError) as exc:
                raise SandboxError("Could not determine container file size") from exc
            if expected_size < 0:
                raise ValueError("Container file has an invalid negative size")
            if max_bytes is not None and expected_size > max_bytes:
                actual_mib = expected_size / (1024 * 1024)
                limit_mib = max_bytes / (1024 * 1024)
                raise ValueError(
                    f"Container file exceeds the allowed size limit: {expected_size:,} bytes "
                    f"({actual_mib:.2f} MiB); limit is {max_bytes:,} bytes "
                    f"({limit_mib:.2f} MiB). Split or reduce the file before loading it."
                )
            if target.exists() or target.is_symlink():
                raise FileExistsError("Download target already exists")
            written = 0
            created = False
            try:
                with target.open("xb") as stream:
                    created = True
                    while written < expected_size:
                        amount = min(chunk_bytes, expected_size - written)
                        command = (
                            f"dd if={quoted} iflag=skip_bytes,count_bytes skip={written} count={amount} status=none "
                            "| base64 -w0"
                        )
                        chunk = runtime.run(command, session="main", timeout=45)
                        if chunk.get("exit_code") != 0:
                            raise SandboxError("Could not read a container file chunk")
                        try:
                            data = base64.b64decode(str(chunk.get("stdout", "")), validate=True)
                        except Exception as exc:
                            raise SandboxError("Container returned an invalid file chunk") from exc
                        if len(data) != amount:
                            raise SandboxError("Container file chunk was incomplete")
                        stream.write(data)
                        written += len(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                if written != expected_size or target.stat().st_size != expected_size:
                    raise SandboxError("Container file download failed size verification")
                remote_hash = runtime.run(
                    "sha256sum -- " + quoted, session="main", timeout=30,
                )
                if remote_hash.get("exit_code") != 0:
                    raise SandboxError("Could not verify container file checksum")
                expected_hash = str(remote_hash.get("stdout", "")).strip().split()[0]
                digest = hashlib.sha256()
                with target.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                actual_hash = digest.hexdigest()
                if not expected_hash or actual_hash != expected_hash:
                    raise SandboxError("Container file download checksum mismatch")
            except Exception:
                if created:
                    target.unlink(missing_ok=True)
                raise
        return target

    def upload_file(self, conversation_id: str, source: str | Path, display_name: str) -> str:
        """Copy an attachment into the container without mounting host paths."""
        source = Path(source)
        if any(parent.is_symlink() for parent in (source, *source.parents)):
            raise SandboxError("Attachment source cannot traverse symbolic links.")
        source = source.resolve(strict=True)
        allowed_roots = [
            (settings.database_path.parent / "attachments").resolve(),
            (settings.workspace_path / ".attachments").resolve(),
        ]
        if not any(source == root or root in source.parents for root in allowed_roots):
            raise SandboxError("Attachment source is outside LocalAgent's managed attachment directories.")
        if not source.is_file():
            raise SandboxError("Attachment source is not a regular file.")
        if source.stat().st_size > 12 * 1024 * 1024:
            raise SandboxError("Attachment exceeds the 12 MB transfer limit.")
        safe_name = "".join(ch for ch in Path(display_name).name if ch.isalnum() or ch in "._-()")[:100] or "attachment"
        digest = hashlib.sha256(f"{source}:{source.stat().st_size}:{source.stat().st_mtime_ns}".encode()).hexdigest()[:16]
        destination = f"/workspace/uploads/{digest}_{safe_name}"
        quoted = shlex.quote(destination)
        with self._lock_for(conversation_id):
            if self.get_active_operation(conversation_id):
                # Task attachments are uploaded before execution starts. If a
                # task resumes while its own terminal command is still active,
                # reuse the deterministic destination without sending commands
                # into the occupied PTY.
                return destination
            self.ensure_session_idle(conversation_id)
            runtime = self.get_runtime(conversation_id)
            exists = runtime.run(f"test -f {quoted} && test ! -L {quoted}", session="main", timeout=10)
            if exists["exit_code"] == 0:
                return destination
            mkdir_result = runtime.run("mkdir -p /workspace/uploads", session="main", timeout=10)
            if mkdir_result["exit_code"] != 0:
                raise SandboxError("Не удалось создать каталог /workspace/uploads в контейнере.")
            upload = getattr(runtime, "upload_file", None)
            if callable(upload):
                upload(str(source), destination)
            else:
                runtime.run(f": > {quoted}", session="main", timeout=10)
                with source.open("rb") as handle:
                    while True:
                        chunk = handle.read(256 * 1024)
                        if not chunk:
                            break
                        encoded = base64.b64encode(chunk).decode("ascii")
                        command = f"printf '%s' {shlex.quote(encoded)} | base64 -d >> {quoted}"
                        result = runtime.run(command, session="main", timeout=30)
                        if result["exit_code"] != 0:
                            runtime.run(f"rm -f -- {quoted}", session="main", timeout=10)
                            raise SandboxError("Не удалось передать вложение в Linux-среду.")
            verify = runtime.run(f"test $(stat -c %s -- {quoted}) -eq {source.stat().st_size}", session="main", timeout=10)
            if verify["exit_code"] != 0:
                runtime.run(f"rm -f -- {quoted}", session="main", timeout=10)
                raise SandboxError("Проверка размера переданного вложения не пройдена.")
        return destination

    def delete(self, conversation_id: str) -> None:
        with self._lock_for(conversation_id):
            record = db.get_sandbox(conversation_id)
            runtime = self._runtimes.get(conversation_id)
            if not record and runtime is None:
                try:
                    container_id = self._find_container(conversation_id)
                except SandboxError as exc:
                    logger.warning("Could not inspect possible orphan sandbox for conversation %s (%s)",
                                   conversation_id, type(exc).__name__)
                    return
                if container_id is None:
                    return
            else:
                container_id = self._find_container(conversation_id)
            if record and container_id and container_id != record.get("container_id"):
                raise SandboxError("ID контейнера по label не совпадает с записью БД; удаление отменено.")
            if runtime:
                self._runtimes.pop(conversation_id, None)
                try:
                    runtime.close_session("main")
                except Exception as exc:
                    # Container removal is the authoritative final cleanup.
                    logger.warning("Could not close main Bash session for conversation %s (%s)",
                                   conversation_id, type(exc).__name__)
                detach = getattr(runtime, "detach", None)
                if callable(detach):
                    detach()
            if container_id:
                self._podman("rm", "-f", container_id)
            db.delete_sandbox(conversation_id)
            if record and container_id is None:
                # Idempotent deletion: a missing container is already removed.
                return

    def restore_all(self) -> dict[str, str]:
        """Reconnect to persisted conversation containers without creating new ones."""
        if not settings.sandbox_enabled:
            return {record["conversation_id"]: "disabled" for record in db.list_sandboxes()}
        outcomes: dict[str, str] = {}
        for record in db.list_sandboxes():
            conversation_id = record["conversation_id"]
            try:
                self.get_runtime(conversation_id)
                outcomes[conversation_id] = "restored"
            except Exception as exc:
                db.update_sandbox_status(conversation_id, "error")
                logger.warning("Sandbox restore failed for conversation %s (%s)",
                               conversation_id, type(exc).__name__)
                outcomes[conversation_id] = f"unavailable: {type(exc).__name__}"
        return outcomes

    def detach_all(self) -> None:
        """Drop local clients on shutdown; do not close remote sessions/containers."""
        with self._guard:
            runtimes, self._runtimes = self._runtimes, {}
        for runtime in runtimes.values():
            detach = getattr(runtime, "detach", None)
            if callable(detach):
                detach()


sandbox_manager = SandboxManager()
