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
        self._new_sessions: set[str] = set()

    def _lock_for(self, conversation_id: str):
        with self._guard:
            return self._locks.setdefault(conversation_id, threading.RLock())

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

    def get_runtime(self, conversation_id: str) -> SandboxRuntime:
        if not conversation_id:
            raise ValueError("conversation_id is required")
        if not settings.sandbox_enabled:
            raise SandboxError("Новая sandbox отключена настройкой SANDBOX_ENABLED.")
        # Only the real host-backed manager enforces the shared proxy lifecycle;
        # injected command runners remain deterministic in unit tests.
        if getattr(settings, "sandbox_proxy_required", True) and self._uses_default_runner:
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
            if record:
                container_id = record["container_id"]
                if found_id and found_id != container_id:
                    raise SandboxError("ID контейнера в БД не совпадает с контейнером по label.")
                if not found_id:
                    raise SandboxError("Контейнер диалога отсутствует. Автоматическое пересоздание запрещено, чтобы не потерять данные.")
                inspect_data = self._inspect(found_id)
                if not (inspect_data.get("State") or {}).get("Running"):
                    self._podman("start", found_id)
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
            session_created = runtime.ensure_session("main")
            if session_created:
                with self._guard:
                    self._new_sessions.add(conversation_id)
            # The auth token is needed by the SWE-ReX server only. Remove it
            # from the persistent interactive shell before the model can use it.
            cleared = runtime.run("unset SWEREX_AUTH_TOKEN", session="main", timeout=10)
            if cleared.get("timed_out") and cleared.get("shell_recovered") is True:
                # A stale interactive shell may have survived a server/client
                # reconnect. The adapter reset it; retry this idempotent cleanup.
                cleared = runtime.run("unset SWEREX_AUTH_TOKEN", session="main", timeout=10)
            if cleared.get("exit_code") != 0 or cleared.get("timed_out"):
                raise SandboxError("Не удалось очистить переменную авторизации или проверить Bash-сессию.")
            if getattr(settings, "sandbox_proxy_required", True) and self._uses_default_runner:
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
            return {"state": "failed", "exit_code": 2, "stdout": syntax.stdout or "",
                    "stderr": (syntax.stderr or "Invalid Bash syntax")[:4000], "syntax_error": True}
        runtime = self.get_runtime(conversation_id)
        self._ensure_network_proxy(runtime, command)
        starter = getattr(runtime, "start_operation", None)
        if not callable(starter):
            raise SandboxError("Подключённый SandboxRuntime не поддерживает фоновые операции SWE-ReX.")
        result = starter(command, session="main", timeout=timeout, idempotency_key=idempotency_key)
        operation_id = str(result.get("operation_id") or "")
        if not operation_id:
            raise SandboxError("SWE-ReX не вернул operation_id для фоновой операции.")
        with self._guard:
            self._active_operations[conversation_id] = {"operation_id": operation_id, "session": "main"}
        return {"operation_id": operation_id, "session": "main", "state": result.get("state", "running"),
                "created_at": result.get("created_at"), "command": command, "background": True}

    @staticmethod
    def _terminal_control_text(output: str, state: str, elapsed: float, idle_seconds: float,
                               *, waiting_input: bool = False) -> str:
        text = output or ""
        if waiting_input:
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
                    "started_monotonic": recovered_now, "last_output_monotonic": recovered_now,
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
                "output": "", "command": command, "started_monotonic": started_now,
                "last_output_monotonic": started_now,
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
                self.send_operation_input(conversation_id, operation_id, command, sendline=True)

        command_started_at = float(active.get("started_monotonic", time.monotonic()))
        call_started_at = time.monotonic()
        idle_limit = max(0.1, float(getattr(settings, "terminal_idle_timeout_seconds", 10)))
        max_wait = max(idle_limit, float(getattr(settings, "terminal_max_wait_seconds", 300)))
        last_output_at = float(active.get("last_output_monotonic", call_started_at))
        state = "running"
        latest: dict[str, Any] = {}
        while True:
            now = time.monotonic()
            elapsed = now - command_started_at
            if now - call_started_at >= max_wait:
                break
            latest = self.get_operation(
                conversation_id, operation_id,
                cursor=int(active.get("cursor", 0)),
                max_output_chars=1_048_576,
                wait_timeout=min(1.0, max(0.1, max_wait - elapsed)),
            )
            chunk = str(latest.get("stdout") or latest.get("output") or "")
            if chunk:
                active["output"] = str(active.get("output") or "") + chunk
                last_output_at = time.monotonic()
                active["last_output_monotonic"] = last_output_at
            active["output_truncated"] = bool(active.get("output_truncated")) or bool(
                latest.get("stdout_truncated") or latest.get("output_truncated")
            )
            active["cursor"] = int(latest.get("next_cursor", latest.get("cursor", active.get("cursor", 0))) or 0)
            state = str(latest.get("state") or "failed")
            if state in {"completed", "interrupted", "timed_out", "failed"}:
                break
            if state == "waiting_input":
                break
            now = time.monotonic()
            if now - last_output_at >= idle_limit:
                break

        elapsed = max(0.0, time.monotonic() - command_started_at)
        output = str(active.get("output") or "")
        waiting = state == "waiting_input"
        still_running = state not in {"completed", "interrupted", "timed_out", "failed"}
        if state == "failed" and latest.get("stderr"):
            diagnostic = str(latest.get("stderr"))
            if diagnostic and diagnostic not in output:
                output = (output.rstrip() + "\n" + diagnostic).lstrip("\n")
        if state == "timed_out":
            diagnostic = str(latest.get("stderr") or "Команда превысила установленный жёсткий timeout.")
            if diagnostic and diagnostic not in output:
                output = (output.rstrip() + "\n" + diagnostic).lstrip("\n")

        shell_state = "busy" if still_running else "unknown"
        if not still_running:
            try:
                # A completed operation must leave the persistent Bash session
                # usable for the next command. create_session is also the
                # supported existence probe: SessionExistsError means it is alive.
                created_now = bool(self.get_runtime(conversation_id).ensure_session("main"))
                with self._guard:
                    created_during_recovery = conversation_id in self._new_sessions
                    self._new_sessions.discard(conversation_id)
                recreated = created_now or (recovered_operation and created_during_recovery)
                shell_state = "recreated" if recreated else "available"
                if recreated:
                    output = (output.rstrip() + "\n" +
                              "[Bash-сессия завершилась вместе с процессом и была создана заново; "
                              "переменные и функции предыдущей сессии не сохранены.]").lstrip("\n")
            except Exception as session_exc:
                shell_state = "unavailable"
                output = (output.rstrip() + "\n" +
                          "[Не удалось подтвердить/восстановить Bash-сессию: " +
                          f"{type(session_exc).__name__}: {str(session_exc)[:300]}]").lstrip("\n")
        text = self._terminal_control_text(
            output, "running" if still_running else state, elapsed,
            max(0.0, time.monotonic() - last_output_at),
            waiting_input=waiting,
        )
        if active.get("output_truncated"):
            text = (text.rstrip() + "\n" +
                    "[Внимание: SWE-ReX сообщил об ограничении объёма одного фрагмента вывода; "
                    "экранный вывод может быть неполным.]").lstrip("\n")
        return {
            "_terminal_text": text,
            "_terminal_output": output,
            "_terminal_command": active.get("command"),
            "state": state if not still_running else ("waiting_input" if waiting else "running"),
            "operation_id": operation_id if still_running else None,
            "cursor": active.get("cursor", 0),
            "exit_code": latest.get("exit_code"),
            "timed_out": latest.get("timed_out", state == "timed_out"),
            "interrupted": latest.get("interrupted", state == "interrupted"),
            "duration_seconds": latest.get("duration_seconds", elapsed),
            "output_truncated": bool(active.get("output_truncated")),
            "shell_state": shell_state,
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
        getter = getattr(runtime, "get_operation", None)
        if not callable(getter):
            raise SandboxError("Подключённый SandboxRuntime не поддерживает polling фоновых операций.")
        result = getter(operation_id, session="main", cursor=cursor,
                        max_output_chars=max_output_chars, wait_timeout=wait_timeout)
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
                    "AND o.tool_name IN ('terminal','terminal_start','terminal_poll') "
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
                if payload.get("state") in {"completed", "interrupted", "timed_out", "failed"}:
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
