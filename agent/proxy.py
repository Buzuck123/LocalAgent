"""Lifecycle and bounded health metadata for the shared Xray proxy container.

The proxy is shared infrastructure: this manager may start/connect it, but never
stops or removes it when a conversation sandbox is deleted.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

PROXY_CONTAINER = "localagent-xray-proxy"


class ProxyUnavailable(RuntimeError):
    def __init__(self, message: str, status: dict[str, Any] | None = None):
        super().__init__(message)
        self.status = status or {"state": "offline", "error": message}


class XrayProxyManager:
    def __init__(self, command_runner: Callable[..., Any], podman_path: str = "podman", container: str = PROXY_CONTAINER):
        self._run = command_runner
        self._podman = podman_path
        self._container = container
        self._lock = threading.Lock()
        self._last_status: dict[str, Any] = {"state": "unknown", "checked_at": None}
        self._last_check = 0.0

    def _cmd(self, *args: str, check: bool = True):
        return self._run([self._podman, *args], check=check)

    @staticmethod
    def _inspect_payload(stdout: str) -> dict[str, Any]:
        try:
            value = json.loads(stdout)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value[0]
            if isinstance(value, dict):
                return value
        except (TypeError, ValueError):
            pass
        raise ProxyUnavailable("Podman вернул некорректный inspect общего Xray-прокси.")

    def ensure(self, network: str, *, force: bool = False) -> dict[str, Any]:
        """Ensure the known proxy container is running and attached to `network`.

        Calls are single-flight and cached briefly to avoid concurrent sandbox
        creation causing repeated starts/inspect operations.
        """
        if not network or not isinstance(network, str):
            raise ProxyUnavailable("Не задана Podman-сеть для Xray-прокси.")
        with self._lock:
            if not force and self._last_check and time.monotonic() - self._last_check < 10:
                if self._last_status.get("state") in {"ready", "online", "degraded"}:
                    return dict(self._last_status)
            try:
                inspected = self._cmd("inspect", self._container, check=False)
                if inspected.returncode != 0:
                    raise ProxyUnavailable(
                        f"Общий контейнер {self._container} не найден; конфигурация Xray не создаётся автоматически."
                    )
                info = self._inspect_payload(inspected.stdout)
                running = bool((info.get("State") or {}).get("Running"))
                started = False
                if not running:
                    self._cmd("start", self._container)
                    started = True
                    info = self._inspect_payload(self._cmd("inspect", self._container).stdout)
                    running = bool((info.get("State") or {}).get("Running"))
                if not running:
                    raise ProxyUnavailable(f"Не удалось запустить общий контейнер {self._container}.")

                networks = (info.get("NetworkSettings") or {}).get("Networks") or {}
                connected = network in networks
                if not connected:
                    self._cmd("network", "inspect", network)
                    self._cmd("network", "connect", "--alias", self._container, network, self._container)
                    refreshed = self._inspect_payload(self._cmd("inspect", self._container).stdout)
                    networks = (refreshed.get("NetworkSettings") or {}).get("Networks") or {}
                    connected = network in networks
                if not connected:
                    raise ProxyUnavailable(f"Xray запущен, но не подключён к сети {network}.")
                status = {
                    "state": "ready", "container": self._container, "running": True,
                    "network": network, "network_connected": True, "started_by_localagent": started,
                    "socks_reachable": None, "external_internet": None,
                    "checked_at": time.time(),
                    "last_success_at": self._last_status.get("last_success_at"),
                    "note": "Проверены lifecycle и подключение к сети; SOCKS5 и внешний egress требуют проверки из sandbox.",
                }
                self._last_status = status
                self._last_check = time.monotonic()
                return dict(status)
            except ProxyUnavailable as exc:
                status = dict(exc.status)
                status.update({"state": "offline", "checked_at": time.time(),
                               "failed_stage": "container_lifecycle",
                               "last_success_at": self._last_status.get("last_success_at")})
                self._last_status = status
                self._last_check = time.monotonic()
                raise
            except Exception as exc:
                status = {"state": "offline", "container": self._container,
                          "network": network, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                          "failed_stage": "container_lifecycle", "last_success_at": self._last_status.get("last_success_at"),
                          "checked_at": time.time()}
                self._last_status = status
                self._last_check = time.monotonic()
                raise ProxyUnavailable("Не удалось обеспечить запуск общего Xray-прокси.", status) from exc

    def record_runtime_probe(self, result: dict[str, Any]) -> dict[str, Any]:
        """Record a probe executed inside the sandbox network namespace."""
        with self._lock:
            status = dict(self._last_status)
            status["socks_reachable"] = result.get("socks_reachable") is True
            status["dns_resolved"] = result.get("dns_resolved") is True
            status["privoxy_egress"] = result.get("privoxy_egress") is True
            status["external_internet"] = result.get("external_internet") is True
            status["probe_error"] = str(result.get("error") or "")[:500] or None
            status["last_error"] = status["probe_error"]
            status["failed_stage"] = result.get("failed_stage")
            status["probed_at"] = time.time()
            if result.get("external_internet") is True and result.get("socks_reachable") is True:
                status["last_success_at"] = status["probed_at"]
            status["state"] = "online" if (
                status.get("running") is True and status["network_connected"]
                and status["dns_resolved"] and status["socks_reachable"]
                and status["privoxy_egress"] and status["external_internet"]
            ) else "degraded"
            self._last_status = status
            return dict(status)

    def status(self) -> dict[str, Any]:
        return dict(self._last_status)
