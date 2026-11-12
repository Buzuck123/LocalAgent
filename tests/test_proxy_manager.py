import json
import subprocess
import threading
import time

import pytest

from agent.proxy import ProxyUnavailable, XrayProxyManager


def _inspect(running=True, networks=None):
    return json.dumps([{
        "State": {"Running": running},
        "NetworkSettings": {"Networks": networks or {}},
    }])


class FakePodman:
    def __init__(self, *, running=False, networks=None, missing=False):
        self.running = running
        self.networks = dict(networks or {})
        self.missing = missing
        self.calls = []

    def __call__(self, args, check=True):
        self.calls.append(args[1:])
        cmd = args[1:]
        if cmd[:2] == ["inspect", "localagent-xray-proxy"]:
            if self.missing:
                return subprocess.CompletedProcess(args, 125, "", "not found")
            return subprocess.CompletedProcess(args, 0, _inspect(self.running, self.networks), "")
        if cmd[:2] == ["start", "localagent-xray-proxy"]:
            self.running = True
            return subprocess.CompletedProcess(args, 0, "localagent-xray-proxy", "")
        if cmd[:2] == ["network", "inspect"]:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if cmd[:2] == ["network", "connect"]:
            self.networks[cmd[-2]] = {"Aliases": ["localagent-xray-proxy"]}
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(f"Unexpected Podman command: {cmd}")


def test_proxy_starts_existing_stopped_container_and_connects_network():
    podman = FakePodman(running=False)
    manager = XrayProxyManager(podman)
    result = manager.ensure("localagent-sandbox-internal-test")
    assert result["state"] == "ready"
    assert result["started_by_localagent"] is True
    assert result["network_connected"] is True
    assert ["start", "localagent-xray-proxy"] in podman.calls
    assert any(call[:2] == ["network", "connect"] for call in podman.calls)


def test_proxy_already_running_and_attached_is_reused():
    podman = FakePodman(running=True, networks={"sandbox-net": {}})
    manager = XrayProxyManager(podman)
    first = manager.ensure("sandbox-net")
    call_count = len(podman.calls)
    second = manager.ensure("sandbox-net")
    assert first["state"] == second["state"] == "ready"
    assert len(podman.calls) == call_count


def test_missing_proxy_fails_closed_without_creating_or_replacing_it():
    podman = FakePodman(missing=True)
    manager = XrayProxyManager(podman)
    with pytest.raises(ProxyUnavailable, match="не найден"):
        manager.ensure("sandbox-net")
    assert not any(call[:1] == ["run"] for call in podman.calls)


def test_proxy_status_does_not_claim_socks_or_egress_were_verified():
    manager = XrayProxyManager(FakePodman(running=True, networks={"sandbox-net": {}}))
    status = manager.ensure("sandbox-net")
    assert status["socks_reachable"] is None
    assert status["external_internet"] is None


def test_runtime_probe_distinguishes_online_from_degraded():
    manager = XrayProxyManager(FakePodman(running=True, networks={"sandbox-net": {}}))
    manager.ensure("sandbox-net")
    degraded = manager.record_runtime_probe({"dns_resolved": True, "socks_reachable": True,
        "privoxy_egress": False, "external_internet": False, "error": "proxy failed"})
    assert degraded["state"] == "degraded"
    assert degraded["probe_error"] == "proxy failed"
    online = manager.record_runtime_probe({"dns_resolved": True, "socks_reachable": True,
        "privoxy_egress": True, "external_internet": True})
    assert online["state"] == "online"


def test_network_command_classifier_covers_common_package_and_http_tools():
    from agent.sandbox_manager import SandboxManager
    assert SandboxManager._command_requires_network("apt-get update")
    assert SandboxManager._command_requires_network("python3 -c 'import requests; requests.get(\"https://example.com\")'")
    assert SandboxManager._command_requires_network("curl https://example.com")
    assert not SandboxManager._command_requires_network("pytest -q")


def test_network_command_health_check_retries_once_after_degraded_probe():
    from agent.sandbox_manager import SandboxManager
    manager = SandboxManager(command_runner=lambda *a, **k: None)
    manager._uses_default_runner = True
    probes = iter([
        {"state": "degraded", "probe_error": "temporary DNS failure"},
        {"state": "online"},
    ])
    manager._probe_proxy_from_sandbox = lambda runtime: next(probes)
    ensured = []
    manager._proxy_manager.ensure = lambda network, force=False: ensured.append((network, force)) or {"state": "ready"}
    assert manager._ensure_network_proxy(object(), "pip install example") == {"state": "online"}
    from agent.config import settings
    assert ensured == [(settings.sandbox_network, True)]


def test_network_command_is_blocked_if_bounded_proxy_retry_stays_degraded():
    from agent.sandbox_manager import SandboxManager, SandboxError
    manager = SandboxManager(command_runner=lambda *a, **k: None)
    manager._uses_default_runner = True
    manager._probe_proxy_from_sandbox = lambda runtime: {"state": "degraded", "probe_error": "SOCKS handshake failed"}
    manager._proxy_manager.ensure = lambda network, force=False: {"state": "ready"}
    with pytest.raises(SandboxError, match="Интернет песочницы недоступен"):
        manager._ensure_network_proxy(object(), "apt-get update")


def test_proxy_status_retains_last_success_and_failed_stage():
    manager = XrayProxyManager(FakePodman(running=True, networks={"sandbox-net": {}}))
    manager.ensure("sandbox-net")
    online = manager.record_runtime_probe({"dns_resolved": True, "socks_reachable": True,
        "privoxy_egress": True, "external_internet": True})
    last_success = online["last_success_at"]
    degraded = manager.record_runtime_probe({"dns_resolved": True, "socks_reachable": False,
        "privoxy_egress": False, "external_internet": False, "failed_stage": "socks5_handshake",
        "error": "handshake failed"})
    assert degraded["last_success_at"] == last_success
    assert degraded["failed_stage"] == "socks5_handshake"
    assert degraded["last_error"] == "handshake failed"


def test_proxy_start_failure_is_reported_without_restart_loop():
    class FailedStart(FakePodman):
        def __call__(self, args, check=True):
            if args[1:2] == ["start"]:
                self.calls.append(args[1:])
                raise subprocess.CalledProcessError(125, args, stderr="start failed")
            return super().__call__(args, check=check)

    podman = FailedStart(running=False)
    manager = XrayProxyManager(podman)
    with pytest.raises(ProxyUnavailable):
        manager.ensure("sandbox-net")
    assert len([call for call in podman.calls if call[:1] == ["start"]]) == 1


@pytest.mark.parametrize("probe,stage", [
    ({"dns_resolved": False, "socks_reachable": False, "privoxy_egress": False,
      "external_internet": False, "failed_stage": "dns"}, "dns"),
    ({"dns_resolved": True, "socks_reachable": False, "privoxy_egress": False,
      "external_internet": False, "failed_stage": "socks5_tcp"}, "socks5_tcp"),
    ({"dns_resolved": True, "socks_reachable": False, "privoxy_egress": False,
      "external_internet": False, "failed_stage": "socks5_handshake"}, "socks5_handshake"),
    ({"dns_resolved": True, "socks_reachable": True, "privoxy_egress": False,
      "external_internet": False, "failed_stage": "privoxy_egress"}, "privoxy_egress"),
])
def test_proxy_probe_failures_keep_the_exact_failed_stage(probe, stage):
    manager = XrayProxyManager(FakePodman(running=True, networks={"sandbox-net": {}}))
    manager.ensure("sandbox-net")
    status = manager.record_runtime_probe(probe)
    assert status["state"] == "degraded"
    assert status["failed_stage"] == stage
