import socket
from types import SimpleNamespace

import pytest

from agent import tools


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/", "http://localhost/",
    "http://anything.localhost/", "http://printer.local/", "http://service.internal/",
    "http://host.lan/", "http://example.test/", "http://user:pass@example.com/",
    "http://example.com:8080/", "https://example.com:8443/",
    "http://example.com\\@127.0.0.1/", "http://example.com/has space", "http://example.com:bad/",
])
def test_public_url_rejects_unsafe_or_unsupported_urls(url):
    with pytest.raises(ValueError):
        tools._validate_public_url(url)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://10.0.0.2/", "http://169.254.169.254/latest/meta-data/",
    "http://[::1]/", "http://[::ffff:127.0.0.1]/",
])
def test_public_url_rejects_non_global_ip_literals(url):
    with pytest.raises(ValueError):
        tools._validate_public_url(url)


def test_public_url_rejects_hostname_resolving_to_private_address(monkeypatch):
    def fake_getaddrinfo(host, port, type):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.10", port))]
    monkeypatch.setattr(tools.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ValueError, match="non-public"):
        tools._validate_public_url("https://example.com/")


def test_public_url_accepts_public_hostname(monkeypatch):
    def fake_getaddrinfo(host, port, type):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
    monkeypatch.setattr(tools.socket, "getaddrinfo", fake_getaddrinfo)
    assert tools._validate_public_url("https://example.com/path") == "https://example.com/path"


def test_redirect_to_private_host_is_rejected_before_second_request(monkeypatch):
    calls = []
    class FakeResponse:
        is_redirect = True
        is_permanent_redirect = False
        headers = {"Location": "http://127.0.0.1/admin"}
        def close(self): pass
    class FakeSession:
        trust_env = True
        def get(self, url, **kwargs):
            calls.append(url)
            return FakeResponse()
        def close(self): pass
    def fake_validate(url):
        if "127.0.0.1" in url:
            raise ValueError("Private, local, or non-public IP addresses are not allowed")
        return url
    monkeypatch.setattr(tools, "_validate_public_url", fake_validate)
    monkeypatch.setattr(tools.requests, "Session", FakeSession)
    with pytest.raises(ValueError, match="Private"):
        tools.open_web_page("https://public.example/")
    assert calls == ["https://public.example/"]


def test_search_result_is_marked_untrusted(monkeypatch):
    class FakeDDGS:
        def __init__(self, **kwargs): pass
        def text(self, *args, **kwargs):
            return iter([{"title": "Result", "href": "https://example.com/", "body": "Snippet"}])
    monkeypatch.setattr(tools, "DDGS", FakeDDGS)
    result = tools.web_search("example", max_results=1)
    assert len(result["results"]) == 1
    assert "untrusted data" in result["note"]


def test_api_version_matches_version_module():
    from agent import web
    from agent.version import __version__
    assert web.version() == {"name": "LocalAgent", "version": __version__}
    assert web.app.version == __version__
