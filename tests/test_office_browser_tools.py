import pytest

from agent import tools


def test_public_url_validator_blocks_local_and_nonstandard_ports(monkeypatch):
    monkeypatch.setattr(
        tools.socket,
        "getaddrinfo",
        lambda *a, **k: [(None, None, None, None, ("93.184.216.34", 0))],
    )
    with pytest.raises(ValueError):
        tools._validate_public_url("http://127.0.0.1/")
    with pytest.raises(ValueError):
        tools._validate_public_url("https://example.com:8443/")
    assert tools._validate_public_url("https://example.com/") == "https://example.com/"


def test_current_browser_and_computer_tools_are_exposed():
    names = {item["function"]["name"] for item in tools.TOOL_DEFINITIONS}
    assert {"terminal", "publish_file", "load_file_for_model", "web_search", "open_browser_page", "get_tool_result"} <= names


def test_retired_office_and_sqlite_tools_are_not_exposed_or_dispatchable():
    retired = {
        "read_excel", "write_excel", "read_word", "write_word", "excel_aggregate",
        "sqlite_create_database", "sqlite_read", "sqlite_query", "sqlite_execute",
        "sqlite_import_excel", "sqlite_export_excel",
    }
    names = {item["function"]["name"] for item in tools.TOOL_DEFINITIONS}
    assert not retired & names
    for name in retired:
        with pytest.raises(ValueError, match="Unknown tool"):
            tools.dispatch_tool(name, {})
