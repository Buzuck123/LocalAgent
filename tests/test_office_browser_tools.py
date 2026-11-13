import pytest

from agent import tools


def test_current_computer_tools_are_exposed_without_host_web_tools():
    names = {item["function"]["name"] for item in tools.TOOL_DEFINITIONS}
    assert {"terminal", "publish_file", "load_file_for_model", "get_tool_result"} <= names
    assert not {"web_search", "open_browser_page"} & names


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
